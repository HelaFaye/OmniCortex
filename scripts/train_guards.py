"""
Guards for fine-tuning (finetune.py): keep a model close to what it was, and stop a run that is collapsing.

**KL anchor** (`--kl-anchor BETA`, LoRA and QLoRA): the loss gains BETA times the KL divergence from the base model's
next-token distribution to the tuned model's, on the reply tokens of the anchor rows (the general replay rows a
dataset mixes in: `source` replay, anchor or general). The base model is the same weights with the adapter switched
off, so nothing is loaded twice. New knowledge is learned from the other rows; on the replay rows the model is pulled
back toward what it already said, which is the forgetting that replay alone only dilutes.

**Collapse guard** (`--guard-every N`): every N steps, greedy replies to a few fixed probe prompts (taken from the
anchor rows) are compared with the replies before training. The run stops when:
- **diversity collapses:** the share of distinct word pairs across the replies falls below `--guard-min-diversity`
  of its value before training (a model that says the same thing to everything);
- **abstention jumps:** the share of replies that decline, or that are empty, rises by more than
  `--guard-max-abstain-rise`;
- **it drifts from the base:** the mean KL from the base model on the probe replies passes `--guard-max-kl` (nats
  per token; LoRA and QLoRA only, as the base is the adapter switched off);
- **the loss is not a number.**

A stopped run still saves what it trained, says why in `omnicortex-finetune.json` (`guard`), and finetune.py exits
with status 5, so a retraining cycle knows not to use it.
"""
from __future__ import annotations

import math

ANCHOR_SOURCES = ("replay", "anchor", "general")
ABSTAIN = ("i don't know", "i do not know", "i'm not sure", "i am not sure", "cannot answer", "can't answer",
           "not able to answer", "i'm unable", "i am unable", "no information")


def is_anchor(rec: dict) -> int:
    return int(str(rec.get("source", "")).lower() in ANCHOR_SOURCES)


def distinct2(texts: list[str]) -> float:
    """Distinct word pairs over all word pairs, across the replies: 1 when nothing repeats, near 0 when it loops."""
    pairs = []
    for t in texts:
        w = t.lower().split()
        pairs += list(zip(w, w[1:]))
    return len(set(pairs)) / len(pairs) if pairs else 0.0


def abstain_rate(texts: list[str]) -> float:
    return sum(any(m in t.lower() for m in ABSTAIN) for t in texts if t.strip()) / max(1, len(texts))


def empty_rate(texts: list[str]) -> float:
    return sum(not t.strip() for t in texts) / max(1, len(texts))


def kl_base_tuned(base_logits, tuned_logits, mask):
    """Mean over masked positions of KL(base || tuned) between next-token distributions."""
    import torch.nn.functional as F
    lp_t = F.log_softmax(tuned_logits.float(), dim=-1)
    lp_b = F.log_softmax(base_logits.float(), dim=-1)
    kl = (lp_b.exp() * (lp_b - lp_t)).sum(-1)
    m = mask.to(kl.dtype)
    return (kl * m).sum() / m.sum().clamp(min=1)


def anchored_trainer(base_cls, beta: float):
    """A subclass of base_cls (an SFTTrainer) whose loss adds beta * KL(base || tuned) on anchor rows' replies."""
    import torch

    class Anchored(base_cls):
        kl_sum, kl_n = 0.0, 0

        def compute_loss(self, model, inputs, return_outputs=False, **kw):
            anchor = inputs.pop("anchor", None)
            loss, outputs = super().compute_loss(model, inputs, return_outputs=True, **kw)
            if anchor is not None and beta > 0 and bool(anchor.any()) and "labels" in inputs:
                labels = inputs["labels"]
                mask = (labels[:, 1:] != -100) & anchor.bool()[:, None]
                if bool(mask.any()):
                    with torch.no_grad(), model.disable_adapter():
                        base = model(input_ids=inputs["input_ids"], attention_mask=inputs.get("attention_mask")).logits
                    kl = kl_base_tuned(base[:, :-1], outputs.logits[:, :-1], mask)
                    loss = loss + beta * kl
                    self.kl_sum += float(kl.detach())
                    self.kl_n += 1
            return (loss, outputs) if return_outputs else loss

        def log(self, logs, *a, **kw):
            if self.kl_n:
                logs["kl_anchor"] = round(self.kl_sum / self.kl_n, 6)
                self.kl_sum, self.kl_n = 0.0, 0
            return super().log(logs, *a, **kw)

    return Anchored


def anchor_collator(collate):
    """Wraps a collator: takes each example's anchor flag out before collating and puts it back as a tensor."""
    import torch

    def wrapped(examples):
        flags = [int(e.pop("anchor", 0)) for e in examples]
        batch = collate(examples)
        batch["anchor"] = torch.tensor(flags)
        return batch
    return wrapped


class CollapseGuard:
    """Built as a transformers TrainerCallback by guard_callback(); kept plain here so its rules test without torch."""

    def __init__(self, min_diversity=0.5, max_abstain_rise=0.3, max_kl=None):
        self.min_diversity, self.max_abstain_rise, self.max_kl = min_diversity, max_abstain_rise, max_kl
        self.before, self.history, self.tripped = None, [], None

    def judge(self, step: int, replies: list[str], kl: float | None = None, loss: float | None = None) -> str | None:
        m = {"step": step, "diversity": round(distinct2(replies), 4), "abstain": round(abstain_rate(replies), 4),
             "empty": round(empty_rate(replies), 4)}
        if kl is not None:
            m["kl"] = round(kl, 5)
        self.history.append(m)
        if self.before is None:
            self.before = m
            return None
        why = None
        if loss is not None and not math.isfinite(loss):
            why = f"the loss is {loss}"
        elif self.before["diversity"] > 0 and m["diversity"] < self.min_diversity * self.before["diversity"]:
            why = (f"reply diversity fell from {self.before['diversity']:.2f} to {m['diversity']:.2f} "
                   f"(limit {self.min_diversity:.0%} of the start)")
        elif m["empty"] - self.before["empty"] > self.max_abstain_rise:
            why = f"empty replies rose from {self.before['empty']:.0%} to {m['empty']:.0%}"
        elif m["abstain"] - self.before["abstain"] > self.max_abstain_rise:
            why = f"abstention rose from {self.before['abstain']:.0%} to {m['abstain']:.0%}"
        elif self.max_kl is not None and kl is not None and kl > self.max_kl:
            why = f"KL from the base model reached {kl:.3f} nats per token (limit {self.max_kl})"
        if why:
            self.tripped = {"step": step, "why": why}
        return why

    def report(self) -> dict:
        return {"limits": {"min_diversity": self.min_diversity, "max_abstain_rise": self.max_abstain_rise,
                           "max_kl": self.max_kl}, "history": self.history, "stopped": self.tripped}


def guard_callback(guard: CollapseGuard, prompts: list, tok, every: int, max_new_tokens: int = 48, kl: bool = False):
    """A TrainerCallback that runs the guard every `every` steps (and once before training)."""
    import torch
    from transformers import TrainerCallback

    def probe(model):
        model_was_training = model.training
        model.eval()
        replies, kls = [], []
        with torch.no_grad():
            for msgs in prompts:
                ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt",
                                              return_dict=True)
                ids = {k: v.to(model.device) for k, v in ids.items()}
                out = model.generate(**ids, max_new_tokens=max_new_tokens, do_sample=False,
                                     pad_token_id=tok.pad_token_id or tok.eos_token_id)
                n = ids["input_ids"].shape[1]
                replies.append(tok.decode(out[0, n:], skip_special_tokens=True))
                if kl and out.shape[1] > n:
                    tuned = model(input_ids=out).logits
                    with model.disable_adapter():
                        base = model(input_ids=out).logits
                    mask = torch.zeros(out.shape[0], out.shape[1] - 1, dtype=torch.bool, device=out.device)
                    mask[:, n - 1:] = True
                    kls.append(float(kl_base_tuned(base[:, :-1], tuned[:, :-1], mask)))
        if model_was_training:
            model.train()
        return replies, (sum(kls) / len(kls) if kls else None)

    class Guard(TrainerCallback):
        last_loss = None

        def on_train_begin(self, args, state, control, model=None, **kw):
            guard.judge(0, *probe(model))

        def on_log(self, args, state, control, logs=None, **kw):
            if logs and "loss" in logs:
                self.last_loss = float(logs["loss"])

        def on_step_end(self, args, state, control, model=None, **kw):
            if every and state.global_step % every == 0:
                replies, k = probe(model)
                why = guard.judge(state.global_step, replies, k, self.last_loss)
                if why:
                    print(f"collapse guard: stopping at step {state.global_step}: {why}", flush=True)
                    control.should_training_stop = True
            return control

    return Guard()
