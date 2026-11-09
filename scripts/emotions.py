#!/usr/bin/env python3
"""
Emotion vectors: how a model represents emotion concepts, where they are active, and steering with them.

Follows Anthropic's "Emotion Concepts and their Function in a Large Language Model" (Sofroniew,
Kauvar, Saunders, Chen et al., 2026; docs/CREDITS.md):

  1. stories   a model writes short stories in which a character feels each emotion, without naming
               it; and emotionally neutral dialogues for denoising
  2. build     the model under study reads them. Per layer, the residual stream is averaged over each
               story's tokens from the 50th on; an emotion's vector is the mean over its stories minus
               the mean over all emotions. The top principal components of the neutral dialogues'
               activations (enough to explain 50% of their variance) are projected out.
               Held-out stories check that each vector picks out its own emotion, layer by layer
  3. look      trace (per-token activation on any text), scan (the passages of a corpus that activate
               each vector most), sweep (activation as one quantity in a prompt changes), the map
               (principal components: the paper finds valence and arousal), logit effects
  4. steer     add s x |h| x v to the residual stream (s is a fraction of the residual stream's norm at
               that token; the paper sweeps -0.1..0.1), in the decoding lab or here; or export a
               llama.cpp control vector

    python scripts/emotions.py stories --writer http://127.0.0.1:1234/v1@writer-model --topics 8 --out runs/emotions/stories
    python scripts/emotions.py build --model Qwen/Qwen2.5-0.5B-Instruct --stories runs/emotions/stories --out runs/emotions/qwen
    python scripts/emotions.py trace --emotions runs/emotions/qwen --text "I can't believe it's finally over!" --out t.json
    python scripts/emotions.py steer --emotions runs/emotions/qwen --prompt "Hi." --mix calm=0.05 --mix desperate=-0.05

The default list is the paper's 171 emotion words; the topics are our own (the paper's 100 are in
its appendix; pass them with --topics-file to use them).
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

# The paper's full list (its appendix "Full list of emotions").
EMOTIONS = [
    "afraid", "alarmed", "alert", "amazed", "amused", "angry", "annoyed", "anxious", "aroused", "ashamed",
    "astonished", "at ease", "awestruck", "bewildered", "bitter", "blissful", "bored", "brooding", "calm",
    "cheerful", "compassionate", "contemptuous", "content", "defiant", "delighted", "dependent", "depressed",
    "desperate", "disdainful", "disgusted", "disoriented", "dispirited", "distressed", "disturbed", "docile",
    "droopy", "dumbstruck", "eager", "ecstatic", "elated", "embarrassed", "empathetic", "energized", "enraged",
    "enthusiastic", "envious", "euphoric", "exasperated", "excited", "exuberant", "frightened", "frustrated",
    "fulfilled", "furious", "gloomy", "grateful", "greedy", "grief-stricken", "grumpy", "guilty", "happy",
    "hateful", "heartbroken", "hope", "hopeful", "horrified", "hostile", "humiliated", "hurt", "hysterical",
    "impatient", "indifferent", "indignant", "infatuated", "inspired", "insulted", "invigorated", "irate",
    "irritated", "jealous", "joyful", "jubilant", "kind", "lazy", "listless", "lonely", "loving", "mad",
    "melancholy", "miserable", "mortified", "mystified", "nervous", "nostalgic", "obstinate", "offended",
    "on edge", "optimistic", "outraged", "overwhelmed", "panicked", "paranoid", "patient", "peaceful",
    "perplexed", "playful", "pleased", "proud", "puzzled", "rattled", "reflective", "refreshed", "regretful",
    "rejuvenated", "relaxed", "relieved", "remorseful", "resentful", "resigned", "restless", "sad", "safe",
    "satisfied", "scared", "scornful", "self-confident", "self-conscious", "self-critical", "sensitive",
    "sentimental", "serene", "shaken", "shocked", "skeptical", "sleepy", "sluggish", "smug", "sorry", "spiteful",
    "stimulated", "stressed", "stubborn", "stuck", "sullen", "surprised", "suspicious", "sympathetic", "tense",
    "terrified", "thankful", "thrilled", "tired", "tormented", "trapped", "triumphant", "troubled", "uneasy",
    "unhappy", "unnerved", "unsettled", "upset", "valiant", "vengeful", "vibrant", "vigilant", "vindictive",
    "vulnerable", "weary", "worn out", "worried", "worthless"]

# A smaller starting set for slow machines: broad coverage of valence and arousal.
CORE = ["happy", "sad", "afraid", "angry", "calm", "desperate", "surprised", "disgusted", "proud", "ashamed",
        "loving", "lonely", "excited", "bored", "anxious", "relieved", "grateful", "guilty", "hopeful", "frustrated",
        "puzzled", "nervous", "content", "hostile"]

TOPICS = [
    "a delayed train at a crowded station", "a first day at a new job", "an old friend's unexpected phone call",
    "a family dinner after a long time apart", "losing a wallet in a market", "a hospital waiting room",
    "a job interview that runs long", "moving into a new apartment", "a dog that will not come home",
    "a letter found in a second-hand book", "a power cut during a storm", "a school play on opening night",
    "a broken phone on a hiking trail", "a neighbour's loud renovation", "a surprise birthday party",
    "a missed flight connection", "a bakery closing down", "a long queue at a government office",
    "an exam result arriving by email", "a garden ruined by frost", "a borrowed car with a flat tyre",
    "a wedding speech", "an argument about who pays the bill", "the last day before retirement",
    "a stranger returning a lost bag", "a child's first bicycle ride", "a promotion going to someone else",
    "a medical test result", "a reunion with a sibling", "a house sale falling through"]

NEUTRAL_TOPICS = [
    "converting units of measurement", "formatting a table of data", "a Python function that sorts a list",
    "the boiling point of water at altitude", "how to reset a router", "summarising a technical report",
    "the order of planets from the sun", "naming conventions for files", "scheduling a recurring meeting",
    "translating a sentence into French", "how photosynthesis works", "writing a SQL query that joins two tables",
    "the history of the printing press", "calculating compound interest", "organising files into folders",
    "proofreading a paragraph for spelling", "the capital cities of Europe", "setting up a spreadsheet formula",
    "the difference between TCP and UDP", "a recipe's ingredient quantities for six people"]


def progress(done: int, total: int, label: str) -> None:
    print(f"PROGRESS {done}/{total} {label}", flush=True)


# ====================================================================== 1. stories

def story_prompt(emotion: str, topic: str, n: int) -> str:
    # follows the structure and constraints of the paper's story prompt (its appendix), in our words
    return (f"Write {n} different short stories, each about one paragraph long, based on this premise.\n\n"
            f"Premise: {topic}\n\nEach story follows a character who feels {emotion}.\n\n"
            "Number them [story 1], [story 2] and so on, each on its own line before the story. The stories are "
            "unrelated fresh starts: vary the phrasing, and mix first-person and third-person narration.\n\n"
            f"Never use the word '{emotion}' or a direct synonym of it. Show the feeling only through what the "
            "character does, their body and physical sensations, what they say and how, their thoughts, and the "
            "situation around them. A reader should still be able to tell clearly what they feel.")


def neutral_prompt(topic: str, n: int) -> str:
    return (f"Write {n} different dialogues between a person and an AI assistant about: {topic}.\n\n"
            "Number them [dialogue 1], [dialogue 2] and so on. Each has 1 to 5 exchanges, with each turn on its own "
            "line starting 'Person:' or 'AI:'. Some may begin with a short instruction to the assistant.\n\n"
            "The dialogues must be completely neutral: no feelings expressed or implied by either side (no "
            "gratitude, frustration, enthusiasm, worry or concern), no emotionally charged subjects, plain and "
            "matter-of-fact language throughout.")


def split_numbered(text: str, word: str) -> list[str]:
    parts = re.split(rf"\[\s*{word}\s*\d+\s*\]", text, flags=re.I)
    out = [p.strip() for p in parts[1:] if p.strip()] if len(parts) > 1 else [text.strip()]
    return [p for p in out if len(p.split()) >= 25]


class Writer:
    """Text from an OpenAI-compatible endpoint (URL@model) or a local Hugging Face model."""

    def __init__(self, endpoint: str | None, model_id: str | None, device: str = "auto", max_tokens: int = 1200,
                 temperature: float = 0.9):
        self.endpoint, self.max_tokens, self.temperature = endpoint, max_tokens, temperature
        self.model = None
        if not endpoint:
            if not model_id:
                raise SystemExit("who writes the stories? --writer URL@model or --writer-model HF-id")
            import features as FX
            self.model, self.tok, self.dev = FX.load_model(model_id, "auto", device)

    def __call__(self, prompt: str) -> str:
        if self.endpoint:
            url, model = self.endpoint, None
            if "@" in url.split("//", 1)[-1]:
                url, model = url.rsplit("@", 1)
            import urllib.request
            u = url.rstrip("/") + ("" if url.rstrip("/").endswith("/v1") else "/v1") + "/chat/completions"
            body = {"messages": [{"role": "user", "content": prompt}], "max_tokens": self.max_tokens,
                    "temperature": self.temperature}
            if model:
                body["model"] = model
            req = urllib.request.Request(u, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=600) as r:
                txt = json.loads(r.read())["choices"][0]["message"].get("content") or ""
            return txt.split("</think>", 1)[-1]
        import torch
        tok = self.tok
        text = tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True) \
            if getattr(tok, "chat_template", None) else prompt
        ids = tok(text, return_tensors="pt", add_special_tokens=not getattr(tok, "chat_template", None)).to(self.dev)
        with torch.no_grad():
            out = self.model.generate(**ids, max_new_tokens=self.max_tokens, do_sample=self.temperature > 0,
                                      temperature=max(self.temperature, 1e-5), top_p=0.95,
                                      pad_token_id=tok.pad_token_id or tok.eos_token_id)
        return tok.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True).split("</think>", 1)[-1]


def pick_emotions(a) -> list[str]:
    if getattr(a, "emotions_file", None):
        return [l.strip() for l in Path(a.emotions_file).read_text().splitlines() if l.strip()]
    if getattr(a, "emotions", None):
        return [e.strip() for e in ",".join(a.emotions).split(",") if e.strip()]
    return CORE if getattr(a, "preset", "paper") == "core" else EMOTIONS


def cmd_stories(a) -> int:
    emotions = pick_emotions(a)
    topics = ([l.strip() for l in Path(a.topics_file).read_text().splitlines() if l.strip()] if a.topics_file
              else TOPICS)
    rng = random.Random(a.seed)
    topics = topics if a.topics <= 0 else rng.sample(topics, min(a.topics, len(topics)))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    w = Writer(a.writer, a.writer_model, a.device, a.max_tokens, a.temperature)
    sp, npth = out / "stories.jsonl", out / "neutral.jsonl"
    have = set()
    if sp.exists() and not a.overwrite:
        for l in sp.read_text().splitlines():
            if l.strip():
                r = json.loads(l)
                have.add((r["emotion"], r["topic"]))
    jobs = [(e, t) for e in emotions for t in topics if (e, t) not in have]
    nt = [t for t in NEUTRAL_TOPICS][: a.neutral_topics] if a.neutral_topics else []
    total = len(jobs) + (len(nt) if not npth.exists() or a.overwrite else 0)
    done = 0
    with sp.open("w" if a.overwrite else "a") as f:
        for e, t in jobs:
            try:
                for s in split_numbered(w(story_prompt(e, t, a.per_call)), "story"):
                    if re.search(rf"\b{re.escape(e)}\b", s, re.I) and not a.keep_named:
                        continue                                      # it named the emotion: leave it out
                    f.write(json.dumps({"emotion": e, "topic": t, "text": s}) + "\n")
                f.flush()
            except Exception as ex:                                   # noqa: BLE001
                print(f"{e} / {t}: {type(ex).__name__}: {ex}", flush=True)
            done += 1
            progress(done, total, f"story prompts ({e}: {t})")
    if nt and (not npth.exists() or a.overwrite):
        with npth.open("w") as f:
            for t in nt:
                try:
                    for s in split_numbered(w(neutral_prompt(t, a.per_call)), "dialogue"):
                        f.write(json.dumps({"topic": t, "text": s}) + "\n")
                except Exception as ex:                               # noqa: BLE001
                    print(f"neutral / {t}: {type(ex).__name__}: {ex}", flush=True)
                done += 1
                progress(done, total, f"neutral dialogues ({t})")
    n = sum(1 for _ in sp.open())
    print(f"{n} stories in {sp}" + (f"; {sum(1 for _ in npth.open())} neutral dialogues in {npth}" if npth.exists() else ""))
    return 0


# ====================================================================== 2. build

def _read_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []


def story_means(model, tok, texts: list[str], layers: list[int], start: int, dev, label: str):
    """[n, len(layers), d] mean residual stream (after each layer) over tokens from `start` (or from a third
    of the way in for shorter texts), plus per-layer mean token norm."""
    import torch
    rows, norms = [], torch.zeros(len(layers))
    with torch.no_grad():
        for i, t in enumerate(texts):
            ids = tok(t, return_tensors="pt")["input_ids"][:, :512].to(dev)
            hs = model(input_ids=ids, output_hidden_states=True, use_cache=False).hidden_states
            s = start if ids.shape[1] > start + 8 else max(1, ids.shape[1] // 3)
            h = torch.stack([hs[l + 1][0, s:] for l in layers]).float()          # [L, T, d]
            rows.append(h.mean(1).cpu())
            norms += h.norm(dim=-1).mean(1).cpu()
            if (i + 1) % 10 == 0 or i == len(texts) - 1:
                progress(i + 1, len(texts), label)
    return torch.stack(rows), norms / max(1, len(texts))


def neutral_tokens(model, tok, texts: list[str], layers: list[int], dev, max_tokens: int = 20000):
    import torch
    out = [[] for _ in layers]
    n = 0
    with torch.no_grad():
        for t in texts:
            ids = tok(t, return_tensors="pt")["input_ids"][:, :512].to(dev)
            hs = model(input_ids=ids, output_hidden_states=True, use_cache=False).hidden_states
            for j, l in enumerate(layers):
                out[j].append(hs[l + 1][0, 1:].float().cpu())
            n += ids.shape[1]
            if n >= max_tokens:
                break
    return [torch.cat(o) for o in out]


def denoise_basis(X, var: float = 0.5):
    """Top principal components of X explaining `var` of its variance -> [d, k] orthonormal."""
    import torch
    Xc = X - X.mean(0)
    U, S_, V = torch.linalg.svd(Xc, full_matrices=False)
    ev = S_ ** 2 / (S_ ** 2).sum()
    k = int((torch.cumsum(ev, 0) < var).sum()) + 1
    return V[:k].t().contiguous(), k


def cmd_build(a) -> int:
    import torch
    import features as FX
    from concept_edit import decoder_layers
    model, tok, dev = FX.load_model(a.model, a.dtype, a.device)
    n_layers = len(decoder_layers(model))
    layers = list(range(n_layers)) if a.layers in (None, ["all"]) else [int(x) for x in a.layers]
    sdir = Path(a.stories)
    stories = _read_jsonl(sdir / "stories.jsonl" if sdir.is_dir() else sdir)
    neutral = _read_jsonl(sdir / "neutral.jsonl" if sdir.is_dir() else Path(a.neutral or ""))
    if a.neutral:
        neutral = _read_jsonl(Path(a.neutral))
    names = sorted({s["emotion"] for s in stories}, key=lambda e: (EMOTIONS.index(e) if e in EMOTIONS else 999, e))
    by = {e: [s["text"] for s in stories if s["emotion"] == e] for e in names}
    names = [e for e in names if len(by[e]) >= 2]
    if len(names) < 2:
        raise SystemExit("need stories for at least two emotions, two each")
    rng = random.Random(a.seed)
    train, test = {}, {}
    for e in names:
        xs = by[e][:]
        rng.shuffle(xs)
        k = max(1, int(round(len(xs) * a.holdout))) if len(xs) >= 4 else 0
        test[e], train[e] = xs[:k], xs[k:]
    flat = [(e, t, "train") for e in names for t in train[e]] + [(e, t, "test") for e in names for t in test[e]]
    print(f"{len(names)} emotions, {sum(len(v) for v in train.values())} training and "
          f"{sum(len(v) for v in test.values())} held-out stories; layers {layers[0]}..{layers[-1]}", flush=True)
    M, norms = story_means(model, tok, [t for _, t, _ in flat], layers, a.start, dev, "stories read")
    lab = [e for e, _, _ in flat]
    split = [s for _, _, s in flat]
    tr = torch.tensor([s == "train" for s in split])
    te = ~tr
    idx = {e: i for i, e in enumerate(names)}
    y = torch.tensor([idx[e] for e in lab])
    E, L = len(names), len(layers)
    means = torch.stack([M[tr & (y == i)].mean(0) for i in range(E)])          # [E, L, d]
    center = means.mean(0)                                                       # [L, d]
    V = means - center                                                           # [E, L, d]
    removed = {}
    V_raw = V.clone()
    if neutral and a.denoise > 0:
        print(f"denoising with {len(neutral)} neutral dialogues", flush=True)
        if len(neutral) < 50:
            print(f"note: {len(neutral)} neutral dialogues is few; their principal components are noisy and can take "
                  "emotion signal with them. Compare the two accuracies below (or --denoise 0).", flush=True)
        NT = neutral_tokens(model, tok, [n["text"] for n in neutral], layers, dev)
        for j, l in enumerate(layers):
            P, k = denoise_basis(NT[j], a.denoise)
            V[:, j] = V[:, j] - (V[:, j] @ P) @ P.t()
            removed[l] = k
    U = V / V.norm(dim=-1, keepdim=True).clamp(min=1e-8)

    def heldout(Uv):
        a1, a5 = {}, {}
        Z = (M[te] - center)                                                      # [n, L, d]
        for j, l in enumerate(layers):
            s = Z[:, j] @ Uv[:, j].t()                                            # [n, E]
            top = s.topk(min(5, E), dim=1).indices
            a1[l] = float((top[:, 0] == y[te]).float().mean())
            a5[l] = float((top == y[te][:, None]).any(1).float().mean())
        return a1, a5
    acc, acc5, acc_raw = {}, {}, {}
    if bool(te.any()):
        acc, acc5 = heldout(U)
        if removed:
            acc_raw = heldout(V_raw / V_raw.norm(dim=-1, keepdim=True).clamp(min=1e-8))[0]
    default_layer = layers[min(range(L), key=lambda j: abs(layers[j] - round(2 * n_layers / 3)))]
    best = max(acc, key=acc.get) if acc else default_layer
    # projection statistics over all story tokens are approximated by story means (for z-scores)
    proj = torch.einsum("nld,eld->nle", M - center, U)
    mu, sd = proj.mean(0), proj.std(0).clamp(min=1e-6)                           # [L, E]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"names": names, "layers": layers, "vectors": V.permute(1, 0, 2).contiguous().half(),
                "center": center.half(), "norms": norms, "mu": mu, "sd": sd, "model": a.model,
                "default_layer": default_layer, "best_layer": best, "removed_pcs": removed,
                "format": "omnicortex-emotions/1"}, out / "emotions.pt")
    j = layers.index(default_layer)
    Uj = U[:, j]
    cos = (Uj @ Uj.t()).numpy().round(3).tolist()
    Vc = V[:, j] - V[:, j].mean(0)
    _, S_, Vt = torch.linalg.svd(Vc, full_matrices=False)
    pcs = (Vc @ Vt[:2].t())
    ev = (S_ ** 2 / (S_ ** 2).sum())[:5].tolist()
    if "happy" in names and float(pcs[names.index("happy"), 0]) < 0:
        pcs[:, 0] = -pcs[:, 0]                                                    # positive valence to the right
    info = {"model": a.model, "names": names, "layers": layers, "n_layers": n_layers, "default_layer": default_layer,
            "best_layer": best, "accuracy": acc, "accuracy_top5": acc5, "accuracy_raw": acc_raw, "chance": 1 / E,
            "counts": {e: [len(train[e]), len(test[e])] for e in names}, "removed_pcs": removed,
            "start_token": a.start, "pca": {"layer": default_layer, "xy": pcs.numpy().round(4).tolist(),
                                             "explained": [round(x, 4) for x in ev]},
            "cosine": cos, "norms": {str(l): round(float(n), 3) for l, n in zip(layers, norms)},
            "created": time.strftime("%Y-%m-%d %H:%M:%S"), "stories": str(sdir)}
    try:
        from logit_lens import final_norm
        g = getattr(final_norm(model), "weight", None)
        Wu = model.get_output_embeddings().weight.detach().float()
        W = Uj.to(Wu.device) * (g.detach().float() if g is not None else 1)
        logits = W @ Wu.t()
        info["logits"] = {e: {"up": [tok.decode([int(t)]) for t in logits[i].topk(8).indices],
                              "down": [tok.decode([int(t)]) for t in (-logits[i]).topk(8).indices]}
                          for i, e in enumerate(names)}
    except Exception as ex:                                                       # noqa: BLE001
        print(f"no logit effects: {type(ex).__name__}: {ex}", flush=True)
    if a.judge:
        import features as FX2
        va = {}
        for i, e in enumerate(names):
            try:
                r = FX2._chat(a.judge, [{"role": "user", "content":
                              f"Rate the emotion concept '{e}' on valence (-1 very negative to 1 very positive) and "
                              "arousal (-1 very calm, low energy to 1 very intense, high energy). Reply with two numbers "
                              "separated by a comma, nothing else."}], 20)
                nums = [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", r)[:2]]
                if len(nums) == 2:
                    va[e] = nums
            except Exception as ex:                                               # noqa: BLE001
                print(f"judge {e}: {ex}", flush=True)
            progress(i + 1, len(names), "emotions rated for valence and arousal")
        info["valence_arousal"] = va
        if len(va) > 3:
            import numpy as np
            ks = [e for e in names if e in va]
            xs = np.array([float(pcs[names.index(e), 0]) for e in ks])
            ys = np.array([float(pcs[names.index(e), 1]) for e in ks])
            info["pc_corr"] = {"pc1_valence": float(np.corrcoef(xs, [va[e][0] for e in ks])[0, 1]),
                               "pc2_arousal": float(np.corrcoef(ys, [va[e][1] for e in ks])[0, 1])}
    (out / "emotions.json").write_text(json.dumps(info, indent=1))
    if acc:
        print("held-out accuracy by layer (top-1, chance %.3f): " % (1 / E)
              + ", ".join(f"{l}: {acc[l]:.2f}" for l in layers), flush=True)
    if acc_raw:
        print("  without denoising:                          "
              + ", ".join(f"{l}: {acc_raw[l]:.2f}" for l in layers), flush=True)
        if max(acc_raw.values()) > max(acc.values()) + 0.05:
            print("denoising lowered the best held-out accuracy; with this neutral set, --denoise 0 may be better",
                  flush=True)
    print(f"default layer {default_layer} (two thirds of the depth, as in the paper); best held-out layer {best}; "
          f"wrote {out}/emotions.pt and emotions.json")
    return 0


# ====================================================================== 3. look

class Vectors:
    def __init__(self, path: str | Path, device="cpu"):
        import torch
        p = Path(path)
        d = torch.load(p / "emotions.pt" if p.is_dir() else p, map_location="cpu", weights_only=False)
        self.names, self.layers = d["names"], d["layers"]
        self.V = d["vectors"].float()                                               # [L, E, d]
        self.U = self.V / self.V.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        self.center, self.mu, self.sd = d["center"].float(), d["mu"], d["sd"]
        self.default_layer, self.model = d["default_layer"], d.get("model")
        self.norms = d["norms"]
        self.U = self.U.to(device)

    def j(self, layer: int | None) -> int:
        layer = self.default_layer if layer is None else layer
        if layer not in self.layers:
            raise SystemExit(f"no vectors for layer {layer}; have {self.layers[0]}..{self.layers[-1]}")
        return self.layers.index(layer)

    def unit(self, name: str, layer: int | None = None):
        if name not in self.names:
            raise KeyError(f"no emotion {name!r}")
        return self.U[self.j(layer), self.names.index(name)]


def encode_text(tok, text: str | None, messages: list | None = None, reply: str | None = None):
    """-> (ids, index where the reply starts)."""
    if messages is not None or reply is not None:
        from logit_lens import encode
        return encode(tok, None, messages or [], reply or "")
    return tok(text)["input_ids"], 0


def trace(model, tok, vec: Vectors, ids: list[int], layer: int | None = None, top: int = 3) -> dict:
    import torch
    j = vec.j(layer)
    l = vec.layers[j]
    dev = next(model.parameters()).device
    with torch.no_grad():
        hs = model(input_ids=torch.tensor([ids], device=dev), output_hidden_states=True, use_cache=False).hidden_states
    h = hs[l + 1][0].float()
    s = ((h - vec.center[j].to(h.device)) @ vec.U[j].to(h.device).t()).cpu()           # [T, E]
    z = (s - vec.mu[j]) / vec.sd[j]
    tops = z.topk(min(top, z.shape[1]), dim=1)
    return {"layer": l, "names": vec.names, "tokens": [tok.decode([t]) for t in ids],
            "z": z.numpy().round(3).tolist(),
            "top": [[[vec.names[int(i)], round(float(v), 2)] for v, i in zip(tv, ti)]
                    for tv, ti in zip(tops.values, tops.indices)]}


def cmd_trace(a) -> int:
    import features as FX
    vec = Vectors(a.emotions)
    model, tok, dev = FX.load_model(a.model or vec.model, a.dtype, a.device)
    text = Path(a.file).read_text() if a.file else a.text
    msgs = json.loads(Path(a.messages).read_text()) if a.messages else None
    ids, start = encode_text(tok, text, msgs, a.reply)
    r = trace(model, tok, vec, ids, a.layer, a.top)
    r["reply_start"] = start
    if a.emotion:
        k = vec.names.index(a.emotion)
        line = "".join(f"[{t}]" if row[k] > 1.5 else t for t, row in zip(r["tokens"], r["z"]))
        print(f"{a.emotion} (z > 1.5 bracketed):\n{line}")
    else:
        for t, tp in list(zip(r["tokens"], r["top"]))[-40:]:
            print(f"{t!r:>16}  " + "  ".join(f"{n} {v:+.1f}" for n, v in tp))
    if a.out:
        Path(a.out).write_text(json.dumps(r))
        print(f"wrote {a.out}")
    return 0


def cmd_scan(a) -> int:
    """The passages of a corpus that activate each emotion vector most (the paper's first validation)."""
    import torch
    import features as FX
    vec = Vectors(a.emotions)
    model, tok, dev = FX.load_model(a.model or vec.model, a.dtype, a.device)
    docs = FX.read_corpus(a.corpus, a.field, a.split, a.config, a.limit_docs, tok)
    j = vec.j(a.layer)
    l = vec.layers[j]
    best = {e: [] for e in vec.names}
    allz = []
    for i, d in enumerate(docs):
        ids = tok(d)["input_ids"][: a.ctx]
        if len(ids) < 8:
            continue
        with torch.no_grad():
            hs = model(input_ids=torch.tensor([ids], device=dev), output_hidden_states=True, use_cache=False).hidden_states
        z = (((hs[l + 1][0].float() - vec.center[j].to(dev)) @ vec.U[j].to(dev).t()).cpu() - vec.mu[j]) / vec.sd[j]
        z[:1] = float("-inf")
        allz.append(z[1:].flatten()[::7])
        for k, e in enumerate(vec.names):
            m, p = z[:, k].max(0)
            best[e].append((float(m), i, int(p)))
        progress(i + 1, len(docs), "documents scanned")
    thr = float(torch.cat(allz).quantile(0.9)) if allz else 0.0
    out = {"layer": l, "threshold_z": round(thr, 3), "emotions": {}}
    for k, e in enumerate(vec.names):
        rows = []
        for m, i, p in sorted(best[e], reverse=True)[: a.top]:
            ids = tok(docs[i])["input_ids"][: a.ctx]
            s, t = max(0, p - a.left), min(len(ids), p + a.right + 1)
            rows.append({"doc": i, "peak": round(m, 2), "tokens": [tok.decode([x]) for x in ids[s:t]], "at": p - s})
        out["emotions"][e] = rows
    Path(a.out).write_text(json.dumps(out))
    print(f"wrote {a.out} ({len(docs)} documents, top {a.top} per emotion; tokens above z {thr:.2f} are the top 10%)")
    return 0


def cmd_sweep(a) -> int:
    """Activation just before the reply as one quantity in the prompt changes (the paper's Tylenol example)."""
    import torch
    import features as FX
    vec = Vectors(a.emotions)
    model, tok, dev = FX.load_model(a.model or vec.model, a.dtype, a.device)
    j = vec.j(a.layer)
    l = vec.layers[j]
    names = a.emotion or vec.names
    rows = []
    for v in a.values:
        text = a.template.replace("{x}", str(v))
        if getattr(tok, "chat_template", None) and not a.raw:
            text = tok.apply_chat_template([{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True)
            ids = tok(text, add_special_tokens=False)["input_ids"]
        else:
            ids = tok(text)["input_ids"]
        with torch.no_grad():
            hs = model(input_ids=torch.tensor([ids], device=dev), output_hidden_states=True, use_cache=False).hidden_states
        z = (((hs[l + 1][0, -1].float() - vec.center[j].to(dev)) @ vec.U[j].to(dev).t()).cpu() - vec.mu[j]) / vec.sd[j]
        rows.append({"value": v, "z": {e: round(float(z[vec.names.index(e)]), 3) for e in names}})
        print(f"{v:>10}  " + "  ".join(f"{e} {rows[-1]['z'][e]:+.2f}" for e in names[:8]), flush=True)
    if a.out:
        Path(a.out).write_text(json.dumps({"layer": l, "template": a.template, "rows": rows}, indent=1))
        print(f"wrote {a.out}")
    return 0


# ====================================================================== 4. steer

def parse_mix(items: list[str]) -> dict[str, float]:
    mix = {}
    for it in items or []:
        name, _, s = it.rpartition("=")
        if not name:
            raise SystemExit(f"--mix {it!r}: use emotion=strength, e.g. calm=0.05")
        mix[name.strip()] = float(s)
    return mix


class Steer:
    """Add sum_e s_e x |h_t| x v_e to the residual stream after the given layers (s in fractions of the
    residual stream's norm at each token)."""

    def __init__(self, model, vec: Vectors, mix: dict, layers: list[int] | None = None):
        from concept_edit import decoder_layers
        self.blocks = decoder_layers(model)
        self.layers = layers or [vec.default_layer]
        self.dirs = {}
        for l in self.layers:
            v = sum(s * vec.unit(e, l) for e, s in mix.items())
            self.dirs[l] = v
        self.hooks = []

    def __enter__(self):
        for l, v in self.dirs.items():
            def hook(_m, _i, out, v=v):
                h = out[0] if isinstance(out, tuple) else out
                h = h + h.norm(dim=-1, keepdim=True) * v.to(h.device, h.dtype)
                return (h, *out[1:]) if isinstance(out, tuple) else h
            self.hooks.append(self.blocks[l].register_forward_hook(hook))
        return self

    def __exit__(self, *exc):
        for h in self.hooks:
            h.remove()
        self.hooks = []


def cmd_steer(a) -> int:
    import torch
    import features as FX
    vec = Vectors(a.emotions)
    model, tok, dev = FX.load_model(a.model or vec.model, a.dtype, a.device)
    mix = parse_mix(a.mix)
    layers = [int(x) for x in a.layers] if a.layers else None
    text = tok.apply_chat_template([{"role": "user", "content": a.prompt}], tokenize=False, add_generation_prompt=True) \
        if getattr(tok, "chat_template", None) else a.prompt
    ids = tok(text, return_tensors="pt", add_special_tokens=not getattr(tok, "chat_template", None)).to(dev)

    def gen():
        with torch.no_grad():
            o = model.generate(**ids, max_new_tokens=a.max_tokens, do_sample=False,
                               pad_token_id=tok.pad_token_id or tok.eos_token_id)
        return tok.decode(o[0, ids["input_ids"].shape[1]:], skip_special_tokens=True)
    base = gen()
    with Steer(model, vec, mix, layers):
        steered = gen()
    print(f"[plain]\n{base}\n\n[{', '.join(f'{e} {s:+g}' for e, s in mix.items())}]\n{steered}")
    if a.out:
        Path(a.out).write_text(json.dumps({"prompt": a.prompt, "mix": mix, "layers": layers or [vec.default_layer],
                                           "plain": base, "steered": steered}, indent=1))
    return 0


def cmd_cvec(a) -> int:
    """A llama.cpp control vector for a mix: fixed at every token (scaled by the layer's typical norm,
    since llama.cpp cannot scale by each token's norm)."""
    try:
        import gguf
    except ImportError:
        raise SystemExit("pip install gguf (the package that ships with llama.cpp)")
    vec = Vectors(a.emotions)
    mix = parse_mix(a.mix)
    layers = [int(x) for x in a.layers] if a.layers else [vec.default_layer]
    w = gguf.GGUFWriter(a.out, "controlvector")
    w.add_string("controlvector.model_hint", a.arch)
    w.add_uint32("controlvector.layer_count", len(layers))
    for l in layers:
        if l < 1:
            print(f"skipping layer {l}: llama.cpp control vectors start at layer 1")
            continue
        j = vec.j(l)
        v = sum(s * vec.unit(e, l) for e, s in mix.items()) * float(vec.norms[j])
        w.add_tensor(f"direction.{l}", v.cpu().numpy().astype("float32"))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"wrote {a.out}: llama-server --control-vector {a.out} (scale 1 = the mix as given)")
    return 0


# ====================================================================== CLI

def _common(p, model_req=False):
    p.add_argument("--model", required=model_req, help="HF id or folder (default: the one the vectors were built on)")
    p.add_argument("--dtype", default="auto", choices=["auto", "float32", "bfloat16", "float16"])
    p.add_argument("--device", default="auto")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0].strip(),
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("stories", help="have a model write the emotion stories and neutral dialogues")
    s.add_argument("--writer", help="URL@model of an OpenAI-compatible endpoint")
    s.add_argument("--writer-model", help="or a local Hugging Face model")
    s.add_argument("--preset", choices=["paper", "core"], default="paper",
                   help="paper: the paper's 171 emotions; core: 24 spread over valence and arousal")
    s.add_argument("--emotions", nargs="+", help="or these emotions (comma or space separated)")
    s.add_argument("--emotions-file", help="or one emotion per line")
    s.add_argument("--topics", type=int, default=8, help="premises per emotion (0: all)")
    s.add_argument("--topics-file", help="premises, one per line (default: 30 built in)")
    s.add_argument("--per-call", type=int, default=3, help="stories per request")
    s.add_argument("--neutral-topics", type=int, default=20, help="neutral dialogue topics (0: none)")
    s.add_argument("--keep-named", action="store_true", help="keep stories that name their emotion anyway")
    s.add_argument("--max-tokens", type=int, default=1200)
    s.add_argument("--temperature", type=float, default=0.9)
    s.add_argument("--overwrite", action="store_true", help="start over instead of adding to what is there")
    s.add_argument("--device", default="auto")
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--out", required=True, help="folder for stories.jsonl and neutral.jsonl")

    b = sub.add_parser("build", help="emotion vectors from the stories")
    _common(b, True)
    b.add_argument("--stories", required=True, help="folder from `stories`, or a stories .jsonl")
    b.add_argument("--neutral", help="neutral dialogues .jsonl (default: neutral.jsonl next to the stories)")
    b.add_argument("--layers", nargs="+", help="decoder layers (default: all)")
    b.add_argument("--start", type=int, default=50, help="average from this token of each story (the paper: 50)")
    b.add_argument("--denoise", type=float, default=0.5,
                   help="project out the neutral PCs explaining this much variance (0: off; the paper: 0.5)")
    b.add_argument("--holdout", type=float, default=0.2, help="share of each emotion's stories held out")
    b.add_argument("--judge", help="URL@model that rates each emotion's valence and arousal, for the map")
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--out", required=True)

    t = sub.add_parser("trace", help="per-token emotion activation over a text or a reply")
    _common(t)
    t.add_argument("--emotions", required=True, help="folder from build")
    t.add_argument("--text")
    t.add_argument("--file", help="or a text file")
    t.add_argument("--messages", help="or a chat JSON (with --reply)")
    t.add_argument("--reply")
    t.add_argument("--layer", type=int, help="default: the vectors' default layer")
    t.add_argument("--emotion", help="highlight one emotion")
    t.add_argument("--top", type=int, default=3)
    t.add_argument("--out")

    c = sub.add_parser("scan", help="the passages of a corpus that activate each emotion most")
    _common(c)
    c.add_argument("--emotions", required=True)
    c.add_argument("--corpus", required=True, help="as for features.py collect")
    c.add_argument("--field")
    c.add_argument("--split", default="train")
    c.add_argument("--config")
    c.add_argument("--limit-docs", type=int, default=500)
    c.add_argument("--ctx", type=int, default=256)
    c.add_argument("--layer", type=int)
    c.add_argument("--top", type=int, default=5)
    c.add_argument("--left", type=int, default=20)
    c.add_argument("--right", type=int, default=8)
    c.add_argument("--out", required=True)

    w = sub.add_parser("sweep", help="activation before the reply as a quantity in the prompt changes")
    _common(w)
    w.add_argument("--emotions", required=True)
    w.add_argument("--template", required=True, help="prompt with {x}, e.g. 'I took {x} mg of Tylenol. OK?'")
    w.add_argument("--values", nargs="+", required=True)
    w.add_argument("--emotion", nargs="+", help="which emotions to report (default: all)")
    w.add_argument("--layer", type=int)
    w.add_argument("--raw", action="store_true", help="do not wrap the prompt in the chat template")
    w.add_argument("--out")

    st = sub.add_parser("steer", help="compare a reply with and without an emotion mix added")
    _common(st)
    st.add_argument("--emotions", required=True)
    st.add_argument("--prompt", required=True)
    st.add_argument("--mix", action="append", required=True, help="emotion=strength (fraction of |h|), repeatable")
    st.add_argument("--layers", nargs="+", help="default: the vectors' default layer")
    st.add_argument("--max-tokens", type=int, default=80)
    st.add_argument("--out")

    v = sub.add_parser("cvec", help="a llama.cpp control vector for an emotion mix")
    v.add_argument("--emotions", required=True)
    v.add_argument("--mix", action="append", required=True)
    v.add_argument("--layers", nargs="+")
    v.add_argument("--arch", default="llama", help="controlvector.model_hint")
    v.add_argument("--out", required=True)

    a = p.parse_args(argv)
    return {"stories": cmd_stories, "build": cmd_build, "trace": cmd_trace, "scan": cmd_scan, "sweep": cmd_sweep,
            "steer": cmd_steer, "cvec": cmd_cvec}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
