"""
Shared machinery for the H-Neurons ROCm pipeline.

Everything that touches model internals lives here so the extractor and the
token scorer cannot drift apart. The three things this module gets right that
the upstream repo does not:

  - hooks only text decoder down_proj modules (a vision tower's MLPs would
    otherwise corrupt the flat-index -> (layer, neuron) map)
  - captures to CPU float32 inside the hook, so device_map offload works
  - builds the scored sequence as generation prompt + response token ids,
    because chat templates for reasoning models strip <think> from prior turns
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from oc_constants import TEXT_LAYER_RE, THINK_CLOSE, normalise_piece


def text_config(cfg):
    """Multimodal configs nest the language model dims under text_config."""
    return getattr(cfg, "text_config", cfg)


def load_model(model_path, gpu_mem=None, cpu_mem="40GiB", trust_remote_code=True):
    """Load in bf16, optionally split across GPU and CPU.

    Never quantize here. CETT measures activation magnitude, which is exactly
    what quantization perturbs.
    """
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=trust_remote_code
    )
    kwargs = dict(torch_dtype=torch.bfloat16, trust_remote_code=trust_remote_code)
    if gpu_mem:
        kwargs["device_map"] = "auto"
        kwargs["max_memory"] = {0: gpu_mem, "cpu": cpu_mem}
    else:
        kwargs["device_map"] = {"": "cpu"}
    model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
    model.eval()
    return model, tokenizer


class CETTManager:
    """Captures each text layer's down_proj input and that layer's output norm.

    CETT(layer, token, neuron) = |a| * ||W[:, neuron]|| / ||layer_output||

    i.e. how much of the layer's contribution to the residual stream that one
    neuron is responsible for at that token position.
    """

    def __init__(self, model):
        self.acts = {}
        self.out_norms = {}
        self.hooks = []
        self.weight_norms = {}
        self.layer_ids = []

        for name, module in model.named_modules():
            if "down_proj" not in name:
                continue
            m = TEXT_LAYER_RE.search(name)
            if not m:
                continue  # vision tower or other non-decoder MLP
            idx = int(m.group(1))
            self.layer_ids.append(idx)
            self.hooks.append(module.register_forward_hook(self._make_hook(idx)))
            self.weight_norms[idx] = torch.norm(
                module.weight.data.float(), dim=0
            ).cpu()

        self.layer_ids.sort()
        if not self.layer_ids:
            raise RuntimeError(
                "No text decoder down_proj modules matched. Adjust TEXT_LAYER_RE; "
                "run preflight.py to list the module names for this model."
            )
        if set(self.layer_ids) != set(range(len(self.layer_ids))):
            raise RuntimeError(
                f"Layer indices not contiguous 0..N-1: {self.layer_ids[:8]}..."
            )

        self.n_layers = len(self.layer_ids)
        self.n_neurons = self.weight_norms[0].shape[0]

    def _make_hook(self, idx):
        def hook_fn(module, inputs, output):
            a = inputs[0].detach()
            if a.dim() == 3:
                a = a.squeeze(0)
            self.acts[idx] = a.to("cpu", torch.float32)
            o = output.detach()
            if o.dim() == 3:
                o = o.squeeze(0)
            self.out_norms[idx] = torch.norm(
                o.to("cpu", torch.float32), dim=-1, keepdim=True
            )
        return hook_fn

    def clear(self):
        self.acts.clear()
        self.out_norms.clear()

    def remove(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()

    def cett(self):
        """[layers, tokens, neurons] on CPU."""
        a = torch.stack([self.acts[i] for i in self.layer_ids]).abs()
        wn = torch.stack([self.weight_norms[i] for i in self.layer_ids])
        norms = torch.stack([self.out_norms[i] for i in self.layer_ids])
        return (a * wn.unsqueeze(1)) / (norms + 1e-8)


def build_sequence(tokenizer, question, response):
    """Exact tokens: generation prompt then the response as generated."""
    out = tokenizer.apply_chat_template(
        [{"role": "user", "content": question}],
        add_generation_prompt=True,
        return_tensors="pt",
    )
    # Depending on the transformers version and the template, this comes back
    # as a tensor, a BatchEncoding, or a plain list of ids.
    if hasattr(out, "input_ids"):
        out = out["input_ids"]
    if not torch.is_tensor(out):
        out = torch.tensor(out)
    prompt_ids = out[0] if out.dim() == 2 else out
    resp_ids = tokenizer(
        response, add_special_tokens=False, return_tensors="pt"
    )["input_ids"][0]
    return torch.cat([prompt_ids, resp_ids]), len(prompt_ids)



def find_think_end(pieces, start, end):
    """Token index just past </think>, or `start` if absent."""
    for i in range(start, end):
        if THINK_CLOSE in pieces[i]:
            return i + 1
    tail = "".join(pieces[start:end])
    if THINK_CLOSE in tail:
        cut = tail.index(THINK_CLOSE) + len(THINK_CLOSE)
        acc = 0
        for i in range(start, end):
            acc += len(pieces[i])
            if acc >= cut:
                return i + 1
    return start


def find_regions(tokenizer, full_ids, prompt_len, answer_tokens, skip_think=True):
    """{region: (start, end) or None} in token index space."""
    n = len(full_ids)
    pieces = [tokenizer.decode([t]) for t in full_ids]
    search_from = find_think_end(pieces, prompt_len, n) if skip_think else prompt_len

    regions = {
        "input": (0, prompt_len),
        "output": (search_from, n),
        "answer_tokens": None,
    }
    if answer_tokens:
        want = [normalise_piece(t) for t in answer_tokens]
        norm = [normalise_piece(p) for p in pieces]
        m = len(want)
        for i in range(search_from, n - m + 1):
            if norm[i:i + m] == want:
                regions["answer_tokens"] = (i, i + m)
                break
    return regions
