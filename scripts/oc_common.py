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


def load_model(model_path, gpu_mem=None, cpu_mem="40GiB", trust_remote_code=False):
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

    How activations leave the GPU (`capture`), all giving the same numbers:
      sync    each layer's input is copied to the CPU as the layer runs (the
              original way; the GPU waits for every copy)
      stream  copied asynchronously into reused page-locked host buffers on a
              side stream, so the copy of layer N overlaps layer N+1's compute;
              the model's own precision is kept (bf16 is not squeezed into fp16,
              whose range it can exceed)
      device  kept on the GPU in the model's precision; CETT and the per-region
              aggregates are computed there and only those small results are
              copied. Fastest, but holds layers x tokens x neurons on the GPU
      auto    device when the GPU has room for this sequence, else stream;
              sync on the CPU
    Each layer's output norm is reduced on its device first (one number per
    token), so only that crosses the bus.
    """

    MODES = ("auto", "sync", "stream", "device")

    def __init__(self, model, capture="sync"):
        if capture not in self.MODES:
            raise ValueError(f"capture: one of {', '.join(self.MODES)}")
        self.capture = capture
        self.acts = {}
        self.out_norms = {}
        self.hooks = []
        self.weight_norms = {}
        self._wn_dev = {}
        self.layer_ids = []
        self._pinned = {}
        self._streams = {}          # device -> side stream for the copies
        self._mode = "sync"

        for name, module in model.named_modules():
            if "down_proj" not in name:
                continue
            m = TEXT_LAYER_RE.search(name)
            if not m:
                continue  # vision tower or other non-decoder MLP
            idx = int(m.group(1))
            self.layer_ids.append(idx)
            self.hooks.append(module.register_forward_hook(self._make_hook(idx)))
            wn = torch.norm(module.weight.data.float(), dim=0)
            self.weight_norms[idx] = wn.cpu()
            self._wn_dev[idx] = wn

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
        self._on_gpu = self._wn_dev[self.layer_ids[0]].is_cuda

    # -- choosing a mode for the next forward pass
    def prepare(self, n_tokens):
        """Call before each forward pass (with its token count) so auto can pick device or stream."""
        mode = self.capture
        if not self._on_gpu:
            mode = "sync"
        elif mode == "auto":
            need = self.n_layers * n_tokens * self.n_neurons * 2 * 1.3   # bf16/fp16 activations + headroom
            try:
                free = min(torch.cuda.mem_get_info(d)[0] for d in {w.device for w in self._wn_dev.values()})
            except Exception:
                free = 0
            mode = "device" if need < 0.6 * free else "stream"
        self._mode = mode
        return mode

    def pause(self):
        """Skip capturing (e.g. while generating) until the next prepare()."""
        self._mode = "off"

    def _pinned_buf(self, idx, shape, dtype):
        b = self._pinned.get(idx)
        n = 1
        for x in shape:
            n *= x
        if b is None or b.numel() < n or b.dtype != dtype:
            b = torch.empty(max(n, 1), dtype=dtype, pin_memory=True)
            self._pinned[idx] = b
        return b[:n].view(shape)

    def _make_hook(self, idx):
        def hook_fn(module, inputs, output):
            if self._mode == "off":
                return
            a = inputs[0].detach()
            if a.dim() == 3:
                a = a.squeeze(0)
            o = output.detach()
            if o.dim() == 3:
                o = o.squeeze(0)
            norm = torch.norm(o.float(), dim=-1, keepdim=True)        # reduced where it was computed
            if self._mode == "device":
                self.acts[idx] = a
                self.out_norms[idx] = norm
            elif self._mode == "stream":
                cur = torch.cuda.current_stream(a.device)
                s = self._streams.get(a.device)
                if s is None:
                    s = self._streams[a.device] = torch.cuda.Stream(device=a.device)
                s.wait_stream(cur)
                with torch.cuda.stream(s):
                    buf = self._pinned_buf(idx, tuple(a.shape), a.dtype)
                    buf.copy_(a, non_blocking=True)
                    nbuf = torch.empty(norm.shape, dtype=norm.dtype, pin_memory=True)
                    nbuf.copy_(norm, non_blocking=True)
                a.record_stream(s)                  # keep the GPU memory until the copy is done
                norm.record_stream(s)
                self.acts[idx] = buf
                self.out_norms[idx] = nbuf
            else:
                self.acts[idx] = a.to("cpu", torch.float32)
                self.out_norms[idx] = norm.to("cpu")
        return hook_fn

    def clear(self):
        self.acts.clear()
        self.out_norms.clear()

    def remove(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()

    def _sync(self):
        if self._mode == "stream":
            for d in {w.device for w in self._wn_dev.values()}:
                torch.cuda.synchronize(d)

    def _layer(self, i, tokens=None):
        """One layer's CETT [tokens, neurons] as float32, where its activations are."""
        a, nrm = self.acts[i], self.out_norms[i]
        if tokens is not None:
            a, nrm = a[tokens], nrm[tokens]
        wn = self._wn_dev[i] if a.is_cuda else self.weight_norms[i]
        return (a.float().abs() * wn) / (nrm.float() + 1e-8)

    def cett(self):
        """[layers, tokens, neurons] on CPU."""
        self._sync()
        return torch.stack([self._layer(i).cpu() for i in self.layer_ids])

    def aggregate(self, spans, method="mean"):
        """[layers, neurons] on CPU: CETT over the token spans [(start, end), ...] joined, by mean or max.
        Only the spans are computed, on whichever device holds the activations."""
        self._sync()
        idx = torch.cat([torch.arange(s, e) for s, e in spans]) if spans else torch.arange(0)
        if idx.numel() == 0:
            return None
        rows = []
        for i in self.layer_ids:
            t = idx.to(self.acts[i].device)
            c = self._layer(i, t)
            rows.append((c.mean(0) if method == "mean" else c.max(0)[0]).cpu())
        return torch.stack(rows)


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
