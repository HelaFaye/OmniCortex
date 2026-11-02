#!/usr/bin/env python3
"""
Capture a time-resolved trace: CETT per token, not aggregated over a region.

The regular extractor reduces over token spans, which is what the classifier
needs and what keeps a sample to ~1MB. To watch activity evolve across a
response you need the token axis kept, and spans are arbitrary [start, end)
ranges -- so emitting one span per token gives [tokens, layers, neurons] with
no change to cett-dump at all.

Size is why this is per-sample rather than a mode of the main pipeline. One
500-token sequence on a 32x14336 model is 459 MB unbinned. --bin-neurons folds
the neuron axis by max-pooling (max, not mean: H-Neurons are under 0.1% of
neurons and averaging erases them), which brings 512 bins down to about 16 MB.

    python scripts/trace_sample.py \\
        --binary ~/llama.cpp/build/bin/llama-cett-dump \\
        --gguf ~/models/Qwen3-8B-Q6_K.gguf \\
        --tokenizer Qwen/Qwen3-8B \\
        --input_path data/consistency_samples.jsonl --qid <id> \\
        --out runs/trace-<id> --bin-neurons 512

Writes a OmniCortex session with one sample holding the full trace, so
viz/timeline.py and anything else that reads records can consume it.
"""

import argparse
import json
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "viz"))
from extract_activations_gguf import (expert_col_norms, read_aggregate,  # noqa
                                      read_tokens, weight_col_norms)
from records import Recorder  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--binary", default=None,
                   help="llama-cett-dump; defaults to $NS_CETT")
    p.add_argument("--gguf", default=None,
                   help="model file; defaults to $NS_GGUF")
    p.add_argument("--tokenizer",
                   help="Optional: without it the tokenizer and "
                        "chat template come from the GGUF.")
    p.add_argument("--input_path", required=True)
    p.add_argument("--qid", help="which sample; default the first")
    p.add_argument("--out", required=True, help="session directory")
    p.add_argument("--bin-neurons", type=int, default=512,
                   help="max-pool the neuron axis to this many bins; 0 = keep all")
    p.add_argument("--stride", type=int, default=1,
                   help="one span per N tokens; raise it for long responses")
    p.add_argument("--max_tokens", type=int, default=1024)
    p.add_argument("--ngl", type=int, default=99)
    p.add_argument("--batch", type=int, default=4096)
    p.add_argument("--classifier", help="models/classifier.npz, to score tokens")
    p.add_argument("--no_trust_remote_code", action="store_true")
    return p.parse_args()


def bin_axis(a, bins):
    """Max-pool the last axis into `bins` columns."""
    n = a.shape[-1]
    if not bins or bins >= n:
        return a
    edges = np.linspace(0, n, bins + 1).astype(int)
    out = np.empty(a.shape[:-1] + (bins,), dtype=a.dtype)
    for i in range(bins):
        lo, hi = edges[i], max(edges[i + 1], edges[i] + 1)
        out[..., i] = a[..., lo:hi].max(axis=-1)
    return out



def classifier_cells(coef, n_layers, n_neurons, bins):
    """H-neurons (positive classifier weights) as (layer, column) cells in the
    trace's binned layout, and the per-column weight (the largest positive
    weight in each bin) that the viewers use to order columns."""
    w = np.clip(np.asarray(coef, dtype=np.float32).reshape(n_layers, n_neurons), 0, None)
    col_w = bin_axis(w, bins)
    cells = [[int(l), int(c)] for l, c in zip(*np.nonzero(col_w > 0))]
    return cells, col_w


def _require_paths(args):
    """Fall back to the environment, then fail with a sentence, not a
    numpy traceback. An empty --gguf used to surface as
    `FileNotFoundError: ''` four frames deep inside np.memmap."""
    import os as _os
    args.gguf = args.gguf or _os.environ.get("NS_GGUF", "")
    args.binary = args.binary or _os.environ.get("NS_CETT", "")
    problems = []
    if not args.gguf:
        problems.append("--gguf is empty and $NS_GGUF is not set")
    elif not _os.path.isfile(args.gguf):
        problems.append(f"--gguf does not exist: {args.gguf}")
    if not args.binary:
        problems.append("--binary is empty and $NS_CETT is not set")
    elif not _os.access(args.binary, _os.X_OK):
        problems.append(f"--binary is not executable: {args.binary}")
    if problems:
        raise SystemExit("\n".join(problems) +
                         "\n\nRun `source env.sh` in this shell first, or pass "
                         "the paths explicitly.")


def main():
    args = parse_args()
    _require_paths(args)
    if args.tokenizer:
        from transformers import AutoConfig, AutoTokenizer
        trc = not args.no_trust_remote_code
        tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=trc)
        cfg = AutoConfig.from_pretrained(args.tokenizer, trust_remote_code=trc)
        tcfg = getattr(cfg, "text_config", cfg)
        n_layers = tcfg.num_hidden_layers
        _gguf_tok = None
    else:
        # The GGUF carries the vocab, the chat template and the layer count,
        # so nothing here needs transformers or torch.
        import gguf_tokenizer
        _gguf_tok = gguf_tokenizer.load(args.gguf)
        if not _gguf_tok.has_template:
            raise SystemExit("this GGUF has no chat template; pass --tokenizer")

        class _Adapter:
            def __init__(self, t):
                self.t = t

            def apply_chat_template(self, messages, add_generation_prompt=True,
                                    tokenize=False):
                return self.t.render_chat(messages, add_generation_prompt)

            def decode(self, ids):
                return self.t.decode(ids)

        tok = _Adapter(_gguf_tok)
        import gguf as _g
        _r = _g.GGUFReader(args.gguf)

        def _kv(k):
            f = _r.fields.get(k)
            if f is None:
                return None
            try:
                v = f.contents()
                return v.decode("utf-8") if isinstance(v, bytes) else v
            except Exception:
                return None

        _arch = _kv("general.architecture")
        n_layers = int(_kv(f"{_arch}.block_count") or 0)
        if not n_layers:
            raise SystemExit(f"could not read block_count from {args.gguf}")
        print(f"tokenizer from GGUF: {len(_gguf_tok.tokens):,} tokens, "
              f"{n_layers} layers")

    rec = None
    with open(args.input_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            qid = next(iter(obj))
            if args.qid is None or qid == args.qid:
                rec, args.qid = obj[qid], qid
                break
    if rec is None:
        raise SystemExit(f"{args.qid or 'no sample'} not found in {args.input_path}")

    prompt = tok.apply_chat_template(
        [{"role": "user", "content": rec["question"]}],
        add_generation_prompt=True, tokenize=False)
    text = prompt + rec["response"]

    os.makedirs(args.out, exist_ok=True)
    work = os.path.join(args.out, "_work")
    os.makedirs(work, exist_ok=True)

    # Pass 1: token ids, so the spans are expressed in the tokenization that
    # will produce the activations.
    m1 = os.path.join(work, "toks.jsonl")
    with open(m1, "w", encoding="utf-8") as f:
        f.write(json.dumps({"id": args.qid, "text": text},
                           ensure_ascii=False) + "\n")
    base = [args.binary, "-m", args.gguf]
    # -c is required even here: without it llama.cpp reserves the model's full
    # trained context as KV cache (8 GB on this model) for a pass that only
    # tokenizes, and the OOM killer takes it.
    subprocess.run(base + ["--tokenize-only", "-ngl", "0", "-c", "4096",
                           "--manifest", m1, "--outdir", work], check=True)
    ids = read_tokens(os.path.join(work, f"{args.qid}.toks"))
    n_tok = len(ids)
    if n_tok > args.max_tokens:
        raise SystemExit(f"{n_tok} tokens exceeds --max_tokens {args.max_tokens}")

    spans = [[t, min(t + args.stride, n_tok)]
             for t in range(0, n_tok, args.stride)]
    print(f"{n_tok} tokens -> {len(spans)} spans (stride {args.stride})")

    m2 = os.path.join(work, "spans.jsonl")
    with open(m2, "w", encoding="utf-8") as f:
        f.write(json.dumps({"id": args.qid, "text": text, "spans": spans},
                           ensure_ascii=False) + "\n")
    subprocess.run(base + ["-ngl", str(args.ngl), "-b", str(args.batch),
                           "-c", str(args.batch), "--n-layers", str(n_layers),
                           "--manifest", m2, "--outdir", work], check=True)

    _, agg, seen, counts, n_experts = read_aggregate(
        os.path.join(work, f"{args.qid}.bin"))
    # agg is [n_spans, n_layers, (experts,) n_neurons]; the span axis is time.
    if n_experts > 1:
        agg = agg.reshape(agg.shape[0], n_layers, -1)
        col = expert_col_norms(args.gguf, n_layers, n_experts,
                               agg.shape[-1] // n_experts).reshape(n_layers, -1)
    else:
        col = np.stack([weight_col_norms(args.gguf, n_layers)[l]
                        for l in range(n_layers)])
    trace = (agg * col[None, :, :]).astype(np.float32)
    print(f"trace {trace.shape} ({trace.nbytes / 1e6:.0f} MB before binning)")

    scores, h_cells, col_weight = None, None, None
    if args.classifier:
        blob = np.load(args.classifier)
        coef, intercept = blob["coef"], float(blob["intercept"])
        flat = trace.reshape(trace.shape[0], -1)
        if flat.shape[1] == coef.shape[0]:
            scores = flat @ coef + intercept
            prob = 1 / (1 + np.exp(-np.clip(scores, -60, 60)))
            print(f"per-token scores: min {scores.min():.3f}, max {scores.max():.3f}; "
                  f"{int((prob >= 0.5).sum())} of {len(prob)} tokens at p >= 0.5")
            if n_experts <= 1:
                h_cells, col_weight = classifier_cells(coef, n_layers, trace.shape[-1], args.bin_neurons)
        else:
            print(f"classifier has {coef.shape[0]} weights but the trace has "
                  f"{flat.shape[1]}; skipping scores")

    binned = bin_axis(trace, args.bin_neurons)
    print(f"binned  {binned.shape} ({binned.nbytes / 1e6:.1f} MB)")

    pieces = (_gguf_tok.pieces(ids) if _gguf_tok is not None
              else [tok.decode([int(i)]) for i in ids])
    meta = {"model": os.path.basename(args.gguf),
            "quant": None, "fingerprint": None,
            "n_layers": n_layers, "n_neurons": binned.shape[-1],
            "kind": "trace", "stride": args.stride,
            "bin_neurons": args.bin_neurons,
            "n_neurons_unbinned": trace.shape[-1]}
    with Recorder(args.out, meta) as r:
        # Stored as one array with the span axis first; readers treat axis 0 as
        # time. `agg` keeps the name so existing consumers can still open it.
        r.add(args.qid, agg=binned.reshape(-1, binned.shape[-1]),
              tokens=ids, scores=scores,
              kind="trace", n_frames=binned.shape[0],
              n_layers=n_layers, pieces=pieces,
              verdict=rec.get("judge"), question=rec["question"],
              h_cells=h_cells, col_weight=None if col_weight is None else col_weight.tolist())
        r.event("trace", qid=args.qid, frames=binned.shape[0],
                tokens=n_tok, stride=args.stride)
    print(f"\nwrote {args.out}\n  python viz/timeline.py {args.out}")


if __name__ == "__main__":
    main()
