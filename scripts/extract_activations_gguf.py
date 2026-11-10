#!/usr/bin/env python3
"""
Stage 4, alternative path: extract activations via llama.cpp instead of PyTorch.

Same output as extract_activations.py -- fp16 .npy of shape [n_layers,
n_neurons] per sample, plus neuron_index.json -- so stages 5 through 9 are
unchanged. What differs is everything upstream:

    extract_activations.py       bf16 safetensors, PyTorch, ROCm or CPU, ~18GB
    extract_activations_gguf.py  Q6_K GGUF, llama.cpp, Vulkan, ~8GB

That size difference is the point. On a 16GB card the quantized path fits with
room for the KV cache; the bf16 path does not, and needs CPU offload.

The division of labour: llama-cett-dump captures the raw down_proj inputs and
output norms, this script supplies the weight column norms from the GGUF and
does the CETT arithmetic, so the maths lives in exactly one place.

Note the weights differ between paths. This measures the quantized model you
actually serve rather than the bf16 one you do not, which for a fixed-quant
deployment is arguably the more honest measurement -- but it does mean neurons
found here are calibrated to this quant. Do not mix outputs from the two paths
in one training set.

    python scripts/extract_activations_gguf.py \\
        --binary ~/llama.cpp/build/bin/llama-cett-dump \\
        --gguf ~/models/Qwen3-8B-Q6_K.gguf \\
        --tokenizer Qwen/Qwen3-8B \\
        --input_path data/answer_tokens.jsonl \\
        --ids_path data/train_qids.json \\
        --output_root data/activations \\
        --ngl 99 --batch 4096 \\
        --locations answer_tokens all_except_answer_tokens
"""

import argparse
import json
import os
import struct
import subprocess
import sys

import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from oc_constants import THINK_CLOSE, normalise_piece  # noqa: E402

MAGIC = b"CETT"
FORMAT_VERSION = 1


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--binary", default=None, help="path to llama-cett-dump (default $OC_CETT)")
    p.add_argument("--gguf", default=None, help="the model to extract from (default $OC_GGUF)")
    p.add_argument("--tokenizer", default=None,
                   help="HF repo or path. Optional: without it the tokenizer "
                        "is read from the GGUF, which avoids needing "
                        "transformers (and therefore torch) at all.")
    p.add_argument("--input_path", required=True)
    p.add_argument("--ids_path", required=True)
    p.add_argument("--output_root", required=True)
    p.add_argument("--locations", nargs="+", default=["answer_tokens"],
                   choices=["input", "output", "answer_tokens",
                            "all_except_answer_tokens"])
    p.add_argument("--method", choices=["mean", "max"], default="mean")
    p.add_argument("--ngl", type=int, default=99)
    p.add_argument("--batch", type=int, default=4096,
                   help="Must exceed the longest sequence: the tool requires a "
                        "single decode so each layer yields one record.")
    p.add_argument("--max_tokens", type=int, default=2048)
    p.add_argument("--device", help="passed through, e.g. Vulkan0")
    p.add_argument("--keep_think", action="store_true")
    p.add_argument("--loops", type=int, help="looped models (Dory): the loop count, if not the GGUF's own")
    p.add_argument("--no_trust_remote_code", action="store_true")
    return p.parse_args()


def read_tokens(path):
    """Phase-1 output: just the token ids."""
    with open(path, "rb") as f:
        blob = f.read()
    if blob[:4] != MAGIC:
        raise ValueError(f"{path} is not a cett dump")
    ver, n_tok = struct.unpack_from("<II", blob, 4)
    if ver != FORMAT_VERSION:
        raise ValueError(f"dump format v{ver}, expected v{FORMAT_VERSION}")
    return np.frombuffer(blob, dtype="<i4", count=n_tok, offset=12)


def read_aggregate(path):
    """Phase-2 output -> (token_ids, agg [n_spans, n_layers, n_ff], seen).

    `agg` holds mean or max over each span of |a| / ||layer output||. The
    weight column norm is applied by the caller: it is a per-neuron constant
    and both mean and max commute with it, so keeping it out of the C++ means
    the tool never needs the weights.
    """
    with open(path, "rb") as f:
        blob = f.read()
    if blob[:4] != MAGIC:
        raise ValueError(f"{path} is not a cett dump")
    ver, n_tok = struct.unpack_from("<II", blob, 4)
    if ver != FORMAT_VERSION:
        raise ValueError(f"dump format v{ver}, expected v{FORMAT_VERSION}")
    off = 12
    token_ids = np.frombuffer(blob, dtype="<i4", count=n_tok, offset=off)
    off += n_tok * 4
    n_spans, n_layers, n_experts, n_ff = struct.unpack_from("<iiii", blob, off)
    off += 16
    cells = n_spans * n_layers * n_experts
    counts = np.frombuffer(blob, dtype="<i4", count=cells, offset=off)
    off += cells * 4
    seen = np.frombuffer(blob, dtype=np.uint8, count=cells, offset=off)
    off += cells
    agg = np.frombuffer(blob, dtype="<f2", count=cells * n_ff, offset=off)
    # Dense models report n_experts=1, so the expert axis collapses and every
    # downstream consumer sees the shape it already expects.
    shape = (n_spans, n_layers, n_ff) if n_experts == 1 \
        else (n_spans, n_layers, n_experts, n_ff)
    view = agg.reshape(shape).astype(np.float32)
    seen = seen.reshape((n_spans, n_layers) if n_experts == 1
                        else (n_spans, n_layers, n_experts))
    return token_ids, view, seen, counts.reshape(seen.shape), n_experts


def looped_layout(gguf_path, loops=None):
    """(blocks, passes, arch) for a looped GGUF -- one whose metadata has <arch>.num_loops, like NVIDIA's Dory --
    else None. cett-dump then runs with --by-weight blocks, and records pass k of block N as layer k*blocks + N
    (docs/DORY.md), so the layer count is blocks x passes."""
    import gguf
    try:
        r = gguf.GGUFReader(gguf_path)
    except (OSError, ValueError):
        return None                 # unreadable here: not treated as looped (cett-dump reports the file itself)

    def kv(key):
        f = r.fields.get(key)
        if f is None:
            return None
        v = f.contents()
        return v.decode("utf-8") if isinstance(v, bytes) else v
    arch = kv("general.architecture")
    n = kv(f"{arch}.num_loops")
    if n is None:
        return None
    return int(kv(f"{arch}.block_count")), int(loops or n), arch


def looped_args(layout, loops_override=False):
    """cett-dump flags for a looped layout (and the loop count, when it is not the GGUF's own)."""
    if not layout:
        return []
    blocks, passes, arch = layout
    out = ["--by-weight", str(blocks)]
    if loops_override:
        out += ["--override-kv", f"{arch}.num_loops=int:{passes}"]
    return out


def layer_norms(gguf_path, n_layers, layout=None):
    """[n_layers, n_ff] column norms, in cett-dump's layer order: for a looped layout, each pass repeats the blocks'
    norms, and blocks without an MLP (attention-only) get zeros (cett-dump leaves them unseen)."""
    if not layout:
        norms = weight_col_norms(gguf_path, n_layers)
        return np.stack([norms[i] for i in range(n_layers)])
    blocks, passes, _ = layout
    norms = weight_col_norms(gguf_path, blocks, allow_missing=True)
    n_ff = max(len(v) for v in norms.values())
    base = np.stack([norms[i] if i in norms and len(norms[i]) == n_ff else np.zeros(n_ff, np.float32)
                     for i in range(blocks)])
    return np.tile(base, (passes, 1))[:n_layers]


def weight_col_norms(gguf_path, n_layers, allow_missing=False):
    """||W[:, j]|| per layer, from the same file the activations came from."""
    try:
        import gguf
    except ImportError:
        raise SystemExit("needs the gguf package: pip install gguf")
    reader = gguf.GGUFReader(gguf_path)
    by_name = {t.name: t for t in reader.tensors}
    out = {}
    for layer in range(n_layers):
        name = f"blk.{layer}.ffn_down.weight"
        t = by_name.get(name)
        if t is None:
            if allow_missing:
                continue            # a block without an MLP
            raise SystemExit(f"{name} not in {gguf_path}")
        w = gguf.quants.dequantize(t.data, t.tensor_type).astype(np.float32)
        # ffn_down is [n_embd, n_ff] once numpy reverses GGUF's dim order;
        # we want the norm down each n_ff column.
        if w.ndim != 2:
            w = w.reshape(-1, out[0].shape[0]) if out else w
        n_ff = max(w.shape)
        if w.shape[0] == n_ff:
            w = w.T
        out[layer] = np.linalg.norm(w, axis=0)
    return out


def expert_col_norms(gguf_path, n_layers, n_experts, n_ff):
    """[n_layers, n_experts, n_ff] column norms from the 3D expert tensors."""
    import gguf
    reader = gguf.GGUFReader(gguf_path)
    by_name = {t.name: t for t in reader.tensors}
    out = np.zeros((n_layers, n_experts, n_ff), dtype=np.float32)
    for layer in range(n_layers):
        name = f"blk.{layer}.ffn_down_exps.weight"
        t = by_name.get(name)
        if t is None:
            raise SystemExit(f"{name} not in {gguf_path}")
        w = gguf.quants.dequantize(t.data, t.tensor_type).astype(np.float32)
        # [n_expert, n_ff, n_embd] once numpy reverses GGUF's dim order
        w = w.reshape(n_experts, n_ff, -1)
        out[layer] = np.linalg.norm(w, axis=2)
    return out


def find_regions(pieces, prompt_len, answer_tokens, skip_think,
                 answer_text=None):
    n = len(pieces)
    search_from = prompt_len
    if skip_think:
        for i in range(prompt_len, n):
            if THINK_CLOSE in pieces[i]:
                search_from = i + 1
                break
    regions = {"input": (0, prompt_len), "output": (search_from, n),
               "answer_tokens": None}
    if not answer_tokens and answer_text:
        # No stage 2 output: locate the answer by string, which is what the
        # GGUF path does. Equivalent result, one fewer tool in the chain.
        import gguf_tokenizer
        regions["answer_tokens"] = gguf_tokenizer.find_answer_span(
            pieces, answer_text, start=search_from)
        return regions
    if answer_tokens:
        want = [normalise_piece(t) for t in answer_tokens]
        norm = [normalise_piece(p) for p in pieces]
        m = len(want)
        for i in range(search_from, n - m + 1):
            if norm[i:i + m] == want:
                regions["answer_tokens"] = (i, i + m)
                break
    return regions


def build_spans(regions, locations, n_tok):
    """-> (spans, mapping). all_except_answer_tokens needs two disjoint ranges,
    so a location may own more than one span; the token counts come back so a
    mean over the union can be re-weighted correctly."""
    spans, mapping = [], {}
    for loc in locations:
        if loc == "all_except_answer_tokens":
            ans = regions["answer_tokens"]
            if ans is None:
                continue
            a, b = ans
            out_start = regions["output"][0]
            parts = [(out_start, a), (b, n_tok)]
        else:
            r = regions[loc]
            if r is None:
                continue
            parts = [r]
        idxs = []
        for st, en in parts:
            if en > st:
                idxs.append(len(spans))
                spans.append([int(st), int(en)])
        if idxs:
            mapping[loc] = idxs
    return spans, mapping


def run_tool(args, manifest, outdir, tokenize_only, n_layers=None):
    cmd = [args.binary, "-m", args.gguf, "--manifest", manifest,
           "--outdir", outdir]
    if tokenize_only:
        # -c is required even here: without it llama.cpp reserves the
        # model's full trained context (often 128k-262k, gigabytes of KV) for a
        # pass that only tokenizes, and the OOM killer takes it.
        cmd += ["--tokenize-only", "-ngl", "0", "-c", "4096"]
    else:
        cmd += ["-ngl", str(args.ngl), "-b", str(args.batch),
                "-c", str(args.batch), "--n-layers", str(n_layers)]
        if args.method == "max":
            cmd += ["--max"]
        cmd += looped_args(getattr(args, "layout", None), getattr(args, "loops", None) is not None)
    if args.device:
        cmd += ["--device", args.device]
    print("running:", " ".join(cmd))
    return subprocess.run(cmd).returncode


class _GGufAdapter:
    """Gives the GGUF tokenizer the two methods this script uses."""

    def __init__(self, t):
        self.t = t

    def apply_chat_template(self, messages, add_generation_prompt=True,
                            tokenize=False):
        return self.t.render_chat(messages, add_generation_prompt)

    def decode(self, ids):
        return self.t.decode(ids)

    def pieces(self, ids):
        # <think> and </think> must survive: the answer is searched for after
        # </think>, and without it "York" matches inside the reasoning first.
        out = []
        for i in ids:
            raw = self.t.tokens[int(i)] if 0 <= int(i) < len(self.t.tokens) else ""
            out.append(raw if raw in ("<think>", "</think>")
                       else self.t.decode_id(i))
        return out



def _require_paths(args):
    """Fall back to the environment, then fail with a sentence, not a
    numpy traceback. An empty --gguf used to surface as
    `FileNotFoundError: ''` four frames deep inside np.memmap."""
    import os as _os
    args.gguf = args.gguf or _os.environ.get("OC_GGUF", "")
    args.binary = args.binary or _os.environ.get("OC_CETT", "")
    problems = []
    if not args.gguf:
        problems.append("--gguf is empty and $OC_GGUF is not set")
    elif not _os.path.isfile(args.gguf):
        problems.append(f"--gguf does not exist: {args.gguf}")
    if not args.binary:
        problems.append("--binary is empty and $OC_CETT is not set")
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
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(
            args.tokenizer, trust_remote_code=not args.no_trust_remote_code)
        use_gguf_tok = False
    else:
        # The GGUF carries the vocab and the chat template, so nothing here
        # needs transformers or torch.
        import gguf_tokenizer
        gt = gguf_tokenizer.load(args.gguf)
        if not gt.has_template:
            raise SystemExit(
                "this GGUF has no chat template; pass --tokenizer instead")
        tok = _GGufAdapter(gt)
        use_gguf_tok = True
        print(f"tokenizer from GGUF: {len(gt.tokens):,} tokens")

    with open(args.ids_path) as f:
        ids = json.load(f)
    targets = set(ids["t"] + ids["f"])
    with open(args.input_path, encoding="utf-8") as f:
        samples = [json.loads(line) for line in f]

    os.makedirs(args.output_root, exist_ok=True)
    tok_dir = os.path.join(args.output_root, "_toks")
    dump_dir = os.path.join(args.output_root, "_dumps")
    os.makedirs(tok_dir, exist_ok=True)
    os.makedirs(dump_dir, exist_ok=True)
    for loc in args.locations:
        os.makedirs(os.path.join(args.output_root, loc), exist_ok=True)

    skipped = {"not_target": 0, "too_long": 0, "no_answer_span": 0, "failed": 0,
               "span_not_captured": 0, "prompt_boundary_merged": 0}

    # ---- phase 0: render sequences -----------------------------------------
    wanted, m1 = [], os.path.join(args.output_root, "_manifest_toks.jsonl")
    prompt_lines = []
    with open(m1, "w", encoding="utf-8") as mf:
        for sample in samples:
            qid = next(iter(sample))
            if qid not in targets:
                skipped["not_target"] += 1
                continue
            data = sample[qid]
            prompt = tok.apply_chat_template(
                [{"role": "user", "content": data["question"]}],
                add_generation_prompt=True, tokenize=False)
            # Measured in phase 1 by tokenizing the prompt on its own with the
            # same binary, so it is in the exact tokenization that produces
            # the activations (a transformers tokenizer can disagree).
            prompt_len = None
            prompt_lines.append(json.dumps(
                {"id": f"{qid}__prompt", "text": prompt}, ensure_ascii=False))
            mf.write(json.dumps({"id": qid, "text": prompt + data["response"]},
                                ensure_ascii=False) + "\n")
            wanted.append((qid, data, prompt_len, prompt))
    print(f"{len(wanted)} sequences")

    # ---- phase 1: tokenize -------------------------------------------------
    # Spans have to be expressed in the tokenization that will produce the
    # activations, so this pass gets the ids with no forward passes.
    print("\nphase 1/2: tokenizing")
    if prompt_lines:
        m0 = os.path.join(args.output_root, "_manifest_prompts.jsonl")
        with open(m0, "w", encoding="utf-8") as pf:
            pf.write("\n".join(prompt_lines) + "\n")
        if run_tool(args, m0, tok_dir, tokenize_only=True) != 0:
            raise SystemExit("tokenization of rendered prompts failed")
    if run_tool(args, m1, tok_dir, tokenize_only=True) != 0:
        raise SystemExit("tokenization of prompt+response sequences failed")

    # ---- phase 2: spans, then the forward passes ---------------------------
    m2 = os.path.join(args.output_root, "_manifest_spans.jsonl")
    plans, n_layers = [], None
    with open(m1, encoding="utf-8") as src, open(m2, "w", encoding="utf-8") as mf:
        for (qid, data, prompt_len, prompt), line in zip(wanted, src):
            tpath = os.path.join(tok_dir, f"{qid}.toks")
            if not os.path.exists(tpath):
                skipped["failed"] += 1
                continue
            token_ids = read_tokens(tpath)
            if len(token_ids) > args.max_tokens:
                skipped["too_long"] += 1
                continue
            pieces = (tok.pieces(token_ids) if use_gguf_tok
                      else [tok.decode([int(i)]) for i in token_ids])
            ppath = os.path.join(tok_dir, f"{qid}__prompt.toks")
            if not os.path.exists(ppath):
                skipped["failed"] += 1
                continue
            pids = read_tokens(ppath)
            # Common prefix of the prompt's own ids and the full sequence's.
            # Usually the whole prompt; one short when the boundary token
            # merged with the start of the response, which is then counted
            # as response (never as prompt).
            prompt_len = 0
            limit = min(len(pids), len(token_ids))
            while prompt_len < limit and pids[prompt_len] == token_ids[prompt_len]:
                prompt_len += 1
            if prompt_len < len(pids):
                skipped["prompt_boundary_merged"] += 1
            regions = find_regions(pieces, prompt_len,
                                   data.get("answer_tokens", []),
                                   skip_think=not args.keep_think,
                                   answer_text=data.get("answer"))
            if regions["answer_tokens"] is None:
                skipped["no_answer_span"] += 1
            spans, mapping = build_spans(regions, args.locations, len(token_ids))
            if not spans:
                continue
            rec = json.loads(line)
            rec["spans"] = spans
            mf.write(json.dumps(rec, ensure_ascii=False) + "\n")
            plans.append((qid, spans, mapping))

    if not plans:
        raise SystemExit("nothing to extract; check --locations and answer spans")

    if args.tokenizer:
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(
            args.tokenizer, trust_remote_code=not args.no_trust_remote_code)
        n_layers = getattr(cfg, "text_config", cfg).num_hidden_layers
    else:
        # Layer count from the GGUF itself: no transformers, no torch.
        import gguf
        _r = gguf.GGUFReader(args.gguf)
        def _kv(key):
            fld = _r.fields.get(key)
            if fld is None:
                return None
            v = fld.contents()
            return v.decode("utf-8") if isinstance(v, bytes) else v
        n_layers = int(_kv(f"{_kv('general.architecture')}.block_count") or 0)
        if not n_layers:
            raise SystemExit(f"could not read block_count from {args.gguf}")
    args.layout = looped_layout(args.gguf, args.loops)
    if args.layout:
        n_layers = args.layout[0] * args.layout[1]
        print(f"looped model: {args.layout[0]} blocks x {args.layout[1]} passes; pass k of block N is layer k*"
              f"{args.layout[0]}+N")

    print(f"\nphase 2/2: {len(plans)} forward passes, {n_layers} layers")
    if run_tool(args, m2, dump_dir, tokenize_only=False, n_layers=n_layers) != 0:
        raise SystemExit("llama-cett-dump activation pass failed")

    # ---- assemble ----------------------------------------------------------
    col_norms, written = None, 0
    for qid, spans, mapping in tqdm(plans, desc="assembling"):
        path = os.path.join(dump_dir, f"{qid}.bin")
        if not os.path.exists(path):
            skipped["failed"] += 1
            continue
        _, agg, seen, counts, n_experts = read_aggregate(path)

        if col_norms is None:
            n_ff = agg.shape[-1]
            if n_experts > 1:
                col_norms = expert_col_norms(args.gguf, n_layers, n_experts,
                                             n_ff)
            else:
                col_norms = layer_norms(args.gguf, n_layers, args.layout)
            with open(os.path.join(args.output_root, "neuron_index.json"), "w") as f:
                meta = {"n_layers": int(n_layers), "n_neurons": int(n_ff),
                        "model_path": args.gguf,
                        "source": "llama.cpp cett-dump"}
                if args.layout:
                    meta.update(looped={"blocks": args.layout[0], "passes": args.layout[1]},
                                layer_order="layer = pass * blocks + block")
                if n_experts > 1:
                    meta["n_experts"] = int(n_experts)
                    meta["order"] = ("flat = (layer * n_experts + expert) "
                                     "* n_neurons + neuron")
                    meta["note"] = ("MoE. Feature count is n_layers * "
                                    "n_experts * n_neurons; check the "
                                    "classifier RAM estimate before training.")
                else:
                    meta["order"] = "flat = layer * n_neurons + neuron"
                json.dump(meta, f, indent=2)

        wrote = False
        for loc, idxs in mapping.items():
            parts = [agg[i] for i in idxs if seen[i].any()]
            if not parts:
                continue
            if len(parts) == 1:
                red = parts[0]
            elif args.method == "max":
                red = np.maximum.reduce(parts)
            else:
                # Re-weight by token count so the union's mean is the true mean
                # rather than the mean of two per-span means.
                counts = [spans[i][1] - spans[i][0] for i in idxs if seen[i].any()]
                total = float(sum(counts))
                red = sum(p * (c / total) for p, c in zip(parts, counts))
            # apply the per-neuron weight norm, held back until now
            np.save(os.path.join(args.output_root, loc, f"act_{qid}.npy"),
                    (red * col_norms).astype(np.float16))
            wrote = True
        if wrote:
            written += 1
        else:
            # Dumps from cett-dump builds before the ubatch fix never captured
            # tokens past position 512, so any answer there landed here.
            skipped["span_not_captured"] += 1

    print(f"\nwrote {written} samples; skipped {skipped}")
    print(f"intermediate dumps in {dump_dir} can be deleted once you are happy")


if __name__ == "__main__":
    main()
