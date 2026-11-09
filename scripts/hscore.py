#!/usr/bin/env python3
"""
Score one reply with an H-Neuron classifier: how strongly did the neurons
associated with confident wrong answers fire while the model wrote it?

    python scripts/hscore.py --gguf model.gguf --classifier models/classifier.npz \\
        --prompt "Who wrote Hamlet?" --response "Christopher Marlowe."

One prefill over prompt + response with llama-cett-dump, CETT averaged over the
response tokens, then the classifier's linear score. Uses the GGUF's own chat
template and tokenizer, so no transformers or torch.

Read the number as a weak, relative signal. The classifier was trained on
short factual answers (the answer-token span); here it sees a whole reply,
and reported AUROCs are in the 0.65-0.75 range. Track it over many replies
(model_stats.py does) rather than trusting any single score.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
import threading

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


class HScorer:
    def __init__(self, binary: str, gguf: str, classifier: str | None, ngl: int = 99, batch: int = 4096,
                 device: str = "", timeout: float = 600, tokenizer=None):
        """`classifier` may be None: profile() then works, score() and trace() need one."""
        self.binary, self.gguf, self.ngl, self.batch, self.device = binary, gguf, ngl, batch, device
        self.timeout = timeout
        self.coef, self.intercept, self.n_layers, self.n_neurons = None, 0.0, None, None
        if classifier:
            blob = np.load(classifier)
            self.coef = np.asarray(blob["coef"], dtype=np.float32)
            self.intercept = float(blob["intercept"])
            self.n_layers = int(blob["n_layers"]) if "n_layers" in blob else None
            self.n_neurons = int(blob["n_neurons"]) if "n_neurons" in blob else None
        if self.n_layers and self.n_neurons and self.coef.size != self.n_layers * self.n_neurons:
            raise ValueError("classifier coefficient size does not match its n_layers x n_neurons "
                             "(MoE classifiers are not supported for live scoring)")
        self._norms = None
        self._tok = tokenizer        # anything with render_chat/decode/has_template
        self.lock = threading.Lock()   # one prefill at a time per model

    def _tokenizer(self):
        if self._tok is None:
            import gguf_tokenizer
            self._tok = gguf_tokenizer.load(self.gguf)
            if not self._tok.has_template:
                raise ValueError("this GGUF has no chat template; cannot render the prompt")
        return self._tok

    def _col_norms(self, n_layers):
        if self._norms is None:
            from extract_activations_gguf import weight_col_norms
            norms = weight_col_norms(self.gguf, n_layers)
            self._norms = np.stack([norms[i] for i in range(n_layers)])
        return self._norms

    def _run(self, args):
        cmd = [self.binary, "-m", self.gguf, *args]
        if self.device:
            cmd += ["--device", self.device]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=self.timeout)
        if r.returncode != 0:
            raise RuntimeError(f"cett-dump failed: {(r.stderr or r.stdout)[-400:]}")

    def _prepare(self, messages: list[dict], response: str, work: str):
        """Tokenize prompt and prompt+response with the binary; -> (text, ids, start).
        The response starts where the prompt's own tokenization ends; if the
        boundary merged into one token, back off to the common prefix."""
        from extract_activations_gguf import read_tokens
        prompt = self._tokenizer().render_chat(messages, True)
        text = prompt + response
        m1 = os.path.join(work, "t.jsonl")
        with open(m1, "w", encoding="utf-8") as f:
            f.write(json.dumps({"id": "p", "text": prompt}, ensure_ascii=False) + "\n")
            f.write(json.dumps({"id": "r", "text": text}, ensure_ascii=False) + "\n")
        self._run(["--tokenize-only", "-ngl", "0", "-c", "4096", "--manifest", m1, "--outdir", work])
        ids = read_tokens(os.path.join(work, "r.toks"))
        p_ids = read_tokens(os.path.join(work, "p.toks"))
        n_tok = len(ids)
        if n_tok > self.batch:
            raise ValueError(f"{n_tok} tokens exceeds batch {self.batch}")
        start = 0
        while start < min(len(p_ids), n_tok) and p_ids[start] == ids[start]:
            start += 1
        if start >= n_tok:
            raise ValueError("empty response region")
        return text, ids, start

    def _dump(self, text: str, spans: list, work: str):
        from extract_activations_gguf import read_aggregate
        m2 = os.path.join(work, "s.jsonl")
        with open(m2, "w", encoding="utf-8") as f:
            f.write(json.dumps({"id": "r", "text": text, "spans": spans}, ensure_ascii=False) + "\n")
        n_layers = self.n_layers or self._layers_from_gguf()
        self._run(["-ngl", str(self.ngl), "-b", str(self.batch), "-c", str(self.batch),
                   "--n-layers", str(n_layers), "--manifest", m2, "--outdir", work])
        _, agg, _seen, _counts, n_experts = read_aggregate(os.path.join(work, "r.bin"))
        if n_experts > 1:
            raise ValueError("MoE models are not supported for live scoring")
        return agg, n_layers

    def profile(self, messages: list[dict], response: str) -> dict:
        """CETT of every MLP neuron averaged over the response: {cett [L, N],
        n_tokens}, plus score and prob when a classifier is loaded. One prefill."""
        with self.lock, tempfile.TemporaryDirectory(prefix="hscore-") as work:
            text, ids, start = self._prepare(messages, response, work)
            agg, n_layers = self._dump(text, [[start, -1]], work)
        cett = (agg[0] * self._col_norms(n_layers)).astype(np.float32)
        out = {"cett": cett, "n_tokens": int(len(ids) - start)}
        if self.coef is not None:
            feats = cett.ravel()
            if feats.size != self.coef.size:
                raise ValueError(f"feature size {feats.size} != classifier {self.coef.size}: "
                                 "wrong classifier for this model?")
            s = float(feats @ self.coef + self.intercept)
            out.update(score=s, prob=1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, s)))))
        return out

    def score(self, messages: list[dict], response: str) -> dict:
        if self.coef is None:
            raise ValueError("no classifier loaded")
        p = self.profile(messages, response)
        return {"score": p["score"], "prob": p["prob"], "n_tokens": p["n_tokens"]}

    def trace(self, messages: list[dict], response: str, bins: int = 512, max_frames: int = 512) -> dict:
        """Per-token CETT over the response (one span per token, or per `stride`
        tokens when the reply is longer than `max_frames`), scored token by token.
        -> {frames [T, L, bins], scores [T] (logits), prob [T], pieces [T], stride,
            h_cells, col_weight, n_layers}"""
        from trace_sample import bin_axis, classifier_cells
        if self.coef is None:
            raise ValueError("no classifier loaded")
        with self.lock, tempfile.TemporaryDirectory(prefix="htrace-") as work:
            text, ids, start = self._prepare(messages, response, work)
            n = len(ids) - start
            stride = max(1, -(-n // max_frames))
            spans = [[start + i, min(start + i + stride, len(ids))] for i in range(0, n, stride)]
            agg, n_layers = self._dump(text, spans, work)
        trace = (agg * self._col_norms(n_layers)[None]).astype(np.float32)
        flat = trace.reshape(trace.shape[0], -1)
        if flat.shape[1] != self.coef.size:
            raise ValueError(f"feature size {flat.shape[1]} != classifier {self.coef.size}: wrong classifier?")
        scores = flat @ self.coef + self.intercept
        tok = self._tokenizer()
        resp_ids = [int(i) for i in ids[start:]]
        per = tok.pieces(resp_ids) if hasattr(tok, "pieces") else [tok.decode([i]) for i in resp_ids]
        pieces = ["".join(per[i:i + stride]) for i in range(0, n, stride)]
        cells, col_w = classifier_cells(self.coef, n_layers, trace.shape[-1], bins)
        return {"frames": bin_axis(trace, bins), "scores": scores.astype(np.float32),
                "prob": (1 / (1 + np.exp(-np.clip(scores, -60, 60)))).astype(np.float32),
                "pieces": pieces, "stride": stride, "h_cells": cells, "col_weight": col_w,
                "n_layers": n_layers, "tokens": resp_ids}

    def _layers_from_gguf(self) -> int:
        import gguf
        r = gguf.GGUFReader(self.gguf)
        arch = r.fields["general.architecture"].contents()
        arch = arch.decode() if isinstance(arch, bytes) else arch
        self.n_layers = int(r.fields[f"{arch}.block_count"].contents())
        return self.n_layers


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--binary", default=os.environ.get("OC_CETT"))
    p.add_argument("--gguf", default=os.environ.get("OC_GGUF"))
    p.add_argument("--classifier", required=True)
    p.add_argument("--prompt", help="the question (or --messages)")
    p.add_argument("--messages", help="a JSON file with the chat messages before the reply")
    p.add_argument("--response", help="the reply (or --response-file)")
    p.add_argument("--response-file")
    p.add_argument("--ngl", type=int, default=99)
    p.add_argument("--batch", type=int, default=4096)
    p.add_argument("--trace-out", help="score token by token and save the trace (.npz) for a 3D view "
                                       "elsewhere, e.g. Studio on another computer")
    a = p.parse_args(argv)
    if not a.binary or not a.gguf:
        raise SystemExit("need --binary and --gguf (or source env.sh)")
    msgs = (json.load(open(a.messages, encoding="utf-8")) if a.messages
            else [{"role": "user", "content": a.prompt or ""}])
    response = open(a.response_file, encoding="utf-8").read() if a.response_file else a.response
    if response is None:
        raise SystemExit("need --response or --response-file")
    s = HScorer(a.binary, a.gguf, a.classifier, a.ngl, a.batch)
    if a.trace_out:
        save_trace(a.trace_out, s.trace(msgs, response))
        print(json.dumps({"trace": a.trace_out}))
        return 0
    print(json.dumps(s.score(msgs, response), indent=2))
    return 0


def save_trace(path: str, r: dict) -> None:
    """A trace() result as a compact .npz (frames in float16)."""
    with open(path, "wb") as f:
        np.savez_compressed(f, frames=r["frames"].astype(np.float16), scores=r["scores"], prob=r["prob"],
                            tokens=np.asarray(r["tokens"], dtype=np.int64), col_weight=np.asarray(r["col_weight"]),
                            meta=np.frombuffer(json.dumps({"pieces": r["pieces"], "stride": r["stride"],
                                                           "h_cells": r["h_cells"], "n_layers": r["n_layers"]}
                                                          ).encode(), dtype=np.uint8))


def load_trace(data) -> dict:
    """The inverse of save_trace, from a path or bytes."""
    import io
    blob = np.load(io.BytesIO(data) if isinstance(data, (bytes, bytearray)) else data, allow_pickle=False)
    meta = json.loads(bytes(blob["meta"]).decode())
    return {"frames": blob["frames"].astype(np.float32), "scores": blob["scores"], "prob": blob["prob"],
            "tokens": [int(t) for t in blob["tokens"]], "col_weight": blob["col_weight"], **meta}


if __name__ == "__main__":
    raise SystemExit(main())
