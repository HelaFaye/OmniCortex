"""Fit a linear map from model A's layer to model B's (ridge regression on mean-pooled residual streams).

Pooled per text, so the two tokenizers never have to agree on tokens. The map is centred: W @ (h_a - mu_a) + mu_b.
R² on a held-out fifth of the texts says how much of B's layer A's explains: below about 0.3 the translation is
mostly noise. The same skeleton maps SAE or transcoder codes: encode both sides with their dictionaries first."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))


def pooled(model_id, layer, texts, device="auto"):
    import interventions as IV
    M = IV.load(model_id, device, "float32")
    torch = M.torch
    out = []
    for t in texts:
        ids = M.encode(t)[:, :512]
        with IV.Hooks(M) as H:
            H.on("resid_post", layer, key="h")
            with torch.no_grad():
                M.m(ids)
            out.append(H.grab["h"][0].float().mean(0).cpu().numpy())
    return np.stack(out), M.describe()


def fit(A, B, alpha=10.0, seed=0):
    n = len(A)
    idx = np.random.default_rng(seed).permutation(n)
    test, train = idx[: max(1, n // 5)], idx[max(1, n // 5):]
    mu_a, mu_b = A[train].mean(0), B[train].mean(0)
    Xa, Xb = A[train] - mu_a, B[train] - mu_b
    W = np.linalg.solve(Xa.T @ Xa + alpha * np.eye(A.shape[1]), Xa.T @ Xb).T       # [d_b, d_a]
    pred = (A[test] - mu_a) @ W.T + mu_b
    ss_res = ((B[test] - pred) ** 2).sum()
    ss_tot = ((B[test] - B[test].mean(0)) ** 2).sum()
    return W, mu_a, mu_b, float(1 - ss_res / max(ss_tot, 1e-12))


def main(argv=None):
    ap = argparse.ArgumentParser()
    for k in ("model_a", "model_b", "texts", "out"):
        ap.add_argument("--" + k.replace("_", "-"), required=True)
    ap.add_argument("--layer-a", type=int, required=True)
    ap.add_argument("--layer-b", type=int, required=True)
    ap.add_argument("--alpha", type=float, default=10.0)
    ap.add_argument("--device", default="auto")
    a = ap.parse_args(argv)
    texts = [l.strip() for l in Path(a.texts).read_text(encoding="utf-8").splitlines() if l.strip()]
    if len(texts) < 20:
        raise SystemExit("give at least 20 texts (a few hundred for a usable map)")
    A, da = pooled(a.model_a, a.layer_a, texts, a.device)
    B, db = pooled(a.model_b, a.layer_b, texts, a.device)
    W, mu_a, mu_b, r2 = fit(A, B, a.alpha)
    np.savez(a.out, W=W, mu_a=mu_a, mu_b=mu_b, meta=json.dumps({"a": a.model_a, "layer_a": a.layer_a, "b": a.model_b,
                                                               "layer_b": a.layer_b, "r2": r2, "texts": len(texts)}))
    print(json.dumps({"r2_held_out": round(r2, 4), "d_a": A.shape[1], "d_b": B.shape[1], "out": a.out,
                      "verdict": "usable" if r2 >= 0.3 else "weak: more texts, closer layers, or a lower alpha"}))


if __name__ == "__main__":
    main()
