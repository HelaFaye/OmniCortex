#!/usr/bin/env python3
"""Export a trained classifier for OC_CLASSIFIER.

Flat little-endian float32: [n_layers * n_ff] coefficients, then one intercept.
Deliberately dumb -- the server should not need numpy or a parser to read it.

    python llama-tools/server-activations/export_classifier_bin.py \\
        models/classifier.npz models/classifier.bin --gguf model.gguf

The server streams |a| / ||layer output||, i.e. CETT without the down_proj
column norm (see oc_activations.h). The classifier was trained on full CETT,
so --gguf folds each column norm into its coefficient:

    coef_j * CETT_j  ==  (coef_j * ||W[:, j]||) * |a_j| / ||out||

and the server's per-token score is then on the classifier's own scale. Use
the GGUF the server will load (any quantization of the same weights is close;
the norms come from the dequantized tensors). Without --gguf the raw
coefficients are written and scores are only comparable to each other.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("src", help="classifier.npz from scripts/classifier.py")
    p.add_argument("dst", help="output .bin for OC_CLASSIFIER")
    p.add_argument("--gguf", help="model GGUF: fold its down_proj column norms into the weights (recommended)")
    a = p.parse_args(argv)
    blob = np.load(a.src)
    coef = np.asarray(blob["coef"], dtype=np.float64).ravel()
    intercept = float(blob["intercept"])
    n_layers = int(blob["n_layers"]) if "n_layers" in blob.files else None
    if a.gguf:
        from extract_activations_gguf import weight_col_norms
        if not n_layers:
            raise SystemExit("classifier.npz has no n_layers; re-train with scripts/classifier.py")
        norms = weight_col_norms(a.gguf, n_layers)
        wn = np.concatenate([np.asarray(norms[l], dtype=np.float64) for l in range(n_layers)])
        if wn.size != coef.size:
            raise SystemExit(f"{a.gguf} has {wn.size} down_proj columns, classifier has {coef.size}: wrong model")
        coef = coef * wn
    with open(a.dst, "wb") as f:
        f.write(coef.astype("<f4").tobytes())
        f.write(np.asarray([intercept], dtype="<f4").tobytes())
    print(f"wrote {a.dst}: {coef.size} weights + intercept ({coef.size * 4 + 4} bytes)"
          + ("" if a.gguf else "; no --gguf, so scores are not on the classifier's scale"))
    if n_layers:
        print(f"  expects a model with {n_layers} layers x {coef.size // n_layers} neurons;"
              " the server refuses a mismatch")


if __name__ == "__main__":
    main()
