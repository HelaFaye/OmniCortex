"""Carry a direction from A's layer into B's through a fitted bridge: W @ v, normalised (a direction, not a state)."""
import argparse
import json

import numpy as np


def translate(bridge, v):
    W = bridge["W"]
    v = np.asarray(v, dtype=np.float64).reshape(-1)
    if v.shape[0] != W.shape[1]:
        raise ValueError(f"the vector has {v.shape[0]} values; model A's layer has {W.shape[1]}")
    u = W @ v
    return u / max(np.linalg.norm(u), 1e-12)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--bridge", required=True)
    ap.add_argument("--vector", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    br = np.load(a.bridge)
    u = translate(br, np.load(a.vector))
    np.save(a.out, u.astype(np.float32))
    meta = json.loads(str(br["meta"]))
    print(json.dumps({"out": a.out, "into": f"{meta['b']} layer {meta['layer_b']}", "bridge_r2": meta["r2"]}))


if __name__ == "__main__":
    main()
