"""Subject colour-coding: which neurons go with which subject, laid onto 3D payloads."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "viz"))
import neuron_review as nr  # noqa: E402


def make_store(tmp_path, L=4, B=32):
    s = nr.ReviewStore(tmp_path)
    rng = np.random.default_rng(0)
    for k in range(36):
        subj = ["math", "code", "writing"][k % 3]
        v = rng.normal(0, 0.05, (L, B)).astype(np.float32)
        if subj == "math":
            v[2, 3:6] += 1.0
        if subj == "code":
            v[1, 20:22] += 1.0
        s.record("m", v, source="testqa", kind="test", subjects=[subj], verdict="correct")
    return s


def test_subject_map(tmp_path):
    m = nr.subject_map(make_store(tmp_path), "m")
    assert set(m["subjects"]) == {"math", "code"}                    # writing has nothing of its own
    assert {(l, b) for l, b, _ in m["cells"]["math"]} == {(2, 3), (2, 4), (2, 5)}
    assert {(l, b) for l, b, _ in m["cells"]["code"]} == {(1, 20), (1, 21)}
    assert m["layers"] == 4 and m["bins"] == 32 and m["counts"]["math"] == 12
    assert nr.subject_map(make_store(tmp_path / "x"), "nobody")["subjects"] == []


def test_payload_carries_cell_subjects(tmp_path):
    import bloom
    frames, fields = bloom.demo_trace()
    T, L, N = frames.shape
    B = 64 if N > 64 else N
    # a map that claims every bin of layer 0 for "math"
    smap = {"subjects": ["math"], "cells": {"math": [[0, b, 1.0] for b in range(B)]}, "bins": B, "layers": L}
    blob, meta = bloom.build_payload(None, None, 97.0, 40000, demo=True, subjects=smap)
    assert meta["subjects"] == ["math"] and len(meta["cell_subject"]) == meta["cells"]
    import struct
    n = struct.unpack("<iii", blob[:12])[1]
    ly = np.frombuffer(blob[12:12 + 4 * n], "<i4")
    cs = np.array(meta["cell_subject"])
    assert (cs[ly == 0] == 0).all() and (cs[ly != 0] == -1).all()
    _, meta2 = bloom.build_payload(None, None, 97.0, 40000, demo=True, subjects={"subjects": ["x"], "cells": {},
                                                                                 "bins": B, "layers": L + 1})
    assert meta2["cell_subject"] is None                              # a map for another model shape is ignored
