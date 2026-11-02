"""Flagging semantics shared by the viewers, and the --demo payload."""
import struct
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "viz"))
sys.path.insert(0, str(ROOT / "scripts"))
import bloom  # noqa: E402
import timeline  # noqa: E402


def _field(T=50, L=4, N=32, seed=0):
    rng = np.random.default_rng(seed)
    return rng.gamma(2.0, 0.02, size=(T, L, N)).astype(np.float32)


def test_clean_reply_has_no_flags_but_relative_mode_always_does():
    frames = _field()
    clean = {"scores": (np.random.default_rng(1).normal(-3.0, 0.5, 50)).tolist()}   # p ~ 0.05 throughout
    mask = np.zeros((4, 32), bool)
    mask[2, :4] = True
    a, f, info = timeline.classify_frames(frames, clean, mask, 90.0)
    assert info["mode"] == "absolute" and not info["flagged_tokens"].any() and not f.any()
    _, f_rel, info_rel = timeline.classify_frames(frames, clean, mask, 90.0, relative_z=1.0)
    assert info_rel["flagged_tokens"].sum() > 0          # why relative is never the default


def test_flags_follow_probability_and_cells_need_a_profile():
    frames = _field()
    s = np.full(50, -3.0)
    s[20:23] = 2.0                                       # p ~ 0.88 on three tokens
    a, f, info = timeline.classify_frames(frames, {"scores": s.tolist()}, None, 90.0)
    assert list(np.nonzero(info["flagged_tokens"])[0]) == [20, 21, 22]
    assert not f.any() and info["cells"].startswith("none")   # no profile: no flagged cells
    mask = timeline.resolve_mask(None, {"h_cells": [[1, 3], [2, 5]]}, (4, 32))
    frames[20:23, 1, 3] = 5.0
    _, f, _ = timeline.classify_frames(frames, {"scores": s.tolist()}, mask, 90.0)
    assert f[20:23, 1, 3].all() and f.sum() == 3
    a, f, info = timeline.classify_frames(frames, {}, mask, 90.0)
    assert info["mode"] == "unscored" and not f.any()
    _, _, sm = timeline.classify_frames(frames, {"scores": s.tolist()}, None, 90.0, smooth=3)
    assert sm["prob"][19] > info_prob_of(-3.0)          # smoothing spreads risk to neighbours


def info_prob_of(logit):
    return 1 / (1 + np.exp(-logit))


def test_demo_payload_layout_flags_and_band():
    blob, meta = bloom.build_payload(None, None, 97.0, 120000, demo=True)
    T, N, L = struct.unpack("<iii", blob[:12])
    assert (T, N, L) == (meta["frames"], meta["cells"], meta["layers"])
    assert len(blob) == 12 + 8 * N + 4 * T * N + T * N        # the layout both clients decode
    assert "synthetic" in meta["model"] and meta["mode"] == "absolute"
    flagged = {meta["labels"][i].strip() for i in meta["flagged"]}
    assert {"moved", "Lyon"} & flagged and flagged <= {"moved", "to", "Lyon", "1923", "1931", "in", "it", "was", ","}
    assert all(meta["prob"][i] >= 0.5 for i in meta["flagged"])
    # weight order: H-neuron cells sit in the leftmost columns of their layer
    o = 12
    xs = np.frombuffer(blob[o + 4 * N:o + 8 * N], "<i4")
    state = np.frombuffer(blob[o + 8 * N + 4 * T * N:], "u1").reshape(T, N)
    hot_cols = xs[(state == 2).any(axis=0)]
    assert hot_cols.size and hot_cols.max() < meta["h_band"] and meta["order"] == "weight"


def test_clients_share_sizes_and_rate():
    import re
    js = (ROOT / "viz" / "bloom.py").read_text()
    gd = (ROOT / "plugins" / "godot-client" / "client" / "main.gd").read_text()
    for name in ("SIZE_IDLE", "SIZE_ACTIVE", "SIZE_FLAG", "FPS"):
        a = float(re.search(rf"{name} = ([\d.]+)", js).group(1))
        b = float(re.search(rf"const {name} := ([\d.]+)", gd).group(1))
        assert a == b, name
    assert "billboard_keep_scale = true" in (ROOT / "plugins" / "godot-client" / "client" / "main.tscn").read_text()


def test_default_theme_is_dark_blue_and_red_orange():
    import bloom
    import timeline
    t = bloom.THEMES["dark"]
    assert (t["active"], t["halluc"]) == ("#3D8BFF", "#FF4D1A")
    for mod in (bloom, timeline):
        src = open(mod.__file__).read()
        assert 'p.add_argument("--theme", default="dark"' in src
    studio_src = "".join((Path(bloom.__file__).parent / f).read_text() for f in ("studio.py", "studio_page.py"))
    assert 'or "dark"' in studio_src and "localStorage.getItem('ns-theme')||'dark'" in studio_src
