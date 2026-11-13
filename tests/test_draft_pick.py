"""Draft model selection for speculative decoding (scripts/draft_pick.py)."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import draft_pick as D  # noqa: E402


def gguf_with_vocab(path: Path, tokens: list[str], model="gpt2", bos=1, eos=2, pad_mb=0.0, arch="llama"):
    gguf = pytest.importorskip("gguf")
    import numpy as np
    w = gguf.GGUFWriter(str(path), arch)
    w.add_tokenizer_model(model)
    w.add_token_list(tokens)
    w.add_bos_token_id(bos)
    w.add_eos_token_id(eos)
    w.add_tensor("pad", np.zeros(max(1, int(pad_mb * 2**20 / 4)), dtype=np.float32))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return str(path)


def test_compatibility_rules():
    base = {"model": "gpt2", "n": 5000, "bos": 1, "eos": 2, "tokens": {0: "a", 1: "b", 97 * 11: "z"}}
    assert D.compatible(base, dict(base)) == (True, "same vocabulary")
    assert not D.compatible(base, {**base, "model": "llama"})[0]
    assert D.compatible(base, {**base, "n": 5100})[0]                    # within the slack
    assert "vocab" in D.compatible(base, {**base, "n": 6000})[1]
    assert "BOS" in D.compatible(base, {**base, "eos": 3})[1]
    assert "token 1" in D.compatible(base, {**base, "tokens": {0: "a", 1: "B"}})[1]
    assert not D.compatible(base, None)[0]


def test_candidates_from_real_files(tmp_path):
    vocab = [f"t{i}" for i in range(3000)]
    big = gguf_with_vocab(tmp_path / "big.gguf", vocab, pad_mb=8)
    good = gguf_with_vocab(tmp_path / "small.gguf", vocab, pad_mb=0.8)
    tiny = gguf_with_vocab(tmp_path / "tiny.gguf", vocab, pad_mb=0.1)
    other = gguf_with_vocab(tmp_path / "other.gguf", [f"u{i}" for i in range(3000)], pad_mb=0.8)
    halfsize = gguf_with_vocab(tmp_path / "half.gguf", vocab, pad_mb=4)
    mm = gguf_with_vocab(tmp_path / "mmproj-big.gguf", vocab, pad_mb=0.5)
    c = D.candidates(big, [big, good, tiny, other, halfsize, mm])
    assert [x["name"] for x in c] == ["small.gguf", "tiny.gguf"]       # about a tenth first; the rest refused
    assert c[0]["same_arch"] and 0.05 < c[0]["ratio"] < 0.15


def test_devices_and_placement():
    text = """Available devices:
  Vulkan0: NVIDIA GeForce (16304 MiB, 15000 MiB free)
  Vulkan1: AMD Radeon Graphics (iGPU) (8192 MiB, 7000 MiB free)
"""
    devs = D.parse_devices(text)
    assert [d["name"] for d in devs] == ["Vulkan0", "Vulkan1"] and devs[1]["desc"] == "AMD Radeon Graphics (iGPU)"
    pl = D.place(devs, [], 1 * 2**30, 10 * 2**30)                       # unpinned: big on Vulkan0, draft on Vulkan1
    assert pl["device"] == "Vulkan0" and pl["draft_device"] == "Vulkan1" and pl["draft_ngl"] == 99
    pl = D.place(devs, ["Vulkan1"], 1 * 2**30)
    assert pl["draft_device"] == "Vulkan0" and pl["device"] is None
    pl = D.place(devs, ["Vulkan0"], 20 * 2**30)                          # no room anywhere else
    assert pl["draft_device"] is None and pl["draft_ngl"] == 0 and "CPU" in pl["why"]
    assert D.place(devs[:1], [], 2**30)["draft_ngl"] == 0                # one GPU: the CPU drafts


def test_studio_api(tmp_path):
    import test_studio
    gen = test_studio.studio_srv.__wrapped__(tmp_path)
    base = next(gen)
    try:
        code, r = test_studio.call(base, "/api/models/draft?model=nobody")
        assert code == 404
        code, r = test_studio.call(base, "/api/models/draft?model=coder-gguf/coder-7b-q4_k_m")
        assert code == 400 and "cannot read" in r["error"]          # the fixture's files are only a header
        code, page = test_studio.call(base, "/")
        assert "suggestDraft" in page and "Suggest a draft model and device" in page
    finally:
        try:
            next(gen)
        except StopIteration:
            pass
