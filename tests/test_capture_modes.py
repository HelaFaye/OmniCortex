"""CETTManager's capture modes (scripts/oc_common.py) give the same CETT, and aggregate() matches cett()."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))
torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("sentencepiece")


@pytest.fixture(scope="module")
def model(tmp_path_factory):
    from tiny_models import make_tiny_llama
    from transformers import AutoModelForCausalLM
    hf = make_tiny_llama(tmp_path_factory.mktemp("tiny"), layers=3, hidden=32, ff=64)
    return AutoModelForCausalLM.from_pretrained(str(hf), torch_dtype=torch.float32).eval()


IDS = torch.tensor([[1, 5, 9, 14, 3, 22, 7, 11, 4, 2]])


def run(model, capture, force=None):
    from oc_common import CETTManager
    m = CETTManager(model, capture=capture)
    try:
        m.prepare(IDS.shape[1])
        if force:
            m._mode = force                      # exercise a GPU path's bookkeeping on the CPU
        with torch.inference_mode():
            model(IDS)
        return m, m.cett()
    finally:
        m.remove()


def test_cpu_falls_back_to_sync(model):
    m, _ = run(model, "auto")
    assert m._mode == "sync"


def test_device_path_matches_sync(model):
    _, ref = run(model, "sync")
    m, dev = run(model, "device", force="device")
    assert dev.shape == ref.shape == (3, IDS.shape[1], 64)
    assert torch.allclose(dev, ref, atol=1e-6)


def test_aggregate_matches_cett_slices(model):
    from oc_common import CETTManager
    m = CETTManager(model, capture="sync")
    try:
        m.prepare(IDS.shape[1])
        with torch.inference_mode():
            model(IDS)
        full = m.cett()
        spans = [(1, 4), (6, 9)]
        sel = torch.cat([full[:, 1:4], full[:, 6:9]], dim=1)
        assert torch.allclose(m.aggregate(spans, "mean"), sel.mean(1), atol=1e-6)
        assert torch.allclose(m.aggregate(spans, "max"), sel.max(1)[0], atol=1e-6)
        assert m.aggregate([], "mean") is None
    finally:
        m.remove()


def test_pause_skips_capture(model):
    from oc_common import CETTManager
    m = CETTManager(model)
    try:
        m.pause()
        with torch.inference_mode():
            model(IDS)
        assert not m.acts
    finally:
        m.remove()


def test_bad_mode_rejected(model):
    from oc_common import CETTManager
    with pytest.raises(ValueError):
        CETTManager(model, capture="zero-copy")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_gpu_modes_agree(model):
    g = model.to("cuda")
    ids = IDS.to("cuda")
    from oc_common import CETTManager
    out = {}
    for mode in ("sync", "stream", "device"):
        m = CETTManager(g, capture=mode)
        m.prepare(ids.shape[1])
        with torch.inference_mode():
            g(ids)
        out[mode] = m.cett()
        m.remove()
    model.to("cpu")
    assert torch.allclose(out["stream"], out["sync"], atol=1e-5)
    assert torch.allclose(out["device"], out["sync"], atol=1e-5)
