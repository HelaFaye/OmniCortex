"""scripts/safe_load.py: PyTorch files load with the weights-only loader, so a .pt from elsewhere can't run code."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
torch = pytest.importorskip("torch")
import safe_load  # noqa: E402


class Payload:
    """Stands in for an object a malicious pickle would build."""


def test_omnicortex_files_load(tmp_path):
    p = tmp_path / "v.pt"
    torch.save({"names": ["joy", "fear"], "layers": [3, 5], "vectors": torch.ones(2, 2, 4).half(),
                "cfg": {"d_in": 4, "arch": "relu"}, "k": 32}, p)
    d = safe_load.torch_load(p)
    assert d["names"] == ["joy", "fear"] and d["vectors"].shape == (2, 2, 4)


def test_other_objects_are_refused_unless_trusted(tmp_path, monkeypatch):
    p = tmp_path / "x.pt"
    torch.save({"obj": Payload()}, p)
    monkeypatch.delenv("OC_TRUST_PICKLE", raising=False)
    with pytest.raises(ValueError, match="OC_TRUST_PICKLE"):
        safe_load.torch_load(p)
    monkeypatch.setenv("OC_TRUST_PICKLE", "1")
    assert isinstance(safe_load.torch_load(p)["obj"], Payload)
