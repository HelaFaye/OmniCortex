"""CLI safety defaults for the model preflight utility."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import preflight  # noqa: E402


def test_remote_code_disabled_by_default(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["preflight.py", "--model_path", "example/model"])
    assert preflight.parse_args().trust_remote_code is False


def test_remote_code_requires_explicit_opt_in(monkeypatch):
    monkeypatch.setattr(sys, "argv", [
        "preflight.py", "--model_path", "example/model", "--trust-remote-code"
    ])
    assert preflight.parse_args().trust_remote_code is True


def test_legacy_remote_code_flag_remains_supported(monkeypatch):
    monkeypatch.setattr(sys, "argv", [
        "preflight.py", "--model_path", "example/model", "--trust_remote_code"
    ])
    assert preflight.parse_args().trust_remote_code is True
