"""The hardware compatibility matrix is complete, honest about evidence, and the doc matches it."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import compat  # noqa: E402


def test_matrix_complete_and_doc_in_sync():
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "compat.py"), "check"], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout


def test_rules_catch_bad_rows():
    m = {"features": {"a": "", "b": ""}, "profiles": {"x": {
        "a": {"status": "verified"}, "b": {"status": "unsupported"}, "c": {"status": "expected"}}}}
    errs = compat.problems(m, ["x", "y"])
    assert any("y: no row" in e for e in errs) and any("verified needs evidence" in e for e in errs)
    assert any("needs a note" in e for e in errs) and any("x.c: not a feature" in e for e in errs)


def test_no_gpu_cell_claims_verified_without_evidence():
    m = compat.load()
    for pid, row in m["profiles"].items():
        for f, c in row.items():
            if c["status"] == "verified":
                assert "CI" in c["evidence"] or "docs/" in c["evidence"], (pid, f)
