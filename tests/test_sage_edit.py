"""SAGE: gated activation edits that leave a protected subspace untouched."""
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))
torch = pytest.importorskip("torch")
import sage_edit as sg  # noqa: E402

LLAMA = Path(os.environ.get("OC_LLAMA") or "~/llama.cpp").expanduser()


def test_edit_never_moves_the_protected_component():
    torch.manual_seed(0)
    d, H = 12, 3
    R = torch.linalg.qr(torch.randn(d, 2))[0]
    e = sg.SageEdits(1, H, d, R.unsqueeze(0))
    with torch.no_grad():
        e.a.normal_()
        e.m.normal_()
    e.training = False
    z = torch.randn(5, H, d)
    z2 = e.edit(0, z)
    assert torch.allclose(z2 @ R, z @ R, atol=1e-5)          # along R: identical
    assert not torch.allclose(z2, z)                          # elsewhere: edited
    e.project_()
    assert float((e.a[0] @ R).abs().max().detach()) < 1e-5 and float((e.m[0] @ R).abs().max().detach()) < 1e-5
    # closed gates: no edit at all
    with torch.no_grad():
        e.la.fill_(-20)
        e.lm.fill_(-20)
    assert torch.allclose(e.edit(0, z), z, atol=1e-6)
    assert float(e.l0().detach()) < 1e-3


def test_train_chat_export(tmp_path):
    pytest.importorskip("transformers")
    pytest.importorskip("sentencepiece")
    import tiny_models
    hf = tiny_models.make_tiny_llama(tmp_path / "tiny", layers=3)
    (tmp_path / "c.txt").write_text("\n".join(f"I am not sure about question {i}" for i in range(6)))
    (tmp_path / "b.txt").write_text("\n".join(f"The answer to question {i} is plainly this" for i in range(6)))
    data = tmp_path / "data"
    data.mkdir()
    (data / "sft.jsonl").write_text("".join(json.dumps({"messages": [{"role": "user", "content": f"add {i} and 1"},
                                                                    {"role": "assistant", "content": str(i + 1)}]})
                                            + "\n" for i in range(12)))
    out = tmp_path / "sage"
    assert sg.main(["train", "--model", str(hf), "--data", str(data), "--out", str(out), "--protect-concept",
                    str(tmp_path / "c.txt"), "--protect-baseline", str(tmp_path / "b.txt"), "--max-steps", "8",
                    "--grad-accum", "1", "--lr", "0.05", "--cpu", "--method", "pca", "--k", "2"]) == 0
    rep = json.loads((out / "sage.json").read_text())
    assert rep["protected_dims"] == 2 and rep["steps"] == 8
    assert rep["max_change_along_protected"] < 1e-3            # the invariant holds in the real model
    assert rep["val_loss_before"] is not None and rep["val_loss_after"] is not None
    assert sg.main(["chat", "--edits", str(out / "edits.pt"), "--cpu", "--max-new-tokens", "4", "hi"]) == 0
    cv = tmp_path / "sage.gguf"
    assert sg.main(["export", "--edits", str(out / "edits.pt"), "--cvector", str(cv), "--llama", str(LLAMA)]) == 0
    assert cv.stat().st_size > 0
    if not (LLAMA / "build" / "bin" / "llama-server").exists():
        pytest.skip("needs llama-server to load the control vector")
    gguf = tmp_path / "tiny.gguf"
    r = subprocess.run([sys.executable, str(LLAMA / "convert_hf_to_gguf.py"), str(hf), "--outfile", str(gguf),
                        "--outtype", "f16"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-1500:]
    import socket
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    p = subprocess.Popen([str(LLAMA / "build" / "bin" / "llama-server"), "-m", str(gguf), "--port", str(port),
                          "-c", "256", "--control-vector", str(cv)], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    try:
        ok = False
        for _ in range(100):
            if p.poll() is not None:
                break
            try:
                ok = urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1).status == 200
                if ok:
                    break
            except Exception:
                time.sleep(0.2)
        assert ok, (p.stderr.read() or b"").decode()[-1500:] if p.poll() is not None else "not ready"
    finally:
        p.terminate()
        p.wait(10)
