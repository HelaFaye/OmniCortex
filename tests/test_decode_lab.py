"""Decoding lab: SLED's logit evolution, adaptive steering, and the OpenAI-compatible variants server."""
import json
import sys
import threading
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))
torch = pytest.importorskip("torch")
import decode_lab as dl  # noqa: E402


def test_sled_follows_the_layers_evolution():
    final = torch.tensor([3.0, 2.9, 1.0, 0.0, -1.0, -2.0, -3.0, -4.0])
    # the earlier layer favoured token 1; going up the layers the model moved toward token 0.
    # SLED reads that trend as the latent answer and continues it: token 0's narrow lead grows.
    early = [torch.tensor([1.0, 3.0, 1.0, 0.0, -1.0, -2.0, -3.0, -4.0])]
    out = dl.sled_logits(final, early, alpha=5.0, k=4)
    assert torch.isinf(out[4:]).all() and out[4:].lt(0).all()          # outside the top k: dropped
    assert float(out[0] - out[1]) > float(final[0] - final[1]) + 1
    # the opposite trend (layers moving toward token 1) turns the choice to token 1
    back = dl.sled_logits(torch.tensor([3.0, 2.9, 1.0, 0.0]), [torch.tensor([5.0, 1.0, 1.0, 0.0])], alpha=5.0, k=4)
    assert int(back.argmax()) == 1
    # layers that agree with the final logits carry no information: top k unchanged
    same = dl.sled_logits(final, [final.clone()], alpha=5.0, k=4)
    assert torch.allclose(same[:4], final[:4])


def test_vectors_from_concept_edit(tmp_path):
    basis = torch.zeros(4, 6, 2)
    basis[2, 3, 0] = 2.0                      # after layer 1, first direction: e3
    torch.save({"basis": basis}, tmp_path / "directions.pt")
    v = dl.load_vector(str(tmp_path / "directions.pt"), 0, layer=1)
    assert torch.allclose(v, torch.eye(6)[3])


@pytest.fixture(scope="module")
def lab(tmp_path_factory):
    pytest.importorskip("transformers")
    pytest.importorskip("sentencepiece")
    import tiny_models
    tmp = tmp_path_factory.mktemp("dl")
    hf = tiny_models.make_tiny_llama(tmp / "tiny", layers=4)
    from transformers import AutoConfig
    d = AutoConfig.from_pretrained(hf).hidden_size
    vec = torch.randn(d, generator=torch.Generator().manual_seed(0))
    torch.save(vec, tmp / "v.pt")
    variants = {"plain": {}, "sled": {"sled": {"alpha": 2, "k": 5, "layers": "all"}},
                "steer": {"steer": {"vectors": str(tmp / "v.pt"), "layer": 1, "alpha": 8.0}},
                "adaptive": {"steer": {"vectors": str(tmp / "v.pt"), "layer": 1, "alpha": 8.0, "adapt": "spread",
                                       "target": 0.5}},
                "attn": {"steer": {"vectors": str(tmp / "v.pt"), "layer": 1, "alpha": 8.0, "adapt": "attention"}}}
    return dl.Lab(str(hf), variants, "float32"), hf


def test_plain_variant_is_greedy_decoding(lab):
    lb, hf = lab
    msgs = [{"role": "user", "content": "Count: one two"}]
    r = lb.generate(msgs, "plain", max_tokens=8)
    ids = lb.tok(lb.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True), return_tensors="pt",
                 add_special_tokens=False)["input_ids"]
    with torch.no_grad():
        ref = lb.model.generate(ids, max_new_tokens=8, do_sample=False, pad_token_id=lb.tok.eos_token_id)
    want = lb.tok.decode(ref[0, ids.shape[1]:], skip_special_tokens=True)
    assert want.startswith(r["text"]) and len(r["spread"]) >= 1
    assert all(a == 0 for a in r["alpha"])


def test_variants_differ_and_adapt(lab):
    lb, _ = lab
    msgs = [{"role": "user", "content": "Tell me a story"}]
    plain = lb.generate(msgs, "plain", max_tokens=12)
    steer = lb.generate(msgs, "steer", max_tokens=12)
    assert steer["text"] != plain["text"] and set(steer["alpha"]) == {8.0}
    ad = lb.generate(msgs, "adaptive", max_tokens=12)
    # alpha follows the previous token's spread: alpha * clip(u / 0.5, 0, 2)
    for a, u in zip(ad["alpha"], ad["spread"]):
        assert 0 <= a <= 16.0
    assert len(set(ad["alpha"])) > 1
    at = lb.generate(msgs, "attn", max_tokens=6)
    assert all(0 <= a <= 16.0 for a in at["alpha"]) and any(a > 0 for a in at["alpha"])
    sl = lb.generate(msgs, "sled", max_tokens=12)
    assert sl["n_out"] >= 1
    with pytest.raises(KeyError):
        lb.generate(msgs, "nope")


def test_openai_server(lab):
    lb, _ = lab
    srv = ThreadingHTTPServer(("127.0.0.1", 0), dl.make_handler(lb))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_port}"
    try:
        ms = json.loads(urllib.request.urlopen(base + "/v1/models").read())
        assert [m["id"] for m in ms["data"]] == ["plain", "sled", "steer", "adaptive", "attn"]
        body = {"model": "sled", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 5}
        req = urllib.request.Request(base + "/v1/chat/completions", json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
        r = json.loads(urllib.request.urlopen(req).read())
        assert r["model"] == "sled" and r["usage"]["completion_tokens"] <= 5 and "spread" in r["decode_lab"]
        req = urllib.request.Request(base + "/v1/chat/completions", json.dumps({**body, "stream": True}).encode(),
                                     {"Content-Type": "application/json"})
        lines = [ln.decode().strip() for ln in urllib.request.urlopen(req) if ln.strip()]
        assert lines[-1] == "data: [DONE]"
        streamed = "".join(json.loads(ln[5:])["choices"][0]["delta"].get("content", "") for ln in lines[:-1])
        assert streamed == r["choices"][0]["message"]["content"]
    finally:
        srv.shutdown()


def test_variant_specs_are_checked():
    assert dl.parse_variants([]) == {"plain": {}}
    with pytest.raises(SystemExit):
        dl.parse_variants(["x={\"beam\": 4}"])
    with pytest.raises(SystemExit):
        dl.parse_variants(["x={\"steer\": {\"alpha\": 1}}"])
