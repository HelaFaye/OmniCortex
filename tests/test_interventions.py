"""Component interventions (scripts/interventions.py) on a tiny random Llama."""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))
torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("sentencepiece")
import interventions as IV  # noqa: E402


@pytest.fixture(scope="module")
def M(tmp_path_factory):
    from tiny_models import make_tiny_llama
    hf = make_tiny_llama(tmp_path_factory.mktemp("tiny"), layers=3, hidden=32, ff=64)
    return IV.load(str(hf), "cpu", "float32")


PROMPT = "the cat sat on the mat and the dog"


def pair(M):
    """Two prompts of equal token length that differ in one word, and two answer token ids."""
    from tiny_models import WORDS
    enc = {w: M.tok("the " + w)["input_ids"] for w in WORDS}
    a, b = next((x, y) for x in WORDS for y in WORDS if x < y and len(enc[x]) == len(enc[y])
                and sum(i != j for i, j in zip(enc[x], enc[y])) == 1)
    clean, corrupt = f"the {a}", f"the {b}"
    t = int(M.logits(M.encode(clean))[-1].argmax())
    alt = int(M.logits(M.encode(corrupt))[-1].argmax())
    return clean, corrupt, f"#{t}", f"#{alt if alt != t else (t + 1) % M.W_U.shape[0]}"


def test_trace_attributions_add_up_to_the_logit(M):
    r = IV.trace(M, PROMPT, top_k=10_000)
    assert r["schema_version"] == 1 and r["operation"] == "trace" and r["model"]["n_layers"] == 3
    ids = M.encode(PROMPT)
    H = IV.Hooks(M)
    with H:
        H.on("resid_pre", 0, key="emb").on("resid_post", 2, key="final")
        logits = M.logits(ids)
    read = M.readout(H.grab["final"][0, -1])
    tid = r["target_token_id"]
    emb = float(read(H.grab["emb"][0, -1])[tid])
    heads = sum(h["logit_attribution"] for h in r["top_heads"])
    neurons = sum(n["logit_attribution"] for n in r["top_neurons"])
    assert len(r["top_heads"]) == 3 * M.n_heads and len(r["top_neurons"]) == 3 * 64
    assert heads + neurons + emb == pytest.approx(float(logits[-1, tid]), abs=1e-3)   # exact for Llama: no biases


def test_activations_and_patch(M):
    a = IV.activations(M, PROMPT)
    assert len(a["layers"]) == 3 and set(a["layers"][0]) >= {"resid_post", "attn_out", "mlp_out", "mlp_post"}
    r = IV.patch(M, PROMPT, 1, "mlp_out")
    assert r["logit_delta"] != 0 and r["positions"] == [a["n_tokens"] - 1]
    same = IV.patch(M, PROMPT, 1, "neuron", index=3, positions="0")       # a neuron at the first token only
    assert same["operation"] == "patch" and same["index"] == 3
    mean = IV.patch(M, PROMPT, 0, "head", index=1, mode="mean", positions="all")
    assert mean["mode"] == "mean"
    with pytest.raises(ValueError, match="layers"):
        IV.patch(M, PROMPT, 9, "mlp_out")
    with pytest.raises(ValueError, match="index"):
        IV.patch(M, PROMPT, 0, "head")


def test_activation_patching_restores_fully_at_the_last_layer(M):
    clean, corrupt, t, alt = pair(M)
    r = IV.act_patch(M, clean, corrupt, t, alt, by="resid", layers="0,2", positions="all")
    first = r["grid"][0]["restored_by_position"]
    # patching every position's residual into layer 0 makes the corrupted run the clean run
    whole = IV.act_patch(M, clean, corrupt, t, alt, by="resid", layers="0", positions="all")
    assert len(first) == len(r["tokens"])
    pos_differs = [i for i, (a, b) in enumerate(zip(M.encode(clean)[0], M.encode(corrupt)[0])) if a != b]
    assert first[pos_differs[0]] == pytest.approx(1.0, abs=1e-3)        # the differing token carries it all
    assert whole["corrupt_logit_diff"] != whole["clean_logit_diff"]
    h = IV.act_patch(M, clean, corrupt, t, alt, by="head")
    assert len(h["grid"]) == 3 * M.n_heads and all("restored" in g for g in h["grid"])
    with pytest.raises(ValueError, match="different lengths"):
        IV.act_patch(M, clean, "the", t)


def test_path_patching_and_circuit(M):
    clean, corrupt, t, _ = pair(M)
    r = IV.path_patch(M, clean, corrupt, t, None, 0, 1, [(1, "attn"), (2, "mlp")])
    assert r["operation"] == "path-patch" and r["receivers"] == [{"layer": 1, "input": "attn"}, {"layer": 2, "input": "mlp"}]
    assert r["effect"] is None or isinstance(r["effect"], float)
    with pytest.raises(ValueError, match="after"):
        IV.path_patch(M, clean, corrupt, t, None, 1, 0, [(1, "attn")])
    c = IV.circuit(M, clean, top_k=3, corrupt=corrupt)
    kinds = {x["kind"] for x in c["components"]}
    assert kinds == {"head", "neuron"} and len(c["components"]) == 6
    assert "interactions" in c["method"]
    abl = [abs(x["ablation_logit_delta"]) for x in c["components"]]
    assert abl == sorted(abl, reverse=True)


def test_cli(M, tmp_path, capsys):
    path = M.m.config._name_or_path
    out = tmp_path / "t.json"
    assert IV.main(["trace", "--model", path, "--prompt", PROMPT, "--top-k", "3", "--device", "cpu", "--out", str(out)]) == 0
    assert json.loads(out.read_text())["operation"] == "trace"
    assert IV.main(["patch", "--model", path, "--prompt", PROMPT, "--layer", "7", "--device", "cpu"]) == 1


def test_saelens_release_download(tmp_path):
    pytest.importorskip("safetensors")
    import sae as S
    from safetensors.torch import save_file

    def fake_download(repo_id, allow_patterns):
        assert repo_id == "org/scope" and allow_patterns[0] == "layer_3/width_16/*"
        d = tmp_path / "hub" / "layer_3" / "width_16"
        d.mkdir(parents=True)
        (d / "cfg.json").write_text(json.dumps({"d_in": 8, "d_sae": 16, "hook_name": "blocks.3.hook_resid_post",
                                                "architecture": "standard", "model_name": "org/base"}))
        save_file({"W_enc": torch.randn(8, 16), "W_dec": torch.randn(16, 8), "b_enc": torch.zeros(16),
                   "b_dec": torch.zeros(8)}, str(d / "sae_weights.safetensors"))
        return str(tmp_path / "hub")
    reg = {"org-scope-res": {"repo_id": "org/scope", "model": "org/base",
                             "saes": [{"id": "layer_3/width_16", "path": "layer_3/width_16", "l0": 12}]}}
    folder, info = S.from_release("org-scope-res", "layer_3/width_16", reg, fake_download)
    d, meta = S.load(folder)
    assert d.n == 16 and info["l0"] == 12 and info["model"] == "org/base"
    with pytest.raises(ValueError, match="has no"):
        S.from_release("org-scope-res", "layer_9", reg, fake_download)
    with pytest.raises(ValueError, match="no SAELens release"):
        S.from_release("other", "x", reg, fake_download)


def test_explain_plan_orders_what_can_run():
    import explain
    none = {"model": "m", "gguf": "/m.gguf", "hf": None, "cett": False, "classifier": False, "torch": False,
            "transformers": False, "saes": 0, "transcoders": 0, "emotions": 0}
    p = {r["id"]: r for r in explain.plan(none)}
    assert p["weights"]["available"] and p["weights"]["recommended"] and not p["risk"]["available"]
    assert "Build llama.cpp" in p["risk"]["why"] and "PyTorch" in p["trace"]["why"]
    full = {**none, "hf": "org/m", "cett": True, "classifier": True, "torch": True, "transformers": True, "transcoders": 2}
    p = {r["id"]: r for r in explain.plan(full, "the capital of", " Paris")}
    assert p["risk"]["recommended"] and p["graph"]["available"] and "kind=iv_circuit" in p["trace"]["action"]["href"]
    inv = {"dicts": [{"model": "org/m", "target_site": None}, {"model": "org/m", "target_site": "mlp_out"},
                     {"model": "other/x"}], "emotions": [{"model": "other/n"}]}
    c = explain.capabilities({"id": "m", "path": "/x/m.gguf"}, {"hf_model": "org/m"}, None, inv)
    assert (c["saes"], c["transcoders"], c["emotions"], c["gguf"]) == (1, 1, 0, "/x/m.gguf")


def test_explain_api(tmp_path):
    import test_studio
    gen = test_studio.studio_srv.__wrapped__(tmp_path)
    base = next(gen)
    try:
        code, r = test_studio.call(base, "/api/explain/plan?model=coder-gguf/coder-7b-q4_k_m&prompt=hi")
        assert code == 200 and r["capabilities"]["gguf"].endswith(".gguf")
        assert {p["id"] for p in r["plan"]} >= {"risk", "trace", "weights"}
        assert test_studio.call(base, "/api/explain/plan?model=nope")[0] == 404
        assert "explainReply" in test_studio.call(base, "/")[1]
    finally:
        try:
            next(gen)
        except StopIteration:
            pass
