"""Feature dictionaries, circuits and emotion vectors, end to end on a tiny random model (no network)."""
import json
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "viz"))
sys.path.insert(0, str(ROOT / "tests"))
from test_studio import call, studio_srv  # noqa: E402,F401

WORDS = ("the a cat dog sat ran on under mat rug happy sad fear calm angry quiet loud house tree river is was "
         "and but then very not it he she they saw heard felt went came home away today night day bright dark "
         "cold warm slow fast").split()


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    from tokenizers import Tokenizer, models, pre_tokenizers
    # Llama architecture: transformers 5 forces Qwen's own tokenizer class onto qwen2 models, which a toy
    # word-level vocabulary does not survive
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast
    d = tmp_path_factory.mktemp("tiny")
    vocab = {w: i for i, w in enumerate(["<unk>", "<pad>", "</s>"] + WORDS)}
    tk = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    tk.pre_tokenizer = pre_tokenizers.Whitespace()
    PreTrainedTokenizerFast(tokenizer_object=tk, unk_token="<unk>", pad_token="<pad>", eos_token="</s>").save_pretrained(d)
    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=len(vocab), hidden_size=32, intermediate_size=64, num_hidden_layers=3,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=128,
                      tie_word_embeddings=False)
    LlamaForCausalLM(cfg).save_pretrained(d)
    import random
    rng = random.Random(0)
    corpus = d / "corpus.txt"
    corpus.write_text("\n\n".join(" ".join(rng.choice(WORDS) for _ in range(40)) for _ in range(120)))
    return d


@pytest.fixture(scope="module")
def built(tiny, tmp_path_factory):
    """Activations, a residual dictionary, two transcoders, descriptions and a neuron map."""
    import features as FX
    ws = tmp_path_factory.mktemp("ws")
    runs = ws / "runs" / "features"
    acts = runs / "acts"
    assert FX.main(["collect", "--model", str(tiny), "--corpus", str(tiny / "corpus.txt"), "--site", "resid", "mlp_in",
                    "mlp_out", "neurons", "--layers", "0", "1", "--max-tokens", "3000", "--ctx", "32", "--batch", "4",
                    "--shard-rows", "1000", "--device", "cpu", "--out", str(acts)]) == 0
    common = ["--acts", str(acts), "--arch", "topk", "--k", "8", "--expansion", "4", "--batch", "256", "--steps", "150",
              "--lr", "3e-3", "--check-seqs", "8", "--device", "cpu"]
    assert FX.main(["train", "--site", "resid", "--layer", "1", "--out", str(runs / "r1")] + common) == 0
    for l in (0, 1):
        assert FX.main(["train", "--site", "mlp_in", "--target-site", "mlp_out", "--layer", str(l),
                        "--out", str(runs / f"tc{l}")] + common) == 0
    assert FX.main(["interpret", "--sae", str(runs / "r1"), "--device", "cpu", "--top", "5"]) == 0
    hn = ws / "h.json"
    hn.write_text(json.dumps({"by_layer": {"1": [0, 1, 2, 3, 4, 5, 6, 7]}}))
    assert FX.main(["neurons", "--sae", str(runs / "r1"), "--h-neurons", str(hn), "--device", "cpu"]) == 0
    return {"ws": ws, "runs": runs, "acts": acts, "model": tiny}


# ---------------------------------------------------------------- dictionaries

def test_sae_recovers_known_directions():
    import sae as S
    torch.manual_seed(0)
    d, n_true = 32, 64
    D = torch.nn.functional.normalize(torch.randn(n_true, d), dim=1)

    def gen(b):
        c = torch.zeros(b, n_true)
        c.scatter_(1, torch.randint(0, n_true, (b, 3)), torch.rand(b, 3) * 3 + 0.5)
        return c @ D * 5

    def it():
        while True:
            yield gen(512), None
    for arch, kw in (("topk", {}), ("relu", {"target_l0": 4}), ("jumprelu", {"target_l0": 4})):
        m = S.SAE(d, 128, arch, k=3)
        m.in_scale.fill_(S.norm_scale([gen(2048)], d))
        m.out_scale.fill_(float(m.in_scale))
        S.train(m, it(), 1000, lr=3e-3, log_every=1000, **kw)
        found = (D @ m.directions().t()).max(1).values
        assert float((found > 0.9).float().mean()) > 0.9, arch


def test_relu_folds_decoder_norms():
    import sae as S
    m = S.SAE(8, 16, "relu")
    x = torch.randn(4, 8)
    before = m.reconstruct(x)[0]
    m.fold_decoder_norms()
    assert torch.allclose(m.W_dec.norm(dim=1), torch.ones(16), atol=1e-5)
    assert torch.allclose(m.reconstruct(x)[0], before, atol=1e-5)


def test_collect_train_interpret_neurons(built):
    runs, acts = built["runs"], built["acts"]
    meta = json.loads((acts / "meta.json").read_text())
    assert meta["tokens"] == 3000 and {"resid.1", "mlp_in.0", "neurons.1"} <= set(meta["points"])
    info = json.loads((runs / "r1" / "sae.json").read_text())
    assert info["site"] == "resid" and info["layer"] == 1 and info["eval"]["l0"] == pytest.approx(8, abs=0.01)
    assert {"clean", "spliced", "delta"} <= set(info["splice"])
    tc = json.loads((runs / "tc0" / "sae.json").read_text())
    assert tc["target_site"] == "mlp_out" and "recovered" in tc["splice"]
    fj = json.loads((runs / "r1" / "features.json").read_text())
    assert fj["n_features"] == 128 and len(fj["features"]) == 128 and fj["alive"] > 0
    f = next(e for e in fj["features"] if e["freq"] > 0)
    assert {"up", "down", "nb", "xy", "hist"} <= set(f)
    ex = json.loads((runs / "r1" / "examples" / "00000.json").read_text())[str(f["id"])]
    assert ex["top"] and ex["buckets"] and len(ex["top"][0]["tok"]) == len(ex["top"][0]["act"])
    top = ex["top"][0]
    assert top["act"][top["at"]] > 0                                     # the marked token is where it fires
    nj = json.loads((runs / "r1" / "neurons.json").read_text())
    assert set(nj["method"]) == {"weights", "correlation"}
    row = nj["features"][str(f["id"])]
    assert row["writes"] and row["reads"] and row["corr"] and 0 <= row["h_share"] <= 1


def test_import_saelens_folder(tmp_path):
    import sae as S
    import features as FX
    from safetensors.torch import save_file
    src = tmp_path / "saelens"
    src.mkdir()
    save_file({"W_enc": torch.randn(32, 64), "W_dec": torch.randn(64, 32), "b_enc": torch.zeros(64),
               "b_dec": torch.zeros(32)}, str(src / "sae_weights.safetensors"))
    (src / "cfg.json").write_text(json.dumps({"d_in": 32, "d_sae": 64, "hook_name": "blocks.2.hook_resid_post",
                                              "activation_fn_str": "relu", "apply_b_dec_to_input": True}))
    assert FX.main(["import", "--path", str(src), "--model", "m", "--out", str(tmp_path / "imp")]) == 0
    sae, info = S.load(tmp_path / "imp")
    assert (info["site"], info["layer"], sae.arch, sae.n, sae.pre_bias) == ("resid", 2, "relu", 64, True)
    assert S.site_from_hook("blocks.5.hook_resid_pre") == ("resid", 4)


# ---------------------------------------------------------------- circuits

def test_frozen_attention_and_norms(tiny):
    import feature_circuits as FC
    model, tok, _ = FC.load_for_attribution(str(tiny), "cpu")
    for p in model.parameters():
        p.requires_grad_(True)
    ids = torch.tensor([[3, 5, 7, 9, 11]])
    att = model.model.layers[1].self_attn
    out = model(input_ids=ids).logits
    with FC.Frozen(model):
        out2 = model(input_ids=ids).logits
    assert torch.allclose(out, out2, atol=1e-5)
    out2[0, -1, 4].backward()
    assert att.q_proj.weight.grad is None or float(att.q_proj.weight.grad.abs().sum()) == 0
    assert att.k_proj.weight.grad is None or float(att.k_proj.weight.grad.abs().sum()) == 0
    assert float(att.v_proj.weight.grad.abs().sum()) > 0


def test_graph_intervene_attribute(built):
    import feature_circuits as FC
    import features as FX
    runs, out = built["runs"], built["ws"] / "runs" / "graphs" / "g.json"
    assert FC.main(["graph", "--transcoders", str(runs / "tc*"), "--prompt", "the cat sat on the mat",
                    "--max-features", "40", "--node-threshold", "0.99", "--edge-threshold", "0.99", "--device", "cpu",
                    "--out", str(out)]) == 0
    g = json.loads(out.read_text())
    N = {n["i"]: n for n in g["nodes"]}
    kinds = {n["kind"] for n in g["nodes"]}
    assert {"logit", "token"} <= kinds and g["targets"] and 0 <= g["completeness_score"] <= 1
    for e in g["edges"]:                                   # causal: earlier layer, same or earlier position
        s, t = N[e["s"]], N[e["t"]]
        if t["kind"] == "feature":
            assert s["layer"] < t["layer"] and s["pos"] <= t["pos"]
    feats = [n for n in g["nodes"] if n["kind"] == "feature"]
    assert feats, "some features should survive a 0.99 threshold"
    model, tok, dev = FC.load_for_attribution(str(built["model"]), "cpu")
    tcs = FC.load_transcoders([str(runs / "tc*")], dev)
    f = max(feats, key=lambda n: n["act"])
    r = FC.intervene(model, tok, tcs, g["prompt"], [{"layer": f["layer"], "feature": f["feature"], "pos": None,
                                                     "value": 50.0, "multiply": True}], watch=[(f["layer"], f["pos"], f["feature"])])
    assert r["kl"] > 0 and r["watched"][0]["after"] == pytest.approx(r["watched"][0]["before"] * 50, rel=1e-3)
    assert FC.parse_set(["3:17=x2", "1:4@2=0"]) == [{"layer": 3, "feature": 17, "pos": None, "value": 2.0, "multiply": True},
                                                    {"layer": 1, "feature": 4, "pos": 2, "value": 0.0, "multiply": False}]
    import sae as S
    sae, info = S.load(runs / "r1")
    a = FC.output_attribution(model, tok, sae, info, "the cat sat on the", " mat", " rug", ablate=5)
    assert a["features"] and all(row["pos"] > 0 for row in a["features"][:10])   # the first position is left out
    assert all("ablation" in row for row in a["features"][:5])
    with pytest.raises(SystemExit):
        FC.load_transcoders([str(runs / "r1")], dev)        # a plain dictionary is not a transcoder
    del FX


# ---------------------------------------------------------------- emotions

@pytest.fixture(scope="module")
def emo(tiny, built):
    import emotions as EM
    sdir = built["ws"] / "runs" / "emotions" / "stories"
    sdir.mkdir(parents=True)
    import random
    rng = random.Random(1)
    words = {"happy": "happy bright warm", "sad": "sad dark cold", "angry": "angry loud fast", "calm": "calm quiet slow"}
    with (sdir / "stories.jsonl").open("w") as f:
        for e, w in words.items():
            for k in range(6):
                text = " ".join(rng.choice(WORDS[:20] + w.split() * 6) for _ in range(30))
                f.write(json.dumps({"emotion": e, "topic": "t", "text": text}) + "\n")
    with (sdir / "neutral.jsonl").open("w") as f:
        for k in range(6):
            f.write(json.dumps({"topic": "n", "text": " ".join(rng.choice(WORDS[:20]) for _ in range(30))}) + "\n")
    out = built["ws"] / "runs" / "emotions" / "tiny"
    assert EM.main(["build", "--model", str(tiny), "--stories", str(sdir), "--start", "2", "--device", "cpu",
                    "--out", str(out)]) == 0
    return out


def test_emotion_build_trace_sweep_steer(tiny, emo, tmp_path):
    import emotions as EM
    info = json.loads((emo / "emotions.json").read_text())
    assert info["names"] == ["angry", "calm", "happy", "sad"] and info["default_layer"] == 2
    assert set(info["accuracy"]) == set(info["accuracy_raw"]) and info["removed_pcs"]
    assert len(info["cosine"]) == 4 and len(info["pca"]["xy"]) == 4 and info["logits"]["happy"]["up"]
    t = tmp_path / "t.json"
    assert EM.main(["trace", "--emotions", str(emo), "--text", "the dog was happy and warm", "--device", "cpu",
                    "--out", str(t)]) == 0
    tr = json.loads(t.read_text())
    assert len(tr["z"]) == len(tr["tokens"]) == 6 and len(tr["z"][0]) == 4
    s = tmp_path / "s.json"
    assert EM.main(["sweep", "--emotions", str(emo), "--template", "the {x} cat", "--values", "happy", "sad",
                    "--raw", "--device", "cpu", "--out", str(s)]) == 0
    assert len(json.loads(s.read_text())["rows"]) == 2
    st = tmp_path / "st.json"
    assert EM.main(["steer", "--emotions", str(emo), "--prompt", "the cat", "--mix", "happy=2", "--max-tokens", "5",
                    "--device", "cpu", "--out", str(st)]) == 0
    r = json.loads(st.read_text())
    assert r["mix"] == {"happy": 2.0} and isinstance(r["steered"], str)
    assert EM.split_numbered("[story 1] " + "word " * 30 + "[story 2] " + "other " * 30, "story")[1].startswith("other")


def test_lab_steering_changes_logits(tiny, built, emo):
    import feature_lab as FL
    import features as FX
    model, tok, dev = FX.load_model(str(tiny), "float32", "cpu")
    ids = torch.tensor([[3, 5, 7, 9]])
    with torch.no_grad():
        base = model(input_ids=ids).logits
        with FL.Steering(model, {"feature": {"sae": str(built["runs"] / "r1"), "set": {"0": 20}}}):
            clamped = model(input_ids=ids).logits
        with FL.Steering(model, {"emotion": {"file": str(emo), "mix": {"happy": 0.5}}}):
            moved = model(input_ids=ids).logits
        after = model(input_ids=ids).logits
    assert not torch.allclose(base, clamped) and not torch.allclose(base, moved)
    assert torch.allclose(base, after)                                     # hooks removed afterwards
    with pytest.raises(ValueError):
        FL.validate({"feature": {"sae": "x"}})
    import decode_lab
    v = decode_lab.parse_variants(['f={"feature": {"sae": "x", "set": {"1": 2}}}', 'e={"emotion": {"file": "y", "mix": {"calm": 0.1}}}'])
    assert set(v) == {"f", "e"}
    with pytest.raises(SystemExit):
        decode_lab.parse_variants(['bad={"emotion": {"file": "y"}}'])


# ---------------------------------------------------------------- Studio

def test_features_api(built, emo):
    import features_api as FA
    ws = str(built["ws"])
    lst = FA.list_all(ws)
    assert {d["name"] for d in lst["dicts"]} == {"r1", "tc0", "tc1"} and lst["acts"] and lst["emotions"] and lst["stories"]
    assert lst["graphs"] and lst["graphs"][0]["kind"] == "attribution"
    d = FA.allowed(str(built["runs"] / "r1"), ws)
    s = FA.dictionary(d)
    assert s["interpreted"] and len(s["freq"]) == 128
    fid = next(i for i, f in enumerate(s["freq"]) if f > 0)
    f = FA.feature(d, fid)
    assert f["examples"]["top"] and isinstance(f["examples"]["top"][0]["tok"][0], str) and f["neurons"]
    FA.set_label(d, fid, "a test label")
    assert FA.feature(d, fid)["label"] == "a test label" and FA.dictionary(d)["label"][fid] == "a test label"
    assert any(h["id"] == fid for h in FA.search(d, "test label"))
    with pytest.raises(ValueError):
        FA.set_label(d, fid, "two\nlines")
    for bad in ("/etc/passwd", ws + "/runs/../../x"):
        with pytest.raises(ValueError):
            FA.allowed(bad, ws)
    assert FA.emotions(FA.allowed(str(emo), ws))["names"]


def test_studio_routes_and_lab_guard():
    import studio
    with pytest.raises(ValueError):
        studio.features_lab_call("http://example.com:8090", "info", None)
    with pytest.raises(ValueError):
        studio.features_lab_call("http://127.0.0.1:8090", "../x", {})


def test_features_page_and_api(studio_srv):
    import urllib.request
    with urllib.request.urlopen(studio_srv + "/features", timeout=20) as r:
        page = r.read().decode()
    assert "Studio Features" in page and "nsProfiles" in page
    code, j = call(studio_srv, "/api/features/list")
    assert code == 200 and set(j) >= {"dicts", "acts", "emotions", "stories", "graphs"}
    code, j = call(studio_srv, "/api/features/dict?path=/etc")
    assert code == 400
    code, j = call(studio_srv, "/api/features/lab/inspect", {"lab": "http://example.com", "text": "x"})
    assert code == 400


def test_job_kinds_match_script_flags():
    import subprocess
    import jobs
    for kind, spec in jobs.SPECS.items():
        if not kind.startswith(("feat_", "emo_")):
            continue
        out = subprocess.run([sys.executable] + [str(x) for x in spec["argv"]] + ["--help"], capture_output=True,
                             text=True, timeout=120).stdout
        missing = [f["flag"] for f in spec["fields"] if f["flag"] and f["flag"] not in out]
        assert not missing, (kind, missing)


# ---------------------------------------------------------------- watching while generating

def test_live_watch_lab_frames_and_session(tiny, built, emo, tmp_path):
    import bloom
    import decode_lab
    import live_watch

    class Sink:
        tier, frames = "sparse", []

        def send(self, fr):
            self.frames.append(fr)

    sink = Sink()
    rec = tmp_path / "traces"
    lab = decode_lab.Lab(str(tiny), {"plain": {}}, "float32", publisher=sink,
                         watch={"cett": True, "bins": 16, "emotions": str(emo), "sae": [str(built["runs"] / "r1")],
                                "record": str(rec)})
    r = lab.generate([{"role": "user", "content": "the cat sat"}], "plain", 6)
    n = r["n_out"]
    assert n >= 1 and len(sink.frames) == n and len(r["watch"]["tokens"]) == n
    fr = sink.frames[0]
    assert fr["src"] == "lab" and fr["t"] == "sparse" and fr["l"] == 3
    assert all(0 <= l < 3 and 0 <= b < 16 for l, b, _ in fr["c"])
    assert len(fr["emo"]) == 3 and {e for e, _ in fr["emo"]} <= {"angry", "calm", "happy", "sad"}
    assert all(abs(z) >= 0.5 for z in fr.get("ez", {}).values())
    assert fr["feat"] and all(d == "r1" and v > 0 for d, _, v, _ in fr["feat"])
    assert r["watch"]["emotions"]["names"] == ["angry", "calm", "happy", "sad"]

    # the recorded session: what the 3D replay, the timeline and Godot read
    man = json.loads((Path(r["session"]) / "manifest.json").read_text())
    assert Path(r["session"]).parent == rec and man["kind"] == "trace" and man["source"] == "decode_lab"
    _blob, meta = bloom.build_payload(r["session"], None, 90.0, 5000)
    assert meta["frames"] == n and meta["layers"] == 3 and meta["neurons"] == 16
    assert len(meta["emotions"]["z"]) == n and len(meta["emotions"]["names"]) == 4
    assert len(meta["features"]) == n and meta["source"] == "decode_lab"

    # a request can turn either off, but cannot pick the folder
    sink.frames.clear()
    r2 = lab.generate([{"role": "user", "content": "the dog"}], "plain", 3, extra={"record": False})
    assert "session" not in r2 and r2["watch"]["tokens"]
    r3 = lab.generate([{"role": "user", "content": "the dog"}], "plain", 3, extra={"watch": False})
    assert "watch" not in r3 and "session" not in r3
    lab.watch.pop("record")
    r4 = lab.generate([{"role": "user", "content": "the dog"}], "plain", 3, extra={"record": True})
    assert "session" not in r4

    # binned tier and emotions-only frames
    w = live_watch.Watcher(lab.model, lab.tok, cett=True, bins=8)
    with w, torch.no_grad():
        lab.model(input_ids=torch.tensor([[3, 4, 5]]))
        f = w.step(" x")
    b = w.live_frame(f, "binned")
    assert b["t"] == "binned" and len(b["v"]) == 3 and len(b["v"][0]) == 8
    w = live_watch.Watcher(lab.model, lab.tok, cett=False, emotions=lab.watch["_loaded"][0])
    with w, torch.no_grad():
        lab.model(input_ids=torch.tensor([[3, 4, 5]]))
        f = w.live_frame(w.step(" y"))
    assert f["t"] == "none" and f["emo"]
    assert not w.hooks                                                       # removed on exit


def test_publisher_and_stream_push(capsys):
    import threading
    import urllib.request
    from http.server import ThreadingHTTPServer
    import live_watch
    import stream
    p = live_watch.Publisher("http://127.0.0.1:9")                          # nothing listens there
    p.send({"i": 0})
    p.send({"i": 1})
    assert capsys.readouterr().out.count("not reachable") == 1
    stream.STATE.update({"token": "tk", "pushed": 0})
    srv = ThreadingHTTPServer(("127.0.0.1", 0), stream.Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        url = f"http://127.0.0.1:{srv.server_address[1]}"
        live_watch.Publisher(url, "tk").send({"i": 0, "tok": "x", "t": "none", "s": 0, "emo": [["calm", 1.2]]})
        assert stream.STATE["pushed"] > 0                                    # the simulation steps aside
        with pytest.raises(urllib.error.HTTPError):
            urllib.request.urlopen(urllib.request.Request(url + "/api/push", data=b"{}"), timeout=5)
    finally:
        srv.shutdown()


def test_decode_lab_watch_flags():
    import subprocess
    out = subprocess.run([sys.executable, str(ROOT / "scripts" / "decode_lab.py"), "--help"], capture_output=True,
                         text=True, timeout=120).stdout
    for flag in ("--watch-cett", "--watch-bins", "--watch-emotions", "--watch-sae", "--publish", "--record"):
        assert flag in out
