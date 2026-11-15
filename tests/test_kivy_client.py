"""The Kivy client's API layer (clients/kivy/omnicortex_client/api.py), against a fake Studio over real TLS:
pairing with certificate pinning, models, streaming chat, the reply checks, the review lookup and the saved
pairing. No Kivy needed."""
import hashlib
import json
import os
import ssl
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "clients" / "kivy"))
from omnicortex_client import api  # noqa: E402

TOKEN = "dev-token-123"


class FakeStudio(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, obj):
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def _authed(self):
        return self.headers.get("Authorization") == f"Bearer {TOKEN}"

    def do_GET(self):
        if not self._authed():
            return self._send(401, {"error": "token needed"})
        if self.path == "/v1/models":
            return self._send(200, {"data": [{"id": "auto"}, {"id": "acme/coder-7b"}]})
        if self.path.startswith("/api/lookup?"):
            return self._send(200, {"model": self.path.split("=", 1)[1], "stats": [{"key": "accuracy", "value": 0.8}]})
        self._send(404, {"error": "no"})

    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers["Content-Length"])) or b"{}")
        if self.path == "/api/pair/claim":
            if req.get("code") != "ABCDEFGHJKMN":
                return self._send(403, {"error": "pairing code is wrong, used or expired"})
            return self._send(200, {"token": TOKEN, "device_id": "d1", "expires": None,
                                    "fingerprint": self.server.fp})
        if not self._authed():
            return self._send(401, {"error": "token needed"})
        if self.path == "/v1/chat/completions":
            if req.get("stream"):
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                for piece in ("Can", "berra", "."):
                    self.wfile.write(b"data: " + json.dumps({"choices": [{"delta": {"content": piece}}]}).encode()
                                     + b"\n\n")
                    self.wfile.flush()
                self.wfile.write(b"data: [DONE]\n\n")
                return
            return self._send(200, {"choices": [{"message": {"content": "Canberra."}}]})
        if self.path == "/api/trace":
            return self._send(200, {"pieces": list(req["text"]), "prob": [0.1] * len(req["text"]), "flagged": [],
                                    "max": 0.1, "mean": 0.1})
        if self.path == "/api/truth":
            return self._send(200, {"verdict": "consistent", "alternatives": req["alternatives"]})
        self._send(404, {"error": "no"})


@pytest.fixture(scope="module")
def studio(tmp_path_factory):
    pytest.importorskip("cryptography")
    import oc_security as sec
    d = tmp_path_factory.mktemp("tls")
    cert, key = d / "c.pem", d / "k.pem"
    assert sec.main(["selfsigned", "--host", "127.0.0.1", "--cert", str(cert), "--key", str(key)]) == 0
    pem = cert.read_text()
    fp = hashlib.sha256(ssl.PEM_cert_to_DER_cert(pem[pem.index("-----BEGIN"):])).hexdigest()
    s = ThreadingHTTPServer(("127.0.0.1", 0), FakeStudio)
    s.fp = fp
    s.socket = sec.server_ssl_context(str(cert), str(key)).wrap_socket(s.socket, server_side=True)
    threading.Thread(target=s.serve_forever, daemon=True).start()
    yield f"https://127.0.0.1:{s.server_port}", fp
    s.shutdown()


def oc_alphabet_ok():
    import oc_pairing
    return api.CODE_ALPHABET == oc_pairing.CODE_ALPHABET


def test_links_parse_like_oc_pairing():
    import oc_pairing
    fp = "ab" * 32
    for link in (f"https://h:7870/pair#c=ABCD-EFGH-JKMN&fp={fp}&a=p", "http://10.0.0.2:7870/pair#c=abcdefghjkmn"):
        mine = api.parse_pair_link(link)
        base, code, f = oc_pairing.parse_link(link)
        assert (mine["base"], mine["code"], mine["fingerprint"]) == (base, code, f)
    with pytest.raises(api.StudioError):
        api.parse_pair_link("https://h:7870/pair#c=ABCD")           # https without a fingerprint
    with pytest.raises(api.StudioError):
        api.parse_pair_link("https://h:7870/pair#c=ABCD&fp=xyz")    # malformed fingerprint
    with pytest.raises(api.StudioError):
        api.parse_pair_link("https://h:7870/pair#fp=" + fp)         # no code


def test_pair_and_use_over_pinned_tls(studio, tmp_path):
    base, fp = studio
    p = api.Studio.pair(f"{base}/pair#c=ABCD-EFGH-JKMN&fp={fp}&a=p", name="phone")
    assert p.token == TOKEN and p.fingerprint == fp and p.expires is None
    st = api.Studio(p)
    assert st.warning == "" and st.models() == ["auto", "acme/coder-7b"]
    msgs = [{"role": "user", "content": "Capital of Australia?"}]
    assert st.chat(msgs) == "Canberra."
    assert "".join(st.chat_stream(msgs)) == "Canberra."
    assert st.check(msgs, "Canberra.")["max"] == 0.1
    assert st.truth(msgs, "Canberra.", ["Sydney."])["alternatives"] == ["Sydney."]
    assert st.lookup("acme/coder-7b")["stats"][0]["key"] == "accuracy"
    path = api.save(p, str(tmp_path))
    assert oc_alphabet_ok()
    assert oct(os.stat(path).st_mode & 0o777) == "0o600"
    assert api.load(str(tmp_path)) == p
    api.forget(str(tmp_path))
    assert api.load(str(tmp_path)) is None


def test_wrong_certificate_and_bad_code_refused(studio):
    base, fp = studio
    with pytest.raises(api.StudioError, match="does not match"):
        api.Studio.pair(f"{base}/pair#c=ABCD-EFGH-JKMN&fp={'0' * 64}")
    with pytest.raises(api.StudioError, match="403"):
        api.Studio.pair(f"{base}/pair#c=WRONG-CODE-XX&fp={fp}")
    st = api.Studio(api.Pairing(base=base, token="nope", fingerprint=fp))
    with pytest.raises(api.StudioError, match="401"):
        st.models()


def test_plain_http_warns_off_loopback():
    assert api.Studio(api.Pairing(base="http://127.0.0.1:7870", token="t")).warning == ""
    assert "VPN" in api.Studio(api.Pairing(base="http://100.64.0.2:7870", token="t")).warning


# ------------------------------------------------------------------ the app's controller (no Kivy)

def test_controller_pairs_chats_and_checks(studio, tmp_path):
    from omnicortex_client import controller as C
    base, fp = studio
    c = C.Client(data_dir=str(tmp_path))
    assert not c.load()
    c.pair(f"{base}/pair#c=ABCD-EFGH-JKMN&fp={fp}", "  ")
    assert c.pairing.name == "device" and C.Client(data_dir=str(tmp_path)).load()   # saved
    assert c.refresh_models() == ["auto", "acme/coder-7b"]
    c.choose("acme/coder-7b")
    assert api.load(str(tmp_path)).model == "acme/coder-7b"
    pieces, done, errors = [], [], []
    reply = c.send("Capital of Australia?", lambda m: pieces.append(m.content), done.append, errors.append,
                   background=False)
    assert pieces == ["Can", "Canberra", "Canberra."] and done == [reply] and not errors and not c.busy
    assert [m.role for m in c.messages] == ["user", "assistant"] and reply.model == "acme/coder-7b"
    assert c.send("   ", None, None, None) is None                      # nothing to send
    c.check(reply, done.append, errors.append, background=False)
    assert "No token flagged" in reply.note and reply.check == "Canberra."
    c.truth(reply, ["Sydney.", " ", ""], done.append, errors.append, background=False)
    assert reply.note.startswith("The model's activations agree") and not errors
    out = []
    c.lookup("acme/coder-7b", out.append, errors.append, background=False)
    assert "accuracy" in out[0] or "Nothing known" in out[0]
    c.forget()
    assert c.pairing is None and not C.Client(data_dir=str(tmp_path)).load()
    with pytest.raises(api.StudioError, match="not paired"):
        c.studio


def test_controller_reports_errors_and_runs_callbacks_through_post(studio, tmp_path):
    from omnicortex_client import controller as C
    base, fp = studio
    posted = []
    c = C.Client(data_dir=str(tmp_path), post=lambda fn: (posted.append(fn), fn()))
    c.pairing = api.Pairing(base=base, token="revoked", fingerprint=fp)
    errors = []
    c.send("hello", None, None, errors.append, background=False)
    assert "401" in errors[0] and posted and not c.busy
    c.lookup("", None, errors.append)
    assert "type a model" in errors[-1]


def test_display_text():
    from omnicortex_client import controller as C
    trace = {"pieces": ["Syd", "ney", " [x]"], "prob": [0.9, 0.55, 0.1], "flagged": [0, 1], "threshold": 0.5}
    assert C.risk_markup(trace) == "[color=ff5c5c]Syd[/color][color=ffae42]ney[/color] &bl;x&br;"
    assert C.risk_summary(trace) == "2 of 3 tokens flagged; highest risk 0.90, mean 0.52."
    assert C.risk_summary({"pieces": [], "prob": []}) == "No tokens were scored."
    t = C.truth_summary({"verdict": "knows_but_says_otherwise", "answer": {"text": "Sydney", "p_true": 0.01},
                         "alternatives": [{"text": "Canberra", "p_true": 0.7}], "best_alternative": "Canberra",
                         "alt_leads_at": ["model.layers.12"], "stage": "model.layers.25", "stage_auc": 0.93})
    assert "favour another answer" in t and "“Canberra”: p = 0.70" in t and "model.layers.12" in t
    assert "does not show intent" in t
    rec = {"model": "org/m", "available": True,
           "stats": [{"key": "tok_s", "group": "performance", "label": "Decode", "value": 31.2, "unit": "tok/s",
                      "variant": "Q4_K_M · llama.cpp"},
                     {"key": "accuracy", "group": "quality", "label": "Accuracy", "value": 0.71, "unit": "ratio",
                      "n": 400}]}
    s = C.lookup_summary(rec)
    assert s.index("Performance") < s.index("Quality") and "Accuracy: 71% (n = 400)" in s and "31.2 tok/s" in s
    assert "Nothing known" in C.lookup_summary({"model": "x", "available": False,
                                                 "sources": [{"id": "b", "ok": False, "error": "offline"}]})


def test_kivy_app_screens_build():
    """The real app's layout and screens, on a virtual display (Kivy opens a window for text inputs and scroll
    views). Skipped without Kivy or Xvfb."""
    import importlib.util
    import shutil
    import subprocess
    if importlib.util.find_spec("kivy") is None:     # not imported here: Kivy would read pytest's arguments
        pytest.skip("needs Kivy")
    if not shutil.which("xvfb-run"):
        pytest.skip("needs xvfb-run")
    code = """
import sys
sys.path.insert(0, %r)
from kivy.lang import Builder
from omnicortex_client import app as A
from omnicortex_client.controller import Message
Builder.load_string(A.KV)
for s in (A.PairScreen(name="pair"), A.ChatScreen(name="chat"), A.LookupScreen(name="lookup")):
    assert s.children, s.name
v = A.MessageView()
v.show(Message("assistant", "a [b] c", model="acme/coder-7b"))
assert v.header == "acme/coder-7b" and v.body == "a &bl;b&br; c", (v.header, v.body)
v.show(Message("user", "hi"))
assert v.header == "You"
print("SCREENS-OK")
""" % str(ROOT / "clients" / "kivy")
    r = subprocess.run(["xvfb-run", "-a", sys.executable, "-c", code], capture_output=True, text=True, timeout=120,
                       env={**os.environ, "KIVY_NO_ARGS": "1", "KIVY_NO_CONSOLELOG": "1"})
    assert "SCREENS-OK" in r.stdout, r.stdout[-2000:] + r.stderr[-2000:]
