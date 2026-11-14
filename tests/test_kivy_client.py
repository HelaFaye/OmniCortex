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
    for link in (f"https://h:7870/pair#c=ABCD-EFGH-JKMN&fp={fp}&a=p", f"http://10.0.0.2:7870/pair#c=abcdefghjkmn"):
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
