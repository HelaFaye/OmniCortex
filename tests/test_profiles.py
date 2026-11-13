"""Setting profiles, every script option as a control, and search."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "viz"))
sys.path.insert(0, str(ROOT / "scripts"))
import jobs  # noqa: E402
from test_studio import call, studio_srv  # noqa: E402,F401

HELP = """usage: x.py [-h] [--model MODEL] [--kv] [--dtype {auto,float32}] [--layers LAYERS [LAYERS ...]]

options:
  -h, --help            show this help message and exit
  --model MODEL         HF id or folder
  --kv                  also try quantized KV caches (can change
                        answers slightly)
  --dtype {auto,float32}
  -o OUT, --out OUT     where to write
  --layers LAYERS [LAYERS ...]
                        layers to use
  --tls, --no-tls       serve HTTPS
"""


def test_parse_help():
    o = {x["flag"]: x for x in jobs._parse_help(HELP)}
    assert set(o) == {"--model", "--kv", "--dtype", "--out", "--layers", "--tls"}
    assert o["--kv"]["takes_value"] is False and o["--kv"]["help"].endswith("answers slightly)")
    assert o["--dtype"]["choices"] == ["auto", "float32"] and o["--layers"]["multi"] and o["--layers"]["metavar"] == "LAYERS"
    assert o["--tls"]["boolean_optional"] and o["--out"]["takes_value"]


def test_extra_options_are_checked(monkeypatch):
    monkeypatch.setattr(jobs, "script_options", lambda kind: jobs._parse_help(HELP))
    assert jobs._extra_argv("x", {"--kv": True, "--dtype": "auto", "--layers": "3 5", "--tls": False}) == \
        ["--kv", "--dtype", "auto", "--layers", "3", "5", "--no-tls"]
    assert jobs._extra_argv("x", {"--model": "-5"}) == ["--model", "-5"]          # negative numbers are values
    for bad in ({"--rm": "x"}, {"--model": "--evil"}, {"--dtype": "int8"}, {"--model": "a\nb"}):
        with pytest.raises(ValueError):
            jobs._extra_argv("x", bad)


def test_real_scripts_offer_their_other_options():
    opts = {o["flag"] for o in jobs.script_options("testqa")}
    assert "--max-concurrency" in opts and "--endpoint" not in opts and "--concurrency" not in opts                  # form fields are not repeated
    argv = jobs.build_argv("testqa", {"endpoint": "m=http://h/v1@m", "out": "r.json", "__extra": {"--max-concurrency": "2"}})
    assert argv[-2:] == ["--max-concurrency", "2"]


def test_profiles_api(studio_srv):
    sc = "job:testqa"
    code, p = call(studio_srv, "/api/profiles?scope=" + sc)
    assert code == 200 and p["profiles"] == {} and p["default"] is None
    call(studio_srv, "/api/profiles", {"scope": sc, "action": "save", "name": "fast", "values": {"per_subject": "5"}})
    call(studio_srv, "/api/profiles", {"scope": sc, "action": "save", "name": "full", "values": {"per_subject": "0"}})
    code, p = call(studio_srv, "/api/profiles", {"scope": sc, "action": "default", "name": "fast"})
    assert p["default"] == "fast" and set(p["profiles"]) == {"fast", "full"}
    code, p = call(studio_srv, "/api/profiles", {"scope": sc, "action": "rename", "name": "fast", "new_name": "quick"})
    assert p["default"] == "quick" and "fast" not in p["profiles"]
    code, p = call(studio_srv, "/api/profiles", {"scope": sc, "action": "delete", "name": "quick"})
    assert p["default"] is None and set(p["profiles"]) == {"full"}
    for bad in ({"scope": "a b", "action": "save", "name": "x", "values": {}},
                {"scope": sc, "action": "save", "name": "", "values": {}},
                {"scope": sc, "action": "save", "name": "a/b", "values": {}},
                {"scope": sc, "action": "default", "name": "nope"},
                {"scope": sc, "action": "explode", "name": "x"}):
        assert call(studio_srv, "/api/profiles", bad)[0] == 400, bad
    # other tools keep their own profiles
    assert call(studio_srv, "/api/profiles?scope=model-settings")[1]["profiles"] == {}


def test_search_and_profiles_on_pages(studio_srv):
    code, j = call(studio_srv, "/api/help/index")
    assert code == 200 and any(p["name"] == "ROLES" for p in j["pages"])
    for path in ("/", "/jobs", "/settings", "/system"):
        _, page = call(studio_srv, path)
        assert "nsFind" in page and "nsProfiles" in page, path
    _, home = call(studio_srv, "/")
    assert 'id="mprof"' in home and "__MODEL_DEFAULTS__" not in home and '"ctx": 8192' in home
    code, opts = call(studio_srv, "/api/jobs/options?kind=testqa")
    assert code == 200 and any(o["flag"] == "--max-concurrency" for o in opts)
    assert call(studio_srv, "/api/jobs/options?kind=nope")[0] == 404
