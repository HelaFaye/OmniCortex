"""Named themes: UI colours, chart colours and the 3D viewer theme, from one file."""
import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "viz"))
sys.path.insert(0, str(ROOT / "tests"))
from test_studio import call, studio_srv  # noqa: E402,F401


def test_every_theme_is_complete():
    d = json.loads((ROOT / "viz" / "themes.json").read_text())
    viz = {k for k in d if not k.startswith("_")}
    for name, t in d["_studio"].items():
        assert t["base"] in ("dark", "light") and t["viz"] in viz, name
        assert len(t["chart"]) >= 2, name
        if t["ui"]:
            assert {"bg", "panel", "fg", "mut", "line", "acc", "ok", "no"} <= set(t["ui"]), name


def test_api_default_and_viewer_cookie(studio_srv):
    import studio
    st, j = call(studio_srv, "/api/themes")
    assert st == 200 and "arcade" in j["themes"] and j["default"] in j["themes"]
    st, r = call(studio_srv, "/api/settings/theme", {"theme": "nope"})
    assert st == 400
    st, r = call(studio_srv, "/api/settings/theme", {"theme": "arcade"})
    assert st == 200 and studio.STATE["viz_theme"] == "arcade"
    h = studio.Handler.__new__(studio.Handler)
    h.headers = {"Cookie": "oc_viz=ember"}
    assert h._viz_theme({"x": 1})["active"] == "#EF9F27"
    h.headers = {"Cookie": "oc_viz=nonsense"}
    t = h._viz_theme({"x": 1})
    assert t["x"] == 1 and "jewel" in t["widgets"]          # the 3D theme carries the widget settings (WIDGETS.md)
    with urllib.request.urlopen(studio_srv + "/settings") as r:
        page = r.read().decode()
    assert "id=\"appearance\"" in page and "ns-theme-vars" in page and "function nsApplyTheme" in page
