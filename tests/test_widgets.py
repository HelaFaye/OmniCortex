"""Widgets (viz/widgets.py): settings checks, merging, and Studio serving them with the 3D theme."""
import json
import sys
import urllib.request
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "viz"))
sys.path.insert(0, str(ROOT / "tests"))
import widgets  # noqa: E402
from test_studio import studio_srv  # noqa: E402,F401


def test_validate_and_merge():
    # icosphere and tesseract come from the jewel-styles add-on: only valid while it provides them
    assert widgets.validate({"jewel": {"style": "tesseract", "speed": 2}}, ["icosphere", "tesseract"]) == \
        {"jewel": {"style": "tesseract", "speed": 2.0}}
    with pytest.raises(ValueError):
        widgets.validate({"jewel": {"style": "tesseract"}})
    assert widgets.spec(["icosphere"])["jewel"]["style"]["choices"] == ["core", "icosphere", "off"]
    for bad in ({"jewel": {"style": "cube"}}, {"jewel": {"speed": 9}}, {"jewel": {"nope": 1}}, {"x": {}},
                {"emotion_graph": {"labels": "yes"}}, {"jewel": {"speed": True}}):
        with pytest.raises(ValueError):
            widgets.validate(bad)
    m = widgets.merged({"widgets": {"jewel": {"glow": 0.5, "junk": 1}}}, {"jewel": {"style": "icosphere"}})
    assert m["jewel"] == {"style": "icosphere", "speed": 1.0, "glow": 0.5, "scale": 0.9}
    assert m["nav_gizmo"]["sway"] is True
    t = widgets.with_widgets({"bg": "#000"})
    assert t["bg"] == "#000" and t["widgets"]["jewel"]["style"] == "core"


def test_pages_carry_the_widgets():
    import bloom
    import settings_page
    import stream
    for page in (bloom.PAGE, stream.PAGE, settings_page.PAGE):
        assert "const NSWidgets" in page and "/*OC_WIDGETS*/" not in page
    assert "id=\"jewel\"" in bloom.PAGE and "id=\"jewel\"" in stream.PAGE and "wdPrev" in settings_page.PAGE
    # Godot and web draw the same styles: the core ones, and the add-on's in both its scripts
    gd = (ROOT / "plugins" / "godot-client" / "client" / "widgets" / "jewel.gd").read_text()
    for style in widgets.SPEC["jewel"]["style"]["choices"]:
        assert f'"{style}"' in gd and f"'{style}'" in widgets.WIDGETS_JS
    addon = ROOT / "plugins" / "jewel-styles"
    m = json.loads((addon / "plugin.json").read_text())
    for style in m["widgets"]["jewel_styles"]:
        assert f'"{style}"' in (addon / "page" / "styles.gd").read_text()
        assert f"'{style}'" in (addon / "page" / "styles.js").read_text()
        assert f"'{style}'" not in widgets.WIDGETS_JS                     # no longer drawn by the core


def _post(base, path, body):
    req = urllib.request.Request(base + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def test_studio_saves_and_serves_widgets(studio_srv):  # noqa: F811
    j = json.loads(urllib.request.urlopen(studio_srv + "/api/themes", timeout=10).read())
    assert set(j["widgets"]["spec"]) == set(widgets.SPEC) and j["widgets"]["values"]["jewel"]["style"] == "core"
    import urllib.request as u
    assets = json.loads(u.urlopen(studio_srv + "/api/addons/widgets", timeout=10).read())
    assert "tesseract" in assets["styles"] and assets["js"][0]["url"] == "/plugins/jewel-styles/styles.js"
    assert u.urlopen(studio_srv + assets["gd"][0]["url"], timeout=10).read().startswith(b"extends RefCounted")
    code, r = _post(studio_srv, "/api/settings/widgets", {"widgets": {"jewel": {"style": "cube"}}})
    assert code == 400
    code, r = _post(studio_srv, "/api/settings/widgets", {"widgets": {"jewel": {"style": "tesseract"}}})
    assert code == 200 and r["values"]["jewel"]["style"] == "tesseract"
    j = json.loads(urllib.request.urlopen(studio_srv + "/api/themes", timeout=10).read())
    assert j["widgets"]["values"]["jewel"]["style"] == "tesseract"
    _post(studio_srv, "/api/settings/widgets", {"widgets": {}})
