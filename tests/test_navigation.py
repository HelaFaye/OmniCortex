"""The sidebar is on every page, and every item in it leads somewhere real."""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "viz"))
sys.path.insert(0, str(ROOT / "scripts"))
import ui_links  # noqa: E402
from test_studio import call, studio_srv  # noqa: E402,F401


def test_sidebar_items_lead_to_pages(studio_srv):
    js = ui_links.js()
    items = re.findall(r"\{href:'([^']+)',(?:tab:'(\w+)',)?label:'([^']+)'", js)
    labels = [i[2] for i in items]
    assert labels[:3] == ["Chat", "My Models", "Discover"] and "Settings" in labels
    _, home = call(studio_srv, "/")
    for href, tab, label in items:
        path = href.split("#")[0] or "/"
        code, page = call(studio_srv, path)
        assert code == 200 and "nsRail" in page, (label, path)
        if tab:
            assert f'data-tab="{tab}"' in home, label
            assert href.split("#")[1] in js, label                     # the main page's hash names it
    assert 'href="/jobs" class="chip"' not in home                     # the old header row is gone
    assert ">Discover</button>" in home


def test_context_usage_by_part(studio_srv):
    msgs = [{"role": "user", "content": "one two three four"}, {"role": "assistant", "content": "five six"}]
    code, c = call(studio_srv, "/api/context", {"messages": msgs, "documents": "a passage of eight characters"})
    assert code == 200 and c["method"] == "estimate" and c["n_ctx"] is None
    parts = {p["key"]: p["tokens"] for p in c["parts"]}
    assert set(parts) == {"system", "documents", "tools", "messages"} and parts["documents"] > 0
    assert c["total"] == sum(parts.values()) and c["turns"] == 2 and c["reply_reserve"] > 0
    # With a model running, its tokenizer counts (the fake server: one token per word) and its window is known.
    call(studio_srv, "/v1/chat/completions", {"model": "coder-gguf/coder-7b-q4_k_m",
                                              "messages": [{"role": "user", "content": "hi"}]})
    code, c = call(studio_srv, "/api/context", {"messages": [{"role": "system", "content": "be brief"}] + msgs})
    parts = {p["key"]: p["tokens"] for p in c["parts"]}
    assert c["method"] == "tokenizer" and c["n_ctx"] == 8192 and parts["system"] == 2
    assert parts["messages"] == 8                  # "user: one two three four" + "assistant: five six"
    _, home = call(studio_srv, "/")
    assert 'id="ctxStrip"' in home and 'id="side"' in home


def test_sidebar_groups():
    js = ui_links.js()
    groups = {g: re.findall(r"label:'([^']+)'", body)
              for g, body in re.findall(r"\{label:'([^']+)',icon:'\w+',tip:'[^']*'(?:,mobile:1)?,kids:\[(.*?)\]\}", js, re.S)}
    assert groups == {"Models": ["My Models", "Discover"], "Tasks": ["Jobs", "Projects", "Workers", "Distribution"],
                      "Studio": ["Features", "Graphs", "Review", "Lab"], "Connect": ["Apps", "Remote hosts", "Devices"]}
    for fn in ("nsSub", "nsGroupHref", "ns-fly"):                   # sibling tabs, last-used page, flyout
        assert fn in js
    # the Godot client groups its tabs the same way (its 3D view is its own; Workers is web-only)
    gd = (ROOT / "plugins" / "godot-client" / "client" / "app.gd").read_text()
    gd_groups = {g: re.findall(r'"([^"]+)"', body)
                 for g, body in re.findall(r'"(\w+)": \[([^\]]*)\]', re.search(r"const GROUPS := \{(.*?)\}", gd).group(1))}
    for g, kids in gd_groups.items():
        assert g in groups and set(kids) - {"3D"} <= set(groups[g]), (g, kids)
    tour = re.findall(r"\{path:'([^']+)'", js)
    hrefs = {h.split("#")[0] or "/" for h in re.findall(r"href:'([^']+)'", js)}
    assert set(tour) <= hrefs                                         # the tour only stops on real pages
