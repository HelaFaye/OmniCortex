"""Contextual compression: passages trimmed to what the question needs; long chats cut or summarized."""
import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "viz"))
sys.path.insert(0, str(ROOT / "scripts"))
import context_compress as cc  # noqa: E402
from test_studio import call, studio_srv  # noqa: E402,F401

DOC = ("The museum opened in 1901. It has three floors. The cafe serves soup on Mondays. "
       "The roof garden closes in winter. Tickets for the Egyptian wing cost 12 euros. Children enter free. "
       "The gift shop sells postcards. Parking is behind the station.")


def test_passages_keep_what_the_question_needs():
    hits = [{"source": "guide.txt", "text": DOC, "score": 1.0}]
    out, info = cc.compress_passages("How much is a ticket for the Egyptian wing?", hits, budget=40, neighbours=0)
    assert "12 euros" in out[0]["text"] and "soup" not in out[0]["text"]
    assert info["after"] < info["before"] and out[0]["compressed"] and out[0]["source"] == "guide.txt"
    # with neighbours the next sentence comes along, for context
    out, _ = cc.compress_passages("Egyptian wing tickets", hits, budget=200, neighbours=1)
    assert "Children enter free" in out[0]["text"]
    # a question with no matching words still leaves something to cite
    out, _ = cc.compress_passages("zebra", hits, budget=40)
    assert out[0]["text"]


def test_fit_history():
    sys_m = {"role": "system", "content": "be brief"}
    convo = [{"role": "user" if i % 2 == 0 else "assistant", "content": "word " * 50} for i in range(10)]
    assert cc.fit_history([sys_m] + convo, 10_000) == 0
    cut = cc.fit_history([sys_m] + convo, 300)
    assert cut > 0 and cut % 2 == 0 and cut <= 6        # lands on a user message; the last 4 stay
    assert cc.fit_history([sys_m] + convo, 1) == 6
    p = cc.summary_prompt(convo[:2], previous="earlier")
    assert "EARLIER SUMMARY: earlier" in p[1]["content"]


def stream(base, body):
    req = urllib.request.Request(base + "/api/chat", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    events = []
    with urllib.request.urlopen(req, timeout=30) as r:
        for line in r.read().decode().splitlines():
            if line.startswith("data:") and line[5:].strip() != "[DONE]":
                events.append(json.loads(line[5:]))
    return events


def test_chat_cuts_or_summarizes(studio_srv, monkeypatch):
    import studio
    monkeypatch.setattr(studio, "_ctx_size", lambda: 2400)        # a small window, so the chat overflows
    convo = [{"role": "user" if i % 2 == 0 else "assistant", "content": f"turn {i} " + "words " * 300}
             for i in range(9)]
    model = "coder-gguf/coder-7b-q4_k_m"
    ev = stream(studio_srv, {"model": model, "messages": convo, "compress": {"history": "summarize"}})
    comp = next(e["compressed"] for e in ev if "compressed" in e)["history"]
    assert comp["history"] == "summarized" and comp["upto"] >= 2 and comp["summary"]
    text = "".join(c["delta"].get("content", "") for e in ev for c in e.get("choices", []))
    assert "summary=yes" in text
    # the next request sends the summary back: no new summary needed while it fits
    ev = stream(studio_srv, {"model": model, "messages": convo, "compress": {"history": "summarize"},
                             "summary": {"upto": comp["upto"], "text": comp["summary"]}})
    again = next(e["compressed"] for e in ev if "compressed" in e)["history"]
    assert again["history"] in ("summary", "summarized") and again["upto"] >= comp["upto"]
    ev = stream(studio_srv, {"model": model, "messages": convo, "compress": {"history": "cut"}})
    cut = next(e["compressed"] for e in ev if "compressed" in e)["history"]
    assert cut["history"] == "cut" and cut["dropped"] >= 2
    # off: nothing is touched
    ev = stream(studio_srv, {"model": model, "messages": convo[:1]})
    assert not any("compressed" in e for e in ev)
    # the context count follows the summary
    _, c = call(studio_srv, "/api/context", {"messages": convo, "compress": {"history": "summarize"},
                                            "summary": {"upto": 4, "text": "short"}})
    assert c["summarized"] == 4 and c["turns"] == 6
