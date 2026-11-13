"""Job: collect the checkable claims from Studio's saved chats into a CSV."""
import argparse
import csv
import json
from pathlib import Path

from claims import spot


def text_of(c):
    if isinstance(c, list):
        return " ".join(p.get("text", "") for p in c if isinstance(p, dict))
    return str(c or "")


p = argparse.ArgumentParser()
p.add_argument("--chats", required=True)
p.add_argument("--out", required=True)
p.add_argument("--model")
a = p.parse_args()
rows = 0
with open(Path(a.out).expanduser(), "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["chat", "model", "question", "kind", "claim", "context"])
    for fp in sorted(Path(a.chats).expanduser().glob("*.json")):
        try:
            chat = json.loads(fp.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        q = ""
        for m in chat.get("messages") or []:
            if m.get("role") == "user":
                q = text_of(m.get("content"))
                continue
            if m.get("role") != "assistant" or (a.model and m.get("model") != a.model):
                continue
            t = text_of(m.get("content"))
            for s, e, kind in spot(t):
                w.writerow([fp.stem, m.get("model", ""), q[:300], kind, t[s:e], t[max(0, s - 60):e + 60]])
                rows += 1
print(f"{rows} claims written to {a.out}")
