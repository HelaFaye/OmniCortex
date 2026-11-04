"""Check: repeated passages inside the reply, and phrases reused from earlier replies in the chat."""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import friction  # noqa: E402

d = json.load(sys.stdin)
text = str(d.get("text") or "")
words = re.findall(r"\S+", text)
grams = [" ".join(words[i:i + 4]) for i in range(max(0, len(words) - 3))]
rep = (len(grams) - len(set(grams))) / max(1, len(grams))
earlier = [friction._text(m) for m in d.get("messages") or [] if m.get("role") == "assistant"]
reused = [p for p, _ in friction.repeated_phrases(earlier + [text])[:6] if p in text]
spans = []
for p in reused:
    for m in re.finditer(re.escape(p), text):
        spans.append([m.start(), m.end(), "also in earlier replies"])
print(json.dumps({"label": f"{rep:.0%} of 4-word runs repeated" + (f", {len(reused)} reused phrases" if reused else ""),
                  "score": round(rep, 3), "flag": rep > 0.2, "note": "; ".join(reused), "spans": spans[:100]}))
