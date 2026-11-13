"""Reply check: reads {model, messages, text} as JSON on stdin, prints the result as JSON."""
import json
import sys

from claims import spot

req = json.load(sys.stdin)
spans = spot(req.get("text") or "")
kinds = {}
for _, _, k in spans:
    kinds[k] = kinds.get(k, 0) + 1
print(json.dumps({
    "label": f"{len(spans)} claim(s) to verify" if spans else "no checkable claims",
    "score": len(spans),
    "flag": False,
    "note": ", ".join(f"{n} {k}{'s' if n > 1 else ''}" for k, n in sorted(kinds.items())),
    "spans": spans}))
