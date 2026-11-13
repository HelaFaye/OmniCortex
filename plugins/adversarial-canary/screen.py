"""Check (no model call): signs that the user's message or pasted text tries to instruct the model."""
import json
import sys

import probes

d = json.load(sys.stdin)
users = [m for m in d.get("messages") or [] if m.get("role") == "user"]
last = users[-1].get("content") if users else ""
last = last if isinstance(last, str) else " ".join(p.get("text", "") for p in last if isinstance(p, dict))
signs = probes.injection_signs(last)
print(json.dumps({"label": "injection signs: " + ", ".join(signs) if signs else "no injection signs", "flag": bool(signs),
                  "score": len(signs)}))
