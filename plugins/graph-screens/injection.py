"""Graph node: signs of prompt injection in text a model will read. Reads {"params", "inputs"}, prints outputs."""
import json
import re
import sys

SIGNS = [
    ("asks the model to ignore its instructions", r"(?i)\b(ignore|disregard|forget)\b[^.\n]{0,40}\b(previous|prior|above|earlier|all)\b[^.\n]{0,20}\b(instructions?|rules|prompts?|directions)"),
    ("claims to be a system or developer message", r"(?im)^\s*(\[?(system|developer|assistant)\]?\s*[:>]|<\|?(system|im_start)\|?>)"),
    ("asks for the system prompt or secrets", r"(?i)\b(reveal|print|show|repeat|leak)\b[^.\n]{0,40}\b(system prompt|instructions|api key|password|secret)"),
    ("tells the model to call tools or run commands", r"(?i)\b(call|use|invoke|run|execute)\b[^.\n]{0,30}\b(the )?(tool|function|command|shell|terminal)\b"),
    ("hidden characters", r"[​-‏‪-‮⁠-⁤\U000E0000-\U000E007F]"),
    ("tells the model it is now someone else", r"(?i)\byou are now\b|\bnew (role|persona|instructions)\b|\bjailbreak\b|\bDAN\b"),
    ("sends data somewhere", r"(?i)\b(send|post|upload|forward|exfiltrate)\b[^.\n]{0,40}\b(to|at)\b[^.\n]{0,20}(https?://|@)"),
]
d = json.load(sys.stdin)
text = str(d["inputs"].get("text", ""))
found = []
for what, rx in SIGNS:
    hits = re.findall(rx, text)
    if hits:
        found.append({"sign": what, "count": len(hits)})
out = {"findings": found}
if len(found) >= int(d["params"].get("threshold", 2)):
    out["fail"] = "possible prompt injection: " + "; ".join(f["sign"] for f in found)
else:
    out["pass"] = text
print(json.dumps(out))
