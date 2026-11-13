"""Graph node: risky patterns in code. Reads {"params", "inputs"} on stdin, prints its outputs."""
import json
import re
import sys

RULES = [
    ("high", "shell command built from input", r"subprocess\.\w+\([^)]*shell\s*=\s*True|os\.system\(|os\.popen\("),
    ("high", "eval or exec", r"\b(eval|exec)\s*\("),
    ("high", "destructive command", r"rm\s+-rf\s+[/~]|shutil\.rmtree\(\s*['\"]/|mkfs|dd\s+if=.*of=/dev/"),
    ("high", "hard-coded secret", r"(?i)(api[_-]?key|secret|password|token)\s*=\s*['\"][^'\"]{8,}['\"]"),
    ("medium", "certificate checks turned off", r"verify\s*=\s*False|CERT_NONE|check_hostname\s*=\s*False|InsecureRequestWarning"),
    ("medium", "unpickling data", r"pickle\.loads?\(|yaml\.load\((?![^)]*Loader=yaml\.SafeLoader)"),
    ("medium", "SQL built with string formatting", r"(?i)execute\(\s*f?['\"].*(select|insert|update|delete).*(%s|\{|\+)"),
    ("medium", "downloads piped into a shell", r"(curl|wget)[^|\n]*\|\s*(sudo\s+)?(ba|z)?sh"),
    ("low", "broad exception swallowed", r"except\s*(Exception)?\s*:\s*pass"),
    ("low", "world-writable permissions", r"chmod\s+(-R\s+)?777|0o?777"),
]
ORDER = {"low": 0, "medium": 1, "high": 2}
d = json.load(sys.stdin)
code = str(d["inputs"].get("code", ""))
worst = ORDER.get(d["params"].get("fail_on", "medium"), 1)
found = []
for sev, what, rx in RULES:
    for m in re.finditer(rx, code):
        line = code.count("\n", 0, m.start()) + 1
        found.append({"severity": sev, "what": what, "line": line, "text": m.group(0)[:80]})
bad = [f for f in found if ORDER[f["severity"]] >= worst]
out = {"findings": found}
if bad:
    out["fail"] = "; ".join(f"line {f['line']}: {f['what']} ({f['severity']})" for f in bad[:12])
else:
    out["pass"] = code
print(json.dumps(out))
