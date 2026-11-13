"""Graph node: the canary probes on a prompt before the main model answers it."""
import json
import sys

import probes

d = json.load(sys.stdin)
prompt = str(d["inputs"].get("prompt", ""))
model = d["params"].get("model") or None
msgs = [{"role": "user", "content": prompt}]
report = []
try:
    first = probes.plugin_api.chat(model, msgs, 200)
    st = probes.stability(model, msgs, first, int(d["params"].get("variants", 2)))
    report.append({"probe": "stability", "agreement": st})
    c = probes.canary(model, msgs)
    if c is not None:
        report.append({"probe": "canary", **c})
except Exception as e:
    print(json.dumps({"error": f"canary: {e}"}))
    raise SystemExit(0)
signs = probes.injection_signs(prompt)
if signs:
    report.append({"probe": "screen", "signs": signs})
bad = (st is not None and st < 0.35) or any(r.get("obeyed") for r in report) or bool(signs)
print(json.dumps({"report": report, **({"fail": "canary: " + json.dumps(report)} if bad else {"pass": prompt})}))
