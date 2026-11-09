#!/usr/bin/env python3
"""Reproduce one sample's span search with nothing caught."""
import json, os, subprocess, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gguf_tokenizer

if len(sys.argv) > 1 or not os.environ.get("OC_GGUF") or not os.environ.get("OC_CETT"):
    raise SystemExit("usage: source env.sh && python scripts/debug_span_search.py\n"
                     "Re-runs the answer-span search for the first training qid in\n"
                     "data/train_qids.json using cett-dump's tokenizer (needs OC_GGUF, OC_CETT).")
GGUF = os.environ["OC_GGUF"]; CETT = os.environ["OC_CETT"]

qid = json.load(open("data/train_qids.json"))["t"][0]
rec = None
for line in open("data/consistency_samples.jsonl", encoding="utf-8"):
    d = json.loads(line)
    if qid in d:
        rec = d[qid]; break
print(f"qid      : {qid}")
print(f"answer   : {rec.get('answer')!r}")
print(f"response : {rec['response'][:120]!r}")
print(f"fields   : {sorted(rec.keys())}")

gt = gguf_tokenizer.load(GGUF)
prompt = gt.render_chat([{"role": "user", "content": rec["question"]}], True)
text = prompt + rec["response"]
print(f"\nprompt rendered, {len(prompt)} chars")

work = tempfile.mkdtemp()
man = os.path.join(work, "m.jsonl")
open(man, "w", encoding="utf-8").write(
    json.dumps({"id": qid, "text": text}, ensure_ascii=False) + "\n")
r = subprocess.run([CETT, "-m", GGUF, "--tokenize-only", "-ngl", "0",
                    "--manifest", man, "--outdir", work],
                   capture_output=True, text=True)
print(f"tokenize exit: {r.returncode}")
tf = os.path.join(work, f"{qid}.toks")
if not os.path.exists(tf):
    print("NO .toks FILE. stderr tail:"); print(r.stderr[-600:]); sys.exit(1)

import struct
b = open(tf, "rb").read()
n = struct.unpack_from("<I", b, 4)[1-1] if False else struct.unpack_from("<I", b, 8)[0]
print(f"toks file {len(b)} bytes")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import importlib.util, types
for m in ("torch","transformers"):
    sys.modules.setdefault(m, types.ModuleType(m))
for nme in ("AutoModelForCausalLM","AutoTokenizer","AutoConfig"):
    setattr(sys.modules["transformers"], nme, object)
spec = importlib.util.spec_from_file_location("ex", "scripts/extract_activations_gguf.py")
ex = importlib.util.module_from_spec(spec); sys.modules["ex"] = ex
try: spec.loader.exec_module(ex)
except SystemExit: pass

ids = ex.read_tokens(tf)
pieces = gt.pieces(ids)
print(f"\n{len(ids)} tokens")
print(f"first 12 pieces : {pieces[:12]}")
print(f"'</think>' present: {any('</think>' in p for p in pieces)}")
print(f"THINK_CLOSE const : {ex.THINK_CLOSE!r}")

acc = pl = 0
for i, p in enumerate(pieces):
    acc += len(p)
    if acc >= len(prompt):
        pl = i + 1; break
print(f"prompt_len resolved: {pl}")
print(f"pieces around it   : {pieces[max(0,pl-3):pl+6]}")

import inspect
print(f"\nfind_regions signature: {inspect.signature(ex.find_regions)}")
reg = ex.find_regions(pieces, pl, [], True, answer_text=rec.get("answer"))
print(f"regions: {reg}")
if reg.get("answer_tokens"):
    a, z = reg["answer_tokens"]
    print(f"ANSWER SPAN OK -> {''.join(pieces[a:z])!r}")
else:
    tail = "".join(pieces[pl:])
    print(f"NO SPAN. answer={rec.get('answer')!r}")
    print(f"decoded tail (300 chars): {tail[:300]!r}")
