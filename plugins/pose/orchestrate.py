"""
The refinement orchestrator (docs/POSE.md): watches the approved pool and, once --threshold new approved
pairs have accumulated, runs one refinement cycle:

  1. pause      creates <staging>/PAUSE; stream.py --staging holds while it exists
  2. export     the approved pairs as a dataset (staging.py export)
  3. sweep      finetune.py --dpo (QLoRA, NF4, bf16 where the GPU has it) for every --lrs x --alphas pair,
                each a short run; reads each run's training log (omnicortex-finetune.json, or TensorBoard
                event files with --report-to tensorboard)
  4. pick       the run with the largest final reward margin (chosen minus rejected) whose loss stayed finite
                and never rose above --explode x its first value; ties go to the lower learning rate
  5. promote    writes the winner's values into --promote (a JSON or YAML settings file), and keeps a copy of
                the previous one
  6. export     optionally merges the winner into a GGUF (merge_export.py) and, with --install TARGET, swaps
                it in: the current TARGET is kept as TARGET.bak-<time> first, the new file is moved in with an
                atomic rename, so a model server that reopens the file gets the new weights
  7. resume     removes PAUSE; records which pairs this cycle used, so they are not counted again

Without --model it stops after the export, which is how to try the loop without a GPU.

H-neurons: llama-cett-dump reads text only, so it cannot trace a mistake on a video frame. The vision side has
its own path: scripts/clip_neurons.py finds the vision encoder's overconfident neurons from images it gets
consistently wrong (the approved frames are such images), classifier.py turns them into h_neurons.json, and
--suppress h_neurons.json --suppress-scale 0.5 downscales them in the merged projector (suppress_mmproj.py).

    python orchestrate.py --staging runs/pose/staging --work runs/pose/cycles --threshold 50 --watch \\
        --model org/pose-vlm --lrs 5e-6,1e-5 --alphas 16,32 --promote configs/pose.yaml
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import staging as S

HERE = Path(__file__).resolve().parent
SCRIPTS = HERE.parents[1] / "scripts"


def dpo_curve(run: Path) -> list[dict]:
    """[{step, loss, margin}] from finetune.py's log, else from TensorBoard event files."""
    meta = run / "omnicortex-finetune.json"
    rows = []
    if meta.exists():
        log = (json.loads(meta.read_text()).get("log") or {}).get("dpo") or []
        for r in log:
            if "loss" in r or "rewards/margins" in r:
                rows.append({"step": r.get("step"), "loss": r.get("loss"), "margin": r.get("rewards/margins")})
    if not rows:
        try:
            from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        except ImportError:
            return rows
        for ev in sorted(run.rglob("events.out.tfevents.*")):
            acc = EventAccumulator(str(ev.parent))
            acc.Reload()
            tags = acc.Tags().get("scalars", [])
            loss = {e.step: e.value for e in acc.Scalars("train/loss")} if "train/loss" in tags else {}
            marg = {e.step: e.value for e in acc.Scalars("train/rewards/margins")} if "train/rewards/margins" in tags else {}
            rows += [{"step": s, "loss": loss.get(s), "margin": marg.get(s)} for s in sorted(set(loss) | set(marg))]
    return rows


def judge(curve: list[dict], explode: float) -> dict:
    losses = [r["loss"] for r in curve if isinstance(r.get("loss"), (int, float))]
    margins = [r["margin"] for r in curve if isinstance(r.get("margin"), (int, float))]
    if not losses or not margins:
        return {"ok": False, "why": "no training log"}
    if any(not math.isfinite(x) for x in losses + margins):
        return {"ok": False, "why": "loss or margin became NaN/inf"}
    if max(losses) > explode * max(losses[0], 1e-6):
        return {"ok": False, "why": f"loss rose to {max(losses):.3g} from {losses[0]:.3g}"}
    tail = margins[-max(1, len(margins) // 4):]                  # the last quarter, to smooth noise
    return {"ok": True, "margin": sum(tail) / len(tail), "min_loss": min(losses), "final_loss": losses[-1]}


def pick(results: list[dict]) -> dict | None:
    ok = [r for r in results if r["judge"]["ok"]]
    return max(ok, key=lambda r: (round(r["judge"]["margin"], 6), -r["lr"])) if ok else None


def promote(path: Path, values: dict) -> None:
    if path.exists():
        shutil.copyfile(path, path.with_name(path.name + ".prev"))
    if path.suffix in (".yaml", ".yml"):
        import yaml
        cur = yaml.safe_load(path.read_text()) if path.exists() else {}
        path.write_text(yaml.safe_dump({**(cur or {}), **values}, sort_keys=False))
    else:
        cur = json.loads(path.read_text()) if path.exists() else {}
        path.write_text(json.dumps({**cur, **values}, indent=1))


def install(new: Path, target: Path) -> Path:
    """Back up target, then move new in place atomically. -> the backup's path."""
    bak = target.with_name(f"{target.name}.bak-{int(time.time())}")
    if target.exists():
        shutil.copy2(target, bak)
    tmp = target.with_name(target.name + ".incoming")
    shutil.copyfile(new, tmp)
    os.replace(tmp, target)
    return bak


def run(cmd: list[str], log) -> int:
    log("$ " + " ".join(cmd))
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in p.stdout.splitlines()[-40:]:
        log("  " + line)
    return p.returncode


def cycle(a, st: Path, work: Path, new_ids: list[str], n: int, log=print) -> dict:
    out = work / f"cycle-{n:03d}"
    (st / "PAUSE").write_text(str(out))
    rec = {"cycle": n, "pairs": len(new_ids), "started": time.time()}
    try:
        rec["data"] = S.export(st, out / "data")
        if not a.model:
            rec["note"] = "exported only (no --model)"
            return rec
        results = []
        for lr, alpha in itertools.product(a.lrs, a.alphas):
            cand = out / f"lr{lr:g}-a{alpha}"
            cmd = [sys.executable, str(SCRIPTS / "finetune.py"), "--model", a.model, "--data", str(out / "data"),
                   "--out", str(cand), "--method", a.method, "--dpo", "--dpo-lr", f"{lr:g}", "--lora-alpha", str(alpha),
                   "--max-steps", str(a.sft_steps), "--dpo-max-steps", str(a.steps), "--logging-steps", "1"]
            if rec["data"]["vision"]:
                cmd.append("--vision")
            if a.report_to:
                cmd += ["--report-to", a.report_to]
            code = run(cmd, log)
            j = judge(dpo_curve(cand), a.explode) if code == 0 else {"ok": False, "why": f"finetune.py exited {code}"}
            results.append({"lr": lr, "alpha": alpha, "out": str(cand), "judge": j})
            log(f"  lr {lr:g}, alpha {alpha}: " + (f"margin {j['margin']:.4f}" if j["ok"] else j["why"]))
        rec["sweep"] = results
        best = pick(results)
        rec["best"] = best
        if best is None:
            rec["note"] = "no run passed; nothing promoted"
            return rec
        if a.promote:
            promote(Path(a.promote), {"dpo_lr": best["lr"], "lora_alpha": best["alpha"]})
            rec["promoted"] = a.promote
        if a.gguf:
            gg = out / "merged"
            cmd = [sys.executable, str(SCRIPTS / "merge_export.py"), "--base", a.model, "--adapter", best["out"],
                   "--out", str(gg), "--gguf", a.gguf] + (["--vision"] if rec["data"]["vision"] else [])
            if run(cmd, log) != 0:
                rec["note"] = "merge failed; nothing installed"
                return rec
            files = sorted(gg.rglob("*.gguf"), key=lambda p: p.stat().st_size, reverse=True)
            main_file = next((f for f in files if "mmproj" not in f.name), None)
            proj = next((f for f in files if "mmproj" in f.name), None)
            if a.suppress and proj is not None:
                sup = proj.with_name(proj.stem + "-suppressed.gguf")
                if run([sys.executable, str(SCRIPTS / "suppress_mmproj.py"), "--mmproj", str(proj), "--h_neurons",
                        a.suppress, "--scale", f"{a.suppress_scale:g}", "--out", str(sup)], log) == 0:
                    rec["mmproj"] = str(sup)
            rec["gguf"] = str(main_file) if main_file else None
            if a.install and main_file:
                rec["backup"] = str(install(main_file, Path(a.install)))
                rec["installed"] = a.install
        return rec
    finally:
        (st / "PAUSE").unlink(missing_ok=True)
        rec["finished"] = time.time()
        out.mkdir(parents=True, exist_ok=True)
        (out / "cycle.json").write_text(json.dumps(rec, indent=1, default=str))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--staging", required=True)
    ap.add_argument("--work", required=True, help="where each cycle's data, runs and report go")
    ap.add_argument("--threshold", type=int, default=50, help="new approved pairs that start a cycle")
    ap.add_argument("--watch", action="store_true", help="keep watching (otherwise check once)")
    ap.add_argument("--poll", type=float, default=30.0, help="seconds between checks")
    ap.add_argument("--model", help="HF id or path of the pose model to refine (bf16/fp16 weights)")
    ap.add_argument("--method", default="qlora", choices=["qlora", "lora"])
    ap.add_argument("--lrs", default="5e-6,1e-5,2e-5", type=lambda s: [float(x) for x in s.split(",")])
    ap.add_argument("--alphas", default="16,32", type=lambda s: [int(x) for x in s.split(",")])
    ap.add_argument("--steps", type=int, default=60, help="DPO steps per sweep run")
    ap.add_argument("--sft-steps", type=int, default=20, help="SFT steps on the chosen poses before DPO")
    ap.add_argument("--explode", type=float, default=3.0)
    ap.add_argument("--report-to", default="", help="tensorboard, to also write event files")
    ap.add_argument("--promote", help="settings file (JSON or YAML) to write the winning values into")
    ap.add_argument("--gguf", nargs="?", const="Q8_0", help="merge the winner into a GGUF of this quant")
    ap.add_argument("--suppress", help="h_neurons.json (clip_neurons.py + classifier.py) for the merged projector")
    ap.add_argument("--suppress-scale", type=float, default=0.5)
    ap.add_argument("--install", help="model file to replace with the merged GGUF (kept as .bak-<time>)")
    a = ap.parse_args(argv)
    st, work = Path(a.staging), Path(a.work)
    work.mkdir(parents=True, exist_ok=True)
    state_p = work / "orchestrator.json"
    state = json.loads(state_p.read_text()) if state_p.exists() else {"used": [], "cycles": 0}
    while True:
        approved = [it["id"] for it in S.items(st, "approved")]
        new = [i for i in approved if i not in set(state["used"])]
        print(f"{len(new)} new approved pairs (threshold {a.threshold})", flush=True)
        if len(new) >= a.threshold:
            state["cycles"] += 1
            rec = cycle(a, st, work, new, state["cycles"], log=lambda s: print(s, flush=True))
            state["used"] += new
            state_p.write_text(json.dumps(state, indent=1))
            print(json.dumps({k: rec.get(k) for k in ("cycle", "pairs", "note", "best", "installed")}, default=str))
        if not a.watch:
            return 0
        time.sleep(a.poll)


if __name__ == "__main__":
    raise SystemExit(main())
