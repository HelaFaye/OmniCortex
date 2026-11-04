"""
The deficit sieve (docs/POSE.md): checks every actor's skeleton, frame by frame, against what a human body
can do, and turns each failure into a staged DPO pair (the pose as tracked = rejected, a corrected pose =
chosen) for review.

Checks (all thresholds are flags):
    hyperextension  a knee bending forwards past straight by more than --knee-limit degrees (the knee behind
                    the hip-ankle line, measured against the body's forward direction)
    inversion       an elbow or knee whose bend direction flips between frames while it is clearly bent:
                    the joint snapped inside out
    velocity        a joint moving faster than --max-speed m/s (hands and feet: --max-speed-ends)
    bone_length     a bone more than --bone-tol off its median length for that actor (a stretched limb)
    floor_clip      a foot below the floor by more than --floor-eps
    foot_slip       a foot on the floor in consecutive frames while sliding faster than --slip-speed m/s

Corrections (chosen): the failing joint, and the joints below it, interpolated between the nearest frames
before and after in which they passed every check (within --window frames). Without such frames: clamped
(a hyper-extended knee moved onto the hip-ankle line; a clipped foot raised to the floor; a jump held at the
previous position).

    python sieve.py --frames runs/pose/frames.jsonl --out runs/pose/staging [--images runs/pose/images]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

import pose_core as P

LIMB = {"l_knee": ("l_hip", "l_ankle"), "r_knee": ("r_hip", "r_ankle"),
        "l_elbow": ("l_shoulder", "l_wrist"), "r_elbow": ("r_shoulder", "r_wrist")}
BELOW = {}
for _p, _c in P.BONES:
    BELOW.setdefault(_p, []).append(_c)
ENDS = {"l_wrist", "r_wrist", "l_ankle", "r_ankle", "l_toe", "r_toe", "head"}
FEET = ("l_ankle", "r_ankle", "l_toe", "r_toe")
PROMPT = ("Estimate the 3D pose of each person in this frame. Output JSON: joints in metres, y up, for the "
          "topology hips, spine, neck, head, shoulders, elbows, wrists, hips, knees, ankles, toes. Every joint "
          "must be anatomically possible: no hyper-extended or inverted joints, no stretched bones, feet on or "
          "above the floor.")


def subtree(j: str) -> list[str]:
    out, todo = [], [j]
    while todo:
        x = todo.pop()
        out.append(x)
        todo += BELOW.get(x, [])
    return out


def forward(j: dict):
    """The body's forward direction: across the hips (right to left) crossed with up."""
    if "l_hip" not in j or "r_hip" not in j:
        return None
    side = P.sub(j["l_hip"], j["r_hip"])
    f = P.cross(side, [0, 1, 0])
    return P.unit(f) if P.norm(f) > 1e-6 else None


def knee_hyper(j: dict, knee: str) -> float | None:
    """Degrees past straight, positive when the knee bends the wrong way."""
    hip, ankle = LIMB[knee]
    f = forward(j)
    if f is None or not all(k in j for k in (hip, knee, ankle)):
        return None
    a, b = P.sub(j[knee], j[hip]), P.sub(j[ankle], j[knee])
    if P.norm(a) < 1e-6 or P.norm(b) < 1e-6:
        return None
    ang = math.degrees(math.acos(max(-1.0, min(1.0, P.dot(P.unit(a), P.unit(b))))))
    # a normal knee is in front of the hip-ankle line; behind it is hyperextension
    off = P.dot(P.sub(j[knee], P.mid(j[hip], j[ankle])), f)
    return ang if off < 0 else -ang


def bend_normal(j: dict, joint: str):
    a, c = LIMB[joint]
    if not all(k in j for k in (a, joint, c)):
        return None, 0.0
    u, v = P.sub(j[joint], j[a]), P.sub(j[c], j[joint])
    if P.norm(u) < 1e-6 or P.norm(v) < 1e-6:
        return None, 0.0
    ang = math.degrees(math.acos(max(-1.0, min(1.0, P.dot(P.unit(u), P.unit(v))))))
    return P.unit(P.cross(u, v)), ang


def detect(frames: list[dict], cfg: argparse.Namespace) -> list[dict]:
    """-> anomalies [{frame index, frame, t, track, joint, error, value, limit}]."""
    out = []
    tracks = sorted({tid for f in frames for tid in f["actors"]}, key=lambda x: (len(x), x))
    floor = cfg.floor
    if floor is None:
        ys = sorted(a["joints"][k][1] for f in frames for a in f["actors"].values() for k in FEET if k in a["joints"])
        floor = ys[max(0, int(len(ys) * 0.05) - 1)] if ys else 0.0
    for tid in tracks:
        seq = [(i, f) for i, f in enumerate(frames) if tid in f["actors"]]
        lengths = {}
        for _, f in seq:
            j = f["actors"][tid]["joints"]
            for p, c in P.BONES:
                if p in j and c in j:
                    lengths.setdefault(c, []).append(P.norm(P.sub(j[c], j[p])))
        med = {c: statistics.median(v) for c, v in lengths.items() if len(v) >= 3}
        prev, prev_normals = None, {}
        for i, f in seq:
            j = f["actors"][tid]["joints"]

            def flag(joint, error, value, limit):
                out.append({"index": i, "frame": f["frame"], "t": f["t"], "track": tid, "joint": joint,
                            "error": error, "value": round(value, 4), "limit": limit})
            for knee in ("l_knee", "r_knee"):
                h = knee_hyper(j, knee)
                if h is not None and h > cfg.knee_limit:
                    flag(knee, "hyperextension", h, cfg.knee_limit)
            for joint in LIMB:
                n, ang = bend_normal(j, joint)
                pn = prev_normals.get(joint)
                if n is not None and pn is not None and ang > cfg.bent and pn[1] > cfg.bent \
                        and P.dot(n, pn[0]) < math.cos(math.radians(150)):
                    flag(joint, "inversion", math.degrees(math.acos(max(-1, min(1, P.dot(n, pn[0]))))), 150)
                if n is not None:
                    prev_normals[joint] = (n, ang)
            for c, m in med.items():
                p = P.PARENT[c]
                if p in j and c in j and m > 1e-6:
                    r = P.norm(P.sub(j[c], j[p])) / m
                    if abs(r - 1) > cfg.bone_tol:
                        flag(c, "bone_length", r, cfg.bone_tol)
            for k in FEET:
                if k in j and j[k][1] < floor - cfg.floor_eps:
                    flag(k, "floor_clip", floor - j[k][1], cfg.floor_eps)
            if prev is not None:
                pi, pf = prev
                dt = f["t"] - pf["t"]
                pj = pf["actors"][tid]["joints"]
                if dt > 0:
                    for k in j:
                        if k not in pj:
                            continue
                        d = P.sub(j[k], pj[k])
                        speed = P.norm(d) / dt
                        lim = cfg.max_speed_ends if k in ENDS else cfg.max_speed
                        if speed > lim:
                            flag(k, "velocity", speed, lim)
                        if k in FEET and j[k][1] < floor + cfg.contact and pj[k][1] < floor + cfg.contact:
                            hs = math.hypot(d[0], d[2]) / dt
                            if hs > cfg.slip_speed:
                                flag(k, "foot_slip", hs, cfg.slip_speed)
            prev = (i, f)
    for a in out:
        a["floor"] = round(floor, 4)
    return out


def correct(frames: list[dict], anomaly: dict, bad: set, window: int) -> tuple[dict, str]:
    """-> (corrected joints for the actor in that frame, how)."""
    i, tid, joint = anomaly["index"], anomaly["track"], anomaly["joint"]
    j = {k: list(v) for k, v in frames[i]["actors"][tid]["joints"].items()}
    fix = subtree(joint)
    if anomaly["error"] == "foot_slip":
        fix = [joint]

    def good(k):
        return tid in frames[k]["actors"] and not any((k, tid, x) in bad for x in fix) \
            and all(x in frames[k]["actors"][tid]["joints"] for x in fix)
    before = next((k for k in range(i - 1, max(-1, i - window - 1), -1) if good(k)), None)
    after = next((k for k in range(i + 1, min(len(frames), i + window + 1)) if good(k)), None)
    if before is not None and after is not None:
        ta, tb, t = frames[before]["t"], frames[after]["t"], frames[i]["t"]
        w = (t - ta) / (tb - ta) if tb > ta else 0.5
        ja, jb = frames[before]["actors"][tid]["joints"], frames[after]["actors"][tid]["joints"]
        root = P.PARENT.get(joint)
        for x in fix:                         # keep the interpolated limb attached to this frame's parent joint
            off = P.lerp(P.sub(ja[x], ja[root]), P.sub(jb[x], jb[root]), w) if root in ja and root in jb and root in j \
                else None
            j[x] = [round(v, 5) for v in (P.add(j[root], off) if off else P.lerp(ja[x], jb[x], w))]
        return j, f"interpolated between frames {frames[before]['frame']} and {frames[after]['frame']}"
    if anomaly["error"] == "hyperextension":
        hip, ankle = LIMB[joint]
        seg = P.sub(j[ankle], j[hip])
        t = P.dot(P.sub(j[joint], j[hip]), seg) / max(P.dot(seg, seg), 1e-9)
        j[joint] = [round(v, 5) for v in P.add(j[hip], P.mul(seg, t))]
        return j, "clamped onto the hip-ankle line"
    if anomaly["error"] == "floor_clip":
        lift = anomaly["floor"] - j[joint][1]
        for x in fix:
            j[x][1] = round(j[x][1] + lift, 5)
        return j, "raised to the floor"
    if before is not None:
        jb = frames[before]["actors"][tid]["joints"]
        for x in fix:
            j[x] = list(jb[x])
        return j, f"held at frame {frames[before]['frame']}"
    return j, "unchanged: no valid neighbouring frames"


def pose_text(joints: dict) -> str:
    return json.dumps({"joints": {k: [round(x, 3) for x in v] for k, v in joints.items()}}, separators=(",", ":"))


def stage(frames: list[dict], anomalies: list[dict], out: Path, images: Path | None, window: int,
          model: str = "") -> int:
    """Write one pending item per (frame, actor): every failing joint of that frame corrected together."""
    (out / "pending").mkdir(parents=True, exist_ok=True)
    bad = {(a["index"], a["track"], a["joint"]) for a in anomalies}
    groups = {}
    for a in anomalies:
        groups.setdefault((a["index"], a["track"]), []).append(a)
    n = 0
    for (i, tid), items in sorted(groups.items()):
        f = frames[i]
        actor = f["actors"][tid]
        chosen = {k: list(v) for k, v in actor["joints"].items()}
        how = []
        for a in items:
            fixed, h = correct(frames, a, bad, window)
            for x in subtree(a["joint"]):
                chosen[x] = fixed[x]
            how.append(f"{a['joint']} {a['error']}: {h}")
        if all(P.norm(P.sub(chosen[k], actor["joints"][k])) < 1e-4 for k in chosen):
            continue                     # the "fix" changes nothing (e.g. a hand landing back where it belongs)
        iid = hashlib.sha1(f"{model}|{f['frame']}|{tid}|{json.dumps(actor['joints'], sort_keys=True)}".encode()).hexdigest()[:12]
        img = None
        if images is not None:
            for ext in (".jpg", ".png", ".jpeg", ".webp"):
                p = images / f"{f['frame']:06d}{ext}"
                if p.exists():
                    img = str(p)
                    break
        item = {"id": iid, "status": "pending", "frame": f["frame"], "t": f["t"], "track": tid, "model": model,
                "image": img, "px": actor.get("px"), "prompt": PROMPT,
                "rejected": actor["joints"], "chosen": chosen,
                "failures": [{k: a[k] for k in ("joint", "error", "value", "limit")} for a in items], "fix": how}
        p = out / "pending" / f"{iid}.json"
        if not p.exists() and not (out / "approved" / f"{iid}.json").exists() \
                and not (out / "discarded" / f"{iid}.json").exists():
            p.write_text(json.dumps(item, indent=1))
            n += 1
    return n


def args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames", required=True, help="frame payloads, one JSON per line (extract.py writes them)")
    ap.add_argument("--out", required=True, help="staging folder: pending/, approved/, discarded/")
    ap.add_argument("--images", help="folder of the video's frames as 000012.jpg, for review overlays and vision DPO")
    ap.add_argument("--model", default="", help="the pose model's name, kept with each item")
    ap.add_argument("--knee-limit", type=float, default=10.0)
    ap.add_argument("--bent", type=float, default=25.0, help="degrees of bend below which inversion is not checked")
    ap.add_argument("--max-speed", type=float, default=8.0)
    ap.add_argument("--max-speed-ends", type=float, default=14.0)
    ap.add_argument("--bone-tol", type=float, default=0.25)
    ap.add_argument("--floor", type=float, help="floor height in metres (default: estimated from the feet)")
    ap.add_argument("--floor-eps", type=float, default=0.05)
    ap.add_argument("--contact", type=float, default=0.03, help="height above the floor that counts as contact")
    ap.add_argument("--slip-speed", type=float, default=0.35)
    ap.add_argument("--window", type=int, default=8, help="frames to look for valid neighbours")
    ap.add_argument("--report", help="also write every anomaly as JSON here")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    a = args(argv)
    frames = P.read_frames(a.frames)
    an = detect(frames, a)
    n = stage(frames, an, Path(a.out), Path(a.images) if a.images else None, a.window, a.model)
    kinds = {}
    for x in an:
        kinds[x["error"]] = kinds.get(x["error"], 0) + 1
    print(f"{len(frames)} frames, {len(an)} anomalies " + json.dumps(kinds) + f"; {n} new items staged in {a.out}/pending")
    if a.report:
        Path(a.report).write_text(json.dumps(an, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
