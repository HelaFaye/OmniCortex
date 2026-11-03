"""
Skeletons, frame payloads, multi-person tracking and joint rotations for the pose add-on (docs/POSE.md).

A frame payload (one JSON object per line in a .jsonl file, and what the stream sends):

    {"frame": 12, "t": 0.4, "actors": {"1": {
        "joints": {"hips": [x, y, z], ...},        metres, y up
        "rot":    {"l_knee": [w, x, y, z], ...},   per bone, relative to the rest (T) pose; swing only, see below
        "px":     {"hips": [u, v], ...},           where the joint is in the video frame, 0..1 (optional)
        "conf":   {"hips": 0.97, ...}}}}           detector confidence (optional)

Rotations are the shortest-arc rotation from a bone's rest direction to its current direction ("swing"). The
twist about the bone's own axis cannot be recovered from joint positions alone; estimators that output full
rotations (WHAM, SMPL-X fits) should pass theirs through instead.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

JOINTS = ["hips", "spine", "neck", "head",
          "l_shoulder", "l_elbow", "l_wrist", "r_shoulder", "r_elbow", "r_wrist",
          "l_hip", "l_knee", "l_ankle", "l_toe", "r_hip", "r_knee", "r_ankle", "r_toe"]
PARENT = {"spine": "hips", "neck": "spine", "head": "neck",
          "l_shoulder": "neck", "l_elbow": "l_shoulder", "l_wrist": "l_elbow",
          "r_shoulder": "neck", "r_elbow": "r_shoulder", "r_wrist": "r_elbow",
          "l_hip": "hips", "l_knee": "l_hip", "l_ankle": "l_knee", "l_toe": "l_ankle",
          "r_hip": "hips", "r_knee": "r_hip", "r_ankle": "r_knee", "r_toe": "r_ankle"}
BONES = [(p, c) for c, p in PARENT.items()]
# rest (T-pose) bone directions, y up, the actor facing +z, its left on +x
REST = {"spine": (0, 1, 0), "neck": (0, 1, 0), "head": (0, 1, 0),
        "l_shoulder": (1, 0, 0), "l_elbow": (1, 0, 0), "l_wrist": (1, 0, 0),
        "r_shoulder": (-1, 0, 0), "r_elbow": (-1, 0, 0), "r_wrist": (-1, 0, 0),
        "l_hip": (1, 0, 0), "l_knee": (0, -1, 0), "l_ankle": (0, -1, 0), "l_toe": (0, 0, 1),
        "r_hip": (-1, 0, 0), "r_knee": (0, -1, 0), "r_ankle": (0, -1, 0), "r_toe": (0, 0, 1)}

# MediaPipe Pose landmarker (33 landmarks) -> these joints. Joints MediaPipe lacks are midpoints.
MEDIAPIPE = {"l_shoulder": 11, "r_shoulder": 12, "l_elbow": 13, "r_elbow": 14, "l_wrist": 15, "r_wrist": 16,
             "l_hip": 23, "r_hip": 24, "l_knee": 25, "r_knee": 26, "l_ankle": 27, "r_ankle": 28,
             "l_toe": 31, "r_toe": 32, "head": 0}
# OpenPose BODY_25 keypoint order -> these joints
BODY25 = {"neck": 1, "r_shoulder": 2, "r_elbow": 3, "r_wrist": 4, "l_shoulder": 5, "l_elbow": 6, "l_wrist": 7,
          "hips": 8, "r_hip": 9, "r_knee": 10, "r_ankle": 11, "l_hip": 12, "l_knee": 13, "l_ankle": 14,
          "head": 0, "l_toe": 19, "r_toe": 22}


# -- small vector maths (no numpy needed, so engine-side scripts can reuse this file) ------------------------
def sub(a, b): return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
def add(a, b): return [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
def mul(a, s): return [a[0] * s, a[1] * s, a[2] * s]
def dot(a, b): return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
def cross(a, b): return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
def norm(a): return math.sqrt(dot(a, a))
def lerp(a, b, t): return [a[i] + (b[i] - a[i]) * t for i in range(3)]


def unit(a):
    n = norm(a)
    return [a[0] / n, a[1] / n, a[2] / n] if n > 1e-12 else [0.0, 0.0, 0.0]


def mid(a, b):
    return [(a[i] + b[i]) / 2 for i in range(3)]


def shortest_arc(u, v) -> list[float]:
    """Unit quaternion [w, x, y, z] rotating direction u onto direction v."""
    u, v = unit(u), unit(v)
    d = dot(u, v)
    if d < -0.999999:                                     # opposite: any perpendicular axis
        ax = cross([1, 0, 0], u)
        if norm(ax) < 1e-6:
            ax = cross([0, 1, 0], u)
        ax = unit(ax)
        return [0.0, ax[0], ax[1], ax[2]]
    c = cross(u, v)
    q = [1 + d, c[0], c[1], c[2]]
    n = math.sqrt(sum(x * x for x in q))
    return [x / n for x in q]


def rotations(joints: dict) -> dict:
    """Swing rotation of each bone from its rest direction, keyed by the bone's child joint."""
    out = {}
    for p, c in BONES:
        if p in joints and c in joints and c in REST:
            d = sub(joints[c], joints[p])
            if norm(d) > 1e-9:
                out[c] = [round(x, 6) for x in shortest_arc(REST[c], d)]
    return out


def complete(joints: dict) -> dict:
    """Fill hips, spine and neck from the hips and shoulders when an estimator lacks them."""
    j = dict(joints)
    if "hips" not in j and "l_hip" in j and "r_hip" in j:
        j["hips"] = mid(j["l_hip"], j["r_hip"])
    if "neck" not in j and "l_shoulder" in j and "r_shoulder" in j:
        j["neck"] = mid(j["l_shoulder"], j["r_shoulder"])
    if "spine" not in j and "hips" in j and "neck" in j:
        j["spine"] = mid(j["hips"], j["neck"])
    return j


# -- multi-person tracking -------------------------------------------------------------------------------
class Tracker:
    """Keeps a persistent TrackID per performer by matching each detection's pelvis (hips) to the nearest track,
    closest pairs first. A track that misses `patience` frames ends; its id is never reused."""

    def __init__(self, max_dist: float = 0.6, patience: int = 15):
        self.max_dist, self.patience = max_dist, patience
        self.tracks: dict[int, dict] = {}
        self.next_id = 1

    def update(self, detections: list[dict]) -> dict[str, dict]:
        pairs = []
        for di, d in enumerate(detections):
            h = d["joints"].get("hips")
            for tid, t in self.tracks.items():
                if h is not None:
                    pairs.append((norm(sub(h, t["hips"])), di, tid))
        pairs.sort()
        used_d, used_t, out = set(), set(), {}
        for dist, di, tid in pairs:
            if dist > self.max_dist or di in used_d or tid in used_t:
                continue
            used_d.add(di), used_t.add(tid)
            out[str(tid)] = detections[di]
        for di, d in enumerate(detections):
            if di not in used_d and "hips" in d["joints"]:
                out[str(self.next_id)] = d
                self.next_id += 1
        for tid in list(self.tracks):
            if str(tid) not in out:
                self.tracks[tid]["missed"] += 1
                if self.tracks[tid]["missed"] > self.patience:
                    del self.tracks[tid]
        for tid, d in out.items():
            self.tracks[int(tid)] = {"hips": d["joints"]["hips"], "missed": 0}
        return out


def frame(index: int, t: float, actors: dict[str, dict]) -> dict:
    out = {}
    for tid, a in actors.items():
        j = complete(a["joints"])
        rec = {"joints": {k: [round(x, 5) for x in v] for k, v in j.items() if k in JOINTS},
               "rot": a.get("rot") or rotations(j)}
        for k in ("px", "conf"):
            if a.get(k):
                rec[k] = a[k]
        out[tid] = rec
    return {"frame": index, "t": round(t, 5), "actors": out}


def read_frames(path) -> list[dict]:
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def from_landmarks(points, mapping: dict, px=None, conf=None, flip_y: bool = False) -> dict:
    """One detection from an estimator's landmark list (each [x, y, z]) through MEDIAPIPE or BODY25."""
    j, p, c = {}, {}, {}
    for name, i in mapping.items():
        if i < len(points) and points[i] is not None:
            x, y, z = points[i][:3]
            j[name] = [float(x), -float(y) if flip_y else float(y), float(z)]
            if px is not None and i < len(px) and px[i] is not None:
                p[name] = [round(float(px[i][0]), 4), round(float(px[i][1]), 4)]
            if conf is not None and i < len(conf) and conf[i] is not None:
                c[name] = round(float(conf[i]), 3)
    return {"joints": complete(j), "px": complete2d(p), "conf": c}


def complete2d(px: dict) -> dict:
    p = dict(px)
    for a, b, m in (("l_hip", "r_hip", "hips"), ("l_shoulder", "r_shoulder", "neck")):
        if m not in p and a in p and b in p:
            p[m] = [round((p[a][0] + p[b][0]) / 2, 4), round((p[a][1] + p[b][1]) / 2, 4)]
    if "spine" not in p and "hips" in p and "neck" in p:
        p["spine"] = [round((p["hips"][0] + p["neck"][0]) / 2, 4), round((p["hips"][1] + p["neck"][1]) / 2, 4)]
    return p
