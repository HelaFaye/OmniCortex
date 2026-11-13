"""
Stream frame payloads to engines without blocking whatever produces them (docs/POSE.md).

    --vmc HOST:PORT     the VMC protocol (OSC over UDP; VSeeFace, VMC4B for Blender, EVMC4U for Unity and other
                        receivers): one performer per port, the first on PORT, the next on PORT+1, ...
                        Bones are Unity humanoid names, local rotations, Unity's left-handed axes
    --json HOST:PORT    one JSON datagram per frame: the payload itself plus "stress" per joint (0..1) when
                        --staging is given. The Godot and Blender clients in engine/ read this

It replays --frames at its own timing (--loop to repeat), or with --follow keeps reading new lines as
extract.py writes them. With --staging it holds while <staging>/PAUSE exists (the orchestrator is
swapping the model). UDP is fire-and-forget: a slow or absent receiver never stalls the stream.

    python stream.py --frames runs/pose/frames.jsonl --vmc 127.0.0.1:39539 --json 127.0.0.1:39570 --loop
"""
from __future__ import annotations

import argparse
import json
import socket
import struct
import time
from pathlib import Path

import pose_core as P

# rotation key (the bone's child joint) -> the Unity humanoid bone that rotation belongs to
VMC_BONES = {"spine": "Hips", "neck": "Spine", "head": "Neck",
             "l_shoulder": "LeftShoulder", "l_elbow": "LeftUpperArm", "l_wrist": "LeftLowerArm",
             "r_shoulder": "RightShoulder", "r_elbow": "RightUpperArm", "r_wrist": "RightLowerArm",
             "l_knee": "LeftUpperLeg", "l_ankle": "LeftLowerLeg", "l_toe": "LeftFoot",
             "r_knee": "RightUpperLeg", "r_ankle": "RightLowerLeg", "r_toe": "RightFoot"}
PARENT_ROT = {"neck": "spine", "head": "neck", "l_shoulder": "neck", "r_shoulder": "neck",
              "l_elbow": "l_shoulder", "l_wrist": "l_elbow", "r_elbow": "r_shoulder", "r_wrist": "r_elbow",
              "l_knee": "spine", "r_knee": "spine", "l_ankle": "l_knee", "r_ankle": "r_knee",
              "l_toe": "l_ankle", "r_toe": "r_ankle"}


# -- OSC ------------------------------------------------------------------------------------------------
def _pad(b: bytes) -> bytes:
    return b + b"\0" * (4 - len(b) % 4)


def osc(address: str, *argv) -> bytes:
    tags, data = ",", b""
    for x in argv:
        if isinstance(x, str):
            tags += "s"
            data += _pad(x.encode())
        elif isinstance(x, int):
            tags += "i"
            data += struct.pack(">i", x)
        else:
            tags += "f"
            data += struct.pack(">f", float(x))
    return _pad(address.encode()) + _pad(tags.encode()) + data


def bundle(msgs: list[bytes]) -> bytes:
    out = _pad(b"#bundle") + struct.pack(">Q", 1)              # 1 = "immediately"
    for m in msgs:
        out += struct.pack(">i", len(m)) + m
    return out


# -- quaternions ----------------------------------------------------------------------------------------
def qmul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return [aw * bw - ax * bx - ay * by - az * bz, aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx, aw * bz + ax * by - ay * bx + az * bw]


def qinv(q):
    return [q[0], -q[1], -q[2], -q[3]]


def to_unity(v):
    """Right-handed (x = the performer's left) -> Unity's left-handed axes: mirror x."""
    return [-v[0], v[1], v[2]]


def q_to_unity(q):
    return [q[0], q[1], -q[2], -q[3]]


def vmc_messages(actor: dict, t: float) -> list[bytes]:
    j, rot = actor["joints"], actor.get("rot") or P.rotations(actor["joints"])
    msgs = [osc("/VMC/Ext/OK", 1), osc("/VMC/Ext/T", float(t))]
    if "hips" in j:
        hp = to_unity(j["hips"])
        msgs.append(osc("/VMC/Ext/Root/Pos", "root", 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0))
        hq = q_to_unity(rot.get("spine", [1, 0, 0, 0]))
        msgs.append(osc("/VMC/Ext/Bone/Pos", "Hips", *hp, hq[1], hq[2], hq[3], hq[0]))
    for key, bone in VMC_BONES.items():
        if key == "spine" or key not in rot:
            continue
        par = rot.get(PARENT_ROT.get(key, ""), [1, 0, 0, 0])
        q = q_to_unity(qmul(qinv(par), rot[key]))                # local to the parent bone
        msgs.append(osc("/VMC/Ext/Bone/Pos", bone, 0.0, 0.0, 0.0, q[1], q[2], q[3], q[0]))
    return msgs


def stress_by_frame(staging: Path) -> dict:
    """{(frame, track): {joint: 0..1}} from staged items: 1 at a failing joint, 0.5 at its neighbours."""
    out = {}
    for d in ("pending", "approved"):
        for fp in (staging / d).glob("*.json") if (staging / d).is_dir() else []:
            it = json.loads(fp.read_text())
            s = out.setdefault((it["frame"], str(it["track"])), {})
            for f in it.get("failures") or []:
                j = f["joint"]
                s[j] = 1.0
                for k in (P.PARENT.get(j), *[c for c, p in P.PARENT.items() if p == j]):
                    if k:
                        s[k] = max(s.get(k, 0.0), 0.5)
    return out


def addr(s: str):
    host, port = s.rsplit(":", 1)
    return host, int(port)


def frames_source(path: Path, follow: bool, loop: bool):
    while True:
        with open(path, encoding="utf-8") as f:
            while True:
                line = f.readline()
                if line:
                    if line.strip():
                        yield json.loads(line)
                    continue
                if not follow:
                    break
                time.sleep(0.02)
        if not loop:
            return


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames", required=True)
    ap.add_argument("--vmc", help="HOST:PORT")
    ap.add_argument("--json", help="HOST:PORT")
    ap.add_argument("--staging", help="the sieve's folder, to send per-joint stress")
    ap.add_argument("--follow", action="store_true", help="keep reading as the file grows")
    ap.add_argument("--loop", action="store_true")
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--max-frames", type=int, default=0, help="stop after this many (0: never)")
    a = ap.parse_args(argv)
    if not (a.vmc or a.json):
        raise SystemExit("give --vmc and/or --json")
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    vmc, js = (addr(a.vmc) if a.vmc else None), (addr(a.json) if a.json else None)
    stress = stress_by_frame(Path(a.staging)) if a.staging else {}
    slots: dict[str, int] = {}
    t0, f0, sent = None, None, 0
    pause = Path(a.staging) / "PAUSE" if a.staging else None
    for f in frames_source(Path(a.frames), a.follow, a.loop):
        while pause is not None and pause.exists():       # the orchestrator is swapping the model
            time.sleep(0.25)
            t0 = None
        if t0 is None or f["t"] < (f0 or 0):
            t0, f0 = time.monotonic(), f["t"]
        if not a.follow:
            wait = (f["t"] - f0) / a.speed - (time.monotonic() - t0)
            if wait > 0:
                time.sleep(wait)
        if vmc:
            for tid, actor in f["actors"].items():
                slot = slots.setdefault(tid, len(slots))
                try:
                    sock.sendto(bundle(vmc_messages(actor, f["t"])), (vmc[0], vmc[1] + slot))
                except OSError:
                    pass
        if js:
            out = dict(f)
            if stress:
                out["stress"] = {tid: stress.get((f["frame"], tid), {}) for tid in f["actors"]}
            try:
                sock.sendto(json.dumps(out, separators=(",", ":")).encode()[:65000], js)
            except OSError:
                pass
        sent += 1
        if a.max_frames and sent >= a.max_frames:
            break
    print(f"sent {sent} frames")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
