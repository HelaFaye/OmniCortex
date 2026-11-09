"""The pose add-on (plugins/pose): skeletons and tracking, the deficit sieve, staging and export, the orchestrator,
the VMC / JSON stream, OpenPose import, and the engine clients' shared stress maths."""
import json
import math
import shutil
import socket
import struct
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
POSE = ROOT / "plugins" / "pose"
sys.path.insert(0, str(POSE))
sys.path.insert(0, str(POSE / "engine" / "blender" / "omnicortex_pose"))
import orchestrate as O  # noqa: E402
import pose_core as P  # noqa: E402
import pose_stress as PS  # noqa: E402
import sieve as SV  # noqa: E402
import staging as ST  # noqa: E402
import stream as SM  # noqa: E402


@pytest.fixture(autouse=True, scope="module")
def _isolate_modules():
    """These add-ons' scripts have generic names (stream, plan, check, ...): forget them and their folders after
    this file, so later tests import OmniCortex's own modules of the same names."""
    paths = list(sys.path)
    yield
    for name, mod in list(sys.modules.items()):
        f = getattr(mod, "__file__", "") or ""
        if f and "plugins" in Path(f).parts:
            del sys.modules[name]
    sys.path[:] = paths


def standing(dx=0.0):
    return {"l_hip": [0.1 + dx, 0.9, 0], "r_hip": [-0.1 + dx, 0.9, 0], "l_knee": [0.1 + dx, 0.5, 0.04],
            "r_knee": [-0.1 + dx, 0.5, 0.04], "l_ankle": [0.1 + dx, 0.08, 0], "r_ankle": [-0.1 + dx, 0.08, 0],
            "l_toe": [0.1 + dx, 0.0, 0.12], "r_toe": [-0.1 + dx, 0.0, 0.12], "l_shoulder": [0.2 + dx, 1.4, 0],
            "r_shoulder": [-0.2 + dx, 1.4, 0], "l_elbow": [0.45 + dx, 1.4, 0], "r_elbow": [-0.45 + dx, 1.4, 0],
            "l_wrist": [0.7 + dx, 1.4, 0], "r_wrist": [-0.7 + dx, 1.4, 0], "head": [dx, 1.65, 0]}


def clip(n=30, edit=None):
    frames = []
    for i in range(n):
        j = standing()
        if edit:
            edit(i, j)
        frames.append(P.frame(i, i / 30, {"1": {"joints": j, "px": {k: [0.5 + v[0] / 4, 0.9 - v[1] / 2] for k, v in j.items()}}}))
    return frames


def cfg(**kw):
    return SV.args(["--frames", "x", "--out", "y", *sum((["--" + k.replace("_", "-"), str(v)] for k, v in kw.items()), [])])


def test_rotations_and_tracking():
    q = P.shortest_arc([0, 1, 0], [1, 0, 0])
    w, x, y, z = q
    assert math.isclose(w, math.cos(math.pi / 4), abs_tol=1e-6) and math.isclose(z, -math.sin(math.pi / 4), abs_tol=1e-6)
    assert P.shortest_arc([0, 1, 0], [0, -1, 0])[0] == 0.0                     # opposite: 180 degrees
    rest = {"hips": [0, 1, 0], "spine": [0, 1.2, 0], "l_hip": [0.1, 1, 0], "l_knee": [0.1, 0.5, 0]}
    rot = P.rotations(rest)
    assert all(abs(rot[k][0] - 1) < 1e-6 for k in ("spine", "l_hip", "l_knee"))  # the rest pose: no rotation
    tr = P.Tracker(max_dist=0.5)
    a, b = {"joints": {"hips": [0, 1, 0]}}, {"joints": {"hips": [2, 1, 0]}}
    first = tr.update([a, b])
    ids = {k: v["joints"]["hips"][0] for k, v in first.items()}
    swapped = tr.update([{"joints": {"hips": [2.1, 1, 0]}}, {"joints": {"hips": [0.1, 1, 0]}}])
    assert {k: round(v["joints"]["hips"][0]) for k, v in swapped.items()} == {k: round(x) for k, x in ids.items()}
    new = tr.update([{"joints": {"hips": [9, 1, 0]}}])                          # far away: a new performer
    assert set(new) == {"3"}
    f = P.frame(0, 0.0, {"1": {"joints": {"l_hip": [1, 1, 0], "r_hip": [-1, 1, 0], "l_shoulder": [1, 2, 0], "r_shoulder": [-1, 2, 0]}}})
    j = f["actors"]["1"]["joints"]
    assert j["hips"] == [0, 1, 0] and j["neck"] == [0, 2, 0] and j["spine"] == [0, 1.5, 0]


def test_sieve_finds_each_failure_and_corrects_it(tmp_path):
    def edit(i, j):
        if i == 10:
            j["l_knee"] = [0.1, 0.5, -0.12]                     # bent backwards
        if i == 15:
            j["r_wrist"] = [3.0, 1.4, 0]                        # teleported hand (also a stretched forearm)
        if i == 20:
            for k in ("l_ankle", "l_toe"):
                j[k] = [j[k][0], j[k][1] - 0.2, j[k][2]]         # through the floor
        if 22 <= i <= 25:
            for k in ("r_ankle", "r_toe"):
                j[k] = [j[k][0], j[k][1], j[k][2] + 0.05 * (i - 22)]   # sliding while planted
    frames = clip(edit=edit)
    an = SV.detect(frames, cfg())
    kinds = {(a["error"], a["joint"], a["frame"]) for a in an}
    assert ("hyperextension", "l_knee", 10) in kinds
    assert any(e == "velocity" and j == "r_wrist" and f == 15 for e, j, f in kinds)
    assert any(e == "bone_length" and j == "r_wrist" for e, j, _ in kinds)
    assert any(e == "floor_clip" and j in ("l_ankle", "l_toe") and f == 20 for e, j, f in kinds)
    assert any(e == "foot_slip" and j in ("r_ankle", "r_toe") for e, j, _ in kinds)
    assert not any(f in (0, 1, 2, 5) for _, _, f in kinds)                       # a clean frame stays clean
    knee = next(a for a in an if a["error"] == "hyperextension")
    fixed, how = SV.correct(frames, knee, {(a["index"], a["track"], a["joint"]) for a in an}, 8)
    assert "interpolated" in how and fixed["l_knee"][2] > 0                          # back in front of the leg
    n = SV.stage(frames, an, tmp_path, None, 8, "pose-model")
    assert n >= 3 and SV.stage(frames, an, tmp_path, None, 8, "pose-model") == 0     # re-running adds nothing
    it = json.loads(next((tmp_path / "pending").glob("*.json")).read_text())
    assert it["chosen"] != it["rejected"] and it["model"] == "pose-model" and it["failures"]


def test_staging_review_and_export(tmp_path):
    frames = clip(edit=lambda i, j: j.update({"l_knee": [0.1, 0.5, -0.12]}) if i == 10 else None)
    img = tmp_path / "img"
    img.mkdir()
    from PIL import Image
    Image.new("RGB", (32, 32), "white").save(img / "000010.png")
    an = SV.detect(frames, cfg())
    SV.stage(frames, an, tmp_path / "st", img, 8)
    run = lambda c: json.loads(subprocess.run([sys.executable, "check.py"], cwd=POSE, capture_output=True, text=True,  # noqa: E731
                                              input=json.dumps({"text": json.dumps({"dir": str(tmp_path / "st"), **c})})).stdout)
    lst = run({"op": "list"})
    assert lst["counts"]["pending"] == 1
    iid = lst["items"][0]["id"]
    got = run({"op": "get", "id": iid})
    assert got["item"]["image_data"].startswith("data:image/png;base64,")
    assert run({"op": "approve", "id": iid})["counts"] == {"pending": 0, "approved": 1, "discarded": 0}
    assert run({"op": "get", "id": "../etc"})["flag"]
    ex = ST.export(tmp_path / "st", tmp_path / "data")
    assert ex["vision"] and ex["files"][0] == "dpo_vision.jsonl"
    row = json.loads((tmp_path / "data" / "dpo_vision.jsonl").read_text().splitlines()[0])
    assert (tmp_path / "data" / row["images"][0]).exists() and row["prompt"][0]["content"][0] == {"type": "image"}
    assert json.loads(row["rejected"][0]["content"][0]["text"])["joints"]["l_knee"][2] < 0


def test_orchestrator_pieces(tmp_path):
    good = [{"step": i, "loss": 0.7 - 0.01 * i, "margin": 0.1 * i} for i in range(10)]
    exploded = [{"step": 0, "loss": 0.7, "margin": 0}, {"step": 1, "loss": 9.0, "margin": 2}]
    nan = [{"step": 0, "loss": float("nan"), "margin": 0.1}]
    assert O.judge(good, 3)["ok"] and not O.judge(exploded, 3)["ok"] and not O.judge(nan, 3)["ok"]
    res = [{"lr": 1e-5, "alpha": 16, "judge": O.judge(good, 3)},
           {"lr": 5e-6, "alpha": 32, "judge": O.judge(good, 3)},
           {"lr": 2e-5, "alpha": 16, "judge": O.judge(exploded, 3)}]
    assert O.pick(res)["lr"] == 5e-6                                             # a tie goes to the lower rate
    run = tmp_path / "run"
    run.mkdir()
    (run / "neuronscope-finetune.json").write_text(json.dumps({"log": {"dpo": [{"step": 1, "loss": 0.6, "rewards/margins": 0.2}]}}))
    assert O.dpo_curve(run) == [{"step": 1, "loss": 0.6, "margin": 0.2}]
    cfgp = tmp_path / "stage2.yaml"
    cfgp.write_text("other: 1\n")
    O.promote(cfgp, {"dpo_lr": 5e-6})
    import yaml
    assert yaml.safe_load(cfgp.read_text()) == {"other": 1, "dpo_lr": 5e-6} and (tmp_path / "stage2.yaml.prev").exists()
    target, new = tmp_path / "model.gguf", tmp_path / "new.gguf"
    target.write_bytes(b"old")
    new.write_bytes(b"new")
    bak = O.install(new, target)
    assert target.read_bytes() == b"new" and bak.read_bytes() == b"old"
    frames = clip(edit=lambda i, j: j.update({"l_knee": [0.1, 0.5, -0.12]}) if i == 10 else None)
    SV.stage(frames, SV.detect(frames, cfg()), tmp_path / "st", None, 8)
    for it in ST.items(tmp_path / "st", "pending"):
        ST.move(tmp_path / "st", it["id"], "approved")
    assert O.main(["--staging", str(tmp_path / "st"), "--work", str(tmp_path / "w"), "--threshold", "1"]) == 0
    rec = json.loads((tmp_path / "w" / "cycle-001" / "cycle.json").read_text())
    assert rec["data"]["pairs"] == 1 and not (tmp_path / "st" / "PAUSE").exists()
    O.main(["--staging", str(tmp_path / "st"), "--work", str(tmp_path / "w"), "--threshold", "1"])
    assert not (tmp_path / "w" / "cycle-002").exists()                           # the same pairs are not counted twice


def _osc_parse(b: bytes):
    def s(off):
        end = b.index(b"\0", off)
        return b[off:end].decode(), (end + 4) & ~3
    addr, off = s(0)
    tags, off = s(off)
    vals = []
    for t in tags[1:]:
        if t == "s":
            v, off = s(off)
        elif t == "i":
            v, off = struct.unpack(">i", b[off:off + 4])[0], off + 4
        else:
            v, off = struct.unpack(">f", b[off:off + 4])[0], off + 4
        vals.append(v)
    return addr, vals


def test_stream_vmc_and_json(tmp_path):
    a, v = _osc_parse(SM.osc("/VMC/Ext/Bone/Pos", "Hips", 1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0))
    assert a == "/VMC/Ext/Bone/Pos" and v[0] == "Hips" and v[1:4] == [1.0, 2.0, 3.0]
    assert SM.bundle([b"abcd"]).startswith(b"#bundle\0")
    rest = {k: list(v) for k, v in standing().items()}
    rest["l_knee"] = [0.1, 0.5, 0.0]
    rest["l_ankle"] = [0.1, 0.08, 0.0]
    actor = P.frame(0, 0, {"1": {"joints": rest}})["actors"]["1"]
    msgs = {(_osc_parse(m)[1][0] if _osc_parse(m)[1] else None): _osc_parse(m)[1] for m in SM.vmc_messages(actor, 0.0)
            if _osc_parse(m)[0] == "/VMC/Ext/Bone/Pos"}
    knee = msgs["LeftLowerLeg"]                                                  # straight leg: no local bend
    assert abs(knee[-1] - 1) < 1e-4
    frames = clip(edit=lambda i, j: j.update({"l_knee": [0.1, 0.5, -0.12]}) if i == 3 else None)
    fp = tmp_path / "f.jsonl"
    fp.write_text("".join(json.dumps(f) + "\n" for f in frames[:6]))
    SV.stage(frames, SV.detect(frames, cfg()), tmp_path / "st", None, 8)
    rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    rx.bind(("127.0.0.1", 0))
    rx.settimeout(5)
    SM.main(["--frames", str(fp), "--json", f"127.0.0.1:{rx.getsockname()[1]}", "--staging", str(tmp_path / "st"), "--speed", "1000"])
    got = [json.loads(rx.recv(65536)) for _ in range(6)]
    rx.close()
    assert got[3]["stress"]["1"]["l_knee"] == 1.0 and got[3]["stress"]["1"]["l_hip"] == 0.5 and got[0]["stress"]["1"] == {}


def test_openpose_import(tmp_path):
    import extract
    d = tmp_path / "op"
    d.mkdir()
    for i in range(3):
        k2 = [0.0] * 75
        pts = {1: (960, 300), 2: (900, 300), 5: (1020, 300), 8: (960, 600), 9: (920, 600), 12: (1000, 600),
               10: (920, 800), 13: (1000, 800), 11: (920, 1000), 14: (1000, 1000)}
        for idx, (x, y) in pts.items():
            k2[idx * 3:idx * 3 + 3] = [x + 10 * i, y, 0.9]
        (d / f"clip_{i:012d}_keypoints.json").write_text(json.dumps({"people": [{"pose_keypoints_2d": k2}]}))
    out = tmp_path / "frames.jsonl"
    assert extract.main(["--openpose", str(d), "--out", str(out)]) == 0
    fr = P.read_frames(out)
    assert len(fr) == 3 and set(fr[0]["actors"]) == {"1"} and set(fr[2]["actors"]) == {"1"}
    j = fr[0]["actors"]["1"]["joints"]
    assert j["neck"][1] > j["hips"][1] > j["l_knee"][1]                          # image y flipped to y up
    assert fr[0]["actors"]["1"]["px"]["hips"] == [0.5, 0.5556]


def test_engine_stress_maths_and_downloads(tmp_path):
    assert PS.ramp(0) == (0.0, 1.0, 0.1, 1.0) and PS.ramp(1) == (1.0, 0.0, 0.1, 1.0) and PS.ramp(0.5)[:2] == (1.0, 1.0)
    g2j = PS.guess_map(["thigh.L", "shin.L", "mixamorig:Hips", "spine.001"])
    assert g2j == {"thigh.L": "l_hip", "shin.L": "l_knee", "mixamorig:Hips": "hips", "spine.001": "spine"}
    assert PS.vertex_stress({"shin.L": 0.75, "thigh.L": 0.25}, {"l_knee": 1.0}, g2j) == 0.75
    assert PS.vertex_stress({}, {"l_knee": 1.0}) == 0.0
    assert PS.clip_stress({(1, "1"): {"l_knee": 0.4}, (2, "1"): {"l_knee": 1.0}}, "1", [1, 2]) == {"l_knee": 1.0}
    sys.path.insert(0, str(ROOT / "viz"))
    import plugins as PL
    pm = PL.Plugins([ROOT / "plugins"], tmp_path / "s.json")
    pm.set_enabled("pose", True)
    import io
    import zipfile
    names = zipfile.ZipFile(io.BytesIO(pm.download("pose", "omnicortex_pose.zip"))).namelist()
    assert "omnicortex_pose/__init__.py" in names and "omnicortex_pose/pose_stress.py" in names
    gd = zipfile.ZipFile(io.BytesIO(pm.download("pose", "omnicortex-pose-godot.zip"))).namelist()
    assert {"omnicortex-pose-godot/" + f for f in ("pose_receiver.gd", "stress_skin.gdshader", "xray_outline.gdshader")} <= set(gd)
    node = shutil.which("node")
    if node:
        html = (POSE / "page" / "index.html").read_text()
        js = html.split("<script>")[1].split("</script>")[0]
        (tmp_path / "p.js").write_text(js)
        subprocess.run([node, "--check", str(tmp_path / "p.js")], check=True)
