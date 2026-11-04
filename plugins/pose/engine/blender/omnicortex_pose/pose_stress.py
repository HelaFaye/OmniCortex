"""Joint stress -> per-vertex stress and colour, shared by the Blender add-on and the tests (no bpy needed)."""


def ramp(s: float) -> tuple[float, float, float, float]:
    """0 green, 0.5 yellow, 1 red (the Godot shader's ramp), as linear RGBA."""
    s = max(0.0, min(1.0, float(s)))
    return (min(1.0, 2 * s), min(1.0, 2 * (1 - s)), 0.1, 1.0)


def vertex_stress(weights: dict[str, float], joint_stress: dict[str, float], group_to_joint: dict[str, str] | None = None) -> float:
    """A vertex's stress: its vertex groups' joint stress blended by the group weights (normalised)."""
    g2j = group_to_joint or {}
    tot = sum(w for w in weights.values() if w > 0)
    if tot <= 0:
        return 0.0
    return sum(w * joint_stress.get(g2j.get(g, g), 0.0) for g, w in weights.items() if w > 0) / tot


def clip_stress(per_frame: dict, track: str, frames: list[int]) -> dict[str, float]:
    """The highest stress each joint reached over the frames (the baked log keeps the worst moment)."""
    out: dict[str, float] = {}
    for f in frames:
        for j, s in (per_frame.get((f, track)) or {}).items():
            out[j] = max(out.get(j, 0.0), s)
    return out


# Rigify / Mixamo / generic bone names for pose_core's joints, tried in order when the user gives no map
BONE_GUESS = {
    "hips": ["hips", "pelvis", "mixamorig:Hips", "torso"], "spine": ["spine", "spine.001", "mixamorig:Spine"],
    "neck": ["neck", "mixamorig:Neck"], "head": ["head", "mixamorig:Head"],
    "l_shoulder": ["upper_arm.L", "shoulder.L", "mixamorig:LeftArm", "LeftUpperArm"],
    "l_elbow": ["forearm.L", "mixamorig:LeftForeArm", "LeftLowerArm"], "l_wrist": ["hand.L", "mixamorig:LeftHand", "LeftHand"],
    "r_shoulder": ["upper_arm.R", "shoulder.R", "mixamorig:RightArm", "RightUpperArm"],
    "r_elbow": ["forearm.R", "mixamorig:RightForeArm", "RightLowerArm"], "r_wrist": ["hand.R", "mixamorig:RightHand", "RightHand"],
    "l_hip": ["thigh.L", "mixamorig:LeftUpLeg", "LeftUpperLeg"], "l_knee": ["shin.L", "mixamorig:LeftLeg", "LeftLowerLeg"],
    "l_ankle": ["foot.L", "mixamorig:LeftFoot", "LeftFoot"], "l_toe": ["toe.L", "mixamorig:LeftToeBase", "LeftToes"],
    "r_hip": ["thigh.R", "mixamorig:RightUpLeg", "RightUpperLeg"], "r_knee": ["shin.R", "mixamorig:RightLeg", "RightLowerLeg"],
    "r_ankle": ["foot.R", "mixamorig:RightFoot", "RightFoot"], "r_toe": ["toe.R", "mixamorig:RightToeBase", "RightToes"],
}


def guess_map(group_names) -> dict[str, str]:
    """vertex group (bone) name -> joint, from the common rig naming schemes."""
    low = {g.lower(): g for g in group_names}
    out = {}
    for joint, names in BONE_GUESS.items():
        for n in names:
            if n.lower() in low:
                out[low[n.lower()]] = joint
                break
    return out
