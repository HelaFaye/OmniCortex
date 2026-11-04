"""OmniCortex pose add-on for Blender 4 (docs/POSE.md). Install: download omnicortex_pose.zip (the pose add-on's page in Studio),
then Edit -> Preferences -> Add-ons -> Install from disk; the "OmniCortex" tab appears in the 3D view's sidebar.

  Listen        receive stream.py's JSON datagrams (--json 127.0.0.1:39570) and move one empty per joint per
                performer, coloured by stress (green -> red)
  Bake stress   for the selected mesh (skinned to an armature): read a frames file and the sieve's staging
                folder, compute each vertex's stress from its bone weights, and write the worst moment of the
                clip into a colour attribute "omnicortex_stress"; each bone also gets an animated custom
                property "oc_stress" over the frames
  Export FBX    the selection as FBX with vertex colours and custom properties, so the quality log travels with
                the asset into Maya, 3ds Max or MotionBuilder

FBX carries one set of vertex colours, not a colour per frame: the per-frame log is in the bones' animated
"oc_stress" property (exported as FBX custom properties); the vertex colours hold the clip's worst moment.
Untested in this repository's CI (no Blender there); its stress maths is pose_stress.py's, which is tested.
"""
bl_info = {"name": "OmniCortex pose", "author": "OmniCortex", "version": (0, 1, 0), "blender": (4, 0, 0),
           "location": "View3D > Sidebar > OmniCortex", "category": "Animation"}

import json
import socket
from pathlib import Path

import bpy

try:
    from . import pose_stress as PS
except ImportError:                       # run as a single script
    import pose_stress as PS

_SOCK = {"s": None}


def _stress_by_frame(staging: Path) -> dict:
    out = {}
    for d in ("pending", "approved"):
        for fp in (staging / d).glob("*.json") if (staging / d).is_dir() else []:
            it = json.loads(fp.read_text())
            s = out.setdefault((it["frame"], str(it["track"])), {})
            for f in it.get("failures") or []:
                s[f["joint"]] = 1.0
    return out


class OC_OT_listen(bpy.types.Operator):
    bl_idname = "omnicortex.listen"
    bl_label = "Listen"
    bl_description = "Receive OmniCortex pose frames over UDP"
    _timer = None

    def modal(self, context, event):
        if event.type == "ESC" or _SOCK["s"] is None:
            return self.cancel(context) or {"CANCELLED"}
        if event.type != "TIMER":
            return {"PASS_THROUGH"}
        last = None
        try:
            while True:
                last = _SOCK["s"].recv(65536)
        except BlockingIOError:
            pass
        if last:
            f = json.loads(last.decode())
            coll = bpy.data.collections.get("OmniCortex pose") or bpy.data.collections.new("OmniCortex pose")
            if coll.name not in context.scene.collection.children:
                context.scene.collection.children.link(coll)
            for tid, a in f.get("actors", {}).items():
                st = (f.get("stress") or {}).get(tid, {})
                for j, p in a["joints"].items():
                    name = f"oc_{tid}_{j}"
                    ob = bpy.data.objects.get(name)
                    if ob is None:
                        ob = bpy.data.objects.new(name, None)
                        ob.empty_display_type = "SPHERE"
                        ob.empty_display_size = 0.04
                        coll.objects.link(ob)
                    ob.location = (p[0], -p[2], p[1])          # y-up metres -> Blender's z-up
                    ob.color = PS.ramp(st.get(j, 0.0))
        return {"PASS_THROUGH"}

    def execute(self, context):
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.bind(("0.0.0.0", context.scene.oc_port))
        s.setblocking(False)
        _SOCK["s"] = s
        self._timer = context.window_manager.event_timer_add(1 / 30, window=context.window)
        context.window_manager.modal_handler_add(self)
        return {"RUNNING_MODAL"}

    def cancel(self, context):
        if self._timer:
            context.window_manager.event_timer_remove(self._timer)
        if _SOCK["s"]:
            _SOCK["s"].close()
            _SOCK["s"] = None


class OC_OT_bake(bpy.types.Operator):
    bl_idname = "omnicortex.bake"
    bl_label = "Bake stress"
    bl_description = "Write the sieve's joint stress into the selected mesh's vertex colours and its bones' oc_stress"

    def execute(self, context):
        ob = context.active_object
        sc = context.scene
        if ob is None or ob.type != "MESH":
            self.report({"ERROR"}, "select the skinned mesh")
            return {"CANCELLED"}
        per = _stress_by_frame(Path(bpy.path.abspath(sc.oc_staging)))
        frames = [json.loads(l)["frame"] for l in Path(bpy.path.abspath(sc.oc_frames)).read_text().splitlines() if l.strip()]
        g2j = PS.guess_map([g.name for g in ob.vertex_groups])
        worst = PS.clip_stress(per, sc.oc_track, frames)
        names = {g.index: g.name for g in ob.vertex_groups}
        attr = ob.data.color_attributes.get("omnicortex_stress") or \
            ob.data.color_attributes.new("omnicortex_stress", "BYTE_COLOR", "POINT")
        for v in ob.data.vertices:
            w = {names[g.group]: g.weight for g in v.groups if g.group in names}
            attr.data[v.index].color = PS.ramp(PS.vertex_stress(w, worst, g2j))
        arm = ob.find_armature()
        if arm is not None:
            j2b = {j: g for g, j in g2j.items()}
            for i, f in enumerate(frames):
                st = per.get((f, sc.oc_track), {})
                for j, b in j2b.items():
                    pb = arm.pose.bones.get(b)
                    if pb is not None:
                        pb["oc_stress"] = float(st.get(j, 0.0))
                        pb.keyframe_insert(data_path='["oc_stress"]', frame=sc.frame_start + i)
        self.report({"INFO"}, f"baked {len(frames)} frames; {sum(1 for s in worst.values() if s > 0)} joints flagged")
        return {"FINISHED"}


class OC_OT_export(bpy.types.Operator):
    bl_idname = "omnicortex.export_fbx"
    bl_label = "Export FBX"
    filepath: bpy.props.StringProperty(subtype="FILE_PATH", default="//pose_stress.fbx")

    def execute(self, context):
        bpy.ops.export_scene.fbx(filepath=bpy.path.abspath(self.filepath), use_selection=True, use_custom_props=True,
                                 colors_type="SRGB", bake_anim=True)
        return {"FINISHED"}

    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {"RUNNING_MODAL"}


class OC_PT_panel(bpy.types.Panel):
    bl_label = "OmniCortex pose"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "OmniCortex"

    def draw(self, context):
        sc, col = context.scene, self.layout.column()
        col.prop(sc, "oc_port")
        col.operator("omnicortex.listen")
        col.separator()
        col.prop(sc, "oc_frames")
        col.prop(sc, "oc_staging")
        col.prop(sc, "oc_track")
        col.operator("omnicortex.bake")
        col.operator("omnicortex.export_fbx")


CLASSES = (OC_OT_listen, OC_OT_bake, OC_OT_export, OC_PT_panel)


def register():
    bpy.types.Scene.oc_port = bpy.props.IntProperty(name="UDP port", default=39570)
    bpy.types.Scene.oc_frames = bpy.props.StringProperty(name="Frames", subtype="FILE_PATH")
    bpy.types.Scene.oc_staging = bpy.props.StringProperty(name="Staging", subtype="DIR_PATH")
    bpy.types.Scene.oc_track = bpy.props.StringProperty(name="Performer", default="1")
    for c in CLASSES:
        bpy.utils.register_class(c)


def unregister():
    for c in reversed(CLASSES):
        bpy.utils.unregister_class(c)
    for p in ("oc_port", "oc_frames", "oc_staging", "oc_track"):
        delattr(bpy.types.Scene, p)
