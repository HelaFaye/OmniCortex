extends Node3D
## OmniCortex pose receiver for Godot 4 (docs/POSE.md).
##
## Listens for stream.py's JSON datagrams (--json HOST:PORT) and draws each performer as a stick figure whose
## joints and bones glow from green (stable) through yellow to red (a joint the deficit sieve flagged), using the
## per-joint "stress" the stream sends with --staging. Each figure also gets an x-ray pass, so a failing joint stays
## visible behind walls and set pieces.
##
## For a rigged character instead of the stick figure: give its MeshInstance3D the material in stress_skin.gdshader
## and set `skin_targets` to {actor_id: mesh_path}; the joint stress is passed to the shader per bone.
##
## Untested in this repository's CI (no Godot there): the maths it relies on is pose_core.py's, which is tested.

@export var port := 39570
@export var joint_radius := 0.035
@export var xray := true
@export var skin_targets := {}          # "1": NodePath to a skinned MeshInstance3D
@export var bone_names := {}            # joint name -> skeleton bone name, for skin_targets

const BONES := [["hips","spine"],["spine","neck"],["neck","head"],["neck","l_shoulder"],["l_shoulder","l_elbow"],
	["l_elbow","l_wrist"],["neck","r_shoulder"],["r_shoulder","r_elbow"],["r_elbow","r_wrist"],["hips","l_hip"],
	["l_hip","l_knee"],["l_knee","l_ankle"],["l_ankle","l_toe"],["hips","r_hip"],["r_hip","r_knee"],
	["r_knee","r_ankle"],["r_ankle","r_toe"]]

var _udp := PacketPeerUDP.new()
var _actors := {}                       # id -> {"root": Node3D, "joints": {}, "bones": []}

func _ready() -> void:
	var err := _udp.bind(port, "0.0.0.0")
	if err != OK:
		push_error("OmniCortex pose: cannot listen on UDP %d (%s)" % [port, error_string(err)])

static func ramp(s: float) -> Color:
	## 0 green, 0.5 yellow, 1 red: the same ramp as pose_stress.py
	s = clampf(s, 0.0, 1.0)
	return Color(minf(1.0, 2.0 * s), minf(1.0, 2.0 * (1.0 - s)), 0.1)

func _material(c: Color, on_top: bool) -> StandardMaterial3D:
	var m := StandardMaterial3D.new()
	m.albedo_color = c
	m.emission_enabled = true
	m.emission = c
	m.emission_energy_multiplier = 1.5
	if on_top:
		m.no_depth_test = true
		m.transparency = BaseMaterial3D.TRANSPARENCY_ALPHA
		m.albedo_color.a = 0.35
		m.render_priority = 10
	return m

func _actor(id: String) -> Dictionary:
	if _actors.has(id):
		return _actors[id]
	var root := Node3D.new()
	root.name = "Actor_" + id
	add_child(root)
	_actors[id] = {"root": root, "joints": {}, "bones": []}
	for b in BONES:
		var mi := MeshInstance3D.new()
		var cyl := CylinderMesh.new()
		cyl.top_radius = joint_radius * 0.5
		cyl.bottom_radius = joint_radius * 0.5
		cyl.height = 1.0
		mi.mesh = cyl
		root.add_child(mi)
		_actors[id]["bones"].append(mi)
	return _actors[id]

func _joint(a: Dictionary, name: String) -> MeshInstance3D:
	if a["joints"].has(name):
		return a["joints"][name]
	var mi := MeshInstance3D.new()
	var sph := SphereMesh.new()
	sph.radius = joint_radius
	sph.height = joint_radius * 2.0
	mi.mesh = sph
	a["root"].add_child(mi)
	a["joints"][name] = mi
	return mi

func _paint(mi: MeshInstance3D, s: float) -> void:
	var m := _material(ramp(s), false)
	if xray:
		m.next_pass = _material(ramp(s), true)
	mi.material_override = m

func _process(_dt: float) -> void:
	var last = null
	while _udp.get_available_packet_count() > 0:
		last = _udp.get_packet()               # only the newest frame matters
	if last == null:
		return
	var f = JSON.parse_string(last.get_string_from_utf8())
	if typeof(f) != TYPE_DICTIONARY:
		return
	var stress: Dictionary = f.get("stress", {})
	for id in f.get("actors", {}):
		var a := _actor(str(id))
		var joints: Dictionary = f["actors"][id].get("joints", {})
		var st: Dictionary = stress.get(id, {})
		for jn in joints:
			var p = joints[jn]
			var mi := _joint(a, jn)
			mi.position = Vector3(p[0], p[1], p[2])
			_paint(mi, float(st.get(jn, 0.0)))
		for i in BONES.size():
			var b = BONES[i]
			var mi: MeshInstance3D = a["bones"][i]
			if not (joints.has(b[0]) and joints.has(b[1])):
				mi.visible = false
				continue
			var p0 := Vector3(joints[b[0]][0], joints[b[0]][1], joints[b[0]][2])
			var p1 := Vector3(joints[b[1]][0], joints[b[1]][1], joints[b[1]][2])
			var d := p1 - p0
			mi.visible = d.length() > 0.001
			if mi.visible:
				# a cylinder's axis is Y: face the far joint (-Z), then tip -Z onto Y, then stretch to the bone
				var up := Vector3.UP if absf(d.normalized().dot(Vector3.UP)) < 0.99 else Vector3.FORWARD
				mi.look_at_from_position((p0 + p1) * 0.5, p1, up)
				mi.rotate_object_local(Vector3.RIGHT, PI / 2)
				mi.scale = Vector3(1, d.length(), 1)
				_paint(mi, maxf(float(st.get(b[0], 0.0)), float(st.get(b[1], 0.0))))
		_skin(str(id), st)

func _skin(id: String, st: Dictionary) -> void:
	if not skin_targets.has(id):
		return
	var mesh := get_node_or_null(skin_targets[id]) as MeshInstance3D
	if mesh == null or mesh.skeleton == NodePath():
		return
	var skel := mesh.get_node(mesh.skeleton) as Skeleton3D
	var arr := PackedFloat32Array()
	arr.resize(128)
	for jn in st:
		var bn: String = bone_names.get(jn, jn)
		var idx := skel.find_bone(bn)
		if idx >= 0 and idx < 128:
			arr[idx] = float(st[jn])
	var mat := mesh.get_active_material(0) as ShaderMaterial
	if mat:
		mat.set_shader_parameter("stress", arr)
