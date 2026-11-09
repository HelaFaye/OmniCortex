extends Node3D
## OmniCortex Godot client.
##
## Fetches the same api/meta, api/theme and api/trace the three.js client
## uses. All thresholding and flagging happened in Python; this only draws, so
## the frontends cannot drift on what counts as a flagged token or cell.
##
##   OC_API=http://127.0.0.1:7880 godot --path plugins/godot-client/client
##   OC_FRAME=146 OC_PAUSED=1 ...    # open on one token, paused (for screenshots)
##   OC_API=https://studio:7870/viz/<id> OC_TOKEN=... godot --path plugins/godot-client/client   # a reply checked in Studio
##
## MultiMesh keeps the whole field in one draw call, and glow runs at half
## resolution, because the GPU is usually also running the model.
##
## The full client (app.gd) embeds this scene and calls open() with a trace or
## a neuron-review view from Studio; run on its own, it reads OC_API.
##
## Layout: x = neuron column (ordered by classifier weight when the trace has a
## classifier, so the H-neurons form a band at the left), y = layer. Flagged
## H-neurons are drawn as rings in a second MultiMesh on top of the field.
##
## Everything 2D sits on the UI CanvasLayer, so it stays put on screen however
## the camera moves: the risk strip, the emotion graph with its jewel, the
## navigation gizmo and the search line (widgets/, docs/WIDGETS.md).
## Camera: drag to orbit, Shift + drag or middle-drag to pan, wheel to zoom;
## numpad 1/3/7 front/right/top (Ctrl for the opposite side), 5 perspective or
## orthographic, Home or numpad . to frame everything, S to sway.

const Jewel := preload("res://widgets/jewel.gd")
const EmotionGraph := preload("res://widgets/emotion_graph.gd")
const NavGizmo := preload("res://widgets/nav_gizmo.gd")
const ChatBox := preload("res://widgets/chat_box.gd")
const EMO_PALETTE := ["#FF4D1A", "#3D8BFF", "#1D9E75", "#E7B65C", "#B07CFF", "#c9c9d2", "#5DCAA5", "#F28FAD"]

const DEFAULT_API := "http://127.0.0.1:7880"
# Shared with the three.js client (viz/bloom.py); tests/test_bloom_demo.py keeps them equal.
const SIZE_IDLE := 1.1
const SIZE_ACTIVE := 2.4
const SIZE_FLAG := 7.0
const FPS := 8.0

var api: String
var token: String
var meta: Dictionary
var theme: Dictionary
var n_frames := 0
var n_cells := 0
var layer_of := PackedInt32Array()
var neuron_of := PackedInt32Array()
var intensity := PackedFloat32Array()
var state := PackedByteArray()
var prob := PackedFloat32Array()
var flagged_set := {}
var threshold := 0.5

var standalone := true       # false when app.gd embeds this scene
var layer_lo := 0            # layer filter: cells outside [layer_lo, layer_hi] are hidden
var layer_hi := 1 << 20
var n_layers := 0

var frame := 0
var playing := true
var accum := 0.0
var orbit := 0.0
var xs := 0.06               # x units per neuron column, fitted to the trace in _build
var center := Vector3.ZERO
var radius := 520.0

var flags: MultiMeshInstance3D
var info: RichTextLabel
var strip: RiskStrip
var context: RichTextLabel
var jewel
var graph
var gizmo
var search
var graph_hidden := false

# camera: orbit around target; yaw 0 looks at the field from the front
var yaw := 0.0
var pitch := -0.29
var dist := 520.0
var target := Vector3.ZERO
var sway_on := true
var dragging := 0               # 1 orbit, 2 pan

@onready var cells: MultiMeshInstance3D = $Cells
@onready var hud: Label = $UI/HUD
@onready var cam: Camera3D = $Camera3D


## Risk over the whole reply: per-token line, threshold, flagged stretches and
## a playhead. Click to jump to a token.
class RiskStrip extends Control:
	var owner_node
	func _gui_input(e: InputEvent) -> void:
		if e is InputEventMouseButton and e.pressed and owner_node.n_frames > 1:
			owner_node.seek(int(round(e.position.x / size.x * (owner_node.n_frames - 1))))
	func _draw() -> void:
		var n: int = owner_node.n_frames
		var w := size.x
		var h := size.y
		draw_rect(Rect2(Vector2.ZERO, size), Color(0.086, 0.086, 0.118, 0.92))
		if n < 2:
			return
		var hal := Color(owner_node.theme.halluc)
		var x := func(i): return float(i) / (n - 1) * w
		var y := func(p): return h - 2.0 - (h - 4.0) * p
		if owner_node.prob.size() == n:
			var bw: float = max(2.0, w / n)
			for i in owner_node.flagged_set:
				draw_rect(Rect2(x.call(i) - bw / 2.0, 0, bw, h), Color(hal.r, hal.g, hal.b, 0.2))
			var ty: float = y.call(owner_node.threshold)
			var dx := 0.0
			while dx < w:
				draw_line(Vector2(dx, ty), Vector2(min(dx + 4.0, w), ty), Color(0.54, 0.54, 0.57), 1.0)
				dx += 8.0
			var pts := PackedVector2Array()
			for i in n:
				pts.append(Vector2(x.call(i), y.call(owner_node.prob[i])))
			draw_polyline(pts, hal, 1.5, true)
		else:
			draw_string(ThemeDB.fallback_font, Vector2(8, h / 2 + 4),
				"no classifier scores in this trace: nothing can be flagged", HORIZONTAL_ALIGNMENT_LEFT, -1, 12,
				Color(0.54, 0.54, 0.57))
		draw_rect(Rect2(x.call(owner_node.frame) - 1, 0, 2, h), Color.WHITE)


func _ready() -> void:
	_build_ui()
	if not standalone:
		hud.text = "pick a trace or a review view"
		return
	api = OS.get_environment("OC_API")
	if api == "":
		api = DEFAULT_API
	open(api, OS.get_environment("OC_TOKEN"))


## Load a payload: base is a URL serving api/meta, api/theme and api/trace
## (viz/bloom.py, Studio's /viz/<id>/ or /review/view/<id>/).
func open(base: String, tok: String = "") -> void:
	api = base.trim_suffix("/")
	token = tok
	n_cells = 0
	flagged_set = {}
	prob = PackedFloat32Array()
	frame = 0
	if flags:
		flags.queue_free()
		flags = null
	cells.multimesh = null
	strip.visible = false
	context.visible = false
	for w in [graph, jewel, gizmo, search]:
		w.visible = false
	info.text = ""
	hud.text = "loading %s…" % api
	_fetch("/api/meta", _on_meta)


func set_layer_range(lo: int, hi: int) -> void:
	layer_lo = lo
	layer_hi = hi
	_paint()


## Save what is on screen to user://shots/ (F12). -> the file's absolute path.
func screenshot() -> String:
	DirAccess.make_dir_recursive_absolute("user://shots")
	var path := "user://shots/%s-%d.png" % [Time.get_datetime_string_from_system().replace(":", "-"), frame]
	get_viewport().get_texture().get_image().save_png(path)
	return ProjectSettings.globalize_path(path)


func _build_ui() -> void:
	info = RichTextLabel.new()
	info.bbcode_enabled = true
	info.fit_content = true
	info.scroll_active = false
	info.position = Vector2(14, 10)
	info.size = Vector2(640, 120)
	info.add_theme_font_size_override("normal_font_size", 14)
	info.mouse_filter = Control.MOUSE_FILTER_IGNORE
	$UI.add_child(info)
	strip = RiskStrip.new()
	strip.owner_node = self
	strip.anchor_left = 0.0
	strip.anchor_right = 1.0
	strip.anchor_top = 1.0
	strip.anchor_bottom = 1.0
	strip.offset_left = 14
	strip.offset_right = -14
	strip.offset_top = -100
	strip.offset_bottom = -54
	$UI.add_child(strip)
	context = RichTextLabel.new()
	context.bbcode_enabled = true
	context.scroll_active = false
	context.anchor_left = 0.0
	context.anchor_right = 1.0
	context.anchor_top = 1.0
	context.anchor_bottom = 1.0
	context.offset_left = 14
	context.offset_right = -14
	context.offset_top = -48
	context.offset_bottom = -8
	context.add_theme_font_size_override("normal_font_size", 15)
	context.mouse_filter = Control.MOUSE_FILTER_IGNORE
	$UI.add_child(context)
	strip.visible = false
	context.visible = false
	# the emotion graph, with the jewel at its right; the jewel stays when the graph is hidden
	graph = EmotionGraph.new()
	graph.anchor_left = 0.0
	graph.anchor_right = 1.0
	graph.anchor_top = 1.0
	graph.anchor_bottom = 1.0
	graph.offset_left = 14
	graph.offset_right = -124
	graph.offset_top = -160
	graph.offset_bottom = -106
	graph.seek.connect(seek)
	$UI.add_child(graph)
	jewel = Jewel.new()
	jewel.anchor_left = 1.0
	jewel.anchor_right = 1.0
	jewel.anchor_top = 1.0
	jewel.anchor_bottom = 1.0
	jewel.offset_left = -116
	jewel.offset_right = -14
	jewel.offset_top = -204
	jewel.offset_bottom = -102
	jewel.toggled.connect(_toggle_graph)
	$UI.add_child(jewel)
	graph_hidden = bool(jewel.load_state().get("graph_hidden", false))
	gizmo = NavGizmo.new()
	gizmo.anchor_left = 1.0
	gizmo.anchor_right = 1.0
	gizmo.offset_left = -110
	gizmo.offset_right = -14
	gizmo.offset_top = 52
	gizmo.offset_bottom = 288
	gizmo.orbit.connect(_orbit_by)
	gizmo.view.connect(set_view)
	gizmo.zoom.connect(_zoom)
	gizmo.pan.connect(_pan)
	gizmo.frame_all.connect(func(): set_view("home"))
	gizmo.projection.connect(_toggle_projection)
	gizmo.sway.connect(_toggle_sway)
	$UI.add_child(gizmo)
	search = ChatBox.new()
	search.anchor_left = 1.0
	search.anchor_right = 1.0
	search.offset_left = -440
	search.offset_right = -14
	search.offset_top = 12
	search.offset_bottom = 44
	search.jump.connect(_jump)
	$UI.add_child(search)
	for w in [graph, jewel, gizmo, search]:
		w.visible = false


func _orbit_by(d: Vector2) -> void:
	_user_moved()
	yaw += d.x
	pitch = clampf(pitch + d.y, -1.55, 1.55)


## The sway button and S: on or off, remembered for next time.
func _toggle_sway() -> void:
	sway_on = not sway_on
	gizmo.save_setting("sway", sway_on)


func _jump(i: int) -> void:
	playing = false
	seek(i)


func _toggle_graph() -> void:
	graph_hidden = not graph_hidden
	jewel.save_state({"graph_hidden": graph_hidden})
	_layout_tracks()


## Which bottom widgets show for this trace: the emotion graph and jewel need an
## emotion track; the risk strip needs classifier scores, or shows why there are none.
func _layout_tracks() -> void:
	var emo: Dictionary = meta.get("emotions", {})
	var has_emo: bool = emo.has("z") and n_cells > 0
	var has_risk := prob.size() == n_frames and n_frames > 0
	jewel.visible = has_emo
	graph.visible = has_emo and not graph_hidden
	strip.visible = n_cells > 0 and (has_risk or not has_emo)
	# the graph takes the strip's place when there is no risk to draw
	var top := -160 if strip.visible else -100
	graph.offset_top = top
	graph.offset_bottom = top + 54 if strip.visible else -54
	jewel.offset_top = graph.offset_bottom - 102
	jewel.offset_bottom = graph.offset_bottom
	graph.offset_right = -124


## Hand the widgets this trace's tracks.
func _widget_data() -> void:
	var emo: Dictionary = meta.get("emotions", {})
	var names: Array = emo.get("names", [])
	var cols := []
	for k in names.size():
		cols.append(EMO_PALETTE[k % EMO_PALETTE.size()])
	var d := {"names": names, "z": emo.get("z", []), "colors": cols, "frame": frame}
	graph.set_data(d)
	jewel.set_data(d)
	search.set_data({"labels": meta.get("labels", []), "features": meta.get("features", []), "emotions": emo})


func _fetch(path: String, cb: Callable, binary := false) -> void:
	var r := HTTPRequest.new()
	add_child(r)
	r.request_completed.connect(
		func(_res, code, _h, body):
			r.queue_free()
			if code != 200:
				hud.text = "backend returned %d for %s\nis viz/bloom.py (or Studio) running, and OC_TOKEN set if needed?" % [code, path]
				return
			cb.call(body if binary else JSON.parse_string(body.get_string_from_utf8()))
	)
	var headers := PackedStringArray()
	if token != "":
		headers.append("Authorization: Bearer " + token)
	var err := r.request(api + path, headers)
	if err != OK:
		hud.text = "cannot reach %s" % api


func _on_meta(d) -> void:
	meta = d
	n_frames = int(meta.frames)
	threshold = float(meta.get("threshold", 0.5))
	# JSON numbers arrive as floats; key the set by int.
	for i in meta.get("flagged", []):
		flagged_set[int(i)] = true
	var p = meta.get("prob")
	if p != null:
		prob = PackedFloat32Array(p)
	_fetch("/api/theme", _on_theme)


func _on_theme(d) -> void:
	theme = d
	# The Environment owns the background now (BG_COLOR), so set that rather
	# than the clear colour, which BG_COLOR paints over.
	var env: Environment = $WorldEnvironment.environment
	if env:
		env.background_color = Color(theme.bg)
		env.fog_light_color = Color(theme.bg)
	for w in [graph, jewel, gizmo, search]:
		w.set_theme_data(theme)
	sway_on = bool(gizmo.setting("sway", true))
	gizmo.visible = bool(gizmo.setting("visible", true)) and n_cells > 0
	_fetch("/api/trace", _on_trace, true)


func _on_trace(body: PackedByteArray) -> void:
	# Layout matches the docstring in viz/bloom.py exactly.
	var T := body.decode_s32(0)
	var N := body.decode_s32(4)
	var L := body.decode_s32(8)
	n_frames = T
	n_cells = N
	var o := 12
	layer_of = body.slice(o, o + N * 4).to_int32_array()
	o += N * 4
	neuron_of = body.slice(o, o + N * 4).to_int32_array()
	o += N * 4
	intensity = body.slice(o, o + T * N * 4).to_float32_array()
	o += T * N * 4
	state = body.slice(o, o + T * N)
	var start := OS.get_environment("OC_FRAME")
	if start.is_valid_int():
		frame = clampi(start.to_int(), 0, max(0, T - 1))
	playing = OS.get_environment("OC_PAUSED") != "1"
	hud.text = ""
	context.visible = true
	n_layers = L
	_build(L)
	gizmo.visible = bool(gizmo.setting("visible", true))
	search.visible = true
	_widget_data()
	_layout_tracks()


func _ring_texture() -> ImageTexture:
	# A solid core inside a ring: flagged H-neurons read as a different kind of
	# mark, not just a brighter dot, whatever the colour vision.
	var n := 64
	var img := Image.create(n, n, false, Image.FORMAT_RGBA8)
	for yy in n:
		for xx in n:
			var r := Vector2(xx + 0.5 - n / 2.0, yy + 0.5 - n / 2.0).length() / (n / 2.0)
			var a := 0.0
			if r < 0.44:
				a = 1.0
			elif r < 0.72:
				a = 0.15
			elif r < 1.0:
				a = clampf((1.0 - r) / 0.12, 0.0, 1.0)
			img.set_pixel(xx, yy, Color(1, 1, 1, a))
	return ImageTexture.create_from_image(img)


func _multimesh(count: int) -> MultiMesh:
	var mm := MultiMesh.new()
	mm.transform_format = MultiMesh.TRANSFORM_3D
	mm.use_colors = true
	var quad := QuadMesh.new()
	quad.size = Vector2(1.6, 1.6)
	mm.mesh = quad
	mm.instance_count = count
	return mm


func _build(layers: int) -> void:
	# Fit the field to the view: neuron columns span 640 units whatever the
	# trace's width (512 bins or 14336 neurons); layers are 8 units apart. The
	# same numbers as the three.js client in viz/bloom.py.
	var width := 640.0
	xs = width / max(1, int(meta.get("neurons", 512)))
	var height := layers * 8.0
	center = Vector3(width / 2.0, height / 2.0 - 40.0, 0.0)
	radius = 0.65 * max(width, height)
	set_view("home")
	var mm := _multimesh(n_cells)
	for i in n_cells:
		mm.set_instance_transform(i, Transform3D(Basis.IDENTITY,
			Vector3(neuron_of[i] * xs, layer_of[i] * 8.0, 0.0)))
	cells.multimesh = mm
	# Flagged H-neurons: their own MultiMesh, unshaded, drawn over the field.
	flags = MultiMeshInstance3D.new()
	var fm := StandardMaterial3D.new()
	fm.shading_mode = BaseMaterial3D.SHADING_MODE_UNSHADED
	fm.transparency = BaseMaterial3D.TRANSPARENCY_ALPHA
	fm.billboard_mode = BaseMaterial3D.BILLBOARD_ENABLED
	fm.billboard_keep_scale = true    # without it billboards ignore instance scale
	fm.vertex_color_use_as_albedo = true
	fm.no_depth_test = true
	fm.render_priority = 10
	fm.albedo_texture = _ring_texture()
	flags.material_override = fm
	flags.multimesh = _multimesh(n_cells)
	add_child(flags)
	_paint()


func seek(i: int) -> void:
	frame = clampi(i, 0, n_frames - 1)
	_paint()


func _paint() -> void:
	if cells.multimesh == null:
		return
	var mm := cells.multimesh
	var fmm := flags.multimesh
	var base := frame * n_cells
	var c_idle := Color(0.16, 0.16, 0.22)
	var c_act := Color(theme.active)
	var c_hal := Color(theme.halluc)
	for i in n_cells:
		var v: float = intensity[base + i]
		var st: int = state[base + i]
		var col: Color = c_idle
		var gain := 0.35
		if st >= 1:
			col = c_act
			gain = 0.6 + 0.8 * v
		mm.set_instance_color(i, Color(col.r * gain, col.g * gain, col.b * gain, 1.0 if st > 0 else 0.35))
		var s: float = (SIZE_ACTIVE if st >= 1 else SIZE_IDLE) * (0.6 + 0.9 * v)
		if layer_of[i] < layer_lo or layer_of[i] > layer_hi:
			s = 0.0
			st = 0
		mm.set_instance_transform(i, Transform3D(Basis.IDENTITY.scaled(Vector3(s, s, s)),
			Vector3(neuron_of[i] * xs, layer_of[i] * 8.0, 0.0)))
		# Full colour, never pushed past 1, so the hue survives the glow instead of washing to white.
		var fs: float = SIZE_FLAG * (0.8 + 0.5 * v) if st == 2 else 0.0
		fmm.set_instance_color(i, c_hal)
		fmm.set_instance_transform(i, Transform3D(Basis.IDENTITY.scaled(Vector3(fs, fs, fs)),
			Vector3(neuron_of[i] * xs, layer_of[i] * 8.0, 0.5)))
	_update_text()
	strip.queue_redraw()
	if graph:
		graph.frame = frame
		graph.queue_redraw()
		jewel.frame = frame


func _hex(c: Color) -> String:
	return "#" + c.to_html(false)


# Emotion and feature tracks recorded by the decoding lab (scripts/live_watch.py), for the current token.
func _tracks_text() -> String:
	var out := ""
	var emo: Dictionary = meta.get("emotions", {})
	if emo.has("z") and frame < emo.z.size():
		var pal := ["#FF4D1A", "#3D8BFF", "#1D9E75", "#E7B65C", "#B07CFF", "#c9c9d2"]
		var rows := []
		for k in range(emo.names.size()):
			rows.append([absf(float(emo.z[frame][k])), k])
		rows.sort_custom(func(a, b): return a[0] > b[0])
		var parts := []
		for r in rows.slice(0, 4):
			var k: int = r[1]
			var z := float(emo.z[frame][k])
			parts.append("[color=%s]%s %s%.1f[/color]" % [pal[k % pal.size()], str(emo.names[k]), "+" if z > 0 else "", z])
		out += "\nemotions: " + " · ".join(parts)
	var feats: Array = meta.get("features", [])
	if frame < feats.size() and feats[frame].size() > 0:
		var parts := []
		for f in feats[frame].slice(0, 5):
			var label := str(f[3]) if str(f[3]) != "" else "#%d" % int(f[1])
			parts.append("%s [color=#8a8a92]%.2f[/color]" % [label.replace("[", "(").replace("]", ")"), float(f[2])])
		out += "\nfeatures: " + " · ".join(parts)
	return out


func _update_text() -> void:
	var flagged: bool = flagged_set.has(frame)
	var hal := _hex(Color(theme.halluc))
	var act := _hex(Color(theme.active))
	var mode := str(meta.get("mode", "unscored"))
	var mode_text := "tokens flagged at risk ≥ %d%% (the classifier's own threshold)" % int(round(threshold * 100))
	var legend: Dictionary = meta.get("legend", {"active": "active", "halluc": "H-neuron on a flagged token"})
	var unit := "time buckets" if mode == "review" else "tokens"
	if mode == "review":
		mode_text = "%s · cells: %s\nstrip: %s" % [str(meta.get("question", "")), str(meta.get("cells_note", "")),
			str(meta.get("strip", ""))]
	elif mode == "relative":
		mode_text = "[color=#ff9b8a]relative mode: some tokens are always flagged[/color]"
	elif mode == "unscored" and str(meta.get("source", "")) == "decode_lab":
		mode_text = "decoding-lab recording: CETT, emotions and features per token (no classifier, nothing is flagged)"
	elif mode == "unscored":
		mode_text = "[color=#ff9b8a]no classifier scores: nothing can be flagged[/color]"
	var cells_note := "" if mode == "review" else str(meta.get("cells_note", ""))
	var order := "x: neurons by classifier weight, H-neurons at left · y: layer" if meta.get("order") == "weight" else "x: neuron index · y: layer"
	var risk := ""
	if prob.size() == n_frames:
		risk = "   %s [b]%d%%[/b]" % ["error rate" if mode == "review" else "risk", int(round(prob[frame] * 100))]
	info.text = ("[b]%s[/b]   [color=#8a8a92]%d/%d %s[/color]%s%s\n" % [str(meta.get("model", "trace")), frame + 1,
		n_frames, unit, risk, ("   [color=%s][b]FLAGGED[/b][/color]" % hal) if flagged else ""]) + \
		("[color=%s]●[/color] %s   %s\n" % [act, str(legend.get("active", "")),
			("[color=%s]◉[/color] %s" % [hal, str(legend.get("halluc"))]) if str(legend.get("halluc", "")) != "" else ""]) + \
		("[color=#8a8a92]%s%s\n%s[/color]" % [mode_text, ("\n" + cells_note) if cells_note.begins_with("none") else "", order]) + \
		_tracks_text()
	# The reply around the current token, shaded by risk; the current token boxed.
	var labels: Array = meta.get("labels", [])
	var lo := maxi(0, frame - 14)
	var hi := mini(labels.size(), frame + 15)
	var out := ""
	for i in range(lo, hi):
		var t := str(labels[i]).replace("[", "(").replace("]", ")").replace("\n", " ")
		if prob.size() == n_frames and prob[i] > 0.15:
			var a := clampf((prob[i] - 0.15) / 0.85, 0.0, 1.0) * 0.6
			var bg := Color(theme.halluc)
			bg.a = a
			t = "[bgcolor=%s]%s[/bgcolor]" % ["#" + bg.to_html(true), t]
		if flagged_set.has(i):
			t = "[u]%s[/u]" % t
		if i == frame:
			t = "[b][color=#ffffff]⟦%s⟧[/color][/b]" % t
		out += t
	context.text = ("…" if lo > 0 else "") + out + ("…" if hi < labels.size() else "")


func _process(delta: float) -> void:
	if n_cells == 0:
		return
	if playing:
		accum += delta
		if accum >= 1.0 / FPS:
			accum = 0.0
			frame = (frame + 1) % n_frames
			_paint()
	# The sway is a slow swing in front of the field, not a full orbit: from
	# behind, the billboards are the same picture mirrored. Moving the view stops it.
	if sway_on:
		orbit += delta * 0.15
	var y := yaw + (0.35 * sin(orbit) if sway_on else 0.0)
	var dir := Vector3(sin(y) * cos(pitch), -sin(pitch), cos(y) * cos(pitch))
	cam.position = target + dir * dist
	cam.look_at(target, Vector3.UP if absf(pitch) < 1.5 else Vector3.FORWARD)
	if cam.projection == Camera3D.PROJECTION_ORTHOGONAL:
		cam.size = dist * 1.1
	if gizmo:
		gizmo.set_camera(cam.global_transform.basis, cam.projection == Camera3D.PROJECTION_ORTHOGONAL, sway_on)


## Preset views (the gizmo's axis balls, numpad keys): front, back, left, right, top, bottom, home.
func set_view(name: String) -> void:
	var views := {"front": [0.0, 0.0], "back": [PI, 0.0], "right": [PI / 2, 0.0], "left": [-PI / 2, 0.0],
		"top": [0.0, -1.55], "bottom": [0.0, 1.55], "home": [0.0, -0.29]}
	if not views.has(name):
		return
	yaw = views[name][0]
	pitch = views[name][1]
	if name == "home":
		target = center
		dist = radius * 1.05
		orbit = 0.0
	else:
		_user_moved()


func _user_moved() -> void:
	sway_on = false
	orbit = 0.0


func _zoom(steps: float) -> void:
	dist = clampf(dist * pow(0.88, steps), 20.0, 5000.0)


func _pan(rel: Vector2) -> void:
	_user_moved()
	var b := cam.global_transform.basis
	var k := dist / 600.0
	target += (-b.x * rel.x + b.y * rel.y) * k


func _toggle_projection() -> void:
	cam.projection = Camera3D.PROJECTION_PERSPECTIVE if cam.projection == Camera3D.PROJECTION_ORTHOGONAL \
		else Camera3D.PROJECTION_ORTHOGONAL


func _unhandled_input(e: InputEvent) -> void:
	if e is InputEventMouseButton:
		if e.button_index == MOUSE_BUTTON_WHEEL_UP and e.pressed:
			_zoom(1.0)
		elif e.button_index == MOUSE_BUTTON_WHEEL_DOWN and e.pressed:
			_zoom(-1.0)
		elif e.button_index in [MOUSE_BUTTON_LEFT, MOUSE_BUTTON_MIDDLE]:
			dragging = 0
			if e.pressed:
				dragging = 2 if e.button_index == MOUSE_BUTTON_MIDDLE or e.shift_pressed else 1
		return
	if e is InputEventMouseMotion and dragging > 0:
		if dragging == 1:
			_user_moved()
			yaw -= e.relative.x * 0.006
			pitch = clampf(pitch - e.relative.y * 0.006, -1.55, 1.55)
		else:
			_pan(e.relative)
		return
	if e is InputEventKey and e.pressed:
		match e.keycode:
			KEY_KP_1:
				set_view("back" if e.ctrl_pressed else "front")
			KEY_KP_3:
				set_view("left" if e.ctrl_pressed else "right")
			KEY_KP_7:
				set_view("bottom" if e.ctrl_pressed else "top")
			KEY_KP_5:
				_toggle_projection()
			KEY_HOME, KEY_KP_PERIOD:
				set_view("home")
			KEY_KP_ADD, KEY_EQUAL:
				_zoom(1.0)
			KEY_KP_SUBTRACT, KEY_MINUS:
				_zoom(-1.0)
			KEY_S:
				_toggle_sway()
			KEY_SPACE:
				playing = not playing
			KEY_LEFT:
				seek((frame - 1 + n_frames) % n_frames)
			KEY_RIGHT:
				seek((frame + 1) % n_frames)
			KEY_N:
				# next flagged token
				for k in range(1, n_frames + 1):
					var j := (frame + k) % n_frames
					if flagged_set.has(j):
						seek(j)
						break
			KEY_F12:
				hud.text = "saved " + screenshot()
			KEY_ESCAPE:
				if standalone:
					get_tree().quit()
