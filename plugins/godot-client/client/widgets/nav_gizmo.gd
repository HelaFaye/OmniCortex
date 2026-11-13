extends "res://widgets/widget.gd"
## Viewport navigation in the manner of Blender's: an axis gizmo that turns
## with the camera (drag it to orbit, click an axis ball to look down that
## axis), and buttons beside it to zoom, pan, frame everything, switch
## perspective and orthographic, and pick a preset view.
##
## The widget only asks; the 3D view does it. Signals carry what was asked:
##   orbit(delta: Vector2)  drag on the gizmo, in radians
##   view(name: String)     front, back, left, right, top, bottom, home
##   zoom(steps: float)     + in, - out
##   pan(delta: Vector2)    drag from the hand button, in pixels
##   frame_all()            fit the field in view
##   projection()           toggle perspective / orthographic
##   sway()                 the slow automatic sway on or off
## Call set_camera(basis) each frame so the gizmo shows where the camera looks.

signal orbit(delta: Vector2)
signal view(name: String)
signal zoom(steps: float)
signal pan(delta: Vector2)
signal frame_all
signal projection
signal sway

const AXES := [["x", Vector3.RIGHT, Color("#ff4d5a"), "right", "left"],
	["y", Vector3.UP, Color("#7ed957"), "top", "bottom"],
	["z", Vector3.BACK, Color("#4d8bff"), "front", "back"]]

var cam_basis := Basis.IDENTITY
var ortho := false
var swaying := true
var dragging := 0               # 0 none, 1 orbit (on the gizmo), 2 zoom, 3 pan
var balls := []                 # [screen centre, view name] for hit tests
var buttons := []               # [rect, id]
var hover := ""


func _init() -> void:
	widget_id = "nav_gizmo"
	custom_minimum_size = Vector2(96, 236)
	mouse_filter = Control.MOUSE_FILTER_STOP


func set_camera(b: Basis, is_ortho: bool, is_swaying: bool) -> void:
	cam_basis = b
	ortho = is_ortho
	swaying = is_swaying
	queue_redraw()


func _gizmo_center() -> Vector2:
	return Vector2(size.x / 2, 46)


func _draw() -> void:
	var c := _gizmo_center()
	var r := 34.0
	var bg := Color(1, 1, 1, 0.08 if hover == "gizmo" or dragging == 1 else 0.0)
	draw_circle(c, r + 10, bg)
	# axes as the camera sees them: farther ones first
	var inv := cam_basis.inverse()
	var items := []
	for a in AXES:
		for sgn in [1.0, -1.0]:
			var v: Vector3 = inv * (a[1] * sgn)
			items.append([v.z, v, a, sgn])
	items.sort_custom(func(p, q): return p[0] < q[0])
	balls = []
	for it in items:
		var v: Vector3 = it[1]
		var a: Array = it[2]
		var p := c + Vector2(v.x, -v.y) * r
		var col: Color = a[2]
		if it[3] > 0:
			draw_line(c, p, col, 2.0, true)
			draw_circle(p, 8.0, col)
			draw_string(ThemeDB.fallback_font, p + Vector2(-4, 4), str(a[0]).to_upper(), HORIZONTAL_ALIGNMENT_LEFT, -1, 11, Color(0.05, 0.05, 0.08))
			balls.append([p, a[3]])
		else:
			draw_circle(p, 6.5, Color(col, 0.35))
			draw_arc(p, 6.5, 0, TAU, 16, col, 1.2, true)
			balls.append([p, a[4]])
	# buttons: zoom, pan, frame, projection, sway, presets
	buttons = []
	var y := 96.0
	var labels := [["zoom", "⌕", "Drag up or down to zoom (or use the wheel)"],
		["pan", "✋", "Drag to pan (or Shift + drag in the view)"],
		["frame", "⛶", "Frame everything"],
		["proj", "▣" if ortho else "◇", "Orthographic" if ortho else "Perspective"],
		["sway", "∿", "Gentle sway on or off" ]]
	for b in labels:
		var rect := Rect2(size.x / 2 - 15, y, 30, 26)
		var on: bool = (b[0] == "sway" and swaying) or hover == b[0]
		draw_rect(rect, Color(1, 1, 1, 0.16 if on else 0.07))
		draw_string(ThemeDB.fallback_font, rect.position + Vector2(8, 18), b[1], HORIZONTAL_ALIGNMENT_LEFT, -1, 14, Color(0.85, 0.85, 0.9))
		buttons.append([rect, b[0], b[2]])
		y += 28


func _hit(p: Vector2) -> String:
	for b in balls:
		if p.distance_to(b[0]) <= 9.0:
			return "view:" + str(b[1])
	if p.distance_to(_gizmo_center()) <= 44:
		return "gizmo"
	for b in buttons:
		if (b[0] as Rect2).has_point(p):
			return b[1]
	return ""


func _gui_input(e: InputEvent) -> void:
	if e is InputEventMouseMotion:
		if dragging == 1:
			orbit.emit(Vector2(-e.relative.x, -e.relative.y) * 0.01)
		elif dragging == 2:
			zoom.emit(-e.relative.y * 0.05)
		elif dragging == 3:
			pan.emit(e.relative)
		else:
			var h := _hit(e.position)
			if h != hover:
				hover = h
				tooltip_text = ""
				for b in buttons:
					if b[1] == h:
						tooltip_text = b[2]
				if h.begins_with("view:"):
					tooltip_text = "Look from the " + h.substr(5)
				elif h == "gizmo":
					tooltip_text = "Drag to orbit; click an axis to look along it"
				queue_redraw()
		accept_event()
	elif e is InputEventMouseButton:
		if e.button_index == MOUSE_BUTTON_WHEEL_UP:
			zoom.emit(1.0)
		elif e.button_index == MOUSE_BUTTON_WHEEL_DOWN:
			zoom.emit(-1.0)
		elif e.button_index == MOUSE_BUTTON_LEFT:
			if e.pressed:
				var h := _hit(e.position)
				if h.begins_with("view:"):
					view.emit(h.substr(5))
				elif h == "gizmo":
					dragging = 1
				elif h == "zoom":
					dragging = 2
				elif h == "pan":
					dragging = 3
				elif h == "frame":
					frame_all.emit()
				elif h == "proj":
					projection.emit()
				elif h == "sway":
					sway.emit()
			else:
				dragging = 0
			queue_redraw()
		accept_event()


func _notification(what: int) -> void:
	if what == NOTIFICATION_MOUSE_EXIT and dragging == 0:
		hover = ""
		queue_redraw()
