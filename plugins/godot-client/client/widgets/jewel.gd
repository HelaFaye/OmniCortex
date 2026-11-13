extends "res://widgets/widget.gd"
## The emotion jewel: one glance at the model's emotion vectors on the current token.
##
## Reads set_data({"names": [...], "z": [[z per emotion] per token], "colors": [...], "frame": i}),
## the same track the emotion graph draws, so the jewel's colours are the lanes' colours.
##
##   colour     the lanes' colours mixed by |z| squared (negative z mixed in darker)
##   size, glow how far the strongest emotion is from its story mean (|z| of 3 = full)
##   motion     each emotion's ribbon, face bulge or 4D turn follows its own z
##
## Styles (settings "style"): core (a reactor core with an orbiting ribbon per
## emotion) and off; add-ons add more (the jewel-styles add-on: icosphere and
## tesseract) through add_styles(), which takes an object with STYLES and
## draw(style, jewel, centre, radius).
## Values glide toward each token's numbers rather than jump, so stepping
## through a reply reads as one continuous motion.
## Click: hide or show the emotion graph (signal toggled). Right-click: next style.

signal toggled

const BUILT_IN := ["core", "off"]

var names: Array = []
var series: Array = []
var colors: Array = []
var frame := 0
var t := 0.0
var cur: PackedFloat32Array = PackedFloat32Array()     # smoothed z per emotion
var mag := 0.0                                          # smoothed max |z| / 3
var mix := Color(0.4, 0.55, 1.0)

var extra: Dictionary = {}     # style -> the add-on object that draws it


func _init() -> void:
	widget_id = "jewel"
	custom_minimum_size = Vector2(96, 96)
	mouse_filter = Control.MOUSE_FILTER_STOP
	tooltip_text = "Emotion jewel: click to hide or show the emotion graph, right-click for the next style"
	var additive := CanvasItemMaterial.new()          # light adds up where it overlaps, as in the original
	additive.blend_mode = CanvasItemMaterial.BLEND_MODE_ADD
	material = additive


func set_data(d: Dictionary) -> void:
	names = d.get("names", [])
	series = d.get("z", [])
	colors = d.get("colors", [])
	frame = int(d.get("frame", 0))
	if cur.size() != names.size():
		cur = PackedFloat32Array()
		cur.resize(names.size())


## Styles from an add-on: obj has STYLES (names) and draw(style, jewel, centre, radius).
func add_styles(obj) -> void:
	for s in obj.STYLES:
		extra[str(s)] = obj
	queue_redraw()


func styles() -> Array:
	return ["core"] + extra.keys() + ["off"]


func target() -> PackedFloat32Array:
	var out := PackedFloat32Array()
	out.resize(names.size())
	if frame < series.size():
		var row: Array = series[frame]
		for k in mini(row.size(), out.size()):
			out[k] = float(row[k])
	return out


func color_of(k: int) -> Color:
	if k < colors.size():
		return Color(str(colors[k]))
	return Color.from_hsv(fmod(0.13 * k, 1.0), 0.6, 1.0)


func _process(delta: float) -> void:
	if not is_visible_in_tree():
		return
	var sp: float = float(setting("speed", 1.0))
	t += delta * sp
	var goal := target()
	var a := 1.0 - exp(-delta * 5.0)
	var mx := 0.0
	var acc := Vector3.ZERO
	var wsum := 0.0
	for k in cur.size():
		cur[k] = lerpf(cur[k], goal[k], a)
		mx = maxf(mx, absf(cur[k]))
		var w := cur[k] * cur[k]
		var c := color_of(k)
		if cur[k] < 0:
			c = c.darkened(0.5)
		acc += Vector3(c.r, c.g, c.b) * w
		wsum += w
	mag = lerpf(mag, clampf(mx / 3.0, 0.0, 1.0), a)
	if wsum > 1e-4:
		var m := acc / wsum
		mix = mix.lerp(Color(m.x, m.y, m.z), a)
	queue_redraw()


func _gui_input(e: InputEvent) -> void:
	if e is InputEventMouseButton and e.pressed:
		if e.button_index == MOUSE_BUTTON_LEFT:
			toggled.emit()
		elif e.button_index == MOUSE_BUTTON_RIGHT:
			var all := styles()
			var i := all.find(str(setting("style", "core")))
			save_setting("style", all[(i + 1) % all.size()])
		accept_event()


func _draw() -> void:
	var c := size / 2.0
	var r := minf(size.x, size.y) * 0.5 * float(setting("scale", 0.9))
	var glow: float = float(setting("glow", 1.0))
	var style := str(setting("style", "core"))
	if style == "off" or names.is_empty():
		draw_arc(c, r * 0.35, 0, TAU, 32, Color(0.4, 0.4, 0.45, 0.5), 1.5, true)
		return
	# a soft halo, wider and brighter as emotions rise
	for i in 6:
		var f := 1.0 - i / 6.0
		var hc := mix
		hc.a = 0.05 * glow * (0.4 + mag) * f
		draw_circle(c, r * (0.45 + 0.55 * (1.0 - f)) * (0.7 + 0.3 * mag), hc)
	if extra.has(style):
		extra[style].draw(style, self, c, r)
	else:
		_draw_core(c, r, glow)          # also when a style's add-on is not here


# ------------------------------------------------------------- core

## A reactor core in the manner of the original Xbox's start-up: a glowing heart
## with energy swirling inside it, and one ribbon of light per emotion on its own
## tilted orbit. A ribbon's length, width and speed follow its emotion's |z|;
## positive z turns one way, negative the other. The jewel draws additively
## (see _init), so light builds up where it overlaps. Same drawing as viz/widgets.py.
func _draw_core(c: Vector2, r: float, glow: float) -> void:
	var n := cur.size()
	var back := []
	var front := []
	for k in n:
		var z := cur[k]
		var s := clampf(absf(z) / 3.0, 0.0, 1.0)
		var tilt := 0.35 + 1.1 * float(k) / maxf(1.0, n)
		var spin := float(k) * TAU / maxf(1.0, n)
		var head := t * (0.8 + 1.6 * s) * (1.0 if z >= 0 else -1.0) + spin * 1.7
		var length := 0.9 + 2.6 * s
		var steps := 34
		var col := color_of(k)
		var rr := r * (0.66 + 0.14 * k / maxf(1.0, n))
		for j in steps:
			var p0 := _orbit(c, rr, tilt, spin, head - length * float(j) / steps)
			var p1 := _orbit(c, rr, tilt, spin, head - length * float(j + 1) / steps)
			var f := 1.0 - float(j) / steps
			var seg := [p0[0], p1[0], col, f * (0.45 + 0.55 * s), (2.2 + 4.5 * s) * (0.35 + 0.65 * f)]
			if p0[1] < 0:
				back.append(seg)
			else:
				front.append(seg)
	_ribbons(back, 0.35)
	# the heart: a smooth glow, the mix colour at the rim and nearly white in the middle
	var pulse := 1.0 + 0.08 * sin(t * (2.0 + 5.0 * mag)) * (0.3 + mag)
	var cr := r * (0.27 + 0.18 * mag) * pulse
	var rings := 22
	for i in rings:
		var f := float(i) / rings
		var col := mix.lerp(Color(1, 1, 1), f * f * 0.9)
		col.a = (0.05 + 0.1 * f) * minf(1.0, 0.6 + 0.4 * glow)
		draw_circle(c, cr * 1.7 * (1.0 - f), col)
	for i in 3:                                     # energy turning inside the heart
		var a0 := t * (1.3 + 0.4 * i) * (-1.0 if i % 2 else 1.0) + i * 2.1
		var col := mix.lerp(Color(1, 1, 1), 0.6)
		col.a = 0.25 + 0.35 * mag
		draw_arc(c, cr * (0.45 + 0.17 * i), a0, a0 + 1.4 + mag, 24, col, 1.5 + 2.0 * mag, true)
	_ribbons(front, 1.0)


## Each ribbon twice: a wide faint glow, then a bright line.
func _ribbons(list: Array, f: float) -> void:
	for sgm in list:
		var cc: Color = sgm[2]
		cc.a = sgm[3] * f * 0.22
		draw_line(sgm[0], sgm[1], cc, sgm[4] * 3.2, true)
	for sgm in list:
		var cc: Color = (sgm[2] as Color).lerp(Color(1, 1, 1), 0.3)
		cc.a = sgm[3] * f
		draw_line(sgm[0], sgm[1], cc, sgm[4], true)


## A point on a tilted, turned circle -> [screen point, depth].
func _orbit(c: Vector2, rr: float, tilt: float, spin: float, u: float) -> Array:
	var p := Vector3(cos(u) * rr, sin(u) * rr * cos(tilt), sin(u) * rr * sin(tilt))
	var q := Vector3(p.x * cos(spin) - p.y * sin(spin), p.x * sin(spin) + p.y * cos(spin), p.z)
	return [c + Vector2(q.x, q.y), q.z]
