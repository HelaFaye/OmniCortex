extends "res://widgets/widget.gd"
## Emotion lanes over the whole reply: one line per emotion (z against its
## story mean), a zero line, a playhead, and the names in their lane colours.
## Click to jump to a token (signal seek).
##
## Reads set_data({"names", "z": [[z per emotion] per token], "colors", "frame"}).
## Settings: "lanes" (how many emotions to draw, the strongest first; 6),
## "labels" (true).

signal seek(i: int)

var names: Array = []
var series: Array = []
var colors: Array = []
var frame := 0


func _init() -> void:
	widget_id = "emotion_graph"
	mouse_filter = Control.MOUSE_FILTER_STOP
	tooltip_text = "Emotion activations per token (z against the stories). Click to jump to a token."


func set_data(d: Dictionary) -> void:
	names = d.get("names", [])
	series = d.get("z", [])
	colors = d.get("colors", [])
	frame = int(d.get("frame", 0))
	queue_redraw()


func _gui_input(e: InputEvent) -> void:
	if e is InputEventMouseButton and e.pressed and e.button_index == MOUSE_BUTTON_LEFT and series.size() > 1:
		seek.emit(int(round(clampf(e.position.x / size.x, 0.0, 1.0) * (series.size() - 1))))
		accept_event()


func _draw() -> void:
	var w := size.x
	var h := size.y
	draw_rect(Rect2(Vector2.ZERO, size), Color(0.086, 0.086, 0.118, 0.92))
	var n := series.size()
	if n < 2 or names.is_empty():
		draw_string(ThemeDB.fallback_font, Vector2(8, h / 2 + 4), "no emotion track in this trace",
			HORIZONTAL_ALIGNMENT_LEFT, -1, 12, Color(0.54, 0.54, 0.57))
		return
	var mx := 3.0
	var peak := []
	for k in names.size():
		peak.append([0.0, k])
	for row in series:
		for k in mini(row.size(), names.size()):
			var a := absf(float(row[k]))
			mx = maxf(mx, a)
			peak[k][0] = maxf(peak[k][0], a)
	peak.sort_custom(func(a, b): return a[0] > b[0])
	var lanes := mini(int(setting("lanes", 6)), names.size())
	draw_line(Vector2(0, h / 2), Vector2(w, h / 2), Color(0.16, 0.16, 0.2), 1.0)
	var lx := 6.0
	for p in peak.slice(0, lanes):
		var k: int = p[1]
		var col := Color(str(colors[k])) if k < colors.size() else Color.from_hsv(fmod(0.13 * k, 1.0), 0.6, 1.0)
		var pts := PackedVector2Array()
		for i in n:
			var z := float(series[i][k]) if k < series[i].size() else 0.0
			pts.append(Vector2(float(i) / (n - 1) * w, h / 2 - (h / 2 - 3) * z / mx))
		draw_polyline(pts, col, 1.4, true)
		if setting("labels", true):
			var label := str(names[k])
			draw_string(ThemeDB.fallback_font, Vector2(lx, 12), label, HORIZONTAL_ALIGNMENT_LEFT, -1, 11, col)
			lx += ThemeDB.fallback_font.get_string_size(label, HORIZONTAL_ALIGNMENT_LEFT, -1, 11).x + 10
	var px := float(frame) / (n - 1) * w
	draw_rect(Rect2(px - 1, 0, 2, h), Color.WHITE)
