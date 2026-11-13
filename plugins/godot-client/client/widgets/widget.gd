extends Control
## Base for OmniCortex's HUD widgets (docs/WIDGETS.md).
##
## A widget is a Control that draws one thing and takes its data and settings
## from outside: set_data() with the current token's values, set_settings()
## with the theme's "widgets" block merged with the viewer's own choices.
## Its own state (shown or hidden, a style picked here) persists in
## user://widgets.json under its widget_id, so it survives restarts.
##
## The web pages have the same widgets (viz/widgets.py) with the same settings
## keys, so a theme looks the same in the browser and in Godot.

const STORE := "user://widgets.json"

var widget_id := "widget"
var settings: Dictionary = {}
var theme_colors: Dictionary = {}


## The theme (api/theme): its colours, and its "widgets" block for this widget.
func set_theme_data(t: Dictionary) -> void:
	theme_colors = t
	var w: Dictionary = t.get("widgets", {})
	var mine: Dictionary = w.get(widget_id, {})
	var merged := mine.duplicate()
	merged.merge(load_state().get("settings", {}), true)   # the viewer's own picks win
	set_settings(merged)


func set_settings(s: Dictionary) -> void:
	settings = s
	queue_redraw()


func setting(key: String, default):
	return settings.get(key, default)


## The current token's values; each widget documents what it reads.
func set_data(_d: Dictionary) -> void:
	pass


static func _all() -> Dictionary:
	if not FileAccess.file_exists(STORE):
		return {}
	var f := FileAccess.open(STORE, FileAccess.READ)
	var d = JSON.parse_string(f.get_as_text()) if f else null
	return d if d is Dictionary else {}


func load_state() -> Dictionary:
	return _all().get(widget_id, {})


func save_state(patch: Dictionary) -> void:
	var all := _all()
	var mine: Dictionary = all.get(widget_id, {})
	mine.merge(patch, true)
	all[widget_id] = mine
	var f := FileAccess.open(STORE, FileAccess.WRITE)
	if f:
		f.store_string(JSON.stringify(all))


## Remember one setting chosen in this client (it overrides the theme's).
func save_setting(key: String, value) -> void:
	var s: Dictionary = load_state().get("settings", {})
	s[key] = value
	save_state({"settings": s})
	settings[key] = value
	queue_redraw()
