extends "res://widgets/widget.gd"
## A compact search and chat line for the 3D view.
##
## Typing searches the open trace as you go: its tokens, the features recorded
## at each token, and emotion names. Pick a hit (or press Enter on the first)
## to jump to that token. Start with ">" to send the text to the chat model
## instead (signal ask); the host shows the answer with show_answer().
##
## set_data({"labels": [token text], "features": [[...] per token], "emotions": {"names", "z"}})
## Settings: "max_hits" (8).

signal jump(i: int)
signal ask(text: String)

var line: LineEdit
var hits: ItemList
var answer: RichTextLabel
var labels: Array = []
var feats: Array = []
var emo: Dictionary = {}
var can_chat := false


func _init() -> void:
	widget_id = "chat_box"
	custom_minimum_size = Vector2(300, 32)
	mouse_filter = Control.MOUSE_FILTER_PASS


func _ready() -> void:
	var v := VBoxContainer.new()
	v.set_anchors_preset(Control.PRESET_FULL_RECT)
	v.mouse_filter = Control.MOUSE_FILTER_IGNORE
	add_child(v)
	line = LineEdit.new()
	line.placeholder_text = "search tokens, features, emotions · > to ask the model"
	line.clear_button_enabled = true
	line.text_changed.connect(_search)
	line.text_submitted.connect(_submit)
	v.add_child(line)
	hits = ItemList.new()
	hits.custom_minimum_size = Vector2(0, 150)
	hits.visible = false
	hits.item_activated.connect(_pick)
	hits.item_clicked.connect(func(i, _p, _b): _pick(i))
	v.add_child(hits)
	answer = RichTextLabel.new()
	answer.fit_content = true
	answer.bbcode_enabled = true
	answer.visible = false
	answer.custom_minimum_size = Vector2(0, 24)
	answer.add_theme_font_size_override("normal_font_size", 13)
	v.add_child(answer)


func set_data(d: Dictionary) -> void:
	labels = d.get("labels", [])
	feats = d.get("features", [])
	emo = d.get("emotions", {})


func _search(q: String) -> void:
	hits.clear()
	q = q.strip_edges().to_lower()
	if q == "" or q.begins_with(">"):
		hits.visible = false
		if q.begins_with(">") and not can_chat:
			answer.visible = true
			answer.text = "[color=#8a8a92]asking needs the full client connected to Studio[/color]"
		return
	var cap := int(setting("max_hits", 8))
	for i in labels.size():
		if str(labels[i]).to_lower().contains(q):
			_add("token %d: %s" % [i, str(labels[i]).strip_edges()], i)
			if hits.item_count >= cap:
				break
	for i in feats.size():
		if hits.item_count >= cap * 2:
			break
		for f in feats[i]:
			if str(f[3]).to_lower().contains(q) or ("#%d" % int(f[1])) == q:
				_add("feature %s at token %d (%.2f)" % [str(f[3]) if str(f[3]) != "" else "#%d" % int(f[1]), i, float(f[2])], i)
				break
	var names: Array = emo.get("names", [])
	var z: Array = emo.get("z", [])
	for k in names.size():
		if str(names[k]).to_lower().contains(q) and z.size() > 0:
			var best := 0
			for i in z.size():
				if float(z[i][k]) > float(z[best][k]):
					best = i
			_add("%s peaks at token %d (z %.1f)" % [str(names[k]), best, float(z[best][k])], best)
	hits.visible = hits.item_count > 0


func _add(text: String, i: int) -> void:
	hits.add_item(text)
	hits.set_item_metadata(hits.item_count - 1, i)


func _pick(i: int) -> void:
	jump.emit(int(hits.get_item_metadata(i)))
	hits.visible = false


func _submit(text: String) -> void:
	var q := text.strip_edges()
	if q.begins_with(">"):
		if can_chat and q.length() > 1:
			answer.visible = true
			answer.text = "[color=#8a8a92]asking…[/color]"
			ask.emit(q.substr(1).strip_edges())
		return
	if hits.item_count > 0:
		_pick(0)


func show_answer(text: String) -> void:
	answer.visible = true
	answer.text = text.replace("[", "(").replace("]", ")")
