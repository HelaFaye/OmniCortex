extends Control
## OmniCortex's Godot client: all of Studio, plus a native 3D view.
##
## Client-server by default. Studio runs on the machine with the GPU and the
## models; this runs anywhere else on the network (or on the same machine if
## you must), so drawing never takes GPU memory from the model. It uses the
## same HTTP API as Studio's web pages, so the two cannot drift.
##
##   godot --path plugins/godot-client/client                                   # pick or add a Studio under Connect
##   NS_STUDIO=https://gpu-box:7870 NS_TOKEN=… godot --path plugins/godot-client/client
##   NS_TAB=Review godot --path plugins/godot-client/client                       # open on a tab (or a page inside a group)
##   NS_SHOTS=/tmp/shots NS_STUDIO=… godot --path plugins/godot-client/client     # screenshot every tab, then quit
##
## Beyond the web pages: the 3D view is native (orbit, layer filter, F12
## screenshots, review periods as frames), several Studios are remembered, and
## a paired device's token works for the parts a device may use.

const VizScene := preload("res://main.tscn")
const BG := Color(0.082, 0.082, 0.078)
const PANEL := Color(0.114, 0.114, 0.106)
const MUTED := Color(0.6, 0.6, 0.58)
const ACCENT := Color(0.435, 0.576, 1.0)
const BAD := Color(1.0, 0.478, 0.42)
const GOOD := Color(0.298, 0.761, 0.494)

# Phrases in Studio's messages -> where to go (the same places the web pages link to).
const LINKS := [
	["Setup\\s*(->|→)\\s*Build( llama\\.cpp)?", "job:build_llama"],
	["Setup\\s*(->|→)\\s*Hardware", "web:/setup#hardware"],
	["Setup\\s*(->|→)\\s*Requirements", "tab:Setup"],
	["Settings\\s*(->|→)\\s*Workspaces", "tab:Settings"],
	["Settings\\s*(->|→)\\s*Models per subject", "tab:Settings"],
	["Settings\\s*(->|→)\\s*Tool approval", "tab:Settings"],
	["Jobs\\s*(->|→)\\s*Evaluate", "job:testqa"],
	["Jobs\\s*(->|→)\\s*Review", "job:review_testqa"],
	["Studio\\s*(->|→)\\s*Models", "tab:Models"],
	["Studio\\s*(->|→)\\s*Tools", "web:/#tools"],
]
const OWNER_TABS := ["Jobs", "Projects", "Review", "Features", "Settings", "Setup", "Lab"]
# Tabs grouped as in the web sidebar: each group is one tab holding its pages as sub-tabs.
const GROUPS := {"Tasks": ["Jobs", "Projects"], "Studio": ["Features", "Review", "3D", "Lab"]}

var studio: StudioClient
var tabs: TabContainer
var conn_label: RichTextLabel
var viz
var specs: Dictionary = {}
var job_defaults: Dictionary = {}
var models: Array = []
var pages: Dictionary = {}          # tab name -> its root container
var groups: Dictionary = {}         # group name -> its TabContainer
var ui: Dictionary = {}             # named widgets


func _ready() -> void:
	studio = StudioClient.new()
	add_child(studio)
	studio.changed.connect(_on_connection)
	set_anchors_preset(Control.PRESET_FULL_RECT)
	var bg := ColorRect.new()
	bg.color = BG
	bg.set_anchors_preset(Control.PRESET_FULL_RECT)
	add_child(bg)
	var root := VBoxContainer.new()
	root.set_anchors_preset(Control.PRESET_FULL_RECT)
	root.add_theme_constant_override("separation", 0)
	add_child(root)
	root.add_child(_header())
	tabs = TabContainer.new()
	tabs.size_flags_vertical = Control.SIZE_EXPAND_FILL
	root.add_child(tabs)
	_build_connect()
	_build_chat()
	_build_models()
	_build_jobs()
	_build_projects()
	_build_features()
	_build_review()
	_build_3d()
	_build_lab()
	_build_settings()
	_build_setup()
	tabs.tab_changed.connect(_on_tab)
	_auto_connect()


# ---------------------------------------------------------------- widgets

func _page(name: String, scroll := true) -> VBoxContainer:
	var m := MarginContainer.new()
	m.name = name
	for side in ["left", "right", "top", "bottom"]:
		m.add_theme_constant_override("margin_" + side, 14)
	var box := VBoxContainer.new()
	box.add_theme_constant_override("separation", 8)
	box.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	if scroll:
		var sc := ScrollContainer.new()
		sc.horizontal_scroll_mode = ScrollContainer.SCROLL_MODE_DISABLED
		sc.add_child(box)
		m.add_child(sc)
	else:
		box.size_flags_vertical = Control.SIZE_EXPAND_FILL
		m.add_child(box)
	_parent_for(name).add_child(m)
	pages[name] = box
	return box


## The container a page goes in: its group's sub-tabs, or the top row.
func _parent_for(name: String) -> TabContainer:
	for g in GROUPS:
		if name in GROUPS[g]:
			if not groups.has(g):
				var t := TabContainer.new()
				t.name = g
				t.size_flags_vertical = Control.SIZE_EXPAND_FILL
				t.tab_changed.connect(func(_i):
					if tabs.get_current_tab_control() == t:
						_on_tab(tabs.current_tab))
				tabs.add_child(t)
				groups[g] = t
			return groups[g]
	return tabs


## The page showing under top tab i (a group shows its current sub-tab).
func _page_name(i: int) -> String:
	var t := tabs.get_tab_title(i)
	if groups.has(t):
		var g: TabContainer = groups[t]
		return g.get_tab_title(g.current_tab)
	return t


func _h(text: String, size := 17) -> Label:
	var l := Label.new()
	l.text = text
	l.add_theme_font_size_override("font_size", size)
	return l


func _note(text := "") -> RichTextLabel:
	var r := RichTextLabel.new()
	r.bbcode_enabled = true
	r.fit_content = true
	r.scroll_active = false
	r.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	r.add_theme_color_override("default_color", MUTED)
	r.add_theme_font_size_override("normal_font_size", 13)
	r.meta_clicked.connect(_on_link)
	r.text = text
	return r


func _btn(text: String, cb: Callable) -> Button:
	var b := Button.new()
	b.text = text
	b.pressed.connect(cb)
	return b


func _row(children: Array = []) -> HBoxContainer:
	var h := HBoxContainer.new()
	h.add_theme_constant_override("separation", 8)
	for c in children:
		h.add_child(c)
	return h


func _line(placeholder := "", text := "") -> LineEdit:
	var e := LineEdit.new()
	e.placeholder_text = placeholder
	e.text = text
	e.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	return e


func _option(items: Array, selected := "") -> OptionButton:
	var o := OptionButton.new()
	for it in items:
		o.add_item(str(it))
		o.set_item_metadata(o.item_count - 1, it)
		if str(it) == selected:
			o.select(o.item_count - 1)
	return o


func _sel(o: OptionButton):
	return o.get_item_metadata(o.selected) if o.selected >= 0 else null


func _card(parent: Control, title: String) -> VBoxContainer:
	var p := PanelContainer.new()
	var sb := StyleBoxFlat.new()
	sb.bg_color = PANEL
	sb.set_corner_radius_all(8)
	sb.content_margin_left = 12
	sb.content_margin_right = 12
	sb.content_margin_top = 10
	sb.content_margin_bottom = 10
	p.add_theme_stylebox_override("panel", sb)
	var v := VBoxContainer.new()
	v.add_theme_constant_override("separation", 6)
	p.add_child(v)
	if title != "":
		v.add_child(_h(title, 15))
	parent.add_child(p)
	return v


func _esc(s) -> String:
	return str(s).replace("[", "(").replace("]", ")")


## Studio's message text with its section references as links.
func _linkify(text: String) -> String:
	var out := _esc(text)
	for pair in LINKS:
		var rx := RegEx.new()
		rx.compile("(" + pair[0] + ")")
		var m := rx.search(out)
		if m:
			out = out.substr(0, m.get_start()) + "[url=%s]%s[/url]" % [pair[1], m.get_string()] + out.substr(m.get_end())
	return out


func _err(text: String) -> String:
	return "[color=#%s]%s[/color]" % [BAD.to_html(false), _linkify(text)]


## Links: tab:<name>, job:<kind>[?field=value&…], web:/path (opens Studio in the browser), 3d:<url>.
func _on_link(meta) -> void:
	var m := str(meta)
	if m.begins_with("tab:"):
		_go(m.substr(4))
	elif m.begins_with("job:"):
		var q := m.substr(4).split("?")
		var pre := {}
		if q.size() > 1:
			for kv in q[1].split("&"):
				var p := kv.split("=")
				if p.size() == 2:
					pre[p[0]] = p[1].uri_decode()
		_open_job(q[0], pre)
	elif m.begins_with("web:"):
		OS.shell_open(studio.base + m.substr(4))
	elif m.begins_with("3d:"):
		_open_3d(m.substr(3))
	elif m == "pstart:force":
		_pact("start", {"force": true})
	elif m.begins_with("restart:"):
		studio.post_json("/api/restart", {})


func _go(tab: String) -> void:
	for i in tabs.get_tab_count():
		if tabs.get_tab_title(i) == tab:
			tabs.current_tab = i
			return
	for g in groups:
		var gc: TabContainer = groups[g]
		for j in gc.get_tab_count():
			if gc.get_tab_title(j) == tab:
				gc.current_tab = j          # first, so the group opens on this page
				_go(g)
				return


# ---------------------------------------------------------------- header + connect

func _header() -> Control:
	var p := PanelContainer.new()
	var sb := StyleBoxFlat.new()
	sb.bg_color = PANEL
	sb.content_margin_left = 14
	sb.content_margin_right = 14
	sb.content_margin_top = 6
	sb.content_margin_bottom = 6
	p.add_theme_stylebox_override("panel", sb)
	var h := _row()
	var t := _h("OmniCortex", 16)
	h.add_child(t)
	conn_label = _note("not connected")
	h.add_child(conn_label)
	h.add_child(_btn("Open in browser", _open_browser))
	p.add_child(h)
	return p


func _open_browser() -> void:
	if studio.base != "":
		OS.shell_open(studio.base + "/")


func _build_connect() -> void:
	var v := _page("Connect")
	var c := _card(v, "Connect to a Studio")
	c.add_child(_note("Studio serves the models on the computer with the GPU; this client can run on any other computer on your network, so the 3D views never take GPU memory from the model. On the GPU machine start Studio for the network (see docs/SECURITY.md: a token and TLS), then connect here with the [b]owner token[/b] for everything, or [b]pair[/b] this computer for models, chat and checks."))
	ui.url = _line("Studio address, e.g. https://gpu-box.lan:7870", "http://127.0.0.1:7870")
	ui.token = _line("owner token (empty if Studio has none)")
	ui.token.secret = true
	ui.cert = _line("Studio's certificate (PEM) to trust, for a self-signed Studio")
	ui.cname = _line("name for this connection", "studio")
	c.add_child(_row([_h("Address", 13), ui.url]))
	c.add_child(_row([_h("Token", 13), ui.token]))
	c.add_child(_row([_h("Certificate", 13), ui.cert]))
	c.add_child(_row([_h("Save as", 13), ui.cname, _btn("Connect", _do_connect)]))
	var pc := _card(v, "Or pair this computer")
	pc.add_child(_note("In Studio (on the GPU machine) open [b]Link[/b], start pairing, and paste the link or code here. A paired computer may use models, chat and reply checks; jobs, projects and settings need the owner token."))
	ui.paircode = _line("pairing link or code")
	pc.add_child(_row([ui.paircode, _btn("Pair", _do_pair)]))
	ui.connmsg = _note()
	v.add_child(ui.connmsg)
	var sc := _card(v, "Saved Studios")
	ui.saved = VBoxContainer.new()
	sc.add_child(ui.saved)
	_list_saved()


func _list_saved() -> void:
	for ch in ui.saved.get_children():
		ch.queue_free()
	var rows: Array = studio.saved()
	if rows.is_empty():
		ui.saved.add_child(_note("none yet"))
	for s in rows:
		var r := _row([_h(s.name, 13), _note(s.url)])
		r.add_child(_btn("Use", _use_saved.bind(s)))
		r.add_child(_btn("Forget", _forget.bind(s.name)))
		ui.saved.add_child(r)


func _forget(name: String) -> void:
	studio.forget(name)
	_list_saved()


func _use_saved(s: Dictionary) -> void:
	ui.url.text = s.url
	ui.token.text = s.token
	ui.cert.text = s.cert
	ui.cname.text = s.name
	_do_connect()


func _auto_connect() -> void:
	var env_url := OS.get_environment("NS_STUDIO")
	if env_url != "":
		ui.url.text = env_url
		ui.token.text = OS.get_environment("NS_TOKEN")
		_do_connect()
		return
	for s in studio.saved():
		if s.last:
			_use_saved(s)
			return


func _do_connect() -> void:
	ui.connmsg.text = "connecting…"
	studio.connect_to(ui.url.text, ui.token.text, ui.cert.text, func(ok):
		if ok:
			studio.save_connection(ui.cname.text if ui.cname.text != "" else "studio", studio.base, studio.token, studio.cert_path)
			_list_saved()
			ui.connmsg.text = "[color=#%s]connected[/color] as %s" % [GOOD.to_html(false), studio.role]
			var start := OS.get_environment("NS_TAB")
			_go(start if start != "" else "Chat")
			if OS.get_environment("NS_SHOTS") != "":
				_capture_all(OS.get_environment("NS_SHOTS"))
		else:
			ui.connmsg.text = _err(studio.last_error))


func _do_pair() -> void:
	ui.connmsg.text = "pairing…"
	studio.pair(ui.url.text, ui.paircode.text, "OmniCortex Godot on " + OS.get_name(), func(ok, msg):
		if ok:
			ui.token.text = studio.token
			_do_connect()
		ui.connmsg.text = msg if ok else _err(msg))


func _on_connection() -> void:
	if studio.role == "":
		conn_label.text = "not connected" + (" · " + _err(studio.last_error) if studio.last_error != "" else "")
	else:
		var st := studio.status
		conn_label.text = "%s · %s · model: %s · workspace %s" % [studio.base, studio.role,
			str(st.get("id")) if st.get("id") else "none loaded", str(st.get("workspace", "?"))]
	var device := studio.role == "device"
	for g in groups:
		var gc: TabContainer = groups[g]
		for j in gc.get_tab_count():
			gc.set_tab_disabled(j, device and gc.get_tab_title(j) in OWNER_TABS)
		if gc.is_tab_disabled(gc.current_tab):
			for j in gc.get_tab_count():
				if not gc.is_tab_disabled(j):
					gc.current_tab = j
					break
	for i in tabs.get_tab_count():
		var t := tabs.get_tab_title(i)
		var owner_only: bool = t in OWNER_TABS or (groups.has(t) and GROUPS[t].all(func(p): return p in OWNER_TABS))
		tabs.set_tab_disabled(i, studio.role == "" and t != "Connect" or (device and owner_only))
	if studio.role != "":
		_load_models()
		studio.get_json("/api/jobs/defaults", func(c, d): job_defaults = d if c == 200 else {})
		_load_widget_addons()


func _on_tab(i: int) -> void:
	match _page_name(i):
		"Models": _load_models()
		"Features": _load_features()
		"Jobs": _load_jobs()
		"Projects": _load_projects()
		"Review": _load_review()
		"3D": _load_traces()
		"Settings": _load_settings()
		"Setup": _load_setup()
		"Lab": _load_lab()


# ---------------------------------------------------------------- chat

var chat_msgs: Array = []


func _build_chat() -> void:
	var v := _page("Chat", false)
	ui.chatmodel = OptionButton.new()
	ui.chatcheck = CheckBox.new()
	ui.chatcheck.text = "check each reply"
	v.add_child(_row([_h("Model", 13), ui.chatmodel, ui.chatcheck, _btn("New chat", _new_chat)]))
	ui.chatlog = RichTextLabel.new()
	ui.chatlog.bbcode_enabled = true
	ui.chatlog.scroll_following = true
	ui.chatlog.size_flags_vertical = Control.SIZE_EXPAND_FILL
	ui.chatlog.meta_clicked.connect(_on_link)
	ui.chatlog.add_theme_font_size_override("normal_font_size", 15)
	v.add_child(ui.chatlog)
	ui.chatin = TextEdit.new()
	ui.chatin.custom_minimum_size = Vector2(0, 80)
	ui.chatin.placeholder_text = "Message (Ctrl+Enter sends)"
	ui.chatin.gui_input.connect(func(e):
		if e is InputEventKey and e.pressed and e.keycode == KEY_ENTER and e.ctrl_pressed:
			_send()
			ui.chatin.accept_event())
	v.add_child(_row([ui.chatin, _btn("Send", _send)]))
	ui.chatin.size_flags_horizontal = Control.SIZE_EXPAND_FILL


func _new_chat() -> void:
	chat_msgs = []
	ui.chatlog.text = ""


func _send() -> void:
	var text: String = ui.chatin.text.strip_edges()
	if text == "":
		return
	ui.chatin.text = ""
	chat_msgs.append({"role": "user", "content": text})
	ui.chatlog.append_text("\n[b]you[/b]  %s\n" % _esc(text))
	var model = _sel(ui.chatmodel)
	var body := {"messages": chat_msgs, "stream": false}
	if model != null and model != "(loaded model)":
		body.model = model
	ui.chatlog.append_text("[color=#%s]…[/color]\n" % MUTED.to_html(false))
	studio.post_json("/v1/chat/completions", body, func(code, d):
		if code != 200:
			ui.chatlog.append_text(_err(str(d.get("error", "HTTP %d" % code))) + "\n")
			return
		var msg: Dictionary = d.choices[0].message
		var reply := str(msg.get("content", ""))
		chat_msgs.append({"role": "assistant", "content": reply})
		var n := chat_msgs.size() - 1
		var route := ""
		if d.has("route") and typeof(d.route) == TYPE_DICTIONARY and d.route.get("reason"):
			route = "  [color=#%s](%s)[/color]" % [MUTED.to_html(false), _esc(d.route.reason)]
		ui.chatlog.append_text("[b]%s[/b]%s\n%s\n[url=check:%d]check this reply[/url]\n" % [_esc(d.get("model", "model")), route, _esc(reply), n])
		if ui.chatcheck.button_pressed:
			_check(n))


func _check(n: int) -> void:
	var model = _sel(ui.chatmodel)
	var msgs := chat_msgs.slice(0, n)
	ui.chatlog.append_text("[color=#%s]checking reply %d (one extra pass over it)…[/color]\n" % [MUTED.to_html(false), n])
	var body := {"messages": msgs, "text": chat_msgs[n].content}
	if model != null and model != "(loaded model)" and model != "auto":
		body.model = model
	studio.post_json("/api/trace", body, func(code, d):
		if code != 200:
			var links := ""
			for l in d.get("links", []):
				links += "  [url=web:%s]%s[/url]" % [l.href, _esc(l.label)]
			ui.chatlog.append_text("check failed: " + _err(str(d.get("error"))) + links + "\n")
			return
		var nflag: int = d.flagged.size()
		var verdict := "[color=#%s]%d of %d tokens flagged[/color]" % [BAD.to_html(false), nflag, d.prob.size()] if nflag else "[color=#%s]no tokens flagged[/color]" % GOOD.to_html(false)
		var label := ""
		if d.get("review"):
			label = "  was it right? [url=label:%s|%s|correct]right[/url] / [url=label:%s|%s|wrong]wrong[/url]" % [d.review.model, d.review.id, d.review.model, d.review.id]
		ui.chatlog.append_text("risk: %s · peak %d%% · [url=3d:%s/%s]open in 3D[/url]%s\n" % [verdict, int(d.max * 100), studio.base, d.url, label]))


func _chat_link(meta) -> void:
	var m := str(meta)
	if m.begins_with("check:"):
		_check(int(m.substr(6)))
	elif m.begins_with("label:"):
		var p := m.substr(6).split("|")
		studio.post_json("/api/review/label", {"model": p[0], "id": p[1], "verdict": p[2]}, func(c, d):
			ui.chatlog.append_text(("[color=#%s]marked %s[/color]\n" % [GOOD.to_html(false), p[2]]) if c == 200 else _err(str(d.get("error"))) + "\n")
		)
	else:
		_on_link(meta)


# ---------------------------------------------------------------- models

func _build_models() -> void:
	var v := _page("Models")
	ui.modelmsg = _note()
	v.add_child(_row([_btn("Refresh", _load_models), _btn("Unload", func(): studio.post_json("/api/unload", {}, func(c, d): _load_models())), ui.modelmsg]))
	ui.modellist = VBoxContainer.new()
	v.add_child(ui.modellist)
	# chat links go through _chat_link (checks and labels), the rest through _on_link
	ui.chatlog.meta_clicked.disconnect(_on_link)
	ui.chatlog.meta_clicked.connect(_chat_link)


func _load_models() -> void:
	studio.get_json("/api/models", func(code, d):
		if code != 200:
			ui.modelmsg.text = _err(str(d.get("error", code)))
			return
		models = d
		var cur = _sel(ui.chatmodel)
		ui.chatmodel.clear()
		for id in ["(loaded model)", "auto"]:
			ui.chatmodel.add_item(id)
			ui.chatmodel.set_item_metadata(ui.chatmodel.item_count - 1, id)
		for m in models:
			ui.chatmodel.add_item(m.id)
			ui.chatmodel.set_item_metadata(ui.chatmodel.item_count - 1, m.id)
			if m.id == cur:
				ui.chatmodel.select(ui.chatmodel.item_count - 1)
		# Nothing chosen yet and nothing loaded: auto when it can route, else the first model (it loads on demand).
		if cur == null and not studio.status.get("id") and models.size():
			ui.chatmodel.select(1 if studio.status.get("auto_ready") else 2)
		for ch in ui.modellist.get_children():
			ch.queue_free()
		if models.is_empty():
			ui.modellist.add_child(_note("No .gguf files in the workspace's models folders: " + _linkify("Settings -> Workspaces") + "."))
		for m in models:
			var c := _card(ui.modellist, m.id)
			var s = m.get("stats")
			var bits := ["%.1f GiB" % (float(m.size) / 1073741824.0)]
			for k in ["quant", "arch"]:
				if m.get(k):
					bits.append(str(m[k]))
			var line := " · ".join(bits)
			if s and s.get("graded", {}).get("n", 0) > 0:
				var g = s.graded
				line += " · right %d%% · hallucinated %d%% · n %d" % [int(g.accuracy * 100), int(g.hallucination_rate * 100), int(g.n)]
			c.add_child(_note(line))
			if s and not s.get("eligible", false):
				var ep := "m=%s/v1@%s" % [studio.base, m.id]
				c.add_child(_note("no graded results: auto routing skips it · [url=job:testqa?endpoint=%s&publish_stats=%s&out=runs/testqa-%s.json]grade it with TestQA[/url]" % [ep.uri_encode(), studio.base.uri_encode(), str(m.id).replace("/", "_")]))
			c.add_child(_row([_btn("Load", _load_model.bind(m)), _btn("Use in chat", _pick_chat.bind(m.id))])))


func _load_model(m: Dictionary) -> void:
	var id: String = m.id
	ui.modelmsg.text = "loading %s…" % id
	studio.post_json("/api/load", {"path": m.path, "settings": m.get("settings", {})}, func(c, d):
		ui.modelmsg.text = ("[color=#%s]loaded %s[/color]" % [GOOD.to_html(false), id]) if c == 200 and d.get("ok", true) else _err(str(d.get("message", d.get("error", c))))
		studio.connect_to(studio.base, studio.token, studio.cert_path))


func _pick_chat(id: String) -> void:
	for i in ui.chatmodel.item_count:
		if ui.chatmodel.get_item_metadata(i) == id:
			ui.chatmodel.select(i)
	_go("Chat")


# ---------------------------------------------------------------- jobs

var cur_job := ""


func _build_jobs() -> void:
	var v := _page("Jobs")
	var c := _card(v, "New job")
	ui.jobkind = OptionButton.new()
	ui.jobkind.item_selected.connect(func(_i): _job_form({}))
	c.add_child(ui.jobkind)
	ui.jobhelp = _note()
	c.add_child(ui.jobhelp)
	ui.jobform = GridContainer.new()
	ui.jobform.columns = 2
	c.add_child(ui.jobform)
	ui.jobmsg = _note()
	c.add_child(_row([_btn("Run", _run_job), ui.jobmsg]))
	var l := _card(v, "Jobs")
	ui.joblist = VBoxContainer.new()
	l.add_child(ui.joblist)
	ui.joblog = TextEdit.new()
	ui.joblog.editable = false
	ui.joblog.custom_minimum_size = Vector2(0, 260)
	l.add_child(_row([_btn("Refresh", _load_jobs), _btn("Cancel selected", func(): studio.post_json("/api/jobs/cancel", {"id": cur_job}, func(c2, d): _load_jobs()))]))
	l.add_child(ui.joblog)


func _load_jobs() -> void:
	if specs.is_empty():
		studio.get_json("/api/jobs/specs", func(code, d):
			if code != 200:
				ui.jobmsg.text = _err(str(d.get("error", code)))
				return
			specs = d
			ui.jobkind.clear()
			var groups := {}
			for k in specs:
				groups.get_or_add(specs[k].group, []).append(k)
			for g in groups:
				ui.jobkind.add_separator(g)
				for k in groups[g]:
					ui.jobkind.add_item(specs[k].title)
					ui.jobkind.set_item_metadata(ui.jobkind.item_count - 1, k)
			_job_form({}))
	studio.get_json("/api/jobs", func(code, d):
		if code != 200:
			return
		for ch in ui.joblist.get_children():
			ch.queue_free()
		for j in d.slice(0, 15):
			var b := _btn("%s  %s  %s" % [j.status, j.title, Time.get_datetime_string_from_unix_time(int(j.started))], func(): _show_job(j.id))
			b.alignment = HORIZONTAL_ALIGNMENT_LEFT
			ui.joblist.add_child(b))


func _show_job(id: String) -> void:
	cur_job = id
	studio.get_json("/api/jobs/%s/log" % id, func(c, d):
		ui.joblog.text = str(d.get("log", d)) if c == 200 else str(d.get("error", c))
		ui.joblog.scroll_vertical = 1e9)


func _open_job(kind: String, prefill: Dictionary) -> void:
	_go("Jobs")
	if specs.is_empty():
		_load_jobs()
		await get_tree().create_timer(1.0).timeout
	for i in ui.jobkind.item_count:
		if ui.jobkind.get_item_metadata(i) == kind:
			ui.jobkind.select(i)
	_job_form(prefill)
	if not prefill.is_empty():
		ui.jobmsg.text = "filled in from the link: check the values, then Run"


func _suggest(f: Dictionary) -> String:
	var h := str(f.get("help", "")).to_lower()
	if f.name == "gguf" and not "quant" in h and job_defaults.get("gguf"):
		return str(job_defaults.gguf)
	if (f.name == "binary" or f.name == "cett") and job_defaults.get("cett"):
		return str(job_defaults.cett)
	if f.name == "server" and "llama-server" in h and job_defaults.get("server"):
		return str(job_defaults.server)
	if f.name == "llama" and job_defaults.get("llama"):
		return str(job_defaults.llama)
	if f.name == "record" and "trace session" in h and job_defaults.get("traces"):
		return str(job_defaults.traces)
	return ""


func _job_form(prefill: Dictionary) -> void:
	for ch in ui.jobform.get_children():
		ch.queue_free()
	var kind = _sel(ui.jobkind)
	if kind == null or not specs.has(kind):
		return
	var s: Dictionary = specs[kind]
	ui.jobhelp.text = _esc(s.help) + ("  · runs in %s" % job_defaults.root if job_defaults.get("root") else "")
	for f in s.fields:
		var lab := Label.new()
		lab.text = str(f.name).replace("_", " ") + (" *" if f.required else "")
		ui.jobform.add_child(lab)
		var w: Control
		var d = f.default
		if d is Array:
			d = " ".join(d)
		var val := str(prefill.get(f.name, "" if d == null else str(d)))
		if val == "":
			val = _suggest(f)
		if f.kind == "bool":
			w = CheckBox.new()
			w.button_pressed = val in ["1", "true", "True"]
		elif f.choices != null:
			w = _option(f.choices, val)
		else:
			w = _line(str(f.get("help", "")), val)
		w.name = "f_" + f.name
		w.size_flags_horizontal = Control.SIZE_EXPAND_FILL
		ui.jobform.add_child(w)


func _run_job() -> void:
	var kind = _sel(ui.jobkind)
	var vals := {}
	for f in specs[kind].fields:
		var w = ui.jobform.get_node("f_" + f.name)
		if w is CheckBox:
			vals[f.name] = w.button_pressed
		elif w is OptionButton:
			vals[f.name] = _sel(w)
		else:
			vals[f.name] = w.text.strip_edges()
	studio.post_json("/api/jobs", {"kind": kind, "values": vals}, func(c, d):
		if c == 200:
			ui.jobmsg.text = "[color=#%s]started[/color]" % GOOD.to_html(false)
			_load_jobs()
			_show_job(d.id)
		else:
			ui.jobmsg.text = _err(str(d.get("error", c))))


# ---------------------------------------------------------------- projects

var cur_project := ""


func _build_projects() -> void:
	var v := _page("Projects")
	var c := _card(v, "New project")
	ui.ptitle = _line("title")
	ui.ptext = TextEdit.new()
	ui.ptext.custom_minimum_size = Vector2(0, 110)
	ui.ptext.placeholder_text = "Paste the project's notes, task list or README"
	ui.prepo = _line("local checkout to survey (optional), on the Studio machine")
	c.add_child(ui.ptitle)
	c.add_child(ui.ptext)
	c.add_child(ui.prepo)
	ui.pmsg = _note()
	c.add_child(_row([_btn("Analyze into a draft plan", _new_project), ui.pmsg]))
	ui.plist = _card(v, "Projects")
	ui.pview = VBoxContainer.new()
	v.add_child(ui.pview)


func _new_project() -> void:
	studio.post_json("/api/projects", {"title": ui.ptitle.text, "text": ui.ptext.text, "repo": ui.prepo.text}, func(c, d):
		if c == 200:
			cur_project = d.id
			_load_projects()
		else:
			ui.pmsg.text = _err(str(d.get("error", c))))


func _load_projects() -> void:
	studio.get_json("/api/projects", func(c, d):
		if c != 200:
			return
		for ch in ui.plist.get_children().slice(1):
			ch.queue_free()
		for p in d.projects:
			var b := _btn("%s  (%s)" % [p.title, p.status], _pick_project.bind(p.id))
			b.alignment = HORIZONTAL_ALIGNMENT_LEFT
			ui.plist.add_child(b)
		if cur_project == "" and d.projects.size():
			cur_project = d.projects[0].id
		if cur_project != "":
			_show_project())


func _pick_project(id: String) -> void:
	cur_project = id
	_show_project()


func _pact(action: String, body := {}) -> void:
	studio.post_json("/api/projects/%s/%s" % [cur_project, action], body, func(c, d):
		if c == 409 and d.has("missing"):
			ui.pmsg.text = _err(str(d.error)) + "  [url=pstart:force]start anyway[/url]"
		elif c != 200:
			ui.pmsg.text = _err(str(d.get("error", c)))
		_show_project())


func _show_project() -> void:
	studio.get_json("/api/projects/" + cur_project, func(c, p):
		for ch in ui.pview.get_children():
			ch.queue_free()
		if c != 200:
			return
		var card := _card(ui.pview, "%s  (%s, v%d)" % [p.title, p.status, int(p.version)])
		var acts := _row()
		if p.status == "draft":
			acts.add_child(_btn("Approve plan", func(): _pact("approve")))
		if p.status in ["approved", "paused"]:
			acts.add_child(_btn("Start", func(): _pact("start")))
		if p.status == "running":
			acts.add_child(_btn("Pause", func(): _pact("pause")))
		acts.add_child(_btn("Check requirements", func(): studio.get_json("/api/projects/%s/requirements" % cur_project, func(_c, _d): _show_project(), 120.0)))
		card.add_child(acts)
		var rq: Dictionary = p.get("requirements", {})
		if rq.get("last"):
			var last: Dictionary = rq.last
			card.add_child(_note("requirements: %s%s" % [last.status, (" · missing " + ", ".join(last.missing)) if last.missing.size() else ""]))
		for t in p.tasks:
			if t.status == "dropped":
				continue
			var tc := _card(ui.pview, "%s  %s  [%s]" % [t.id, t.title, t.status])
			tc.add_child(_note(_esc(t.detail).left(400)))
			if t.status == "review" and t.attempts.size():
				var a = t.attempts[-1]
				var out := TextEdit.new()
				out.text = str(a.get("text", ""))
				out.editable = false
				out.custom_minimum_size = Vector2(0, 160)
				tc.add_child(out)
				var fb := _line("feedback for the next attempt")
				tc.add_child(_row([fb, _btn("Accept", func(): _pact("review", {"task": t.id, "accept": true, "feedback": fb.text})),
					_btn("Send back", func(): _pact("review", {"task": t.id, "accept": false, "feedback": fb.text}))]))
			for q in t.get("questions", []):
				if not q.get("answer"):
					var ans := _line(str(q.q))
					tc.add_child(_row([ans, _btn("Answer", func(): _pact("answer", {"task": t.id, "answers": [ans.text]}))])))


# ---------------------------------------------------------------- review

func _build_features() -> void:
	var v := _page("Features")
	v.add_child(_note("Feature dictionaries and emotion vectors found in the workspace. Reading, circuits and steering "
		+ "open in the browser; replies the decoding lab recorded with --watch-* play in Studio -> 3D with their "
		+ "emotion lanes and features."))
	v.add_child(_row([_btn("Record activations", func(): _open_job("feat_collect", {})),
		_btn("Train a dictionary", func(): _open_job("feat_train", {})),
		_btn("Build emotion vectors", func(): _open_job("emo_build", {})),
		_btn("Decoding lab", func(): _open_job("decode_lab", {})),
		_btn("Open Features", func(): OS.shell_open(studio.base + "/features"))]))
	ui.featlist = VBoxContainer.new()
	v.add_child(ui.featlist)


## JSON numbers arrive as floats: 12.0 -> "12".
func _num(v) -> String:
	if v == null:
		return "?"
	if (typeof(v) == TYPE_FLOAT) and v == floor(v):
		return str(int(v))
	return str(v)


func _load_features() -> void:
	studio.get_json("/api/features/list", func(c, d):
		for ch in ui.featlist.get_children():
			ch.queue_free()
		if c != 200:
			ui.featlist.add_child(_note(_err(str(d.get("error", c)))))
			return
		var dc := _card(ui.featlist, "Dictionaries (%d)" % d.dicts.size())
		if d.dicts.is_empty():
			dc.add_child(_note("None yet: record activations, then train one."))
		for x in d.dicts:
			var stats := "%s · %s %s.%s · %s features" % [_esc(str(x.get("model", "?")).get_file()), str(x.get("arch", "")),
				str(x.get("site", "")), _num(x.get("layer")), _num(x.get("n_features"))]
			if x.get("recovered") != null:
				stats += " · recovers %d%%" % int(round(float(x.recovered) * 100))
			if not x.get("interpreted"):
				stats += " · not described yet"
			var path: String = x.path
			dc.add_child(_row([_h(str(x.name), 14), _note(stats),
				_btn("Open", func(): OS.shell_open(studio.base + "/features#dicts?path=" + path.uri_encode()))]))
		var ec := _card(ui.featlist, "Emotion vectors (%d)" % d.emotions.size())
		if d.emotions.is_empty():
			ec.add_child(_note("None yet: write stories (Jobs -> Interpret), then build the vectors."))
		for x in d.emotions:
			var acc := "" if x.get("best_accuracy") == null else " · best layer %s, %d%% accurate" % [_num(x.best_layer), int(round(float(x.best_accuracy) * 100))]
			ec.add_child(_row([_h(str(x.name), 14), _note("%s · %s emotions%s" % [_esc(str(x.get("model", "?")).get_file()), _num(x.get("n")), acc]),
				_btn("Open", func(): OS.shell_open(studio.base + "/features#emotions"))]))
		if not d.graphs.is_empty():
			var gc := _card(ui.featlist, "Graphs (%d)" % d.graphs.size())
			for x in d.graphs:
				gc.add_child(_row([_h(str(x.name), 14), _note(_esc(str(x.get("prompt", "")).left(80))),
					_btn("Open", func(): OS.shell_open(studio.base + "/features#circuits"))])))


func _build_review() -> void:
	var v := _page("Review")
	var c := _card(v, "Neuron review")
	ui.rvmodel = OptionButton.new()
	ui.rvstat = _option(["association", "risk", "firing"])
	ui.rvby = _option(["week", "day", "month", "all"])
	ui.rvsubject = _line("subjects, comma-separated (empty: all)")
	c.add_child(_row([_h("Model", 13), ui.rvmodel, _h("Statistic", 13), ui.rvstat, _h("By", 13), ui.rvby]))
	c.add_child(_row([ui.rvsubject, _btn("Summary", _review_summary), _btn("Open 3D over time", _review_3d)]))
	ui.rvout = _note("Every checked reply is recorded; add TestQA runs with " + _linkify("Jobs -> Review") + ".")
	c.add_child(ui.rvout)


func _load_review() -> void:
	studio.get_json("/api/review", func(c, d):
		if c != 200:
			ui.rvout.text = _err(str(d.get("error", c)))
			return
		ui.rvmodel.clear()
		for m in d.models:
			ui.rvmodel.add_item("%s (%d)" % [m.model, int(m.observations)])
			ui.rvmodel.set_item_metadata(ui.rvmodel.item_count - 1, m.model)
		if d.models.is_empty():
			ui.rvout.text = "No observations yet: check replies in Chat, or [url=job:review_testqa]ingest a TestQA run[/url]."
		else:
			_review_summary())


func _review_body() -> Dictionary:
	var subj := []
	for s in ui.rvsubject.text.split(","):
		if s.strip_edges() != "":
			subj.append(s.strip_edges())
	return {"model": _sel(ui.rvmodel), "stat": _sel(ui.rvstat), "by": _sel(ui.rvby), "subjects": subj, "top": 12}


func _review_summary() -> void:
	studio.post_json("/api/review/summary", _review_body(), func(c, r):
		if c != 200:
			ui.rvout.text = _err(str(r.get("error", c)))
			return
		var t := "[b]%d replies[/b] · %d right · %d wrong%s\n" % [int(r.n), int(r.n_right), int(r.n_wrong),
			(" · error rate %d%%" % int(r.error_rate * 100)) if r.error_rate != null else ""]
		if r.note != "":
			t += _esc(r.note) + "\n"
		for row in r.top:
			t += "layer %d bin %d  %+.2f\n" % [int(row.layer), int(row.bin), float(row.value)]
		for b in r.timeline:
			t += "%s  %d replies%s\n" % [b.label, int(b.n), ("  error rate %d%%" % int(b.error_rate * 100)) if b.error_rate != null else ""]
		ui.rvout.text = t)


func _review_3d() -> void:
	studio.post_json("/api/review/view", _review_body(), func(c, r):
		if c == 200:
			_open_3d(studio.base + r.url)
		else:
			ui.rvout.text = _err(str(r.get("error", c))))


# ---------------------------------------------------------------- 3D

func _build_3d() -> void:
	var m := MarginContainer.new()
	m.name = "3D"
	m.add_theme_constant_override("margin_left", 10)
	_parent_for("3D").add_child(m)
	var h := HBoxContainer.new()
	m.add_child(h)
	var side := VBoxContainer.new()
	side.custom_minimum_size = Vector2(300, 0)
	h.add_child(side)
	side.add_child(_h("Traces", 15))
	ui.traces = ItemList.new()
	ui.traces.size_flags_vertical = Control.SIZE_EXPAND_FILL
	ui.traces.item_activated.connect(func(i): _open_3d(studio.base + "/viz/%s/" % ui.traces.get_item_metadata(i)))
	side.add_child(ui.traces)
	side.add_child(_note("Double-click a checked reply or a decoding-lab recording. Review views open from Studio -> Review."))
	side.add_child(_h("Layers", 13))
	ui.lo = SpinBox.new()
	ui.hi = SpinBox.new()
	ui.hi.max_value = 999
	ui.hi.value = 999
	for sp in [ui.lo, ui.hi]:
		sp.value_changed.connect(_on_layers)
	side.add_child(_row([ui.lo, _h("to", 13), ui.hi]))
	ui.shot = _note()
	side.add_child(_row([_btn("Play/pause", _toggle_play), _btn("Screenshot", _shot)]))
	side.add_child(ui.shot)
	side.add_child(_note("Space play/pause · ←/→ step · N next flagged · F12 screenshot · drag orbit, Shift+drag pan, wheel zoom · numpad 1/3/7 views, 5 ortho, Home frame, S sway · click the jewel to hide the emotion graph ([url=web:/settings#widgets]widget settings[/url])"))
	var vc := SubViewportContainer.new()
	vc.stretch = true
	vc.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	vc.size_flags_vertical = Control.SIZE_EXPAND_FILL
	h.add_child(vc)
	var sv := SubViewport.new()
	sv.handle_input_locally = true
	vc.add_child(sv)
	viz = VizScene.instantiate()
	viz.standalone = false
	sv.add_child(viz)
	# the 3D view's search line asks the model through Studio (">question"), with the reply as context
	viz.search.can_chat = true
	viz.search.ask.connect(_ask_from_3d)


func _ask_from_3d(q: String) -> void:
	var reply := "".join(viz.meta.get("labels", []))
	var msgs := [{"role": "system", "content": "The user is looking at this reply in a 3D view of the model's activations:\n" + reply.left(6000)},
		{"role": "user", "content": q}]
	studio.post_json("/v1/chat/completions", {"messages": msgs, "stream": false, "max_tokens": 300}, func(code, d):
		if code != 200:
			viz.search.show_answer("error: " + str(d.get("error", code)))
			return
		viz.search.show_answer(str(d.choices[0].message.get("content", ""))))


## Widget add-ons (the jewel-styles add-on, …): Studio lists their GDScript; each is compiled here and
## handed to the 3D view's jewel. Only add-ons the owner turned on are listed.
func _load_widget_addons() -> void:
	studio.get_json("/api/addons/widgets", func(c, d):
		if c != 200 or typeof(d) != TYPE_DICTIONARY:
			return
		for item in d.get("gd", []):
			studio.call_api(HTTPClient.METHOD_GET, str(item.url), null, func(code, bytes):
				if code != 200:
					return
				var sc := GDScript.new()
				sc.source_code = bytes.get_string_from_utf8()
				if sc.reload() != OK:
					push_warning("widget add-on %s did not compile" % str(item.url))
					return
				var obj = sc.new()
				if viz and viz.jewel and "STYLES" in obj:
					viz.jewel.add_styles(obj), true))


func _on_layers(_v: float) -> void:
	viz.set_layer_range(int(ui.lo.value), int(ui.hi.value))


func _toggle_play() -> void:
	viz.playing = not viz.playing


func _shot() -> void:
	ui.shot.text = viz.screenshot()


func _load_traces() -> void:
	studio.get_json("/api/traces", func(c, d):
		if c != 200:
			return
		ui.traces.clear()
		for t in d.traces:
			ui.traces.add_item("%s  %s" % [t.id, str(t.get("model", ""))])
			ui.traces.set_item_metadata(ui.traces.item_count - 1, t.id))


func _open_3d(url: String) -> void:
	_go("3D")
	viz.open(url, studio.token)


# ---------------------------------------------------------------- settings

func _build_settings() -> void:
	var v := _page("Settings")
	var w := _card(v, "Workspaces")
	w.add_child(_note("Where models, llama.cpp, pipeline data and Studio's state live on the Studio machine. Edit the folders in the browser ([url=web:/settings#workspaces]Settings → Workspaces[/url]); switch here."))
	ui.wslist = VBoxContainer.new()
	w.add_child(ui.wslist)
	var s := _card(v, "Models per subject")
	s.add_child(_note("With model auto, each subject goes to the model measured best for it, unless you pick one."))
	ui.subjgrid = GridContainer.new()
	ui.subjgrid.columns = 3
	s.add_child(ui.subjgrid)
	ui.subjmsg = _note()
	s.add_child(_row([_btn("Save", _save_subjects), ui.subjmsg]))
	var t := _card(v, "Tool approval")
	t.add_child(_note("Quick settings by kind of operation; ordered rules with argument regexes are edited in the browser ([url=web:/settings#tools]Settings → Tool approval[/url])."))
	ui.catgrid = GridContainer.new()
	ui.catgrid.columns = 2
	t.add_child(ui.catgrid)
	ui.rulesnote = _note()
	t.add_child(ui.rulesnote)
	ui.toolmsg = _note()
	t.add_child(_row([_btn("Save quick settings", _save_cats), ui.toolmsg]))


var settings: Dictionary = {}


func _load_settings() -> void:
	studio.get_json("/api/workspaces", func(c, d):
		for ch in ui.wslist.get_children():
			ch.queue_free()
		if c != 200:
			return
		for ws in d.workspaces:
			var r := _row([_h(("● " if ws.active else "○ ") + ws.name, 13), _note(_esc(ws.resolved.root) + " · " + _esc(", ".join(ws.resolved.models_dirs)))])
			if not ws.active:
				r.add_child(_btn("Use", _use_ws.bind(ws.name)))
			ui.wslist.add_child(r))
	studio.get_json("/api/settings", func(c, d):
		if c != 200:
			return
		settings = d
		for ch in ui.subjgrid.get_children():
			ch.queue_free()
		for subj in d.subjects:
			var row: Dictionary = d.subjects[subj]
			ui.subjgrid.add_child(_h(subj, 13))
			ui.subjgrid.add_child(_note(_esc(row.best) if row.best else "no measured best yet"))
			var o := _option(["measured best"] + d.models, row.chosen if row.chosen else "measured best")
			o.name = "s_" + subj
			ui.subjgrid.add_child(o)
		for ch in ui.catgrid.get_children():
			ch.queue_free()
		ui.catgrid.add_child(_h("default", 13))
		var od := _option(["ask", "allow", "deny"], d.tool_rules.default)
		od.name = "c_default"
		ui.catgrid.add_child(od)
		for k in d.tool_kinds:
			ui.catgrid.add_child(_h(str(k).replace("_", " "), 13))
			var o2 := _option(["default", "ask", "allow", "deny"], d.tool_rules.categories.get(k, "default"))
			o2.name = "c_" + k
			ui.catgrid.add_child(o2)
		ui.rulesnote.text = "%d rule(s) before these: %s" % [d.tool_rules.rules.size(), ", ".join(d.tool_rules.rules.map(func(r): return "%s %s/%s" % [r.action, r.server, r.tool]))])


func _use_ws(name: String) -> void:
	studio.post_json("/api/workspaces/use", {"name": name}, func(_c, dd):
		_load_settings()
		if dd.get("restart_needed"):
			ui.subjmsg.text = "workspace %s is active after Studio restarts · [url=restart:]restart now[/url]" % name)


func _save_subjects() -> void:
	var vals := {}
	for subj in settings.subjects:
		var pick = _sel(ui.subjgrid.get_node("s_" + subj))
		vals[subj] = null if pick == "measured best" else pick
	studio.post_json("/api/settings/subjects", {"subject_models": vals}, func(c, d):
		ui.subjmsg.text = "saved" if c == 200 else _err(str(d.get("error", c))))


func _save_cats() -> void:
	var rules: Dictionary = settings.tool_rules.duplicate(true)
	rules.default = _sel(ui.catgrid.get_node("c_default"))
	for k in settings.tool_kinds:
		rules.categories[k] = _sel(ui.catgrid.get_node("c_" + k))
	studio.post_json("/api/settings/tool-rules", {"rules": rules}, func(c, d):
		ui.toolmsg.text = "saved" if c == 200 else _err(str(d.get("error", c))))


# ---------------------------------------------------------------- setup

func _build_setup() -> void:
	var v := _page("Setup")
	var c := _card(v, "Health check")
	ui.health = _note("…")
	c.add_child(_row([_btn("Run checks", _load_setup), _btn("Build llama.cpp", func(): _open_job("build_llama", {})),
		_btn("Restart Studio", func(): studio.post_json("/api/restart", {}, func(_c, _d): ui.health.text = "restarting…"))]))
	c.add_child(ui.health)
	var r := _card(v, "Requirements")
	ui.reqs = _note("…")
	r.add_child(ui.reqs)


func _load_setup() -> void:
	ui.health.text = "checking…"
	studio.get_json("/api/doctor", func(c, d):
		if c != 200:
			ui.health.text = _err(str(d.get("error", c)))
			return
		var t := ""
		for row in d.get("checks", []):
			if row.status in ["warn", "fail"]:
				t += "[color=#%s]%s[/color] %s: %s%s\n" % [(BAD if row.status == "fail" else Color(0.9, 0.71, 0.36)).to_html(false), row.status,
					_esc(row.name), _esc(row.detail), ("\n    fix: " + _linkify(row.fix)) if row.fix else ""]
		ui.health.text = t if t != "" else "[color=#%s]everything checked is fine[/color]" % GOOD.to_html(false), 240.0)
	studio.get_json("/api/requirements", func(c, d):
		if c != 200:
			return
		var t := ""
		for k in d.features:
			var f: Dictionary = d.features[k]
			t += "[b]%s[/b] %s%s\n" % [f.status, _esc(f.title), (" · " + _esc(f.note)) if f.get("note") else ""]
			for it in f.items:
				if it.status != "ok":
					t += "    %s: %s%s\n" % [_esc(it.name), _esc(it.get("have") if it.get("have") else "not found"),
						("  → " + _linkify(str(it.fix))) if it.get("fix") else ""]
		ui.reqs.text = t, 120.0)


# ---------------------------------------------------------------- lab

func _build_lab() -> void:
	var v := _page("Lab")
	v.add_child(_note("Labs and live views run on the Studio machine; they open in the browser."))
	ui.lablist = VBoxContainer.new()
	v.add_child(ui.lablist)


func _load_lab() -> void:
	studio.get_json("/api/lab", func(c, d):
		for ch in ui.lablist.get_children():
			ch.queue_free()
		if c != 200:
			ui.lablist.add_child(_note(_err(str(d.get("error", c)))))
			return
		for name in d.services:
			var s: Dictionary = d.services[name]
			var card := _card(ui.lablist, "%s%s" % [name.replace("_", " "), "  (running)" if s.get("running") else ""])
			card.add_child(_note(_esc(s.get("desc", ""))))
			var host := studio.base.split("://")[-1].split(":")[0]
			card.add_child(_row([_btn("Start", func(): studio.post_json("/api/lab/start", {"name": name}, func(_c, _d): _load_lab())),
				_btn("Open", func(): OS.shell_open("http://%s:%d/" % [host, int(s.get("port", 0))])),
				_btn("Stop", func(): studio.post_json("/api/lab/stop", {"name": name}, func(_c, _d): _load_lab()))])))


## Screenshot every tab (docs and checks): NS_SHOTS=<dir>. NS_SHOT_3D=<url>
## opens that trace or review view in the 3D tab first.
func _capture_all(dir: String) -> void:
	DirAccess.make_dir_recursive_absolute(dir)
	var v3d := OS.get_environment("NS_SHOT_3D")
	var all: Array = []
	for i in tabs.get_tab_count():
		var top := tabs.get_tab_title(i)
		if tabs.is_tab_disabled(i):
			continue
		if groups.has(top):
			var gc: TabContainer = groups[top]
			for j in gc.get_tab_count():
				if not gc.is_tab_disabled(j):
					all.append(gc.get_tab_title(j))
		else:
			all.append(top)
	for i in all.size():
		var t: String = all[i]
		_go(t)
		if t == "3D" and v3d != "":
			viz.open(v3d, studio.token)
			var w := OS.get_environment("NS_SHOT_WAIT")
			await get_tree().create_timer(float(w) if w.is_valid_float() else 6.0).timeout
		elif t == "Chat" and OS.get_environment("NS_SHOT_CHAT") != "":
			ui.chatin.text = OS.get_environment("NS_SHOT_CHAT")
			_send()
			await get_tree().create_timer(6.0).timeout
		else:
			await get_tree().create_timer(3.0).timeout
		await RenderingServer.frame_post_draw
		get_viewport().get_texture().get_image().save_png("%s/godot-%02d-%s.png" % [dir, i, t.to_lower()])
	get_tree().quit()


func _unhandled_input(e: InputEvent) -> void:
	if e is InputEventKey and e.pressed and e.keycode == KEY_F12 and viz and _page_name(tabs.current_tab) == "3D":
		ui.shot.text = viz.screenshot()
