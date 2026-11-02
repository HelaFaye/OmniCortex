class_name StudioClient
extends Node
## Talks to a OmniCortex Studio over HTTP(S): the same API the web pages use.
##
## The Godot client is built to run on a different computer from Studio, so
## drawing never takes GPU memory from the model: Studio serves models on the
## machine with the GPU, this draws. Connections (URL, token, a trusted
## certificate for a self-signed Studio) are saved in user://connections.cfg.

signal changed                      # connected, disconnected or role changed

const CFG := "user://connections.cfg"

var base := ""                      # e.g. https://studio.lan:7870
var token := ""
var cert_path := ""                 # PEM to trust (Studio's self-signed certificate)
var role := ""                      # "owner" or "device" once connected
var status: Dictionary = {}
var last_error := ""


func headers(json := false) -> PackedStringArray:
	var h := PackedStringArray()
	if token != "":
		h.append("Authorization: Bearer " + token)
	if json:
		h.append("Content-Type: application/json")
	return h


func _tls() -> TLSOptions:
	if cert_path != "" and FileAccess.file_exists(cert_path):
		var c := X509Certificate.new()
		if c.load(cert_path) == OK:
			return TLSOptions.client(c)
	return TLSOptions.client()


## method: HTTPClient.METHOD_GET or METHOD_POST. cb(code: int, data) gets parsed
## JSON (or the raw bytes when raw is true; or {"error": …} on failure).
func call_api(method: int, path: String, body = null, cb: Callable = Callable(), raw := false,
		timeout := 120.0) -> void:
	if base == "":
		if cb.is_valid():
			cb.call(0, {"error": "not connected to a Studio"})
		return
	var r := HTTPRequest.new()
	r.timeout = timeout
	r.set_tls_options(_tls())
	add_child(r)
	r.request_completed.connect(func(result, code, _h, bytes):
		r.queue_free()
		if result != HTTPRequest.RESULT_SUCCESS:
			last_error = "cannot reach %s (%s)" % [base, _result_name(result)]
			if cb.is_valid():
				cb.call(0, {"error": last_error})
			return
		if raw:
			if cb.is_valid():
				cb.call(code, bytes)
			return
		var data = JSON.parse_string(bytes.get_string_from_utf8())
		if data == null:
			data = {"error": bytes.get_string_from_utf8().left(300)}
		if typeof(data) == TYPE_DICTIONARY and data.has("error") and typeof(data.error) == TYPE_DICTIONARY:
			data.error = str(data.error.get("message", data.error))
		if cb.is_valid():
			cb.call(code, data))
	var err := r.request(base + path, headers(body != null), method,
		"" if body == null else JSON.stringify(body))
	if err != OK and cb.is_valid():
		r.queue_free()
		cb.call(0, {"error": "request failed (%d)" % err})


func get_json(path: String, cb: Callable, timeout := 120.0) -> void:
	call_api(HTTPClient.METHOD_GET, path, null, cb, false, timeout)


func post_json(path: String, body: Dictionary, cb: Callable = Callable(), timeout := 600.0) -> void:
	call_api(HTTPClient.METHOD_POST, path, body, cb, false, timeout)


func _result_name(r: int) -> String:
	match r:
		HTTPRequest.RESULT_CANT_CONNECT: return "connection refused"
		HTTPRequest.RESULT_CANT_RESOLVE: return "unknown host"
		HTTPRequest.RESULT_TLS_HANDSHAKE_ERROR: return "TLS failed: trust Studio's certificate under Connect"
		HTTPRequest.RESULT_TIMEOUT: return "timed out"
		HTTPRequest.RESULT_CONNECTION_ERROR: return "connection error"
	return "error %d" % r


## Check the connection; sets role and status, then emits changed.
func connect_to(url: String, tok: String, cert := "", cb: Callable = Callable()) -> void:
	base = url.strip_edges().trim_suffix("/")
	if base != "" and not base.begins_with("http"):
		base = "http://" + base
	token = tok.strip_edges()
	cert_path = cert.strip_edges()
	get_json("/api/status", func(code, d):
		if code == 200:
			status = d
			role = str(d.get("role", "owner"))
			last_error = ""
		else:
			role = ""
			last_error = str(d.get("error", "HTTP %d" % code)) if typeof(d) == TYPE_DICTIONARY else "HTTP %d" % code
			if code == 401:
				last_error = "Studio wants a token: paste the owner token, or pair this computer"
		changed.emit()
		if cb.is_valid():
			cb.call(code == 200), 15.0)


## Claim a pairing code or link from Studio's Link panel. -> cb(ok, message).
func pair(url: String, code_or_link: String, device_name: String, cb: Callable) -> void:
	var code := code_or_link.strip_edges()
	var i := code.find("c=")
	if i >= 0:
		code = code.substr(i + 2).split("&")[0]
	base = url.strip_edges().trim_suffix("/")
	token = ""
	post_json("/api/pair/claim", {"code": code, "name": device_name}, func(c, d):
		if c == 200 and d.has("token"):
			token = str(d.token)
			cb.call(true, "paired" + ("; access ends " + Time.get_datetime_string_from_unix_time(int(d.expires)) if d.get("expires") else ""))
		else:
			cb.call(false, str(d.get("error", "pairing failed (%d)" % c))), 20.0)


func is_owner() -> bool:
	return role == "owner"


# ------------------------------------------------------------ saved connections

func saved() -> Array:
	var cf := ConfigFile.new()
	if cf.load(CFG) != OK:
		return []
	var out := []
	for s in cf.get_sections():
		out.append({"name": s, "url": cf.get_value(s, "url", ""), "token": cf.get_value(s, "token", ""),
			"cert": cf.get_value(s, "cert", ""), "last": cf.get_value(s, "last", false)})
	return out


func save_connection(name: String, url: String, tok: String, cert: String) -> void:
	var cf := ConfigFile.new()
	cf.load(CFG)
	for s in cf.get_sections():
		cf.set_value(s, "last", false)
	cf.set_value(name, "url", url)
	cf.set_value(name, "token", tok)
	cf.set_value(name, "cert", cert)
	cf.set_value(name, "last", true)
	cf.save(CFG)


func forget(name: String) -> void:
	var cf := ConfigFile.new()
	if cf.load(CFG) == OK and cf.has_section(name):
		cf.erase_section(name)
		cf.save(CFG)
