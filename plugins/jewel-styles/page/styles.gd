extends RefCounted
## Extra emotion-jewel styles for the Godot client (the jewel-styles add-on). The client loads this from
## Studio while the add-on is on: styles names them, draw(style, jewel, centre, radius) draws one during
## the jewel's _draw, reading its smoothed values (cur, mix, mag, t, color_of). Same drawing as styles.js.

const STYLES := ["icosphere", "tesseract"]

var ico_v: Array = []          # unit vectors
var ico_f: Array = []          # triangles (index triples)
var dirs: Array = []           # one direction per emotion slot, spread over the sphere


func draw(style: String, j, c: Vector2, r: float) -> void:
	if style == "icosphere":
		_draw_ico(j, c, r)
	elif style == "tesseract":
		_draw_tesseract(j, c, r)


# ------------------------------------------------------------- icosphere

func _init() -> void:
	var g := (1.0 + sqrt(5.0)) / 2.0
	var v := [Vector3(-1, g, 0), Vector3(1, g, 0), Vector3(-1, -g, 0), Vector3(1, -g, 0),
		Vector3(0, -1, g), Vector3(0, 1, g), Vector3(0, -1, -g), Vector3(0, 1, -g),
		Vector3(g, 0, -1), Vector3(g, 0, 1), Vector3(-g, 0, -1), Vector3(-g, 0, 1)]
	var f := [[0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11], [1, 5, 9], [5, 11, 4], [11, 10, 2],
		[10, 7, 6], [7, 1, 8], [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9], [4, 9, 5], [2, 4, 11],
		[6, 2, 10], [8, 6, 7], [9, 8, 1]]
	for i in v.size():
		v[i] = v[i].normalized()
	# one subdivision: 80 faces, enough to show where it bulges
	var mid := {}
	var nf := []
	for tri in f:
		var m := []
		for e in [[tri[0], tri[1]], [tri[1], tri[2]], [tri[2], tri[0]]]:
			var key := "%d_%d" % [mini(e[0], e[1]), maxi(e[0], e[1])]
			if not mid.has(key):
				v.append(((v[e[0]] + v[e[1]]) / 2.0).normalized())
				mid[key] = v.size() - 1
			m.append(mid[key])
		nf += [[tri[0], m[0], m[2]], [tri[1], m[1], m[0]], [tri[2], m[2], m[1]], [m[0], m[1], m[2]]]
	ico_v = v
	ico_f = nf


## n directions spread evenly over the sphere (a Fibonacci lattice).
func _spread(n: int) -> Array:
	var out := []
	for i in n:
		var y := 1.0 - 2.0 * (i + 0.5) / maxf(1.0, n)
		var rad := sqrt(maxf(0.0, 1.0 - y * y))
		var th := PI * (3.0 - sqrt(5.0)) * i
		out.append(Vector3(cos(th) * rad, y, sin(th) * rad))
	return out


## The sphere is pushed out (positive z) or in (negative) around each emotion's
## direction, and each face takes the colours of the emotions bulging it.
func _draw_ico(j, c: Vector2, r: float) -> void:
	if dirs.size() != j.cur.size():
		dirs = _spread(j.cur.size())
	var rot: Basis = Basis(Vector3.UP, j.t * 0.5) * Basis(Vector3.RIGHT, 0.4 + 0.2 * sin(j.t * 0.3))
	var pts := []
	var wts := []
	for v in ico_v:
		var d := 0.0
		var w := []
		for k in j.cur.size():
			var f := pow(maxf(0.0, v.dot(dirs[k])), 3.0)
			d += j.cur[k] / 3.0 * f
			w.append(f * absf(j.cur[k]))
		var p: Vector3 = rot * (v * (0.55 + 0.35 * clampf(d, -0.6, 1.2))) * r
		pts.append(p)
		wts.append(w)
	var light := Vector3(-0.4, -0.6, 0.7).normalized()
	var faces := []
	for tri in ico_f:
		var a: Vector3 = pts[tri[0]]
		var b: Vector3 = pts[tri[1]]
		var e: Vector3 = pts[tri[2]]
		var nrm := (b - a).cross(e - a).normalized()
		if nrm.z < 0:
			continue
		faces.append([(a.z + b.z + e.z) / 3.0, tri, nrm])
	faces.sort_custom(func(x, y): return x[0] < y[0])
	for fc in faces:
		var tri: Array = fc[1]
		var col := Color(0, 0, 0)
		var ws := 0.0
		for i in tri:
			for k in j.cur.size():
				var w: float = wts[i][k]
				var ck: Color = j.color_of(k)
				if j.cur[k] < 0:
					ck = ck.darkened(0.5)
				col += ck * w
				ws += w
		col = j.mix.darkened(0.35) if ws < 1e-4 else col / ws
		var shade := 0.35 + 0.65 * maxf(0.0, (fc[2] as Vector3).dot(light))
		col = Color(col.r * shade, col.g * shade, col.b * shade, 0.95)
		var poly := PackedVector2Array([c + Vector2(pts[tri[0]].x, pts[tri[0]].y),
			c + Vector2(pts[tri[1]].x, pts[tri[1]].y), c + Vector2(pts[tri[2]].x, pts[tri[2]].y)])
		j.draw_colored_polygon(poly, col)
		j.draw_polyline(poly + PackedVector2Array([poly[0]]), Color(col.lightened(0.3), 0.35), 1.0, true)


# ------------------------------------------------------------- tesseract

## A 4D cube. It turns in the XW plane with the positive emotions and in YW
## with the negative ones, faster the stronger they are; the inner cube takes
## the mix colour and each outer edge one emotion's colour.
func _draw_tesseract(j, c: Vector2, r: float) -> void:
	var pos := 0.0
	var neg := 0.0
	for z in j.cur:
		pos += maxf(0.0, z)
		neg += maxf(0.0, -z)
	var axw: float = j.t * (0.3 + 0.5 * clampf(pos / 3.0, 0.0, 2.0))
	var ayw: float = j.t * (0.2 + 0.5 * clampf(neg / 3.0, 0.0, 2.0))
	var axy: float = j.t * 0.25
	var proj := []
	for i in 16:
		var p := [1.0 if i & 1 else -1.0, 1.0 if i & 2 else -1.0, 1.0 if i & 4 else -1.0, 1.0 if i & 8 else -1.0]
		var x: float = p[0] * cos(axw) - p[3] * sin(axw)
		var w: float = p[0] * sin(axw) + p[3] * cos(axw)
		var y: float = p[1] * cos(ayw) - w * sin(ayw)
		w = p[1] * sin(ayw) + w * cos(ayw)
		var x2: float = x * cos(axy) - p[2] * sin(axy)
		var z2: float = x * sin(axy) + p[2] * cos(axy)
		var k4 := 1.0 / (2.6 - w)
		var v3 := Vector3(x2, y, z2) * k4
		var k3 := 1.0 / (3.0 - v3.z)
		proj.append([c + Vector2(v3.x, v3.y) * k3 * r * 2.4, w])
	var n: int = j.cur.size()
	var e := 0
	for i in 16:
		for b in 4:
			var q := i ^ (1 << b)
			if q < i:
				continue
			var inner: bool = proj[i][1] < 0 and proj[q][1] < 0
			var col: Color = j.mix if inner or n == 0 else j.color_of(e % n)
			var s: float = j.mag if inner or n == 0 else clampf(absf(j.cur[e % n]) / 3.0, 0.0, 1.0)
			col.a = 0.35 + 0.65 * s
			j.draw_line(proj[i][0], proj[q][0], col, 1.0 + 2.5 * s, true)
			e += 1
	for p in proj:
		j.draw_circle(p[0], 1.5 + 2.0 * j.mag, j.mix.lightened(0.4))
