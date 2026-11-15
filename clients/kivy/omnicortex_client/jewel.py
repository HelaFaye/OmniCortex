"""
The emotion jewel (docs/WIDGETS.md), drawn with Kivy's canvas: the "core" style of Studio's browser widget and the
Godot client. It reads Studio's /api/emotions answer: {names, z [token][lane], colors}.

- The core's colour is the lanes' colours mixed by |z|^2; it pulses with the strongest emotion.
- Each emotion has a ribbon on its own orbit. Its length follows |z| (3 = a full sweep), and it turns one way for
  a positive z and the other way for a negative one.
- It plays the reply token by token, then holds the reply's mean. A tap replays it.
"""
from __future__ import annotations

import math

from kivy.clock import Clock
from kivy.graphics import Color, Ellipse, Line
from kivy.properties import NumericProperty, StringProperty
from kivy.uix.widget import Widget


def hex_rgb(h: str) -> tuple[float, float, float]:
    h = h.lstrip("#")
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    return tuple(int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))


def mix(colors: list[str], z: list[float]) -> tuple[float, float, float]:
    """The lanes' colours mixed by |z|^2; an emotion below its mean is mixed in darker."""
    w = [v * v for v in z]
    tot = sum(w) or 1.0
    out = [0.0, 0.0, 0.0]
    for c, wi, v in zip(colors, w, z):
        rgb = hex_rgb(c)
        k = 1.0 if v >= 0 else 0.55
        for i in range(3):
            out[i] += rgb[i] * k * wi / tot
    return tuple(out) if sum(w) else (0.5, 0.5, 0.55)


class EmotionJewel(Widget):
    frame = NumericProperty(0)          # the token shown; len(z) = the reply's mean
    caption = StringProperty("")

    def __init__(self, **kw):
        super().__init__(**kw)
        self.names, self.z, self.colors, self.mean = [], [], [], []
        self.phase = []
        self._ev = None
        self.bind(pos=self.draw, size=self.draw, frame=self.draw)

    def set_data(self, r: dict):
        self.names, self.z = list(r.get("names") or []), [list(row) for row in r.get("z") or []]
        self.colors = list(r.get("colors") or ["#cccccc"] * len(self.names))
        self.mean = list(r.get("mean") or ([sum(c) / len(self.z) for c in zip(*self.z)] if self.z else []))
        self.phase = [i * 2 * math.pi / max(1, len(self.names)) for i in range(len(self.names))]
        self.play()

    def current(self) -> list[float]:
        if not self.z:
            return []
        return self.z[int(self.frame)] if self.frame < len(self.z) else self.mean

    def play(self, *_):
        if self._ev:
            self._ev.cancel()
        self.frame = 0
        self._ev = Clock.schedule_interval(self._tick, 1 / 12)

    def _tick(self, dt):
        z = self.current()
        for i, v in enumerate(z):
            self.phase[i] += dt * (0.6 + min(abs(v), 3)) * (1 if v >= 0 else -1)
        if self.frame < len(self.z):
            self.frame += 1
            tok = self.frame - 1
            strongest = max(range(len(z)), key=lambda i: abs(z[i])) if z else None
            self.caption = (f"token {tok + 1}/{len(self.z)}: {self.names[strongest]} {z[strongest]:+.1f}"
                            if strongest is not None else "")
        else:
            self.caption = "the reply's mean (tap to replay)"
            self.draw()
        return True

    def on_touch_down(self, touch):
        if self.collide_point(*touch.pos) and self.z:
            self.play()
            return True
        return super().on_touch_down(touch)

    def on_parent(self, *_):
        if self.parent is None and self._ev:          # removed from the screen: stop animating
            self._ev.cancel()
            self._ev = None

    def draw(self, *_):
        self.canvas.clear()
        z = self.current()
        if not z:
            return
        cx, cy = self.center
        R = min(self.width, self.height) / 2 * 0.95
        peak = max(abs(v) for v in z)
        with self.canvas:
            r, g, b = mix(self.colors, z)
            core = R * (0.28 + 0.08 * min(peak, 3) / 3)
            Color(r, g, b, 0.25)                                   # halo, wider as emotions rise
            halo = core * (1.3 + 0.4 * min(peak, 3) / 3)
            Ellipse(pos=(cx - halo, cy - halo), size=(2 * halo, 2 * halo))
            Color(r, g, b, 1)
            Ellipse(pos=(cx - core, cy - core), size=(2 * core, 2 * core))
            n = len(z)
            for i, v in enumerate(z):
                rad = core + (R - core) * (i + 1) / (n + 1)
                sweep = 20 + 320 * min(abs(v), 3) / 3
                start = math.degrees(self.phase[i]) % 360
                cr, cg, cb = hex_rgb(self.colors[i % len(self.colors)])
                k = 1.0 if v >= 0 else 0.55
                Color(cr * k, cg * k, cb * k, 0.9)
                Line(circle=(cx, cy, rad, start, start + sweep), width=1 + 2.5 * min(abs(v), 3) / 3)
