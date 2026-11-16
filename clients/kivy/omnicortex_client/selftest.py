"""
The app's self-test: drives the real screens through a whole session, for an emulator or a desktop run in CI.

It runs only when `oc_selftest.json` is in the app's data folder. On Android that is the app's private storage,
which only `adb shell run-as` can write, and only on a debug build. Nothing in the app creates it.

    {"link": "https://10.0.2.2:8790/pair#c=…&fp=…", "name": "emulator", "shots": true}

Each step logs `[OC-SELFTEST] <step> ok|FAIL <detail>`, and the run ends with `[OC-SELFTEST] DONE pass=N fail=M`,
in Kivy's log (logcat on Android). Screenshots go to the data folder as `selftest-<n>-<step>.png`. A test harness
reads both: tests/android/run_selftest.sh, and tests/test_kivy_client.py on the desktop.
"""
from __future__ import annotations

import json
import os
import time

from kivy.clock import Clock
from kivy.logger import Logger

FILE = "oc_selftest.json"


def config(data_dir: str) -> dict | None:
    p = os.path.join(data_dir or "", FILE)
    if not data_dir or not os.path.exists(p):
        return None
    try:
        with open(p) as f:
            return json.load(f)
    except (OSError, ValueError) as e:
        Logger.error(f"OC-SELFTEST: cannot read {p} ({e})")
        return None


class SelfTest:
    def __init__(self, app, cfg: dict, data_dir: str):
        self.app, self.cfg, self.dir = app, cfg, data_dir
        self.passed, self.failed, self.n = 0, 0, 0
        self.steps = [
            ("pair", self.pair), ("models", self.models), ("chat", self.chat), ("risk", self.risk),
            ("truth", self.truth), ("emotions", self.emotions), ("fix", self.fix), ("review", self.review),
            ("scan", self.scan), ("restart", self.restart),
        ]

    # ------------------------------------------------------------- plumbing
    def log(self, step, ok, detail=""):
        if ok:
            self.passed += 1
        else:
            self.failed += 1
        Logger.info(f"OC-SELFTEST: {step} {'ok' if ok else 'FAIL'} {detail}".rstrip())

    def shot(self, name):
        if not self.cfg.get("shots", True):
            return
        from kivy.core.window import Window
        self.n += 1
        # Window.screenshot appends a counter to the name; the harness globs for it
        Window.screenshot(name=os.path.join(self.dir, f"selftest-{self.n:02d}-{name}.png"))

    def start(self):
        Logger.info("OC-SELFTEST: START")
        Clock.schedule_once(lambda dt: self.next(), 1.0)

    def next(self):
        if not self.steps:
            Logger.info(f"OC-SELFTEST: DONE pass={self.passed} fail={self.failed}")
            if self.cfg.get("quit", False):
                Clock.schedule_once(lambda dt: self.app.stop(), 0.5)
            return
        name, fn = self.steps.pop(0)
        try:
            fn(name)
        except Exception as e:                      # a step that raises is a failure, and the run goes on
            self.log(name, False, f"{type(e).__name__}: {e}")
            Clock.schedule_once(lambda dt: self.next(), 0.2)

    def wait(self, name, cond, detail, timeout=60.0, then_shot=True):
        """Poll cond() on the interface thread; log ok when it holds, FAIL at the timeout."""
        end = time.time() + timeout

        def tick(dt):
            try:
                ok = cond()
            except Exception as e:
                ok, err = False, f"{type(e).__name__}: {e}"
            else:
                err = ""
            if ok:
                if then_shot:
                    self.shot(name)
                self.log(name, True, detail() if callable(detail) else detail)
                Clock.schedule_once(lambda dt: self.next(), 0.3)
                return False
            if time.time() > end:
                self.shot(name + "-timeout")
                status = self.app.root.get_screen(self.app.root.current).ids.get("status")
                self.log(name, False, err or f"timed out; status: {getattr(status, 'text', '')[:300]}")
                Clock.schedule_once(lambda dt: self.next(), 0.3)
                return False
            return True
        Clock.schedule_interval(tick, 0.25)

    def reply(self):
        return next(m for m in reversed(self.app.client.messages) if m.role == "assistant")

    def view(self):
        return self.app.views[id(self.reply())]

    # ------------------------------------------------------------- steps
    def pair(self, name):
        if self.app.client.pairing:
            self.app.unpair()
        scr = self.app.root.get_screen("pair")
        scr.ids.link.text = self.cfg["link"]
        scr.ids.name.text = self.cfg.get("name", "selftest")
        self.shot("pair-screen")
        scr.pair(scr.ids.link.text, scr.ids.name.text)
        self.wait(name, lambda: self.app.client.pairing is not None and self.app.root.current == "chat",
                  lambda: f"paired with {self.app.client.pairing.base}, fingerprint "
                          f"{(self.app.client.pairing.fingerprint or 'none')[:16]}")

    def models(self, name):
        self.wait(name, lambda: len(self.app.client.models) > 0,
                  lambda: "models: " + ", ".join(self.app.client.models), then_shot=False)

    def chat(self, name):
        self.app.send("What is the capital of Australia?")
        self.wait(name, lambda: not self.app.client.busy and self.reply().content.strip() != "",
                  lambda: f"streamed reply: {self.reply().content!r}")

    def risk(self, name):
        self.app.check(self.view())
        self.wait(name, lambda: "flagged" in self.reply().note.lower() or "no token" in self.reply().note.lower(),
                  lambda: self.reply().note)

    def truth(self, name):
        r = self.reply()
        r.note = ""
        self.app.client.truth(r, ["Canberra"], on_done=self.app.view_for,
                              on_error=lambda e: self.app.say(e, error=True))
        self.wait(name, lambda: "activations" in self.reply().note, lambda: self.reply().note.splitlines()[0])

    def emotions(self, name):
        self.app.emotions(self.view())
        self.wait(name, lambda: self.reply().emotions is not None and self.view().has_jewel
                  and self.view().ids.jewel.frame >= len(self.reply().emotions.get("z") or []),
                  lambda: self.reply().note, timeout=90)

    def fix(self, name):
        self.app.client.fix(self.reply(), "The capital of Australia is Canberra.", "selftest",
                            on_done=self.app.view_for, on_error=lambda e: self.app.say(e, error=True))
        self.wait(name, lambda: self.reply().note.startswith("Kept for training"), lambda: self.reply().note)

    def review(self, name):
        self.app.root.current = "lookup"
        scr = self.app.root.get_screen("lookup")
        scr.result = ""
        self.app.lookup(self.cfg.get("lookup", "Qwen/Qwen2.5-1.5B-Instruct"))
        self.wait(name, lambda: scr.result and scr.result != "Looking up…", lambda: scr.result.splitlines()[0])

    def scan(self, name):
        """The QR scanner starts without crashing. What it shows depends on the device (Play services, a camera),
        so a scan screen or a clear error message both count; a crash or an exception does not."""
        from . import qr
        self.app.root.current = "pair"
        seen = {}
        qr.scan(lambda link: seen.setdefault("link", link), lambda msg: seen.setdefault("error", msg))
        if not qr.on_android():
            self.wait(name, lambda: bool(seen), lambda: seen.get("error") or seen.get("link"), timeout=40,
                      then_shot=False)
            return
        # Google's scanner opens its own screen over the app, which only the harness can photograph
        Logger.info("OC-SELFTEST: scan-open, photograph the screen now")

        def back(dt):
            if "error" in seen and qr.CODE_FAILED in seen["error"]:
                self.log(name, False, seen["error"])             # a bug in the app, not the device
            elif "error" in seen and qr.DOWNLOADING in seen["error"]:
                self.log(name, True, "Google's scanner answered (its module is still downloading): " + seen["error"])
            elif "error" in seen:
                self.log(name, True, "no scanner on this device, and the app says so: " + seen["error"])
            else:
                self.log(name, True, "the scanner opened" + (f" and read {seen['link']}" if "link" in seen else ""))
            # the harness photographs the scanner and closes it with the Back key (tests/android/run_selftest.sh)
            Clock.schedule_once(lambda dt: self.next(), 6)
        Clock.schedule_once(back, 12)

    def restart(self, name):
        """The pairing survives a restart: a fresh controller loads it from the private data folder."""
        from .controller import Client
        c = Client(data_dir=self.dir)
        ok = c.load() and c.pairing.token == self.app.client.pairing.token
        self.app.root.current = "chat"
        self.log(name, ok, f"saved pairing reloaded from {self.dir}" if ok else "the saved pairing did not reload")
        Clock.schedule_once(lambda dt: self.next(), 0.3)
