#!/usr/bin/env python3
"""
Neuron aging and decay (docs/INTEGRATED_OPTIONS.md): a correction that has not been needed for a while slowly
fades, and comes back at full strength the moment it is needed again.

What decays: the scale of the LoRA adapter Studio serves a model with, which is OmniCortex's live suppression
and correction strength (the slider on the model, /lora-adapters). Each model's configured scale is its full
strength; with no trigger for --grace days the live scale moves toward --floor x that strength, halving the
remaining distance every --half-life days. A trigger sets it straight back to full:

    a reply whose H-neuron risk (the model's classifier) is at or above --risk
    a Fix This Mistake correction for that model
    a TestQA failure reported to Studio for it

Every change is logged (decay.json beside Studio's settings), so the Settings page and the review can show when a
model's corrections were eased off and why. Off by default: it changes how a model answers without asking.

Not done: per-neuron ageing inside the weights. A suppressed or edited GGUF has its scales baked in; only an
adapter can be eased live. For a baked edit, the log tells you when to rebuild it with a smaller scale.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

DAY = 86400.0


def live_scale(full: float, last_trigger: float | None, now: float, grace_days: float = 7, half_life_days: float = 14,
               floor: float = 0.5) -> float:
    """The scale a correction should have now, given when it last mattered."""
    if full == 0 or last_trigger is None:
        return full
    idle = (now - last_trigger) / DAY - grace_days
    if idle <= 0:
        return full
    target = full * floor
    return round(target + (full - target) * math.pow(0.5, idle / max(half_life_days, 1e-6)), 4)


class Log:
    def __init__(self, path):
        self.path = Path(path)
        self.d = json.loads(self.path.read_text()) if self.path.exists() else {"models": {}}

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.d, indent=1))

    def model(self, mid: str) -> dict:
        return self.d["models"].setdefault(mid, {"last_trigger": None, "scale": None, "events": []})

    def trigger(self, mid: str, why: str, now: float | None = None) -> None:
        m = self.model(mid)
        now = time.time() if now is None else now
        first = m["last_trigger"] is None
        m["last_trigger"] = now
        if first or not m["events"] or m["events"][-1].get("why") != why or now - m["events"][-1]["t"] > 3600:
            m["events"] = (m["events"] + [{"t": now, "kind": "trigger", "why": why}])[-200:]
        self.save()

    def decide(self, mid: str, full: float, cfg: dict, now: float | None = None) -> float | None:
        """-> the new live scale when it should change (by more than 0.01), else None."""
        now = time.time() if now is None else now
        m = self.model(mid)
        if m["last_trigger"] is None:
            m["last_trigger"] = now            # start the clock when decay first sees a model
            self.save()
        s = live_scale(full, m["last_trigger"], now, cfg["grace_days"], cfg["half_life_days"], cfg["floor"])
        cur = m["scale"] if m["scale"] is not None else full
        if abs(s - cur) < 0.01:
            return None
        m["scale"] = s
        m["events"] = (m["events"] + [{"t": now, "kind": "decay" if s < cur else "restore", "scale": s, "full": full}])[-200:]
        self.save()
        return s
