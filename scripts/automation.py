"""
Automation rules: react to how a computer is being used and how much power it draws.

A rule watches one computer ("local" for the one Studio runs on, or "host:<name>" for a remote host)
and, while its condition holds, applies actions to it:

  when (one of)
    activity   its owner is using it: keyboard or mouse within idle_under seconds, a game controller,
               or a VR session (scripts/presence.py; a host reports it through its agent)
    power      a Home Assistant sensor: average power (W) over the period, or energy (kWh) used since
               the period began, above a limit. Periods: second, minute, hour, day, week, month, year;
               month and year can start on a billing day.
    manual     set through the automation API (Home Assistant automations, scripts)

  then (any of)
    unload     unload the models on that computer
    busy       give it no new work (project tasks, worker models, remote jobs)
    requeue    tasks running there go back to the queue for another computer, without counting as
               failures
    throttle   at most one task at a time there

  until
    clear      as long as the condition holds (activity: plus `cooldown` seconds of quiet)
    period     until the rule's period ends (power: e.g. the next billing day)

The evaluation here is pure: inputs in, decisions out. Studio does the readings and applies the
actions.
"""
from __future__ import annotations

import calendar
import json
import time
import urllib.parse
import urllib.request
import uuid
from datetime import datetime, timedelta

PERIODS = ("second", "minute", "hour", "day", "week", "month", "year")
ACTIONS = ("unload", "busy", "requeue", "throttle")
KINDS = ("activity", "power", "manual")


# ---------------------------------------------------------------- periods

def period_bounds(period: str, now: float, reset_day: int = 1, week_start: int = 0) -> tuple[float, float]:
    """(start, end) of the calendar period containing `now`, in local time. month/year begin on
    reset_day (a billing day; clamped to the month's length); weeks on week_start (0 = Monday)."""
    if period not in PERIODS:
        raise ValueError(f"period: one of {', '.join(PERIODS)}")
    t = datetime.fromtimestamp(now)
    if period == "second":
        s = t.replace(microsecond=0)
        return s.timestamp(), s.timestamp() + 1
    if period == "minute":
        s = t.replace(second=0, microsecond=0)
        return s.timestamp(), s.timestamp() + 60
    if period == "hour":
        s = t.replace(minute=0, second=0, microsecond=0)
        return s.timestamp(), s.timestamp() + 3600
    if period == "day":
        s = t.replace(hour=0, minute=0, second=0, microsecond=0)
        return s.timestamp(), (s + timedelta(days=1)).timestamp()
    if period == "week":
        s = (t - timedelta(days=(t.weekday() - week_start) % 7)).replace(hour=0, minute=0, second=0, microsecond=0)
        return s.timestamp(), (s + timedelta(days=7)).timestamp()

    def at(y, m):
        d = min(max(1, int(reset_day)), calendar.monthrange(y, m)[1])
        return datetime(y, m, d)
    if period == "month":
        s = at(t.year, t.month)
        if t < s:
            y, m = (t.year, t.month - 1) if t.month > 1 else (t.year - 1, 12)
            s = at(y, m)
        y, m = (s.year, s.month + 1) if s.month < 12 else (s.year + 1, 1)
        return s.timestamp(), at(y, m).timestamp()
    s = at(t.year, 1)                                  # year: from the billing day of January
    if t < s:
        s = at(t.year - 1, 1)
    return s.timestamp(), at(s.year + 1, 1).timestamp()


# ---------------------------------------------------------------- measurements

def energy_used(samples: list[tuple[float, float]], start: float) -> float | None:
    """kWh used since `start` from a cumulative meter (Home Assistant total_increasing): the rises
    after start, counting a drop as a meter reset (the new reading is all new use)."""
    pts = sorted((t, v) for t, v in samples if v is not None)
    if not pts:
        return None
    before = [v for t, v in pts if t <= start]
    prev = before[-1] if before else None
    used = 0.0
    for t, v in pts:
        if t <= start:
            continue
        if prev is None:
            prev = v
            continue
        used += v - prev if v >= prev else v
        prev = v
    return used


def average_power(samples: list[tuple[float, float]], start: float, now: float) -> float | None:
    """Time-weighted mean of a power sensor (W) over [start, now]: each reading holds until the next."""
    pts = sorted((t, v) for t, v in samples if v is not None)
    if not pts:
        return None
    before = [(t, v) for t, v in pts if t <= start]
    seq = ([(start, before[-1][1])] if before else []) + [(t, v) for t, v in pts if start < t <= now]
    if not seq:
        return None
    if len(seq) == 1 or now <= seq[0][0]:
        return seq[-1][1]
    total = 0.0
    for (t0, v0), (t1, _) in zip(seq, seq[1:] + [(now, None)]):
        total += v0 * max(0.0, t1 - t0)
    span = now - seq[0][0]
    return total / span if span > 0 else seq[-1][1]


class HomeAssistant:
    """The parts of Home Assistant's REST API rules need: a sensor's state and its history."""

    def __init__(self, url: str, token: str, timeout: float = 10):
        self.url, self.token, self.timeout = url.rstrip("/"), token, timeout

    def _get(self, path: str):
        req = urllib.request.Request(self.url + path, headers={"Authorization": f"Bearer {self.token}",
                                                               "Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.loads(r.read())

    def state(self, entity: str) -> dict:
        d = self._get("/api/states/" + urllib.parse.quote(entity))
        try:
            v = float(d.get("state"))
        except (TypeError, ValueError):
            v = None
        return {"value": v, "unit": (d.get("attributes") or {}).get("unit_of_measurement"),
                "name": (d.get("attributes") or {}).get("friendly_name") or entity}

    def history(self, entity: str, start: float, end: float | None = None) -> list[tuple[float, float]]:
        iso = lambda t: datetime.fromtimestamp(t).astimezone().isoformat()          # noqa: E731
        q = {"filter_entity_id": entity, "minimal_response": "", "no_attributes": ""}
        if end:
            q["end_time"] = iso(end)
        d = self._get(f"/api/history/period/{urllib.parse.quote(iso(start))}?" + urllib.parse.urlencode(q))
        out = []
        for row in (d[0] if d else []):
            try:
                ts = datetime.fromisoformat(str(row.get("last_changed") or row.get("last_updated"))).timestamp()
                out.append((ts, float(row.get("state"))))
            except (TypeError, ValueError):
                continue
        return out


# ---------------------------------------------------------------- rules

def clean_rule(r: dict) -> dict:
    """Check a rule from the page or the API; fill defaults."""
    if not isinstance(r, dict):
        raise ValueError("a rule is an object")
    when = dict(r.get("when") or {})
    kind = when.get("kind")
    if kind not in KINDS:
        raise ValueError(f"when.kind: one of {', '.join(KINDS)}")
    target = str(r.get("target") or "local")
    if target != "local" and not (target.startswith("host:") and len(target) > 5):
        raise ValueError("target: local or host:<name>")
    then = [a for a in r.get("then") or [] if a in ACTIONS]
    if not then:
        raise ValueError(f"then: at least one of {', '.join(ACTIONS)}")
    out = {"id": str(r.get("id") or "r" + uuid.uuid4().hex[:10]), "name": str(r.get("name") or kind)[:80],
           "enabled": bool(r.get("enabled", True)), "target": target, "then": then}
    if kind == "activity":
        out["when"] = {"kind": kind, "idle_under": max(5.0, float(when.get("idle_under", 60))),
                       "gamepad": bool(when.get("gamepad", True)), "vr": bool(when.get("vr", True)),
                       "cooldown": max(0.0, float(when.get("cooldown", 300)))}
        out["until"] = "clear"
    elif kind == "power":
        if not str(when.get("entity") or "").strip():
            raise ValueError("when.entity: the Home Assistant sensor, e.g. sensor.pc_power")
        measure = when.get("measure", "power")
        if measure not in ("power", "energy"):
            raise ValueError("when.measure: power (W) or energy (kWh)")
        if when.get("period", "minute") not in PERIODS:
            raise ValueError(f"when.period: one of {', '.join(PERIODS)}")
        out["when"] = {"kind": kind, "entity": str(when["entity"]).strip(), "measure": measure,
                       "period": when.get("period", "minute"), "limit": float(when["limit"]),
                       "reset_day": int(when.get("reset_day", 1)), "week_start": int(when.get("week_start", 0))}
        out["until"] = r.get("until") if r.get("until") in ("clear", "period") else "period"
    else:
        out["when"] = {"kind": kind}
        out["until"] = "clear"
    return out


def evaluate(rule: dict, inputs: dict, now: float, state: dict | None = None) -> dict:
    """Does the rule hold now? inputs: {"presence": {target: snapshot}, "power": {entity: {value,
    samples}}, "manual": {rule id: bool}}. state: this rule's previous result (for cooldowns).
    -> {active, why, until (timestamp or None), reading}"""
    state = state or {}
    w = rule["when"]
    if w["kind"] == "manual":
        on = bool((inputs.get("manual") or {}).get(rule["id"]))
        return {"active": on, "why": "set through the automation API" if on else "", "until": None, "reading": None}
    if w["kind"] == "activity":
        snap = (inputs.get("presence") or {}).get(rule["target"]) or {}
        reasons = []
        if w.get("vr", True) and snap.get("vr"):
            reasons.append("a VR session is running")
        if w.get("gamepad", True) and snap.get("gamepad"):
            reasons.append("a game controller is in use")
        if snap.get("idle_s") is not None and snap["idle_s"] < w["idle_under"]:
            reasons.append(f"keyboard or mouse used {int(snap['idle_s'])} s ago")
        if reasons:
            return {"active": True, "why": "; ".join(reasons), "until": None, "reading": snap, "last_seen": now}
        last = state.get("last_seen") or 0
        if state.get("active") and now - last < w["cooldown"]:          # stay yielded a while after the last input
            return {"active": True, "why": f"quiet for {int(now - last)} s of {int(w['cooldown'])} s",
                    "until": last + w["cooldown"], "reading": snap, "last_seen": last}
        known = snap.get("idle_s") is not None or snap.get("gamepad") is not None or snap.get("vr") is not None
        return {"active": False, "why": "" if known else "this computer cannot tell whether it is in use",
                "until": None, "reading": snap}
    # power
    start, end = period_bounds(w["period"], now, w.get("reset_day", 1), w.get("week_start", 0))
    p = (inputs.get("power") or {}).get(w["entity"]) or {}
    samples = p.get("samples") or []
    if w["measure"] == "energy":
        val = energy_used(samples, start)
        unit = "kWh"
    else:
        val = average_power(samples, start, now) if samples else p.get("value")
        unit = "W"
    if val is None:
        return {"active": False, "why": "no reading from Home Assistant", "until": None, "reading": None}
    over = val > w["limit"]
    held = rule.get("until") == "period" and state.get("active") and (state.get("until") or 0) > now
    return {"active": over or bool(held),
            "why": (f"{val:.3g} {unit} this {w['period']} is over the limit of {w['limit']:g} {unit}" if over else
                    f"over the limit earlier this {w['period']}" if held else ""),
            "until": end if (over or held) and rule.get("until") == "period" else None,
            "reading": {"value": round(val, 4), "unit": unit, "period_start": start, "period_end": end}}


def combine(rules: list[dict], results: dict) -> dict:
    """Per target: the actions in force and why. -> {target: {actions:set, reasons:[...], until}}"""
    out: dict = {}
    for r in rules:
        res = results.get(r["id"]) or {}
        if not r.get("enabled") or not res.get("active"):
            continue
        t = out.setdefault(r["target"], {"actions": set(), "reasons": [], "until": None, "rules": []})
        t["actions"] |= set(r["then"])
        t["reasons"].append(f"{r['name']}: {res.get('why')}")
        t["rules"].append(r["id"])
        if res.get("until"):
            t["until"] = max(t["until"] or 0, res["until"])
    return out
