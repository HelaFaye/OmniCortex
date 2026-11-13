"""Automation rules: periods with billing days, energy and power readings, activity with a cooldown."""
import sys
from datetime import datetime
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import automation as au  # noqa: E402

ts = lambda *a: datetime(*a).timestamp()           # noqa: E731


def test_periods_and_billing_day():
    s, e = au.period_bounds("month", ts(2026, 3, 10, 12), reset_day=15)
    assert (s, e) == (ts(2026, 2, 15), ts(2026, 3, 15))
    s, e = au.period_bounds("month", ts(2026, 3, 20), reset_day=15)
    assert (s, e) == (ts(2026, 3, 15), ts(2026, 4, 15))
    assert au.period_bounds("month", ts(2026, 2, 28, 9), reset_day=31)[0] == ts(2026, 2, 28)   # clamped
    assert au.period_bounds("week", ts(2026, 10, 22, 9))[0] == ts(2026, 10, 19)               # Monday
    assert au.period_bounds("minute", ts(2026, 1, 1, 0, 5, 30)) == (ts(2026, 1, 1, 0, 5), ts(2026, 1, 1, 0, 6))
    assert au.period_bounds("year", ts(2026, 6, 1))[0] == ts(2026, 1, 1)
    with pytest.raises(ValueError):
        au.period_bounds("fortnight", 0)


def test_energy_and_power():
    meter = [(100, 50.0), (200, 50.4), (300, 50.9), (400, 0.2), (500, 0.5)]     # a reset at t=400
    assert au.energy_used(meter, 150) == pytest.approx(0.4 + 0.5 + 0.2 + 0.3)
    w = [(0, 100.0), (60, 300.0)]
    assert au.average_power(w, 0, 120) == pytest.approx(200.0)
    assert au.average_power(w, 30, 90) == pytest.approx((30 * 100 + 30 * 300) / 60)
    assert au.average_power([], 0, 1) is None


def test_power_rule_holds_until_the_billing_period_ends():
    r = au.clean_rule({"name": "monthly cap", "target": "host:rig", "then": ["busy", "unload"],
                       "when": {"kind": "power", "entity": "sensor.rig_energy", "measure": "energy", "period": "month",
                                "limit": 30, "reset_day": 5}})
    now = ts(2026, 3, 20)
    inp = {"power": {"sensor.rig_energy": {"samples": [(ts(2026, 3, 5, 1), 100.0), (ts(2026, 3, 19), 135.0)]}}}
    res = au.evaluate(r, inp, now)
    assert res["active"] and "35 kWh this month" in res["why"] and res["until"] == ts(2026, 4, 5)
    combined = au.combine([r], {r["id"]: res})
    assert combined["host:rig"]["actions"] == {"busy", "unload"} and combined["host:rig"]["until"] == ts(2026, 4, 5)
    under = au.evaluate(r, {"power": {"sensor.rig_energy": {"samples": [(ts(2026, 3, 5, 1), 100.0),
                                                                          (ts(2026, 3, 19), 110.0)]}}}, now)
    assert not under["active"]
    assert not au.evaluate(r, {}, now)["active"]                                   # no reading: no action


def test_activity_rule_with_cooldown():
    r = au.clean_rule({"target": "host:rig", "then": ["unload", "busy", "requeue"],
                       "when": {"kind": "activity", "idle_under": 60, "cooldown": 300}})
    on = au.evaluate(r, {"presence": {"host:rig": {"idle_s": 3, "gamepad": False, "vr": False}}}, 1000)
    assert on["active"] and "keyboard or mouse" in on["why"]
    later = au.evaluate(r, {"presence": {"host:rig": {"idle_s": 200, "gamepad": False, "vr": False}}}, 1200, on)
    assert later["active"] and "quiet" in later["why"]                             # still within the cooldown
    off = au.evaluate(r, {"presence": {"host:rig": {"idle_s": 500, "gamepad": False, "vr": False}}}, 1400, later)
    assert not off["active"]
    vr = au.evaluate(r, {"presence": {"host:rig": {"idle_s": None, "gamepad": None, "vr": True}}}, 1000)
    assert vr["active"] and "VR" in vr["why"]
    blind = au.evaluate(r, {"presence": {"host:rig": {"idle_s": None, "gamepad": None, "vr": None}}}, 1000)
    assert not blind["active"] and "cannot tell" in blind["why"]


def test_rule_checks():
    for bad in ({"when": {"kind": "weather"}, "then": ["busy"]}, {"when": {"kind": "manual"}, "then": []},
                {"when": {"kind": "power", "limit": 1}, "then": ["busy"]}, {"target": "rig", "when": {"kind": "manual"},
                                                                             "then": ["busy"]}):
        with pytest.raises((ValueError, KeyError)):
            au.clean_rule(bad)
