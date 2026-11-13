"""Will a model run here: full GPU offload, partial, CPU, or too large; and which download to suggest."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "viz"))
sys.path.insert(0, str(ROOT / "scripts"))
import studio  # noqa: E402

G = 2 ** 30


class Ram16:
    ram, ram_available = 16 * G, 14 * G


def test_any_vendor(monkeypatch):
    monkeypatch.setattr(studio, "Host", Ram16)
    monkeypatch.setattr(studio, "nvidia_gpus", lambda: [])
    radeon = [{"backend": "rocm", "name": "Radeon", "memory_total": 16 * G, "memory_free": 15 * G, "reserve": G // 2,
               "unified": False}]
    monkeypatch.setattr(studio, "accel_devices", lambda: radeon)
    assert studio.fit_estimate(8 * G)["level"] == "gpu"
    assert studio.fit_estimate(20 * G)["level"] == "partial"
    assert studio.fit_estimate(40 * G)["level"] == "no"
    # an integrated GPU shares RAM: there is no "partial", it fits in the shared memory or it does not
    apu = [{"backend": "vulkan", "name": "Radeon Graphics", "memory_total": 12 * G, "memory_free": 10 * G,
            "reserve": G, "unified": True}]
    monkeypatch.setattr(studio, "accel_devices", lambda: apu)
    f = studio.fit_estimate(6 * G)
    assert f["level"] == "gpu" and "shared" in f["note"]
    assert studio.fit_estimate(10 * G)["level"] == "no"
    monkeypatch.setattr(studio, "accel_devices", lambda: [])
    assert studio.fit_estimate(6 * G)["level"] == "cpu" and studio.fit_estimate(20 * G)["level"] == "no"
    assert studio.fit_estimate(0)["level"] is None


def test_recommended_download(monkeypatch):
    monkeypatch.setattr(studio, "Host", Ram16)
    monkeypatch.setattr(studio, "nvidia_gpus", lambda: [{"memory_free": int(7.5 * G), "memory_total": 8 * G}])
    files = [("Q2_K", 3.2), ("Q4_K_M", 4.9), ("Q6_K", 6.6), ("Q8_0", 8.5), ("F16", 16.1)]
    monkeypatch.setattr(studio, "_hf_get", lambda url, timeout=25: {"siblings": [
        {"rfilename": f"M-{q}.gguf", "size": int(s * G)} for q, s in files] + [
        {"rfilename": "mmproj-M-f16.gguf", "size": int(0.6 * G)}]})
    out = studio.hub_files("x/M-GGUF")
    assert [g["quant"] for g in out][:2] == ["F16", "Q2_K"] or out[0]["size"] <= out[1]["size"]
    assert [g["size"] for g in out] == sorted(g["size"] for g in out)               # smallest first
    rec = [g for g in out if g.get("recommended")]
    assert len(rec) == 1 and rec[0]["quant"] == "Q4_K_M"                           # largest that fully fits
    assert {g["quant"]: g["fit"]["level"] for g in out}["Q8_0"] == "partial"
