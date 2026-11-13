"""Audio stems: a band map, problem flags, and a frequency-domain repair. numpy and the standard library only."""
import argparse
import json
import struct
import sys
import wave
from pathlib import Path

import numpy as np

BANDS = [(20, 60, "sub"), (60, 250, "bass"), (250, 500, "low-mid"), (500, 2000, "mid"), (2000, 5000, "presence"),
         (5000, 9000, "sibilance"), (9000, 20000, "air")]
# what a band sticking out usually means
MEANING = {"bass": "boom", "low-mid": "mud", "presence": "harshness", "sibilance": "sibilance", "air": "hiss"}


def read_wav(path):
    with wave.open(str(path), "rb") as w:
        n, ch, sw, sr = w.getnframes(), w.getnchannels(), w.getsampwidth(), w.getframerate()
        raw = w.readframes(n)
    if sw == 2:
        x = np.frombuffer(raw, "<i2").astype(np.float32) / 32768
    elif sw == 3:
        b = np.frombuffer(raw, np.uint8).reshape(-1, 3)
        x = ((b[:, 0].astype(np.int32) | (b[:, 1].astype(np.int32) << 8) | (b[:, 2].astype(np.int32) << 16)) << 8 >> 8).astype(np.float32) / 8388608
    elif sw == 4:
        x = np.frombuffer(raw, "<i4").astype(np.float32) / 2147483648
    elif sw == 1:
        x = (np.frombuffer(raw, np.uint8).astype(np.float32) - 128) / 128
    else:
        raise ValueError(f"{path}: {sw * 8}-bit WAV not supported")
    return x.reshape(-1, ch), sr, sw


def write_wav(path, x, sr, sw=2):
    x = np.clip(x, -1, 1)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(x.shape[1])
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((x * 32767).astype("<i2").tobytes())


def band_db(x, sr):
    # each channel's power, averaged: a mono mixdown would hide whatever is out of phase between channels
    win = np.hanning(len(x))
    spec = np.mean([np.abs(np.fft.rfft(x[:, c] * win)) ** 2 for c in range(x.shape[1])], axis=0)
    f = np.fft.rfftfreq(len(x), 1 / sr)
    out = {}
    for lo, hi, name in BANDS:
        m = (f >= lo) & (f < min(hi, sr / 2))
        if m.any():
            out[name] = 10 * np.log10(spec[m].mean() + 1e-20)
    return out


def analyze(x, sr):
    peak = float(np.abs(x).max())
    rms = float(np.sqrt((x ** 2).mean()) + 1e-12)
    rep = {"seconds": round(len(x) / sr, 2), "channels": x.shape[1], "rate": sr,
           "peak_db": round(20 * np.log10(peak + 1e-12), 2), "rms_db": round(20 * np.log10(rms), 2),
           "crest_db": round(20 * np.log10((peak + 1e-12) / rms), 2),
           "clipped": round(float((np.abs(x) >= 0.999).mean()), 6), "dc": [round(float(v), 5) for v in x.mean(0)]}
    if x.shape[1] == 2:
        a, b = x[:, 0] - x[:, 0].mean(), x[:, 1] - x[:, 1].mean()
        rep["stereo_correlation"] = round(float((a * b).sum() / (np.sqrt((a * a).sum() * (b * b).sum()) + 1e-12)), 3)
    bd = band_db(x, sr)
    rep["bands_db"] = {k: round(v, 2) for k, v in bd.items()}
    names = list(bd)
    flags = []
    # Music falls off with frequency, so a band is judged against the spectrum's own tilt (a straight line through
    # every band's level against log frequency, fitted without the band itself), not against its neighbours.
    centre = {b[2]: np.log10(np.sqrt(b[0] * min(b[1], sr / 2))) for b in BANDS if b[2] in bd}
    for n in names:
        others = [k for k in names if k != n]
        if len(others) < 3:
            break
        k, c = np.polyfit([centre[o] for o in others], [bd[o] for o in others], 1)
        excess = bd[n] - float(k * centre[n] + c)
        limit = {"presence": 4.0, "sibilance": 5.0, "bass": 6.0, "low-mid": 5.0, "air": 8.0}.get(n)
        if limit is not None and excess > limit:
            flags.append({"band": n, "problem": MEANING.get(n, "excess"), "excess_db": round(excess - limit + 1, 2)})
    if rep["clipped"] > 0.0005:
        flags.append({"problem": "clipping", "share": rep["clipped"]})
    if any(abs(d) > 0.01 for d in rep["dc"]):
        flags.append({"problem": "DC offset", "dc": rep["dc"]})
    if rep.get("stereo_correlation", 1) < -0.2:
        flags.append({"problem": "phase cancellation (channels out of phase)", "correlation": rep["stereo_correlation"]})
    rep["flags"] = flags
    return rep


def repair(x, sr, gains_db: dict, block=8192):
    """Smooth per-band gains in the frequency domain (overlap-add, Hann windows), DC removed."""
    x = x - x.mean(0)
    if not gains_db:
        return x
    hop, win = block // 2, np.hanning(block).astype(np.float32)
    f = np.fft.rfftfreq(block, 1 / sr)
    g = np.ones_like(f)
    for (lo, hi), db in gains_db.items():
        edge = (lo + hi) / 2 * 0.15
        ramp = np.clip(np.minimum((f - lo + edge) / (2 * edge), (hi + edge - f) / (2 * edge)), 0, 1)
        g *= 10 ** (db * ramp / 20)
    out = np.zeros_like(x)
    norm = np.zeros(len(x), np.float32)
    pad = np.vstack([x, np.zeros((block, x.shape[1]), x.dtype)])
    for s in range(0, len(x), hop):
        seg = pad[s:s + block] * win[:, None]
        y = np.fft.irfft(np.fft.rfft(seg, axis=0) * g[:, None], n=block, axis=0) * win[:, None]
        e = min(block, len(x) - s)
        out[s:s + e] += y[:e]
        norm[s:s + e] += win[:e] ** 2
    return out / np.maximum(norm, 1e-6)[:, None]


def midi_report(path):
    try:
        import mido
    except ImportError:
        return {"error": "pip install mido to check MIDI files"}
    mf = mido.MidiFile(str(path))
    flags, on = [], {}
    for ti, tr in enumerate(mf.tracks):
        t, vels = 0, []
        for msg in tr:
            t += msg.time
            if msg.type == "note_on" and msg.velocity > 0:
                if vels and msg.velocity - np.median(vels[-16:]) > 50:
                    flags.append({"track": ti, "tick": t, "problem": "velocity spike", "velocity": msg.velocity})
                vels.append(msg.velocity)
                on[(ti, msg.channel, msg.note)] = t
            elif msg.type in ("note_off", "note_on"):
                on.pop((ti, msg.channel, msg.note), None)
    flags += [{"track": k[0], "note": k[2], "problem": "stuck note (never released)", "since_tick": v} for k, v in on.items()]
    return {"tracks": len(mf.tracks), "flags": flags[:200]}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--repair", action="store_true")
    ap.add_argument("--gain", nargs="*", default=[])
    ap.add_argument("--max-cut", type=float, default=6.0)
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    manual = {}
    for gspec in a.gain or []:
        rng, db = gspec.rsplit(":", 1)
        lo, hi = rng.split("-")
        manual[(float(lo), float(hi))] = float(db)
    report = {}
    for p in map(Path, a.inputs):
        if p.suffix.lower() in (".mid", ".midi"):
            report[p.name] = midi_report(p)
            continue
        x, sr, sw = read_wav(p)
        rep = analyze(x, sr)
        if a.repair or manual:
            gains = dict(manual)
            for fl in rep["flags"]:
                if "band" in fl:
                    lo, hi = next((b[0], b[1]) for b in BANDS if b[2] == fl["band"])
                    gains[(lo, hi)] = -min(a.max_cut, fl["excess_db"])
            y = repair(x, sr, gains)
            dst = out / f"{p.stem}.repaired.wav"
            write_wav(dst, y, sr)
            rep["repaired"] = {"file": str(dst), "gains_db": {f"{int(k[0])}-{int(k[1])}": v for k, v in gains.items()},
                               "after": {k: v for k, v in analyze(y, sr).items() if k in ("bands_db", "flags", "peak_db")}}
        report[p.name] = rep
        print(f"{p.name}: " + (", ".join(f.get("problem") + (f" ({f['band']}, +{f['excess_db']} dB)" if "band" in f else "")
                                        for f in rep["flags"]) or "nothing flagged"))
    (out / "stems-report.json").write_text(json.dumps(report, indent=1))
    print(f"report: {out / 'stems-report.json'}")


if __name__ == "__main__":
    sys.exit(main())
