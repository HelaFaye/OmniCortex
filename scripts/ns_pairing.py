#!/usr/bin/env python3
"""
Device pairing and linked hosts: use one machine's models from another
(LM Link-style), without sharing the master token.

On the host, the owner creates a one-time pairing code (Studio: Connect → Devices, or
`POST /api/pair/start`). It is shown as a link:

    https://studio.lan:7870/pair#c=K7QX-M2PA-9DTE&fp=3f9a…   (single use, short-lived)

Each link grants either persistent access (until revoked) or temporary access
that expires after a duration the host's owner chose; the host's policy
(PairingPolicy, Studio's --pair-* flags) sets how long links stay claimable,
which durations are offered, the longest allowed, and whether persistent
access may be granted at all.

`fp` is the SHA-256 of the host's TLS certificate. Whoever claims the link
checks that the server presents exactly that certificate before sending the
code, so a self-signed LAN certificate is pinned instead of clicked through,
and a machine in the middle cannot take the code. The claim returns a device
token of its own: revocable on the host without touching anything else, and
stored there only as a hash.

    python scripts/ns_pairing.py claim 'https://studio.lan:7870/pair#c=…&fp=…' --name laptop \\
        --out ~/.config/neuronscope/studio-laptop.token

A browser opening the link is paired the same way and logged in. Another
Studio can claim it as a *linked host* and then serve the remote's models
next to its own.
"""
from __future__ import annotations

import argparse
import hashlib
import hmac
import http.client
import json
import os
import re
import secrets
import ssl
import sys
import threading
import time
import urllib.parse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ns_security as sec  # noqa: E402

CODE_ALPHABET = "ABCDEFGHJKMNPQRSTUVWXYZ23456789"   # no 0/O, 1/I/L
CODE_LEN = 12                                        # ~59 bits; single use, short-lived, throttled
CODE_TTL = 300
EXPIRED_KEEP = 30 * 86400                            # expired devices stay listed this long, then go


def parse_duration(v) -> int:
    """'90m', '8h', '7d', '2w' or plain seconds -> seconds."""
    if isinstance(v, (int, float)):
        return int(v)
    m = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*([smhdw]?)\s*", str(v).lower())
    if not m:
        raise ValueError(f"not a duration: {v!r} (use e.g. 90m, 8h, 7d)")
    return int(float(m.group(1)) * {"": 1, "s": 1, "m": 60, "h": 3600, "d": 86400, "w": 604800}[m.group(2)])


def fmt_duration(sec: int) -> str:
    for unit, n in (("d", 86400), ("h", 3600), ("m", 60)):
        if sec >= n and sec % n == 0:
            return f"{sec // n}{unit}"
    return f"{sec}s"


class PairingPolicy:
    """What the host allows, set by its owner (Studio flags):

    code_ttl          how long a pairing link can be claimed (seconds)
    presets           temporary-access durations offered when pairing
    max_ttl           the longest temporary access allowed
    allow_persistent  whether access that lasts until revoked may be granted
    default           what a new pairing grants unless chosen otherwise
                      ("persistent" or a duration)
    """

    def __init__(self, code_ttl=CODE_TTL, presets=("1h", "8h", "1d", "7d", "30d"), max_ttl="90d",
                 allow_persistent=True, default="persistent"):
        self.code_ttl = max(30, min(parse_duration(code_ttl), 86400))
        self.max_ttl = parse_duration(max_ttl)
        self.presets = sorted({parse_duration(p) for p in presets if parse_duration(p) <= self.max_ttl})
        self.allow_persistent = bool(allow_persistent)
        if default == "persistent" and not self.allow_persistent:
            default = self.presets[-1] if self.presets else self.max_ttl
        self.default = None if default == "persistent" else min(parse_duration(default), self.max_ttl)

    def access(self, persistent=None, ttl=None) -> int | None:
        """Resolve a requested lifetime against the policy -> seconds, or None for persistent."""
        if persistent is None and ttl in (None, ""):
            return self.default
        if persistent:
            if not self.allow_persistent:
                raise ValueError("this host only grants temporary access (persistent pairing is disabled)")
            return None
        sec = parse_duration(ttl if ttl not in (None, "") else (self.default or self.max_ttl))
        if sec < 60:
            raise ValueError("temporary access must last at least a minute")
        if sec > self.max_ttl:
            raise ValueError(f"this host allows temporary access of at most {fmt_duration(self.max_ttl)}")
        return sec

    def describe(self) -> dict:
        return {"code_ttl": self.code_ttl, "presets": [fmt_duration(p) for p in self.presets],
                "max_ttl": fmt_duration(self.max_ttl), "allow_persistent": self.allow_persistent,
                "default": "persistent" if self.default is None else fmt_duration(self.default)}


def _hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def normalize_code(code: str) -> str:
    return "".join(c for c in code.upper() if c in CODE_ALPHABET)


def cert_fingerprint(certfile: str) -> str:
    """SHA-256 of a PEM certificate's DER bytes, lowercase hex."""
    pem = Path(certfile).read_text()
    der = ssl.PEM_cert_to_DER_cert(pem[pem.index("-----BEGIN CERTIFICATE-----"):])
    return hashlib.sha256(der).hexdigest()


class DeviceRegistry:
    """Paired devices: {id, name, token_sha256, created, last_seen, expires}.
    `expires` is None for persistent access (until revoked) or a Unix time for
    temporary access; an expired device is refused on every request. Tokens are
    never stored, only their hashes; the file is 0600."""

    def __init__(self, path: str | Path, policy: PairingPolicy | None = None):
        self.path = Path(path).expanduser()
        self.policy = policy or PairingPolicy()
        self.lock = threading.Lock()
        self.codes: dict[str, dict] = {}        # pairing code -> {expires, access}
        try:
            self.devices = json.loads(self.path.read_text()).get("devices", [])
        except (FileNotFoundError, ValueError):
            self.devices = []
        for d in self.devices:
            d.setdefault("expires", None)        # files from before expiry existed: persistent

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        sec.write_secret_file(self.path, json.dumps({"devices": self.devices}, indent=1))

    def new_code(self, persistent=None, ttl=None) -> tuple[str, float, int | None]:
        """-> (code, code expiry, access seconds or None for persistent)."""
        access = self.policy.access(persistent, ttl)
        code = "".join(secrets.choice(CODE_ALPHABET) for _ in range(CODE_LEN))
        exp = time.time() + self.policy.code_ttl
        with self.lock:
            now = time.time()
            self.codes = {c: v for c, v in self.codes.items() if v["expires"] > now}
            self.codes[code] = {"expires": exp, "access": access}
        return code, exp, access

    def claim(self, code: str, name: str) -> dict | None:
        code = normalize_code(code)
        with self.lock:
            c = self.codes.pop(code, None)            # single use, even when expired
            if c is None or c["expires"] < time.time():
                return None
            token = sec.generate_token()
            now = time.time()
            dev = {"id": secrets.token_hex(6), "name": re.sub(r"[^\w .@-]", "", name)[:60] or "device",
                   "token_sha256": _hash(token), "created": now, "last_seen": None,
                   "expires": None if c["access"] is None else now + c["access"]}
            self.devices.append(dev)
            self._prune(now)
            self._save()
        return {"device_id": dev["id"], "name": dev["name"], "token": token, "expires": dev["expires"]}

    def check(self, token: str | None) -> dict | None:
        if not token:
            return None
        h = _hash(token)
        found = None
        for d in self.devices:                        # compare against all, in constant time each
            if hmac.compare_digest(d["token_sha256"], h):
                found = d
        if found is None:
            return None
        now = time.time()
        if found.get("expires") is not None and found["expires"] <= now:
            return None                               # temporary access has run out
        if not found.get("last_seen") or now - found["last_seen"] > 60:
            found["last_seen"] = now
            with self.lock:
                self._save()
        return found

    def _prune(self, now: float) -> None:
        self.devices = [d for d in self.devices
                        if d.get("expires") is None or d["expires"] > now - EXPIRED_KEEP]

    def list(self) -> list[dict]:
        now = time.time()
        out = []
        for d in self.devices:
            e = d.get("expires")
            out.append({**{k: d.get(k) for k in ("id", "name", "created", "last_seen", "expires")},
                        "persistent": e is None, "expired": e is not None and e <= now})
        return out

    def update(self, dev_id: str, persistent=None, ttl=None) -> dict:
        """Change a device's access: make it persistent, or set it to expire `ttl`
        from now (extend, shorten, or renew an expired device)."""
        access = self.policy.access(persistent, ttl)
        with self.lock:
            d = next((x for x in self.devices if x["id"] == dev_id), None)
            if d is None:
                raise KeyError(dev_id)
            d["expires"] = None if access is None else time.time() + access
            self._save()
        return next(x for x in self.list() if x["id"] == dev_id)

    def revoke(self, dev_id: str) -> bool:
        with self.lock:
            n = len(self.devices)
            self.devices = [d for d in self.devices if d["id"] != dev_id]
            if len(self.devices) != n:
                self._save()
                return True
        return False


# ---------------------------------------------------------------- client side

def parse_link(link: str) -> tuple[str, str, str]:
    """-> (base URL, code, fingerprint or '')."""
    u = urllib.parse.urlsplit(link.strip())
    frag = urllib.parse.parse_qs(u.fragment)
    code = normalize_code((frag.get("c") or [""])[0])
    fp = (frag.get("fp") or [""])[0].lower().replace(":", "")
    if u.scheme not in ("http", "https") or not u.netloc or not code:
        raise ValueError("not a pairing link (expected https://host:port/pair#c=CODE&fp=FINGERPRINT)")
    if fp and not re.fullmatch(r"[0-9a-f]{64}", fp):
        raise ValueError("bad certificate fingerprint in the link")
    return f"{u.scheme}://{u.netloc}", code, fp


class PinnedHTTPSConnection(http.client.HTTPSConnection):
    """HTTPS that trusts exactly one certificate, by SHA-256 fingerprint."""

    def __init__(self, host, fingerprint: str, **kw):
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE                # replaced by the fingerprint check below
        super().__init__(host, context=ctx, **kw)
        self.fingerprint = fingerprint

    def connect(self):
        super().connect()
        der = self.sock.getpeercert(binary_form=True)
        if not der or not hmac.compare_digest(hashlib.sha256(der).hexdigest(), self.fingerprint):
            self.sock.close()
            raise ssl.SSLError("server certificate does not match the pinned fingerprint")


_WARNED: set = set()


def request(base: str, method: str, path: str, body: dict | None = None, token: str = "", fingerprint: str = "",
            timeout: float = 30, stream: bool = False):
    """JSON request to a Studio, pinning `fingerprint` when given. With stream=True
    the open response is returned (the caller reads and closes it)."""
    u = urllib.parse.urlsplit(base)
    if u.scheme == "https":
        conn = (PinnedHTTPSConnection(u.netloc, fingerprint, timeout=timeout) if fingerprint
                else http.client.HTTPSConnection(u.netloc, timeout=timeout, context=ssl.create_default_context()))
    else:
        if not sec.is_loopback(u.hostname or "") and base not in _WARNED:
            _WARNED.add(base)
            print(f"warning: {base} is plain HTTP; the device token crosses the network unencrypted "
                  "(fine only inside a VPN)", file=sys.stderr)
        conn = http.client.HTTPConnection(u.netloc, timeout=timeout)
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    conn.request(method, path, body=None if body is None else json.dumps(body).encode(), headers=headers)
    r = conn.getresponse()
    if stream:
        return r
    data = r.read()
    conn.close()
    try:
        out = json.loads(data or b"{}")
    except ValueError:
        out = {"error": data[:200].decode(errors="replace")}
    if r.status >= 400:
        raise RuntimeError(f"{base}{path}: HTTP {r.status}: {out.get('error', out)}")
    return out


def claim(link: str, name: str) -> dict:
    base, code, fp = parse_link(link)
    if base.startswith("https") and not fp:
        raise ValueError("an https pairing link must carry the certificate fingerprint (fp=)")
    r = request(base, "POST", "/api/pair/claim", {"code": code, "name": name}, fingerprint=fp)
    return {"url": base, "token": r["token"], "device_id": r["device_id"], "fingerprint": fp,
            "expires": r.get("expires")}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("claim", help="pair this machine using a link from the host")
    c.add_argument("link")
    c.add_argument("--name", default=os.uname().nodename if hasattr(os, "uname") else "device")
    c.add_argument("--out", required=True, help="file for the device token (written 0600)")
    a = p.parse_args(argv)
    r = claim(a.link, a.name)
    sec.write_secret_file(Path(a.out).expanduser(), r["token"])
    print(json.dumps({"url": r["url"], "device_id": r["device_id"], "fingerprint": r["fingerprint"],
                      "token_file": str(Path(a.out).expanduser()),
                      "access": "persistent (until revoked)" if r["expires"] is None else
                      "temporary, until " + time.strftime("%Y-%m-%d %H:%M", time.localtime(r["expires"]))},
                     indent=1))
    print("use it as: Authorization: Bearer $(cat token_file); pin the fingerprint, or pass the cert "
          "with --cafile in OmniCortex clients", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
