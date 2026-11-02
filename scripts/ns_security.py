#!/usr/bin/env python3
"""Shared security helpers for OmniCortex network services.

Every service that listens on a socket (the resumable transfer receiver, the
remote tuning worker, the WebRTC signaling server) goes through this module so
the rules are identical everywhere:

* loopback binds may run without a token;
* any non-loopback bind REQUIRES a bearer token;
* any non-loopback bind REQUIRES TLS unless the operator explicitly passes
  ``--allow-plaintext`` (meaning: "this port is only reachable through a
  WireGuard/Tailscale tunnel or an HTTPS reverse proxy");
* tokens are compared in constant time and never accepted on argv silently;
* request bodies are bounded;
* repeated authentication failures from one address are throttled.
"""
from __future__ import annotations

import hashlib
import hmac
import ipaddress
import json
import os
import secrets
import socket
import ssl
import stat
import sys
import threading
import time
from pathlib import Path

TOKEN_ENV = "NS_TRANSFER_TOKEN"
TOKEN_BYTES = 32            # 256-bit bearer tokens
MIN_TOKEN_CHARS = 32        # refuse obviously weak hand-typed tokens on remote binds
MAX_JSON_BYTES = 64 * 1024
MAX_CHUNK_BYTES = 8 * 1024 * 1024


class SecurityConfigError(RuntimeError):
    """Raised when a service is asked to start in an unsafe configuration."""


# ---------------------------------------------------------------- tokens

def generate_token() -> str:
    """A 256-bit URL-safe bearer token."""
    return secrets.token_urlsafe(TOKEN_BYTES)


def token_matches(presented: str | None, expected: str) -> bool:
    """Constant-time comparison. An empty ``expected`` never matches here;
    callers decide separately whether auth is required at all."""
    if not expected or presented is None:
        return False
    return secrets.compare_digest(presented.encode("utf-8"), expected.encode("utf-8"))


def bearer_ok(header: str | None, expected: str) -> bool:
    """Check an ``Authorization: Bearer ...`` header.

    If ``expected`` is empty the service was started without auth, which
    :func:`check_bind` only permits on loopback, so the request is allowed."""
    if not expected:
        return True
    header = header or ""
    if not header.startswith("Bearer "):
        return False
    return token_matches(header[7:], expected)


def write_secret_file(path: Path, value: str) -> Path:
    """Write ``value`` to ``path`` with 0600 permissions (created atomically)."""
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(path.parent, 0o700)
    except OSError:
        pass
    tmp = path.with_name(path.name + ".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(value.strip() + "\n")
    os.replace(tmp, path)
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass
    return path


def read_secret_file(path: Path) -> str:
    path = Path(path).expanduser()
    if os.name == "posix":
        mode = path.stat().st_mode
        if mode & (stat.S_IRWXG | stat.S_IRWXO):
            print(f"warning: {path} is readable by group/other; run: chmod 600 {path}",
                  file=sys.stderr)
    return path.read_text(encoding="utf-8").strip()


def resolve_token(cli_token: str = "", token_file: str = "", *, warn_argv: bool = True) -> str:
    """Pick a token from (in order) --token-file, $NS_TRANSFER_TOKEN, --token.

    Passing the secret on the command line still works for compatibility but
    prints a warning: argv is visible to every local user via ``ps``."""
    if token_file:
        return read_secret_file(Path(token_file))
    env = os.environ.get(TOKEN_ENV, "").strip()
    if env:
        return env
    if cli_token and warn_argv:
        print("warning: --token exposes the secret in the process list; prefer "
              f"--token-file or ${TOKEN_ENV}", file=sys.stderr)
    return cli_token.strip()


# ---------------------------------------------------------------- signed results

RESULT_SIG_CONTEXT = b"neuronscope-signed-result-v1"


def _result_key(token: str) -> bytes:
    # A derived key, so a signature never doubles as anything the token itself authorises.
    return hmac.new(token.encode(), RESULT_SIG_CONTEXT, hashlib.sha256).digest()


def _canonical(obj: dict) -> bytes:
    return json.dumps({k: v for k, v in obj.items() if k != "sig"}, sort_keys=True,
                      separators=(",", ":"), ensure_ascii=False).encode()


def sign_result(token: str, obj: dict) -> dict:
    """Return obj with an HMAC-SHA256 "sig" over its canonical JSON."""
    out = {k: v for k, v in obj.items() if k != "sig"}
    if token:
        out["sig"] = hmac.new(_result_key(token), _canonical(out), hashlib.sha256).hexdigest()
    return out


def verify_result(token: str, obj: dict) -> bool:
    sig = obj.get("sig")
    if not token or not isinstance(sig, str):
        return False
    return hmac.compare_digest(sig, hmac.new(_result_key(token), _canonical(obj), hashlib.sha256).hexdigest())


# ---------------------------------------------------------------- binds

def is_loopback(host: str) -> bool:
    host = (host or "").strip("[]")
    if host in {"localhost", ""}:
        return host == "localhost"
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        try:
            infos = socket.getaddrinfo(host, None)
        except OSError:
            return False
        return bool(infos) and all(ipaddress.ip_address(i[4][0]).is_loopback for i in infos)


def check_bind(host: str, token: str, *, tls: bool, allow_plaintext: bool) -> list[str]:
    """Validate a listen address against the security baseline.

    Returns a list of warnings; raises :class:`SecurityConfigError` for
    configurations that must not start."""
    warnings: list[str] = []
    if is_loopback(host):
        return warnings
    if not token:
        raise SecurityConfigError(
            f"refusing to listen on {host!r} without a token. Generate one with:\n"
            "  python scripts/ns_security.py token --out ~/.config/neuronscope/transfer.token\n"
            "then pass --token-file, or bind to 127.0.0.1.")
    if len(token) < MIN_TOKEN_CHARS:
        raise SecurityConfigError(
            f"token is too short ({len(token)} chars) for a network-facing service; "
            f"use at least {MIN_TOKEN_CHARS} characters (ns_security.py token generates 256-bit tokens).")
    if not tls:
        if not allow_plaintext:
            raise SecurityConfigError(
                f"refusing plaintext HTTP on {host!r}: the token and file contents would cross the "
                "network unencrypted. Either pass --tls-cert/--tls-key, or pass --allow-plaintext if "
                "this port is reachable only through WireGuard/Tailscale or an HTTPS reverse proxy.")
        warnings.append("plaintext HTTP on a non-loopback address: make sure this port is only "
                        "reachable through a VPN tunnel or a TLS-terminating reverse proxy.")
    return warnings


def loopback_only(host: str, name: str, allow_unauthenticated: bool = False) -> None:
    """For local tools that have no authentication of their own."""
    if is_loopback(host):
        return
    if allow_unauthenticated:
        print(f"warning: {name} has no authentication and is listening on {host}; anyone who can "
              "reach the port can use it.", file=sys.stderr)
        return
    raise SystemExit(
        f"error: {name} has no authentication, so it refuses to listen on {host!r}.\n"
        "Bind to 127.0.0.1 and reach it through an SSH tunnel (ssh -L PORT:127.0.0.1:PORT host) "
        "or run it under viz/hub.py, or pass --allow-unauthenticated on a network you trust.")


def server_ssl_context(certfile: str, keyfile: str) -> ssl.SSLContext:
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.minimum_version = ssl.TLSVersion.TLSv1_2
    ctx.load_cert_chain(certfile, keyfile)
    return ctx


def client_ssl_context(cafile: str = "", insecure: bool = False) -> ssl.SSLContext:
    ctx = ssl.create_default_context(cafile=cafile or None)
    if insecure:
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
    return ctx


def warn_plaintext_url(url: str, token: str) -> None:
    """Client-side counterpart of :func:`check_bind`."""
    from urllib.parse import urlparse
    u = urlparse(url)
    if u.scheme == "http" and not is_loopback(u.hostname or "") and token:
        print(f"warning: sending a bearer token over plaintext HTTP to {u.hostname}; "
              "use https:// or a VPN tunnel.", file=sys.stderr)


# ---------------------------------------------------------------- throttling

class FailureThrottle:
    """Lock an address out after too many failed attempts in a window.

    Used for bad bearer tokens and for WebRTC room joins with a wrong
    invitation secret, so neither can be brute-forced at line rate."""

    def __init__(self, max_failures: int = 10, window: float = 60.0, lockout: float = 300.0):
        self.max_failures = max_failures
        self.window = window
        self.lockout = lockout
        self._fails: dict[str, list[float]] = {}
        self._locked: dict[str, float] = {}
        self._lock = threading.Lock()

    def blocked(self, addr: str) -> bool:
        now = time.monotonic()
        with self._lock:
            until = self._locked.get(addr)
            if until and until > now:
                return True
            if until:
                self._locked.pop(addr, None)
            return False

    def fail(self, addr: str) -> None:
        now = time.monotonic()
        with self._lock:
            xs = [t for t in self._fails.get(addr, []) if now - t < self.window]
            xs.append(now)
            self._fails[addr] = xs
            if len(xs) >= self.max_failures:
                self._locked[addr] = now + self.lockout
                self._fails.pop(addr, None)

    def succeed(self, addr: str) -> None:
        with self._lock:
            self._fails.pop(addr, None)


def add_server_security_args(p) -> None:
    """Common argparse flags for every listening service."""
    p.add_argument("--token", default="", help="bearer token (discouraged: visible in ps; "
                   f"prefer --token-file or ${TOKEN_ENV})")
    p.add_argument("--token-file", default="", help="file containing the bearer token (chmod 600)")
    p.add_argument("--tls-cert", default="", help="PEM certificate; enables HTTPS")
    p.add_argument("--tls-key", default="", help="PEM private key for --tls-cert")
    p.add_argument("--allow-plaintext", action="store_true",
                   help="permit plain HTTP on a non-loopback bind (only behind a VPN or TLS proxy)")


def add_client_security_args(p) -> None:
    p.add_argument("--token", default="", help=f"bearer token (prefer --token-file or ${TOKEN_ENV})")
    p.add_argument("--token-file", default="", help="file containing the bearer token")
    p.add_argument("--cafile", default="", help="CA bundle / self-signed cert to trust for https://")


def main(argv=None) -> int:
    import argparse
    p = argparse.ArgumentParser(description="OmniCortex security utilities")
    sub = p.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("token", help="generate a 256-bit bearer token")
    t.add_argument("--out", default="", help="write to this file with 0600 permissions instead of stdout")
    c = sub.add_parser("selfsigned", help="create a self-signed TLS cert for LAN use (needs `cryptography`)")
    c.add_argument("--host", action="append", default=[], help="DNS name or IP to include (repeatable)")
    c.add_argument("--cert", default="ns-cert.pem")
    c.add_argument("--key", default="ns-key.pem")
    c.add_argument("--days", type=int, default=365)
    a = p.parse_args(argv)
    if a.cmd == "token":
        tok = generate_token()
        if a.out:
            print(f"wrote {write_secret_file(Path(a.out), tok)} (0600)")
        else:
            print(tok)
        return 0
    if a.cmd == "selfsigned":
        return _selfsigned(a.host or ["localhost", "127.0.0.1"], a.cert, a.key, a.days)
    return 1


def _selfsigned(hosts: list[str], cert: str, key: str, days: int) -> int:
    try:
        from cryptography import x509
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import ec
        from cryptography.x509.oid import NameOID
    except ImportError:
        print("pip install cryptography   (or use: openssl req -x509 -newkey ec "
              "-pkeyopt ec_paramgen_curve:P-256 -nodes -keyout ns-key.pem -out ns-cert.pem "
              "-days 365 -subj /CN=neuronscope)", file=sys.stderr)
        return 2
    import datetime
    k = ec.generate_private_key(ec.SECP256R1())
    sans = []
    for h in hosts:
        try:
            sans.append(x509.IPAddress(ipaddress.ip_address(h)))
        except ValueError:
            sans.append(x509.DNSName(h))
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, hosts[0])])
    now = datetime.datetime.now(datetime.timezone.utc)
    crt = (x509.CertificateBuilder().subject_name(name).issuer_name(name)
           .public_key(k.public_key()).serial_number(x509.random_serial_number())
           .not_valid_before(now - datetime.timedelta(minutes=5))
           .not_valid_after(now + datetime.timedelta(days=days))
           .add_extension(x509.SubjectAlternativeName(sans), critical=False)
           .sign(k, hashes.SHA256()))
    Path(cert).write_bytes(crt.public_bytes(serialization.Encoding.PEM))
    write_secret_file(Path(key), k.private_bytes(
        serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption()).decode())
    fp = crt.fingerprint(hashes.SHA256()).hex(":")
    print(f"cert: {cert}\nkey:  {key} (0600)\nSHA-256 fingerprint: {fp}")
    print("Clients: pass --cafile", cert, "(or install it in the browser trust store).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
