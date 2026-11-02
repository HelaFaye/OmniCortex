#!/usr/bin/env python3
"""OmniCortex Model Transfer: browser/WebRTC signaling and file-transfer UI server.

This server only relays signaling metadata (SDP offers/answers, ICE
candidates). File bytes go peer-to-peer over DTLS-encrypted WebRTC
DataChannels and never touch this process.

Room access:
* every room has a short *display code* (for reading aloud / comparing on
  screen) and a 128-bit *invitation key*. The share link carries both in the
  URL fragment, which browsers never send to the server in HTTP requests;
* joining with the code alone sends a *knock* that a member must admit;
* wrong codes/keys and knocks are throttled per client address;
* WebSocket upgrades are accepted only from allowed Origins.

Peer authentication: the browser UI shows a short authentication string
derived from both DTLS certificate fingerprints. If both screens show the same
words, no one (including this signaling server) is in the middle.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import hmac
import json
import secrets
import socket
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from aiohttp import web

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ns_security as sec  # noqa: E402

CODE_ALPHABET = "ABCDEFGHJKMNPQRSTUVWXYZ23456789"
CODE_LEN = 10               # ~49.5 bits: a rendezvous label, not the secret
KEY_BYTES = 16              # 128-bit invitation key: the actual capability
ROOM_TTL = 15 * 60
MAX_PEERS = 4
MAX_KNOCKS = 4
MAX_SIGNAL_BYTES = 128 * 1024
DEFAULT_STUN = ["stun:stun.cloudflare.com:3478", "stun:stun.l.google.com:19302"]

ROOT = Path(__file__).resolve().parents[1]
WEB_ROOT = ROOT / "web"
INDEX = WEB_ROOT / "model_transfer.html"

ROOMS = web.AppKey("rooms", object)
ICE = web.AppKey("ice", dict)
ORIGINS = web.AppKey("allowed_origins", set)
TURN = web.AppKey("turn", dict)


def turn_servers(turn: dict | None, label: str) -> list[dict]:
    """TURN entries for one admitted peer.

    With a shared secret (coturn `use-auth-secret` / `static-auth-secret`, the
    "TURN REST API" scheme) each peer gets its own credentials that expire after
    `ttl` seconds: username "<expiry>:<label>", password
    base64(HMAC-SHA1(secret, username)). Nothing long-lived reaches a browser.
    Static credentials are still supported, but are only ever sent to peers
    inside a room, never from /ice.json."""
    if not turn or not turn.get("urls"):
        return []
    if turn.get("secret"):
        import base64
        user = f"{int(time.time()) + int(turn.get('ttl', 3600))}:{label}"
        cred = base64.b64encode(hmac.new(turn["secret"].encode(), user.encode(), hashlib.sha1).digest()).decode()
        return [{"urls": turn["urls"], "username": user, "credential": cred}]
    return [{"urls": turn["urls"], "username": turn.get("user", ""), "credential": turn.get("credential", "")}]


def new_room_code() -> str:
    return "".join(secrets.choice(CODE_ALPHABET) for _ in range(CODE_LEN))


def new_room_key() -> str:
    return secrets.token_urlsafe(KEY_BYTES)


def normalize_code(code: str) -> str:
    return "".join(c for c in code.upper() if c in CODE_ALPHABET)


def local_addresses() -> list[str]:
    found: list[str] = []
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            addr = info[4][0]
            if not addr.startswith("127.") and addr not in found:
                found.append(addr)
    except OSError:
        pass
    return found


class JoinError(ValueError):
    pass


@dataclass
class Conn:
    """Per-WebSocket state; ``peer`` is set once the connection is in a room."""
    ws: Any
    addr: str
    peer: "Peer | None" = None
    knock: "Knock | None" = None


@dataclass
class Peer:
    ws: Any
    peer_id: str
    room: str
    connected_at: float = field(default_factory=time.time)


@dataclass
class Knock:
    knock_id: str
    conn: Conn
    room: str


@dataclass
class Room:
    code: str
    key: str
    peers: dict[str, Peer] = field(default_factory=dict)
    knocks: dict[str, Knock] = field(default_factory=dict)
    last_empty_at: float | None = None

    def expired(self, now: float) -> bool:
        return not self.peers and self.last_empty_at is not None and now - self.last_empty_at > ROOM_TTL


class RoomManager:
    def __init__(self, throttle: sec.FailureThrottle | None = None) -> None:
        self.rooms: dict[str, Room] = {}
        self.lock = asyncio.Lock()
        self.throttle = throttle or sec.FailureThrottle(max_failures=8, window=60, lockout=600)

    async def create(self, ws) -> tuple[Room, Peer]:
        async with self.lock:
            self._gc_locked()
            room = Room(self._unique_code_locked(), new_room_key())
            self.rooms[room.code] = room
            return room, self._add_peer_locked(room, ws)

    async def join(self, code: str, key: str, ws, addr: str = "") -> tuple[Room, Peer]:
        """Join with display code + invitation key."""
        if self.throttle.blocked(addr):
            raise JoinError("rate-limited")
        async with self.lock:
            self._gc_locked()
            room = self.rooms.get(normalize_code(code))
            if room is None or not sec.token_matches(key, room.key):
                self.throttle.fail(addr)
                # Same error either way: do not reveal which rooms exist.
                raise JoinError("bad-room-or-key")
            if len(room.peers) >= MAX_PEERS:
                raise JoinError("room-full")
            self.throttle.succeed(addr)
            return room, self._add_peer_locked(room, ws)

    async def knock(self, code: str, conn: Conn) -> tuple[Knock, list[Peer]]:
        """Ask to join with the display code only; members must admit."""
        if self.throttle.blocked(conn.addr):
            raise JoinError("rate-limited")
        async with self.lock:
            room = self.rooms.get(normalize_code(code))
            # Every knock counts toward the throttle so codes cannot be enumerated.
            self.throttle.fail(conn.addr)
            if room is None or not room.peers:
                raise JoinError("bad-room-or-key")
            if len(room.knocks) >= MAX_KNOCKS:
                raise JoinError("too-many-knocks")
            k = Knock(secrets.token_hex(8), conn, room.code)
            room.knocks[k.knock_id] = k
            conn.knock = k
            return k, list(room.peers.values())

    async def admit(self, member: Peer, knock_id: str, allow: bool) -> tuple[Knock | None, Room | None, Peer | None]:
        async with self.lock:
            room = self.rooms.get(member.room)
            if room is None or member.peer_id not in room.peers:
                return None, None, None
            k = room.knocks.pop(knock_id, None)
            if k is None:
                return None, None, None
            k.conn.knock = None
            if not allow or len(room.peers) >= MAX_PEERS or k.conn.ws.closed:
                return k, room, None
            peer = self._add_peer_locked(room, k.conn.ws)
            k.conn.peer = peer
            return k, room, peer

    async def drop_knock(self, k: Knock) -> list[Peer]:
        async with self.lock:
            room = self.rooms.get(k.room)
            if room is None:
                return []
            room.knocks.pop(k.knock_id, None)
            return list(room.peers.values())

    async def create_or_join(self, requested: str | None, ws, key: str = "", addr: str = "") -> tuple[Room, Peer]:
        """Back-compat helper used by tests: create when ``requested`` is None."""
        if requested is None:
            return await self.create(ws)
        return await self.join(requested, key, ws, addr)

    async def leave(self, peer: Peer) -> list[Peer]:
        async with self.lock:
            room = self.rooms.get(peer.room)
            if room is None:
                return []
            room.peers.pop(peer.peer_id, None)
            others = list(room.peers.values())
            if not room.peers:
                room.last_empty_at = time.time()
                room.knocks.clear()
            self._gc_locked()
            return others

    async def peers(self, room: str) -> list[Peer]:
        async with self.lock:
            rec = self.rooms.get(room)
            return list(rec.peers.values()) if rec else []

    def _add_peer_locked(self, room: Room, ws) -> Peer:
        peer = Peer(ws=ws, peer_id=secrets.token_hex(8), room=room.code)
        room.peers[peer.peer_id] = peer
        room.last_empty_at = None
        return peer

    def _unique_code_locked(self) -> str:
        while True:
            code = new_room_code()
            if code not in self.rooms:
                return code

    def _gc_locked(self) -> None:
        now = time.time()
        for code, room in list(self.rooms.items()):
            if room.expired(now):
                self.rooms.pop(code, None)


async def send_json(ws, payload: dict[str, Any]) -> None:
    if not ws.closed:
        await ws.send_str(json.dumps(payload, separators=(",", ":")))


def origin_allowed(request: web.Request) -> bool:
    origin = request.headers.get("Origin")
    if origin is None:
        return True  # non-browser client; CSWSH needs a browser
    allowed: set[str] = request.app[ORIGINS]
    if origin in allowed:
        return True
    u = urlparse(origin)
    return bool(u.netloc) and u.netloc == request.host


SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
    "X-Frame-Options": "DENY",
    "Content-Security-Policy": (
        "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; "
        "connect-src 'self' ws: wss:; img-src 'self' data:; frame-ancestors 'none'"),
}


@web.middleware
async def security_headers(request: web.Request, handler):
    resp = await handler(request)
    for k, v in SECURITY_HEADERS.items():
        resp.headers.setdefault(k, v)
    return resp


async def health(_: web.Request) -> web.Response:
    return web.json_response({"ok": True, "service": "neuronscope-model-transfer", "p2p": True})


async def ice_config(request: web.Request) -> web.Response:
    return web.json_response(request.app[ICE])


async def index(_: web.Request) -> web.StreamResponse:
    return web.FileResponse(INDEX)


async def static_file(request: web.Request) -> web.StreamResponse:
    rel = request.match_info["path"].lstrip("/")
    candidate = (WEB_ROOT / rel).resolve()
    if WEB_ROOT.resolve() not in candidate.parents or not candidate.is_file():
        raise web.HTTPNotFound()
    return web.FileResponse(candidate)


async def websocket_handler(request: web.Request) -> web.WebSocketResponse:
    if not origin_allowed(request):
        raise web.HTTPForbidden(text="origin not allowed")
    ws = web.WebSocketResponse(heartbeat=20, max_msg_size=MAX_SIGNAL_BYTES)
    await ws.prepare(request)
    manager: RoomManager = request.app[ROOMS]
    conn = Conn(ws=ws, addr=request.remote or "")

    try:
        async for msg in ws:
            if msg.type == web.WSMsgType.ERROR:
                break
            if msg.type != web.WSMsgType.TEXT:
                continue
            try:
                data = json.loads(msg.data)
                if not isinstance(data, dict):
                    raise ValueError
            except ValueError:
                await send_json(ws, {"type": "error", "error": "bad-json"})
                continue
            kind = data.get("type")
            current = conn.peer

            if kind == "ping":
                await send_json(ws, {"type": "pong"})

            elif kind in ("create", "join", "knock"):
                if current is not None or conn.knock is not None:
                    await send_json(ws, {"type": "error", "error": "already-joined"})
                    continue
                room_code = data.get("room")
                key = data.get("key")
                try:
                    if kind == "create" or (kind == "join" and room_code is None):
                        room, current = await manager.create(ws)
                        conn.peer = current
                        await send_json(ws, {"type": "joined", "selfId": current.peer_id, "room": room.code,
                                             "key": room.key, "peers": [], "creator": True,
                                             "iceServers": turn_servers(request.app.get(TURN), current.peer_id)})
                        continue
                    if not isinstance(room_code, str) or len(room_code) > 32:
                        raise JoinError("bad-room")
                    if kind == "join" and isinstance(key, str) and key:
                        room, current = await manager.join(room_code, key, ws, conn.addr)
                        conn.peer = current
                        await _announce(room, current, include_key=True, turn=request.app.get(TURN))
                    else:
                        k, members = await manager.knock(room_code, conn)
                        await send_json(ws, {"type": "waiting", "room": k.room})
                        for p in members:
                            await send_json(p.ws, {"type": "knock", "knockId": k.knock_id, "addr": conn.addr})
                except JoinError as exc:
                    await send_json(ws, {"type": "error", "error": str(exc)})

            elif kind == "admit":
                if current is None:
                    await send_json(ws, {"type": "error", "error": "not-joined"})
                    continue
                k, room, peer = await manager.admit(current, str(data.get("knockId", "")), bool(data.get("allow")))
                if k is None:
                    continue
                if peer is None:
                    await send_json(k.conn.ws, {"type": "error", "error": "knock-denied"})
                else:
                    await _announce(room, peer, include_key=True, turn=request.app.get(TURN))

            elif kind == "signal":
                if current is None:
                    await send_json(ws, {"type": "error", "error": "not-joined"})
                    continue
                target = data.get("to")
                signal = data.get("data")
                if not isinstance(target, str) or not isinstance(signal, dict):
                    await send_json(ws, {"type": "error", "error": "bad-signal"})
                    continue
                dest = next((p for p in await manager.peers(current.room) if p.peer_id == target), None)
                if dest is not None:
                    await send_json(dest.ws, {"type": "signal", "from": current.peer_id, "data": signal})

            else:
                await send_json(ws, {"type": "error", "error": "unknown-type"})
    finally:
        if conn.knock is not None:
            for p in await manager.drop_knock(conn.knock):
                await send_json(p.ws, {"type": "knock-cancelled", "knockId": conn.knock.knock_id})
        if conn.peer is not None:
            for p in await manager.leave(conn.peer):
                await send_json(p.ws, {"type": "peer-left", "peerId": conn.peer.peer_id})
    return ws


async def _announce(room: Room, peer: Peer, include_key: bool, turn: dict | None = None) -> None:
    existing = [p for p in room.peers.values() if p.peer_id != peer.peer_id]
    msg = {"type": "joined", "selfId": peer.peer_id, "room": room.code,
           "peers": [p.peer_id for p in existing], "iceServers": turn_servers(turn, peer.peer_id)}
    if include_key:
        msg["key"] = room.key
    await send_json(peer.ws, msg)
    for p in existing:
        await send_json(p.ws, {"type": "peer-joined", "peerId": peer.peer_id})


def build_app(ice: dict | None = None, allowed_origins: set[str] | None = None,
              turn: dict | None = None) -> web.Application:
    app = web.Application(middlewares=[security_headers])
    app[ROOMS] = RoomManager()
    app[ICE] = ice or {"mode": "direct", "iceServers": [{"urls": DEFAULT_STUN}]}
    app[TURN] = turn or {}
    app[ORIGINS] = allowed_origins or set()
    app.router.add_get("/", index)
    app.router.add_get("/health", health)
    app.router.add_get("/ice.json", ice_config)
    app.router.add_get("/ws", websocket_handler)
    app.router.add_get("/{path:.*}", static_file)
    return app


def ice_from_args(a) -> dict:
    stun = [s for s in (a.stun or DEFAULT_STUN) if s]
    servers: list[dict] = []
    if stun and not a.no_stun:
        servers.append({"urls": stun})
    # TURN credentials are not public: they go to each peer once it is in a room.
    return {"mode": a.ice_mode, "iceServers": servers, "hasTurn": bool(a.turn_url)}


def turn_from_args(a) -> dict:
    if not a.turn_url:
        return {}
    if a.turn_secret_file:
        return {"urls": a.turn_url, "secret": sec.read_secret_file(Path(a.turn_secret_file)), "ttl": a.turn_ttl}
    if a.turn_credential:
        print("warning: static TURN credentials are long-lived and shared by every peer; prefer "
              "--turn-secret-file with coturn's use-auth-secret", file=sys.stderr)
    return {"urls": a.turn_url, "user": a.turn_user, "credential": a.turn_credential}


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="OmniCortex P2P model transfer (WebRTC) server")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8798)
    p.add_argument("--public-host", default=None, help="hostname/IP to print in share URLs")
    p.add_argument("--tls-cert", default="", help="PEM certificate; serves HTTPS/WSS")
    p.add_argument("--tls-key", default="")
    p.add_argument("--allow-plaintext", action="store_true",
                   help="allow plain HTTP on a non-loopback bind (VPN/reverse-proxy only)")
    p.add_argument("--allowed-origin", action="append", default=[],
                   help="extra Origin allowed to open the WebSocket (e.g. https://transfer.example)")
    p.add_argument("--ice-mode", choices=["direct", "relay", "lan"], default="direct",
                   help="default connectivity: direct (STUN), relay (TURN only, hides IPs), lan (no STUN)")
    p.add_argument("--stun", action="append", default=None, help="STUN URL (repeatable)")
    p.add_argument("--no-stun", action="store_true")
    p.add_argument("--turn-url", action="append", default=[], help="TURN URL, e.g. turns:turn.example:5349")
    p.add_argument("--turn-secret-file", default="",
                   help="coturn static-auth-secret (use-auth-secret): mint short-lived per-peer credentials")
    p.add_argument("--turn-ttl", type=int, default=3600,
                   help="lifetime of minted TURN credentials in seconds; longer than your longest transfer")
    p.add_argument("--turn-user", default="", help="static TURN user (discouraged; see --turn-secret-file)")
    p.add_argument("--turn-credential", default="", help="static TURN password (discouraged)")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    tls = bool(args.tls_cert and args.tls_key)
    if not sec.is_loopback(args.host) and not tls and not args.allow_plaintext:
        print("error: refusing plain HTTP on a non-loopback address. Browsers only expose the APIs this "
              "page needs (crypto.subtle, showDirectoryPicker) in a secure context, and room keys would "
              "cross the network in clear. Pass --tls-cert/--tls-key (ns_security.py selfsigned makes a "
              "LAN cert) or --allow-plaintext behind a TLS reverse proxy.", file=sys.stderr)
        return 2
    if args.ice_mode == "relay" and not args.turn_url:
        print("error: --ice-mode relay needs --turn-url", file=sys.stderr)
        return 2
    app = build_app(ice_from_args(args), set(args.allowed_origin), turn_from_args(args))
    ssl_ctx = sec.server_ssl_context(args.tls_cert, args.tls_key) if tls else None

    scheme = "https" if tls else "http"
    host_for_urls = args.public_host
    if not host_for_urls:
        addrs = local_addresses() if args.host in {"0.0.0.0", "::"} else []
        host_for_urls = addrs[0] if addrs else args.host
    print("OmniCortex Model Transfer")
    print(f"Listening: {scheme}://{args.host}:{args.port}/")
    print(f"Share URL: {scheme}://{host_for_urls}:{args.port}/")
    print(f"ICE mode:  {args.ice_mode}")
    print("File bytes go peer-to-peer over WebRTC (DTLS); this server relays handshake metadata only.")
    web.run_app(app, host=args.host, port=args.port, ssl_context=ssl_ctx, print=None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
