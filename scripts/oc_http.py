"""Bounded HTTP primitives for OmniCortex control services."""
from __future__ import annotations

import socket
import threading
import time
from http.server import ThreadingHTTPServer


class HeaderDeadlineMixin:
    """Apply an absolute deadline to each request's headers."""

    def handle_one_request(self):
        arm = getattr(self.server, "_arm_header_deadline", None)
        if arm:
            arm(self.connection)
        return super().handle_one_request()

    def parse_request(self):
        try:
            return super().parse_request()
        finally:
            complete = getattr(self.server, "_mark_headers_received", None)
            if complete:
                complete(self.connection)


class BoundedThreadingHTTPServer(ThreadingHTTPServer):
    """Bound handler threads and enforce request-header deadlines."""

    daemon_threads = True
    block_on_close = False

    def __init__(
        self, server_address, RequestHandlerClass,
        bind_and_activate=True, max_threads=32, header_timeout=15.0
    ):
        self.max_threads = max(1, min(int(max_threads), 256))
        self.header_timeout = max(0.1, min(float(header_timeout), 300.0))
        self._thread_slots = threading.BoundedSemaphore(self.max_threads)
        self._header_deadlines = {}
        self._header_lock = threading.Lock()
        self._header_stop = threading.Event()
        super().__init__(server_address, RequestHandlerClass, bind_and_activate)
        self._header_watch_thread = threading.Thread(
            target=self._watch_header_deadlines,
            name="oc-http-header-watchdog",
            daemon=True,
        )
        self._header_watch_thread.start()

    def _arm_header_deadline(self, conn):
        with self._header_lock:
            self._header_deadlines[conn] = time.monotonic() + self.header_timeout

    def _mark_headers_received(self, conn):
        with self._header_lock:
            if conn in self._header_deadlines:
                self._header_deadlines[conn] = None

    def _forget_header_deadline(self, conn):
        with self._header_lock:
            self._header_deadlines.pop(conn, None)

    def _watch_header_deadlines(self):
        interval = min(0.25, max(0.05, self.header_timeout / 4))
        while not self._header_stop.wait(interval):
            now = time.monotonic()
            with self._header_lock:
                expired = [
                    conn for conn, deadline in self._header_deadlines.items()
                    if deadline is not None and deadline <= now
                ]
                for conn in expired:
                    self._header_deadlines[conn] = None
            for conn in expired:
                try:
                    conn.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass

    def process_request(self, request, client_address):
        if not self._thread_slots.acquire(blocking=False):
            self.shutdown_request(request)
            return
        self._arm_header_deadline(request)
        try:
            super().process_request(request, client_address)
        except Exception:
            self._forget_header_deadline(request)
            self._thread_slots.release()
            raise

    def process_request_thread(self, request, client_address):
        try:
            super().process_request_thread(request, client_address)
        finally:
            self._forget_header_deadline(request)
            self._thread_slots.release()

    def server_close(self):
        self._header_stop.set()
        super().server_close()
        watcher = getattr(self, "_header_watch_thread", None)
        if watcher and watcher is not threading.current_thread():
            watcher.join(timeout=1)


def content_length(handler, limit: int) -> int:
    """Validate framing and close connections with invalid framing."""
    try:
        transfer_encoding = handler.headers.get_all("Transfer-Encoding", []) or []
        lengths = handler.headers.get_all("Content-Length", []) or []
        if transfer_encoding:
            raise ValueError("Transfer-Encoding is not supported")
        if len(lengths) > 1:
            raise ValueError("duplicate Content-Length")
        if not lengths:
            return 0
        raw = lengths[0].strip()
        if not raw or not raw.isascii() or not raw.isdecimal() or len(raw) > 20:
            raise ValueError("invalid Content-Length")
        size = int(raw)
        if size > limit:
            raise ValueError(f"request body too large (limit {limit} bytes)")
        return size
    except ValueError:
        handler.close_connection = True
        raise


def read_body(
    handler, limit: int, *, required: bool = False, total_timeout: float = 120.0
) -> bytes:
    """Read exactly Content-Length bytes under an absolute deadline."""
    lengths = handler.headers.get_all("Content-Length", []) or []
    if required and not lengths:
        handler.close_connection = True
        raise ValueError("Content-Length required")

    size = content_length(handler, limit)
    if size == 0:
        return b""

    conn = handler.connection
    old_timeout = conn.gettimeout()
    deadline = time.monotonic() + max(0.1, float(total_timeout))
    reader = getattr(handler.rfile, "read1", None) or handler.rfile.read
    body = bytearray()
    try:
        while len(body) < size:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise ValueError("request body deadline exceeded")
            conn.settimeout(min(remaining, 10.0))
            try:
                chunk = reader(min(64 * 1024, size - len(body)))
            except socket.timeout as exc:
                if time.monotonic() >= deadline:
                    raise ValueError("request body deadline exceeded") from exc
                continue
            if not chunk:
                raise ValueError("short request body")
            body.extend(chunk)
    except Exception:
        handler.close_connection = True
        raise
    finally:
        try:
            conn.settimeout(old_timeout)
        except OSError:
            pass
    return bytes(body)
