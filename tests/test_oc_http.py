from email.parser import BytesParser
from http.server import BaseHTTPRequestHandler
from pathlib import Path
import socket
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "remote"))

import oc_http
from omnicortex_remote import http_utils as remote_http


def handler_for(headers, conn=None, rfile=None):
    parsed = BytesParser().parsebytes(headers + b"\r\n\r\n")
    return type("TestHandler", (), {})() if False else __import__(
        "types"
    ).SimpleNamespace(
        headers=parsed,
        connection=conn,
        rfile=rfile,
        close_connection=False,
    )


def expect_value_error(fn):
    try:
        fn()
    except ValueError:
        return
    raise AssertionError("invalid HTTP request was accepted")


def test_content_length_accepts_single_bounded_value():
    h = handler_for(b"Content-Length: 12")
    assert oc_http.content_length(h, 12) == 12


def test_content_length_rejects_ambiguous_framing_and_closes():
    for headers in (
        b"Content-Length: 2\r\nContent-Length: 2",
        b"Transfer-Encoding: chunked\r\nContent-Length: 2",
    ):
        h = handler_for(headers)
        expect_value_error(lambda: oc_http.content_length(h, 10))
        assert h.close_connection is True


def test_content_length_rejects_invalid_or_oversized_values_and_closes():
    for value in (b"-1", b"1, 1", b"999999999999999999999", b"11"):
        h = handler_for(b"Content-Length: " + value)
        expect_value_error(lambda: oc_http.content_length(h, 10))
        assert h.close_connection is True


def test_read_body_requires_length_and_closes():
    h = handler_for(b"")
    expect_value_error(lambda: oc_http.read_body(h, 10, required=True))
    assert h.close_connection is True


def test_read_body_reads_exact_length():
    receiver, sender = socket.socketpair()
    stream = receiver.makefile("rb")
    try:
        sender.sendall(b"hello")
        h = handler_for(b"Content-Length: 5", receiver, stream)
        assert oc_http.read_body(h, 5, total_timeout=1) == b"hello"
    finally:
        stream.close()
        sender.close()
        receiver.close()


def test_read_body_rejects_short_body_and_marks_connection_closed():
    receiver, sender = socket.socketpair()
    stream = receiver.makefile("rb")
    try:
        sender.sendall(b"he")
        sender.shutdown(socket.SHUT_WR)
        h = handler_for(b"Content-Length: 5", receiver, stream)
        expect_value_error(lambda: oc_http.read_body(h, 5, total_timeout=1))
        assert h.close_connection is True
    finally:
        stream.close()
        sender.close()
        receiver.close()


def test_remote_body_reader_closes_on_invalid_framing():
    for headers in (
        b"Transfer-Encoding: chunked\r\nContent-Length: 2",
        b"Content-Length: 2\r\nContent-Length: 2",
        b"Content-Length: 999",
    ):
        h = handler_for(headers)
        expect_value_error(lambda: remote_http.read_body(h, 10))
        assert h.close_connection is True


def test_header_deadline_releases_only_handler_slot():
    class H(oc_http.HeaderDeadlineMixin, BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Length", "2")
            self.end_headers()
            self.wfile.write(b"ok")

        def log_message(self, *args):
            pass

    server = oc_http.BoundedThreadingHTTPServer(
        ("127.0.0.1", 0), H, max_threads=1, header_timeout=0.25
    )
    thread = threading.Thread(
        target=lambda: server.serve_forever(poll_interval=0.01),
        daemon=True,
    )
    thread.start()
    idle = socket.create_connection(server.server_address, timeout=2)
    try:
        # This client consumes the only handler slot but never sends a request line.
        time.sleep(0.45)

        with socket.create_connection(server.server_address, timeout=2) as active:
            active.sendall(
                b"GET / HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n"
            )
            active.settimeout(2)
            with active.makefile("rb") as response:
                status = response.readline()
            assert status.startswith(b"HTTP/1.0 200"), status

        idle.settimeout(1)
        assert idle.recv(1) == b""
    finally:
        idle.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

def test_read_body_retries_socket_timeout_before_absolute_deadline():
    from types import SimpleNamespace

    class Headers:
        def get_all(self, name, default=None):
            if name.lower() == "content-length":
                return ["2"]
            return default if default is not None else []

    class Connection:
        def __init__(self):
            self.timeout = None

        def gettimeout(self):
            return self.timeout

        def settimeout(self, value):
            self.timeout = value

    class FlakyReader:
        def __init__(self):
            self.calls = 0

        def read1(self, size):
            self.calls += 1
            if self.calls == 1:
                raise socket.timeout("temporary timeout")
            return b"ok"

    for helper in (oc_http, remote_http):
        reader = FlakyReader()
        handler = SimpleNamespace(
            headers=Headers(),
            connection=Connection(),
            rfile=reader,
            close_connection=False,
        )

        assert helper.read_body(handler, 2, total_timeout=2) == b"ok"
        assert reader.calls == 2
        assert handler.close_connection is False
