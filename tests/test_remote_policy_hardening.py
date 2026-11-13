import os
from pathlib import Path
import stat
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "remote"))

import oc_workspace
import tool_rules
from omnicortex_remote.agent import Store, Throttle
from remote_hosts import CallHomeHub


def test_workspace_env_rejects_secrets_and_retains_runtime_paths():
    for secret_name in ("HF_TOKEN", "OPENAI_API_KEY"):
        rejected, errors = oc_workspace.validate(
            {"env": {secret_name: "not-for-config"}},
            check_exists=False,
        )
        assert "env" in errors
        assert secret_name in errors["env"]
        assert secret_name not in rejected.get("env", {})

    clean, errors = oc_workspace.validate(
        {"env": {"VK_DRIVER_FILES": "/usr/share/vulkan/icd.d/test.json"}},
        check_exists=False,
    )
    assert not errors
    assert clean["env"] == {
        "VK_DRIVER_FILES": "/usr/share/vulkan/icd.d/test.json"
    }


def test_workspace_resolution_filters_legacy_credential_env():
    resolved = oc_workspace.resolve({"env": {"HF_TOKEN": "secret", "VK_DRIVER_FILES": "/tmp/driver.json"}})
    assert resolved["env"] == {"VK_DRIVER_FILES": "/tmp/driver.json"}


def test_tool_rule_regex_timeout_fails_closed(tmp_path, monkeypatch):
    class SlowPattern:
        def search(self, _value, *, timeout):
            raise TimeoutError("simulated regex timeout")

    monkeypatch.setattr(tool_rules.timeout_regex, "compile", lambda _pattern: SlowPattern())
    with pytest.raises(tool_rules.RuleRegexTimeout):
        tool_rules.cond_holds({"path": "payload", "regex": "safe"}, {"payload": "safe"})
    rules = tool_rules.Rules(tmp_path / "rules.json")
    rules.replace({"default": "allow", "categories": {}, "rules": [{
        "id": "slow-regex", "action": "allow", "args": [
            {"path": "payload", "regex": "safe"}
        ]
    }]})
    decision = rules.decide("test", "run", {"payload": "safe"}, record=False)
    assert decision["action"] == "ask"
    assert "timed out" in decision["reason"]


def test_tool_rule_argument_count_is_bounded():
    values = {f"v{i}": "ok" for i in range(tool_rules.MAX_ARGUMENT_VALUES + 1)}
    with pytest.raises(tool_rules.RuleRegexTimeout):
        tool_rules.cond_holds({"path": "*", "regex": "ok"}, values)


def test_callhome_file_ticket_is_single_use_and_expired_tickets_are_pruned():
    hub = CallHomeHub()
    ticket = hub.ticket("/tmp/payload.bin", "worker", ttl=60)
    assert hub.ticket_path(ticket) == "/tmp/payload.bin"
    assert hub.ticket_path(ticket) is None
    expired = hub.ticket("/tmp/expired.bin", "worker", ttl=60)
    hub.tickets[expired]["expires"] = 0
    assert hub.ticket_path(expired) is None
    assert expired not in hub.tickets


def test_remote_throttle_peer_map_is_bounded():
    throttle = Throttle()
    for number in range(4200):
        throttle.fail(f"198.51.100.{number}")
    assert len(throttle.hits) <= 4096


@pytest.mark.skipif(os.name != "posix", reason="POSIX file modes are not portable to Windows")
def test_remote_store_uses_private_directory_and_files(tmp_path):
    home = tmp_path / "private-state"
    store = Store(home)
    store.write("state.json", {"controller": "token-hash-only"})
    assert stat.S_IMODE(home.stat().st_mode) == 0o700
    assert stat.S_IMODE((home / "state.json").stat().st_mode) == 0o600


def test_tool_rules_save_is_private_and_does_not_follow_temp_symlink(tmp_path):
    import stat

    path = tmp_path / "tool-rules.json"
    victim = tmp_path / "protected-sentinel"
    victim.write_text("must remain unchanged")
    predictable_temp = path.with_suffix(".tmp")

    try:
        predictable_temp.symlink_to(victim)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks unavailable")

    rules = tool_rules.Rules(path)
    rules.save(tool_rules.blank())

    assert victim.read_text() == "must remain unchanged"
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert predictable_temp.is_symlink()


def test_callhome_invites_are_bounded():
    import remote_hosts

    hub = CallHomeHub()
    hub.MAX_INVITES = 2

    hub.invite("https://studio.example", "worker-a", "ab" * 32, "Studio")
    hub.invite("https://studio.example", "worker-b", "cd" * 32, "Studio")

    with pytest.raises(remote_hosts.RemoteError, match="invite"):
        hub.invite("https://studio.example", "worker-c", "ef" * 32, "Studio")

    assert len(hub.invites) == 2


def test_callhome_pending_requests_are_bounded_per_host():
    import threading
    import time
    import remote_hosts

    hub = CallHomeHub()
    hub.MAX_PENDING_PER_HOST = 1
    hub.seen["worker"] = time.time()
    result = {}

    def request():
        try:
            result["response"] = hub.request(
                "worker", {"method": "test"}, timeout=3
            )
        except Exception as exc:
            result["error"] = exc

    thread = threading.Thread(target=request)
    thread.start()
    deadline = time.monotonic() + 2
    while not hub.waiting and time.monotonic() < deadline:
        time.sleep(0.01)

    assert len(hub.waiting) == 1
    with pytest.raises(remote_hosts.RemoteError, match="pending"):
        hub.request("worker", {"method": "second"}, timeout=0.2)

    mid = next(iter(hub.waiting))
    assert hub.next("worker", wait=0)[0]["id"] == mid
    assert hub.reply(mid, {"ok": True}, host="worker")

    thread.join(3)
    assert not thread.is_alive()
    assert result["response"] == {"ok": True}
    assert not hub.waiting
    assert not hub.host_pending


def test_callhome_long_polls_are_bounded_per_host():
    import threading
    import time
    import remote_hosts

    hub = CallHomeHub()
    hub.MAX_POLLS_PER_HOST = 1
    result = []

    thread = threading.Thread(
        target=lambda: result.extend(hub.next("worker", wait=0.3))
    )
    thread.start()

    deadline = time.monotonic() + 2
    while not hub.polling.get("worker") and time.monotonic() < deadline:
        time.sleep(0.01)

    assert hub.polling.get("worker") == 1
    with pytest.raises(remote_hosts.RemoteError, match="long polls"):
        hub.next("worker", wait=0.1)

    thread.join(2)
    assert not thread.is_alive()
    assert not hub.polling
    assert hub.active_polls == 0


def test_callhome_progress_buffer_overflow_cancels_request():
    import threading
    import time
    import remote_hosts

    hub = CallHomeHub()
    hub.MAX_PROGRESS_PIECE_BYTES = 32
    hub.seen["worker"] = time.time()
    errors = []

    def request():
        try:
            hub.request(
                "worker", {"method": "test"}, timeout=3,
                on_progress=lambda _piece: None,
            )
        except Exception as exc:
            errors.append(exc)

    thread = threading.Thread(target=request)
    thread.start()

    deadline = time.monotonic() + 2
    while not hub.waiting and time.monotonic() < deadline:
        time.sleep(0.01)
    mid = next(iter(hub.waiting))
    hub.next("worker", wait=0)

    result = hub.progress(mid, ["x" * 64], host="worker")
    assert result == {"ok": True, "cancel": True}

    thread.join(3)
    assert not thread.is_alive()
    assert len(errors) == 1
    assert isinstance(errors[0], remote_hosts.RemoteError)
    assert "buffer limit" in str(errors[0])
    assert not hub.waiting


def test_callhome_host_registry_is_bounded_and_prunes_idle_hosts():
    import time
    import remote_hosts

    hub = CallHomeHub()
    hub.MAX_HOSTS = 2

    hub.next("worker-a", wait=0)
    hub.next("worker-b", wait=0)

    with pytest.raises(remote_hosts.RemoteError, match="host capacity"):
        hub.next("worker-c", wait=0)

    hub.seen["worker-a"] = time.time() - hub.HOST_IDLE_TTL - 1
    hub.next("worker-c", wait=0)

    assert "worker-a" not in hub.seen
    assert set(hub.seen) <= {"worker-b", "worker-c"}

def test_callhome_rejects_non_object_progress_metadata():
    import remote_hosts

    hub = remote_hosts.CallHomeHub()

    for invalid_meta in (None, "invalid", [], 42):
        message = {
            "method": "tools/call",
            "params": {"_meta": invalid_meta},
        }

        with pytest.raises(
            remote_hosts.RemoteError,
            match="_meta must be an object",
        ):
            hub.request(
                "worker",
                message,
                timeout=1,
                on_progress=lambda _piece: None,
            )

    assert not hub.waiting


def test_callhome_ticket_allows_sequential_range_retries_until_revoked():
    hub = CallHomeHub()
    ticket = hub.ticket("/tmp/payload.bin", "worker", ttl=60)

    # The first HTTP request leases the ticket exclusively.
    assert hub.ticket_acquire(ticket) == "/tmp/payload.bin"
    assert hub.ticket_acquire(ticket) is None

    # A completed HTTP response releases its lease, not the capability.
    hub.ticket_release(ticket, completed=False)
    assert hub.ticket_acquire(ticket) == "/tmp/payload.bin"
    hub.ticket_release(ticket, completed=False)

    # The host can retry again, then the controller revokes after terminal status.
    assert hub.ticket_acquire(ticket) == "/tmp/payload.bin"
    hub.ticket_release(ticket, completed=False)
    hub.ticket_revoke(ticket)

    assert hub.ticket_acquire(ticket) is None
    assert ticket not in hub.tickets


def test_callhome_ticket_path_remains_one_shot():
    hub = CallHomeHub()
    ticket = hub.ticket("/tmp/one-shot.bin", "worker", ttl=60)

    assert hub.ticket_path(ticket) == "/tmp/one-shot.bin"
    assert hub.ticket_path(ticket) is None


def test_callhome_ticket_capacity_does_not_evict_active_lease():
    hub = CallHomeHub()
    hub.max_tickets = 1

    active = hub.ticket("/tmp/active.bin", "worker", ttl=60)
    assert hub.ticket_acquire(active) == "/tmp/active.bin"

    with pytest.raises(Exception, match="ticket capacity is full"):
        hub.ticket("/tmp/second.bin", "worker", ttl=60)

    assert active in hub.tickets
    hub.ticket_release(active, completed=False)
    replacement = hub.ticket("/tmp/second.bin", "worker", ttl=60)
    assert replacement in hub.tickets
    assert active not in hub.tickets


def test_remote_send_limits_active_workers_and_history(tmp_path, monkeypatch):
    import threading
    import time
    import remote_hosts

    host = {
        "kind": "agent",
        "name": "worker",
        "method": "callhome",
    }
    remote = remote_hosts.Remote(lambda: [host], lambda _links: None)

    model = tmp_path / "model.gguf"
    model.write_bytes(b"test model contents")

    worker_count = remote_hosts.MAX_ACTIVE_REMOTE_SENDS
    barrier = threading.Barrier(worker_count + 1)
    release = threading.Event()

    def blocked_hash(_path):
        barrier.wait(timeout=10)
        if not release.wait(timeout=10):
            raise TimeoutError("test did not release blocked send workers")
        return "a" * 64

    monkeypatch.setattr(remote_hosts.nsr, "file_sha256", blocked_hash)

    send_ids = []
    try:
        send_ids = [
            remote.send("worker", str(model))
            for _ in range(worker_count)
        ]

        # Main test thread joins the barrier only after all workers reached hashing.
        barrier.wait(timeout=10)

        assert len(set(send_ids)) == worker_count
        assert all(len(send_id) >= 20 for send_id in send_ids)

        with pytest.raises(
            remote_hosts.RemoteError,
            match="too many active remote sends",
        ):
            remote.send("worker", str(model))
    finally:
        release.set()

    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        with remote.send_lock:
            finished = all(
                remote.sends[sid]["status"] == "failed"
                for sid in send_ids
            )
        if finished:
            break
        time.sleep(0.02)

    with remote.send_lock:
        assert all(remote.sends[sid]["status"] == "failed" for sid in send_ids)

    # All failed workers must release their slots, allowing another send.
    with remote.send_lock:
        remote.sends.clear()
        for number in range(remote_hosts.MAX_REMOTE_SEND_HISTORY):
            sid = f"old-{number}"
            remote.sends[sid] = {
                "id": sid,
                "host": "worker",
                "file": "old.gguf",
                "size": 1,
                "done": 1,
                "status": "done",
            }

    monkeypatch.setattr(
        remote_hosts.nsr,
        "file_sha256",
        lambda _path: "a" * 64,
    )

    new_id = remote.send("worker", str(model))
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        with remote.send_lock:
            status = remote.sends[new_id]["status"]
        if status == "failed":
            break
        time.sleep(0.02)

    with remote.send_lock:
        assert remote.sends[new_id]["status"] == "failed"
        assert len(remote.sends) == remote_hosts.MAX_REMOTE_SEND_HISTORY
        assert new_id in remote.sends
        assert "old-0" not in remote.sends

    snapshot = remote.send_snapshot(200)
    assert len(snapshot) == remote_hosts.MAX_REMOTE_SEND_HISTORY
    snapshot[-1]["status"] = "changed externally"
    with remote.send_lock:
        assert remote.sends[new_id]["status"] == "failed"
