import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import tuning_worker


def make_store(tmp_path, workers=1):
    root = tmp_path / "worker"
    store = tuning_worker.Store(root, tmp_path / "source.gguf", tmp_path / "profile.npz",
                                tmp_path / "suppress.py", max_workers=workers, token="test-token")
    submitted = []
    store.pool.submit = lambda fn, *args: submitted.append((fn, args))
    return store, submitted


def test_submit_rejects_reused_job_id_without_replacing_original(tmp_path):
    store, submitted = make_store(tmp_path)
    original = store.submit(0.5, job_id="stable-id")
    with pytest.raises(ValueError, match="already been used"):
        store.submit(0.7, job_id="stable-id")
    assert store.jobs["stable-id"]["scale"] == original["scale"]
    assert len(submitted) == 1
    store.pool.shutdown(wait=True)


def test_submit_rejects_concurrent_jobs_that_share_output_path(tmp_path):
    store, submitted = make_store(tmp_path)
    store.submit(0.5001, job_id="scale-a")
    with pytest.raises(ValueError, match="output scale"):
        store.submit(0.5002, job_id="scale-b")
    assert set(store.jobs) == {"scale-a"}
    assert len(submitted) == 1
    store.pool.shutdown(wait=True)


def test_pending_queue_is_bounded(tmp_path):
    store, submitted = make_store(tmp_path, workers=1)
    store.submit(-1.0, job_id="one")
    store.submit(-2.0, job_id="two")
    with pytest.raises(ValueError, match="queue is full"):
        store.submit(-3.0, job_id="three")
    assert len(submitted) == 2
    store.pool.shutdown(wait=True)


def test_restart_marks_unfinished_jobs_interrupted(tmp_path):
    root = tmp_path / "worker"
    root.mkdir()
    (root / "worker_state.json").write_text(json.dumps({"jobs": {
        "stuck": {"job_id": "stuck", "scale": 0.5, "status": "running", "submitted": 1}
    }}))
    store = tuning_worker.Store(root, tmp_path / "source.gguf", tmp_path / "profile.npz",
                                tmp_path / "suppress.py", max_workers=1, token="test-token")
    assert store.jobs["stuck"]["status"] == "interrupted"
    store.pool.shutdown(wait=True)


def test_submit_rejects_job_id_with_trailing_newline(tmp_path):
    store, submitted = make_store(tmp_path)
    with pytest.raises(ValueError, match="bad job_id"):
        store.submit(0.5, job_id="invalid\n")
    assert not submitted
    store.pool.shutdown(wait=True)


def test_completed_job_reserves_its_rounded_filename(tmp_path):
    store, submitted = make_store(tmp_path)
    store.jobs["old"] = {
        "job_id": "old",
        "scale": 0.5001,
        "status": "completed",
        "submitted": 1,
    }

    with pytest.raises(ValueError, match="output scale filename"):
        store.submit(0.5002, job_id="different-scale")

    assert not submitted
    store.pool.shutdown(wait=True)


def test_same_size_output_requires_matching_provenance(tmp_path, monkeypatch):
    import shutil

    store, submitted = make_store(tmp_path)
    store.source.write_bytes(b"GGUF-source-content")
    store.profile.write_text("{}")
    store.suppressor.write_text("# suppressor version one\n")

    output = store.root / f"{store.source.stem}-{tuning_worker.scale_tag(0.5)}.gguf"
    output.write_bytes(b"X" * store.source.stat().st_size)

    # Establish the filename's scale identity, but deliberately use stale
    # provenance so the same-size output must be regenerated.
    store._provenance_path(output).write_text(json.dumps({
        "schema": 1,
        "scale_hex": float(0.5).hex(),
        "source_sha256": "0" * 64,
        "profile_sha256": "0" * 64,
        "suppressor_sha256": "0" * 64,
        "output_sha256": "0" * 64,
    }))

    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        source = Path(command[command.index("--gguf") + 1])
        destination = Path(command[command.index("--out") + 1])
        shutil.copyfile(source, destination)
        return type("Result", (), {
            "returncode": 0, "stderr": "", "stdout": ""
        })()

    monkeypatch.setattr(tuning_worker.subprocess, "run", fake_run)

    # Same file size is insufficient: this stale output must be regenerated.
    store.submit(0.5, job_id="fresh")
    fn, args = submitted.pop()
    fn(*args)
    assert store.jobs["fresh"]["status"] == "completed"
    assert store.jobs["fresh"]["resumed"] is False
    assert output.read_bytes() == store.source.read_bytes()
    assert len(calls) == 1

    # Exact inputs and exact scale may safely reuse the recorded output.
    store.submit(0.5, job_id="reuse")
    fn, args = submitted.pop()
    fn(*args)
    assert store.jobs["reuse"]["status"] == "completed"
    assert store.jobs["reuse"]["resumed"] is True
    assert len(calls) == 1

    # A changed profile invalidates provenance even if its size stays the same.
    store.profile.write_text("[]")
    store.submit(0.5, job_id="profile-changed")
    fn, args = submitted.pop()
    fn(*args)
    assert store.jobs["profile-changed"]["status"] == "completed"
    assert store.jobs["profile-changed"]["resumed"] is False
    assert len(calls) == 2

    store.pool.shutdown(wait=True)

def test_state_save_does_not_follow_predictable_tmp_symlink(tmp_path):
    store, _submitted = make_store(tmp_path)

    target = tmp_path / "outside.txt"
    target.write_text("preserve this file")
    link = store.root / "worker_state.tmp"

    try:
        link.symlink_to(target)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks unavailable")

    store.jobs["safe-id"] = {
        "job_id": "safe-id",
        "scale": 0.5,
        "status": "completed",
        "submitted": 1,
    }
    store.save()

    assert target.read_text() == "preserve this file"
    assert store.state.is_file()
    assert not store.state.is_symlink()
    store.pool.shutdown(wait=True)


def test_load_discards_one_malformed_job_without_losing_valid_history(tmp_path):
    root = tmp_path / "worker"
    root.mkdir()
    (root / "worker_state.json").write_text(json.dumps({
        "jobs": {
            "valid": {
                "job_id": "valid",
                "scale": 0.5,
                "status": "completed",
                "submitted": 1,
            },
            "bad-time": {
                "job_id": "bad-time",
                "scale": 0.7,
                "status": "completed",
                "submitted": "not-a-timestamp",
            },
            "bad-status": {
                "job_id": "bad-status",
                "scale": 0.8,
                "status": ["completed"],
                "submitted": 2,
            },
        }
    }))

    store = tuning_worker.Store(
        root, tmp_path / "source.gguf", tmp_path / "profile.npz",
        tmp_path / "suppress.py", max_workers=1, token="test-token",
    )

    assert set(store.jobs) == {"valid"}
    store.pool.shutdown(wait=True)


def test_output_scale_collision_survives_job_history_pruning(tmp_path):
    store, submitted = make_store(tmp_path)

    first_scale = 0.5001
    colliding_scale = 0.5002
    output = store.root / (
        f"{store.source.stem}-{tuning_worker.scale_tag(first_scale)}.gguf"
    )
    output.write_bytes(b"existing model")

    manifest = store._provenance_path(output)
    manifest.write_text(json.dumps({
        "scale_hex": first_scale.hex(),
    }))

    # No historical job record remains; the sidecar still reserves the name.
    assert not store.jobs

    with pytest.raises(ValueError, match="output scale filename"):
        store.submit(colliding_scale, job_id="colliding-scale")

    # The exact same scale may create a new job and verify/reuse its provenance.
    store.submit(first_scale, job_id="same-scale")
    assert len(submitted) == 1
    store.pool.shutdown(wait=True)


def test_existing_output_without_provenance_is_not_silently_reused(tmp_path):
    store, submitted = make_store(tmp_path)

    output = store.root / (
        f"{store.source.stem}-{tuning_worker.scale_tag(0.5)}.gguf"
    )
    output.write_bytes(b"untracked model")

    with pytest.raises(ValueError, match="without valid scale provenance"):
        store.submit(0.5, job_id="untracked-output")

    assert not submitted
    store.pool.shutdown(wait=True)

def test_run_releases_slot_when_initial_running_state_save_fails(tmp_path):
    store, submitted = make_store(tmp_path, workers=1)
    store.submit(0.5, job_id="initial-save-failure")
    fn, args = submitted.pop()

    original_save = store.save
    calls = 0

    def fail_once():
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("simulated initial state write failure")
        original_save()

    store.save = fail_once
    fn(*args)

    assert store.jobs["initial-save-failure"]["status"] == "failed"
    assert calls == 2

    # Both permits must be available again.
    assert store.pending_slots.acquire(blocking=False)
    assert store.pending_slots.acquire(blocking=False)
    store.pending_slots.release()
    store.pending_slots.release()
    store.pool.shutdown(wait=True)


def test_run_releases_slot_even_when_failure_state_save_fails(tmp_path):
    store, submitted = make_store(tmp_path, workers=1)
    store.submit(0.5, job_id="both-save-fail")
    fn, args = submitted.pop()

    store.save = lambda: (_ for _ in ()).throw(
        OSError("simulated persistent storage failure")
    )
    fn(*args)

    assert store.jobs["both-save-fail"]["status"] == "failed"
    assert store.pending_slots.acquire(blocking=False)
    assert store.pending_slots.acquire(blocking=False)
    store.pending_slots.release()
    store.pending_slots.release()
    store.pool.shutdown(wait=True)


def test_close_rejects_new_jobs_and_shuts_down_executor(tmp_path):
    store, submitted = make_store(tmp_path, workers=1)
    store.close()

    with pytest.raises(ValueError, match="shutting down"):
        store.submit(0.5, job_id="after-close")

    assert not submitted
    assert store.pool._shutdown is True


def test_close_waits_for_accepted_queued_jobs(tmp_path):
    store, submitted = make_store(tmp_path, workers=1)
    store.submit(0.5, job_id="accepted")
    assert len(submitted) == 1

    store.close()

    assert store.closed is True
    assert store.pool._shutdown is True
    assert store.jobs["accepted"]["status"] == "queued"
