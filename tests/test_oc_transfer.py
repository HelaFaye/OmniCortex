import threading
from pathlib import Path
from http.server import ThreadingHTTPServer
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import oc_transfer as nt

def server(tmp, token=''):
    nt.Handler.receiver=nt.Receiver(tmp); nt.Handler.token=token
    s=ThreadingHTTPServer(('127.0.0.1',0),nt.Handler); threading.Thread(target=s.serve_forever,daemon=True).start(); return s

def test_hash_and_safe_name(tmp_path):
    p=tmp_path/'a.bin'; p.write_bytes(b'hello'); assert nt.sha256_file(p)==nt.sha256_bytes(b'hello'); assert nt.safe_name('../../x')=='x'

def test_resumable_transfer(tmp_path):
    s=server(tmp_path/'recv','x'); base=f'http://127.0.0.1:{s.server_port}'; src=tmp_path/'model.gguf'; src.write_bytes(b'0123456789'*10000)
    got=nt.send_one(base,src,'x',chunk=1024)
    out=Path(got['path']); assert out.read_bytes()==src.read_bytes(); assert got['status']=='completed'; s.shutdown()

def test_auth(tmp_path):
    s=server(tmp_path/'recv','secret'); base=f'http://127.0.0.1:{s.server_port}'
    try: nt.request(base,'GET','/health')
    except nt.TransferError as e: assert '401' in str(e)
    else: raise AssertionError('unauthenticated request accepted')
    assert nt.request(base,'GET','/health','secret')['ok']; s.shutdown()

def test_batch(tmp_path):
    s=server(tmp_path/'recv'); base=f'http://127.0.0.1:{s.server_port}'; src=tmp_path/'src'; src.mkdir()
    for i in range(3): (src/f'm{i}.gguf').write_bytes(bytes([i])*5000)
    import subprocess
    r=subprocess.run([sys.executable,str(Path(nt.__file__)),'batch','--url',base,'--dir',str(src),'--workers','2'],capture_output=True,text=True)
    assert r.returncode==0, r.stderr
    assert len(list((tmp_path/'recv').glob('m*.gguf')))==3; s.shutdown()

def test_cancel(tmp_path):
    s=server(tmp_path/'recv'); base=f'http://127.0.0.1:{s.server_port}'; m=nt.request(base,'POST','/v1/transfers/init','x',{'name':'a.gguf','size':3,'sha256':nt.sha256_bytes(b'abc')}); c=nt.request(base,'POST',f"/v1/transfers/{m['id']}/cancel",'x',{}); assert c['status']=='cancelled'; s.shutdown()


def test_resume_existing_partial(tmp_path):
    s=server(tmp_path/'recv','x'); base=f'http://127.0.0.1:{s.server_port}'; src=tmp_path/'resume.gguf'; src.write_bytes(b'abcdef'*2000)
    size=src.stat().st_size; digest=nt.sha256_file(src)
    meta=nt.request(base,'POST','/v1/transfers/init','x',{'name':src.name,'size':size,'sha256':digest})
    first=b'abcdef'*100; nt.request(base,'PUT',f"/v1/transfers/{meta['id']}/chunk",'x',first,headers={'Content-Range':f'bytes 0-{len(first)-1}/{size}','X-Chunk-SHA256':nt.sha256_bytes(first)},raw=True)
    final=nt.send_one(base,src,'x',chunk=1024)
    assert Path(final['path']).read_bytes()==src.read_bytes(); s.shutdown()


def test_server_family_selection():
    assert nt.server_class_for_host("127.0.0.1") is nt.oc_http.BoundedThreadingHTTPServer
    assert nt.server_class_for_host("::1") is nt.IPv6ThreadingHTTPServer
    assert nt.server_class_for_host("[::1]") is nt.IPv6ThreadingHTTPServer
    assert nt.server_class_for_host("transfer.example", "ipv6") is nt.IPv6ThreadingHTTPServer
    assert nt.server_class_for_host("transfer.example", "ipv4") is nt.oc_http.BoundedThreadingHTTPServer


def test_normalize_bind_host():
    assert nt.normalize_bind_host("[2001:db8::1]") == "2001:db8::1"
    assert nt.normalize_bind_host("::") == "::"
    try:
        nt.normalize_bind_host("[::1")
    except ValueError:
        pass
    else:
        raise AssertionError("malformed bracketed bind host accepted")


def test_ipv6_loopback_bind_if_available():
    try:
        s = nt.IPv6ThreadingHTTPServer(("::1", 0), nt.Handler)
    except OSError as exc:
        import pytest
        pytest.skip(f"IPv6 loopback unavailable on this host: {exc}")
    try:
        assert ":" in s.server_address[0]
        assert s.server_port > 0
    finally:
        s.server_close()


def test_abandoned_transfer_expires_and_releases_active_slot(tmp_path):
    import time
    receiver = nt.Receiver(tmp_path / "expiry", max_active=1)
    first = receiver.init({"name": "old.gguf", "size": 3, "sha256": nt.sha256_bytes(b"old")})
    old = receiver.get(first["id"])
    old["updated"] = time.time() - nt.INCOMPLETE_TTL - 1
    receiver.save(old)
    second = receiver.init({"name": "new.gguf", "size": 3, "sha256": nt.sha256_bytes(b"new")})
    assert second["status"] == "receiving"
    assert receiver.get(first["id"])["status"] == "expired"


def test_transfer_space_reservation_includes_existing_active_uploads(tmp_path, monkeypatch):
    from types import SimpleNamespace
    receiver = nt.Receiver(tmp_path / "space", max_active=3)
    monkeypatch.setattr(nt.shutil, "disk_usage", lambda _p: SimpleNamespace(free=100))
    receiver.init({"name": "first.gguf", "size": 70, "sha256": nt.sha256_bytes(b"1")})
    try:
        receiver.init({"name": "second.gguf", "size": 40, "sha256": nt.sha256_bytes(b"2")})
    except nt.TransferError as exc:
        assert "active transfers" in str(exc)
    else:
        raise AssertionError("disk reservation ignored an active transfer")


def test_transfer_metadata_storage_is_private(tmp_path):
    import stat

    receiver = nt.Receiver(tmp_path / "private")
    rec = receiver.init({
        "name": "private.gguf",
        "size": 0,
        "sha256": nt.sha256_bytes(b""),
    })
    mp, pp = receiver.paths(rec["id"])

    assert stat.S_IMODE(receiver.meta_dir.stat().st_mode) == 0o700
    assert stat.S_IMODE(mp.stat().st_mode) == 0o600
    assert stat.S_IMODE(pp.stat().st_mode) == 0o600


def test_transfer_rejects_metadata_directory_destinations(tmp_path):
    receiver = nt.Receiver(tmp_path / "reserved")
    payload = {
        "name": "victim.json",
        "size": 0,
        "sha256": nt.sha256_bytes(b""),
        "subdir": ".omnicortex-transfers",
    }
    try:
        receiver.init(payload)
    except nt.TransferError as exc:
        assert "reserved" in str(exc)
    else:
        raise AssertionError("metadata directory accepted as upload destination")


def test_transfer_rejects_metadata_directory_symlink_alias(tmp_path):
    import pytest

    receiver = nt.Receiver(tmp_path / "alias")
    empty_sha = nt.sha256_bytes(b"")
    victim = receiver.init({
        "name": "victim.gguf",
        "size": 0,
        "sha256": empty_sha,
    })

    alias = receiver.root / "metadata-alias"
    try:
        alias.symlink_to(receiver.meta_dir, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("directory symlinks unavailable")

    with pytest.raises(nt.TransferError, match="unsafe or reserved"):
        receiver.init({
            "name": victim["id"] + ".json",
            "subdir": "metadata-alias",
            "size": 0,
            "sha256": empty_sha,
        })

    assert receiver.get(victim["id"])["status"] == "receiving"

def test_transfer_id_rejects_trailing_newline(tmp_path):
    receiver = nt.Receiver(tmp_path / "bad-id")
    rec = receiver.init({
        "name": "valid.gguf",
        "size": 0,
        "sha256": nt.sha256_bytes(b""),
    })
    try:
        receiver.get(rec["id"] + "\n")
    except nt.TransferError:
        pass
    else:
        raise AssertionError("transfer ID accepted a trailing newline")

def test_resume_reconciles_partial_bytes_after_interrupted_metadata_save(tmp_path):
    receiver = nt.Receiver(tmp_path / "crash-recovery")
    payload = {
        "name": "recovery.gguf",
        "size": 8,
        "sha256": nt.sha256_bytes(b"12345678"),
    }

    first = receiver.init(payload)
    _, partial = receiver.paths(first["id"])

    # Simulate a crash after the data reached disk but before metadata.offset
    # was updated.
    partial.write_bytes(b"1234")

    resumed = receiver.init(payload)
    assert resumed["id"] == first["id"]
    assert resumed["offset"] == 4


def test_resume_rejects_partial_file_larger_than_declared_size(tmp_path):
    receiver = nt.Receiver(tmp_path / "oversized-partial")
    payload = {
        "name": "oversized.gguf",
        "size": 3,
        "sha256": nt.sha256_bytes(b"abc"),
    }

    first = receiver.init(payload)
    _, partial = receiver.paths(first["id"])
    partial.write_bytes(b"four")

    resumed = receiver.init(payload)

    assert resumed["id"] != first["id"]
    assert resumed["offset"] == 0
    assert receiver.get(first["id"])["status"] == "failed"
    assert not partial.exists()


def test_corrupt_transfer_metadata_does_not_break_pruning(tmp_path):
    import json
    import time

    receiver = nt.Receiver(tmp_path / "corrupt-metadata")

    malformed_id = "a" * 32
    malformed = receiver.meta_dir / f"{malformed_id}.json"
    malformed.write_text(json.dumps({
        "id": malformed_id,
        "name": "broken.gguf",
        "size": 10,
        "offset": 0,
        "sha256": nt.sha256_bytes(b"0123456789"),
        "status": "receiving",
        "created": time.time(),
        "updated": "not-a-timestamp",
        "subdir": "",
    }))

    # A valid-looking record whose embedded ID does not match its filename.
    mismatched = receiver.meta_dir / f"{'b' * 32}.json"
    mismatched.write_text(json.dumps({
        "id": "c" * 32,
        "name": "mismatched.gguf",
        "size": 10,
        "offset": 0,
        "sha256": nt.sha256_bytes(b"0123456789"),
        "status": "receiving",
        "created": time.time(),
        "updated": time.time(),
        "subdir": "",
    }))

    assert receiver.list() == []

    valid = receiver.init({
        "name": "valid.gguf",
        "size": 0,
        "sha256": nt.sha256_bytes(b""),
    })
    assert valid["status"] == "receiving"

def test_init_recovers_file_published_before_metadata_update(tmp_path):
    receiver = nt.Receiver(tmp_path / "publish-recovery")
    content = b"published payload"
    payload = {
        "name": "published.gguf",
        "size": len(content),
        "sha256": nt.sha256_bytes(content),
    }

    first = receiver.init(payload)
    _, partial = receiver.paths(first["id"])
    dest = receiver.root / payload["name"]

    # Simulate os.replace() succeeding before the metadata save.
    dest.write_bytes(content)
    partial.unlink()

    recovered = receiver.init(payload)

    assert recovered["id"] == first["id"]
    assert recovered["status"] == "completed"
    assert recovered["offset"] == len(content)
    assert Path(recovered["path"]) == dest
    assert not partial.exists()


def test_complete_recovers_matching_published_file(tmp_path):
    receiver = nt.Receiver(tmp_path / "complete-recovery")
    content = b"already published"
    payload = {
        "name": "already.gguf",
        "size": len(content),
        "sha256": nt.sha256_bytes(content),
    }

    meta = receiver.init(payload)
    _, partial = receiver.paths(meta["id"])
    dest = receiver.root / payload["name"]

    dest.write_bytes(content)
    partial.unlink()

    recovered = receiver.complete(meta["id"])

    assert recovered["status"] == "completed"
    assert recovered["offset"] == len(content)
    assert Path(recovered["path"]) == dest
    assert receiver.get(meta["id"])["status"] == "completed"


def test_complete_does_not_accept_mismatched_published_file(tmp_path):
    receiver = nt.Receiver(tmp_path / "mismatch-recovery")
    content = b"expected content"
    payload = {
        "name": "mismatch.gguf",
        "size": len(content),
        "sha256": nt.sha256_bytes(content),
    }

    meta = receiver.init(payload)
    _, partial = receiver.paths(meta["id"])
    dest = receiver.root / payload["name"]

    dest.write_bytes(b"different content")
    partial.unlink()

    try:
        receiver.complete(meta["id"])
    except nt.TransferError as exc:
        assert "matching published file" in str(exc)
    else:
        raise AssertionError("mismatched published file was accepted")

    assert receiver.get(meta["id"])["status"] == "receiving"



def test_chunk_rejects_symlinked_partial_file(tmp_path):
    import pytest

    receiver = nt.Receiver(tmp_path / "chunk-symlink")
    payload = b"abc"
    meta = receiver.init({
        "name": "safe.gguf",
        "size": len(payload),
        "sha256": nt.sha256_bytes(payload),
    })

    _, partial = receiver.paths(meta["id"])
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"protected")
    partial.unlink()
    partial.symlink_to(outside)

    with pytest.raises(nt.TransferError, match="symlink"):
        receiver.chunk(
            meta["id"], 0, len(payload), payload, nt.sha256_bytes(payload)
        )

    assert outside.read_bytes() == b"protected"
    assert receiver.get(meta["id"])["offset"] == 0


def test_chunk_rejects_partial_size_mismatch(tmp_path):
    import pytest

    receiver = nt.Receiver(tmp_path / "chunk-size")
    meta = receiver.init({
        "name": "size.gguf",
        "size": 8,
        "sha256": nt.sha256_bytes(b"12345678"),
    })
    _, partial = receiver.paths(meta["id"])
    partial.write_bytes(b"unexpected")

    with pytest.raises(nt.TransferError, match="size mismatch"):
        receiver.chunk(
            meta["id"], 0, 8, b"1234", nt.sha256_bytes(b"1234")
        )

    assert partial.read_bytes() == b"unexpected"
    assert receiver.get(meta["id"])["offset"] == 0


def test_active_transfers_cannot_reserve_same_destination(tmp_path):
    import pytest

    receiver = nt.Receiver(tmp_path / "destination-collision")
    first_payload = b"first"
    first = {
        "name": "same.gguf",
        "size": len(first_payload),
        "sha256": nt.sha256_bytes(first_payload),
    }
    original = receiver.init(first)

    assert receiver.init(first)["id"] == original["id"]

    second_payload = b"other"
    second = {
        "name": "same.gguf",
        "size": len(second_payload),
        "sha256": nt.sha256_bytes(second_payload),
    }
    with pytest.raises(nt.TransferError, match="reserved"):
        receiver.init(second)

    assert receiver.get(original["id"])["status"] == "receiving"


def test_cancelled_transfer_cannot_be_recovered_as_completed(tmp_path):
    import pytest

    receiver = nt.Receiver(tmp_path / "cancelled-recovery")
    content = b"cancelled"
    meta = receiver.init({
        "name": "cancelled.gguf",
        "size": len(content),
        "sha256": nt.sha256_bytes(content),
    })

    cancelled = receiver.cancel(meta["id"])
    assert cancelled["status"] == "cancelled"

    dest = receiver.root / "cancelled.gguf"
    dest.write_bytes(content)

    with pytest.raises(nt.TransferError, match="not receiving"):
        receiver.complete(meta["id"])

    assert receiver.get(meta["id"])["status"] == "cancelled"


def test_completed_transfer_download_uses_validated_destination(tmp_path):
    receiver = nt.Receiver(tmp_path / "validated-download")
    content = b"download"
    meta = receiver.init({
        "name": "download.gguf",
        "size": len(content),
        "sha256": nt.sha256_bytes(content),
    })
    _, partial = receiver.paths(meta["id"])
    partial.write_bytes(content)
    meta = receiver.get(meta["id"])
    meta["offset"] = len(content)
    receiver.save(meta)

    completed = receiver.complete(meta["id"])
    assert completed["status"] == "completed"
    assert receiver._destination(completed).read_bytes() == content

    completed["path"] = str(tmp_path / "outside.bin")
    assert receiver._destination(completed) == receiver.root / "download.gguf"



def test_receiver_get_rejects_symlinked_metadata(tmp_path):
    import pytest

    receiver = nt.Receiver(tmp_path / "metadata-symlink")
    meta = receiver.init({
        "name": "safe.gguf",
        "size": 0,
        "sha256": nt.sha256_bytes(b""),
    })
    metadata_path, _ = receiver.paths(meta["id"])

    outside = tmp_path / "outside.json"
    outside.write_text('{"id": "external"}')
    metadata_path.unlink()
    metadata_path.symlink_to(outside)

    with pytest.raises(nt.TransferError, match="metadata path is a symlink"):
        receiver.get(meta["id"])

    assert outside.read_text() == '{"id": "external"}'


def test_pull_uses_unique_private_temporary_file(tmp_path, monkeypatch):
    import io
    import stat

    content = b"downloaded model contents"
    digest = nt.sha256_bytes(content)
    out_dir = tmp_path / "downloads"
    out_dir.mkdir()

    # The former predictable temporary path is a planted symlink.
    victim = tmp_path / "protected.bin"
    victim.write_bytes(b"do not overwrite")
    predictable = out_dir / "model.gguf.ns-incoming"
    predictable.symlink_to(victim)

    monkeypatch.setattr(
        nt,
        "request",
        lambda *args, **kwargs: {
            "name": "model.gguf",
            "size": len(content),
            "sha256": digest,
        },
    )

    class Response(io.BytesIO):
        pass

    monkeypatch.setattr(
        nt.urllib.request,
        "urlopen",
        lambda *args, **kwargs: Response(content),
    )

    result = nt.pull(
        "http://receiver.test",
        "a" * 32,
        "test-token",
        out_dir,
    )

    final = out_dir / "model.gguf"
    assert result["sha256"] == digest
    assert final.read_bytes() == content
    assert stat.S_IMODE(final.stat().st_mode) == 0o600
    assert predictable.is_symlink()
    assert victim.read_bytes() == b"do not overwrite"
    assert sorted(p.name for p in out_dir.iterdir()) == [
        "model.gguf",
        "model.gguf.ns-incoming",
    ]


def test_pull_removes_temporary_file_on_checksum_failure(tmp_path, monkeypatch):
    import io

    out_dir = tmp_path / "failed-download"
    out_dir.mkdir()
    content = b"wrong content"

    monkeypatch.setattr(
        nt,
        "request",
        lambda *args, **kwargs: {
            "name": "model.gguf",
            "size": len(content),
            "sha256": "0" * 64,
        },
    )
    monkeypatch.setattr(
        nt.urllib.request,
        "urlopen",
        lambda *args, **kwargs: io.BytesIO(content),
    )

    import pytest
    with pytest.raises(nt.TransferError, match="sha256 mismatch"):
        nt.pull("http://receiver.test", "a" * 32, "token", out_dir)

    assert not (out_dir / "model.gguf").exists()
    assert list(out_dir.iterdir()) == []
