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
    assert nt.server_class_for_host("127.0.0.1") is ThreadingHTTPServer
    assert nt.server_class_for_host("::1") is nt.IPv6ThreadingHTTPServer
    assert nt.server_class_for_host("[::1]") is nt.IPv6ThreadingHTTPServer
    assert nt.server_class_for_host("transfer.example", "ipv6") is nt.IPv6ThreadingHTTPServer
    assert nt.server_class_for_host("transfer.example", "ipv4") is ThreadingHTTPServer


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
