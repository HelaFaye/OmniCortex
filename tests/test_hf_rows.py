"""Hub datasets without the `datasets` package: the Hub's parquet export, cached."""
import io
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import hf_rows  # noqa: E402

pd = pytest.importorskip("pandas")
pytest.importorskip("pyarrow")


class Resp(io.BytesIO):
    def __init__(self, b, headers=None):
        super().__init__(b)
        self.headers = headers or {}

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_parquet_fallback(monkeypatch, tmp_path):
    buf = io.BytesIO()
    pd.DataFrame([{"instance_id": "a-1", "text": "x"}, {"instance_id": "b-2", "text": "y"}]).to_parquet(buf)
    calls = []

    def get(url, timeout=60):
        calls.append(url)
        if url.endswith("/parquet"):
            return Resp(json.dumps({"default": {"test": ["https://hub/p/default/test/0.parquet"],
                                                "dev": ["https://hub/p/default/dev/0.parquet"]}}).encode())
        return Resp(buf.getvalue(), {"Content-Length": str(len(buf.getvalue()))})
    monkeypatch.setattr(hf_rows, "_get", get)
    monkeypatch.setattr(hf_rows, "CACHE", tmp_path)
    monkeypatch.setitem(sys.modules, "datasets", None)          # as if not installed
    rows = hf_rows.load("org/set", "test")
    assert [r["instance_id"] for r in rows] == ["a-1", "b-2"]
    n = len(calls)
    assert hf_rows.load("org/set", "test") == rows and len(calls) == n + 1      # parquet now from the cache
    with pytest.raises(ValueError, match="no split 'train'"):
        hf_rows.load("org/set", "train")
    f = tmp_path / "rows.jsonl"
    f.write_text('{"a": 1}\n{"a": 2}\n')
    assert hf_rows.load(str(f)) == [{"a": 1}, {"a": 2}]
