"""
Rows of a Hugging Face dataset split, with or without the `datasets` package.

    rows = hf_rows.load("princeton-nlp/SWE-bench_Lite_bm25_13K", "test")

With `datasets` installed it is used as usual. Without it, the Hub's own
parquet export of the split (/api/datasets/<repo>/parquet) is downloaded
once into ~/.cache/omnicortex/datasets and read with pandas, so a job does
not fail just because one large package is missing. HF_TOKEN is sent for
gated datasets.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import urllib.parse
import urllib.request
from pathlib import Path

CACHE = Path(os.environ.get("OC_DATASET_CACHE") or Path.home() / ".cache" / "omnicortex" / "datasets")
HUB = "https://huggingface.co"


def _get(url: str, timeout: float = 60):
    req = urllib.request.Request(url, headers={"User-Agent": "omnicortex"})
    tok = os.environ.get("HF_TOKEN")
    if tok:
        req.add_header("Authorization", f"Bearer {tok}")
    return urllib.request.urlopen(req, timeout=timeout)


def parquet_urls(repo: str, split: str, config: str | None = None) -> list[str]:
    with _get(f"{HUB}/api/datasets/{urllib.parse.quote(repo, safe='/')}/parquet") as r:
        tree = json.loads(r.read())
    if not tree:
        raise ValueError(f"{repo}: the Hub has no parquet export for this dataset")
    cfg = config or ("default" if "default" in tree else next(iter(tree)))
    if cfg not in tree:
        raise ValueError(f"{repo}: no config {cfg!r}; have {', '.join(tree)}")
    if split not in tree[cfg]:
        raise ValueError(f"{repo}: no split {split!r} in {cfg}; have {', '.join(tree[cfg])}")
    return tree[cfg][split]


def _download(url: str, progress=True) -> Path:
    CACHE.mkdir(parents=True, exist_ok=True)
    dst = CACHE / (hashlib.sha256(url.encode()).hexdigest()[:20] + ".parquet")
    if dst.exists() and dst.stat().st_size:
        return dst
    tmp = dst.with_suffix(".part")
    with _get(url, timeout=300) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        done = 0
        while True:
            b = r.read(1 << 20)
            if not b:
                break
            f.write(b)
            done += len(b)
            if progress and total:
                # the Jobs page reads "PROGRESS done/total label" lines as a progress bar
                print(f"PROGRESS {done}/{total} downloading {url.rsplit('/', 3)[-3]}", flush=True)
    tmp.replace(dst)
    return dst


def load(spec: str, split: str = "test", config: str | None = None) -> list[dict]:
    """A local .jsonl / .json / .parquet file, or a Hub dataset id."""
    p = Path(spec).expanduser()
    if p.exists():
        if p.suffix == ".jsonl":
            return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
        if p.suffix == ".json":
            return json.loads(p.read_text())
        if p.suffix == ".parquet":
            import pandas as pd
            return pd.read_parquet(p).to_dict("records")
        raise ValueError(f"{spec}: expected .jsonl, .json or .parquet")
    try:
        from datasets import load_dataset
    except ImportError:
        load_dataset = None
    if load_dataset is not None:
        return list(load_dataset(spec, config, split=split) if config else load_dataset(spec, split=split))
    try:
        import pandas as pd
    except ImportError:
        raise SystemExit("reading a Hub dataset needs either the 'datasets' or the 'pandas' package; "
                         "install one (Setup -> Requirements, or pip install pandas pyarrow)")
    print(f"'datasets' is not installed; reading {spec} [{split}] from the Hub's parquet export", flush=True)
    rows = []
    for u in parquet_urls(spec, split, config):
        rows += pd.read_parquet(_download(u)).to_dict("records")
    return rows


if __name__ == "__main__":
    r = load(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "test")
    print(len(r), "rows;", ", ".join(list(r[0])[:12]) if r else "")
