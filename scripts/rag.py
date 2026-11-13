#!/usr/bin/env python3
"""
Local document collections for retrieval-augmented chat.

A collection is a directory of chunks (chunks.jsonl) with a BM25 index built
on load and, optionally, dense vectors (dense.npy) from any OpenAI-compatible
/v1/embeddings endpoint: llama-server with an embedding GGUF (`--embedding`),
LM Studio, or Studio's own `--rag-embed-gguf`. Search fuses the two rankings
with reciprocal-rank fusion; with no embedder it is BM25 alone, which needs no
model and is often enough for names, identifiers and error messages.

    python scripts/rag.py add  --collection notes docs/*.md
    python scripts/rag.py search --collection notes "how does auto routing pick a model"
    python scripts/rag.py add  --collection papers paper.pdf --embed http://127.0.0.1:8081/v1@nomic-embed

Text, Markdown, code, HTML, JSON/CSV are read directly; PDF needs `pypdf`,
DOCX needs `python-docx`. Chunks are ~1000 characters on paragraph
boundaries with overlap, and remember their source file and page.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import io
import json
import math
import re
import shutil
import threading
import time
import urllib.request
from collections import Counter
from pathlib import Path

DEFAULT_ROOT = Path.home() / ".omnicortex" / "rag"
NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_. -]{0,63}$")
TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")
TEXT_EXT = {".txt", ".md", ".markdown", ".rst", ".py", ".js", ".ts", ".tsx", ".jsx", ".c", ".h", ".cpp", ".hpp",
            ".cc", ".rs", ".go", ".java", ".kt", ".swift", ".rb", ".php", ".sh", ".toml", ".yaml", ".yml", ".ini",
            ".cfg", ".json", ".jsonl", ".csv", ".tsv", ".sql", ".tex", ".log", ".css", ".xml"}
STOP = set("a an and are as at be by for from has have in is it its of on or that the this to was were will with "
           "what which who how why when where do does did can i you we they".split())


def tokens(text: str) -> list[str]:
    out = []
    for t in TOKEN_RE.findall(text.lower()):
        if t in STOP:
            continue
        out.append(t)
        if "_" in t:                       # snake_case identifiers also match their parts
            out += [p for p in t.split("_") if p and p not in STOP]
    return out


# ---------------------------------------------------------------- reading

def extract(name: str, data: bytes) -> list[tuple[int | None, str]]:
    """[(page or None, text)] from a file's bytes."""
    ext = Path(name).suffix.lower()
    if ext == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError:
            raise ValueError("PDF needs pypdf: pip install pypdf") from None
        r = PdfReader(io.BytesIO(data))
        return [(i + 1, p.extract_text() or "") for i, p in enumerate(r.pages)]
    if ext == ".docx":
        try:
            import docx
        except ImportError:
            raise ValueError("DOCX needs python-docx: pip install python-docx") from None
        d = docx.Document(io.BytesIO(data))
        return [(None, "\n\n".join(p.text for p in d.paragraphs))]
    text = data.decode("utf-8", errors="replace")
    if ext in (".html", ".htm"):
        text = re.sub(r"(?is)<(script|style).*?</\1>", " ", text)
        text = re.sub(r"(?i)<(br|/p|/div|/h\d|/li)>", "\n", text)
        text = html.unescape(re.sub(r"<[^>]+>", " ", text))
    elif ext not in TEXT_EXT and "\x00" in text[:4096]:
        raise ValueError(f"{name}: binary file type {ext or '(none)'} is not supported")
    return [(None, text)]


def chunk(text: str, size: int = 1000, overlap: int = 150) -> list[str]:
    """Paragraph-packed chunks of about `size` characters, overlapping by `overlap`."""
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    pieces = []
    for p in paras:                        # split paragraphs longer than a chunk
        while len(p) > size:
            cut = p.rfind(" ", 0, size)
            cut = cut if cut > size // 2 else size
            pieces.append(p[:cut])
            p = p[cut:].strip()
        pieces.append(p)
    out, cur = [], ""
    for p in pieces:
        if cur and len(cur) + len(p) + 2 > size:
            out.append(cur)
            cur = cur[-overlap:].split(" ", 1)[-1] if overlap else ""
        cur = (cur + "\n\n" + p).strip() if cur else p
    if cur:
        out.append(cur)
    return out


# ---------------------------------------------------------------- embeddings

class Embedder:
    """OpenAI-compatible /v1/embeddings client: spec is URL[@model]."""

    def __init__(self, spec: str, api_key: str = "", batch: int = 32):
        url, _, model = spec.partition("@")
        self.url = url.rstrip("/")
        if not self.url.endswith("/v1"):
            self.url += "/v1"
        self.model = model or "embedding"
        self.api_key = api_key
        self.batch = batch

    def __call__(self, texts: list[str]):
        import numpy as np
        out = []
        for i in range(0, len(texts), self.batch):
            body = json.dumps({"model": self.model, "input": texts[i:i + self.batch]}).encode()
            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            req = urllib.request.Request(self.url + "/embeddings", body, headers)
            with urllib.request.urlopen(req, timeout=600) as r:
                data = json.loads(r.read())["data"]
            out += [d["embedding"] for d in sorted(data, key=lambda d: d.get("index", 0))]
        v = np.asarray(out, dtype=np.float32)
        return v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-12)


# ---------------------------------------------------------------- collection

class Collection:
    def __init__(self, path: Path):
        self.path = path
        self.lock = threading.Lock()
        self.meta = json.loads((path / "meta.json").read_text()) if (path / "meta.json").exists() else {}
        self.chunks: list[dict] = []
        if (path / "chunks.jsonl").exists():
            self.chunks = [json.loads(l) for l in (path / "chunks.jsonl").read_text(encoding="utf-8").splitlines() if l]
        self.dense = None
        if (path / "dense.npy").exists():
            import numpy as np
            self.dense = np.load(path / "dense.npy")
        self._index()

    @property
    def name(self) -> str:
        return self.path.name

    def _index(self) -> None:
        self.tf = [Counter(tokens(c["text"])) for c in self.chunks]
        self.len = [sum(t.values()) for t in self.tf]
        self.avg = (sum(self.len) / len(self.len)) if self.len else 1.0
        df = Counter()
        for t in self.tf:
            df.update(t.keys())
        n = len(self.tf)
        self.idf = {w: math.log(1 + (n - f + 0.5) / (f + 0.5)) for w, f in df.items()}

    def _save(self) -> None:
        tmp = self.path / "chunks.jsonl.tmp"
        tmp.write_text("".join(json.dumps(c, ensure_ascii=False) + "\n" for c in self.chunks), encoding="utf-8")
        tmp.replace(self.path / "chunks.jsonl")
        if self.dense is not None:
            import numpy as np
            np.save(self.path / "dense.tmp.npy", self.dense)
            (self.path / "dense.tmp.npy").replace(self.path / "dense.npy")
        elif (self.path / "dense.npy").exists():
            (self.path / "dense.npy").unlink()
        (self.path / "meta.json").write_text(json.dumps(self.meta, indent=1))

    def docs(self) -> list[dict]:
        seen: dict[str, dict] = {}
        for c in self.chunks:
            d = seen.setdefault(c["doc"], {"doc": c["doc"], "source": c["source"], "chunks": 0})
            d["chunks"] += 1
        return list(seen.values())

    def info(self) -> dict:
        return {"name": self.name, "docs": len(self.docs()), "chunks": len(self.chunks),
                "dense": self.dense is not None, "embed_model": self.meta.get("embed_model")}

    def add(self, name: str, data: bytes, embedder: Embedder | None = None, size: int = 1000) -> dict:
        doc = hashlib.sha256(data).hexdigest()[:16]
        pages = extract(name, data)
        new = []
        for page, text in pages:
            for i, piece in enumerate(chunk(text, size)):
                new.append({"doc": doc, "source": Path(name).name, "page": page, "i": i, "text": piece})
        if not new:
            raise ValueError(f"{name}: no text found")
        vecs = None
        if embedder is not None:
            if self.chunks and self.dense is None:
                raise ValueError("this collection has no vectors; re-create it to add an embedder")
            if self.meta.get("embed_model") not in (None, embedder.model):
                raise ValueError(f"collection was embedded with {self.meta['embed_model']}, not {embedder.model}")
            vecs = embedder([c["text"] for c in new])
        elif self.dense is not None:
            raise ValueError(f"collection uses dense vectors ({self.meta.get('embed_model')}); configure the embedder")
        with self.lock:
            keep = [i for i, c in enumerate(self.chunks) if c["doc"] != doc]      # re-adding replaces
            if self.dense is not None and len(keep) != len(self.chunks):
                self.dense = self.dense[keep]
            self.chunks = [self.chunks[i] for i in keep] + new
            if vecs is not None:
                import numpy as np
                self.dense = vecs if self.dense is None else np.concatenate([self.dense, vecs])
                self.meta["embed_model"] = embedder.model
            self.meta["updated"] = time.time()
            self._save()
            self._index()
        return {"doc": doc, "source": Path(name).name, "chunks": len(new), "pages": len(pages)}

    def remove(self, doc: str) -> int:
        with self.lock:
            keep = [i for i, c in enumerate(self.chunks) if c["doc"] != doc]
            n = len(self.chunks) - len(keep)
            if self.dense is not None:
                self.dense = self.dense[keep]
            self.chunks = [self.chunks[i] for i in keep]
            self._save()
            self._index()
        return n

    def bm25(self, query: str, k1: float = 1.5, b: float = 0.75) -> list[tuple[float, int]]:
        q = tokens(query)
        out = []
        for i, tf in enumerate(self.tf):
            s = 0.0
            for w in q:
                f = tf.get(w)
                if f:
                    s += self.idf.get(w, 0.0) * f * (k1 + 1) / (f + k1 * (1 - b + b * self.len[i] / self.avg))
            if s > 0:
                out.append((s, i))
        out.sort(reverse=True)
        return out

    def search(self, query: str, k: int = 4, embedder: Embedder | None = None) -> list[dict]:
        if not self.chunks:
            return []
        lex = self.bm25(query)[:50]
        ranks: dict[int, float] = {}
        for r, (_, i) in enumerate(lex):
            ranks[i] = ranks.get(i, 0.0) + 1.0 / (60 + r)
        if embedder is not None and self.dense is not None:
            qv = embedder([query])[0]
            sims = self.dense @ qv
            for r, i in enumerate(sims.argsort()[::-1][:50]):
                ranks[int(i)] = ranks.get(int(i), 0.0) + 1.0 / (60 + r)
        best = sorted(ranks.items(), key=lambda x: -x[1])[:k]
        return [{**{x: self.chunks[i][x] for x in ("source", "page", "text", "doc")}, "score": round(s, 5)}
                for i, s in best]


class RagStore:
    def __init__(self, root: str | Path | None = None, embedder: Embedder | None = None):
        self.root = Path(root or DEFAULT_ROOT).expanduser()
        self.root.mkdir(parents=True, exist_ok=True)
        self.embedder = embedder
        self._cache: dict[str, Collection] = {}
        self.lock = threading.Lock()

    def _dir(self, name: str) -> Path:
        if not NAME_RE.match(name or "") or ".." in name:
            raise ValueError("collection names: letters, digits, space, _ . - (max 64)")
        return self.root / name

    def get(self, name: str, create: bool = False) -> Collection:
        d = self._dir(name)
        with self.lock:
            if name not in self._cache:
                if not d.exists():
                    if not create:
                        raise FileNotFoundError(f"no collection {name!r}")
                    d.mkdir(parents=True)
                self._cache[name] = Collection(d)
            return self._cache[name]

    def list(self) -> list[dict]:
        return [self.get(p.name).info() for p in sorted(self.root.iterdir()) if p.is_dir() and NAME_RE.match(p.name)]

    def drop(self, name: str) -> None:
        d = self._dir(name)
        with self.lock:
            self._cache.pop(name, None)
            if d.exists():
                shutil.rmtree(d)

    def search(self, name: str, query: str, k: int = 4) -> list[dict]:
        c = self.get(name)
        return c.search(query, k, self.embedder if c.dense is not None else None)


def context_message(hits: list[dict]) -> str:
    """System message carrying numbered sources; the model is asked to cite them."""
    parts = ["Use the numbered sources below when they are relevant, and cite them like [1]. If they do not "
             "contain the answer, say so instead of guessing.\n"]
    for n, h in enumerate(hits, 1):
        where = h["source"] + (f", page {h['page']}" if h.get("page") else "")
        parts.append(f"[{n}] ({where})\n{h['text']}\n")
    return "\n".join(parts)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--root", default=str(DEFAULT_ROOT))
    p.add_argument("--embed", help="URL[@model] of an OpenAI-compatible embeddings endpoint")
    sub = p.add_subparsers(dest="cmd", required=True)
    a_ = sub.add_parser("add")
    a_.add_argument("--collection", required=True)
    a_.add_argument("files", nargs="+")
    s_ = sub.add_parser("search")
    s_.add_argument("--collection", required=True)
    s_.add_argument("query")
    s_.add_argument("-k", type=int, default=4)
    sub.add_parser("list")
    d_ = sub.add_parser("drop")
    d_.add_argument("--collection", required=True)
    a = p.parse_args(argv)
    store = RagStore(a.root, Embedder(a.embed) if a.embed else None)
    if a.cmd == "add":
        c = store.get(a.collection, create=True)
        for f in a.files:
            print(json.dumps(c.add(f, Path(f).read_bytes(), store.embedder)))
    elif a.cmd == "search":
        for h in store.search(a.collection, a.query, a.k):
            print(f"{h['score']:.4f}  {h['source']}{'' if not h.get('page') else ' p' + str(h['page'])}: "
                  f"{h['text'][:160]!r}")
    elif a.cmd == "list":
        print(json.dumps(store.list(), indent=1))
    else:
        store.drop(a.collection)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
