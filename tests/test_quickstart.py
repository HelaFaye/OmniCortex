"""The quick-start guides (docs/QUICKSTART.md): every job link opens a real job with real fields and values,
every doc link and anchor exists, and the Jobs page links to each guide."""
import re
import sys
import urllib.parse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "viz"))
sys.path.insert(0, str(ROOT / "scripts"))
import help_page  # noqa: E402
import jobs  # noqa: E402

DOC = (ROOT / "docs" / "QUICKSTART.md").read_text()
LINKS = re.findall(r"\]\(([^)\s]+)\)", DOC)


def test_job_links_prefill_real_fields():
    specs = jobs.public_specs()
    job_links = [h for h in LINKS if h.startswith("/jobs#")]
    assert len(job_links) >= 15
    for href in job_links:
        q = dict(urllib.parse.parse_qsl(href.split("#", 1)[1]))
        kind = q.pop("kind")
        assert kind in specs, href
        fields = {f["name"]: f for f in specs[kind]["fields"]}
        for k, v in q.items():
            assert k in fields, f"{href}: {kind} has no field {k}"
            f = fields[k]
            if f.get("choices"):
                assert v in f["choices"], f"{href}: {v} is not one of {f['choices']}"
            if f.get("kind") == "int":
                int(v)
            if f.get("kind") == "bool":
                assert v in ("1", "true", "on"), href


def test_doc_links_and_anchors_exist():
    html = help_page.page("QUICKSTART")
    assert html and "/jobs#kind=testqa&amp;subject=code" in html
    for href in LINKS:
        if href.endswith(".md") or ".md#" in href:
            assert (ROOT / "docs" / href.split("#")[0]).exists(), href
    jobs_page = (ROOT / "viz" / "jobs.py").read_text()
    anchors = re.findall(r'/help/QUICKSTART#([\w-]+)', jobs_page)
    assert len(anchors) == 6
    for a in anchors:
        assert f'id="{a}"' in html, a


def test_rag_folder_keeps_paths_and_skips_noise(tmp_path):
    import rag
    repo = tmp_path / "repo"
    (repo / "src" / "app").mkdir(parents=True)
    (repo / "src" / "app" / "main.py").write_text("def launch_rocket():\n    return 'liftoff'\n")
    (repo / "README.md").write_text("# Rocket\n\nThe rocket launches from pad 39A.\n")
    (repo / "Makefile").write_text("all:\n\tpython src/app/main.py\n")
    for junk in (".git/config", "node_modules/x/index.js", "build/out.py", ".venv/lib.py"):
        (repo / junk).parent.mkdir(parents=True, exist_ok=True)
        (repo / junk).write_text("launch_rocket noise")
    (repo / "logo.png").write_bytes(b"\x89PNG\x00\x00")
    (repo / "big.txt").write_text("x" * (rag.FOLDER_MAX + 1))
    files = [p.relative_to(repo).as_posix() for p in rag.folder_files(repo)]
    assert files == ["Makefile", "README.md", "src/app/main.py"]

    assert rag.main(["--root", str(tmp_path / "store"), "add", "--collection", "repo", "--folder", str(repo)]) in (0, None)
    c = rag.RagStore(tmp_path / "store").get("repo")
    hits = c.search("launch_rocket", 2)
    assert hits[0]["source"] == "src/app/main.py"
    n = len(c.chunks)
    (repo / "src" / "app" / "main.py").write_text("def launch_rocket():\n    return 'scrubbed'\n")
    c.add("main.py", (repo / "src/app/main.py").read_bytes(), source="src/app/main.py")
    assert len(c.chunks) == n and "scrubbed" in c.search("launch_rocket", 1)[0]["text"]    # same path: replaced
    assert rag.source_name("/abs/../../etc/passwd") == "abs/etc/passwd"
    assert rag.source_name("C:\\\\src\\\\a.py") == "src/a.py" and rag.source_name("..") == "upload.txt"
