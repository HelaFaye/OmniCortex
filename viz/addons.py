"""
Add-ons: Studio's optional parts, and plugins installed from a package file or a package repository
(docs/ADDONS.md).

Two kinds:

  built-in  parts of Studio that not everyone needs (the decoding lab, features and emotions, graphs,
            neuron review, projects, labs, retraining, …). Removing one takes away its pages, API routes,
            jobs, lab services, graph nodes and sidebar entries; reinstalling brings them back. Their code
            stays in the checkout, so a reinstall needs no download. Vision, remote hosts, model transfer
            and device pairing are part of the core and stay.
  packaged  plugins (viz/plugins.py): a folder with plugin.json, installed from a .zip (a file you
            have, or one listed in a repository index), updated, and uninstalled. A newly installed or
            updated add-on is off until the owner reviews and turns it on, as with any plugin.

A repository is an index.json at an https URL (or a local path):

    {"name": "OmniCortex add-ons", "addons": [
      {"name": "comfyui", "version": "1.0", "description": "...", "url": "comfyui-1.0.zip",
       "sha256": "…", "size": 12345, "homepage": "https://…"}]}

`url` may be relative to the index. The download must match `sha256`.

    python viz/addons.py pack plugins/comfyui -o dist/            # -> dist/comfyui-1.0.zip
    python viz/addons.py index dist/                              # -> dist/index.json
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import shutil
import tempfile
import time
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

import plugins as ns_plugins

MAX_PACKAGE = 200 * 1024 * 1024
MAX_FILES = 5000

# Each built-in add-on: what it brings, so removing it can take exactly that away.
#   paths: URL path prefixes (pages and API); nav: sidebar hrefs; jobs: job kinds or groups ("group:Retrain");
#   services: lab hub services (viz/hub.py); nodes: graph node types; requires: add-ons it needs;
#   needs: Python packages it uses (shown, so you know what removing it saves)
# Not here, so always present: chat, models, jobs, settings, setup, permissions, help, vision, remote hosts,
# model transfer and device pairing.
BUILTIN = {
    "decoding_lab": {"title": "Decoding lab", "desc": "A Hugging Face model behind an OpenAI-compatible API with "
                     "decoding variants (SLED, steering, random projections), the logit lens, and live watching of "
                     "neurons, emotions and features.", "jobs": ["decode_lab", "rp_fit", "rp_export", "logit_lens"],
                     "needs": ["torch", "transformers"], "docs": "DECODING.md"},
    "features": {"title": "Features and emotions", "desc": "Feature dictionaries (SAEs and transcoders), reading "
                 "replies, attribution graphs, emotion vectors and steering.",
                 "paths": ["/features", "/api/features/"], "nav": ["/features"], "jobs": ["group:Interpret"],
                 "nodes": ["emotion_screen", "feature_screen"], "requires": ["decoding_lab"],
                 "needs": ["torch", "transformers"], "docs": "FEATURES.md"},
    "graphs": {"title": "Graphs", "desc": "Node graphs that screen, verify and route model inputs and outputs.",
               "paths": ["/graphs", "/api/graphs"], "nav": ["/graphs"], "docs": "GRAPHS.md"},
    "review": {"title": "Neuron review", "desc": "How neurons behave across tests and checked replies, by subject and "
               "over time.", "paths": ["/review", "/api/review"], "nav": ["/review"], "jobs": ["group:Review"],
               "requires": ["hneurons"], "docs": "REVIEW.md"},
    "projects": {"title": "Projects and workers", "desc": "The director: plans from a description, worker models, "
                 "reviews and verification graphs.", "paths": ["/projects", "/api/projects", "/workers", "/api/workers"],
                 "nav": ["/projects", "/workers"], "docs": "DIRECTOR.md"},
    "labs": {"title": "Labs and 3D views", "desc": "The live activation viewer, the 3D replay, the pipeline dashboard, "
             "and the quantization and sweep labs (the transfer lab stays).",
             "services": ["live", "replay", "pipeline", "quant_lab", "scale_sweep", "adaptive_tuning"],
             "jobs": ["group:Views"], "docs": "LABS.md"},
    "documents": {"title": "Documents (RAG)", "desc": "Document collections for chat and graphs.",
                  "paths": ["/api/rag"], "nav": ["/#docs"], "nodes": ["rag_search", "grounding"], "docs": "STUDIO.md"},
    "system": {"title": "System monitor", "desc": "What is running, restarting any part, and the hardware monitor.",
               "paths": ["/system", "/api/system"], "nav": ["/system"], "docs": "SYSTEM.md"},
    "automation": {"title": "Automation and Home Assistant", "desc": "Power and activity rules, and the Home "
                   "Assistant connection.", "paths": ["/api/automation"], "docs": "SYSTEM.md"},
    "retraining": {"title": "Retraining", "desc": "Deficit datasets, fine-tuning (SFT, DPO, QLoRA), retrain cycles, "
                   "concept edits and merges.", "jobs": ["group:Retrain", "group:Merge"],
                   "needs": ["torch", "transformers", "peft", "trl", "bitsandbytes"], "docs": "RETRAINING.md"},
    "benchmarks": {"title": "SWE-bench and LiveBench", "desc": "SWE-bench predictions, the Docker evaluation harness, "
                   "training data from it, and LiveBench.",
                   "jobs": ["swe_predict", "swe_evaluate", "swe_import", "swe_train", "livebench"],
                   "needs": ["datasets", "swebench", "docker"], "docs": "BENCHMARKS.md"},
    "hneurons": {"title": "H-Neuron pipeline", "desc": "Collect, label, extract, classify, tune and export "
                 "hallucination neurons; analysis and tuning jobs; the hallucination check node.",
                 "jobs": ["group:H-Neurons", "group:Analysis", "group:Tuning"], "nodes": ["hallucination_check"],
                 "docs": "PIPELINE.md"},
    "sharing": {"title": "Sharing results", "desc": "Publish model results to a scoreboard.",
                "paths": ["/api/settings/sharing"], "docs": "SHARING.md"},
}


class AddonError(ValueError):
    pass


# ------------------------------------------------------------------ built-ins

class Builtins:
    """Which built-in add-ons are removed (kept in Studio's config as addons_off)."""

    def __init__(self, load, save):
        self._load, self._save = load, save                      # () -> list, list -> None

    def off(self) -> set:
        return {x for x in self._load() or [] if x in BUILTIN}

    def set(self, name: str, installed: bool, with_dependents: bool = False) -> list:
        """Remove or reinstall one. Removing an add-on others need is refused unless with_dependents, which
        removes them too; reinstalling brings back what it needs. -> every add-on changed."""
        if name not in BUILTIN:
            raise AddonError(f"no built-in add-on {name!r}")
        off = self.off()
        if installed:
            todo, changed = [name], []
            while todo:                                    # it and everything it needs
                n = todo.pop()
                if n in off:
                    off.discard(n)
                    changed.append(n)
                todo += BUILTIN[n].get("requires", [])
        else:
            deps = self.dependents(name) - off
            if deps and not with_dependents:
                raise AddonError(f"{', '.join(BUILTIN[d]['title'] for d in sorted(deps))} need"
                                 f"{'s' if len(deps) == 1 else ''} {BUILTIN[name]['title']}: remove "
                                 f"{'it' if len(deps) == 1 else 'them'} too, or keep it")
            changed = [n for n in [name, *sorted(deps)] if n not in off]
            off.update(changed)
        self._save(sorted(off))
        return changed

    def dependents(self, name: str) -> set:
        """Every add-on that needs this one, directly or not."""
        out, todo = set(), [name]
        while todo:
            n = todo.pop()
            for k, v in BUILTIN.items():
                if n in v.get("requires", []) and k not in out:
                    out.add(k)
                    todo.append(k)
        return out

    def service_hidden(self, service: str) -> bool:
        return any(service in BUILTIN[n].get("services", []) for n in self.off())

    def node_hidden(self, node_type: str) -> bool:
        return any(node_type in BUILTIN[n].get("nodes", []) for n in self.off())

    def blocks(self, path: str) -> str | None:
        """The removed add-on that owns this URL path, if any."""
        path = path.split("?", 1)[0]
        for name in self.off():
            for p in BUILTIN[name].get("paths", []):
                p = p.rstrip("/")
                if path == p or path.startswith(p + "/"):
                    return name
        return None

    def job_hidden(self, kind: str, spec: dict) -> bool:
        for name in self.off():
            for j in BUILTIN[name].get("jobs", []):
                if j == kind or (j.startswith("group:") and spec.get("group") == j[6:]):
                    return True
        return False

    def nav_off(self) -> list:
        return [h for name in self.off() for h in BUILTIN[name].get("nav", [])]

    def list(self) -> list:
        off = self.off()
        return [{"name": k, "kind": "built-in", "installed": k not in off, **{x: v.get(x) for x in ("title", "desc", "docs")},
                 "requires": v.get("requires", []), "needed_by": sorted(self.dependents(k)), "needs": v.get("needs", []),
                 "adds": {"pages": v.get("nav", []), "jobs": v.get("jobs", []), "services": v.get("services", []),
                          "nodes": v.get("nodes", [])}} for k, v in BUILTIN.items()]


# ------------------------------------------------------------------ packages

def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _fetch(url: str, limit: int = MAX_PACKAGE) -> bytes:
    if re.match(r"^https?://", url):
        if url.startswith("http://") and not re.match(r"^http://(127\.0\.0\.1|localhost)[:/]", url):
            raise AddonError("add-on repositories must use https (or be on this computer)")
        req = urllib.request.Request(url, headers={"User-Agent": "OmniCortex-addons"})
        with urllib.request.urlopen(req, timeout=60) as r:
            data = r.read(limit + 1)
    else:
        p = Path(url.removeprefix("file://")).expanduser()
        if p.stat().st_size > limit:
            raise AddonError("package larger than 200 MB")
        data = p.read_bytes()
    if len(data) > limit:
        raise AddonError("package larger than 200 MB")
    return data


def unpack(data: bytes, dest_root: Path) -> tuple[str, Path]:
    """Check a package (.zip) and unpack it into dest_root/<name>. -> (name, folder).
    plugin.json may sit at the top or in one top folder. No absolute paths, .., links or devices."""
    try:
        z = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile:
        raise AddonError("not a zip file") from None
    infos = z.infolist()
    if len(infos) > MAX_FILES:
        raise AddonError(f"more than {MAX_FILES} files")
    if sum(i.file_size for i in infos) > 4 * MAX_PACKAGE:
        raise AddonError("unpacks to more than 800 MB")
    names = [i.filename for i in infos]
    for i in infos:
        n = i.filename
        if n.startswith(("/", "\\")) or ".." in Path(n).parts or re.match(r"^[A-Za-z]:", n):
            raise AddonError(f"unsafe path in package: {n}")
        if (i.external_attr >> 16) & 0o170000 in (0o120000, 0o020000, 0o060000, 0o010000):
            raise AddonError(f"links and device files are not allowed: {n}")
    if "plugin.json" in names:
        prefix = ""
    else:
        tops = {n.split("/", 1)[0] for n in names if "/" in n}
        cand = [t for t in tops if f"{t}/plugin.json" in names]
        if len(cand) != 1:
            raise AddonError("no plugin.json at the top of the package (or in one top folder)")
        prefix = cand[0] + "/"
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp) / "pkg"
        for i in infos:
            if not i.filename.startswith(prefix) or i.is_dir():
                continue
            rel = i.filename[len(prefix):]
            out = (tmp / rel).resolve()
            if tmp.resolve() not in out.parents:
                raise AddonError(f"unsafe path in package: {i.filename}")
            out.parent.mkdir(parents=True, exist_ok=True)
            with z.open(i) as src, open(out, "wb") as dst:
                shutil.copyfileobj(src, dst)
        m = ns_plugins.read_manifest(tmp)
        name = m["name"]
        dest = dest_root / name
        old = dest.with_name(f".{name}.old")
        dest_root.mkdir(parents=True, exist_ok=True)
        if old.exists():
            shutil.rmtree(old)
        if dest.exists():
            dest.rename(old)
        shutil.copytree(tmp, dest)
        if old.exists():
            shutil.rmtree(old, ignore_errors=True)
    return name, dest


class Packages:
    """Installing, updating and removing packaged add-ons in the user's plugins folder."""

    def __init__(self, pm: ns_plugins.Plugins, install_dir: str | Path, repos_load, repos_save):
        self.pm, self.dir = pm, Path(install_dir).expanduser()
        self._repos_load, self._repos_save = repos_load, repos_save

    def repos(self) -> list:
        return list(self._repos_load() or [])

    def set_repos(self, urls: list) -> list:
        clean = []
        for u in urls:
            u = str(u).strip()
            if u and (re.match(r"^https://\S+$", u) or re.match(r"^http://(127\.0\.0\.1|localhost)[:/]\S*$", u)
                      or os.path.isabs(os.path.expanduser(u))):
                clean.append(u)
            elif u:
                raise AddonError(f"{u}: an https URL or an absolute path to an index.json")
        self._repos_save(clean)
        return clean

    def catalog(self) -> list:
        """Every repository's add-ons, with whether each is installed and at which version."""
        installed = {p["name"]: p for p in self.pm.scan()}
        out = []
        for repo in self.repos():
            try:
                idx = json.loads(_fetch(repo, 5 * 1024 * 1024))
            except Exception as e:                                         # noqa: BLE001
                out.append({"repo": repo, "error": f"{type(e).__name__}: {e}"[:300]})
                continue
            for a in idx.get("addons") or []:
                if not ns_plugins.NAME.match(str(a.get("name", ""))):
                    continue
                have = installed.get(a["name"])
                ver = ((have or {}).get("manifest") or {}).get("version")
                out.append({"repo": repo, "repo_name": idx.get("name", ""), **{k: a.get(k) for k in
                            ("name", "version", "description", "url", "sha256", "size", "homepage", "author")},
                            "installed": ver if have else None, "update": bool(have and ver != a.get("version"))})
        return out

    def install_bytes(self, data: bytes, sha256: str | None = None) -> dict:
        if sha256 and _sha256(data) != sha256.lower():
            raise AddonError("the download does not match its sha256: not installed")
        name, folder = unpack(data, self.dir)
        # whatever was turned on before, an installed or updated add-on waits for review
        st = self.pm._state()
        if name in st:
            st.pop(name)
            self.pm._save(st)
        return {"name": name, "folder": str(folder), "status": "off", "sha256": _sha256(data)}

    def install_from_repo(self, name: str, repo: str | None = None) -> dict:
        for a in self.catalog():
            if a.get("name") == name and (repo is None or a["repo"] == repo):
                if not a.get("sha256"):
                    raise AddonError(f"{name}: the index gives no sha256, so it cannot be checked")
                url = a["url"] if re.match(r"^https?://|^/", a["url"] or "") else urllib.parse.urljoin(a["repo"], a["url"]) \
                    if re.match(r"^https?://", a["repo"]) else str(Path(a["repo"]).expanduser().parent / a["url"])
                return self.install_bytes(_fetch(url), a["sha256"])
        raise AddonError(f"no add-on {name!r} in the repositories")

    def uninstall(self, name: str) -> None:
        p = next((x for x in self.pm.scan() if x["name"] == name), None)
        if p is None:
            raise AddonError(f"no add-on {name!r}")
        folder = Path(p["folder"]).resolve()
        if self.dir.resolve() not in folder.parents:
            raise AddonError(f"{name} came with OmniCortex ({folder.parent}); turn it off instead, or remove that folder")
        self.pm.set_enabled(name, False)
        shutil.rmtree(folder)


# ------------------------------------------------------------------ packing (for repositories)

def pack(folder: str | Path, out_dir: str | Path) -> Path:
    folder = Path(folder)
    m = ns_plugins.read_manifest(folder)
    out = Path(out_dir) / f"{m['name']}-{m.get('version', '0')}.zip"
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(folder.rglob("*")):
            rel = p.relative_to(folder)
            if p.is_file() and not any(part in ("__pycache__", ".git", ".venv", "venv", ".godot") for part in rel.parts):
                z.write(p, f"{m['name']}/{rel.as_posix()}")
    return out


def index(dist: str | Path, name: str = "OmniCortex add-ons") -> Path:
    dist = Path(dist)
    rows = []
    for z in sorted(dist.glob("*.zip")):
        data = z.read_bytes()
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            mf = next(n for n in zf.namelist() if n.endswith("plugin.json") and n.count("/") <= 1)
            m = json.loads(zf.read(mf))
        rows.append({"name": m["name"], "version": m.get("version", "0"), "description": m.get("description", ""),
                     "author": m.get("author", ""), "homepage": m.get("homepage", ""), "url": z.name,
                     "sha256": _sha256(data), "size": len(data)})
    out = dist / "index.json"
    out.write_text(json.dumps({"name": name, "updated": time.strftime("%Y-%m-%d"), "addons": rows}, indent=1))
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd")
    a = sub.add_parser("pack", help="a plugin folder -> <name>-<version>.zip")
    a.add_argument("folder")
    a.add_argument("-o", "--out", default="dist")
    b = sub.add_parser("index", help="index.json for a folder of packages")
    b.add_argument("dist")
    b.add_argument("--name", default="OmniCortex add-ons")
    x = p.parse_args(argv)
    if x.cmd == "pack":
        print(pack(x.folder, x.out))
    elif x.cmd == "index":
        print(index(x.dist, x.name))
    else:
        p.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
