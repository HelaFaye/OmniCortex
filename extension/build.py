#!/usr/bin/env python3
"""
Package the browser extension (docs/BROWSER_EXTENSION.md):

    dist/omnicortex-extension-chromium.zip   Chrome, Edge, Opera, Brave, Vivaldi (and Edge for Android)
    dist/omnicortex-extension-firefox.zip    Firefox desktop and Firefox for Android (128+)

Safari (macOS, iOS) takes the Chromium folder through Apple's converter, which needs Xcode:
    xcrun safari-web-extension-converter build/extension-chromium

    python extension/build.py [--out dist] [--version 0.1.0]
"""
import argparse
import json
import shutil
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SITES = ["https://huggingface.co/*", "https://hf.co/*", "https://hf-mirror.com/*", "https://modelscope.cn/*",
         "https://www.modelscope.cn/*", "https://lmstudio.ai/*"]


def manifest(browser: str, version: str) -> dict:
    m = {
        "manifest_version": 3,
        "name": "OmniCortex model reviews",
        "short_name": "OmniCortex",
        "version": version,
        "description": "Crowd-sourced OmniCortex reviews of the model on the page: verified accuracy, speed per "
                       "hardware, features and emotion summaries. The icon lights up blue when there is data.",
        "icons": {str(s): f"icons/lit-{s}.png" for s in (16, 32, 48, 128)},
        "action": {"default_popup": "popup.html", "default_title": "OmniCortex",
                   "default_icon": {str(s): f"icons/idle-{s}.png" for s in (16, 32, 48, 128)}},
        "options_ui": {"page": "options.html", "open_in_tab": True},
        "permissions": ["storage", "activeTab"],
        "host_permissions": ["https://raw.githubusercontent.com/*"],
        "optional_host_permissions": ["https://*/*", "http://*/*"],
        "content_scripts": [{"matches": SITES, "js": ["content.js"], "run_at": "document_idle"}],
    }
    if browser == "chromium":
        m["background"] = {"service_worker": "background.js"}
        m["minimum_chrome_version"] = "110"
    else:
        m["background"] = {"scripts": ["sites.js", "normalize.js", "background.js"]}
        m["browser_specific_settings"] = {"gecko": {"id": "omnicortex-reviews@omnicortex", "strict_min_version": "128.0"},
                                          "gecko_android": {"strict_min_version": "128.0"}}
    return m


def build(out: Path, version: str) -> list[Path]:
    out.mkdir(parents=True, exist_ok=True)
    made = []
    for browser in ("chromium", "firefox"):
        stage = ROOT / "build" / f"extension-{browser}"
        shutil.rmtree(stage, ignore_errors=True)
        shutil.copytree(HERE / "src", stage)
        shutil.copytree(HERE / "icons", stage / "icons")
        (stage / "manifest.json").write_text(json.dumps(manifest(browser, version), indent=1))
        z = out / f"omnicortex-extension-{browser}.zip"
        with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zf:
            for f in sorted(stage.rglob("*")):
                if f.is_file():
                    zf.write(f, f.relative_to(stage).as_posix())
        made.append(z)
    return made


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default=str(ROOT / "dist"))
    p.add_argument("--version", default="0.1.0")
    a = p.parse_args()
    for z in build(Path(a.out), a.version):
        print(z)
