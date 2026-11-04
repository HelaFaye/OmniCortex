"""A plugin script's way to call the Studio that runs it: OMNICORTEX_URL and OMNICORTEX_TOKEN come from Studio.

    import plugin_api            # (sys.path gains OmniCortex's scripts/ folder: see a plugin's own scripts)
    plugin_api.chat(model, messages)  -> the reply text
    plugin_api.call("/api/system")   -> JSON
Studio is on this computer, so a certificate it made for itself is accepted for 127.0.0.1 only.
"""
from __future__ import annotations

import json
import os
import ssl
import urllib.request


def base() -> str:
    u = os.environ.get("OMNICORTEX_URL")
    if not u:
        raise RuntimeError("not run by Studio (no OMNICORTEX_URL)")
    return u.rstrip("/")


def call(path: str, body=None, timeout: float = 120):
    req = urllib.request.Request(base() + path, method="POST" if body is not None else "GET",
                                 data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Content-Type": "application/json",
                                          **({"Authorization": "Bearer " + os.environ["OMNICORTEX_TOKEN"]}
                                             if os.environ.get("OMNICORTEX_TOKEN") else {})})
    ctx = None
    if base().startswith("https://127.0.0.1"):
        ctx = ssl.create_default_context()
        ctx.check_hostname, ctx.verify_mode = False, ssl.CERT_NONE
    with urllib.request.urlopen(req, timeout=timeout, context=ctx) as r:
        return json.loads(r.read())


def chat(model: str | None, messages: list[dict], max_tokens: int = 400, temperature: float = 0.0) -> str:
    r = call("/v1/chat/completions", {**({"model": model} if model else {}), "messages": messages,
                                      "max_tokens": max_tokens, "temperature": temperature})
    return ((r.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
