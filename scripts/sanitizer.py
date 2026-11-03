#!/usr/bin/env python3
"""
Prompt sanitizer (docs/INTEGRATED_OPTIONS.md): before a request leaves this computer for a linked host, replace
secrets and names you list with placeholders like [SECRET_1]; put the real values back into the reply here.

Found by pattern: API keys and tokens (OpenAI, Anthropic, GitHub, Hugging Face, AWS, Slack, Google, generic
"key = ..." assignments, bearer tokens, private key blocks, JWTs), e-mail addresses, IPv4 addresses, and any
terms you add (company names, project code names, hostnames), matched as whole words, case-insensitively.

What it does not do: it is pattern matching, not understanding. A secret in a form it has no pattern for, or a
name you did not list, goes through unchanged. Placeholders are per request; the remote model sees the same
placeholder for every copy of one value, so it can still reason about "the same key used twice".
"""
from __future__ import annotations

import re

PATTERNS = [
    ("PRIVATE_KEY", r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]+?-----END [A-Z ]*PRIVATE KEY-----"),
    ("SECRET", r"\bsk-(?:ant-|proj-)?[A-Za-z0-9_-]{20,}\b"),
    ("SECRET", r"\bgh[pousr]_[A-Za-z0-9]{30,}\b"),
    ("SECRET", r"\bgithub_pat_[A-Za-z0-9_]{40,}\b"),
    ("SECRET", r"\bhf_[A-Za-z0-9]{30,}\b"),
    ("SECRET", r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
    ("SECRET", r"\bxox[abpr]-[A-Za-z0-9-]{10,}\b"),
    ("SECRET", r"\bAIza[0-9A-Za-z_-]{35}\b"),
    ("SECRET", r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b"),
    ("SECRET", r"(?i)\bbearer (?P<v>[A-Za-z0-9._~+/-]{20,}=*)"),
    ("SECRET", r"(?i)\b(?:api[_-]?key|secret|token|password|passwd)[\"']?\s{0,3}[:=]\s{0,3}[\"']?(?P<v>[^\s\"',;]{8,})"),
    ("EMAIL", r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    ("IP", r"\b(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)\b"),
]
DEFAULTS = {"enabled": False, "terms": [], "patterns": True, "emails": True, "ips": False}


class Session:
    """One request's placeholders, so the reply can be restored."""

    def __init__(self, cfg: dict | None = None):
        self.cfg = {**DEFAULTS, **(cfg or {})}
        self.map: dict[str, str] = {}             # placeholder -> real value
        self.rev: dict[str, str] = {}
        self.counts: dict[str, int] = {}
        rules = []
        if self.cfg["patterns"]:
            rules += [(k, p) for k, p in PATTERNS if k not in ("EMAIL", "IP")]
        if self.cfg["emails"]:
            rules.append(next(x for x in PATTERNS if x[0] == "EMAIL"))
        if self.cfg["ips"]:
            rules.append(next(x for x in PATTERNS if x[0] == "IP"))
        for t in sorted({str(t).strip() for t in self.cfg["terms"] if str(t).strip()}, key=len, reverse=True):
            rules.append(("NAME", r"(?i)(?<![\w-])" + re.escape(t) + r"(?![\w-])"))
        self.rules = [(k, re.compile(p)) for k, p in rules]

    def _ph(self, kind: str, value: str) -> str:
        key = value.lower() if kind == "NAME" else value
        if key in self.rev:
            return self.rev[key]
        self.counts[kind] = self.counts.get(kind, 0) + 1
        ph = f"[{kind}_{self.counts[kind]}]"
        self.map[ph], self.rev[key] = value, ph
        return ph

    def clean(self, text: str) -> str:
        def sub(kind, m):
            if "v" in m.re.groupindex:                  # replace only the value, keep "Bearer " / "api_key="
                a, b = m.span("v")
                return m.group(0)[:a - m.start()] + self._ph(kind, m.group("v")) + m.group(0)[b - m.start():]
            return self._ph(kind, m.group(0))
        for kind, rx in self.rules:
            text = rx.sub(lambda m, k=kind: sub(k, m), text)
        return text

    def restore(self, text: str) -> str:
        if not self.map:
            return text
        rx = re.compile("|".join(re.escape(p) for p in sorted(self.map, key=len, reverse=True)))
        return rx.sub(lambda m: self.map[m.group(0)], text)

    def clean_messages(self, msgs: list[dict]) -> list[dict]:
        out = []
        for m in msgs:
            c = m.get("content")
            if isinstance(c, str):
                c = self.clean(c)
            elif isinstance(c, list):
                c = [{**p, "text": self.clean(p["text"])} if isinstance(p, dict) and isinstance(p.get("text"), str) else p
                     for p in c]
            out.append({**m, "content": c})
        return out


class StreamRestorer:
    """Restores placeholders in streamed text, holding back a possibly incomplete one ("[SECR") until it ends."""

    def __init__(self, s: Session):
        self.s, self.buf = s, ""

    def feed(self, piece: str) -> str:
        self.buf += piece
        cut = self.buf.rfind("[")
        if cut >= 0 and "]" not in self.buf[cut:] and len(self.buf) - cut <= 24:
            out, self.buf = self.buf[:cut], self.buf[cut:]
        else:
            out, self.buf = self.buf, ""
        return self.s.restore(out)

    def flush(self) -> str:
        out, self.buf = self.buf, ""
        return self.s.restore(out)
