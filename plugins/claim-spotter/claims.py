"""Spotting checkable claims in text. Shared by check.py and collect.py."""
import re

PATTERNS = [
    ("quotation", re.compile(r"[\"“][^\"”\n]{12,300}[\"”]")),
    ("date", re.compile(r"\b(?:\d{1,2} )?(?:January|February|March|April|May|June|July|August|September|October|"
                        r"November|December) (?:\d{1,2}, )?\d{3,4}\b|\b\d{4}-\d{2}-\d{2}\b")),
    ("year", re.compile(r"\b(?:1[0-9]|20)\d{2}\b")),
    ("number", re.compile(r"(?<![\w.])[-+]?\d+(?:,\d{3})*(?:\.\d+)?(?:\s?(?:%|percent\b|million\b|billion\b|km\b|kg\b|m\b|°C|°F))?")),
]


def spot(text: str) -> list:
    """-> [[start, end, kind], ...], longest kinds first, no overlaps."""
    taken, out = [], []
    for kind, rx in PATTERNS:
        for m in rx.finditer(text):
            a, b = m.span()
            if b - a < 2 and kind == "number":
                continue
            if any(a < y and x < b for x, y in taken):
                continue
            taken.append((a, b))
            out.append([a, b, kind])
    return sorted(out)
