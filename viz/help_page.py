"""/help/<page>: OmniCortex's docs (docs/*.md) rendered inside Studio, so links in the
UI can point at them. A small Markdown subset: headings, paragraphs, lists, tables,
code blocks, inline code, bold, italics and links. Everything else is escaped."""
from __future__ import annotations

import html
import re
from pathlib import Path

DOCS = Path(__file__).resolve().parents[1] / "docs"
NAME = re.compile(r"^[A-Z][A-Z0-9_]{1,40}$")


def _inline(s: str) -> str:
    out, parts = [], re.split(r"(`[^`]*`)", s)
    for p in parts:
        if p.startswith("`") and p.endswith("`") and len(p) > 1:
            out.append(f"<code>{html.escape(p[1:-1])}</code>")
            continue
        t = html.escape(p, quote=False)
        t = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", t)
        t = re.sub(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])", r"<i>\1</i>", t)

        def link(m):
            text, href = m.group(1), html.unescape(m.group(2))
            if re.fullmatch(r"[A-Z][A-Z0-9_]+\.md(#[\w-]+)?", href):
                href = "/help/" + href.replace(".md", "")
            elif not href.startswith(("http://", "https://", "/", "#")):
                return text
            return f'<a href="{html.escape(href)}">{text}</a>'
        t = re.sub(r"\[([^\]]+)\]\(([^)\s]+)\)", link, t)
        out.append(t)
    return "".join(out)


def render(md: str) -> str:
    lines, out, i = md.splitlines(), [], 0
    while i < len(lines):
        ln = lines[i]
        if ln.startswith("```"):
            j = i + 1
            while j < len(lines) and not lines[j].startswith("```"):
                j += 1
            out.append("<pre>" + html.escape("\n".join(lines[i + 1:j])) + "</pre>")
            i = j + 1
            continue
        m = re.match(r"(#{1,4})\s+(.*)", ln)
        if m:
            n = len(m.group(1))
            slug = re.sub(r"[^a-z0-9]+", "-", m.group(2).lower()).strip("-")
            out.append(f'<h{n} id="{slug}">{_inline(m.group(2))}</h{n}>')
            i += 1
            continue
        if ln.startswith("|") and i + 1 < len(lines) and re.match(r"^\|[\s:|-]+\|$", lines[i + 1]):
            head = [c.strip() for c in ln.strip("|").split("|")]
            rows, j = [], i + 2
            while j < len(lines) and lines[j].startswith("|"):
                rows.append([c.strip() for c in lines[j].strip("|").split("|")])
                j += 1
            out.append('<div class="tw"><table><tr>' + "".join(f"<th>{_inline(c)}</th>" for c in head) + "</tr>"
                       + "".join("<tr>" + "".join(f"<td>{_inline(c)}</td>" for c in r) + "</tr>" for r in rows)
                       + "</table></div>")
            i = j
            continue
        if re.match(r"^\s*([-*]|\d+\.)\s+", ln):
            ordered = bool(re.match(r"^\s*\d+\.", ln))
            items, j = [], i
            while j < len(lines) and (re.match(r"^\s*([-*]|\d+\.)\s+", lines[j]) or
                                      (lines[j].startswith("  ") and lines[j].strip())):
                if re.match(r"^\s*([-*]|\d+\.)\s+", lines[j]):
                    items.append(re.sub(r"^\s*([-*]|\d+\.)\s+", "", lines[j]))
                else:
                    items[-1] += " " + lines[j].strip()
                j += 1
            tag = "ol" if ordered else "ul"
            out.append(f"<{tag}>" + "".join(f"<li>{_inline(x)}</li>" for x in items) + f"</{tag}>")
            i = j
            continue
        if not ln.strip():
            i += 1
            continue
        para, j = [ln], i + 1
        while j < len(lines) and lines[j].strip() and not re.match(r"^(#|```|\||\s*([-*]|\d+\.)\s)", lines[j]):
            para.append(lines[j])
            j += 1
        out.append("<p>" + _inline(" ".join(x.strip() for x in para)) + "</p>")
        i = j
    return "\n".join(out)


def _ui_js() -> str:
    """The sidebar, tour and ? button every Studio page shares."""
    try:
        import ui_links
    except ImportError:
        return ""
    return ui_links.js()


def page(name: str) -> str | None:
    if not NAME.match(name):
        return None
    f = DOCS / f"{name}.md"
    if not f.is_file():
        return None
    body = render(f.read_text(encoding="utf-8"))
    index = "".join(f'<a href="/help/{p.stem}">{p.stem.replace("_", " ").title()}</a> '
                    for p in sorted(DOCS.glob("*.md")) if NAME.match(p.stem))
    return f"""<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>OmniCortex help: {html.escape(name.replace('_', ' ').title())}</title>
<script>try{{const d=document.documentElement,t=localStorage.getItem('ns-theme')||'dark';d.dataset.theme=localStorage.getItem('ns-theme-base')||(t==='light'?'light':'dark');const v=JSON.parse(localStorage.getItem('ns-theme-vars')||'{{}}');for(const k in v)d.style.setProperty('--'+k,v[k])}}catch{{}}</script>
<style>
:root{{--bg:#fafaf9;--panel:#fff;--fg:#1c1c1a;--mut:#6b6b66;--line:#e4e4df;--acc:#2d5bd7;--code:#f3f3f0}}
@media (prefers-color-scheme:dark){{:root:not([data-theme=light]){{--bg:#151514;--panel:#1d1d1b;--fg:#ececea;--mut:#9a9a94;--line:#33332f;--acc:#6f93ff;--code:#262624}}}}
:root[data-theme=dark]{{--bg:#151514;--panel:#1d1d1b;--fg:#ececea;--mut:#9a9a94;--line:#33332f;--acc:#6f93ff;--code:#262624}}
body{{margin:0;font:15px/1.6 ui-sans-serif,system-ui;background:var(--bg);color:var(--fg)}}
header{{padding:.6rem 16px;border-bottom:1px solid var(--line);background:var(--panel);font-size:13px}}
header a{{color:var(--acc);margin-right:.7rem}}main{{max-width:860px;margin:0 auto;padding:1rem 16px 3rem}}
a{{color:var(--acc)}}code{{font:13px ui-monospace,monospace;background:var(--code);padding:.05rem .3rem;border-radius:4px}}
pre{{font:13px/1.45 ui-monospace,monospace;background:var(--code);padding:.7rem;border-radius:6px;overflow-x:auto}}
.tw{{overflow-x:auto}}table{{border-collapse:collapse;margin:.6rem 0}}td,th{{border-bottom:1px solid var(--line);padding:.35rem .5rem;text-align:left;vertical-align:top;font-size:14px}}
th{{color:var(--mut);font-weight:500}}details{{margin:.5rem 0}}
</style>
<header><a href="/">← Studio</a><details style="display:inline"><summary style="display:inline;cursor:pointer;color:var(--acc)">All help pages</summary><div style="margin-top:.4rem">{index}</div></details></header>
<main>{body}</main>""" + "<script>" + _ui_js() + "</script>"
