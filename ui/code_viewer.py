"""Read-only code viewer: Pygments highlighting, sticky line numbers, search.

Highlighting is done per line (multi-line tokens such as C block comments are
split at newlines) so every line is an independent row that can be
highlighted, filtered or wrapped without breaking markup.
"""

from __future__ import annotations

import html as _html
from typing import List, Optional, Sequence, Tuple

import streamlit as st
from pygments.lexers import CLexer, GasLexer, LlvmLexer, TextLexer
from pygments.token import STANDARD_TYPES

LANGS = {
    "llvm": ("LLVM IR", LlvmLexer),
    "asm": ("Assembly", GasLexer),
    "c": ("C", CLexer),
    "text": ("Text", TextLexer),
}
MAX_LINES = 20_000          # beyond this the viewer shows the head and offers a download
CONTEXT = 2                 # lines of context around matches in "matches only" mode


def _css_class(ttype) -> str:
    while ttype is not None:
        cls = STANDARD_TYPES.get(ttype)
        if cls is not None:
            return cls
        ttype = ttype.parent
    return ""


@st.cache_data(show_spinner=False, max_entries=64)
def highlight_lines(text: str, lang: str) -> List[str]:
    """Return one HTML fragment per source line."""
    lexer = LANGS[lang][1](stripnl=False, ensurenl=False)
    lines: List[List[str]] = [[]]
    for ttype, value in lexer.get_tokens(text):
        cls = _css_class(ttype)
        pieces = value.split("\n")
        for i, piece in enumerate(pieces):
            if i:
                lines.append([])
            if piece:
                safe = _html.escape(piece, quote=False)
                lines[-1].append(f'<span class="{cls}">{safe}</span>' if cls else safe)
    out = ["".join(parts) for parts in lines]
    # get_tokens may add a trailing newline: keep the line count equal to the file's
    n_real = text.count("\n") + (0 if text.endswith("\n") else 1)
    return out[: max(n_real, 1)]


def find_matches(text: str, query: str) -> List[int]:
    if not query:
        return []
    q = query.lower()
    return [i for i, line in enumerate(text.splitlines()) if q in line.lower()]


def _visible_rows(n_lines: int, hits: Sequence[int], only_matches: bool) -> List[Optional[int]]:
    """Line indexes to show; ``None`` marks a collapsed gap."""
    if not only_matches or not hits:
        return list(range(min(n_lines, MAX_LINES)))
    keep = set()
    for h in hits:
        keep.update(range(max(0, h - CONTEXT), min(n_lines, h + CONTEXT + 1)))
    rows: List[Optional[int]] = []
    prev = -1
    for i in sorted(keep):
        if i != prev + 1:
            rows.append(None)
        rows.append(i)
        prev = i
    if prev < n_lines - 1:
        rows.append(None)
    return rows[:MAX_LINES]


def render_viewer(
    text: str,
    lang: str,
    filename: str,
    meta: Sequence[Tuple[str, str]] = (),
    query: str = "",
    only_matches: bool = False,
    wrap: bool = False,
    badge_html: str = "",
) -> str:
    lines = highlight_lines(text, lang)
    n = len(lines)
    hits = find_matches(text, query.strip())
    hit_set = set(hits)
    gutter = max(3, len(str(n)))

    rows_html = []
    for idx in _visible_rows(n, hits, only_matches and bool(query.strip())):
        if idx is None:
            rows_html.append('<div class="cv-gap"><span class="cv-n">⋯</span><span class="cv-c">lines hidden</span></div>')
            continue
        cls = "cv-l hit" if idx in hit_set else "cv-l"
        rows_html.append(f'<div class="{cls}"><span class="cv-n">{idx + 1}</span><span class="cv-c">{lines[idx] or " "}</span></div>')

    size = len(text.encode("utf-8"))
    size_txt = f"{size / 1024:.1f} KB" if size >= 1024 else f"{size} B"
    meta_parts = [f"<span><b>{n:,}</b> lines</span>", f"<span>{size_txt}</span>"]
    meta_parts += [f"<span>{_html.escape(v)}</span>" if not k else f"<span><b>{_html.escape(v)}</b> {_html.escape(k)}</span>" for k, v in meta]
    if query.strip():
        label = "match" if len(hits) == 1 else "matches"
        meta_parts.append(f'<span class="hit">{len(hits):,} {label} in lines</span>')

    trunc = ""
    if n > MAX_LINES and not (only_matches and query.strip()):
        trunc = f'<div class="cv-trunc">Showing the first {MAX_LINES:,} of {n:,} lines. Download the file to see the rest.</div>'

    lang_label = LANGS[lang][0]
    classes = f"cv lang-{lang}" + (" wrap" if wrap else "")
    return (
        f'<div class="{classes}" style="--cv-gutter:{gutter}ch">'
        f'<div class="cv-head"><div class="cv-file">{badge_html}<span class="name">{_html.escape(filename)}</span></div>'
        f'<div class="cv-meta"><span>{lang_label}</span>{"".join(meta_parts)}</div></div>'
        f'<div class="cv-body" tabindex="0" aria-label="{_html.escape(filename)} source"><div class="cv-lines">{"".join(rows_html)}</div></div>'
        f"{trunc}</div>"
    )
