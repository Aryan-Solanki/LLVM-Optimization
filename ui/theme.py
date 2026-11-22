"""Design tokens (docs/design-system.md) and stylesheet injection."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from urllib.parse import quote

import streamlit as st

# ---- colors --------------------------------------------------------------
INK = "#171717"
BODY = "#4d4d4d"
MUTE = "#8f8f8f"
FAINT = "#a1a1a1"
HAIRLINE = "#ebebeb"
HAIRLINE_SOFT = "#f2f2f2"
CANVAS = "#fafafa"
ELEVATED = "#ffffff"
LINK = "#0070f3"
LINK_DEEP = "#0761d1"
VIOLET = "#7928ca"
ERROR = "#ee0000"
WARNING = "#f5a623"

# ---- data encoding (validated with the dataviz palette checker) ----------
# IR and assembly keep the same hue everywhere: table swatches, meters, charts.
SERIES_IR = LINK
SERIES_ASM = VIOLET
SERIES_NEUTRAL = INK          # single measures with no IR/asm identity
BASELINE = MUTE               # O0 reference marks

FONT_SANS = "Geist, Arial, sans-serif"
FONT_MONO = "Geist Mono, ui-monospace, SFMono-Regular, Menlo, monospace"

_CSS_PATH = Path(__file__).with_name("styles.css")

# 16px line icons.  Rendered as CSS masks filled with currentColor, because
# st.html sanitizes inline <svg>.
ICON_PATHS = {
    "check": '<path d="M3.5 8.5l3 3 6-7"/>',
    "x": '<path d="M4.5 4.5l7 7M11.5 4.5l-7 7"/>',
    "minus": '<path d="M4 8h8"/>',
    "circle": '<circle cx="8" cy="8" r="3.5"/>',
    "info": '<circle cx="8" cy="8" r="6.25"/><path d="M8 7.25V11M8 5v.01"/>',
    "alert": '<path d="M8 2.25l6.25 11H1.75z"/><path d="M8 6.5v3M8 11.5v.01"/>',
    "error": '<circle cx="8" cy="8" r="6.25"/><path d="M5.75 5.75l4.5 4.5M10.25 5.75l-4.5 4.5"/>',
    "success": '<circle cx="8" cy="8" r="6.25"/><path d="M5.25 8.25l1.9 1.9 3.6-4.15"/>',
}


def _icon_css() -> str:
    rules = [
        ".lx-i{display:inline-block;flex:none;width:16px;height:16px;background-color:currentColor;"
        "-webkit-mask:var(--lx-i) center/contain no-repeat;mask:var(--lx-i) center/contain no-repeat;}"
    ]
    for name, body in ICON_PATHS.items():
        svg = (
            "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 16 16' fill='none' stroke='black' "
            f"stroke-width='1.6' stroke-linecap='round' stroke-linejoin='round'>{body.replace(chr(34), chr(39))}</svg>"
        )
        rules.append(f'.lx-i-{name}{{--lx-i:url("data:image/svg+xml,{quote(svg)}");}}')
    return "\n".join(rules)


@lru_cache(maxsize=4)
def _css(mtime: float) -> str:
    return _CSS_PATH.read_text(encoding="utf-8") + "\n" + _icon_css()


def inject_css() -> None:
    """Load the design-system stylesheet into the page (no layout space used)."""
    st.html(f"<style>{_css(_CSS_PATH.stat().st_mtime)}</style>")
