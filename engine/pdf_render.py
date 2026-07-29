"""Renders an already-generated gap-export markdown draft into a styled,
typeset PDF.

Pure rendering -- no LLM calls, never imports gaps.py, agent.py, nlp.py,
or pipeline.py, and never imports gap_export.py either. Takes exactly the
text the caller already has (the same content already shown on screen);
never regenerates it, so what downloads always matches what was
displayed -- no drift, no wasted extra LLM call.
"""
from __future__ import annotations

import re
from pathlib import Path

import markdown as md_lib

_FONTS_DIR = Path(__file__).resolve().parent.parent / "assets" / "fonts"


class PdfRenderError(Exception):
    """Message is safe to show directly to the user."""


def _font_face(weight: str, style: str, filename: str) -> str:
    uri = (_FONTS_DIR / filename).as_uri()
    return f'@font-face {{ font-family: "Liberation Serif"; src: url("{uri}"); font-weight: {weight}; font-style: {style}; }}'


_FONT_FACES = "\n".join(
    [
        _font_face("normal", "normal", "LiberationSerif-Regular.ttf"),
        _font_face("bold", "normal", "LiberationSerif-Bold.ttf"),
        _font_face("normal", "italic", "LiberationSerif-Italic.ttf"),
        _font_face("bold", "italic", "LiberationSerif-BoldItalic.ttf"),
    ]
)

def _base_css(gap_number: int) -> str:
    return f"""
{_FONT_FACES}
@page {{
  size: letter; margin: 0.75in 0.625in 0.6in 0.625in;
  @bottom-center {{ content: "Gap {gap_number} \\2014 page " counter(page); font-family: "Liberation Serif", serif; font-size: 8pt; color: #444; }}
}}
body {{ font-family: "Liberation Serif", serif; font-size: 10pt; line-height: 1.16; color: #000; }}
h1 {{ font-size: 18pt; text-align: center; margin: 0 0 12pt 0; }}
h2 {{ font-size: 11pt; margin: 14pt 0 6pt 0; letter-spacing: 0.3pt; }}
h3 {{ font-size: 10.5pt; font-style: italic; margin: 10pt 0 4pt 0; }}
p {{ margin: 0 0 6pt 0; text-align: justify; orphans: 2; widows: 2; }}
strong {{ font-weight: bold; }}
em {{ font-style: italic; }}
hr {{ display: none; }}
"""

_IEEE_EXTRA_CSS = """
.header h2 { text-align: left; }
.body-cols { column-count: 2; column-gap: 0.28in; margin-top: 4pt; }
.body-cols h2 { font-variant: small-caps; }
"""

_SINGLE_COL_EXTRA_CSS = """
.header h2 { text-align: left; }
"""


def _ieee_css(gap_number: int) -> str:
    return _base_css(gap_number) + _IEEE_EXTRA_CSS


def _single_col_css(gap_number: int) -> str:
    return _base_css(gap_number) + _SINGLE_COL_EXTRA_CSS


def _to_html(markdown_content: str) -> str:
    return md_lib.markdown(markdown_content, extensions=["extra"])


def _split_ieee_header(html: str) -> tuple[str, str]:
    """Splits at the second <h2 (the first is Abstract; the second starts
    the real body, e.g. "I. Introduction") so title + abstract + keywords
    span both columns while everything from there on splits into two."""
    matches = list(re.finditer(r"<h2[ >]", html))
    if len(matches) < 2:
        return html, ""
    split_at = matches[1].start()
    return html[:split_at], html[split_at:]


def render_pdf(markdown_content: str, fmt: str, gap_number: int) -> bytes:
    try:
        html_body = _to_html(markdown_content)

        if fmt == "ieee":
            header_html, body_html = _split_ieee_header(html_body)
            page_html = (
                f"<html><head><style>{_ieee_css(gap_number)}</style></head><body>"
                f'<div class="header">{header_html}</div>'
                f'<div class="body-cols">{body_html}</div>'
                f"</body></html>"
            )
        else:
            page_html = f"<html><head><style>{_single_col_css(gap_number)}</style></head><body>{html_body}</body></html>"

        import weasyprint

        return weasyprint.HTML(string=page_html).write_pdf()
    except Exception as e:
        raise PdfRenderError(f"Could not render the PDF: {e}") from e
