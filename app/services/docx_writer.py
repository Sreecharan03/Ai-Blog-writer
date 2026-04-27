"""
Shared markdown-to-Word converter used by both the download API endpoint
and the scripts/md_to_docx.py CLI tool.
"""
from __future__ import annotations

import io
import re
from pathlib import Path
from typing import Union

from docx import Document
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH


# ── Encoding cleanup ──────────────────────────────────────────────────────────
# Family A: UTF-8 3-byte sequences mis-decoded as Latin-1.
# Each Latin-1 code point == its original byte, so we reconstruct the UTF-8
# bytes and re-decode. Covers the full General Punctuation / Misc block.
_LAT1_RE = re.compile(
    'â'                  # original byte 0xE2  -> Latin-1 U+00E2
    '[-¿]'         # continuation byte 2 (0x80-0xBF)
    '[-¿]'         # continuation byte 3 (0x80-0xBF)
)

# Family B: same bytes mis-decoded as CP1252.
# Use single-quoted strings throughout; the content may contain curly quotes.
_CP1252 = [
    ('â€™', '’'),   # right single quote  '
    ('â€œ', '“'),   # left  double quote  "
    ('â€', '”'),   # right double quote  "
    ('â€“', '—'),   # em-dash             -
    ('â€–', '–'),   # en-dash             -
    ('â€¦', '…'),   # ellipsis            ...
    ('â€˜', '‘'),   # left  single quote  '
]


def _lat1_fix(m: re.Match) -> str:
    raw = bytes(ord(c) for c in m.group(0))
    try:
        return raw.decode('utf-8')
    except UnicodeDecodeError:
        return m.group(0)


def clean(text: str) -> str:
    """
    Fix all mojibake, then replace typographic dashes with plain hyphens
    so the output reads naturally without AI-style long dashes.
    """
    text = text.lstrip('﻿')                  # strip BOM
    text = _LAT1_RE.sub(_lat1_fix, text)          # family A: reconstruct
    for bad, good in _CP1252:                     # family B: literal swap
        text = text.replace(bad, good)
    # After mojibake fix, replace any remaining typographic dashes
    text = text.replace('—', ' - ')          # em-dash
    text = text.replace('–', ' - ')          # en-dash
    text = text.replace('‑', '-')            # non-breaking hyphen
    return text


# ── Inline markdown -> Word runs ──────────────────────────────────────────────
_INLINE = re.compile(
    r'\*\*\*(.+?)\*\*\*'
    r'|\*\*(.+?)\*\*'
    r'|\*(.+?)\*'
    r'|`(.+?)`'
)


def _add_inline(para, text: str) -> None:
    text = clean(text)
    last = 0
    for m in _INLINE.finditer(text):
        if m.start() > last:
            para.add_run(text[last:m.start()])
        if m.group(1):
            r = para.add_run(m.group(1)); r.bold = True; r.italic = True
        elif m.group(2):
            r = para.add_run(m.group(2)); r.bold = True
        elif m.group(3):
            r = para.add_run(m.group(3)); r.italic = True
        elif m.group(4):
            r = para.add_run(m.group(4))
            r.font.name = 'Courier New'
            r.font.size = Pt(10)
        last = m.end()
    if last < len(text):
        para.add_run(text[last:])


def _add_hr(doc: Document) -> None:
    para = doc.add_paragraph()
    pPr = para._p.get_or_add_pPr()
    pBdr = OxmlElement('w:pBdr')
    bottom = OxmlElement('w:bottom')
    bottom.set(qn('w:val'), 'single')
    bottom.set(qn('w:sz'), '6')
    bottom.set(qn('w:space'), '1')
    bottom.set(qn('w:color'), 'AAAAAA')
    pBdr.append(bottom)
    pPr.append(pBdr)


# ── Public converter ───────────────────────────────────────────────────────────
def markdown_to_docx(
    title: str,
    markdown: str,
    dest: Union[Path, io.BytesIO],
) -> None:
    """
    Convert *markdown* to a Word document and write to *dest*.

    *dest* may be a filesystem Path or an in-memory BytesIO buffer.
    Seek the buffer to position 0 after this call before reading from it.
    """
    doc = Document()

    for section in doc.sections:
        section.top_margin    = Inches(1.0)
        section.bottom_margin = Inches(1.0)
        section.left_margin   = Inches(1.15)
        section.right_margin  = Inches(1.15)

    tp = doc.add_heading(clean(title), level=0)
    tp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    doc.add_paragraph('')

    for line in markdown.splitlines():
        stripped = line.strip()
        if not stripped:
            continue

        if stripped.startswith('### '):
            h = re.sub(r'\*+(.+?)\*+', r'\1', clean(stripped[4:]))
            doc.add_heading(h, level=3)
        elif stripped.startswith('## '):
            h = re.sub(r'\*+(.+?)\*+', r'\1', clean(stripped[3:]))
            doc.add_heading(h, level=2)
        elif stripped.startswith('# '):
            h = re.sub(r'\*+(.+?)\*+', r'\1', clean(stripped[2:]))
            doc.add_heading(h, level=1)
        elif re.match(r'^-{3,}$', stripped) or re.match(r'^\*{3,}$', stripped):
            _add_hr(doc)
        elif stripped.startswith('- ') or stripped.startswith('* '):
            _add_inline(doc.add_paragraph(style='List Bullet'), stripped[2:])
        elif re.match(r'^\d+[\.\)]\s', stripped):
            _add_inline(
                doc.add_paragraph(style='List Number'),
                re.sub(r'^\d+[\.\)]\s+', '', stripped),
            )
        elif stripped.startswith('> '):
            _add_inline(doc.add_paragraph(style='Quote'), stripped[2:])
        else:
            _add_inline(doc.add_paragraph(), stripped)

    if isinstance(dest, Path):
        dest.parent.mkdir(parents=True, exist_ok=True)
        doc.save(str(dest))
    else:
        doc.save(dest)
