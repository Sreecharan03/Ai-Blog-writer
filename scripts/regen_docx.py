"""
Fetch the latest completed article from GCS and regenerate a clean Word doc
with all pipeline marker / structural fixes applied.

Handles old-pipeline GCS artifacts where META comment, hook prose,
Key Takeaways, and TOC bullets were packed onto the same lines.
"""
import io
import json
import os
import re
import sys
from pathlib import Path

import psycopg2
from google.cloud import storage

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("GOOGLE_APPLICATION_CREDENTIALS",
                      r"D:\Hare Krishna_ai_blog\ai-blog-writer-sa-key.json")

from app.services.docx_writer import markdown_to_docx

DB = dict(
    host="aws-1-ap-southeast-2.pooler.supabase.com",
    port=6543,
    dbname="postgres",
    user="postgres.dppacvjatqcbolulzxud",
    password="Aiblog@2026",
    sslmode="require",
    connect_timeout=8,
)

_STRUCTURAL_HEADER = re.compile(
    r'\*\*(?:Key takeaways|In this article)[:\*]*\*\*',
    re.IGNORECASE,
)
_EMBEDDED_HEADING = re.compile(r'(#{2,3}\s+.+)')


def preprocess_gcs_markdown(md: str) -> str:
    """
    Fix structural issues in old GCS-stored markdown where the assembler
    packed the META comment, hook prose, Key Takeaways, TOC bullets, and
    section headings onto the same lines without newline separators.

    Passes:
      1. Strip embedded <!-- META: ... --> comments from any line.
      2. Truncate any line at an embedded **Key takeaways** / **In this article:** marker.
      3. Split lines that have an embedded ## heading in the middle (e.g., a TOC
         bullet line that has "## Section {#anchor}" appended at the end).
    """
    fixed: list[str] = []

    for raw_line in md.split('\n'):
        line = raw_line

        # Pass 1: strip embedded META comment (may appear mid-line in old artifacts)
        line = re.sub(r'<!--\s*META:[^>]*-->\s*', '', line)

        # Pass 2: truncate at embedded structural block headers
        sh = _STRUCTURAL_HEADER.search(line)
        if sh:
            line = line[:sh.start()].rstrip()

        # Pass 3: split embedded H2/H3 heading out of a bullet/prose line
        #   e.g. "- [Link](#anchor) ## Heading {#id}" → "- [Link](#anchor)" + "## Heading"
        hm = _EMBEDDED_HEADING.search(line)
        if hm and not line.lstrip().startswith('#'):
            prefix = line[:hm.start()].rstrip()
            heading = hm.group(1)
            if prefix:
                fixed.append(prefix)
            fixed.append(heading)
            continue

        fixed.append(line)

    return '\n'.join(fixed)


def clean_markers(md: str) -> str:
    """Strip internal pipeline markers that must not appear in published output."""
    md = re.sub(r'\s*\[F\d+\]', '', md)
    md = re.sub(r'\s*\[VERIFY\]', '', md)
    return md


def fetch_gcs(gcs_uri: str) -> tuple[str, str]:
    bucket_name, obj_path = gcs_uri[5:].split("/", 1)
    gcs = storage.Client()
    raw = gcs.bucket(bucket_name).blob(obj_path).download_as_text()
    data = json.loads(raw)
    draft = data.get("draft") or {}
    markdown = (data.get("draft_markdown") or draft.get("draft_markdown") or "").strip()
    title = (data.get("title") or draft.get("title") or data.get("topic") or "").strip()
    return title, markdown


def main():
    conn = psycopg2.connect(**DB)
    cur = conn.cursor()
    cur.execute(
        """
        SELECT request_id, title, gcs_draft_uri, status
        FROM   public.article_requests
        WHERE  status = 'completed'
          AND  gcs_draft_uri IS NOT NULL
        ORDER BY created_at DESC
        LIMIT 5
        """
    )
    rows = cur.fetchall()
    conn.close()

    if not rows:
        print("No completed articles found.")
        return

    print("Recent completed articles:")
    for i, (rid, title, uri, status) in enumerate(rows):
        print(f"  [{i}] {title!r} | {status} | {rid}")

    row = rows[0]
    request_id, title, gcs_uri, status = row
    print(f"\nUsing: {title!r} ({request_id})")

    gcs_title, markdown = fetch_gcs(gcs_uri)
    title = gcs_title or title or "Article"

    # Stats before cleanup
    fact_count = len(re.findall(r'\[F\d+\]', markdown))
    verify_count = len(re.findall(r'\[VERIFY\]', markdown))
    ita_count = len(re.findall(r'In this article', markdown, re.IGNORECASE))
    print(f"\nBefore cleanup:  [F-id]={fact_count}  [VERIFY]={verify_count}  'In this article'={ita_count}")

    # Fix old-pipeline structural packing issues, then strip markers
    markdown_fixed = preprocess_gcs_markdown(markdown)
    markdown_clean = clean_markers(markdown_fixed)

    out_path = Path(__file__).parent.parent / "outputs" / "test_clean.docx"
    out_path.parent.mkdir(exist_ok=True)
    markdown_to_docx(title, markdown_clean, out_path)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
