"""
Convert all markdown files in outputs/ to properly formatted Word documents.

Usage:
    python scripts/md_to_docx.py
    python scripts/md_to_docx.py --dir outputs --zip

All conversion logic lives in app/services/docx_writer.py.
"""
from __future__ import annotations

import argparse
import re
import sys
import zipfile
from pathlib import Path

# Allow running from the project root without installing the package
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from app.services.docx_writer import markdown_to_docx  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="outputs", help="Directory containing .md files")
    parser.add_argument("--zip", action="store_true", help="Also create a zip of all .docx files")
    args = parser.parse_args()

    src = Path(args.dir)
    md_files = sorted(src.glob("*.md"))

    if not md_files:
        print(f"No .md files found in {src}/")
        return

    docx_files = []
    skipped = []
    for md_path in md_files:
        markdown = md_path.read_text(encoding="utf-8-sig").strip()
        words = len(markdown.split())

        if words < 50:
            skipped.append(md_path.name)
            continue
        if md_path.stem.startswith("draft_"):
            skipped.append(md_path.name)
            continue

        title = md_path.stem.replace("_", " ").strip()
        title = re.sub(r"^\d+\s+", "", title)

        out_path = md_path.with_suffix(".docx")
        markdown_to_docx(title, markdown, out_path)
        docx_files.append(out_path)
        print(f"  OK  {out_path.name}  ({words} words)")

    if skipped:
        print(f"\nSkipped {len(skipped)} file(s): {', '.join(skipped)}")
    print(f"\nConverted {len(docx_files)} files.")

    if args.zip and docx_files:
        from datetime import datetime
        zip_path = src / f"articles_docx_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for f in docx_files:
                zf.write(f, f.name)
        print(f"Zip saved: {zip_path}  ({len(docx_files)} docs)")


if __name__ == "__main__":
    main()
