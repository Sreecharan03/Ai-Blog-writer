"""
Quick extraction test — run before the full pipeline to verify what content
is actually being pulled from a URL.

Usage:
  python scripts/test_extraction.py --url "https://www.cricbuzz.com/profiles/1413/virat-kohli"

Outputs:
  - Static extraction result (trafilatura/readability/BS4)
  - Playwright-rendered extraction result
  - Word counts and first 600 chars of each for comparison
"""

from __future__ import annotations

import argparse
import sys
import time

import requests

# ── path fix so we can import from app/ ──────────────────────────────────────
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.api.url_ingest import extract_from_html, _render_js_if_needed

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)
TIMEOUT = 20


def fetch_static(url: str) -> bytes:
    r = requests.get(
        url,
        headers={"User-Agent": USER_AGENT},
        timeout=TIMEOUT,
        allow_redirects=True,
    )
    r.raise_for_status()
    return r.content


def _divider(label: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {label}")
    print("="*60)


def _show(label: str, text: str, method: str) -> None:
    words = len(text.split())
    chars = len(text)
    _divider(label)
    print(f"  Method  : {method}")
    print(f"  Words   : {words}")
    print(f"  Chars   : {chars}")
    print(f"\n--- First 600 chars ---")
    print(text[:600])
    print(f"\n--- Last 300 chars ---")
    print(text[-300:] if len(text) > 300 else "(same as above)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Test URL extraction quality")
    parser.add_argument("--url", required=True, help="URL to test")
    parser.add_argument("--timeout", type=int, default=30, help="Playwright timeout seconds")
    args = parser.parse_args()

    url = args.url
    print(f"\nTesting extraction for: {url}\n")

    # ── 1. Static fetch + extraction ─────────────────────────────────────────
    print("Step 1: Static HTTP fetch + 3-layer extraction ...")
    t0 = time.monotonic()
    try:
        raw = fetch_static(url)
        headers = {"content-type": "text/html; charset=utf-8"}
        static_text, static_meta = extract_from_html(raw, url, headers)
        static_method = static_meta.get("extraction_method", "unknown")
    except Exception as e:
        static_text = ""
        static_method = f"FAILED: {e}"
    static_elapsed = time.monotonic() - t0
    print(f"  Done in {static_elapsed:.1f}s")

    _show("STATIC EXTRACTION", static_text, static_method)

    static_words = len(static_text.split())
    if static_words >= 150:
        print(f"\n  [PASS] Static extraction has {static_words} words — Playwright would NOT trigger (>= 150 words)")
    else:
        print(f"\n  [SPARSE] Static extraction has {static_words} words — Playwright WILL trigger (< 150 words)")

    # ── 2. Playwright fetch + extraction ─────────────────────────────────────
    print("\nStep 2: Playwright (headless Chromium) render + extraction ...")
    t0 = time.monotonic()
    try:
        rendered = _render_js_if_needed(url, USER_AGENT, args.timeout)
        if rendered:
            pw_text, pw_meta = extract_from_html(rendered, url, {"content-type": "text/html; charset=utf-8"})
            pw_method = pw_meta.get("extraction_method", "unknown") + "+playwright"
        else:
            pw_text = ""
            pw_method = "FAILED: _render_js_if_needed returned None"
    except Exception as e:
        pw_text = ""
        pw_method = f"FAILED: {e}"
    pw_elapsed = time.monotonic() - t0
    print(f"  Done in {pw_elapsed:.1f}s")

    _show("PLAYWRIGHT EXTRACTION", pw_text, pw_method)

    # ── 3. Comparison summary ─────────────────────────────────────────────────
    _divider("COMPARISON SUMMARY")
    sw = len(static_text.split())
    pw = len(pw_text.split())
    print(f"  Static  : {sw:>5} words  ({static_method})")
    print(f"  Playwright: {pw:>3} words  ({pw_method})")
    print(f"  Winner  : {'Playwright' if pw > sw else 'Static'} (+{abs(pw - sw)} words)")

    if pw > sw * 1.5:
        print("\n  [VERDICT] Playwright extracts significantly more — JS-rendered page confirmed.")
        print("            Pipeline will use Playwright for this URL automatically.")
    elif pw > sw:
        print("\n  [VERDICT] Playwright extracts slightly more — marginal JS content.")
    else:
        print("\n  [VERDICT] Static and Playwright are similar — page is likely static HTML.")


if __name__ == "__main__":
    main()
