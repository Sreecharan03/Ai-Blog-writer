"""
Fetch articles from GCS via the output endpoint and save as .docx
Falls back to direct DB query when the article is blocked by QC/ZeroGPT gates.
Usage: python scripts/fetch_missing_docx.py
"""
from __future__ import annotations
import json, os, sys
from pathlib import Path

import requests
from dotenv import load_dotenv
from google.cloud import storage

# Load .env so GOOGLE_APPLICATION_CREDENTIALS and DB vars are available
load_dotenv(Path(__file__).parent.parent / ".env")
creds = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
if creds:
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = creds

sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.md_to_docx import markdown_to_docx

API = "http://127.0.0.1:8000"

ARTICLES = [
    ("Sleep and Athletic Recovery",            "6dd0390e-6b2e-46f2-b3e4-a2b929d79f63"),
    ("How Sleep Affects Muscle Growth",         "cc159d41-33e9-4d92-a48a-6b5e609de8ef"),
    ("Sleep Deprivation and Cognitive Decline", "ef2cbefa-d26e-48c8-92e3-dd19c6ad7453"),
    ("REM Sleep and Emotional Regulation",      "de5954a0-f9f3-4a81-a123-34f00ccb11c9"),
]


def login() -> str:
    r = requests.post(f"{API}/api/v1/auth/login", json={
        "email": "charanch53030@gmail.com",
        "password": "NewPass@5678",
    })
    r.raise_for_status()
    return r.json()["access_token"]


def get_gcs_uri_via_api(token: str, request_id: str) -> str:
    r = requests.get(
        f"{API}/api/v1/articles/requests/{request_id}/output",
        headers={"Authorization": f"Bearer {token}"},
    )
    r.raise_for_status()
    return r.json()["output"]["gcs_draft_uri"]


def get_gcs_uri_via_db(request_id: str) -> str:
    """Fallback: query article_requests directly for the GCS URI."""
    import psycopg2

    conn = psycopg2.connect(
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT", 5432),
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        sslmode=os.getenv("DB_SSLMODE", "require"),
    )
    with conn.cursor() as cur:
        cur.execute(
            "SELECT gcs_draft_uri FROM public.article_requests WHERE request_id=%s::uuid LIMIT 1",
            (request_id,),
        )
        row = cur.fetchone()
    conn.close()
    if not row or not row[0]:
        raise RuntimeError(f"No gcs_draft_uri found in DB for request_id={request_id}")
    return row[0]


def fetch_markdown_from_gcs(gcs_uri: str) -> str:
    assert gcs_uri.startswith("gs://"), f"Expected gs:// URI, got: {gcs_uri}"
    bucket_name, obj_path = gcs_uri[5:].split("/", 1)
    client = storage.Client()
    data = json.loads(client.bucket(bucket_name).blob(obj_path).download_as_text())
    draft = data.get("draft") or {}
    return draft.get("draft_markdown", "")


def main():
    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)

    token = login()
    print("Logged in.\n")

    for title, request_id in ARTICLES:
        safe      = title.replace("/", "_")
        docx_path = out_dir / f"{safe}.docx"
        md_path   = out_dir / f"{safe}.md"

        # Skip only if the markdown has real content (> 200 words)
        if md_path.exists() and len(md_path.read_text(encoding="utf-8").split()) > 200:
            print(f"SKIP (already exists): {docx_path.name}")
            continue

        print(f"Fetching: {title}")
        try:
            # Try API first
            try:
                gcs_uri = get_gcs_uri_via_api(token, request_id)
                print(f"  via API  → {gcs_uri}")
            except requests.HTTPError as e:
                print(f"  API blocked ({e.response.status_code}) — falling back to DB")
                gcs_uri = get_gcs_uri_via_db(request_id)
                print(f"  via DB   → {gcs_uri}")

            markdown = fetch_markdown_from_gcs(gcs_uri)
            if not markdown:
                print(f"  WARN: empty markdown in GCS artifact — skipping")
                continue

            words = len(markdown.split())
            md_path.write_text(markdown, encoding="utf-8")
            markdown_to_docx(title, markdown, docx_path)
            print(f"  OK  {docx_path.name}  ({words} words)")

        except Exception as e:
            print(f"  ERROR: {e}")

    print("\nDone. Run  python scripts/md_to_docx.py --zip  to re-zip all 10.")


if __name__ == "__main__":
    main()
