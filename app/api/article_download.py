"""
GET /api/v1/articles/requests/{request_id}/download

Returns the completed article draft as a Word (.docx) attachment.
Requires only that gcs_draft_uri is set on the article_request row
(i.e. the LangGraph pipeline wrote a draft to GCS).

Does NOT enforce the ZeroGPT gate so you can download even if the AI-score
check was skipped — use the /output endpoint when you need gate enforcement.
"""
from __future__ import annotations

import io
import json
import os
import re
from typing import Any, Optional

import psycopg2
from fastapi import APIRouter, Depends, Header, HTTPException, Query
from fastapi.responses import StreamingResponse
from jose import JWTError, jwt

from google.cloud import storage

from app.services.docx_writer import markdown_to_docx

router = APIRouter(prefix="/api/v1/articles", tags=["article-download"])


# ── settings ──────────────────────────────────────────────────────────────────
try:
    from app.core.config import get_settings
except Exception:
    from app.core.config import settings as _s  # type: ignore

    def get_settings():  # type: ignore
        return _s


def _pick(settings: Any, *names: str, default: Any = None) -> Any:
    for n in names:
        if hasattr(settings, n) and getattr(settings, n) not in (None, ""):
            return getattr(settings, n)
        env_val = os.getenv(n)
        if env_val not in (None, ""):
            return env_val
    return default


# ── auth ──────────────────────────────────────────────────────────────────────
class _Claims:
    def __init__(self, tenant_id: str, user_id: str, role: str):
        self.tenant_id = tenant_id
        self.user_id = user_id
        self.role = role


def _require_claims(authorization: str = Header(..., description="Bearer <JWT>")) -> _Claims:
    if not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Missing Bearer token")
    token = authorization.split(" ", 1)[1].strip()
    secret = os.getenv("JWT_SECRET_KEY") or os.getenv("JWT_SECRET") or os.getenv("SECRET_KEY")
    alg = os.getenv("JWT_ALGORITHM") or "HS256"
    if not secret:
        raise HTTPException(status_code=500, detail="JWT secret not configured")
    try:
        payload = jwt.decode(token, secret, algorithms=[alg])
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")
    tenant_id = payload.get("tenant_id")
    user_id = payload.get("sub")
    role = payload.get("role", "user")
    if not tenant_id or not user_id:
        raise HTTPException(status_code=401, detail="Token missing required claims")
    return _Claims(tenant_id=str(tenant_id), user_id=str(user_id), role=str(role))


# ── DB ────────────────────────────────────────────────────────────────────────
def _db_conn(settings: Any):
    host     = _pick(settings, "DB_HOST", "SUPABASE_DB_HOST", "POSTGRES_HOST")
    port     = int(_pick(settings, "DB_PORT", "SUPABASE_DB_PORT", "POSTGRES_PORT", default=5432))
    dbname   = _pick(settings, "DB_NAME", "SUPABASE_DB_NAME", "POSTGRES_DB", default="postgres")
    user     = _pick(settings, "DB_USER", "SUPABASE_DB_USER", "POSTGRES_USER")
    password = _pick(settings, "DB_PASSWORD", "SUPABASE_DB_PASSWORD", "POSTGRES_PASSWORD")
    sslmode  = _pick(settings, "DB_SSLMODE", "SUPABASE_DB_SSLMODE", default="require")
    return psycopg2.connect(
        host=host, port=port, dbname=dbname,
        user=user, password=password, sslmode=sslmode,
        connect_timeout=8,
    )


# ── GCS ───────────────────────────────────────────────────────────────────────
def _fetch_from_gcs(gcs_uri: str) -> tuple[str, str]:
    """
    Download the GCS artifact JSON and return (title, markdown).
    Raises HTTPException on GCS failure or empty draft.
    """
    assert gcs_uri.startswith("gs://"), f"Unexpected URI: {gcs_uri}"
    bucket_name, obj_path = gcs_uri[5:].split("/", 1)
    try:
        client = storage.Client()
        raw = client.bucket(bucket_name).blob(obj_path).download_as_text()
        data = json.loads(raw)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"GCS download failed: {exc}")

    # Handle two GCS artifact shapes:
    # Shape A (blog pipeline): {"draft_markdown": "...", "title": "..."}
    # Shape B (legacy):        {"draft": {"draft_markdown": "...", "title": "..."}}
    draft    = data.get("draft") or {}
    markdown = (
        data.get("draft_markdown")          # Shape A
        or draft.get("draft_markdown")      # Shape B
        or ""
    ).strip()
    title    = (
        data.get("title")
        or draft.get("title")
        or data.get("topic")
        or ""
    ).strip()

    if not markdown:
        raise HTTPException(
            status_code=404,
            detail="Draft markdown is empty in GCS artifact. Pipeline may still be running.",
        )
    return title, markdown


# ── Endpoint ──────────────────────────────────────────────────────────────────
@router.get(
    "/requests/{request_id}/download",
    summary="Download article as Word (.docx)",
    response_description="Streams a .docx file attachment",
)
def download_article_docx(
    request_id: str,
    claims: _Claims = Depends(_require_claims),
    fallback_title: Optional[str] = Query(
        default=None,
        description="Override filename if the GCS artifact has no title field",
    ),
):
    """
    Stream the completed article draft as a Word document.

    - Looks up `gcs_draft_uri` from `article_requests` (tenant-scoped).
    - Fetches the JSON artifact from GCS.
    - Converts markdown to .docx in memory (no temp files).
    - Returns the file as `Content-Disposition: attachment`.

    **Status gate:** the request must be in `completed` or
    `completed_with_warnings` status. Call `/pipeline/{id}` to poll progress.
    """
    settings = get_settings()

    # ── 1. Look up the request row ───────────────────────────────────────────
    conn = _db_conn(settings)
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT gcs_draft_uri, status, title
                FROM   public.article_requests
                WHERE  tenant_id = %s::uuid
                  AND  request_id = %s::uuid
                LIMIT 1
                """,
                (claims.tenant_id, request_id),
            )
            row = cur.fetchone()
    finally:
        conn.close()

    if not row:
        raise HTTPException(status_code=404, detail="Article request not found")

    gcs_draft_uri, status, topic = row  # topic holds the title column value

    _READY = {"completed", "completed_with_warnings"}
    if status not in _READY:
        raise HTTPException(
            status_code=409,
            detail=(
                f"Article is not ready for download (status='{status}'). "
                "Wait for the pipeline to complete."
            ),
        )

    if not gcs_draft_uri:
        raise HTTPException(
            status_code=404,
            detail="No GCS draft URI recorded. The pipeline may not have written output yet.",
        )

    # ── 2. Fetch markdown from GCS ───────────────────────────────────────────
    gcs_title, markdown = _fetch_from_gcs(str(gcs_draft_uri))

    # Resolve best title: GCS artifact > fallback_title param > DB topic
    title = gcs_title or fallback_title or topic or "Article"

    # ── 3. Convert to .docx in memory ───────────────────────────────────────
    buf = io.BytesIO()
    markdown_to_docx(title, markdown, buf)
    buf.seek(0)

    # ── 4. Build a safe filename ─────────────────────────────────────────────
    safe = re.sub(r"[^\w\s-]", "", title).strip().replace(" ", "_")
    if not safe:
        safe = request_id[:8]
    filename = f"{safe}.docx"

    return StreamingResponse(
        buf,
        media_type=(
            "application/vnd.openxmlformats-officedocument"
            ".wordprocessingml.document"
        ),
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
