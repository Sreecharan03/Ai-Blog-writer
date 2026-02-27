from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, Optional, Tuple

import psycopg2
from fastapi import APIRouter, Depends, Header, HTTPException, Query
from jose import JWTError, jwt
from pydantic import BaseModel, Field

from google.cloud import storage


router = APIRouter(prefix="/api/v1/articles", tags=["article-output"])


# ============================================================
# Settings loader (same pattern)
# ============================================================
try:
    from app.core.config import get_settings  # type: ignore
except Exception:
    from app.core.config import settings as _settings  # type: ignore

    def get_settings():  # type: ignore
        return _settings


def _pick(settings: Any, *names: str, default: Any = None) -> Any:
    for n in names:
        if hasattr(settings, n) and getattr(settings, n) not in (None, ""):
            return getattr(settings, n)
        n_lower = n.lower()
        if hasattr(settings, n_lower) and getattr(settings, n_lower) not in (None, ""):
            return getattr(settings, n_lower)
        env_val = os.getenv(n)
        if env_val not in (None, ""):
            return env_val
    return default


# ============================================================
# Auth
# ============================================================
class Claims(BaseModel):
    tenant_id: str
    user_id: str
    role: str
    exp: int


def require_claims(
    authorization: str = Header(..., description="Bearer <JWT>"),
) -> Claims:
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
    role = payload.get("role")
    exp = payload.get("exp")

    if not tenant_id or not user_id or not role or not exp:
        raise HTTPException(status_code=401, detail="Token missing required claims")

    return Claims(tenant_id=str(tenant_id), user_id=str(user_id), role=str(role), exp=int(exp))


# ============================================================
# DB helpers (Supabase via DB_*)
# ============================================================
def _db_conn(settings: Any):
    host = _pick(settings, "DB_HOST", "SUPABASE_DB_HOST", "POSTGRES_HOST")
    port = int(_pick(settings, "DB_PORT", "SUPABASE_DB_PORT", "POSTGRES_PORT", default=5432))
    dbname = _pick(settings, "DB_NAME", "SUPABASE_DB_NAME", "POSTGRES_DB", default="postgres")
    user = _pick(settings, "DB_USER", "SUPABASE_DB_USER", "POSTGRES_USER")
    password = _pick(settings, "DB_PASSWORD", "SUPABASE_DB_PASSWORD", "POSTGRES_PASSWORD")
    sslmode = _pick(settings, "DB_SSLMODE", "SUPABASE_DB_SSLMODE", "POSTGRES_SSLMODE", default="require")

    missing = [k for k, v in [("host", host), ("user", user), ("password", password)] if not v]
    if missing:
        raise RuntimeError(f"Missing DB settings: {', '.join(missing)}")

    return psycopg2.connect(
        host=host,
        port=port,
        dbname=dbname,
        user=user,
        password=password,
        sslmode=sslmode,
        connect_timeout=8,
    )


def _log_job_event(conn, tenant_id: str, event_type: str, detail: Dict[str, Any], job_id: Optional[str] = None) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO public.job_events (event_id, tenant_id, request_id, job_id, event_type, detail)
            VALUES (%s, %s, %s, %s, %s, %s)
            """,
            (str(uuid.uuid4()), tenant_id, None, job_id, event_type, json.dumps(detail)),
        )


# ============================================================
# GCS helpers
# ============================================================
def _parse_gs_uri(gs_uri: str) -> Tuple[str, str]:
    if not gs_uri.startswith("gs://"):
        raise ValueError(f"Not a gs:// uri: {gs_uri}")
    rest = gs_uri[len("gs://") :]
    parts = rest.split("/", 1)
    bucket = parts[0]
    obj = parts[1] if len(parts) > 1 else ""
    return bucket, obj


def _signed_url_for_gs(gs_uri: str, *, minutes: int = 15) -> Optional[str]:
    """
    Best-effort signed URL. Requires credentials capable of signing (service account JSON works).
    If signing fails, returns None (still returns gcs_draft_uri).
    """
    try:
        bucket, obj = _parse_gs_uri(gs_uri)
        client = storage.Client(project=os.getenv("GCP_PROJECT_ID") or None)
        blob = client.bucket(bucket).blob(obj)
        return blob.generate_signed_url(
            version="v4",
            expiration=timedelta(minutes=int(minutes)),
            method="GET",
        )
    except Exception:
        return None


# ============================================================
# Response models
# ============================================================
class ArticleOutput(BaseModel):
    request_id: str
    tenant_id: str
    kb_id: str

    status: str
    attempt_count: int

    gcs_draft_uri: Optional[str] = None
    draft_fingerprint: Optional[str] = None
    draft_model: Optional[str] = None
    draft_meta: Optional[Dict[str, Any]] = None

    # optional convenience
    draft_signed_url: Optional[str] = None
    draft_signed_url_expires_minutes: Optional[int] = None

    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    last_run_at: Optional[str] = None


class ArticleOutputResponse(BaseModel):
    status: str = "ok"
    output: ArticleOutput


# ============================================================
# Endpoint
# ============================================================
@router.get("/requests/{request_id}/output", response_model=ArticleOutputResponse)
def get_article_output(
    request_id: str,
    claims: Claims = Depends(require_claims),
    signed_url: bool = Query(default=False, description="If true, return a short-lived signed URL for the draft"),
    signed_url_minutes: int = Query(default=15, ge=1, le=60),
):
    """
    Day 17 companion:
    Get output metadata for an article request (draft pointer now; final/QC later).
    """
    settings = get_settings()
    tenant_id = claims.tenant_id

    with _db_conn(settings) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                  request_id::text,
                  tenant_id::text,
                  kb_id::text,
                  status::text,
                  attempt_count,
                  gcs_draft_uri,
                  draft_fingerprint,
                  draft_model,
                  draft_meta,
                  created_at,
                  updated_at,
                  last_run_at
                FROM public.article_requests
                WHERE tenant_id=%s::uuid AND request_id=%s::uuid
                LIMIT 1;
                """,
                (tenant_id, request_id),
            )
            row = cur.fetchone()

        if not row:
            raise HTTPException(status_code=404, detail="Request not found")

        _log_job_event(conn, tenant_id, "article_output_fetched", {"request_id": request_id}, job_id=request_id)
        conn.commit()

    (
        rid,
        tid,
        kb,
        status,
        attempt_count,
        gcs_draft_uri,
        draft_fp,
        draft_model,
        draft_meta,
        created_at,
        updated_at,
        last_run_at,
    ) = row

    draft_signed = None
    if signed_url and gcs_draft_uri:
        draft_signed = _signed_url_for_gs(str(gcs_draft_uri), minutes=int(signed_url_minutes))

    output = ArticleOutput(
        request_id=str(rid),
        tenant_id=str(tid),
        kb_id=str(kb),
        status=str(status),
        attempt_count=int(attempt_count or 0),
        gcs_draft_uri=str(gcs_draft_uri) if gcs_draft_uri else None,
        draft_fingerprint=str(draft_fp) if draft_fp else None,
        draft_model=str(draft_model) if draft_model else None,
        draft_meta=draft_meta if isinstance(draft_meta, dict) else None,
        draft_signed_url=draft_signed,
        draft_signed_url_expires_minutes=int(signed_url_minutes) if (signed_url and draft_signed) else None,
        created_at=created_at.isoformat() if created_at else None,
        updated_at=updated_at.isoformat() if updated_at else None,
        last_run_at=last_run_at.isoformat() if last_run_at else None,
    )

    return ArticleOutputResponse(output=output)