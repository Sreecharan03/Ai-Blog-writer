from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple

import psycopg2
from fastapi import APIRouter, Depends, Header, HTTPException
from jose import JWTError, jwt
from pydantic import BaseModel, Field


router = APIRouter(prefix="/api/v1/articles", tags=["article-state"])


# ============================================================
# Settings loader
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


def _require_admin(claims: Claims) -> None:
    if (claims.role or "").lower() != "tenant_admin":
        raise HTTPException(status_code=403, detail="Only tenant_admin can change request state")


# ============================================================
# DB helpers (Supabase Postgres via DB_*)
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


def _ensure_day18_schema(conn) -> None:
    """
    Day 18 requires clear error reasons + safe retries.
    We add structured error columns if missing.
    """
    with conn.cursor() as cur:
        cur.execute("ALTER TABLE public.article_requests ADD COLUMN IF NOT EXISTS last_error_code text;")
        cur.execute("ALTER TABLE public.article_requests ADD COLUMN IF NOT EXISTS last_error_detail jsonb;")
        cur.execute("ALTER TABLE public.article_requests ADD COLUMN IF NOT EXISTS next_run_at timestamptz;")

        # lock table (already exists from Day 16) but ensure present
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS public.job_locks (
              job_id text PRIMARY KEY,
              lock_token text NOT NULL,
              locked_at timestamptz NOT NULL DEFAULT now(),
              expires_at timestamptz NOT NULL DEFAULT (now() + interval '30 minutes')
            );
            """
        )


def _is_locked(conn, job_id: str) -> bool:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT 1
            FROM public.job_locks
            WHERE job_id=%s AND expires_at >= now()
            LIMIT 1
            """,
            (job_id,),
        )
        return cur.fetchone() is not None


def _force_unlock(conn, job_id: str) -> int:
    with conn.cursor() as cur:
        cur.execute("DELETE FROM public.job_locks WHERE job_id=%s", (job_id,))
        return int(cur.rowcount or 0)


def _fetch_request(conn, tenant_id: str, request_id: str) -> Dict[str, Any]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT
              request_id::text, tenant_id::text, kb_id::text,
              title, keywords, length_target,
              status::text, priority, attempt_count,
              qc_summary, last_error, last_error_code, last_error_detail,
              gcs_draft_uri, draft_fingerprint, draft_model, draft_meta,
              created_at, updated_at, last_run_at, next_run_at
            FROM public.article_requests
            WHERE tenant_id=%s::uuid AND request_id=%s::uuid
            LIMIT 1
            """,
            (tenant_id, request_id),
        )
        row = cur.fetchone()

    if not row:
        raise HTTPException(status_code=404, detail="Request not found")

    keys = [
        "request_id", "tenant_id", "kb_id",
        "title", "keywords", "length_target",
        "status", "priority", "attempt_count",
        "qc_summary", "last_error", "last_error_code", "last_error_detail",
        "gcs_draft_uri", "draft_fingerprint", "draft_model", "draft_meta",
        "created_at", "updated_at", "last_run_at", "next_run_at",
    ]
    out: Dict[str, Any] = {}
    for k, v in zip(keys, row):
        if isinstance(v, datetime):
            out[k] = v.isoformat()
        else:
            out[k] = v
    # normalize arrays/json
    out["keywords"] = list(out.get("keywords") or [])
    return out


# ============================================================
# API models
# ============================================================
class StateResponse(BaseModel):
    status: str = "ok"
    request: Dict[str, Any]


class CancelRequest(BaseModel):
    reason: str = "user_cancelled"
    force_unlock: bool = False


class RetryRequest(BaseModel):
    reason: str = "manual_retry"
    force_unlock: bool = False
    priority: Optional[int] = None  # allow bump priority on retry
    skip_if_qc_pass: bool = False   # if True, don't overwrite a draft that already passes QC


# ============================================================
# Endpoints (Day 18)
# ============================================================

@router.post("/requests/{request_id}/cancel", response_model=StateResponse)
def cancel_request(
    request_id: str,
    payload: CancelRequest,
    claims: Claims = Depends(require_claims),
):
    """
    Contract endpoint for cancel (state transition).:contentReference[oaicite:3]{index=3}
    """
    _require_admin(claims)
    settings = get_settings()
    tenant_id = claims.tenant_id

    with _db_conn(settings) as conn:
        _ensure_day18_schema(conn)

        req = _fetch_request(conn, tenant_id, request_id)
        status = (req.get("status") or "").lower()

        # If in_progress and locked -> either force unlock or reject
        if status == "in_progress" and _is_locked(conn, request_id):
            if not payload.force_unlock:
                conn.commit()
                raise HTTPException(status_code=409, detail="Request is locked/in_progress. Use force_unlock=true to cancel anyway.")
            _force_unlock(conn, request_id)

        if status in ("completed",):
            conn.commit()
            raise HTTPException(status_code=409, detail="Cannot cancel a completed request")

        # idempotent: if already cancelled, return it
        if status == "cancelled":
            _log_job_event(conn, tenant_id, "article_cancel_noop", {"request_id": request_id}, job_id=request_id)
            conn.commit()
            return StateResponse(request=req)

        # perform transition
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE public.article_requests
                SET status='cancelled',
                    last_error=%s,
                    last_error_code=%s,
                    last_error_detail=%s::jsonb,
                    next_run_at=NULL
                WHERE tenant_id=%s::uuid AND request_id=%s::uuid
                """,
                (
                    f"cancelled: {payload.reason}",
                    "cancelled",
                    json.dumps({"reason": payload.reason}),
                    tenant_id,
                    request_id,
                ),
            )

        _log_job_event(conn, tenant_id, "article_cancelled", {"request_id": request_id, "reason": payload.reason}, job_id=request_id)
        conn.commit()

        out = _fetch_request(conn, tenant_id, request_id)
        return StateResponse(request=out)


@router.post("/requests/{request_id}/retry", response_model=StateResponse)
def retry_request(
    request_id: str,
    payload: RetryRequest,
    claims: Claims = Depends(require_claims),
):
    """
    Day 18 retry skeleton:
    - clears error fields
    - transitions to queued
    - optional force unlock (if stuck)
    - does NOT run generation itself (you still call /run)
    """
    _require_admin(claims)
    settings = get_settings()
    tenant_id = claims.tenant_id

    with _db_conn(settings) as conn:
        _ensure_day18_schema(conn)

        req = _fetch_request(conn, tenant_id, request_id)
        status = (req.get("status") or "").lower()

        # If locked -> either force unlock or reject
        if _is_locked(conn, request_id):
            if not payload.force_unlock:
                conn.commit()
                raise HTTPException(status_code=409, detail="Request is locked/in_progress. Use force_unlock=true to retry anyway.")
            _force_unlock(conn, request_id)

        # allowed states to retry
        if status in ("completed",):
            conn.commit()
            raise HTTPException(status_code=409, detail="Request already completed. Create a new request if you want a new article.")

        # Workflow guard: skip retry if current draft already passes QC
        if payload.skip_if_qc_pass:
            qc_summary = req.get("qc_summary")
            if isinstance(qc_summary, dict) and qc_summary.get("qc_pass"):
                conn.commit()
                return StateResponse(status="ok", request=req)

        # Transition to queued (attempt_count is not incremented here; /run increments)
        new_priority = payload.priority if payload.priority is not None else req.get("priority")

        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE public.article_requests
                SET status='queued',
                    priority=%s,
                    last_error=NULL,
                    last_error_code=NULL,
                    last_error_detail=NULL,
                    next_run_at=now()
                WHERE tenant_id=%s::uuid AND request_id=%s::uuid
                """,
                (int(new_priority or 0), tenant_id, request_id),
            )

        _log_job_event(
            conn,
            tenant_id,
            "article_retry_queued",
            {"request_id": request_id, "reason": payload.reason, "priority": int(new_priority or 0)},
            job_id=request_id,
        )
        conn.commit()

        out = _fetch_request(conn, tenant_id, request_id)
        return StateResponse(request=out)