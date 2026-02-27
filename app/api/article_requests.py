from __future__ import annotations

import json
import os
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

import psycopg2
import psycopg2.extras
from fastapi import APIRouter, Depends, Header, HTTPException, Query
from jose import JWTError, jwt
from pydantic import BaseModel, Field, field_validator


# ============================================================
# Router
# ============================================================
router = APIRouter(prefix="/api/v1/articles", tags=["article-requests"])


# ============================================================
# Settings loader (same pattern as your other files)
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
# Auth (same Claims style)
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
            (
                str(uuid.uuid4()),
                tenant_id,
                None,
                job_id,
                event_type,
                json.dumps(detail),
            ),
        )


# ============================================================
# (Optional safety) Ensure schema exists
# - If you already created tables in SQL editor, this is harmless.
# ============================================================
def _ensure_day16_schema(conn) -> None:
    with conn.cursor() as cur:
        # pgcrypto for gen_random_uuid (often already enabled)
        cur.execute("create extension if not exists pgcrypto;")

        # status enum
        cur.execute(
            """
            do $$
            begin
              create type public.article_request_status as enum (
                'queued','in_progress','completed','failed','cancelled','manual_review'
              );
            exception
              when duplicate_object then null;
            end $$;
            """
        )

        # updated_at trigger fn
        cur.execute(
            """
            create or replace function public.set_updated_at()
            returns trigger language plpgsql as $$
            begin
              new.updated_at = now();
              return new;
            end $$;
            """
        )

        # main table
        cur.execute(
            """
            create table if not exists public.article_requests (
              request_id uuid primary key default gen_random_uuid(),
              tenant_id uuid not null,
              kb_id uuid not null,

              title text not null,
              keywords text[] not null default '{}'::text[],
              length_target int not null default 2000,

              status public.article_request_status not null default 'queued',
              priority int not null default 0,

              attempt_count int not null default 0,
              qc_summary jsonb null,
              last_error text null,

              created_at timestamptz not null default now(),
              updated_at timestamptz not null default now()
            );
            """
        )

        # trigger
        cur.execute("drop trigger if exists trg_article_requests_updated_at on public.article_requests;")
        cur.execute(
            """
            create trigger trg_article_requests_updated_at
            before update on public.article_requests
            for each row execute function public.set_updated_at();
            """
        )

        # indexes for list filters
        cur.execute(
            """
            create index if not exists idx_article_requests_tenant_kb_created
              on public.article_requests (tenant_id, kb_id, created_at desc);
            """
        )
        cur.execute(
            """
            create index if not exists idx_article_requests_tenant_status_created
              on public.article_requests (tenant_id, status, created_at desc);
            """
        )
        cur.execute(
            """
            create index if not exists idx_article_requests_kb_status_created
              on public.article_requests (kb_id, status, created_at desc);
            """
        )

        # lock table
        cur.execute(
            """
            create table if not exists public.job_locks (
              job_id text primary key,
              lock_token text not null,
              locked_at timestamptz not null default now(),
              expires_at timestamptz not null default (now() + interval '30 minutes')
            );
            """
        )
        cur.execute("create index if not exists idx_job_locks_expires_at on public.job_locks (expires_at);")
        cur.execute("create index if not exists idx_job_locks_locked_at on public.job_locks (locked_at desc);")


# ============================================================
# Models
# ============================================================
ALLOWED_STATUSES = {"queued", "in_progress", "completed", "failed", "cancelled", "manual_review"}


class ArticleRequestCreate(BaseModel):
    kb_id: str
    title: str
    keywords: List[str] = Field(default_factory=list)
    length_target: int = 2000
    priority: int = 0

    @field_validator("title")
    @classmethod
    def _title_nonempty(cls, v: str) -> str:
        v = (v or "").strip()
        if not v:
            raise ValueError("title is required")
        return v

    @field_validator("length_target")
    @classmethod
    def _len_target(cls, v: int) -> int:
        if v < 300 or v > 10000:
            raise ValueError("length_target must be between 300 and 10000")
        return v

    @field_validator("keywords")
    @classmethod
    def _clean_keywords(cls, v: List[str]) -> List[str]:
        out = []
        seen = set()
        for k in v or []:
            kk = (k or "").strip()
            if not kk:
                continue
            kk = kk[:80]
            if kk.lower() in seen:
                continue
            seen.add(kk.lower())
            out.append(kk)
        return out


class ArticleRequestOut(BaseModel):
    request_id: str
    tenant_id: str
    kb_id: str
    title: str
    keywords: List[str] = Field(default_factory=list)
    length_target: int
    status: str
    priority: int
    attempt_count: int
    qc_summary: Optional[Dict[str, Any]] = None
    last_error: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class ArticleRequestListResponse(BaseModel):
    status: str = "ok"
    count: int
    requests: List[ArticleRequestOut] = Field(default_factory=list)


class ArticleRequestDetailResponse(BaseModel):
    status: str = "ok"
    request: ArticleRequestOut


# ============================================================
# Endpoints
# ============================================================

@router.post("/requests", response_model=ArticleRequestDetailResponse)
def create_article_request(
    payload: ArticleRequestCreate,
    claims: Claims = Depends(require_claims),
):
    """
    Day 16: Create article request (queue entry).
    """
    settings = get_settings()
    tenant_id = claims.tenant_id

    with _db_conn(settings) as conn:
        _ensure_day16_schema(conn)

        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO public.article_requests
                  (tenant_id, kb_id, title, keywords, length_target, status, priority)
                VALUES
                  (%s::uuid, %s::uuid, %s, %s::text[], %s, 'queued', %s)
                RETURNING
                  request_id, tenant_id, kb_id, title, keywords, length_target, status,
                  priority, attempt_count, qc_summary, last_error, created_at, updated_at;
                """,
                (
                    tenant_id,
                    payload.kb_id,
                    payload.title,
                    payload.keywords,
                    int(payload.length_target),
                    int(payload.priority),
                ),
            )
            row = cur.fetchone()

        _log_job_event(
            conn,
            tenant_id,
            "article_request_created",
            {"kb_id": payload.kb_id, "title": payload.title, "length_target": payload.length_target},
            job_id=str(row[0]),
        )
        conn.commit()

    req_out = ArticleRequestOut(
        request_id=str(row[0]),
        tenant_id=str(row[1]),
        kb_id=str(row[2]),
        title=row[3],
        keywords=list(row[4] or []),
        length_target=int(row[5]),
        status=str(row[6]),
        priority=int(row[7]),
        attempt_count=int(row[8]),
        qc_summary=row[9],
        last_error=row[10],
        created_at=row[11].isoformat() if row[11] else None,
        updated_at=row[12].isoformat() if row[12] else None,
    )
    return ArticleRequestDetailResponse(request=req_out)


@router.get("/requests", response_model=ArticleRequestListResponse)
def list_article_requests(
    claims: Claims = Depends(require_claims),
    kb_id: Optional[str] = Query(default=None),
    status: Optional[str] = Query(default=None, description="queued|in_progress|completed|failed|cancelled|manual_review"),
    limit: int = Query(default=20, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
):
    """
    Day 16: List requests (filter by kb_id/status; sorted by created_at desc).
    """
    settings = get_settings()
    tenant_id = claims.tenant_id

    if status and status not in ALLOWED_STATUSES:
        raise HTTPException(status_code=400, detail=f"Invalid status. Allowed: {sorted(ALLOWED_STATUSES)}")

    where = ["tenant_id=%s::uuid"]
    params: List[Any] = [tenant_id]

    if kb_id:
        where.append("kb_id=%s::uuid")
        params.append(kb_id)

    if status:
        where.append("status=%s::public.article_request_status")
        params.append(status)

    where_sql = " AND ".join(where)

    with _db_conn(settings) as conn:
        _ensure_day16_schema(conn)

        with conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT
                  request_id, tenant_id, kb_id, title, keywords, length_target, status,
                  priority, attempt_count, qc_summary, last_error, created_at, updated_at
                FROM public.article_requests
                WHERE {where_sql}
                ORDER BY created_at DESC
                LIMIT %s OFFSET %s;
                """,
                (*params, int(limit), int(offset)),
            )
            rows = cur.fetchall()

        _log_job_event(
            conn,
            tenant_id,
            "article_requests_listed",
            {"kb_id": kb_id, "status": status, "limit": limit, "offset": offset, "returned": len(rows)},
            job_id=None,
        )
        conn.commit()

    out: List[ArticleRequestOut] = []
    for r in rows:
        out.append(
            ArticleRequestOut(
                request_id=str(r[0]),
                tenant_id=str(r[1]),
                kb_id=str(r[2]),
                title=r[3],
                keywords=list(r[4] or []),
                length_target=int(r[5]),
                status=str(r[6]),
                priority=int(r[7]),
                attempt_count=int(r[8]),
                qc_summary=r[9],
                last_error=r[10],
                created_at=r[11].isoformat() if r[11] else None,
                updated_at=r[12].isoformat() if r[12] else None,
            )
        )

    return ArticleRequestListResponse(count=len(out), requests=out)


@router.get("/requests/{request_id}", response_model=ArticleRequestDetailResponse)
def get_article_request(
    request_id: str,
    claims: Claims = Depends(require_claims),
):
    """
    Day 16: Get request detail (includes status + attempt_count + qc_summary).
    """
    settings = get_settings()
    tenant_id = claims.tenant_id

    with _db_conn(settings) as conn:
        _ensure_day16_schema(conn)

        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                  request_id, tenant_id, kb_id, title, keywords, length_target, status,
                  priority, attempt_count, qc_summary, last_error, created_at, updated_at
                FROM public.article_requests
                WHERE tenant_id=%s::uuid AND request_id=%s::uuid
                LIMIT 1;
                """,
                (tenant_id, request_id),
            )
            row = cur.fetchone()

        if not row:
            raise HTTPException(status_code=404, detail="Request not found")

        _log_job_event(conn, tenant_id, "article_request_fetched", {"request_id": request_id}, job_id=request_id)
        conn.commit()

    req_out = ArticleRequestOut(
        request_id=str(row[0]),
        tenant_id=str(row[1]),
        kb_id=str(row[2]),
        title=row[3],
        keywords=list(row[4] or []),
        length_target=int(row[5]),
        status=str(row[6]),
        priority=int(row[7]),
        attempt_count=int(row[8]),
        qc_summary=row[9],
        last_error=row[10],
        created_at=row[11].isoformat() if row[11] else None,
        updated_at=row[12].isoformat() if row[12] else None,
    )
    return ArticleRequestDetailResponse(request=req_out)