from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import psycopg2
import psycopg2.extras
from fastapi import APIRouter, Depends, Header, HTTPException
from jose import JWTError, jwt
from pydantic import BaseModel

router = APIRouter(prefix="/api/v1", tags=["dashboard"])

# ── settings ────────────────────────────────────────────────
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
        env_val = os.getenv(n)
        if env_val not in (None, ""):
            return env_val
    return default


# ── auth ────────────────────────────────────────────────────
class Claims(BaseModel):
    tenant_id: str
    user_id: str
    role: str
    exp: int


def require_claims(authorization: str = Header(..., description="Bearer <JWT>")) -> Claims:
    if not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Missing Bearer token")
    token = authorization.split(" ", 1)[1].strip()
    secret = os.getenv("JWT_SECRET_KEY") or os.getenv("JWT_SECRET") or os.getenv("SECRET_KEY")
    alg = os.getenv("JWT_ALGORITHM", "HS256")
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
    if not all([tenant_id, user_id, role, exp]):
        raise HTTPException(status_code=401, detail="Token missing required claims")
    return Claims(tenant_id=str(tenant_id), user_id=str(user_id), role=str(role), exp=int(exp))


# ── db ──────────────────────────────────────────────────────
def _conn(settings: Any):
    return psycopg2.connect(
        host=_pick(settings, "DB_HOST", "POSTGRES_HOST"),
        port=int(_pick(settings, "DB_PORT", "POSTGRES_PORT", default=5432)),
        dbname=_pick(settings, "DB_NAME", "POSTGRES_DB", default="postgres"),
        user=_pick(settings, "DB_USER", "POSTGRES_USER"),
        password=_pick(settings, "DB_PASSWORD", "POSTGRES_PASSWORD"),
        sslmode=_pick(settings, "DB_SSLMODE", "POSTGRES_SSLMODE", default="require"),
        connect_timeout=8,
    )


_KB_TABLE_CANDIDATES = ("public.knowledge_bases", "public.kbs", "public.kb")


def _kb_table(cur) -> Optional[str]:
    for t in _KB_TABLE_CANDIDATES:
        cur.execute("SELECT to_regclass(%s) IS NOT NULL AS ok", (t,))
        row = cur.fetchone()
        if row and row["ok"]:
            return t
    return None


def _doc_url_col(cur) -> str:
    cur.execute(
        "SELECT column_name FROM information_schema.columns WHERE table_schema='public' AND table_name='documents'",
    )
    cols = {r["column_name"] for r in cur.fetchall()}
    for candidate in ("source_url", "url", "source", "origin_url"):
        if candidate in cols:
            return candidate
    return "doc_id"


# ── endpoint ─────────────────────────────────────────────────
@router.get("/dashboard")
def get_dashboard(claims: Claims = Depends(require_claims)) -> Dict[str, Any]:
    """
    Single-call summary for the authenticated tenant's dashboard.
    Returns article stats, KB count, recent articles, and recent ingestions.
    """
    tid = claims.tenant_id
    settings = get_settings()

    try:
        with _conn(settings) as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:

                # ── 1. Article counts by status ──────────────────
                cur.execute(
                    """
                    SELECT
                        COUNT(*)                                                  AS total,
                        COUNT(*) FILTER (WHERE status = 'completed')              AS completed,
                        COUNT(*) FILTER (WHERE status = 'in_progress')            AS in_progress,
                        COUNT(*) FILTER (WHERE status = 'queued')                 AS queued,
                        COUNT(*) FILTER (WHERE status = 'failed')                 AS failed,
                        COUNT(*) FILTER (WHERE status = 'cancelled')              AS cancelled,
                        COUNT(*) FILTER (WHERE status = 'manual_review')          AS manual_review
                    FROM public.article_requests
                    WHERE tenant_id = %s
                    """,
                    (tid,),
                )
                article_stats = dict(cur.fetchone() or {})

                # ── 2. KB count ──────────────────────────────────
                kb_tbl = _kb_table(cur)
                if kb_tbl:
                    cur.execute(
                        f"SELECT COUNT(*)::int AS total FROM {kb_tbl} WHERE tenant_id = %s",  # noqa: S608
                        (tid,),
                    )
                    kb_total = (cur.fetchone() or {}).get("total", 0)
                else:
                    kb_total = 0

                # ── 3. Recent 5 articles ─────────────────────────
                cur.execute(
                    """
                    SELECT request_id, title, status, kb_id,
                           qc_summary, last_error, created_at, updated_at
                    FROM public.article_requests
                    WHERE tenant_id = %s
                    ORDER BY created_at DESC NULLS LAST
                    LIMIT 5
                    """,
                    (tid,),
                )
                recent_articles: List[Dict] = [dict(r) for r in cur.fetchall()]
                for a in recent_articles:
                    for k in ("created_at", "updated_at"):
                        if a.get(k):
                            a[k] = a[k].isoformat()

                # ── 4. Recent 5 ingestions across all KBs ────────
                url_col = _doc_url_col(cur)
                if kb_tbl:
                    cur.execute(
                        f"""
                        SELECT d.doc_id, d.{url_col} AS source_url, d.kb_id,
                               kb.name AS kb_name, d.created_at
                        FROM public.documents d
                        LEFT JOIN {kb_tbl} kb
                              ON kb.kb_id = d.kb_id AND kb.tenant_id = d.tenant_id
                        WHERE d.tenant_id = %s
                        ORDER BY d.created_at DESC NULLS LAST
                        LIMIT 5
                        """,
                        (tid,),
                    )
                else:
                    cur.execute(
                        f"""
                        SELECT doc_id, {url_col} AS source_url, kb_id, NULL AS kb_name, created_at
                        FROM public.documents
                        WHERE tenant_id = %s
                        ORDER BY created_at DESC NULLS LAST
                        LIMIT 5
                        """,
                        (tid,),
                    )
                recent_ingestions: List[Dict] = [dict(r) for r in cur.fetchall()]
                for r in recent_ingestions:
                    if r.get("created_at"):
                        r["created_at"] = r["created_at"].isoformat()

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Dashboard query failed: {str(exc)[:300]}")

    return {
        "status": "ok",
        "articles": {
            "total": int(article_stats.get("total", 0)),
            "completed": int(article_stats.get("completed", 0)),
            "in_progress": int(article_stats.get("in_progress", 0)),
            "queued": int(article_stats.get("queued", 0)),
            "failed": int(article_stats.get("failed", 0)),
            "cancelled": int(article_stats.get("cancelled", 0)),
            "manual_review": int(article_stats.get("manual_review", 0)),
        },
        "kbs": {
            "total": int(kb_total),
        },
        "recent_articles": recent_articles,
        "recent_ingestions": recent_ingestions,
    }
