from __future__ import annotations

import os
import uuid
from typing import Any, List, Optional

from fastapi import APIRouter, Depends, Header, HTTPException, Query
from pydantic import BaseModel
from jose import JWTError, jwt

import psycopg2
from psycopg2.extras import RealDictCursor
from psycopg2 import sql

# ---------------------------
# Settings loader (supports both patterns)
# ---------------------------
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


def _supabase_conn(settings: Any):
    host = _pick(settings, "SUPABASE_DB_HOST", "DB_HOST", "POSTGRES_HOST")
    port = int(_pick(settings, "SUPABASE_DB_PORT", "DB_PORT", "POSTGRES_PORT", default=5432))
    dbname = _pick(settings, "SUPABASE_DB_NAME", "DB_NAME", "POSTGRES_DB", default="postgres")
    user = _pick(settings, "SUPABASE_DB_USER", "DB_USER", "POSTGRES_USER")
    password = _pick(settings, "SUPABASE_DB_PASSWORD", "DB_PASSWORD", "POSTGRES_PASSWORD")
    sslmode = _pick(settings, "SUPABASE_DB_SSLMODE", "DB_SSLMODE", "POSTGRES_SSLMODE", default="require")

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


# ---------------------------
# Auth (JWT decode)
# ---------------------------
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


# ---------------------------
# Helpers
# ---------------------------
def _require_uuid(value: str, field: str):
    try:
        uuid.UUID(value)
    except Exception:
        raise HTTPException(status_code=400, detail=f"{field} must be a valid UUID")


def _get_table_columns(cur, schema: str, table: str) -> List[str]:
    cur.execute(
        """
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema = %s AND table_name = %s
        """,
        (schema, table),
    )
    return [r["column_name"] for r in cur.fetchall()]


def _best_order_column(cols: List[str]) -> str:
    # prefer newest-first timestamps if present
    for c in ("created_at", "inserted_at", "uploaded_at", "updated_at"):
        if c in cols:
            return c
    # fallback
    return "doc_id" if "doc_id" in cols else cols[0]


# ---------------------------
# API
# ---------------------------
router = APIRouter(prefix="/api/v1/kb", tags=["kb"])


@router.get("/{kb_id}/docs")
def list_kb_docs(
    kb_id: str,
    limit: int = Query(20, ge=1, le=200),
    offset: int = Query(0, ge=0),
    claims: Claims = Depends(require_claims),
):
    """
    List documents ingested into a KB (tenant-safe).
    This endpoint is schema-tolerant: it does not assume optional columns exist.
    """
    _require_uuid(kb_id, "kb_id")

    try:
        settings = get_settings()
        with _supabase_conn(settings) as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cols = _get_table_columns(cur, "public", "documents")
                if "tenant_id" not in cols or "kb_id" not in cols:
                    raise HTTPException(status_code=500, detail="public.documents missing required columns")

                order_col = _best_order_column(cols)

                cur.execute(
                    """
                    SELECT COUNT(*)::int AS count
                    FROM public.documents
                    WHERE tenant_id = %s AND kb_id = %s
                    """,
                    (claims.tenant_id, kb_id),
                )
                total = int((cur.fetchone() or {}).get("count", 0))

                q = sql.SQL(
                    """
                    SELECT *
                    FROM public.documents
                    WHERE tenant_id = %s AND kb_id = %s
                    ORDER BY {order_col} DESC NULLS LAST
                    LIMIT %s OFFSET %s
                    """
                ).format(order_col=sql.Identifier(order_col))

                cur.execute(q, (claims.tenant_id, kb_id, limit, offset))
                docs = cur.fetchall()

        return {
            "status": "ok",
            "tenant_id": claims.tenant_id,
            "kb_id": kb_id,
            "count": total,
            "limit": limit,
            "offset": offset,
            "order_by": order_col,
            "docs": docs,
        }

    except HTTPException:
        raise
    except psycopg2.Error as e:
        return {"status": "not_ok", "error": str(e).split("\n")[0]}
    except Exception as e:
        return {"status": "not_ok", "error": str(e)}


@router.get("/{kb_id}/docs/{doc_id}")
def get_kb_doc(
    kb_id: str,
    doc_id: str,
    claims: Claims = Depends(require_claims),
):
    """
    Get one document metadata (tenant-safe).
    Schema-tolerant (SELECT *).
    """
    _require_uuid(kb_id, "kb_id")
    _require_uuid(doc_id, "doc_id")

    try:
        settings = get_settings()
        with _supabase_conn(settings) as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cols = _get_table_columns(cur, "public", "documents")
                if "tenant_id" not in cols or "kb_id" not in cols:
                    raise HTTPException(status_code=500, detail="public.documents missing required columns")

                # doc_id might have different name in some schemas; but your ingest response shows doc_id.
                id_col = "doc_id" if "doc_id" in cols else None
                if not id_col:
                    raise HTTPException(status_code=500, detail="public.documents missing doc_id column")

                q = sql.SQL(
                    """
                    SELECT *
                    FROM public.documents
                    WHERE tenant_id = %s AND kb_id = %s AND {id_col} = %s
                    LIMIT 1
                    """
                ).format(id_col=sql.Identifier(id_col))

                cur.execute(q, (claims.tenant_id, kb_id, doc_id))
                doc = cur.fetchone()

        if not doc:
            raise HTTPException(status_code=404, detail="Document not found")

        return {"status": "ok", "doc": doc}

    except HTTPException:
        raise
    except psycopg2.Error as e:
        return {"status": "not_ok", "error": str(e).split("\n")[0]}
    except Exception as e:
        return {"status": "not_ok", "error": str(e)}
