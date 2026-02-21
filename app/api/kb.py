# app/api/kb.py
from __future__ import annotations

import os
import sys
import uuid
from typing import Any, Optional, List, Dict

from fastapi import APIRouter, Depends, Header, HTTPException, Query
from pydantic import BaseModel, Field
from jose import JWTError, jwt

import psycopg2
from psycopg2.extras import RealDictCursor


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
        nl = n.lower()
        if hasattr(settings, nl) and getattr(settings, nl) not in (None, ""):
            return getattr(settings, nl)
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
# Auth (self-contained JWT decode)
# ---------------------------
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
# KB table detection (prevents “blind” assumptions)
# ---------------------------
_KB_TABLE_CANDIDATES = ("public.knowledge_bases", "public.kbs", "public.kb")


def _detect_kb_table(conn) -> str:
    with conn.cursor() as cur:
        for t in _KB_TABLE_CANDIDATES:
            cur.execute("SELECT to_regclass(%s) IS NOT NULL AS ok", (t,))
            ok = cur.fetchone()[0]
            if ok:
                return t
    raise RuntimeError(
        "KB table not found. Create one of: public.knowledge_bases / public.kbs / public.kb "
        "with columns: kb_id uuid PK, tenant_id uuid, name text, description text null, scope text, "
        "created_at timestamptz default now(), updated_at timestamptz null."
    )


# ---------------------------
# API models
# ---------------------------
router = APIRouter(prefix="/api/v1/kb", tags=["kb"])


class KBCreateIn(BaseModel):
    name: str = Field(..., min_length=2, max_length=120)
    description: Optional[str] = Field(default=None, max_length=2000)
    scope: str = Field(default="tenant_private")  # keep simple (plan: tenant_private)
    # system_admin only override
    tenant_id: Optional[str] = None


class KBOut(BaseModel):
    kb_id: str
    tenant_id: str
    name: str
    description: Optional[str] = None
    scope: str
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


@router.post("")
def create_kb(payload: KBCreateIn, claims: Claims = Depends(require_claims)):
    target_tenant = payload.tenant_id or claims.tenant_id

    if payload.tenant_id and claims.role != "system_admin":
        raise HTTPException(status_code=403, detail="Only system_admin can create KB for other tenants")

    try:
        # validate tenant uuid when provided
        uuid.UUID(str(target_tenant))
    except Exception:
        raise HTTPException(status_code=400, detail="tenant_id must be a valid UUID")

    settings = get_settings()
    try:
        with _supabase_conn(settings) as conn:
            kb_table = _detect_kb_table(conn)
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                kb_id = str(uuid.uuid4())

                # Try with explicit kb_id first (most schemas)
                try:
                    cur.execute(
                        f"""
                        INSERT INTO {kb_table} (kb_id, tenant_id, name, description, scope)
                        VALUES (%s, %s, %s, %s, %s)
                        RETURNING kb_id, tenant_id, name, description, scope, created_at, updated_at
                        """,
                        (kb_id, target_tenant, payload.name, payload.description, payload.scope),
                    )
                except psycopg2.Error as e:
                    # fallback: if schema has kb_id default generated
                    if "column" in str(e).lower() and "kb_id" in str(e).lower():
                        cur.execute(
                            f"""
                            INSERT INTO {kb_table} (tenant_id, name, description, scope)
                            VALUES (%s, %s, %s, %s)
                            RETURNING kb_id, tenant_id, name, description, scope, created_at, updated_at
                            """,
                            (target_tenant, payload.name, payload.description, payload.scope),
                        )
                    else:
                        raise

                kb = cur.fetchone()

        return {"status": "ok", "kb": kb}

    except psycopg2.Error as e:
        return {"status": "not_ok", "error": str(e).split("\n")[0]}
    except Exception as e:
        return {"status": "not_ok", "error": str(e)}


@router.get("")
def list_kbs(
    claims: Claims = Depends(require_claims),
    tenant_id: Optional[str] = Query(default=None, description="system_admin only"),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
):
    target_tenant = tenant_id or claims.tenant_id
    if tenant_id and claims.role != "system_admin":
        raise HTTPException(status_code=403, detail="Only system_admin can list other tenants")

    settings = get_settings()
    try:
        with _supabase_conn(settings) as conn:
            kb_table = _detect_kb_table(conn)
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    f"""
                    SELECT kb_id, tenant_id, name, description, scope, created_at, updated_at
                    FROM {kb_table}
                    WHERE tenant_id = %s
                    ORDER BY created_at DESC NULLS LAST
                    LIMIT %s OFFSET %s
                    """,
                    (target_tenant, limit, offset),
                )
                rows = cur.fetchall()

        return {"status": "ok", "tenant_id": target_tenant, "count": len(rows), "kbs": rows}

    except psycopg2.Error as e:
        return {"status": "not_ok", "error": str(e).split("\n")[0]}
    except Exception as e:
        return {"status": "not_ok", "error": str(e)}


@router.get("/{kb_id}")
def get_kb(kb_id: str, claims: Claims = Depends(require_claims)):
    try:
        uuid.UUID(kb_id)
    except Exception:
        raise HTTPException(status_code=400, detail="kb_id must be a valid UUID")

    settings = get_settings()
    try:
        with _supabase_conn(settings) as conn:
            kb_table = _detect_kb_table(conn)
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    f"""
                    SELECT kb_id, tenant_id, name, description, scope, created_at, updated_at
                    FROM {kb_table}
                    WHERE tenant_id = %s AND kb_id = %s
                    LIMIT 1
                    """,
                    (claims.tenant_id, kb_id),
                )
                kb = cur.fetchone()

        if not kb:
            raise HTTPException(status_code=404, detail="KB not found")

        return {"status": "ok", "kb": kb}

    except psycopg2.Error as e:
        return {"status": "not_ok", "error": str(e).split("\n")[0]}
    except Exception as e:
        return {"status": "not_ok", "error": str(e)}


@router.delete("/{kb_id}")
def delete_kb(kb_id: str, claims: Claims = Depends(require_claims)):
    try:
        uuid.UUID(kb_id)
    except Exception:
        raise HTTPException(status_code=400, detail="kb_id must be a valid UUID")

    settings = get_settings()
    try:
        with _supabase_conn(settings) as conn:
            kb_table = _detect_kb_table(conn)
            with conn.cursor() as cur:
                cur.execute(
                    f"DELETE FROM {kb_table} WHERE tenant_id = %s AND kb_id = %s",
                    (claims.tenant_id, kb_id),
                )
                deleted = cur.rowcount > 0

        return {"status": "ok", "deleted": deleted, "kb_id": kb_id}

    except psycopg2.Error as e:
        return {"status": "not_ok", "error": str(e).split("\n")[0]}
    except Exception as e:
        return {"status": "not_ok", "error": str(e)}


# ---------------------------
# Real self-test (no mocks)
# ---------------------------
def main() -> int:
    """
    Real DB smoke test:
    - connect to Supabase Postgres
    - detect KB table
    - print row count
    """
    try:
        settings = get_settings()
        with _supabase_conn(settings) as conn:
            try:
                kb_table = _detect_kb_table(conn)
            except Exception as e:
                print(f"[kb.py self-test] KB table missing: {e}")
                return 2

            with conn.cursor() as cur:
                cur.execute(f"SELECT COUNT(*) FROM {kb_table}")
                n = cur.fetchone()[0]
                print(f"[kb.py self-test] OK. Using {kb_table}. Rows={n}")
        return 0
    except Exception as e:
        print(f"[kb.py self-test] FAILED: {e}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
