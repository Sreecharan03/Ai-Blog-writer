from __future__ import annotations

from datetime import datetime, timezone
import os
from typing import Any, Dict, Optional, List

from fastapi import APIRouter, Depends, Header, HTTPException, Query
from pydantic import BaseModel
from jose import JWTError, jwt
import uuid

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
# Auth (self-contained JWT decode)
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


def _is_system_admin(claims: Claims) -> bool:
    return claims.role == "system_admin"


# ---------------------------
# Router
# ---------------------------
router = APIRouter(prefix="/api/v1/finance", tags=["finance"])


@router.get("/pricebook")
def list_pricebook(
    claims: Claims = Depends(require_claims),
    vendor: Optional[str] = Query(default=None),
    item: Optional[str] = Query(default=None),
    as_of: Optional[str] = Query(default=None, description="ISO datetime; returns prices effective at this time"),
    limit: int = Query(default=50, ge=1, le=200),
):
    """
    List pricebook rows (useful to verify different model prices exist).
    """
    as_of_dt = None
    if as_of:
        try:
            as_of_dt = datetime.fromisoformat(as_of.replace("Z", "+00:00"))
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid as_of datetime. Use ISO format.")

    where = []
    params: List[Any] = []

    if vendor:
        where.append("vendor = %s")
        params.append(vendor)
    if item:
        where.append("item = %s")
        params.append(item)
    if as_of_dt:
        where.append("effective_from <= %s")
        params.append(as_of_dt)

    sql = """
        SELECT price_id, vendor, item, unit_price, effective_from
        FROM public.pricebook
    """
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY effective_from DESC NULLS LAST LIMIT %s"
    params.append(limit)

    settings = get_settings()
    with _supabase_conn(settings) as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql, tuple(params))
            rows = cur.fetchall()

    return {"status": "ok", "count": len(rows), "prices": rows}


@router.get("/usage-events")
def list_usage_events(
    claims: Claims = Depends(require_claims),
    tenant_id: Optional[str] = Query(default=None, description="system_admin only"),
    request_id: Optional[str] = Query(default=None),
    vendor: Optional[str] = Query(default=None),
    item: Optional[str] = Query(default=None),
    operation_name: Optional[str] = Query(default=None),
    date_from: Optional[str] = Query(default=None, description="ISO date/time"),
    date_to: Optional[str] = Query(default=None, description="ISO date/time"),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
):
    """
    Ledger view of usage events (Supabase-backed).
    Contract mentioned in plan: list usage events with filters. :contentReference[oaicite:7]{index=7}
    """
    target_tenant = claims.tenant_id
    if tenant_id:
        if not _is_system_admin(claims):
            raise HTTPException(status_code=403, detail="Only system_admin can query other tenants")
        target_tenant = tenant_id

    where = ["tenant_id = %s"]
    params: List[Any] = [target_tenant]

    if request_id:
        where.append("request_id = %s")
        params.append(request_id)
    if vendor:
        where.append("vendor = %s")
        params.append(vendor)
    if item:
        where.append("item = %s")
        params.append(item)
    if operation_name:
        where.append("operation_name = %s")
        params.append(operation_name)

    def _parse_dt(x: str) -> datetime:
        return datetime.fromisoformat(x.replace("Z", "+00:00"))

    if date_from:
        try:
            where.append("created_at >= %s")
            params.append(_parse_dt(date_from))
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid date_from. Use ISO format.")
    if date_to:
        try:
            where.append("created_at <= %s")
            params.append(_parse_dt(date_to))
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid date_to. Use ISO format.")

    sql = f"""
        SELECT
            event_id, tenant_id, user_id, request_id,
            operation_name, vendor, item,
            units, total_cost, metadata, created_at
        FROM public.usage_events
        WHERE {" AND ".join(where)}
        ORDER BY created_at DESC
        LIMIT %s OFFSET %s
    """
    params.extend([limit, offset])

    settings = get_settings()
    with _supabase_conn(settings) as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql, tuple(params))
            events = cur.fetchall()

    return {"status": "ok", "tenant_id": target_tenant, "count": len(events), "events": events}


@router.get("/requests/{request_id}/cost")
def request_cost_breakdown(
    request_id: str,
    claims: Claims = Depends(require_claims),
):
    """
    Cost breakdown for a request.
    """
    import uuid
    try:
        _ = uuid.UUID(request_id)
    except Exception:
        raise HTTPException(status_code=400, detail="request_id must be a valid UUID")

    settings = get_settings()
    with _supabase_conn(settings) as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT
                    operation_name, vendor, item,
                    COUNT(*) AS events,
                    COALESCE(SUM(total_cost), 0) AS total_cost
                FROM public.usage_events
                WHERE tenant_id = %s AND request_id = %s
                GROUP BY operation_name, vendor, item
                ORDER BY total_cost DESC
                """,
                (claims.tenant_id, request_id),
            )
            breakdown = cur.fetchall()

            cur.execute(
                """
                SELECT COALESCE(SUM(total_cost), 0) AS total_cost
                FROM public.usage_events
                WHERE tenant_id = %s AND request_id = %s
                """,
                (claims.tenant_id, request_id),
            )
            total = cur.fetchone()

    return {
        "status": "ok",
        "tenant_id": claims.tenant_id,
        "request_id": request_id,
        "total_cost": (total or {}).get("total_cost", 0),
        "breakdown": breakdown,
    }


@router.get("/tenants/{tenant_id}/summary")
def tenant_monthly_summary(
    tenant_id: str,
    claims: Claims = Depends(require_claims),
    month: Optional[str] = Query(default=None, description="YYYY-MM; defaults to current UTC month"),
):
    """
    Monthly summary for a tenant.
    Contract mentioned in plan: /finance/tenants/{tenant_id}/summary :contentReference[oaicite:9]{index=9}
    """
    if tenant_id != claims.tenant_id and not _is_system_admin(claims):
        raise HTTPException(status_code=403, detail="Tenant mismatch")

    now = datetime.now(timezone.utc)
    if month:
        try:
            y, m = month.split("-")
            year = int(y)
            mon = int(m)
            start = datetime(year, mon, 1, tzinfo=timezone.utc)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid month. Use YYYY-MM")
    else:
        start = datetime(now.year, now.month, 1, tzinfo=timezone.utc)

    # compute end-of-month
    if start.month == 12:
        end = datetime(start.year + 1, 1, 1, tzinfo=timezone.utc)
    else:
        end = datetime(start.year, start.month + 1, 1, tzinfo=timezone.utc)

    settings = get_settings()
    with _supabase_conn(settings) as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT COALESCE(SUM(total_cost), 0) AS total_cost
                FROM public.usage_events
                WHERE tenant_id = %s AND created_at >= %s AND created_at < %s
                """,
                (tenant_id, start, end),
            )
            total = cur.fetchone()

            cur.execute(
                """
                SELECT
                    DATE(created_at) AS day,
                    COALESCE(SUM(total_cost), 0) AS total_cost
                FROM public.usage_events
                WHERE tenant_id = %s AND created_at >= %s AND created_at < %s
                GROUP BY DATE(created_at)
                ORDER BY day ASC
                """,
                (tenant_id, start, end),
            )
            daily = cur.fetchall()

    return {
        "status": "ok",
        "tenant_id": tenant_id,
        "month": start.strftime("%Y-%m"),
        "total_cost": (total or {}).get("total_cost", 0),
        "daily": daily,
    }
