from __future__ import annotations

from datetime import datetime
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
import os
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, Header, HTTPException, Query
from pydantic import BaseModel, Field
from jose import JWTError, jwt

import psycopg2
from psycopg2.extras import RealDictCursor, Json

import uuid
import psycopg2.extras

# ---------------------------
# Settings loader (supports both patterns)
# ---------------------------
try:
    from app.core.config import get_settings  # type: ignore
except Exception:
    # fallback: app.core.config.settings
    from app.core.config import settings as _settings  # type: ignore

    def get_settings():  # type: ignore
        return _settings


def _pick(settings: Any, *names: str, default: Any = None) -> Any:
    for n in names:
        # prefer attributes on settings (supports both UPPER and snake_case)
        if hasattr(settings, n) and getattr(settings, n) not in (None, ""):
            return getattr(settings, n)
        n_lower = n.lower()
        if hasattr(settings, n_lower) and getattr(settings, n_lower) not in (None, ""):
            return getattr(settings, n_lower)
        # fallback to environment variables
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
    settings: Any = Depends(get_settings),
) -> Claims:
    if not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Missing Bearer token")

    token = authorization.split(" ", 1)[1].strip()

    secret = os.getenv("JWT_SECRET_KEY") or os.getenv("JWT_SECRET") or os.getenv("SECRET_KEY")
    alg = os.getenv("JWT_ALGORITHM") or "HS256"
    if not secret:
        raise HTTPException(status_code=500, detail="JWT secret not configured")

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
# API
# ---------------------------
router = APIRouter(prefix="/api/v1/finance", tags=["finance"])


class BudgetUpsertIn(BaseModel):
    monthly_cap: float = Field(..., gt=0)
    per_request_cap: float = Field(..., gt=0)
    # optional override (only system_admin should use it)
    tenant_id: Optional[str] = None


@router.post("/budgets")
def upsert_budget(payload: BudgetUpsertIn, claims: Claims = Depends(require_claims)):
    """
    Set/update budgets (Supabase-backed).
    - tenant_admin: can only write its own tenant
    - system_admin: can write any tenant via tenant_id override
    """
    target_tenant = payload.tenant_id or claims.tenant_id

    if payload.tenant_id and claims.role != "system_admin":
        raise HTTPException(status_code=403, detail="Only system_admin can set budgets for other tenants")

    if target_tenant != claims.tenant_id and claims.role != "system_admin":
        raise HTTPException(status_code=403, detail="Tenant mismatch")

    try:
        settings = get_settings()
        with _supabase_conn(settings) as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                # Find latest budget row for tenant (works even if no UNIQUE constraint exists)
                cur.execute(
                    """
                    SELECT budget_id
                    FROM public.budgets
                    WHERE tenant_id = %s
                    ORDER BY created_at DESC NULLS LAST
                    LIMIT 1
                    """,
                    (target_tenant,),
                )
                row = cur.fetchone()

                if row and row.get("budget_id"):
                    cur.execute(
                        """
                        UPDATE public.budgets
                        SET monthly_cap = %s,
                            per_request_cap = %s,
                            updated_at = NOW()
                        WHERE budget_id = %s
                        RETURNING budget_id, tenant_id, monthly_cap, per_request_cap,
                                  created_at, COALESCE(updated_at, NOW()) AS updated_at
                        """,
                        (payload.monthly_cap, payload.per_request_cap, row["budget_id"]),
                    )
                else:
                    cur.execute(
                        """
                        INSERT INTO public.budgets (tenant_id, monthly_cap, per_request_cap)
                        VALUES (%s, %s, %s)
                        RETURNING budget_id, tenant_id, monthly_cap, per_request_cap,
                                  created_at, COALESCE(updated_at, created_at) AS updated_at
                        """,
                        (target_tenant, payload.monthly_cap, payload.per_request_cap),
                    )

                budget = cur.fetchone()

        return {"status": "ok", "budget": budget}

    except psycopg2.Error as e:
        return {"status": "not_ok", "error": str(e).split("\n")[0]}
    except Exception as e:
        return {"status": "not_ok", "error": str(e)}


@router.get("/budgets")
def get_budget(
    claims: Claims = Depends(require_claims),
    tenant_id: Optional[str] = Query(default=None, description="Only for system_admin"),
):
    target_tenant = tenant_id or claims.tenant_id

    if tenant_id and claims.role != "system_admin":
        raise HTTPException(status_code=403, detail="Only system_admin can query other tenants")

    try:
        settings = get_settings()
        with _supabase_conn(settings) as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT budget_id, tenant_id, monthly_cap, per_request_cap,
                           created_at, COALESCE(updated_at, created_at) AS updated_at
                    FROM public.budgets
                    WHERE tenant_id = %s
                    ORDER BY created_at DESC NULLS LAST
                    LIMIT 1
                    """,
                    (target_tenant,),
                )
                budget = cur.fetchone()

        return {"status": "ok", "tenant_id": target_tenant, "budget": budget}

    except psycopg2.Error as e:
        return {"status": "not_ok", "error": str(e).split("\n")[0]}
    except Exception as e:
        return {"status": "not_ok", "error": str(e)}

# ---------------------------
# Pricebook (Day-5)
# ---------------------------

class PricebookUpsertIn(BaseModel):
    vendor: str = Field(..., min_length=1)
    item: str = Field(..., min_length=1)  # e.g., model name like "gemini-2.5-flash"
    unit_price: float = Field(..., gt=0)  # price per unit (you define unit: per 1K tokens, per call, etc.)
    effective_from: Optional[datetime] = None


@router.post("/pricebook")
def add_pricebook_row(payload: PricebookUpsertIn, claims: Claims = Depends(require_claims)):
    """
    Inserts a new pricebook row (keeps history).
    Later: cost calculator will pick latest effective row.
    """
    try:
        settings = get_settings()
        eff = payload.effective_from or datetime.utcnow()

        with _supabase_conn(settings) as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    """
                    INSERT INTO public.pricebook (vendor, item, unit_price, effective_from)
                    VALUES (%s, %s, %s, %s)
                    RETURNING
                      price_id,
                      vendor,
                      item,
                      unit_price::float8 as unit_price,
                      effective_from,
                      created_at
                    """,
                    (payload.vendor, payload.item, payload.unit_price, eff),
                )
                row = cur.fetchone()

        return {"status": "ok", "price": row}

    except psycopg2.Error as e:
        return {"status": "not_ok", "error": str(e).split("\n")[0]}
    except Exception as e:
        return {"status": "not_ok", "error": str(e)}


@router.get("/pricebook")
def list_pricebook(
    vendor: Optional[str] = Query(default=None),
    item: Optional[str] = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200),
    claims: Claims = Depends(require_claims),
):
    """
    Lists latest pricebook rows (optionally filtered).
    """
    try:
        settings = get_settings()
        where = []
        params = []

        if vendor:
            where.append("vendor = %s")
            params.append(vendor)
        if item:
            where.append("item = %s")
            params.append(item)

        where_sql = ("WHERE " + " AND ".join(where)) if where else ""

        with _supabase_conn(settings) as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    f"""
                    SELECT
                      price_id,
                      vendor,
                      item,
                      unit_price::float8 as unit_price,
                      effective_from,
                      created_at
                    FROM public.pricebook
                    {where_sql}
                    ORDER BY effective_from DESC NULLS LAST
                    LIMIT %s
                    """,
                    (*params, limit),
                )
                rows = cur.fetchall()

        return {"status": "ok", "count": len(rows), "prices": rows}

    except psycopg2.Error as e:
        return {"status": "not_ok", "error": str(e).split("\n")[0]}
    except Exception as e:
        return {"status": "not_ok", "error": str(e)}

# ---------------------------
# Usage Events + Cost Calc (Day-5)
# ---------------------------

class UsageEventIn(BaseModel):
    operation_name: str = Field(..., min_length=1)
    vendor: str = Field(..., min_length=1)
    item: str = Field(..., min_length=1)
    units: int = Field(..., gt=0)
    request_id: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

# ---------------------------
# Helpers
# ---------------------------
def _as_uuid_or_400(value: str, field_name: str) -> uuid.UUID:
    try:
        return uuid.UUID(value)
    except Exception:
        raise HTTPException(status_code=400, detail=f"{field_name} must be a valid UUID")


def _decimal_money(x: Any) -> Decimal:
    # convert numbers safely (avoid float artifacts)
    try:
        d = Decimal(str(x))
    except (InvalidOperation, ValueError):
        raise HTTPException(status_code=400, detail="Invalid numeric value")
    # keep 6 decimal places (enough for token/unit pricing)
    return d.quantize(Decimal("0.000001"), rounding=ROUND_HALF_UP)


def _month_spend(cur, tenant_id: str) -> Decimal:
    cur.execute(
        """
        SELECT COALESCE(SUM(total_cost), 0) AS total
        FROM public.usage_events
        WHERE tenant_id = %s
          AND created_at >= date_trunc('month', now())
        """,
        (tenant_id,),
    )
    row = cur.fetchone() or {}
    return _decimal_money(row.get("total", 0))


def _request_spend(cur, tenant_id: str, request_id: str) -> Decimal:
    cur.execute(
        """
        SELECT COALESCE(SUM(total_cost), 0) AS total
        FROM public.usage_events
        WHERE tenant_id = %s AND request_id = %s
        """,
        (tenant_id, request_id),
    )
    row = cur.fetchone() or {}
    return _decimal_money(row.get("total", 0))


def _latest_budget(cur, tenant_id: str) -> Optional[Dict[str, Any]]:
    cur.execute(
        """
        SELECT budget_id, monthly_cap, per_request_cap
        FROM public.budgets
        WHERE tenant_id = %s
        ORDER BY created_at DESC NULLS LAST
        LIMIT 1
        """,
        (tenant_id,),
    )
    return cur.fetchone()


def _unit_price(cur, vendor: str, item: str) -> Decimal:
    # supports model price changes over time
    cur.execute(
        """
        SELECT unit_price
        FROM public.pricebook
        WHERE vendor = %s AND item = %s
          AND effective_from <= now()
        ORDER BY effective_from DESC
        LIMIT 1
        """,
        (vendor, item),
    )
    row = cur.fetchone()
    if not row:
        raise HTTPException(status_code=400, detail=f"Price not found for {vendor}:{item}")
    return _decimal_money(row["unit_price"])


def _log_job_event(cur, tenant_id: str, request_id: Optional[str], event_type: str, detail: Dict[str, Any]):
    cur.execute(
        """
        INSERT INTO public.job_events (tenant_id, request_id, event_type, detail)
        VALUES (%s, %s, %s, %s)
        """,
        (tenant_id, request_id, event_type, Json(detail)),
    )


# ---------------------------
# Endpoint: POST /usage-events
# ---------------------------
@router.post("/usage-events")
def create_usage_event(payload: UsageEventIn, claims=Depends(require_claims)):
    """
    Inserts a usage_event ONLY if budget allows.
    Preserves idempotency for retries using (tenant_id, request_id, operation_name, metadata.step).
    """
    request_id: Optional[str] = None
    step = str((payload.metadata or {}).get("step", "default"))

    if payload.request_id:
        _as_uuid_or_400(payload.request_id, "request_id")
        request_id = payload.request_id

    settings = get_settings()

    try:
        with _supabase_conn(settings) as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:

                # 1) IDEMPOTENCY FIRST: if already inserted, return it (don’t re-check budget)
                if request_id:
                    cur.execute(
                        """
                        SELECT *
                        FROM public.usage_events
                        WHERE tenant_id = %s
                          AND request_id = %s
                          AND operation_name = %s
                          AND COALESCE(metadata->>'step','default') = %s
                        ORDER BY created_at DESC NULLS LAST
                        LIMIT 1
                        """,
                        (claims.tenant_id, request_id, payload.operation_name, step),
                    )
                    existing = cur.fetchone()
                    if existing:
                        return {"status": "ok", "duplicate": True, "usage_event": existing}

                # 2) price lookup + expected cost
                unit_price = _unit_price(cur, payload.vendor, payload.item)
                expected_cost = (unit_price * _decimal_money(payload.units)).quantize(
                    Decimal("0.000001"), rounding=ROUND_HALF_UP
                )

                # 3) budgets + spends
                budget = _latest_budget(cur, claims.tenant_id)
                month_cap = _decimal_money(budget["monthly_cap"]) if budget else None
                req_cap = _decimal_money(budget["per_request_cap"]) if budget else None

                month_spend = _month_spend(cur, claims.tenant_id)
                request_spend = _request_spend(cur, claims.tenant_id, request_id) if request_id else Decimal("0")

                month_projected = month_spend + expected_cost
                request_projected = request_spend + expected_cost

                # alert thresholds (simple & predictable)
                def _alert(projected: Decimal, cap: Optional[Decimal]) -> str:
                    if not cap or cap <= 0:
                        return "ok"
                    ratio = projected / cap
                    if ratio >= Decimal("1.0"):
                        return "blocked"
                    if ratio >= Decimal("0.95"):
                        return "critical"
                    if ratio >= Decimal("0.80"):
                        return "warn"
                    return "ok"

                month_alert = _alert(month_projected, month_cap)
                req_alert = _alert(request_projected, req_cap)

                # 4) BLOCK if caps exceeded
                reason = None
                if req_cap and request_projected > req_cap:
                    reason = "per_request_cap_exceeded"
                elif month_cap and month_projected > month_cap:
                    reason = "monthly_cap_exceeded"

                if reason:
                    _log_job_event(
                        cur,
                        tenant_id=claims.tenant_id,
                        request_id=request_id,
                        event_type="budget_blocked",
                        detail={
                            "reason": reason,
                            "operation_name": payload.operation_name,
                            "vendor": payload.vendor,
                            "item": payload.item,
                            "units": payload.units,
                            "unit_price": str(unit_price),
                            "expected_cost": str(expected_cost),
                            "month_spend": str(month_spend),
                            "month_cap": str(month_cap) if month_cap else None,
                            "month_projected": str(month_projected),
                            "request_spend": str(request_spend),
                            "request_cap": str(req_cap) if req_cap else None,
                            "request_projected": str(request_projected),
                            "alert_level": "blocked",
                            "step": step,
                        },
                    )
                    # 402 Payment Required is a clean “budget gate” signal for clients
                    raise HTTPException(
                        status_code=402,
                        detail={
                            "status": "blocked",
                            "reason": reason,
                            "tenant_id": claims.tenant_id,
                            "request_id": request_id,
                            "expected_cost": float(expected_cost),
                            "unit_price": float(unit_price),
                            "month_spend": float(month_spend),
                            "month_cap": float(month_cap) if month_cap else None,
                            "month_projected": float(month_projected),
                            "request_spend": float(request_spend),
                            "request_cap": float(req_cap) if req_cap else None,
                            "request_projected": float(request_projected),
                            "alert_level": "blocked",
                        },
                    )

                # 5) INSERT (allowed)
                cur.execute(
                    """
                    INSERT INTO public.usage_events
                        (tenant_id, user_id, request_id, operation_name, vendor, item,
                         units, unit_price, total_cost, metadata)
                    VALUES
                        (%s, %s, %s, %s, %s, %s,
                         %s, %s, %s, %s)
                    RETURNING *
                    """,
                    (
                        claims.tenant_id,
                        claims.user_id,
                        request_id,
                        payload.operation_name,
                        payload.vendor,
                        payload.item,
                        payload.units,
                        float(unit_price),
                        float(expected_cost),
                        Json({**(payload.metadata or {}), "step": step}),
                    ),
                )
                created = cur.fetchone()

        return {
            "status": "ok",
            "duplicate": False,
            "usage_event": created,
            "alerts": {"month": month_alert, "request": req_alert},
        }

    except HTTPException:
        raise
    except psycopg2.Error as e:
        raise HTTPException(status_code=500, detail=str(e).split("\n")[0])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
from datetime import datetime, timezone
from uuid import UUID
from fastapi import Body

class BudgetCheckIn(BaseModel):
    vendor: str
    item: str
    units: float = Field(..., ge=0)
    request_id: Optional[str] = None  # must be UUID if present
    operation_name: Optional[str] = None  # for logging only

class BudgetCheckOut(BaseModel):
    status: str
    allowed: bool
    reason: Optional[str]
    tenant_id: str
    request_id: Optional[str]
    expected_cost: float
    unit_price: float
    month_spend: float
    month_cap: Optional[float]
    month_projected: float
    request_spend: Optional[float]
    request_cap: Optional[float]
    request_projected: Optional[float]
    alert_level: str  # ok | warning | critical | blocked


@router.post("/budget/check", response_model=BudgetCheckOut)
def budget_check(payload: BudgetCheckIn = Body(...), claims: Claims = Depends(require_claims)):
    # ---- validate request_id if provided ----
    rid: Optional[str] = None
    if payload.request_id:
        try:
            rid = str(UUID(payload.request_id))
        except Exception:
            raise HTTPException(status_code=400, detail="request_id must be a valid UUID")

    settings = get_settings()

    now = datetime.now(timezone.utc)

    with _supabase_conn(settings) as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            # ---- 1) latest budget for tenant ----
            cur.execute(
                """
                SELECT monthly_cap, per_request_cap
                FROM public.budgets
                WHERE tenant_id = %s
                ORDER BY created_at DESC NULLS LAST
                LIMIT 1
                """,
                (claims.tenant_id,),
            )
            b = cur.fetchone() or {}
            month_cap = b.get("monthly_cap")
            req_cap = b.get("per_request_cap")

            # ---- 2) pick latest price for vendor+item ----
            cur.execute(
                """
                SELECT unit_price
                FROM public.pricebook
                WHERE vendor = %s AND item = %s
                  AND effective_from <= NOW()
                ORDER BY effective_from DESC
                LIMIT 1
                """,
                (payload.vendor, payload.item),
            )
            pr = cur.fetchone()
            if not pr or pr.get("unit_price") is None:
                raise HTTPException(status_code=400, detail="pricebook missing for vendor/item")

            unit_price = float(pr["unit_price"])
            expected_cost = float(payload.units) * unit_price

            # ---- 3) month spend (current month) ----
            cur.execute(
                """
                SELECT COALESCE(SUM(total_cost), 0) AS month_spend
                FROM public.usage_events
                WHERE tenant_id = %s
                  AND created_at >= date_trunc('month', NOW())
                  AND created_at <  (date_trunc('month', NOW()) + interval '1 month')
                """,
                (claims.tenant_id,),
            )
            month_spend = float((cur.fetchone() or {}).get("month_spend", 0) or 0)
            month_projected = month_spend + expected_cost

            # ---- 4) request spend (if request_id provided) ----
            request_spend = None
            request_projected = None
            if rid:
                cur.execute(
                    """
                    SELECT COALESCE(SUM(total_cost), 0) AS request_spend
                    FROM public.usage_events
                    WHERE tenant_id = %s AND request_id = %s
                    """,
                    (claims.tenant_id, rid),
                )
                request_spend = float((cur.fetchone() or {}).get("request_spend", 0) or 0)
                request_projected = request_spend + expected_cost

            # ---- 5) decide allowed + reason ----
            allowed = True
            reason = None

            # per-request cap check first (if applicable)
            if rid and req_cap is not None and request_projected is not None and request_projected > float(req_cap):
                allowed = False
                reason = "per_request_cap_exceeded"

            # monthly cap
            if allowed and month_cap is not None and month_projected > float(month_cap):
                allowed = False
                reason = "monthly_cap_exceeded"

            # ---- 6) alert levels (80/90/100) ----
            def level(pct: Optional[float]) -> str:
                if pct is None:
                    return "ok"
                if pct >= 1.0:
                    return "blocked"
                if pct >= 0.9:
                    return "critical"
                if pct >= 0.8:
                    return "warning"
                return "ok"

            month_pct = (month_projected / float(month_cap)) if month_cap else None
            req_pct = (request_projected / float(req_cap)) if (rid and req_cap and request_projected is not None) else None

            # pick the worst level
            order = {"ok": 0, "warning": 1, "critical": 2, "blocked": 3}
            alert_level = max([level(month_pct), level(req_pct)], key=lambda x: order[x])

            # ---- 7) log rejection path (job_event) if blocked ----
            if not allowed:
                cur.execute(
                    """
                    INSERT INTO public.job_events (tenant_id, request_id, event_type, detail)
                    VALUES (%s, %s::uuid, %s, %s::jsonb)
                    """,
                    (
                        claims.tenant_id,
                        rid,
                        "budget_exceeded",
                        psycopg2.extras.Json(
                            {
                                "reason": reason,
                                "vendor": payload.vendor,
                                "item": payload.item,
                                "units": float(payload.units),
                                "expected_cost": float(expected_cost),
                                "operation_name": payload.operation_name,
                                "month_spend": float(month_spend),
                                "month_cap": float(month_cap) if month_cap is not None else None,
                                "request_spend": float(request_spend) if request_spend is not None else None,
                                "request_cap": float(req_cap) if req_cap is not None else None,
                                "alert_level": alert_level,
                            }
                        ),
                    ),
                )

    return BudgetCheckOut(
        status="ok",
        allowed=allowed,
        reason=reason,
        tenant_id=claims.tenant_id,
        request_id=rid,
        expected_cost=expected_cost,
        unit_price=unit_price,
        month_spend=month_spend,
        month_cap=float(month_cap) if month_cap is not None else None,
        month_projected=month_projected,
        request_spend=request_spend,
        request_cap=float(req_cap) if req_cap is not None else None,
        request_projected=request_projected,
        alert_level=alert_level,
    )


