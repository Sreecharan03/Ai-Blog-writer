"""
Brand Config API — Layer B endpoints.

GET  /api/v1/config   → return current tenant brand config (or defaults)
PUT  /api/v1/config   → save/update tenant brand config

The config is stored in public.tenant_brand_configs (created on first use).
Every pipeline run reads from this table via prompt_engine.load_brand_context().
"""
from __future__ import annotations

import os
from typing import Any, List, Optional

import psycopg2
import psycopg2.extras
from fastapi import APIRouter, Depends, Header, HTTPException
from jose import JWTError, jwt
from pydantic import BaseModel, Field

router = APIRouter(prefix="/api/v1", tags=["brand-config"])

# ── settings ─────────────────────────────────────────────────────────────────
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


# ── auth ──────────────────────────────────────────────────────────────────────
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


# ── DB ────────────────────────────────────────────────────────────────────────
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
        host=host, port=port, dbname=dbname, user=user,
        password=password, sslmode=sslmode, connect_timeout=8,
    )


# ── Pydantic schemas ──────────────────────────────────────────────────────────
class BrandConfigIn(BaseModel):
    persona: str = Field(
        default="",
        description="One-sentence writer persona. E.g. 'A knowledgeable but approachable health writer.'",
        max_length=500,
    )
    tone_adjectives: List[str] = Field(
        default_factory=list,
        description="3–6 adjectives that define tone. E.g. ['clear', 'evidence-aware', 'non-preachy']",
        max_items=10,
    )
    forbidden_phrases: List[str] = Field(
        default_factory=list,
        description="Brand-specific phrases the model must never output.",
        max_items=50,
    )
    audience_primary: str = Field(
        default="",
        description="Primary reader segment. E.g. 'health-conscious adults aged 28-45'",
        max_length=300,
    )
    audience_pain_points: List[str] = Field(
        default_factory=list,
        description="Top reader problems this content addresses. E.g. ['poor sleep quality', 'low energy']",
        max_items=10,
    )
    reading_level: str = Field(
        default="grade 10-12",
        description="Target reading level. E.g. 'grade 10-12' or 'undergraduate'",
        max_length=50,
    )
    language: str = Field(
        default="en",
        description="BCP-47 language tag. E.g. 'en-IN', 'en-US'",
        max_length=10,
    )
    preferred_pov: str = Field(
        default="second-person",
        description="Default point of view.",
    )
    compliance_note: str = Field(
        default="",
        description="Sector compliance note appended to every article. E.g. 'Always include a consult-a-professional nudge.'",
        max_length=500,
    )


class BrandConfigOut(BrandConfigIn):
    is_configured: bool
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


# ── Routes ────────────────────────────────────────────────────────────────────
@router.get("/config", response_model=BrandConfigOut, summary="Get brand config for your tenant")
def get_brand_config(claims: Claims = Depends(require_claims)):
    from app.services.blog_pipeline.prompt_engine import load_brand_context, _ensure_table

    settings = get_settings()
    conn = _db_conn(settings)
    try:
        _ensure_table(conn)
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                "SELECT * FROM public.tenant_brand_configs WHERE tenant_id = %s::uuid",
                (claims.tenant_id,),
            )
            row = cur.fetchone()
    finally:
        conn.close()

    if not row:
        ctx = load_brand_context.__wrapped__ if hasattr(load_brand_context, "__wrapped__") else None
        return BrandConfigOut(
            is_configured=False,
            persona="",
            tone_adjectives=[],
            forbidden_phrases=[],
            audience_primary="",
            audience_pain_points=[],
            reading_level="grade 10-12",
            language="en",
            preferred_pov="second-person",
            compliance_note="",
        )

    return BrandConfigOut(
        is_configured=bool(row.get("audience_primary")),
        persona=row.get("persona") or "",
        tone_adjectives=list(row.get("tone_adjectives") or []),
        forbidden_phrases=list(row.get("forbidden_phrases") or []),
        audience_primary=row.get("audience_primary") or "",
        audience_pain_points=list(row.get("audience_pain_points") or []),
        reading_level=row.get("reading_level") or "grade 10-12",
        language=row.get("language") or "en",
        preferred_pov=row.get("preferred_pov") or "second-person",
        compliance_note=row.get("compliance_note") or "",
        created_at=str(row["created_at"]) if row.get("created_at") else None,
        updated_at=str(row["updated_at"]) if row.get("updated_at") else None,
    )


@router.put("/config", response_model=BrandConfigOut, summary="Save or update brand config for your tenant")
def put_brand_config(
    body: BrandConfigIn,
    claims: Claims = Depends(require_claims),
):
    from app.services.blog_pipeline.prompt_engine import (
        BrandContext,
        upsert_brand_context,
    )

    ctx = BrandContext(
        persona=body.persona,
        tone_adjectives=body.tone_adjectives,
        forbidden_phrases=body.forbidden_phrases,
        audience_primary=body.audience_primary,
        audience_pain_points=body.audience_pain_points,
        reading_level=body.reading_level,
        language=body.language,
        preferred_pov=body.preferred_pov,
        compliance_note=body.compliance_note,
    )

    settings = get_settings()
    conn = _db_conn(settings)
    try:
        upsert_brand_context(conn, claims.tenant_id, ctx)
    except Exception as exc:
        conn.close()
        raise HTTPException(status_code=500, detail=f"Failed to save config: {exc}")
    finally:
        conn.close()

    return BrandConfigOut(
        is_configured=ctx.is_configured(),
        **body.model_dump(),
    )
