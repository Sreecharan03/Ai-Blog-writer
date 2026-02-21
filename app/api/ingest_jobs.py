from __future__ import annotations

import json
import os
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from fastapi import APIRouter, Depends, Header, HTTPException, Query
from jose import JWTError, jwt
from pydantic import BaseModel

import psycopg2
from psycopg2.extras import RealDictCursor


# ---------------------------
# Settings loader (same pattern as your ingest.py)
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


# ---------------------------
# JWT Claims (same as your ingest.py)
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
# DB connection (Supabase pooler envs)
# ---------------------------
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


def _coerce_json(val: Any) -> Any:
    if val is None:
        return None
    if isinstance(val, (dict, list)):
        return val
    if isinstance(val, (bytes, bytearray)):
        try:
            return json.loads(val.decode("utf-8"))
        except Exception:
            return {"_raw": val.decode("utf-8", errors="replace")}
    if isinstance(val, str):
        try:
            return json.loads(val)
        except Exception:
            return {"_raw": val}
    return {"_raw": str(val)}


def _parse_dt(v: Any) -> Optional[str]:
    if v is None:
        return None
    if isinstance(v, datetime):
        return v.isoformat()
    return str(v)


# ---------------------------
# Response models
# ---------------------------
class JobEventOut(BaseModel):
    event_type: str
    created_at: Optional[str] = None
    detail: Dict[str, Any] = {}


class JobArtifactsOut(BaseModel):
    doc_id: Optional[str] = None
    kb_id: Optional[str] = None
    filename: Optional[str] = None
    fingerprint: Optional[str] = None
    text_fingerprint: Optional[str] = None
    gcs_raw_uri: Optional[str] = None
    gcs_extracted_uri: Optional[str] = None
    extraction_method: Optional[str] = None


class JobProgressOut(BaseModel):
    current: int
    total: int
    pct: float


class IngestJobStatusOut(BaseModel):
    status: str
    tenant_id: str
    job_id: str
    state: str  # running | done | failed | unknown
    started_at: Optional[str] = None
    updated_at: Optional[str] = None
    progress: JobProgressOut
    artifacts: JobArtifactsOut
    recent_events: List[JobEventOut]


# ---------------------------
# Derivation logic (from job_events)
# ---------------------------
_EXPECTED_CORE_EVENTS = [
    "step_started",
    "raw_uploaded",
    # either cache_hit OR extracted_text_saved OR extraction_skipped
    "doc_row_created",
    "step_done",
]


def _derive_state(events: List[Dict[str, Any]]) -> str:
    if not events:
        return "unknown"

    # any explicit failure marker
    for e in reversed(events):
        et = (e.get("event_type") or "").lower()
        if et in ("failed", "error", "step_failed"):
            return "failed"

    # "done" if we see step_done for ingest_file or ingest_url
    for e in reversed(events):
        if e.get("event_type") == "step_done":
            detail = _coerce_json(e.get("detail")) or {}
            step = (detail.get("step") or "").lower()
            if step in ("ingest_file", "ingest_url"):
                return "done"

    return "running"


def _derive_progress(events: List[Dict[str, Any]]) -> Tuple[int, int, float]:
    if not events:
        return (0, len(_EXPECTED_CORE_EVENTS), 0.0)

    seen = set()
    for e in events:
        et = e.get("event_type")
        if et in ("cache_hit", "extracted_text_saved", "extraction_skipped"):
            seen.add("extraction_phase")
        if et in ("step_started", "raw_uploaded", "doc_row_created", "step_done"):
            seen.add(et)

    total = 5  # step_started + raw_uploaded + extraction_phase + doc_row_created + step_done
    current = 0
    for k in ("step_started", "raw_uploaded", "extraction_phase", "doc_row_created", "step_done"):
        if k in seen:
            current += 1
    pct = round((current / total) * 100.0, 2) if total else 0.0
    return (current, total, pct)


def _derive_artifacts(events: List[Dict[str, Any]]) -> JobArtifactsOut:
    a: Dict[str, Any] = {}
    # walk chronologically so later values override earlier
    for e in events:
        et = e.get("event_type")
        d = _coerce_json(e.get("detail")) or {}

        if et == "step_started":
            a["kb_id"] = a.get("kb_id") or d.get("kb_id")
            a["filename"] = a.get("filename") or d.get("filename")

        if et == "raw_uploaded":
            a["gcs_raw_uri"] = a.get("gcs_raw_uri") or d.get("gcs_raw_uri")

        if et in ("extracted_text_saved", "cache_hit"):
            a["gcs_extracted_uri"] = a.get("gcs_extracted_uri") or d.get("gcs_extracted_uri")

        if et in ("cache_registry_upserted",):
            a["text_fingerprint"] = a.get("text_fingerprint") or d.get("text_fingerprint")

        if et in ("doc_row_created",):
            a["doc_id"] = a.get("doc_id") or d.get("doc_id")
            a["fingerprint"] = a.get("fingerprint") or d.get("fingerprint")

        # extraction method appears in extracted_text_saved or your ingest response; capture when present
        if et in ("extracted_text_saved", "extraction_skipped"):
            a["extraction_method"] = a.get("extraction_method") or d.get("method")

    return JobArtifactsOut(**a)


# ---------------------------
# Router (Day-10 Part-1)
# ---------------------------
router = APIRouter(prefix="/api/v1/ingest", tags=["ingest_jobs"])


@router.get("/jobs/{job_id}", response_model=IngestJobStatusOut)
def get_ingest_job_status(
    job_id: str,
    include_events: bool = Query(True, description="Include recent events preview"),
    events_limit: int = Query(20, ge=1, le=200),
    claims: Claims = Depends(require_claims),
):
    """
    Day-10: Ingestion job status (derived from job_events).
    Endpoint per document:
      GET /api/v1/ingest/jobs/{job_id}
    """
    try:
        _ = uuid.UUID(job_id)
    except Exception:
        raise HTTPException(status_code=400, detail="job_id must be a valid UUID")

    settings = get_settings()
    tenant_id = claims.tenant_id

    with _db_conn(settings) as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT event_id, tenant_id, job_id, event_type, detail, created_at
                FROM public.job_events
                WHERE tenant_id = %s::uuid AND job_id = %s
                ORDER BY created_at ASC
                """,
                (tenant_id, job_id),
            )
            rows = cur.fetchall() or []

    if not rows:
        raise HTTPException(status_code=404, detail="Job not found for this tenant")

    # normalize rows
    events: List[Dict[str, Any]] = []
    for r in rows:
        events.append(
            {
                "event_type": r.get("event_type"),
                "detail": _coerce_json(r.get("detail")) or {},
                "created_at": r.get("created_at"),
            }
        )

    state = _derive_state(events)
    current, total, pct = _derive_progress(events)
    artifacts = _derive_artifacts(events)

    started_at = _parse_dt(events[0].get("created_at"))
    updated_at = _parse_dt(events[-1].get("created_at"))

    recent_events: List[JobEventOut] = []
    if include_events:
        slice_events = events[-events_limit:]
        recent_events = [
            JobEventOut(
                event_type=e.get("event_type") or "unknown",
                created_at=_parse_dt(e.get("created_at")),
                detail=e.get("detail") or {},
            )
            for e in slice_events
        ]

    return IngestJobStatusOut(
        status="ok",
        tenant_id=tenant_id,
        job_id=job_id,
        state=state,
        started_at=started_at,
        updated_at=updated_at,
        progress=JobProgressOut(current=current, total=total, pct=pct),
        artifacts=artifacts,
        recent_events=recent_events,
    )


# ---------------------------
# Real self-test (no mock data)
# ---------------------------
if __name__ == "__main__":
    """
    Real DB check only (no inserts).
    Requires env vars for DB connectivity.
    """
    settings = get_settings()
    with _db_conn(settings) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT 1;")
            print("DB OK:", cur.fetchone()[0])
