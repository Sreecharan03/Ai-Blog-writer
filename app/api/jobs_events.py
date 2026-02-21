from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import anyio
import psycopg2
from fastapi import APIRouter, Depends, Header, HTTPException, Query
from fastapi.encoders import jsonable_encoder
from fastapi.responses import StreamingResponse
from jose import JWTError, jwt
from psycopg2.extras import RealDictCursor
from pydantic import BaseModel


# ---------------------------
# Settings loader (same pattern)
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
# JWT Claims
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


def _coerce_json(val: Any) -> Dict[str, Any]:
    if val is None:
        return {}
    if isinstance(val, dict):
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


def _dt_iso(v: Any) -> Optional[str]:
    if v is None:
        return None
    if isinstance(v, datetime):
        return v.isoformat()
    return str(v)


# ---------------------------
# SSE helpers
# ---------------------------
def _sse(event: str, data_obj: Any, event_id: Optional[str] = None) -> str:
    payload = json.dumps(jsonable_encoder(data_obj), ensure_ascii=False)
    lines = []
    if event_id:
        lines.append(f"id: {event_id}")
    lines.append(f"event: {event}")
    lines.append(f"data: {payload}")
    return "\n".join(lines) + "\n\n"


def _sse_comment(comment: str) -> str:
    # comment line keeps connection alive in many proxies
    return f": {comment}\n\n"


# ---------------------------
# Fetch events (sync; will be called via anyio.to_thread)
# Cursor uses tuple (created_at, event_id) to prevent duplicates.
# ---------------------------
def _fetch_events_sync(
    settings: Any,
    tenant_id: str,
    job_id: Optional[str],
    after_created_at: datetime,
    after_event_id: uuid.UUID,
    limit: int,
) -> List[Dict[str, Any]]:
    with _db_conn(settings) as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            if job_id:
                cur.execute(
                    """
                    SELECT event_id, tenant_id, request_id, job_id, event_type, detail, created_at
                    FROM public.job_events
                    WHERE tenant_id = %s::uuid
                      AND job_id = %s
                      AND (created_at, event_id) > (%s, %s::uuid)
                    ORDER BY created_at ASC, event_id ASC
                    LIMIT %s
                    """,
                    (tenant_id, job_id, after_created_at, str(after_event_id), limit),
                )
            else:
                cur.execute(
                    """
                    SELECT event_id, tenant_id, request_id, job_id, event_type, detail, created_at
                    FROM public.job_events
                    WHERE tenant_id = %s::uuid
                      AND (created_at, event_id) > (%s, %s::uuid)
                    ORDER BY created_at ASC, event_id ASC
                    LIMIT %s
                    """,
                    (tenant_id, after_created_at, str(after_event_id), limit),
                )

            rows = cur.fetchall() or []

    # normalize payload
    out: List[Dict[str, Any]] = []
    for r in rows:
        out.append(
            {
                "event_id": str(r.get("event_id")),
                "tenant_id": str(r.get("tenant_id")),
                "request_id": r.get("request_id"),
                "job_id": r.get("job_id"),
                "event_type": r.get("event_type"),
                "detail": _coerce_json(r.get("detail")),
                "created_at": _dt_iso(r.get("created_at")),
                "_created_at_dt": r.get("created_at"),  # for cursor update
            }
        )
    return out


# ---------------------------
# Router
# ---------------------------
router = APIRouter(prefix="/api/v1/jobs", tags=["jobs"])


@router.get("/events")
async def stream_job_events(
    job_id: Optional[str] = Query(None, description="Optional job_id to filter stream"),
    include_past: bool = Query(False, description="If true, stream from beginning for this tenant/job."),
    poll_interval_seconds: float = Query(1.0, ge=0.2, le=10.0),
    heartbeat_seconds: float = Query(15.0, ge=5.0, le=120.0),
    limit_per_poll: int = Query(50, ge=1, le=200),
    tenant_id: Optional[str] = Query(None, description="system_admin only override"),
    claims: Claims = Depends(require_claims),
    last_event_id: Optional[str] = Header(None, alias="Last-Event-ID"),
):
    """
    Day-10 Part-2: SSE stream for job_events.
    GET /api/v1/jobs/events
    """
    # validate job_id if provided
    if job_id:
        try:
            _ = uuid.UUID(job_id)
        except Exception:
            raise HTTPException(status_code=400, detail="job_id must be a valid UUID")

    # tenant scoping
    effective_tenant = claims.tenant_id
    if tenant_id:
        if claims.role != "system_admin":
            raise HTTPException(status_code=403, detail="tenant_id override requires system_admin")
        try:
            _ = uuid.UUID(tenant_id)
        except Exception:
            raise HTTPException(status_code=400, detail="tenant_id must be a valid UUID")
        effective_tenant = tenant_id

    settings = get_settings()

    # cursor init
    zero_uuid = uuid.UUID("00000000-0000-0000-0000-000000000000")

    if include_past:
        cursor_created_at = datetime(1970, 1, 1, tzinfo=timezone.utc)
        cursor_event_id = zero_uuid
    else:
        cursor_created_at = datetime.now(timezone.utc)
        cursor_event_id = zero_uuid

    # allow client to resume using Last-Event-ID (best-effort: we resume *after* that event_id at current cursor time)
    # Note: UUID alone isn’t enough without created_at; we keep created_at cursor and use UUID as tie-breaker.
    # If you want perfect resume, store last (created_at,event_id) on the client; for now this is safe + simple.
    if last_event_id:
        try:
            _ = uuid.UUID(last_event_id)
            cursor_event_id = uuid.UUID(last_event_id)
        except Exception:
            pass

    async def event_gen():
        nonlocal cursor_created_at, cursor_event_id

        # initial hello
        yield "retry: 15000\n\n"
        yield _sse_comment("connected")

        last_heartbeat = datetime.now(timezone.utc)

        try:
            while True:
                # heartbeat
                now = datetime.now(timezone.utc)
                if (now - last_heartbeat).total_seconds() >= heartbeat_seconds:
                    yield _sse_comment("heartbeat")
                    last_heartbeat = now

                # fetch new rows (thread offload)
                try:
                    rows = await anyio.to_thread.run_sync(
                        _fetch_events_sync,
                        settings,
                        effective_tenant,
                        job_id,
                        cursor_created_at,
                        cursor_event_id,
                        limit_per_poll,
                    )
                except Exception as e:
                    # emit an error event but keep stream alive (client can choose to reconnect)
                    yield _sse(
                        event="error",
                        event_id=None,
                        data_obj={
                            "status": "not_ok",
                            "message": "db_fetch_failed",
                            "detail": str(e).split("\n")[0],
                        },
                    )
                    await anyio.sleep(min(5.0, poll_interval_seconds * 2))
                    continue

                if rows:
                    for r in rows:
                        ev_id = r.get("event_id")
                        ev_type = r.get("event_type") or "job_event"
                        created_at_dt = r.get("_created_at_dt")

                        # stream event
                        payload = {
                            "event_id": ev_id,
                            "tenant_id": r.get("tenant_id"),
                            "request_id": r.get("request_id"),
                            "job_id": r.get("job_id"),
                            "event_type": ev_type,
                            "detail": r.get("detail") or {},
                            "created_at": r.get("created_at"),
                        }
                        yield _sse(event="job_event", event_id=ev_id, data_obj=payload)

                        # update cursor
                        if isinstance(created_at_dt, datetime):
                            cursor_created_at = created_at_dt
                        if ev_id:
                            try:
                                cursor_event_id = uuid.UUID(ev_id)
                            except Exception:
                                cursor_event_id = cursor_event_id

                await anyio.sleep(poll_interval_seconds)

        except (anyio.get_cancelled_exc_class(), GeneratorExit):
            # client disconnected
            return

    headers = {
        "Cache-Control": "no-cache, no-transform",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",  # important for nginx
    }

    return StreamingResponse(event_gen(), media_type="text/event-stream", headers=headers)


# ---------------------------
# Real self-test (no inserts)
# ---------------------------
if __name__ == "__main__":
    settings = get_settings()
    with _db_conn(settings) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT 1;")
            print("DB OK:", cur.fetchone()[0])
