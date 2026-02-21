from __future__ import annotations

import uuid
from typing import Optional
import psycopg2.extras

from fastapi import APIRouter
from pydantic import BaseModel

from app.core.config import get_settings

router = APIRouter(prefix="/db", tags=["db"])


class TenantCreateIn(BaseModel):
    name: Optional[str] = None


@router.get("/health")
def db_health():
    """
    REAL DB health check (Supabase Postgres):
    - connects using DB_* env
    - runs SELECT 1
    """
    s = get_settings()

    try:
        import psycopg2

        dsn = (
            f"host={s.db_host} port={s.db_port} dbname={s.db_name} "
            f"user={s.db_user} password={s.db_password} sslmode={s.db_sslmode}"
        )

        conn = psycopg2.connect(dsn, connect_timeout=5)
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT 1;")
                val = cur.fetchone()[0]
        finally:
            conn.close()

        return {"status": "ok", "db": "supabase-postgres", "select_1": val}

    except Exception as e:
        return {"status": "not_ok", "db": "supabase-postgres", "error": str(e)}


@router.get("/info")
def db_info():
    """
    REAL DB info (Supabase Postgres):
    - returns server version, current database, current user
    """
    s = get_settings()

    try:
        import psycopg2

        dsn = (
            f"host={s.db_host} port={s.db_port} dbname={s.db_name} "
            f"user={s.db_user} password={s.db_password} sslmode={s.db_sslmode}"
        )

        conn = psycopg2.connect(dsn, connect_timeout=5)
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT version();")
                version = cur.fetchone()[0]

                cur.execute("SELECT current_database();")
                dbname = cur.fetchone()[0]

                cur.execute("SELECT current_user;")
                user = cur.fetchone()[0]
        finally:
            conn.close()

        return {
            "status": "ok",
            "db": "supabase-postgres",
            "current_database": dbname,
            "current_user": user,
            "version": version,
        }

    except Exception as e:
        return {"status": "not_ok", "db": "supabase-postgres", "error": str(e)}


@router.post("/tenants")
def create_tenant(payload: TenantCreateIn):
    """
    REAL insert into tenants_fin (required before job_events inserts).
    """
    s = get_settings()

    try:
        import psycopg2

        dsn = (
            f"host={s.db_host} port={s.db_port} dbname={s.db_name} "
            f"user={s.db_user} password={s.db_password} sslmode={s.db_sslmode}"
        )

        tenant_id = uuid.uuid4()

        conn = psycopg2.connect(dsn, connect_timeout=5)
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO public.tenants_fin (tenant_id, name)
                    VALUES (%s, %s)
                    RETURNING tenant_id, name, created_at;
                    """,
                    (str(tenant_id), payload.name),
                )
                row = cur.fetchone()
                conn.commit()
        finally:
            conn.close()

        return {
            "status": "ok",
            "tenant": {
                "tenant_id": str(row[0]),
                "name": row[1],
                "created_at": row[2].isoformat() if row[2] else None,
            },
        }

    except Exception as e:
        return {"status": "not_ok", "error": str(e)}


class JobEventCreateIn(BaseModel):
    tenant_id: str
    request_id: Optional[str] = None
    job_id: Optional[str] = None
    event_type: str
    detail: dict = {}


@router.post("/job-events")
def create_job_event(payload: JobEventCreateIn):
    """
    REAL insert into public.job_events (logs table).
    """
    s = get_settings()

    try:
        import psycopg2

        dsn = (
            f"host={s.db_host} port={s.db_port} dbname={s.db_name} "
            f"user={s.db_user} password={s.db_password} sslmode={s.db_sslmode}"
        )

        conn = psycopg2.connect(dsn, connect_timeout=5)
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO public.job_events (tenant_id, request_id, job_id, event_type, detail)
                    VALUES (%s, %s, %s, %s, %s::jsonb)
                    RETURNING event_id, tenant_id, request_id, job_id, event_type, detail, created_at;
                    """,
                    (
                        payload.tenant_id,
                        payload.request_id,
                        payload.job_id,
                        payload.event_type,
                        psycopg2.extras.Json(payload.detail),
                    ),
                )
                row = cur.fetchone()
                conn.commit()
        finally:
            conn.close()

        return {
            "status": "ok",
            "job_event": {
                "event_id": str(row[0]),
                "tenant_id": str(row[1]),
                "request_id": str(row[2]) if row[2] else None,
                "job_id": str(row[3]) if row[3] else None,
                "event_type": row[4],
                "detail": row[5],
                "created_at": row[6].isoformat() if row[6] else None,
            },
        }

    except Exception as e:
        return {"status": "not_ok", "error": str(e)}

@router.get("/job-events/latest")
def latest_job_events(tenant_id: str, limit: int = 20):
    """
    REAL select: fetch latest job_events for a tenant (streaming UI support).
    """
    s = get_settings()

    try:
        import psycopg2

        dsn = (
            f"host={s.db_host} port={s.db_port} dbname={s.db_name} "
            f"user={s.db_user} password={s.db_password} sslmode={s.db_sslmode}"
        )

        conn = psycopg2.connect(dsn, connect_timeout=5)
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT event_id, tenant_id, request_id, job_id, event_type, detail, created_at
                    FROM public.job_events
                    WHERE tenant_id = %s
                    ORDER BY created_at DESC
                    LIMIT %s;
                    """,
                    (tenant_id, limit),
                )
                rows = cur.fetchall()
        finally:
            conn.close()

        events = []
        for r in rows:
            events.append(
                {
                    "event_id": str(r[0]),
                    "tenant_id": str(r[1]),
                    "request_id": str(r[2]) if r[2] else None,
                    "job_id": str(r[3]) if r[3] else None,
                    "event_type": r[4],
                    "detail": r[5],
                    "created_at": r[6].isoformat() if r[6] else None,
                }
            )

        return {"status": "ok", "count": len(events), "events": events}

    except Exception as e:
        return {"status": "not_ok", "error": str(e)}
