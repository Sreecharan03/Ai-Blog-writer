"""
Day 9 (Step A) — Cache Registry (Cloud SQL / Postgres)

Purpose
- Prevent reprocessing when the same content is re-submitted.
- Map fingerprints -> processed artifacts (GCS URIs) + preprocessing_version + indexing state.
- Enforce scope rules:
    * Public URLs -> PUBLIC_GLOBAL reuse across tenants
    * Private uploads -> TENANT_PRIVATE reuse only within tenant
- Cache hit is valid only if:
    * preprocessing_version matches
    * required artifacts exist

This module is DB-agnostic (works with Supabase Postgres or Cloud SQL Postgres)
as long as DB_* env vars (or DB_DSN) are set.

Self-test (REAL):
- Reads the latest document row for TENANT_ID from `public.documents`
- Upserts a cache_registry record using that real doc’s fingerprint + GCS URIs
- Looks it up and validates the hit
"""

from __future__ import annotations

import os
import json
import hashlib
import datetime as dt
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Literal


# Prefer psycopg3, fallback psycopg2
_PSYCOPG_MODE = None
try:
    import psycopg  # type: ignore
    from psycopg.rows import dict_row  # type: ignore

    _PSYCOPG_MODE = "psycopg3"
except Exception:
    psycopg = None  # type: ignore
    dict_row = None  # type: ignore

try:
    import psycopg2  # type: ignore
    import psycopg2.extras  # type: ignore

    if _PSYCOPG_MODE is None:
        _PSYCOPG_MODE = "psycopg2"
except Exception:
    psycopg2 = None  # type: ignore


CacheScope = Literal["PUBLIC_GLOBAL", "TENANT_PRIVATE"]


@dataclass
class CacheRecord:
    cache_id: str
    fingerprint: str
    text_fingerprint: Optional[str]
    scope: CacheScope
    tenant_id: Optional[str]
    source_type: Optional[str]
    canonical_url: Optional[str]
    gcs_raw_uri: Optional[str]
    gcs_extracted_uri: Optional[str]
    extraction_method: Optional[str]
    preprocessing_version: str
    indexing_state: Dict[str, Any]
    meta: Dict[str, Any]
    created_at: str
    updated_at: str


def _sha256_hex(data: bytes) -> str:
    h = hashlib.sha256()
    h.update(data)
    return h.hexdigest()


def normalize_text_for_fingerprint(text: str) -> str:
    """
    Stable normalization to reduce trivial diffs.
    - normalize newlines
    - collapse whitespace runs
    - strip
    NOTE: We do NOT lowercase by default (case can matter for some domains).
    """
    t = text.replace("\r\n", "\n").replace("\r", "\n")
    t = " ".join(t.split())
    return t.strip()


def fingerprint_bytes(data: bytes) -> str:
    return _sha256_hex(data)


def fingerprint_text(text: str) -> str:
    norm = normalize_text_for_fingerprint(text)
    return _sha256_hex(norm.encode("utf-8", errors="ignore"))


def _now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _get_db_dsn() -> str:
    """
    Supports either DB_DSN or individual DB_* env vars.
    Expected env vars commonly used in your project:
      DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD
    """
    dsn = os.getenv("DB_DSN")
    if dsn and dsn.strip():
        return dsn.strip()

    host = os.getenv("DB_HOST")
    port = os.getenv("DB_PORT", "5432")
    name = os.getenv("DB_NAME")
    user = os.getenv("DB_USER")
    pwd = os.getenv("DB_PASSWORD")

    missing = [k for k, v in [("DB_HOST", host), ("DB_NAME", name), ("DB_USER", user), ("DB_PASSWORD", pwd)] if not v]
    if missing:
        raise RuntimeError(f"Missing DB config env vars: {', '.join(missing)} (or set DB_DSN).")

    # psycopg/psycopg2 DSN
    return f"host={host} port={port} dbname={name} user={user} password={pwd}"


def _connect():
    dsn = _get_db_dsn()

    if _PSYCOPG_MODE == "psycopg3":
        assert psycopg is not None
        return psycopg.connect(dsn, row_factory=dict_row)
    if _PSYCOPG_MODE == "psycopg2":
        assert psycopg2 is not None
        conn = psycopg2.connect(dsn)
        return conn

    raise RuntimeError("Neither psycopg (v3) nor psycopg2 is installed. Install one of them.")


class CacheRegistryService:
    """
    Cache registry with scope rules:
      - PUBLIC_GLOBAL: reusable across tenants
      - TENANT_PRIVATE: reusable only within same tenant (but can also fall back to PUBLIC_GLOBAL if allowed)
    """

    def __init__(self, preprocessing_version: str):
        if not preprocessing_version or not preprocessing_version.strip():
            raise ValueError("preprocessing_version is required (e.g. 'v1').")
        self.preprocessing_version = preprocessing_version.strip()

    # --------------------
    # Schema
    # --------------------
    def ensure_schema(self) -> None:
        """
        Creates table + indexes if missing.
        This is safe to run at startup (idempotent).
        """
        ddl = """
        CREATE EXTENSION IF NOT EXISTS pgcrypto;

        CREATE TABLE IF NOT EXISTS public.cache_registry (
            cache_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
            fingerprint text NOT NULL,
            text_fingerprint text NULL,
            scope text NOT NULL CHECK (scope IN ('PUBLIC_GLOBAL','TENANT_PRIVATE')),
            tenant_id uuid NULL,
            source_type text NULL,
            canonical_url text NULL,

            gcs_raw_uri text NULL,
            gcs_extracted_uri text NULL,
            extraction_method text NULL,

            preprocessing_version text NOT NULL,
            indexing_state jsonb NOT NULL DEFAULT '{}'::jsonb,
            meta jsonb NOT NULL DEFAULT '{}'::jsonb,

            created_at timestamptz NOT NULL DEFAULT now(),
            updated_at timestamptz NOT NULL DEFAULT now()
        );

        -- Unique constraints by scope:
        CREATE UNIQUE INDEX IF NOT EXISTS uq_cache_registry_global
            ON public.cache_registry (fingerprint)
            WHERE scope='PUBLIC_GLOBAL';

        CREATE UNIQUE INDEX IF NOT EXISTS uq_cache_registry_tenant
            ON public.cache_registry (tenant_id, fingerprint)
            WHERE scope='TENANT_PRIVATE';

        CREATE INDEX IF NOT EXISTS ix_cache_registry_fingerprint
            ON public.cache_registry (fingerprint);

        CREATE INDEX IF NOT EXISTS ix_cache_registry_tenant_fp
            ON public.cache_registry (tenant_id, fingerprint);

        CREATE INDEX IF NOT EXISTS ix_cache_registry_updated
            ON public.cache_registry (updated_at DESC);
        """
        with _connect() as conn:
            if _PSYCOPG_MODE == "psycopg3":
                with conn.cursor() as cur:
                    cur.execute(ddl)
                conn.commit()
            else:
                cur = conn.cursor()
                cur.execute(ddl)
                conn.commit()
                cur.close()

    # --------------------
    # Upsert
    # --------------------
    def upsert(
        self,
        *,
        fingerprint: str,
        scope: CacheScope,
        tenant_id: Optional[str],
        text_fingerprint: Optional[str] = None,
        source_type: Optional[str] = None,
        canonical_url: Optional[str] = None,
        gcs_raw_uri: Optional[str] = None,
        gcs_extracted_uri: Optional[str] = None,
        extraction_method: Optional[str] = None,
        indexing_state: Optional[Dict[str, Any]] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> CacheRecord:
        """
        Upsert by scope-specific unique index inference:
          - PUBLIC_GLOBAL: ON CONFLICT (fingerprint) WHERE scope='PUBLIC_GLOBAL'
          - TENANT_PRIVATE: ON CONFLICT (tenant_id, fingerprint) WHERE scope='TENANT_PRIVATE'
        """
        if scope == "TENANT_PRIVATE" and not tenant_id:
            raise ValueError("tenant_id is required for TENANT_PRIVATE scope.")

        indexing_state = indexing_state or {}
        meta = meta or {}
        now = _now_iso()

        sql_global = """
        INSERT INTO public.cache_registry
            (fingerprint, text_fingerprint, scope, tenant_id, source_type, canonical_url,
             gcs_raw_uri, gcs_extracted_uri, extraction_method, preprocessing_version,
             indexing_state, meta, updated_at)
        VALUES
            (%(fingerprint)s, %(text_fingerprint)s, 'PUBLIC_GLOBAL', NULL, %(source_type)s, %(canonical_url)s,
             %(gcs_raw_uri)s, %(gcs_extracted_uri)s, %(extraction_method)s, %(preprocessing_version)s,
             %(indexing_state)s::jsonb, %(meta)s::jsonb, now())
        ON CONFLICT (fingerprint) WHERE scope='PUBLIC_GLOBAL'
        DO UPDATE SET
            text_fingerprint = EXCLUDED.text_fingerprint,
            source_type = EXCLUDED.source_type,
            canonical_url = EXCLUDED.canonical_url,
            gcs_raw_uri = EXCLUDED.gcs_raw_uri,
            gcs_extracted_uri = EXCLUDED.gcs_extracted_uri,
            extraction_method = EXCLUDED.extraction_method,
            preprocessing_version = EXCLUDED.preprocessing_version,
            indexing_state = EXCLUDED.indexing_state,
            meta = EXCLUDED.meta,
            updated_at = now()
        RETURNING *;
        """

        sql_tenant = """
        INSERT INTO public.cache_registry
            (fingerprint, text_fingerprint, scope, tenant_id, source_type, canonical_url,
             gcs_raw_uri, gcs_extracted_uri, extraction_method, preprocessing_version,
             indexing_state, meta, updated_at)
        VALUES
            (%(fingerprint)s, %(text_fingerprint)s, 'TENANT_PRIVATE', %(tenant_id)s::uuid, %(source_type)s, %(canonical_url)s,
             %(gcs_raw_uri)s, %(gcs_extracted_uri)s, %(extraction_method)s, %(preprocessing_version)s,
             %(indexing_state)s::jsonb, %(meta)s::jsonb, now())
        ON CONFLICT (tenant_id, fingerprint) WHERE scope='TENANT_PRIVATE'
        DO UPDATE SET
            text_fingerprint = EXCLUDED.text_fingerprint,
            source_type = EXCLUDED.source_type,
            canonical_url = EXCLUDED.canonical_url,
            gcs_raw_uri = EXCLUDED.gcs_raw_uri,
            gcs_extracted_uri = EXCLUDED.gcs_extracted_uri,
            extraction_method = EXCLUDED.extraction_method,
            preprocessing_version = EXCLUDED.preprocessing_version,
            indexing_state = EXCLUDED.indexing_state,
            meta = EXCLUDED.meta,
            updated_at = now()
        RETURNING *;
        """

        params = {
            "fingerprint": fingerprint,
            "text_fingerprint": text_fingerprint,
            "tenant_id": tenant_id,
            "source_type": source_type,
            "canonical_url": canonical_url,
            "gcs_raw_uri": gcs_raw_uri,
            "gcs_extracted_uri": gcs_extracted_uri,
            "extraction_method": extraction_method,
            "preprocessing_version": self.preprocessing_version,
            "indexing_state": json.dumps(indexing_state, ensure_ascii=False),
            "meta": json.dumps(meta, ensure_ascii=False),
            "now": now,
        }

        sql = sql_global if scope == "PUBLIC_GLOBAL" else sql_tenant

        with _connect() as conn:
            row = None
            if _PSYCOPG_MODE == "psycopg3":
                with conn.cursor() as cur:
                    cur.execute(sql, params)
                    row = cur.fetchone()
                conn.commit()
            else:
                cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)  # type: ignore
                cur.execute(sql, params)
                row = cur.fetchone()
                conn.commit()
                cur.close()

        if not row:
            raise RuntimeError("Upsert failed: no row returned.")

        return self._row_to_record(row)

    # --------------------
    # Lookup (cache hit)
    # --------------------
    def lookup(
        self,
        *,
        fingerprint: str,
        tenant_id: Optional[str],
        desired_scope: CacheScope,
        allow_global_fallback: bool = True,
        require_extracted_artifact: bool = True,
    ) -> Optional[CacheRecord]:
        """
        Cache eligibility:
          - If desired_scope=PUBLIC_GLOBAL:
              only return PUBLIC_GLOBAL hits
          - If desired_scope=TENANT_PRIVATE:
              return TENANT_PRIVATE hit for tenant_id if exists
              else if allow_global_fallback: return PUBLIC_GLOBAL hit
        Valid hit only if preprocessing_version matches and artifact requirements are met.
        """
        if desired_scope == "TENANT_PRIVATE" and not tenant_id:
            raise ValueError("tenant_id is required when desired_scope is TENANT_PRIVATE.")

        where_artifact = ""
        if require_extracted_artifact:
            where_artifact = "AND gcs_extracted_uri IS NOT NULL AND length(gcs_extracted_uri) > 0"

        sql_tenant = f"""
        SELECT *
        FROM public.cache_registry
        WHERE scope='TENANT_PRIVATE'
          AND tenant_id = %(tenant_id)s::uuid
          AND fingerprint = %(fingerprint)s
          AND preprocessing_version = %(preprocessing_version)s
          {where_artifact}
        ORDER BY updated_at DESC
        LIMIT 1;
        """

        sql_global = f"""
        SELECT *
        FROM public.cache_registry
        WHERE scope='PUBLIC_GLOBAL'
          AND fingerprint = %(fingerprint)s
          AND preprocessing_version = %(preprocessing_version)s
          {where_artifact}
        ORDER BY updated_at DESC
        LIMIT 1;
        """

        params = {
            "tenant_id": tenant_id,
            "fingerprint": fingerprint,
            "preprocessing_version": self.preprocessing_version,
        }

        def _fetch_one(sql: str) -> Optional[Dict[str, Any]]:
            with _connect() as conn:
                if _PSYCOPG_MODE == "psycopg3":
                    with conn.cursor() as cur:
                        cur.execute(sql, params)
                        r = cur.fetchone()
                    conn.commit()
                    return r
                else:
                    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)  # type: ignore
                    cur.execute(sql, params)
                    r = cur.fetchone()
                    conn.commit()
                    cur.close()
                    return r

        # Tenant-private first if requested
        if desired_scope == "TENANT_PRIVATE":
            r = _fetch_one(sql_tenant)
            if r:
                return self._row_to_record(r)
            if allow_global_fallback:
                r = _fetch_one(sql_global)
                return self._row_to_record(r) if r else None
            return None

        # Public-global requested
        r = _fetch_one(sql_global)
        return self._row_to_record(r) if r else None

    # --------------------
    # Helpers
    # --------------------
    def _row_to_record(self, row: Dict[str, Any]) -> CacheRecord:
        # Normalize JSON fields
        indexing_state = row.get("indexing_state") or {}
        meta = row.get("meta") or {}
        # psycopg2 returns dict; psycopg3 returns dict already due to dict_row
        return CacheRecord(
            cache_id=str(row["cache_id"]),
            fingerprint=row["fingerprint"],
            text_fingerprint=row.get("text_fingerprint"),
            scope=row["scope"],
            tenant_id=str(row["tenant_id"]) if row.get("tenant_id") else None,
            source_type=row.get("source_type"),
            canonical_url=row.get("canonical_url"),
            gcs_raw_uri=row.get("gcs_raw_uri"),
            gcs_extracted_uri=row.get("gcs_extracted_uri"),
            extraction_method=row.get("extraction_method"),
            preprocessing_version=row["preprocessing_version"],
            indexing_state=indexing_state if isinstance(indexing_state, dict) else dict(indexing_state),
            meta=meta if isinstance(meta, dict) else dict(meta),
            created_at=str(row.get("created_at")),
            updated_at=str(row.get("updated_at")),
        )


# --------------------
# REAL Self-test
# --------------------
def _select_latest_document_for_tenant(tenant_id: str) -> Optional[Dict[str, Any]]:
    """
    Reads the latest row from your existing documents table.
    We use this to avoid fake/mock data in self-test.
    Expected columns (based on your /kb/{kb}/docs output):
      fingerprint, text_fingerprint, gcs_raw_uri, gcs_extracted_uri, source_type, source_name
    """
    sql = """
    SELECT
        doc_id,
        tenant_id,
        kb_id,
        fingerprint,
        text_fingerprint,
        gcs_raw_uri,
        gcs_extracted_uri,
        source_type,
        source_name,
        created_at
    FROM public.documents
    WHERE tenant_id = %(tenant_id)s::uuid
    ORDER BY created_at DESC
    LIMIT 1;
    """
    with _connect() as conn:
        if _PSYCOPG_MODE == "psycopg3":
            with conn.cursor() as cur:
                cur.execute(sql, {"tenant_id": tenant_id})
                r = cur.fetchone()
            conn.commit()
            return r
        else:
            cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)  # type: ignore
            cur.execute(sql, {"tenant_id": tenant_id})
            r = cur.fetchone()
            conn.commit()
            cur.close()
            return r


def _main():
    tenant_id = os.getenv("TENANT_ID")
    preprocessing_version = os.getenv("PREPROCESSING_VERSION", "v1")

    if not tenant_id:
        raise RuntimeError("Set TENANT_ID env var for self-test (your tenant UUID).")

    svc = CacheRegistryService(preprocessing_version=preprocessing_version)
    svc.ensure_schema()

    doc = _select_latest_document_for_tenant(tenant_id)
    if not doc:
        raise RuntimeError(
            f"No documents found for tenant_id={tenant_id}. Ingest at least 1 file/url first, then rerun."
        )

    fp = doc["fingerprint"]
    tfp = doc.get("text_fingerprint")
    raw_uri = doc.get("gcs_raw_uri")
    ext_uri = doc.get("gcs_extracted_uri")
    src_type = doc.get("source_type")

    # Decide scope based on source_type (you can refine later in ingest endpoints)
    # For self-test we keep it TENANT_PRIVATE to match your current tenant pipeline.
    scope: CacheScope = "TENANT_PRIVATE"

    rec = svc.upsert(
        fingerprint=fp,
        text_fingerprint=tfp,
        scope=scope,
        tenant_id=tenant_id,
        source_type=src_type,
        canonical_url=None,
        gcs_raw_uri=raw_uri,
        gcs_extracted_uri=ext_uri,
        extraction_method="selftest:from_documents",
        indexing_state={"embedded": False, "chunked": False},
        meta={"selftest_doc_id": str(doc.get("doc_id")), "selftest_kb_id": str(doc.get("kb_id"))},
    )

    hit = svc.lookup(
        fingerprint=fp,
        tenant_id=tenant_id,
        desired_scope="TENANT_PRIVATE",
        allow_global_fallback=True,
        require_extracted_artifact=True,
    )

    if not hit:
        raise RuntimeError("Self-test failed: lookup returned no hit.")

    if hit.fingerprint != fp:
        raise RuntimeError("Self-test failed: fingerprint mismatch.")

    print("✅ CacheRegistry self-test OK")
    print(f"   scope={hit.scope}")
    print(f"   fingerprint={hit.fingerprint[:12]}...")
    print(f"   gcs_extracted_uri={hit.gcs_extracted_uri}")


if __name__ == "__main__":
    _main()
