from __future__ import annotations

import hashlib
import os
import re
import unicodedata
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple

from fastapi import APIRouter, Body, Depends, Header, HTTPException
from fastapi.encoders import jsonable_encoder
from jose import JWTError, jwt
from pydantic import BaseModel, Field

import psycopg2
from psycopg2.extras import Json, RealDictCursor

from google.cloud import storage
from google.api_core.exceptions import PreconditionFailed

try:
    from app.core.config import Config  # type: ignore
except Exception:
    class Config:  # type: ignore
        PREPROCESSING_VERSION = os.getenv("PREPROCESSING_VERSION", "v1")


# -------------------------------------------------------------------
# Settings loader (same pattern as ingest.py)
# -------------------------------------------------------------------
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


# -------------------------------------------------------------------
# JWT Claims (same style as ingest.py)
# -------------------------------------------------------------------
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


# -------------------------------------------------------------------
# DB connection
# -------------------------------------------------------------------
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


def _ensure_job_events_table(conn) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS public.job_events (
              event_id uuid PRIMARY KEY,
              tenant_id uuid NOT NULL,
              request_id uuid NULL,
              job_id uuid NULL,
              event_type text NOT NULL,
              detail jsonb NOT NULL DEFAULT '{}'::jsonb,
              created_at timestamptz NOT NULL DEFAULT now()
            );
            """
        )
        cur.execute("CREATE INDEX IF NOT EXISTS idx_job_events_tenant_created_at ON public.job_events (tenant_id, created_at DESC);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_job_events_job_created_at ON public.job_events (job_id, created_at DESC);")


def _log_job_event(conn, tenant_id: str, event_type: str, detail: Dict[str, Any], job_id: Optional[str] = None) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO public.job_events (event_id, tenant_id, request_id, job_id, event_type, detail)
            VALUES (%s, %s, %s, %s, %s, %s)
            """,
            (
                str(uuid.uuid4()),
                tenant_id,
                None,
                job_id,
                event_type,
                Json(jsonable_encoder(detail)),
            ),
        )


def _ensure_documents_table(conn) -> None:
    # Keep aligned with ingest.py (safe if exists)
    with conn.cursor() as cur:
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS public.documents (
              doc_id uuid PRIMARY KEY,
              tenant_id uuid NOT NULL,
              kb_id uuid NOT NULL,
              fingerprint text NOT NULL,
              scope text NOT NULL DEFAULT 'tenant_private',
              source_type text NOT NULL,
              source_name text,
              mime_type text,
              byte_size bigint,
              gcs_raw_uri text,
              gcs_extracted_uri text,
              text_fingerprint text,
              created_at timestamptz NOT NULL DEFAULT now()
            );
            """
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_documents_tenant_kb_created_at ON public.documents (tenant_id, kb_id, created_at DESC);"
        )


def _ensure_preprocess_tables(conn) -> None:
    with conn.cursor() as cur:
        # status tracking
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS public.preprocess_jobs (
              job_id uuid PRIMARY KEY,
              tenant_id uuid NOT NULL,
              kb_id uuid NOT NULL,
              doc_id uuid NOT NULL,
              state text NOT NULL,
              preprocessing_version text NOT NULL,
              started_at timestamptz NOT NULL,
              updated_at timestamptz NOT NULL,
              progress_current integer NOT NULL DEFAULT 0,
              progress_total integer NOT NULL DEFAULT 0,
              artifacts jsonb NOT NULL DEFAULT '{}'::jsonb,
              error text NULL
            );
            """
        )
        cur.execute("CREATE INDEX IF NOT EXISTS idx_preprocess_jobs_tenant_updated_at ON public.preprocess_jobs (tenant_id, updated_at DESC);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_preprocess_jobs_doc_ver ON public.preprocess_jobs (doc_id, preprocessing_version);")

        # reusable outputs for later chunking/indexing
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS public.preprocess_outputs (
              output_id uuid PRIMARY KEY,
              tenant_id uuid NOT NULL,
              kb_id uuid NOT NULL,
              doc_id uuid NOT NULL,
              preprocessing_version text NOT NULL,
              input_text_fingerprint text NULL,
              clean_fingerprint text NOT NULL,
              gcs_clean_uri text NOT NULL,
              cleaned_chars integer NOT NULL,
              method text NOT NULL,
              meta jsonb NOT NULL DEFAULT '{}'::jsonb,
              created_at timestamptz NOT NULL DEFAULT now(),
              UNIQUE (tenant_id, kb_id, doc_id, preprocessing_version)
            );
            """
        )
        cur.execute("CREATE INDEX IF NOT EXISTS idx_preprocess_outputs_tenant_kb_doc ON public.preprocess_outputs (tenant_id, kb_id, doc_id);")


# -------------------------------------------------------------------
# GCS helpers
# -------------------------------------------------------------------
def _gcs_client(settings: Any) -> storage.Client:
    project_id = _pick(settings, "GCP_PROJECT_ID", "GOOGLE_CLOUD_PROJECT")
    return storage.Client(project=project_id)


def _parse_gs_uri(uri: str) -> Tuple[str, str]:
    if not uri or not uri.startswith("gs://"):
        raise ValueError("Invalid GCS URI (expected gs://bucket/path)")
    rest = uri[5:]
    if "/" not in rest:
        return rest, ""
    bucket, obj = rest.split("/", 1)
    return bucket, obj


def _download_gcs_text(client: storage.Client, gs_uri: str) -> str:
    bucket_name, obj = _parse_gs_uri(gs_uri)
    blob = client.bucket(bucket_name).blob(obj)
    if not blob.exists():
        raise FileNotFoundError(f"GCS object not found: {gs_uri}")
    data = blob.download_as_bytes()
    return data.decode("utf-8", errors="replace")


def _upload_gcs_text(
    client: storage.Client,
    bucket_name: str,
    object_name: str,
    text: str,
) -> str:
    blob = client.bucket(bucket_name).blob(object_name)
    try:
        # Create-only (no overwrite => no delete permission needed)
        blob.upload_from_string(text.encode("utf-8"), content_type="text/plain; charset=utf-8", if_generation_match=0)
    except PreconditionFailed:
        # Object already exists => treat as cache hit / reuse
        pass
    return f"gs://{bucket_name}/{object_name}"


def _norm_prefix(p: str) -> str:
    p = (p or "").strip().replace("\\", "/")
    if not p:
        return ""
    return p if p.endswith("/") else (p + "/")


# -------------------------------------------------------------------
# Cleaning + normalization (Day 11)
# -------------------------------------------------------------------
@dataclass
class CleanStats:
    input_chars: int
    output_chars: int
    lines_in: int
    lines_out: int
    lines_dropped_boilerplate: int
    lines_dropped_noise: int
    lines_deduped: int
    headings_added: int


_BOILERPLATE_PATTERNS = [
    r"^\s*cookie(s)?\s+(policy|preferences)\s*$",
    r"^\s*privacy\s+policy\s*$",
    r"^\s*terms(\s+of\s+use|\s+&\s+conditions|\s+and\s+conditions)?\s*$",
    r"^\s*copyright\s*©?\s*\d{4}.*$",
    r"^\s*all\s+rights\s+reserved\s*$",
    r"^\s*sign\s*(in|up)\s*$",
    r"^\s*log\s*in\s*$",
    r"^\s*register\s*$",
    r"^\s*my\s+account\s*$",
    r"^\s*profile\s*$",
    r"^\s*accept\s+all\s+cookies\s*$",
]

_NOISE_LINE_PATTERNS = [
    r"^\s*$",
    r"^\s*[\|·•\-\—\–_=]{3,}\s*$",  # separators
]


def _sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _normalize_newlines(t: str) -> str:
    t = unicodedata.normalize("NFKC", t or "")
    t = t.replace("\r\n", "\n").replace("\r", "\n")
    return t


def clean_text(
    text: str,
    *,
    remove_boilerplate: bool = True,
    standardize_bullets: bool = True,
    standardize_headings: bool = True,
) -> Tuple[str, CleanStats, Dict[str, Any]]:
    t = _normalize_newlines(text)
    lines = t.split("\n")
    lines_in = len(lines)

    boiler_re = [re.compile(pat, re.IGNORECASE) for pat in _BOILERPLATE_PATTERNS]
    noise_re = [re.compile(pat) for pat in _NOISE_LINE_PATTERNS]

    out = []
    dropped_boiler = 0
    dropped_noise = 0
    deduped = 0
    headings_added = 0

    prev_norm = None

    def is_boiler(line: str) -> bool:
        if not remove_boilerplate:
            return False
        s = line.strip()
        if len(s) > 120:
            return False
        return any(r.match(s) for r in boiler_re)

    def is_noise(line: str) -> bool:
        s = line.strip()
        return any(r.match(s) for r in noise_re)

    bullet_prefixes = ("•", "·", "●", "◦", "▪", "–", "—", "‣", "⁃")

    for line in lines:
        raw = line
        s = raw.strip()

        if is_noise(raw):
            dropped_noise += 1
            continue

        if is_boiler(raw):
            dropped_boiler += 1
            continue

        # collapse internal whitespace but keep indentation minimal
        s = re.sub(r"[ \t]+", " ", s).strip()

        # bullet normalization
        if standardize_bullets and s.startswith(bullet_prefixes):
            s = "- " + s.lstrip("".join(bullet_prefixes)).strip()

        # normalize numbered list "1)" -> "1."
        s = re.sub(r"^(\d+)\)\s+", r"\1. ", s)

        # dedupe consecutive duplicate-ish lines
        norm = s.lower()
        if prev_norm == norm:
            deduped += 1
            continue
        prev_norm = norm

        out.append(s)

    # add heading markers (very conservative)
    if standardize_headings:
        refined = []
        for line in out:
            is_candidate = (
                3 <= len(line) <= 80
                and not line.startswith(("-", "*", "•"))
                and re.match(r"^[A-Za-z0-9][A-Za-z0-9 \-–—:()&/]+$", line) is not None
                and (line.isupper() or line.endswith(":") or re.match(r"^(section|chapter|part)\b", line, re.I))
            )
            if is_candidate and not line.startswith("#"):
                refined.append("## " + line.rstrip(":").strip())
                headings_added += 1
            else:
                refined.append(line)
        out = refined

    joined = "\n".join(out)
    joined = re.sub(r"\n{3,}", "\n\n", joined).strip()

    stats = CleanStats(
        input_chars=len(text or ""),
        output_chars=len(joined),
        lines_in=lines_in,
        lines_out=len(joined.split("\n")) if joined else 0,
        lines_dropped_boilerplate=dropped_boiler,
        lines_dropped_noise=dropped_noise,
        lines_deduped=deduped,
        headings_added=headings_added,
    )

    meta = {
        "remove_boilerplate": remove_boilerplate,
        "standardize_bullets": standardize_bullets,
        "standardize_headings": standardize_headings,
    }
    return joined, stats, meta


# -------------------------------------------------------------------
# API
# -------------------------------------------------------------------
PREPROCESSING_VERSION = getattr(Config, "PREPROCESSING_VERSION", os.getenv("PREPROCESSING_VERSION", "v1"))

router = APIRouter(prefix="/api/v1", tags=["preprocess"])


class PreprocessRequest(BaseModel):
    preprocessing_version: Optional[str] = None
    remove_boilerplate: bool = True
    standardize_bullets: bool = True
    standardize_headings: bool = True


class PreprocessResponse(BaseModel):
    status: str
    kb_id: str
    doc_id: str
    preprocess_job_id: str
    preprocessing_version: str
    input_text_fingerprint: Optional[str] = None
    clean_fingerprint: str
    gcs_clean_uri: str
    cleaned_chars: int
    method: str
    stats: Dict[str, Any] = Field(default_factory=dict)
    meta: Dict[str, Any] = Field(default_factory=dict)


@router.post("/kb/{kb_id}/preprocess/{doc_id}", response_model=PreprocessResponse)
def preprocess_doc(
    kb_id: str,
    doc_id: str,
    req: PreprocessRequest = Body(default=PreprocessRequest()),
    claims: Claims = Depends(require_claims),
):
    """
    Day 11: Stage-2 preprocessing (text cleaning + normalization).
    Input: documents.gcs_extracted_uri
    Output: GCS processed/<tenant>/<kb>/<doc>/clean_<ver>/<clean_fingerprint>.txt
    Tracking: preprocess_jobs + job_events
    """
    try:
        _ = uuid.UUID(kb_id)
        _ = uuid.UUID(doc_id)
    except Exception:
        raise HTTPException(status_code=400, detail="kb_id and doc_id must be valid UUIDs")

    settings = get_settings()
    tenant_id = claims.tenant_id
    ver = (req.preprocessing_version or PREPROCESSING_VERSION or "v1").strip()

    bucket_name = _pick(settings, "GCS_BUCKET_NAME")
    if not bucket_name:
        raise HTTPException(status_code=500, detail="Missing GCS_BUCKET_NAME")
    processed_prefix = _norm_prefix(_pick(settings, "GCS_PROCESSED_PREFIX", default="processed"))

    client = _gcs_client(settings)
    preprocess_job_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc)

    total_steps = 5
    current_step = 0

    with _db_conn(settings) as conn:
        _ensure_job_events_table(conn)
        _ensure_documents_table(conn)
        _ensure_preprocess_tables(conn)

        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO public.preprocess_jobs
                  (job_id, tenant_id, kb_id, doc_id, state, preprocessing_version, started_at, updated_at,
                   progress_current, progress_total, artifacts, error)
                VALUES
                  (%s, %s::uuid, %s::uuid, %s::uuid, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    preprocess_job_id,
                    tenant_id,
                    kb_id,
                    doc_id,
                    "running",
                    ver,
                    now,
                    now,
                    0,
                    total_steps,
                    Json({}),
                    None,
                ),
            )

        def bump(step_name: str, detail: Dict[str, Any]):
            nonlocal current_step
            current_step += 1
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE public.preprocess_jobs
                    SET progress_current=%s, updated_at=%s
                    WHERE job_id=%s
                    """,
                    (current_step, datetime.now(timezone.utc), preprocess_job_id),
                )
            _log_job_event(conn, tenant_id, step_name, detail, job_id=preprocess_job_id)

        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT doc_id, tenant_id, kb_id, gcs_extracted_uri, text_fingerprint, source_name
                FROM public.documents
                WHERE tenant_id=%s::uuid AND kb_id=%s::uuid AND doc_id=%s::uuid
                """,
                (tenant_id, kb_id, doc_id),
            )
            row = cur.fetchone()

        if not row:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE public.preprocess_jobs SET state=%s, error=%s, updated_at=%s WHERE job_id=%s",
                    ("failed", "document_not_found", datetime.now(timezone.utc), preprocess_job_id),
                )
            raise HTTPException(status_code=404, detail="Document not found for this tenant/kb")

        gcs_extracted_uri = (row.get("gcs_extracted_uri") or "").strip()
        input_text_fingerprint = row.get("text_fingerprint")
        source_name = row.get("source_name")

        _log_job_event(conn, tenant_id, "step_started", {"step": "preprocess", "kb_id": kb_id, "doc_id": doc_id, "version": ver}, job_id=preprocess_job_id)

        if not gcs_extracted_uri:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE public.preprocess_jobs SET state=%s, error=%s, updated_at=%s WHERE job_id=%s",
                    ("failed", "missing_gcs_extracted_uri", datetime.now(timezone.utc), preprocess_job_id),
                )
            raise HTTPException(status_code=409, detail="Document has no extracted text (gcs_extracted_uri is null)")

        try:
            extracted_text = _download_gcs_text(client, gcs_extracted_uri)
        except Exception as e:
            err = f"download_failed: {type(e).__name__}: {str(e)[:240]}"
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE public.preprocess_jobs SET state=%s, error=%s, updated_at=%s WHERE job_id=%s",
                    ("failed", err, datetime.now(timezone.utc), preprocess_job_id),
                )
            raise HTTPException(status_code=502, detail=f"Failed to read extracted text from GCS: {gcs_extracted_uri}")

        bump("preprocess_input_loaded", {"gcs_extracted_uri": gcs_extracted_uri, "chars": len(extracted_text)})

        cleaned, stats, clean_meta = clean_text(
            extracted_text,
            remove_boilerplate=req.remove_boilerplate,
            standardize_bullets=req.standardize_bullets,
            standardize_headings=req.standardize_headings,
        )
        if not cleaned.strip():
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE public.preprocess_jobs SET state=%s, error=%s, updated_at=%s WHERE job_id=%s",
                    ("failed", "cleaned_text_empty", datetime.now(timezone.utc), preprocess_job_id),
                )
            raise HTTPException(status_code=422, detail="Preprocess produced empty text (check extraction quality)")

        bump("text_cleaned", {"input_chars": stats.input_chars, "output_chars": stats.output_chars})

        clean_fingerprint = _sha256_text(cleaned)
        clean_object = f"{processed_prefix}{tenant_id}/{kb_id}/{doc_id}/clean_{ver}/{clean_fingerprint}.txt"
        gcs_clean_uri = _upload_gcs_text(client, bucket_name, clean_object, cleaned)
        bump("clean_text_saved", {"gcs_clean_uri": gcs_clean_uri, "clean_fingerprint": clean_fingerprint, "chars": stats.output_chars})

        out_meta = {
            "source_name": source_name,
            "stats": jsonable_encoder(stats.__dict__),
            **clean_meta,
        }
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO public.preprocess_outputs
                  (output_id, tenant_id, kb_id, doc_id, preprocessing_version, input_text_fingerprint,
                   clean_fingerprint, gcs_clean_uri, cleaned_chars, method, meta)
                VALUES
                  (%s, %s::uuid, %s::uuid, %s::uuid, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (tenant_id, kb_id, doc_id, preprocessing_version)
                DO UPDATE SET
                  input_text_fingerprint=EXCLUDED.input_text_fingerprint,
                  clean_fingerprint=EXCLUDED.clean_fingerprint,
                  gcs_clean_uri=EXCLUDED.gcs_clean_uri,
                  cleaned_chars=EXCLUDED.cleaned_chars,
                  method=EXCLUDED.method,
                  meta=EXCLUDED.meta,
                  created_at=now()
                """,
                (
                    str(uuid.uuid4()),
                    tenant_id,
                    kb_id,
                    doc_id,
                    ver,
                    input_text_fingerprint,
                    clean_fingerprint,
                    gcs_clean_uri,
                    int(stats.output_chars),
                    "clean:v1",
                    Json(out_meta),
                ),
            )

        bump("preprocess_output_upserted", {"doc_id": doc_id, "version": ver})

        artifacts = {
            "kb_id": kb_id,
            "doc_id": doc_id,
            "version": ver,
            "input_text_fingerprint": input_text_fingerprint,
            "clean_fingerprint": clean_fingerprint,
            "gcs_clean_uri": gcs_clean_uri,
            "cleaned_chars": int(stats.output_chars),
        }
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE public.preprocess_jobs
                SET state=%s, updated_at=%s, progress_current=%s, progress_total=%s, artifacts=%s, error=NULL
                WHERE job_id=%s
                """,
                ("done", datetime.now(timezone.utc), total_steps, total_steps, Json(artifacts), preprocess_job_id),
            )

        _log_job_event(conn, tenant_id, "step_done", {"step": "preprocess", **artifacts}, job_id=preprocess_job_id)

    return PreprocessResponse(
        status="ok",
        kb_id=kb_id,
        doc_id=doc_id,
        preprocess_job_id=preprocess_job_id,
        preprocessing_version=ver,
        input_text_fingerprint=input_text_fingerprint,
        clean_fingerprint=clean_fingerprint,
        gcs_clean_uri=gcs_clean_uri,
        cleaned_chars=int(stats.output_chars),
        method="clean:v1",
        stats=jsonable_encoder(stats.__dict__),
        meta=clean_meta,
    )


class PreprocessJobResponse(BaseModel):
    status: str
    tenant_id: str
    job_id: str
    state: str
    started_at: str
    updated_at: str
    progress: Dict[str, Any]
    artifacts: Dict[str, Any]
    recent_events: list[Dict[str, Any]] = Field(default_factory=list)


@router.get("/preprocess/jobs/{job_id}", response_model=PreprocessJobResponse)
def get_preprocess_job(
    job_id: str,
    include_events: bool = False,
    events_limit: int = 20,
    claims: Claims = Depends(require_claims),
):
    settings = get_settings()
    tenant_id = claims.tenant_id

    try:
        _ = uuid.UUID(job_id)
    except Exception:
        raise HTTPException(status_code=400, detail="job_id must be a valid UUID")

    with _db_conn(settings) as conn:
        _ensure_job_events_table(conn)
        _ensure_preprocess_tables(conn)

        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT job_id, tenant_id, kb_id, doc_id, state, preprocessing_version,
                       started_at, updated_at, progress_current, progress_total, artifacts, error
                FROM public.preprocess_jobs
                WHERE tenant_id=%s::uuid AND job_id=%s::uuid
                """,
                (tenant_id, job_id),
            )
            row = cur.fetchone()

        if not row:
            raise HTTPException(status_code=404, detail="Job not found")

        events: list[Dict[str, Any]] = []
        if include_events:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT event_type, created_at, detail
                    FROM public.job_events
                    WHERE tenant_id=%s::uuid AND job_id=%s::uuid
                    ORDER BY created_at DESC
                    LIMIT %s
                    """,
                    (tenant_id, job_id, int(max(1, min(events_limit, 200)))),
                )
                events = cur.fetchall() or []

        progress_total = int(row.get("progress_total") or 0)
        progress_current = int(row.get("progress_current") or 0)
        pct = (float(progress_current) / float(progress_total) * 100.0) if progress_total else 0.0

        return PreprocessJobResponse(
            status="ok",
            tenant_id=tenant_id,
            job_id=job_id,
            state=row.get("state") or "unknown",
            started_at=str(row.get("started_at")),
            updated_at=str(row.get("updated_at")),
            progress={"current": progress_current, "total": progress_total, "pct": pct},
            artifacts=row.get("artifacts") or {},
            recent_events=events,
        )
