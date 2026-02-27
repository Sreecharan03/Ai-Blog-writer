from __future__ import annotations

import hashlib
import json
import os
import re
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional, Tuple

import psycopg2
from fastapi import APIRouter, Depends, Header, HTTPException, Query
from jose import JWTError, jwt
from pydantic import BaseModel

from google.cloud import storage
from google.api_core.exceptions import PreconditionFailed


router = APIRouter(prefix="/api/v1/articles", tags=["article-qc"])


# -----------------------------
# Settings loader
# -----------------------------
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


# -----------------------------
# Auth
# -----------------------------
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


def _require_admin(claims: Claims) -> None:
    if (claims.role or "").lower() != "tenant_admin":
        raise HTTPException(status_code=403, detail="Only tenant_admin can run QC")


# -----------------------------
# DB helpers
# -----------------------------
def _db_conn(settings: Any):
    host = _pick(settings, "DB_HOST", "SUPABASE_DB_HOST", "POSTGRES_HOST")
    port = int(_pick(settings, "DB_PORT", "SUPABASE_DB_PORT", "POSTGRES_PORT", default=5432))
    dbname = _pick(settings, "DB_NAME", "SUPABASE_DB_NAME", "POSTGRES_DB", default="postgres")
    user = _pick(settings, "DB_USER", "SUPABASE_DB_USER", "POSTGRES_USER")
    password = _pick(settings, "DB_PASSWORD", "SUPABASE_DB_PASSWORD", "POSTGRES_PASSWORD")
    sslmode = _pick(settings, "DB_SSLMODE", "SUPABASE_DB_SSLMODE", "POSTGRES_SSLMODE", default="require")

    return psycopg2.connect(
        host=host, port=port, dbname=dbname, user=user, password=password, sslmode=sslmode, connect_timeout=8
    )


def _log_job_event(conn, tenant_id: str, event_type: str, detail: Dict[str, Any], job_id: Optional[str] = None) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO public.job_events (event_id, tenant_id, request_id, job_id, event_type, detail)
            VALUES (%s, %s, %s, %s, %s, %s)
            """,
            (str(uuid.uuid4()), tenant_id, None, job_id, event_type, json.dumps(detail)),
        )


def _ensure_day19_schema(conn) -> None:
    with conn.cursor() as cur:
        cur.execute("ALTER TABLE public.article_requests ADD COLUMN IF NOT EXISTS gcs_qc_uri text;")
        cur.execute("ALTER TABLE public.article_requests ADD COLUMN IF NOT EXISTS qc_fingerprint text;")
        cur.execute("ALTER TABLE public.article_requests ADD COLUMN IF NOT EXISTS qc_meta jsonb;")
        # qc_summary already exists in your schema, but keep safe
        cur.execute("ALTER TABLE public.article_requests ADD COLUMN IF NOT EXISTS qc_summary jsonb;")


# -----------------------------
# GCS helpers
# -----------------------------
def _parse_gs_uri(gs_uri: str) -> Tuple[str, str]:
    if not gs_uri.startswith("gs://"):
        raise ValueError(f"Not a gs:// uri: {gs_uri}")
    rest = gs_uri[len("gs://") :]
    parts = rest.split("/", 1)
    return parts[0], (parts[1] if len(parts) > 1 else "")


def _gcs_client(settings: Any) -> storage.Client:
    project_id = _pick(settings, "GCP_PROJECT_ID", "GOOGLE_CLOUD_PROJECT")
    return storage.Client(project=project_id)


def _gcs_download_json(gcs: storage.Client, gs_uri: str) -> Dict[str, Any]:
    b, o = _parse_gs_uri(gs_uri)
    blob = gcs.bucket(b).blob(o)
    if not blob.exists():
        raise FileNotFoundError(gs_uri)
    data = blob.download_as_bytes()
    return json.loads(data.decode("utf-8"))


def _gcs_upload_create_only(
    gcs: storage.Client,
    bucket_name: str,
    object_name: str,
    data: bytes,
    content_type: str,
) -> str:
    blob = gcs.bucket(bucket_name).blob(object_name)
    try:
        blob.upload_from_string(data, content_type=content_type, if_generation_match=0)
    except PreconditionFailed:
        pass
    return f"gs://{bucket_name}/{object_name}"


def _signed_url(gs_uri: str, minutes: int = 15) -> Optional[str]:
    try:
        b, o = _parse_gs_uri(gs_uri)
        client = storage.Client(project=os.getenv("GCP_PROJECT_ID") or None)
        blob = client.bucket(b).blob(o)
        return blob.generate_signed_url(version="v4", expiration=timedelta(minutes=minutes), method="GET")
    except Exception:
        return None


# -----------------------------
# QC metrics
# -----------------------------
_WORD_RE = re.compile(r"[A-Za-z0-9']+")
_SENT_RE = re.compile(r"[.!?]+")


def _count_syllables(word: str) -> int:
    w = re.sub(r"[^a-z]", "", word.lower())
    if not w:
        return 0
    vowels = "aeiouy"
    groups = 0
    prev = False
    for ch in w:
        is_v = ch in vowels
        if is_v and not prev:
            groups += 1
        prev = is_v
    if w.endswith("e") and groups > 1:
        groups -= 1
    return max(groups, 1)


def _readability(text: str) -> Dict[str, float]:
    words = _WORD_RE.findall(text)
    wc = len(words)
    sc = max(1, len(_SENT_RE.findall(text)) or 1)
    syll = sum(_count_syllables(w) for w in words) if wc else 0

    wps = wc / sc
    spw = (syll / wc) if wc else 0.0

    flesch = 206.835 - 1.015 * wps - 84.6 * spw
    fk_grade = 0.39 * wps + 11.8 * spw - 15.59
    return {
        "word_count": float(wc),
        "sentence_count": float(sc),
        "syllable_count": float(syll),
        "flesch_reading_ease": float(flesch),
        "flesch_kincaid_grade": float(fk_grade),
    }


def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


# -----------------------------
# API models
# -----------------------------
class QCResponse(BaseModel):
    status: str = "ok"
    request_id: str
    tenant_id: str
    kb_id: str

    gcs_draft_uri: str
    draft_fingerprint: str
    attempt_count: int

    qc_pass: bool
    qc_metrics: Dict[str, Any]
    thresholds: Dict[str, Any]

    gcs_qc_uri: str
    qc_fingerprint: str
    qc_signed_url: Optional[str] = None
    qc_signed_url_expires_minutes: Optional[int] = None


# -----------------------------
# Endpoint
# -----------------------------
@router.get("/requests/{request_id}/qc", response_model=QCResponse)
def get_qc_report(
    request_id: str,
    claims: Claims = Depends(require_claims),
    signed_url: bool = Query(default=False),
    signed_url_minutes: int = Query(default=15, ge=1, le=60),
):
    """
    Day 19: Local QC (cost=0): wordcount + readability.
    Saves qc_report.json to GCS and stores pointer in Supabase.

    FIXED: If gcs_qc_uri exists, we now READ the QC JSON and return the REAL qc_pass + metrics.
    """
    _require_admin(claims)
    settings = get_settings()
    tenant_id = claims.tenant_id

    # plan thresholds
    thresholds = {
        "word_count_min": 1950,
        "word_count_max": 2050,
        "fk_grade_min": 7.0,
        "fk_grade_max": 9.0,
    }

    bucket_name = _pick(settings, "GCS_BUCKET_NAME")
    if not bucket_name:
        raise HTTPException(status_code=500, detail="Missing GCS_BUCKET_NAME")

    gcs_articles_prefix = _pick(settings, "GCS_PREFIX_ARTICLES", default="articles/")
    gcs_articles_prefix = (gcs_articles_prefix or "").strip().replace("\\", "/")
    if gcs_articles_prefix and not gcs_articles_prefix.endswith("/"):
        gcs_articles_prefix += "/"

    gcs = _gcs_client(settings)

    # ---- Load request row ----
    with _db_conn(settings) as conn:
        _ensure_day19_schema(conn)

        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                  request_id::text, tenant_id::text, kb_id::text,
                  attempt_count,
                  gcs_draft_uri,
                  draft_fingerprint,
                  gcs_qc_uri,
                  qc_fingerprint
                FROM public.article_requests
                WHERE tenant_id=%s::uuid AND request_id=%s::uuid
                LIMIT 1;
                """,
                (tenant_id, request_id),
            )
            row = cur.fetchone()

        if not row:
            raise HTTPException(status_code=404, detail="Request not found")

        rid, tid, kb_id, attempt_count, gcs_draft_uri, draft_fp, gcs_qc_uri, qc_fp = row

        if not gcs_draft_uri or not draft_fp:
            raise HTTPException(status_code=409, detail="No draft found yet. Run /run first.")

        # ---- TRUE cache hit: read QC JSON and return real values ----
        if gcs_qc_uri and qc_fp:
            try:
                report = _gcs_download_json(gcs, str(gcs_qc_uri))
                # (optional safety) ensure report matches this draft fingerprint
                rep_input = (report.get("input") or {}) if isinstance(report, dict) else {}
                rep_draft_fp = str(rep_input.get("draft_fingerprint") or "")

                if rep_draft_fp and str(rep_draft_fp) != str(draft_fp):
                    # stale QC pointer; force recompute
                    raise ValueError("stale_qc_pointer")

                rep_thresholds = report.get("thresholds") if isinstance(report, dict) else None
                rep_metrics = report.get("metrics") if isinstance(report, dict) else None
                rep_qc_pass = bool(report.get("qc_pass")) if isinstance(report, dict) else False

                if not isinstance(rep_thresholds, dict):
                    rep_thresholds = thresholds
                if not isinstance(rep_metrics, dict):
                    # if missing metrics, recompute by forcing below path
                    raise ValueError("missing_metrics")

                qc_url = _signed_url(str(gcs_qc_uri), minutes=int(signed_url_minutes)) if signed_url else None
                conn.commit()

                return QCResponse(
                    request_id=str(rid),
                    tenant_id=str(tid),
                    kb_id=str(kb_id),
                    gcs_draft_uri=str(gcs_draft_uri),
                    draft_fingerprint=str(draft_fp),
                    attempt_count=int(attempt_count or 0),
                    qc_pass=rep_qc_pass,
                    qc_metrics=rep_metrics,
                    thresholds=rep_thresholds,
                    gcs_qc_uri=str(gcs_qc_uri),
                    qc_fingerprint=str(qc_fp),
                    qc_signed_url=qc_url,
                    qc_signed_url_expires_minutes=int(signed_url_minutes) if qc_url else None,
                )

            except Exception:
                # fall through to recompute below
                pass

        _log_job_event(conn, tenant_id, "qc_started", {"request_id": request_id}, job_id=request_id)
        conn.commit()

    # ---- Download draft from GCS ----
    try:
        draft_obj = _gcs_download_json(gcs, str(gcs_draft_uri))
    except FileNotFoundError:
        raise HTTPException(status_code=502, detail=f"Draft artifact missing in GCS: {gcs_draft_uri}")

    draft = draft_obj.get("draft") or {}
    text = (draft.get("draft_markdown") or "").strip()
    if not text:
        raise HTTPException(status_code=502, detail="Draft JSON has no draft_markdown")

    metrics = _readability(text)
    wc = int(metrics["word_count"])
    fk = float(metrics["flesch_kincaid_grade"])

    qc_pass = (thresholds["word_count_min"] <= wc <= thresholds["word_count_max"]) and (
        thresholds["fk_grade_min"] <= fk <= thresholds["fk_grade_max"]
    )

    report = {
        "request_id": str(request_id),
        "tenant_id": str(tenant_id),
        "kb_id": str(kb_id),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input": {"gcs_draft_uri": str(gcs_draft_uri), "draft_fingerprint": str(draft_fp)},
        "thresholds": thresholds,
        "qc_pass": qc_pass,
        "metrics": metrics,
        "notes": "Local QC only (cost=0). ZeroGPT comes later (Day 20).",
    }

    out_bytes = json.dumps(report, ensure_ascii=False, indent=2).encode("utf-8")
    out_fp = _sha256_bytes(out_bytes)

    attempt_no = int(attempt_count or 0)
    qc_obj = f"{gcs_articles_prefix}{tenant_id}/{kb_id}/{request_id}/attempt_{attempt_no}/qc_v1/{out_fp}.json"
    gcs_qc_uri_new = _gcs_upload_create_only(
        gcs,
        bucket_name=bucket_name,
        object_name=qc_obj,
        data=out_bytes,
        content_type="application/json; charset=utf-8",
    )

    qc_url = _signed_url(gcs_qc_uri_new, minutes=int(signed_url_minutes)) if signed_url else None

    # ---- Save pointers back to Supabase ----
    with _db_conn(settings) as conn:
        _ensure_day19_schema(conn)
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE public.article_requests
                SET
                  qc_summary=%s::jsonb,
                  gcs_qc_uri=%s,
                  qc_fingerprint=%s,
                  qc_meta=%s::jsonb
                WHERE tenant_id=%s::uuid AND request_id=%s::uuid
                """,
                (
                    json.dumps({"qc_pass": qc_pass, "word_count": wc, "fk_grade": fk}),
                    gcs_qc_uri_new,
                    out_fp,
                    json.dumps({"attempt_no": attempt_no}),
                    tenant_id,
                    request_id,
                ),
            )

        _log_job_event(conn, tenant_id, "qc_saved", {"gcs_qc_uri": gcs_qc_uri_new, "qc_pass": qc_pass}, job_id=request_id)
        _log_job_event(conn, tenant_id, "qc_done", {"request_id": request_id}, job_id=request_id)
        conn.commit()

    return QCResponse(
        request_id=str(request_id),
        tenant_id=str(tenant_id),
        kb_id=str(kb_id),
        gcs_draft_uri=str(gcs_draft_uri),
        draft_fingerprint=str(draft_fp),
        attempt_count=attempt_no,
        qc_pass=qc_pass,
        qc_metrics=metrics,
        thresholds=thresholds,
        gcs_qc_uri=gcs_qc_uri_new,
        qc_fingerprint=out_fp,
        qc_signed_url=qc_url,
        qc_signed_url_expires_minutes=int(signed_url_minutes) if qc_url else None,
    )