# app/api/article_zerogpt.py
from __future__ import annotations

import hashlib
import json
import os
import re
import uuid
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, Optional, Tuple, List

import psycopg2
from psycopg2.extras import RealDictCursor, Json
import requests
from fastapi import APIRouter, Depends, HTTPException, Header, Query
from jose import JWTError, jwt
from pydantic import BaseModel, Field

from google.cloud import storage
from google.api_core.exceptions import PreconditionFailed


router = APIRouter(prefix="/api/v1/articles", tags=["article-zerogpt"])


# ============================================================
# Settings loader
# ============================================================
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


# ============================================================
# Auth
# ============================================================
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


# ============================================================
# DB helpers (Supabase Postgres)
# ============================================================
def _db_conn(settings: Any):
    host = _pick(settings, "DB_HOST", "SUPABASE_DB_HOST", "POSTGRES_HOST")
    port = int(_pick(settings, "DB_PORT", "SUPABASE_DB_PORT", "POSTGRES_PORT", default=5432))
    dbname = _pick(settings, "DB_NAME", "SUPABASE_DB_NAME", "POSTGRES_DB", default="postgres")
    user = _pick(settings, "DB_USER", "SUPABASE_DB_USER", "POSTGRES_USER")
    password = _pick(settings, "DB_PASSWORD", "SUPABASE_DB_PASSWORD", "POSTGRES_PASSWORD")
    sslmode = _pick(settings, "DB_SSLMODE", "SUPABASE_DB_SSLMODE", "POSTGRES_SSLMODE", default="require")

    return psycopg2.connect(
        host=host,
        port=port,
        dbname=dbname,
        user=user,
        password=password,
        sslmode=sslmode,
        connect_timeout=8,
    )


def _ensure_day20_columns(conn) -> None:
    with conn.cursor() as cur:
        cur.execute("ALTER TABLE public.article_requests ADD COLUMN IF NOT EXISTS gcs_zerogpt_uri text;")
        cur.execute("ALTER TABLE public.article_requests ADD COLUMN IF NOT EXISTS zerogpt_fingerprint text;")
        cur.execute("ALTER TABLE public.article_requests ADD COLUMN IF NOT EXISTS zerogpt_score double precision;")
        cur.execute("ALTER TABLE public.article_requests ADD COLUMN IF NOT EXISTS zerogpt_meta jsonb;")


def _log_job_event(conn, tenant_id: str, event_type: str, detail: Dict[str, Any], request_id: Optional[str] = None) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO public.job_events (event_id, tenant_id, request_id, job_id, event_type, detail)
            VALUES (%s, %s, %s, %s, %s, %s)
            """,
            (
                str(uuid.uuid4()),
                tenant_id,
                request_id,
                None,
                event_type,
                Json(detail),
            ),
        )


def _usage_events_columns(conn) -> List[str]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_schema='public' AND table_name='usage_events'
            """
        )
        return [r[0] for r in cur.fetchall()]


def _log_usage_event_best_effort(
    conn,
    *,
    tenant_id: str,
    request_id: Optional[str],
    operation_name: str,
    tokens: int,
    total_cost: float,
    vendor: Optional[str] = None,
    model: Optional[str] = None,
    detail: Optional[Dict[str, Any]] = None,
) -> None:
    cols = set(_usage_events_columns(conn))
    payload: Dict[str, Any] = {}

    if "event_id" in cols:
        payload["event_id"] = str(uuid.uuid4())
    if "tenant_id" in cols:
        payload["tenant_id"] = tenant_id
    if "request_id" in cols and request_id:
        payload["request_id"] = request_id
    if "operation_name" in cols:
        payload["operation_name"] = operation_name
    if "tokens" in cols:
        payload["tokens"] = int(tokens)
    if "total_cost" in cols:
        payload["total_cost"] = float(total_cost)

    if vendor and "vendor" in cols:
        payload["vendor"] = vendor
    if model and "model" in cols:
        payload["model"] = model

    if detail:
        for k in ("detail", "meta", "metadata"):
            if k in cols:
                payload[k] = Json(detail)
                break

    if not payload or ("tenant_id" not in payload) or ("operation_name" not in payload):
        return

    columns = list(payload.keys())
    values = [payload[c] for c in columns]
    placeholders = ", ".join(["%s"] * len(values))
    col_sql = ", ".join(columns)

    with conn.cursor() as cur:
        cur.execute(f"INSERT INTO public.usage_events ({col_sql}) VALUES ({placeholders})", values)


# ============================================================
# GCS helpers
# ============================================================
def _gcs_client(settings: Any) -> storage.Client:
    project_id = _pick(settings, "GCP_PROJECT_ID", "GOOGLE_CLOUD_PROJECT")
    return storage.Client(project=project_id)


def _parse_gs_uri(gs_uri: str) -> Tuple[str, str]:
    if not gs_uri.startswith("gs://"):
        raise ValueError(f"Not a gs:// uri: {gs_uri}")
    rest = gs_uri[len("gs://") :]
    parts = rest.split("/", 1)
    bucket = parts[0]
    obj = parts[1] if len(parts) > 1 else ""
    return bucket, obj


def _gcs_download_bytes(client: storage.Client, gs_uri: str) -> bytes:
    bucket_name, obj = _parse_gs_uri(gs_uri)
    blob = client.bucket(bucket_name).blob(obj)
    if not blob.exists():
        raise FileNotFoundError(f"GCS object not found: {gs_uri}")
    return blob.download_as_bytes()


def _gcs_download_json(client: storage.Client, gs_uri: str) -> Dict[str, Any]:
    b = _gcs_download_bytes(client, gs_uri)
    try:
        o = json.loads(b.decode("utf-8", errors="replace"))
        return o if isinstance(o, dict) else {}
    except Exception:
        return {}


def _gcs_upload_bytes_create_only(
    client: storage.Client,
    bucket_name: str,
    object_name: str,
    data: bytes,
    content_type: str,
) -> str:
    blob = client.bucket(bucket_name).blob(object_name)
    try:
        blob.upload_from_string(data, content_type=content_type, if_generation_match=0)
    except PreconditionFailed:
        pass
    return f"gs://{bucket_name}/{object_name}"


def _gcs_signed_url(client: storage.Client, gs_uri: str, minutes: int) -> str:
    bucket, obj = _parse_gs_uri(gs_uri)
    blob = client.bucket(bucket).blob(obj)
    return blob.generate_signed_url(version="v4", expiration=timedelta(minutes=int(minutes)), method="GET")


def _norm_prefix(p: str) -> str:
    p = (p or "").strip().replace("\\", "/")
    return p if (not p or p.endswith("/")) else (p + "/")


# ============================================================
# ZeroGPT helpers (THIS MATCHES YOUR WORKING TEST)
# ============================================================
def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _strip_markdown_noise(text: str) -> str:
    text = text or ""
    text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text


def _call_zerogpt_zerogpt_com(base_url: str, api_key: str, text: str, timeout_s: int = 60) -> Dict[str, Any]:
    """
    Calls:
      POST https://api.zerogpt.com/api/detect/detectText
      Header: ApiKey: <key>
      Body:   {"input_text": "..."}
    """
    base_url = (base_url or "").strip().rstrip("/")
    if not base_url:
        raise RuntimeError("Missing ZEROGPT_BASE_URL")

    endpoint_path = os.getenv("ZEROGPT_ENDPOINT_PATH", "/api/detect/detectText")
    if not endpoint_path.startswith("/"):
        endpoint_path = "/" + endpoint_path

    url = base_url + endpoint_path
    headers = {"ApiKey": api_key, "Content-Type": "application/json"}
    payload = {"input_text": text}

    try:
        r = requests.post(url, headers=headers, json=payload, timeout=(10, timeout_s))
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"ZeroGPT request failed (network): {type(e).__name__}: {str(e)[:200]}")

    if r.status_code != 200:
        raise HTTPException(status_code=502, detail=f"ZeroGPT failed: {r.status_code}: {r.text[:500]}")

    try:
        return r.json()
    except Exception:
        return {"raw_text": r.text}


def _extract_score(resp: Dict[str, Any]) -> Tuple[Optional[float], Dict[str, Any]]:
    """
    From your response:
      resp["data"]["fakePercentage"]  (0..100, AI-likelihood)
      resp["data"]["isHuman"]         (0..100)
      resp["data"]["sentences"]       spans-like info (often empty)
      resp["data"]["h"], resp["data"]["hi"], etc.
    We'll define:
      zerogpt_score = fakePercentage
    """
    data = resp.get("data") if isinstance(resp, dict) else None
    if not isinstance(data, dict):
        return None, {"warning": "missing_data_field"}

    score = None
    try:
        score = float(data.get("fakePercentage"))
    except Exception:
        score = None

    meta = {
        "success": resp.get("success"),
        "code": resp.get("code"),
        "message": resp.get("message"),
        "isHuman": data.get("isHuman"),
        "textWords": data.get("textWords"),
        "aiWords": data.get("aiWords"),
        "feedback": data.get("feedback"),
        "additional_feedback": data.get("additional_feedback"),
        "sentences": data.get("sentences"),
        "specialIndexes": data.get("specialIndexes"),
        "specialSentences": data.get("specialSentences"),
        "h": data.get("h"),
        "hi": data.get("hi"),
        "raw_id": data.get("id"),
    }
    # spans_count for UI
    try:
        meta["spans_count"] = len(data.get("sentences") or [])
    except Exception:
        meta["spans_count"] = 0

    return score, meta


# ============================================================
# Response model
# ============================================================
class ZeroGPTResponse(BaseModel):
    status: str
    request_id: str
    tenant_id: str
    kb_id: str
    attempt_no: int

    gcs_draft_uri: str
    draft_fingerprint: str

    zerogpt_score: Optional[float] = None
    gcs_zerogpt_uri: str
    zerogpt_fingerprint: str

    spans_count: int = 0

    zerogpt_signed_url: Optional[str] = None
    zerogpt_signed_url_expires_minutes: Optional[int] = None

    meta: Dict[str, Any] = Field(default_factory=dict)


# ============================================================
# Endpoint (Day 20)
# ============================================================
@router.get("/requests/{request_id}/zerogpt", response_model=ZeroGPTResponse)
def run_zerogpt(
    request_id: str,
    signed_url: bool = Query(False),
    signed_url_minutes: int = Query(15, ge=1, le=120),
    force: bool = Query(False),
    claims: Claims = Depends(require_claims),
):
    settings = get_settings()
    tenant_id = claims.tenant_id

    api_key = os.getenv("ZEROGPT_API_KEY")
    base_url = os.getenv("ZEROGPT_BASE_URL")
    if not api_key or not base_url:
        raise HTTPException(status_code=500, detail="Missing ZEROGPT_API_KEY or ZEROGPT_BASE_URL")

    bucket_name = _pick(settings, "GCS_BUCKET_NAME")
    if not bucket_name:
        raise HTTPException(status_code=500, detail="Missing GCS_BUCKET_NAME")

    gcs = _gcs_client(settings)

    # ---- Load request row + cache-hit check ----
    with _db_conn(settings) as conn:
        _ensure_day20_columns(conn)

        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT request_id, tenant_id, kb_id, status,
                       attempt_count, gcs_draft_uri, draft_fingerprint,
                       gcs_qc_uri, qc_fingerprint,
                       gcs_zerogpt_uri, zerogpt_fingerprint, zerogpt_score, zerogpt_meta
                FROM public.article_requests
                WHERE tenant_id=%s::uuid AND request_id=%s::uuid
                """,
                (tenant_id, request_id),
            )
            row = cur.fetchone()

        if not row:
            raise HTTPException(status_code=404, detail="Request not found for this tenant")

        kb_id = str(row["kb_id"])
        attempt_no = int(row.get("attempt_count") or 0)

        gcs_draft_uri = row.get("gcs_draft_uri")
        draft_fp = row.get("draft_fingerprint")
        if not gcs_draft_uri or not draft_fp:
            raise HTTPException(status_code=409, detail="Draft not found. Run /run first (Day 17).")

        gcs_draft_uri = str(gcs_draft_uri)
        draft_fp = str(draft_fp)

        gcs_qc_uri = row.get("gcs_qc_uri")
        qc_fp = row.get("qc_fingerprint")
        if not gcs_qc_uri or not qc_fp:
            raise HTTPException(status_code=409, detail="QC not found yet. Run /qc first (Day 19).")

        # Read QC JSON from GCS for truth (avoid cached assumptions)
        try:
            qc_obj = _gcs_download_json(gcs, str(gcs_qc_uri))
            qc_pass = bool(qc_obj.get("qc_pass")) if isinstance(qc_obj, dict) else False
        except Exception:
            raise HTTPException(status_code=502, detail="Failed to read QC artifact from GCS.")

        if not qc_pass:
            raise HTTPException(status_code=409, detail="QC failed. Fix draft with /qc-fix before ZeroGPT.")

        # Cache hit is valid only if it matches current draft_fp (important!)
        if (not force) and row.get("gcs_zerogpt_uri") and row.get("zerogpt_fingerprint"):
            meta = row.get("zerogpt_meta") or {}
            cached_for_fp = meta.get("draft_fingerprint") if isinstance(meta, dict) else None
            if str(cached_for_fp or "") == draft_fp:
                out_uri = str(row["gcs_zerogpt_uri"])
                out_fp = str(row["zerogpt_fingerprint"])
                out_score = row.get("zerogpt_score")
                spans_count = int(meta.get("spans_count") or 0) if isinstance(meta, dict) else 0

                signed = _gcs_signed_url(gcs, out_uri, signed_url_minutes) if signed_url else None
                return ZeroGPTResponse(
                    status="ok",
                    request_id=str(row["request_id"]),
                    tenant_id=str(row["tenant_id"]),
                    kb_id=kb_id,
                    attempt_no=attempt_no,
                    gcs_draft_uri=gcs_draft_uri,
                    draft_fingerprint=draft_fp,
                    zerogpt_score=float(out_score) if out_score is not None else None,
                    gcs_zerogpt_uri=out_uri,
                    zerogpt_fingerprint=out_fp,
                    spans_count=spans_count,
                    zerogpt_signed_url=signed,
                    zerogpt_signed_url_expires_minutes=(signed_url_minutes if signed_url else None),
                    meta={"cached": True},
                )

        _log_job_event(conn, tenant_id, "zerogpt_started", {"request_id": request_id}, request_id=request_id)
        conn.commit()

    # ---- Download draft JSON from GCS and extract nested draft_markdown ----
    try:
        draft_bytes = _gcs_download_bytes(gcs, gcs_draft_uri)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Failed to download draft from GCS: {type(e).__name__}: {str(e)[:200]}")

    try:
        draft_obj = json.loads(draft_bytes.decode("utf-8", errors="replace"))
    except Exception:
        draft_obj = {}

    draft_md = ""
    if isinstance(draft_obj, dict):
        d = draft_obj.get("draft")
        if isinstance(d, dict):
            draft_md = str(d.get("draft_markdown") or d.get("content") or d.get("text") or "")
        else:
            draft_md = str(draft_obj.get("draft_markdown") or draft_obj.get("content") or draft_obj.get("text") or "")

    draft_md = _strip_markdown_noise(draft_md)
    if not draft_md.strip():
        raise HTTPException(status_code=422, detail="Draft markdown is empty; cannot run ZeroGPT.")

    # ---- Call ZeroGPT (api.zerogpt.com) ----
    resp = _call_zerogpt_zerogpt_com(base_url=base_url, api_key=api_key, text=draft_md, timeout_s=60)
    score, meta = _extract_score(resp)
    try:
        txt_words = int(meta.get("textWords") or 0) if isinstance(meta, dict) else 0
        ai_words = int(meta.get("aiWords") or 0) if isinstance(meta, dict) else 0
    except Exception:
        txt_words, ai_words = 0, 0
    confidence = "normal"
    if score is not None and score <= 1.0 and ai_words == 0 and txt_words >= 1500:
        confidence = "low"
    if isinstance(meta, dict):
        meta["confidence"] = confidence
    else:
        meta = {"confidence": confidence}

    # ---- Build + store artifact in GCS ----
    artifact = {
        "request_id": request_id,
        "tenant_id": tenant_id,
        "kb_id": kb_id,
        "attempt_no": attempt_no,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input": {
            "gcs_draft_uri": gcs_draft_uri,
            "draft_fingerprint": draft_fp,
            "chars": len(draft_md),
        },
        "zerogpt": {
            "score_fakePercentage": score,
            "meta": meta,
            "raw": resp,
        },
    }

    out_bytes = json.dumps(artifact, ensure_ascii=False, indent=2).encode("utf-8")
    out_fp = _sha256_bytes(out_bytes)

    articles_prefix = _norm_prefix(_pick(settings, "GCS_PREFIX_ARTICLES", default="articles/"))
    out_object = (
        f"{articles_prefix}{tenant_id}/{kb_id}/{request_id}/attempt_{attempt_no}/"
        f"zerogpt_v1/{out_fp}.json"
    )
    out_uri = _gcs_upload_bytes_create_only(
        gcs,
        bucket_name=bucket_name,
        object_name=out_object,
        data=out_bytes,
        content_type="application/json; charset=utf-8",
    )

    spans_count = int(meta.get("spans_count") or 0) if isinstance(meta, dict) else 0

    # ---- Update Supabase + logs ----
    meta_db = dict(meta) if isinstance(meta, dict) else {}
    meta_db["draft_fingerprint"] = draft_fp
    meta_db["endpoint"] = "/api/detect/detectText"
    meta_db["base_url"] = base_url

    with _db_conn(settings) as conn:
        _ensure_day20_columns(conn)

        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE public.article_requests
                SET gcs_zerogpt_uri=%s,
                    zerogpt_fingerprint=%s,
                    zerogpt_score=%s,
                    zerogpt_meta=%s::jsonb,
                    updated_at=%s
                WHERE tenant_id=%s::uuid AND request_id=%s::uuid
                """,
                (
                    out_uri,
                    out_fp,
                    score,
                    json.dumps(meta_db),
                    datetime.now(timezone.utc),
                    tenant_id,
                    request_id,
                ),
            )

        _log_job_event(
            conn,
            tenant_id,
            "zerogpt_saved",
            {"gcs_zerogpt_uri": out_uri, "zerogpt_fingerprint": out_fp, "score": score, "spans_count": spans_count},
            request_id=request_id,
        )

        _log_usage_event_best_effort(
            conn,
            tenant_id=tenant_id,
            request_id=request_id,
            operation_name="zerogpt_score",
            vendor="zerogpt",
            model=None,
            tokens=0,
            total_cost=0.0,
            detail={"chars": len(draft_md), "spans_count": spans_count, "score_fakePercentage": score},
        )

        conn.commit()

    signed = _gcs_signed_url(gcs, out_uri, signed_url_minutes) if signed_url else None

    return ZeroGPTResponse(
        status="ok",
        request_id=request_id,
        tenant_id=tenant_id,
        kb_id=kb_id,
        attempt_no=attempt_no,
        gcs_draft_uri=gcs_draft_uri,
        draft_fingerprint=draft_fp,
        zerogpt_score=score,
        gcs_zerogpt_uri=out_uri,
        zerogpt_fingerprint=out_fp,
        spans_count=spans_count,
        zerogpt_signed_url=signed,
        zerogpt_signed_url_expires_minutes=(signed_url_minutes if signed_url else None),
        meta={"cached": False},
    )
