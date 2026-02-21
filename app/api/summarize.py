from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import psycopg2
import logging
from fastapi import APIRouter, Depends, Header, HTTPException
from jose import JWTError, jwt
from pydantic import BaseModel, Field

from google.cloud import storage
from google.api_core.exceptions import PreconditionFailed
import google.generativeai as genai
import requests


# ============================================================
# Router
# ============================================================
router = APIRouter(prefix="/api/v1/kb", tags=["summarize"])
logger = logging.getLogger("summarize")
if not logger.handlers:
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))


# ============================================================
# Settings loader (same pattern as your ingest.py)
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
# Auth (same Claims style as your ingest.py)
# ============================================================
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


# ============================================================
# DB helpers (for job_events logging)
# ============================================================
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
                json.dumps(detail),
            ),
        )


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


def _gcs_blob_exists(client: storage.Client, bucket_name: str, object_name: str) -> bool:
    return client.bucket(bucket_name).blob(object_name).exists()


def _gcs_upload_bytes(
    client: storage.Client,
    bucket_name: str,
    object_name: str,
    data: bytes,
    content_type: str,
) -> str:
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(object_name)
    try:
        # Create-only (no overwrite => no delete permission needed)
        blob.upload_from_string(data, content_type=content_type, if_generation_match=0)
    except PreconditionFailed:
        # Object already exists => treat as cache hit / reuse
        pass
    return f"gs://{bucket_name}/{object_name}"


def _norm_prefix(p: str) -> str:
    p = (p or "").strip().replace("\\", "/")
    if not p:
        return ""
    return p if p.endswith("/") else (p + "/")


# ============================================================
# Fingerprints
# ============================================================
def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _params_hash(obj: Dict[str, Any]) -> str:
    raw = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return _sha256_bytes(raw)


# ============================================================
# Gemini API (google-generativeai) - allowed models only
# ============================================================
ALLOWED_MODELS = {
    "gemini-2.5-flash",
    "gemini-2.5-pro",
}
GROQ_DEFAULT_MODEL = "llama-3.3-70b-versatile"
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"


@dataclass
class VertexUsage:
    prompt_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0


class VertexGeminiClient:
    def __init__(
        self,
        api_key: Optional[str],
        primary_model: str,
        fallback_model: str,
        *,
        groq_api_key: Optional[str] = None,
        groq_model: Optional[str] = None,
    ):
        if primary_model not in ALLOWED_MODELS:
            raise ValueError(f"primary_model not allowed: {primary_model}")
        if fallback_model not in ALLOWED_MODELS:
            raise ValueError(f"fallback_model not allowed: {fallback_model}")

        self.primary_model = primary_model
        self.fallback_model = fallback_model
        self._model_cache: Dict[str, Any] = {}
        self.last_provider: Optional[str] = None
        self.last_model: Optional[str] = None

        self._gemini_key = api_key
        self._groq_key = groq_api_key
        self._groq_model = groq_model or GROQ_DEFAULT_MODEL

        if not self._gemini_key and not self._groq_key:
            raise ValueError("Missing GEMINI_API_KEY / GOOGLE_API_KEY and GROQ_API_KEY")

        # Configure Gemini only if key is available
        if self._gemini_key:
            genai.configure(api_key=self._gemini_key)

    def _get_model(self, model: str):
        m = self._model_cache.get(model)
        if m is None:
            m = genai.GenerativeModel(model)
            self._model_cache[model] = m
        return m

    def _usage_from_metadata(self, um: Any) -> VertexUsage:
        def _get(obj: Any, attr: str, key: str) -> int:
            if obj is None:
                return 0
            if hasattr(obj, attr):
                return int(getattr(obj, attr) or 0)
            if isinstance(obj, dict):
                return int(obj.get(key) or 0)
            return 0

        return VertexUsage(
            prompt_tokens=_get(um, "prompt_token_count", "prompt_token_count"),
            output_tokens=_get(um, "candidates_token_count", "candidates_token_count"),
            total_tokens=_get(um, "total_token_count", "total_token_count"),
        )

    def generate_json(
        self,
        prompt: str,
        *,
        system: str,
        model: Optional[str] = None,
        max_output_tokens: int = 1024,
        temperature: float = 0.2,
        timeout_s: int = 60,
        retries: int = 4,
    ) -> Tuple[Dict[str, Any], VertexUsage]:
        chosen = model or self.primary_model
        last_err: Optional[Exception] = None

        # Try Gemini first if configured (primary then fallback)
        if self._gemini_key:
            logger.info("LLM: trying Gemini models (primary=%s fallback=%s)", chosen, self.fallback_model)
            for m in [chosen, self.fallback_model] if chosen != self.fallback_model else [chosen]:
                try:
                    out = self._generate_json_once(
                        prompt,
                        system=system,
                        model=m,
                        max_output_tokens=max_output_tokens,
                        temperature=temperature,
                        timeout_s=timeout_s,
                        retries=retries,
                    )
                    self.last_provider = "gemini_api"
                    self.last_model = m
                    logger.info("LLM: Gemini success (model=%s)", m)
                    return out
                except Exception as e:
                    last_err = e
                    logger.warning("LLM: Gemini failed (model=%s, err=%s)", m, str(e)[:300])

        # Gemini unavailable/failed -> fallback to Groq if configured
        if self._groq_key:
            logger.info("LLM: falling back to Groq (model=%s)", self._groq_model)
            out = self._generate_json_groq(
                prompt,
                system=system,
                model=self._groq_model,
                max_output_tokens=max_output_tokens,
                temperature=temperature,
                timeout_s=timeout_s,
                retries=retries,
            )
            self.last_provider = "groq"
            self.last_model = self._groq_model
            logger.info("LLM: Groq success (model=%s)", self._groq_model)
            return out

        raise HTTPException(
            status_code=502,
            detail=f"LLM provider unavailable (Gemini and Groq not configured). Last error: {last_err}",
        )

    def _generate_json_once(
        self,
        prompt: str,
        *,
        system: str,
        model: str,
        max_output_tokens: int,
        temperature: float,
        timeout_s: int,
        retries: int,
    ) -> Tuple[Dict[str, Any], VertexUsage]:
        usage = VertexUsage()
        last_err: Optional[str] = None

        for attempt in range(retries + 1):
            try:
                logger.debug("LLM: Gemini attempt %s/%s (model=%s)", attempt + 1, retries + 1, model)
                prompt_text = f"{system}\n\n{prompt}" if system else prompt
                gen_cfg = genai.types.GenerationConfig(
                    temperature=float(temperature),
                    max_output_tokens=int(max_output_tokens),
                    response_mime_type="application/json",
                )
                resp = self._get_model(model).generate_content(
                    prompt_text,
                    generation_config=gen_cfg,
                    request_options={"timeout": timeout_s},
                )
            except Exception as e:
                last_err = str(e)
                logger.warning("LLM: Gemini request error (attempt=%s, err=%s)", attempt + 1, last_err[:300])
                time.sleep(min(2 ** attempt, 10))
                continue

            usage = self._usage_from_metadata(getattr(resp, "usage_metadata", None))

            text_out = (getattr(resp, "text", "") or "").strip()
            if not text_out:
                try:
                    cand0 = (getattr(resp, "candidates", []) or [])[0]
                    parts = (((getattr(cand0, "content", None) or {}).get("parts")) or [])
                    text_out = (parts[0].get("text") if parts else "") or ""
                except Exception:
                    text_out = ""

            parsed = _safe_json_loads(text_out)
            if parsed is None:
                # try to recover JSON substring
                parsed = _safe_json_loads(_extract_json_object(text_out) or "")
            if parsed is None:
                raise HTTPException(status_code=502, detail="Gemini API returned non-JSON output (could not parse).")

            return parsed, usage

        raise HTTPException(status_code=502, detail=f"Gemini API retry exhausted. Last error: {last_err}")

    def _generate_json_groq(
        self,
        prompt: str,
        *,
        system: str,
        model: str,
        max_output_tokens: int,
        temperature: float,
        timeout_s: int,
        retries: int,
    ) -> Tuple[Dict[str, Any], VertexUsage]:
        if not self._groq_key:
            raise HTTPException(status_code=502, detail="Groq API key not configured")

        usage = VertexUsage()
        last_err: Optional[str] = None
        for attempt in range(retries + 1):
            logger.debug("LLM: Groq attempt %s/%s (model=%s)", attempt + 1, retries + 1, model)
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})

            body = {
                "model": model,
                "messages": messages,
                "temperature": float(temperature),
                "max_tokens": int(max_output_tokens),
                "response_format": {"type": "json_object"},
            }
            headers = {"Authorization": f"Bearer {self._groq_key}", "Content-Type": "application/json"}

            try:
                resp = requests.post(GROQ_API_URL, headers=headers, json=body, timeout=timeout_s)
            except Exception as e:
                last_err = str(e)
                logger.warning("LLM: Groq request error (attempt=%s, err=%s)", attempt + 1, last_err[:300])
                time.sleep(min(2 ** attempt, 10))
                continue

            if resp.status_code in (429, 500, 502, 503, 504):
                last_err = f"{resp.status_code}: {resp.text[:300]}"
                logger.warning("LLM: Groq retryable error (attempt=%s, err=%s)", attempt + 1, last_err)
                time.sleep(min(2 ** attempt, 10))
                continue

            if resp.status_code != 200:
                last_err = f"{resp.status_code}: {resp.text[:500]}"
                logger.warning("LLM: Groq non-200 error (err=%s)", last_err)
                break

            data = resp.json()
            try:
                content = data["choices"][0]["message"]["content"]
            except Exception:
                content = ""

            um = data.get("usage") or {}
            usage.prompt_tokens = int(um.get("prompt_tokens") or 0)
            usage.output_tokens = int(um.get("completion_tokens") or 0)
            usage.total_tokens = int(um.get("total_tokens") or (usage.prompt_tokens + usage.output_tokens))

            parsed = _safe_json_loads(content)
            if parsed is None:
                parsed = _safe_json_loads(_extract_json_object(content) or "")
            if parsed is None:
                last_err = "Groq returned non-JSON output (could not parse)."
                logger.warning("LLM: Groq non-JSON response")
                break

            return parsed, usage

        raise HTTPException(status_code=502, detail=f"Groq API retry exhausted. Last error: {last_err}")

def _safe_json_loads(s: str) -> Optional[Dict[str, Any]]:
    s = (s or "").strip()
    if not s:
        return None
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            return obj
        return None
    except Exception:
        return None


def _extract_json_object(s: str) -> Optional[str]:
    if not s:
        return None
    # naive: first {...} block
    m = re.search(r"\{.*\}", s, flags=re.DOTALL)
    return m.group(0) if m else None


# ============================================================
# Chunk parsing
# ============================================================
@dataclass
class Chunk:
    chunk_id: str
    text: str
    meta: Dict[str, Any]


def _parse_chunks_jsonl(jsonl_bytes: bytes, max_chunks: Optional[int] = None) -> List[Chunk]:
    out: List[Chunk] = []
    for i, line in enumerate(jsonl_bytes.decode("utf-8", errors="replace").splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue

        txt = obj.get("text") or obj.get("content") or ""
        if not isinstance(txt, str):
            txt = str(txt)

        cid = obj.get("chunk_id") or obj.get("id") or f"chunk_{len(out)+1}"
        meta = {k: v for k, v in obj.items() if k not in ("text", "content")}

        out.append(Chunk(chunk_id=str(cid), text=txt, meta=meta))

        if max_chunks and len(out) >= max_chunks:
            break
    return out


# ============================================================
# Prompts (Stage 4)
# ============================================================
CHUNK_SYSTEM = (
    "You are a senior technical analyst. "
    "Return ONLY valid JSON. Do not add markdown. Do not add extra keys."
)

DOC_SYSTEM = (
    "You are a senior architect assistant. "
    "Return ONLY valid JSON. No markdown. No extra keys."
)


def _chunk_prompt(chunk_id: str, chunk_text: str) -> str:
    return (
        "Schema:\n"
        "{\n"
        '  "chunk_id": "string",\n'
        '  "title": "short heading (max 12 words)",\n'
        '  "summary": "1-3 sentences",\n'
        '  "key_points": ["point1", "point2", "point3"],\n'
        '  "tags": ["tag1", "tag2"]\n'
        "}\n\n"
        f"chunk_id: {chunk_id}\n"
        "Chunk text:\n"
        "<<<\n"
        f"{chunk_text}\n"
        ">>>"
    )


def _doc_prompt(chunk_summaries: List[Dict[str, Any]]) -> str:
    # Keep prompt small: send only title + key_points
    compact = []
    for cs in chunk_summaries:
        compact.append(
            {
                "chunk_id": cs.get("chunk_id"),
                "title": cs.get("title"),
                "key_points": cs.get("key_points", [])[:6],
            }
        )
    return (
        "Build a document-level summary and an auto table-of-contents from chunk signals.\n"
        "Return JSON ONLY with this schema:\n"
        "{\n"
        '  "doc_title": "string",\n'
        '  "one_paragraph_summary": "string",\n'
        '  "toc": [ { "section": "string", "chunk_ids": ["c1","c2"], "summary": "string" } ],\n'
        '  "themes": ["t1","t2"],\n'
        '  "recommended_queries": ["q1","q2","q3"]\n'
        "}\n\n"
        "Chunk signals:\n"
        f"{json.dumps(compact, ensure_ascii=False)}"
    )


# ============================================================
# Request/Response models
# ============================================================
class SummarizeRequest(BaseModel):
    # Use your chunk output JSONL (recommended)
    gcs_chunks_uri: Optional[str] = None

    # If not provided, we auto-derive from the latest chunk artifact in DB.
    input_kind: str = "chunks"

    # LLM controls
    primary_model: str = "gemini-2.5-flash"
    fallback_model: str = "gemini-2.5-pro"
    temperature: float = 0.2

    max_chunks: int = 200
    max_chars_per_chunk: int = 8000
    max_output_tokens_chunk: int = 700
    max_output_tokens_doc: int = 1100
    max_concurrent_llm: int = 4


class SummarizeResponse(BaseModel):
    status: str
    kb_id: str
    doc_id: str
    summarize_job_id: str

    input_chunks_uri: str
    input_chunks_fingerprint: str
    params_hash: str

    chunk_count: int
    primary_model: str
    fallback_model: str

    gcs_summary_uri: str
    summary_fingerprint: str

    usage: Dict[str, Any] = Field(default_factory=dict)
    meta: Dict[str, Any] = Field(default_factory=dict)


# ============================================================
# Endpoint
# ============================================================
@router.post("/{kb_id}/summarize/{doc_id}", response_model=SummarizeResponse)
async def summarize_doc(
    kb_id: str,
    doc_id: str,
    req: SummarizeRequest,
    claims: Claims = Depends(require_claims),
):
    """
    Day 13 – Stage 4:
    Read chunk JSONL from GCS → Gemini API → write summary JSON to GCS.
    """
    settings = get_settings()
    tenant_id = claims.tenant_id
    logger.info("summarize: start kb_id=%s doc_id=%s tenant_id=%s", kb_id, doc_id, tenant_id)

    # Gemini + Groq settings
    api_key = _pick(settings, "GEMINI_API_KEY", "GOOGLE_API_KEY")
    groq_api_key = _pick(settings, "GROQ_API_KEY")
    groq_model = _pick(settings, "GROQ_MODEL", default=GROQ_DEFAULT_MODEL)
    logger.info(
        "summarize: provider_config gemini=%s groq=%s groq_model=%s",
        "set" if api_key else "missing",
        "set" if groq_api_key else "missing",
        groq_model,
    )
    if not api_key and not groq_api_key:
        raise HTTPException(status_code=500, detail="Missing GEMINI_API_KEY/GOOGLE_API_KEY and GROQ_API_KEY")

    # GCS settings
    bucket_name = _pick(settings, "GCS_BUCKET_NAME")
    if not bucket_name:
        raise HTTPException(status_code=500, detail="Missing GCS_BUCKET_NAME")
    processed_prefix = _norm_prefix(_pick(settings, "GCS_PROCESSED_PREFIX", default="processed"))

    summarize_job_id = str(uuid.uuid4())

    # derive chunks uri if not provided
    chunks_uri = req.gcs_chunks_uri
    if not chunks_uri:
        # prefer most recent chunk artifact from DB (supports versioned keys)
        with _db_conn(settings) as conn:
            try:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT gcs_chunks_uri
                        FROM public.chunks
                        WHERE tenant_id=%s::uuid AND kb_id=%s::uuid AND doc_id=%s::uuid
                        ORDER BY created_at DESC
                        LIMIT 1
                        """,
                        (tenant_id, kb_id, doc_id),
                    )
                    row = cur.fetchone()
                if row and row[0]:
                    chunks_uri = row[0]
            except psycopg2.Error:
                # chunks table might not exist yet; fall back to legacy path
                pass
    if not chunks_uri:
        # legacy path fallback
        chunks_object = f"{processed_prefix}{tenant_id}/{kb_id}/{doc_id}/chunks_v1.jsonl"
        chunks_uri = f"gs://{bucket_name}/{chunks_object}"
    logger.info("summarize: chunks_uri=%s", chunks_uri)

    # log start
    with _db_conn(settings) as conn:
        _log_job_event(conn, tenant_id, "step_started", {"step": "summarize_doc", "kb_id": kb_id, "doc_id": doc_id}, job_id=summarize_job_id)

    # Load chunks
    gcs = _gcs_client(settings)
    try:
        chunks_bytes = _gcs_download_bytes(gcs, chunks_uri)
    except FileNotFoundError:
        with _db_conn(settings) as conn:
            _log_job_event(conn, tenant_id, "summarize_failed", {"reason": "chunks_not_found", "gcs_chunks_uri": chunks_uri}, job_id=summarize_job_id)
        raise HTTPException(
            status_code=409,
            detail=f"Chunks not found at {chunks_uri}. Run chunk stage first.",
        )
    logger.info("summarize: chunks_loaded bytes=%s", len(chunks_bytes))

    chunks_fp = _sha256_bytes(chunks_bytes)
    chunks = _parse_chunks_jsonl(chunks_bytes, max_chunks=req.max_chunks)
    if not chunks:
        raise HTTPException(status_code=400, detail="No chunks parsed from chunk artifact")
    logger.info("summarize: chunks_parsed count=%s", len(chunks))

    with _db_conn(settings) as conn:
        _log_job_event(
            conn,
            tenant_id,
            "chunks_loaded",
            {"gcs_chunks_uri": chunks_uri, "chunk_count": len(chunks), "chunks_fingerprint": chunks_fp},
            job_id=summarize_job_id,
        )

    # Gemini client
    vtx = VertexGeminiClient(
        api_key=str(api_key) if api_key else None,
        primary_model=req.primary_model,
        fallback_model=req.fallback_model,
        groq_api_key=str(groq_api_key) if groq_api_key else None,
        groq_model=str(groq_model) if groq_model else None,
    )
    logger.info(
        "summarize: llm_request primary=%s fallback=%s max_chunks=%s",
        req.primary_model,
        req.fallback_model,
        req.max_chunks,
    )

    # params hash
    ph = _params_hash(
        {
            "primary_model": req.primary_model,
            "fallback_model": req.fallback_model,
            "temperature": req.temperature,
            "max_chars_per_chunk": req.max_chars_per_chunk,
            "max_output_tokens_chunk": req.max_output_tokens_chunk,
            "max_output_tokens_doc": req.max_output_tokens_doc,
        }
    )

    # ---- stable cache key for this summarize request (input-based) ----
    summary_cache_key = _sha256_bytes(f"{chunks_fp}:{ph}".encode("utf-8"))
    out_object = f"{processed_prefix}{tenant_id}/{kb_id}/{doc_id}/summary_v1/{summary_cache_key}.json"
    gcs_summary_uri = f"gs://{bucket_name}/{out_object}"

    if _gcs_blob_exists(gcs, bucket_name, out_object):
        logger.info("summarize: cache_hit uri=%s", gcs_summary_uri)
        with _db_conn(settings) as conn:
            _log_job_event(
                conn,
                tenant_id,
                "summary_cache_hit",
                {"gcs_summary_uri": gcs_summary_uri, "summary_fingerprint": summary_cache_key, "params_hash": ph},
                job_id=summarize_job_id,
            )
            _log_job_event(conn, tenant_id, "step_done", {"step": "summarize_doc"}, job_id=summarize_job_id)

        return SummarizeResponse(
            status="ok",
            kb_id=kb_id,
            doc_id=doc_id,
            summarize_job_id=summarize_job_id,
            input_chunks_uri=chunks_uri,
            input_chunks_fingerprint=chunks_fp,
            params_hash=ph,
            chunk_count=len(chunks),
            primary_model=req.primary_model,
            fallback_model=req.fallback_model,
            gcs_summary_uri=gcs_summary_uri,
            summary_fingerprint=summary_cache_key,
            usage={"prompt_tokens": 0, "output_tokens": 0, "total_tokens": 0},
            meta={"provider": "cache", "model": "cached", "cache_hit": True},
        )

    # Summarize chunks (bounded concurrency)
    sem = asyncio.Semaphore(max(1, int(req.max_concurrent_llm)))
    usage_total = VertexUsage()

    chunk_summaries: List[Dict[str, Any]] = []

    async def _one_chunk(c: Chunk) -> Dict[str, Any]:
        text = (c.text or "").strip()
        if len(text) > int(req.max_chars_per_chunk):
            text = text[: int(req.max_chars_per_chunk)] + "\n[TRUNCATED]"
        prompt = _chunk_prompt(c.chunk_id, text)

        async with sem:
            parsed, usage = await asyncio.to_thread(
                vtx.generate_json,
                prompt,
                system=CHUNK_SYSTEM,
                max_output_tokens=int(req.max_output_tokens_chunk),
                temperature=float(req.temperature),
            )

        # accumulate usage (best-effort)
        usage_total.prompt_tokens += usage.prompt_tokens
        usage_total.output_tokens += usage.output_tokens
        usage_total.total_tokens += usage.total_tokens

        # harden required fields
        parsed["chunk_id"] = str(parsed.get("chunk_id") or c.chunk_id)
        parsed.setdefault("title", "")
        parsed.setdefault("summary", "")
        parsed.setdefault("key_points", [])
        parsed.setdefault("tags", [])
        return parsed

    tasks = [_one_chunk(c) for c in chunks]
    for coro in asyncio.as_completed(tasks):
        chunk_summaries.append(await coro)
    logger.info("summarize: chunk_summaries_done count=%s", len(chunk_summaries))

    # keep stable order by chunk_id appearance
    order = {c.chunk_id: i for i, c in enumerate(chunks)}
    chunk_summaries.sort(key=lambda x: order.get(str(x.get("chunk_id")), 10**9))

    with _db_conn(settings) as conn:
        _log_job_event(conn, tenant_id, "chunk_summaries_done", {"count": len(chunk_summaries)}, job_id=summarize_job_id)

    # Document summary + TOC
    doc_prompt = _doc_prompt(chunk_summaries)
    doc_summary, usage_doc = await asyncio.to_thread(
        vtx.generate_json,
        doc_prompt,
        system=DOC_SYSTEM,
        max_output_tokens=int(req.max_output_tokens_doc),
        temperature=float(req.temperature),
    )
    logger.info("summarize: doc_summary_done provider=%s model=%s", vtx.last_provider, vtx.last_model)
    usage_total.prompt_tokens += usage_doc.prompt_tokens
    usage_total.output_tokens += usage_doc.output_tokens
    usage_total.total_tokens += usage_doc.total_tokens

    with _db_conn(settings) as conn:
        _log_job_event(conn, tenant_id, "doc_summary_done", {"has_toc": bool(doc_summary.get("toc"))}, job_id=summarize_job_id)

    # Build final artifact
    artifact = {
        "kb_id": kb_id,
        "doc_id": doc_id,
        "tenant_id": tenant_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "models": {"primary": req.primary_model, "fallback": req.fallback_model},
        "input": {"gcs_chunks_uri": chunks_uri, "chunks_fingerprint": chunks_fp, "chunk_count": len(chunks), "params_hash": ph},
        "chunk_summaries": chunk_summaries,
        "doc_summary": doc_summary,
    }

    out_bytes = json.dumps(artifact, ensure_ascii=False, indent=2).encode("utf-8")
    out_fp = _sha256_bytes(out_bytes)

    # write to GCS
    _ = _gcs_upload_bytes(
        gcs,
        bucket_name=bucket_name,
        object_name=out_object,
        data=out_bytes,
        content_type="application/json; charset=utf-8",
    )
    logger.info("summarize: summary_saved uri=%s", gcs_summary_uri)

    with _db_conn(settings) as conn:
        _log_job_event(
            conn,
            tenant_id,
            "summary_saved",
            {"gcs_summary_uri": gcs_summary_uri, "summary_fingerprint": out_fp, "params_hash": ph},
            job_id=summarize_job_id,
        )
        _log_job_event(conn, tenant_id, "step_done", {"step": "summarize_doc"}, job_id=summarize_job_id)

    return SummarizeResponse(
        status="ok",
        kb_id=kb_id,
        doc_id=doc_id,
        summarize_job_id=summarize_job_id,
        input_chunks_uri=chunks_uri,
        input_chunks_fingerprint=chunks_fp,
        params_hash=ph,
        chunk_count=len(chunks),
        primary_model=req.primary_model,
        fallback_model=req.fallback_model,
        gcs_summary_uri=gcs_summary_uri,
        summary_fingerprint=summary_cache_key,
        usage={
            "prompt_tokens": usage_total.prompt_tokens,
            "output_tokens": usage_total.output_tokens,
            "total_tokens": usage_total.total_tokens,
        },
        meta={
            "provider": vtx.last_provider or "unknown",
            "model": vtx.last_model or "unknown",
            "cache_hit": False,
            "output_fingerprint": out_fp,
        },
    )



