# app/api/embed.py
from __future__ import annotations

import hashlib
import json
import os
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import psycopg2
from psycopg2.extras import execute_values

from fastapi import APIRouter, Depends, Header, HTTPException
from jose import JWTError, jwt
from pydantic import BaseModel, Field

from google.cloud import storage
from google.api_core.exceptions import PreconditionFailed

import google.generativeai as genai


# ============================================================
# Router
# ============================================================
router = APIRouter(prefix="/api/v1/kb", tags=["embeddings"])


# ============================================================
# Settings loader (same pattern as your other files)
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
# DB helpers (Supabase Postgres via DB_*)
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


# ============================================================
# Chunk parsing (reads your chunks JSONL format)
# ============================================================
@dataclass
class Chunk:
    chunk_id: str
    text: str


def _parse_chunks_jsonl(jsonl_bytes: bytes, max_chunks: Optional[int] = None) -> List[Chunk]:
    out: List[Chunk] = []
    for line in jsonl_bytes.decode("utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue

        cid = obj.get("chunk_id") or obj.get("id")
        txt = obj.get("text") or obj.get("content") or ""
        if not cid or not isinstance(txt, str):
            continue

        out.append(Chunk(chunk_id=str(cid), text=txt))
        if max_chunks and len(out) >= max_chunks:
            break
    return out


# ============================================================
# Embedding helpers (google.generativeai)
# ============================================================
def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _params_hash(obj: Dict[str, Any]) -> str:
    raw = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return _sha256_bytes(raw)


def _to_vector_literal(v: List[float]) -> str:
    # pgvector accepts: '[1,2,3]'
    # Use repr to preserve precision reasonably.
    return "[" + ",".join(repr(float(x)) for x in v) + "]"


def _extract_embedding_vec(item: Any) -> List[float]:
    # item may be {"embedding":[...]} or {"values":[...]} depending on SDK shape
    if isinstance(item, dict):
        if isinstance(item.get("embedding"), list):
            return [float(x) for x in item["embedding"]]
        if isinstance(item.get("values"), list):
            return [float(x) for x in item["values"]]
    # sometimes it's already the list
    if isinstance(item, list):
        return [float(x) for x in item]
    return []


def _embed_batch(
    model: str,
    texts: List[str],
    *,
    output_dimensionality: int,
    retries: int = 4,
    sleep_base: float = 1.0,
) -> List[List[float]]:
    last_err: Optional[str] = None

    for attempt in range(retries + 1):
        try:
            result = genai.embed_content(
                model=model,
                content=texts,
                output_dimensionality=int(output_dimensionality),
            )
        except Exception as e:
            last_err = str(e)
            time.sleep(min(sleep_base * (2 ** attempt), 10))
            continue

        # normalize embeddings list
        embs: List[Any] = []
        if isinstance(result, dict):
            if "embeddings" in result and isinstance(result.get("embeddings"), list):
                embs = result["embeddings"]
            elif "embedding" in result:
                embs = [result["embedding"]]

        # If API returned unexpected shape, fall back to per-text calls
        if len(embs) != len(texts):
            embs = []
            for t in texts:
                r = genai.embed_content(
                    model=model,
                    content=t,
                    output_dimensionality=int(output_dimensionality),
                )
                if isinstance(r, dict) and "embedding" in r:
                    embs.append(r["embedding"])
                elif isinstance(r, dict) and "embeddings" in r and r["embeddings"]:
                    embs.append(r["embeddings"][0])

        vectors: List[List[float]] = []
        for it in embs:
            vec = _extract_embedding_vec(it)
            vectors.append(vec)

        if vectors and all(len(v) == int(output_dimensionality) for v in vectors):
            return vectors

        last_err = f"Bad embedding shape/dim. got lens={[len(v) for v in vectors[:5]]} expected={output_dimensionality}"
        time.sleep(min(sleep_base * (2 ** attempt), 10))

    raise HTTPException(status_code=502, detail=f"Embedding API failed. Last error: {last_err}")


# ============================================================
# Request/Response
# ============================================================
class EmbedRequest(BaseModel):
    gcs_chunks_uri: Optional[str] = None
    max_chunks: int = 500
    batch_size: int = 32

    embedding_model: str = "models/gemini-embedding-001"
    output_dimensionality: int = 1536

    # If True, we will embed even if DB already has rows for same chunks_fingerprint+params_hash
    force: bool = False


class EmbedResponse(BaseModel):
    status: str
    kb_id: str
    doc_id: str
    embed_job_id: str

    gcs_chunks_uri: str
    chunks_fingerprint: str
    params_hash: str

    chunk_count: int
    embedded_count: int

    embedding_model: str
    output_dimensionality: int

    meta: Dict[str, Any] = Field(default_factory=dict)


# ============================================================
# Endpoint
# ============================================================
@router.post("/{kb_id}/embed/{doc_id}", response_model=EmbedResponse)
def embed_doc(
    kb_id: str,
    doc_id: str,
    req: EmbedRequest,
    claims: Claims = Depends(require_claims),
):
    """
    Day 14.1:
    Read chunks JSONL from GCS -> generate Gemini embeddings (1536) -> upsert into public.chunk_embeddings.
    """
    settings = get_settings()
    tenant_id = claims.tenant_id
    embed_job_id = str(uuid.uuid4())

    # LLM key
    api_key = _pick(settings, "GEMINI_API_KEY", "GOOGLE_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="Missing GEMINI_API_KEY/GOOGLE_API_KEY")

    # Configure GenAI client
    genai.configure(api_key=str(api_key))

    # Determine chunks uri
    chunks_uri = req.gcs_chunks_uri
    if not chunks_uri:
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
                chunks_uri = None

    if not chunks_uri:
        raise HTTPException(status_code=409, detail="gcs_chunks_uri not provided and no chunks found in DB. Run chunk stage first.")

    # log start
    with _db_conn(settings) as conn:
        _log_job_event(conn, tenant_id, "embed_started", {"kb_id": kb_id, "doc_id": doc_id, "gcs_chunks_uri": chunks_uri}, job_id=embed_job_id)
        conn.commit()

    # Load chunks
    gcs = _gcs_client(settings)
    try:
        chunks_bytes = _gcs_download_bytes(gcs, chunks_uri)
    except FileNotFoundError:
        with _db_conn(settings) as conn:
            _log_job_event(conn, tenant_id, "embed_failed", {"reason": "chunks_not_found", "gcs_chunks_uri": chunks_uri}, job_id=embed_job_id)
            conn.commit()
        raise HTTPException(status_code=409, detail=f"Chunks not found at {chunks_uri}. Run chunk stage first.")

    chunks_fp = _sha256_bytes(chunks_bytes)
    chunks = _parse_chunks_jsonl(chunks_bytes, max_chunks=req.max_chunks)
    if not chunks:
        raise HTTPException(status_code=400, detail="No chunks parsed from chunk artifact")

    params_hash = _params_hash(
        {
            "embedding_model": req.embedding_model,
            "output_dimensionality": int(req.output_dimensionality),
        }
    )

    # Optional: skip if already embedded (best-effort)
    if not req.force:
        try:
            with _db_conn(settings) as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT COUNT(*)
                        FROM public.chunk_embeddings
                        WHERE tenant_id=%s::uuid AND kb_id=%s::uuid AND doc_id=%s::uuid
                          AND chunks_fingerprint=%s AND params_hash=%s
                        """,
                        (tenant_id, kb_id, doc_id, chunks_fp, params_hash),
                    )
                    already = int(cur.fetchone()[0] or 0)
            if already >= len(chunks) and already > 0:
                with _db_conn(settings) as conn:
                    _log_job_event(conn, tenant_id, "embed_cache_hit", {"already": already, "chunk_count": len(chunks)}, job_id=embed_job_id)
                    _log_job_event(conn, tenant_id, "embed_done", {"embedded_count": 0, "skipped": True}, job_id=embed_job_id)
                    conn.commit()

                return EmbedResponse(
                    status="ok",
                    kb_id=kb_id,
                    doc_id=doc_id,
                    embed_job_id=embed_job_id,
                    gcs_chunks_uri=chunks_uri,
                    chunks_fingerprint=chunks_fp,
                    params_hash=params_hash,
                    chunk_count=len(chunks),
                    embedded_count=0,
                    embedding_model=req.embedding_model,
                    output_dimensionality=int(req.output_dimensionality),
                    meta={"cache_hit": True, "already": already},
                )
        except Exception:
            # if schema differs (columns missing), ignore skip check and continue
            pass

    # Generate embeddings in batches
    texts = [c.text for c in chunks]
    batch_size = max(1, int(req.batch_size))
    dim = int(req.output_dimensionality)

    vectors_all: List[List[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        vecs = _embed_batch(req.embedding_model, batch, output_dimensionality=dim)
        vectors_all.extend(vecs)

    if len(vectors_all) != len(chunks):
        raise HTTPException(status_code=502, detail=f"Embedding count mismatch. chunks={len(chunks)} embeddings={len(vectors_all)}")

    # Upsert into DB
    rows = []
    for c, vec in zip(chunks, vectors_all):
        if len(vec) != dim:
            raise HTTPException(status_code=502, detail=f"Embedding dim mismatch for chunk_id={c.chunk_id}. got={len(vec)} expected={dim}")
        rows.append(
            (
                tenant_id,
                kb_id,
                doc_id,
                c.chunk_id,
                _to_vector_literal(vec),
                req.embedding_model,
                dim,
                chunks_fp,
                params_hash,
            )
        )

    with _db_conn(settings) as conn:
        with conn.cursor() as cur:
            sql = """
            INSERT INTO public.chunk_embeddings (
              tenant_id, kb_id, doc_id, chunk_id,
              embedding, embedding_model, output_dimensionality,
              chunks_fingerprint, params_hash
            )
            VALUES %s
            ON CONFLICT (tenant_id, kb_id, doc_id, chunk_id)
            DO UPDATE SET
              embedding = EXCLUDED.embedding,
              embedding_model = EXCLUDED.embedding_model,
              output_dimensionality = EXCLUDED.output_dimensionality,
              chunks_fingerprint = EXCLUDED.chunks_fingerprint,
              params_hash = EXCLUDED.params_hash,
              created_at = now()
            """
            template = "(%s::uuid,%s::uuid,%s::uuid,%s,%s::vector,%s,%s,%s,%s)"
            execute_values(cur, sql, rows, template=template, page_size=200)

        _log_job_event(conn, tenant_id, "embed_upserted", {"count": len(rows), "dim": dim, "params_hash": params_hash}, job_id=embed_job_id)
        _log_job_event(conn, tenant_id, "embed_done", {"embedded_count": len(rows)}, job_id=embed_job_id)
        conn.commit()

    return EmbedResponse(
        status="ok",
        kb_id=kb_id,
        doc_id=doc_id,
        embed_job_id=embed_job_id,
        gcs_chunks_uri=chunks_uri,
        chunks_fingerprint=chunks_fp,
        params_hash=params_hash,
        chunk_count=len(chunks),
        embedded_count=len(rows),
        embedding_model=req.embedding_model,
        output_dimensionality=dim,
        meta={"cache_hit": False},
    )