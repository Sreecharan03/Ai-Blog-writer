# app/api/search.py
from __future__ import annotations

import hashlib
import json
import math
import os
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import psycopg2
from fastapi import APIRouter, Depends, Header, HTTPException
from jose import JWTError, jwt
from pydantic import BaseModel, Field

from google.cloud import storage
import google.generativeai as genai


router = APIRouter(prefix="/api/v1/kb", tags=["search"])


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
# DB helpers (Supabase via DB_*)
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
# Chunk parsing (map chunk_id -> text)
# ============================================================
def _parse_chunks_jsonl_to_map(jsonl_bytes: bytes) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for line in jsonl_bytes.decode("utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        cid = obj.get("chunk_id") or obj.get("id")
        txt = obj.get("text") or obj.get("content")
        if cid and isinstance(txt, str):
            out[str(cid)] = txt
    return out


# ============================================================
# Embedding helpers (google.generativeai)
# ============================================================
def _to_vector_literal(v: List[float]) -> str:
    return "[" + ",".join(repr(float(x)) for x in v) + "]"


def _extract_embedding_vec(item: Any) -> List[float]:
    # item may be {"embedding":[...]} or {"values":[...]}
    if isinstance(item, dict):
        if isinstance(item.get("embedding"), list):
            return [float(x) for x in item["embedding"]]
        if isinstance(item.get("values"), list):
            return [float(x) for x in item["values"]]
    if isinstance(item, list):
        return [float(x) for x in item]
    return []


def _embed_query_once(
    model: str,
    text: str,
    *,
    output_dimensionality: int,
    retries: int = 4,
) -> List[float]:
    last_err: Optional[str] = None
    for attempt in range(retries + 1):
        try:
            res = genai.embed_content(
                model=model,
                content=text,
                output_dimensionality=int(output_dimensionality),
            )
            # dict shape
            if isinstance(res, dict):
                if "embedding" in res:
                    v = _extract_embedding_vec(res["embedding"])
                    if len(v) == int(output_dimensionality):
                        return v
                if "embeddings" in res and res["embeddings"]:
                    v = _extract_embedding_vec(res["embeddings"][0])
                    if len(v) == int(output_dimensionality):
                        return v
            last_err = f"Bad embedding response shape: {type(res)}"
        except Exception as e:
            last_err = str(e)

        time.sleep(min(2 ** attempt, 10))

    raise HTTPException(status_code=502, detail=f"Embedding query failed. Last error: {last_err}")


# ============================================================
# Request/Response
# ============================================================
class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)

    # if doc_id is given, we also attach chunk text by loading that doc's chunks jsonl from GCS
    doc_id: Optional[str] = None

    # optionally override chunks uri (if you want to search against a specific artifact)
    gcs_chunks_uri: Optional[str] = None

    top_k: int = 5

    embedding_model: str = "models/gemini-embedding-001"
    output_dimensionality: int = 1536

    # filters
    restrict_to_doc: bool = True  # if doc_id given, default to only search within that doc


class SearchHit(BaseModel):
    chunk_id: str
    score: float
    distance: float
    text: str = ""


class SearchResponse(BaseModel):
    status: str
    kb_id: str
    query: str
    top_k: int
    doc_id: Optional[str] = None

    embedding_model: str
    output_dimensionality: int

    gcs_chunks_uri: Optional[str] = None

    results: List[SearchHit] = Field(default_factory=list)
    meta: Dict[str, Any] = Field(default_factory=dict)


# ============================================================
# Endpoint
# ============================================================
@router.post("/{kb_id}/search", response_model=SearchResponse)
def search_kb(
    kb_id: str,
    req: SearchRequest,
    claims: Claims = Depends(require_claims),
):
    """
    Day 14.2:
    - Embed the query (1536)
    - pgvector search in public.chunk_embeddings using cosine distance (<=>)
    - Return chunk_id + score + distance + chunk text (from chunks JSONL)
    """
    settings = get_settings()
    tenant_id = claims.tenant_id
    job_id = str(uuid.uuid4())

    api_key = _pick(settings, "GEMINI_API_KEY", "GOOGLE_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="Missing GEMINI_API_KEY/GOOGLE_API_KEY")
    genai.configure(api_key=str(api_key))

    top_k = max(1, min(int(req.top_k), 50))
    dim = int(req.output_dimensionality)

    # 1) Embed query
    q_vec = _embed_query_once(req.embedding_model, req.query, output_dimensionality=dim)
    q_literal = _to_vector_literal(q_vec)

    # 2) Query DB for nearest chunks
    with _db_conn(settings) as conn:
        _log_job_event(conn, tenant_id, "search_started", {"kb_id": kb_id, "doc_id": req.doc_id, "top_k": top_k}, job_id=job_id)
        conn.commit()

    where_doc = ""
    params: List[Any] = [tenant_id, kb_id, dim, req.embedding_model, q_literal, q_literal, top_k]
    # SQL uses q_literal twice: once for distance select, once for order-by
    if req.doc_id and req.restrict_to_doc:
        where_doc = " AND doc_id=%s::uuid"
        # insert doc_id after kb_id
        params = [tenant_id, kb_id, req.doc_id, dim, req.embedding_model, q_literal, q_literal, top_k]

    sql = f"""
    SELECT chunk_id,
           (embedding <=> %s::vector) AS distance
    FROM public.chunk_embeddings
    WHERE tenant_id=%s::uuid
      AND kb_id=%s::uuid
      {where_doc}
      AND output_dimensionality=%s
      AND embedding_model=%s
    ORDER BY embedding <=> %s::vector ASC
    LIMIT %s;
    """

    # adjust parameter ordering to match SQL (distance first)
    if req.doc_id and req.restrict_to_doc:
        # distance, tenant, kb, doc, dim, model, order_distance, limit
        sql_params = [q_literal, tenant_id, kb_id, req.doc_id, dim, req.embedding_model, q_literal, top_k]
    else:
        sql_params = [q_literal, tenant_id, kb_id, dim, req.embedding_model, q_literal, top_k]

    rows: List[Tuple[str, float]] = []
    with _db_conn(settings) as conn:
        with conn.cursor() as cur:
            cur.execute(sql, sql_params)
            rows = [(str(r[0]), float(r[1])) for r in cur.fetchall()]
        _log_job_event(conn, tenant_id, "search_db_done", {"returned": len(rows)}, job_id=job_id)
        conn.commit()

    # 3) Load chunk text map (optional but requested)
    chunks_uri = req.gcs_chunks_uri
    chunk_text_map: Dict[str, str] = {}
    if not chunks_uri and req.doc_id:
        # resolve latest chunks uri from public.chunks
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
                        (tenant_id, kb_id, req.doc_id),
                    )
                    r = cur.fetchone()
                if r and r[0]:
                    chunks_uri = str(r[0])
            except Exception:
                chunks_uri = None

    if chunks_uri:
        try:
            gcs = _gcs_client(settings)
            b = _gcs_download_bytes(gcs, chunks_uri)
            chunk_text_map = _parse_chunks_jsonl_to_map(b)
        except Exception:
            chunk_text_map = {}

    # 4) Build results with score + text
    # distance is cosine distance; convert to similarity score = 1 - distance
    results: List[SearchHit] = []
    for cid, dist in rows:
        score = 1.0 - dist
        # clamp
        if score < -1.0:
            score = -1.0
        if score > 1.0:
            score = 1.0
        results.append(
            SearchHit(
                chunk_id=cid,
                distance=float(dist),
                score=float(score),
                text=chunk_text_map.get(cid, ""),
            )
        )

    with _db_conn(settings) as conn:
        _log_job_event(conn, tenant_id, "search_done", {"hits": len(results), "gcs_chunks_uri": chunks_uri}, job_id=job_id)
        conn.commit()

    return SearchResponse(
        status="ok",
        kb_id=kb_id,
        query=req.query,
        top_k=top_k,
        doc_id=req.doc_id,
        embedding_model=req.embedding_model,
        output_dimensionality=dim,
        gcs_chunks_uri=chunks_uri,
        results=results,
        meta={"job_id": job_id, "restricted_to_doc": bool(req.doc_id and req.restrict_to_doc)},
    )