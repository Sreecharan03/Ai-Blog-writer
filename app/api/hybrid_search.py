# app/api/hybrid_search.py
from __future__ import annotations

import hashlib
import json
import os
import re
import time
import uuid
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Tuple

import psycopg2
from fastapi import APIRouter, Depends, Header, HTTPException
from jose import JWTError, jwt
from pydantic import BaseModel, Field

import google.generativeai as genai
from google.cloud import storage


router = APIRouter(prefix="/api/v1/kb", tags=["hybrid-search"])


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
            (str(uuid.uuid4()), tenant_id, None, job_id, event_type, json.dumps(detail)),
        )


def _ensure_retrieval_cache_table(conn) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS public.retrieval_cache (
              cache_key text PRIMARY KEY,
              tenant_id uuid NOT NULL,
              kb_id uuid NOT NULL,
              doc_id uuid NULL,
              created_at timestamptz NOT NULL DEFAULT now(),
              expires_at timestamptz NOT NULL,
              payload jsonb NOT NULL
            );
            """
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_retrieval_cache_lookup ON public.retrieval_cache (tenant_id, kb_id, doc_id, expires_at DESC);"
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
# Hybrid scoring helpers
# ============================================================
_STOP = {
    "the","a","an","and","or","but","to","of","in","on","for","with","at","by","from","as","is","are","was","were",
    "what","does","this","document","talk","about","tell","me","explain","please","give","show"
}

def _tokens(s: str) -> List[str]:
    toks = re.findall(r"[a-z0-9]+", (s or "").lower())
    return [t for t in toks if t and t not in _STOP and len(t) >= 2]

def _keyword_score(query_tokens: List[str], chunk_text: str) -> float:
    if not query_tokens:
        return 0.0
    ctoks = set(re.findall(r"[a-z0-9]+", (chunk_text or "").lower()))
    hit = sum(1 for t in set(query_tokens) if t in ctoks)
    return hit / max(1, len(set(query_tokens)))

def _entity_terms(query: str) -> List[str]:
    # simple heuristic: quoted phrases + CapitalWords
    quoted = re.findall(r'"([^"]+)"', query or "")
    caps = re.findall(r"\b[A-Z][a-zA-Z0-9_]{2,}\b", query or "")
    terms = []
    for q in quoted:
        qt = q.strip()
        if qt:
            terms.append(qt.lower())
    for c in caps:
        terms.append(c.lower())
    # also include longer normal tokens (acts as "entity-ish" keywords)
    terms.extend([t for t in _tokens(query) if len(t) >= 5])
    # dedupe
    out = []
    seen = set()
    for t in terms:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out

def _entity_boost(ent_terms: List[str], chunk_text: str, max_boost: float = 0.12) -> float:
    if not ent_terms:
        return 0.0
    txt = (chunk_text or "").lower()
    hits = sum(1 for e in ent_terms if e in txt)
    # up to max_boost
    return min(max_boost, 0.03 * hits)

def _to_vector_literal(v: List[float]) -> str:
    return "[" + ",".join(repr(float(x)) for x in v) + "]"

def _embed_query(
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
            if isinstance(res, dict):
                if "embedding" in res and isinstance(res["embedding"], dict):
                    v = res["embedding"].get("values") or res["embedding"].get("embedding")
                    if isinstance(v, list) and len(v) == int(output_dimensionality):
                        return [float(x) for x in v]
                if "embedding" in res and isinstance(res["embedding"], list) and len(res["embedding"]) == int(output_dimensionality):
                    return [float(x) for x in res["embedding"]]
                if "embeddings" in res and res["embeddings"]:
                    it = res["embeddings"][0]
                    if isinstance(it, dict):
                        v = it.get("values") or it.get("embedding")
                        if isinstance(v, list) and len(v) == int(output_dimensionality):
                            return [float(x) for x in v]
            last_err = f"bad response shape: {type(res)}"
        except Exception as e:
            last_err = str(e)
        time.sleep(min(2 ** attempt, 10))
    raise HTTPException(status_code=502, detail=f"Embedding failed. Last error: {last_err}")


def _sha256(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


# ============================================================
# Request/Response
# ============================================================
class HybridSearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    doc_id: Optional[str] = None
    top_k: int = 5

    # candidate pool from vector DB (we rerank with keywords/entities)
    candidate_k: int = 40

    # weights
    alpha_vector: float = 0.75
    beta_keyword: float = 0.25

    embedding_model: str = "models/gemini-embedding-001"
    output_dimensionality: int = 1536

    # diversify settings
    diversify: bool = True
    min_chunk_gap: int = 1  # avoid adjacent chunks if possible
    one_per_section: bool = True

    # cache
    use_cache: bool = True
    cache_ttl_s: int = 300

    # optional: if you want to force against a specific chunk artifact
    gcs_chunks_uri: Optional[str] = None
    restrict_to_doc: bool = True


class HybridHit(BaseModel):
    chunk_id: str
    score: float
    vector_score: float
    keyword_score: float
    entity_boost: float
    distance: float
    chunk_index: Optional[int] = None
    section_path: Optional[str] = None
    text: str = ""


class HybridSearchResponse(BaseModel):
    status: str
    kb_id: str
    query: str
    top_k: int
    doc_id: Optional[str] = None
    gcs_chunks_uri: Optional[str] = None
    results: List[HybridHit] = Field(default_factory=list)
    meta: Dict[str, Any] = Field(default_factory=dict)


# ============================================================
# Endpoint
# ============================================================
@router.post("/{kb_id}/hybrid-search", response_model=HybridSearchResponse)
def hybrid_search(
    kb_id: str,
    req: HybridSearchRequest,
    claims: Claims = Depends(require_claims),
):
    """
    Day 15:
    Vector + Keyword + Entity boosting + Diversification + Retrieval Cache (TTL).
    """
    settings = get_settings()
    tenant_id = claims.tenant_id
    job_id = str(uuid.uuid4())

    api_key = _pick(settings, "GEMINI_API_KEY", "GOOGLE_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="Missing GEMINI_API_KEY/GOOGLE_API_KEY")
    genai.configure(api_key=str(api_key))

    top_k = max(1, min(int(req.top_k), 50))
    candidate_k = max(top_k, min(int(req.candidate_k), 200))
    dim = int(req.output_dimensionality)

    # ---------- cache lookup ----------
    cache_key = _sha256(
        json.dumps(
            {
                "tenant_id": tenant_id,
                "kb_id": kb_id,
                "doc_id": req.doc_id if req.restrict_to_doc else None,
                "q": req.query,
                "top_k": top_k,
                "candidate_k": candidate_k,
                "alpha": float(req.alpha_vector),
                "beta": float(req.beta_keyword),
                "model": req.embedding_model,
                "dim": dim,
                "div": bool(req.diversify),
                "gap": int(req.min_chunk_gap),
                "one_per_section": bool(req.one_per_section),
                "gcs_chunks_uri": req.gcs_chunks_uri,
            },
            sort_keys=True,
        )
    )

    if req.use_cache:
        with _db_conn(settings) as conn:
            _ensure_retrieval_cache_table(conn)
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT payload
                    FROM public.retrieval_cache
                    WHERE cache_key=%s
                      AND tenant_id=%s::uuid
                      AND kb_id=%s::uuid
                      AND (doc_id IS NOT DISTINCT FROM %s::uuid)
                      AND expires_at > now()
                    """,
                    (cache_key, tenant_id, kb_id, req.doc_id if req.restrict_to_doc else None),
                )
                row = cur.fetchone()
            if row and row[0]:
                payload = row[0]
                return HybridSearchResponse(**payload)

    with _db_conn(settings) as conn:
        _log_job_event(conn, tenant_id, "hybrid_search_started", {"kb_id": kb_id, "doc_id": req.doc_id, "top_k": top_k}, job_id=job_id)
        conn.commit()

    # ---------- embed query ----------
    q_vec = _embed_query(req.embedding_model, req.query, output_dimensionality=dim)
    q_literal = _to_vector_literal(q_vec)

    # ---------- vector candidates from DB (join to chunks for chunk_index/section_path + gcs uri) ----------
    where_doc = ""
    sql_params: List[Any] = [q_literal, tenant_id, kb_id, dim, req.embedding_model, q_literal, candidate_k]

    if req.doc_id and req.restrict_to_doc:
        where_doc = " AND ce.doc_id=%s::uuid"
        sql_params = [q_literal, tenant_id, kb_id, req.doc_id, dim, req.embedding_model, q_literal, candidate_k]

    sql = f"""
    SELECT ce.chunk_id,
           (ce.embedding <=> %s::vector) AS distance,
           ch.chunk_index,
           ch.section_path,
           ch.gcs_chunks_uri
    FROM public.chunk_embeddings ce
    LEFT JOIN public.chunks ch
      ON ch.tenant_id=ce.tenant_id
     AND ch.kb_id=ce.kb_id
     AND ch.doc_id=ce.doc_id
     AND ch.chunk_id = ce.chunk_id::uuid
    WHERE ce.tenant_id=%s::uuid
      AND ce.kb_id=%s::uuid
      {where_doc}
      AND ce.output_dimensionality=%s
      AND ce.embedding_model=%s
    ORDER BY ce.embedding <=> %s::vector ASC
    LIMIT %s;
    """

    candidates: List[Tuple[str, float, Optional[int], Optional[str], Optional[str]]] = []
    with _db_conn(settings) as conn:
        with conn.cursor() as cur:
            cur.execute(sql, sql_params)
            for r in cur.fetchall():
                candidates.append((str(r[0]), float(r[1]), r[2], r[3], r[4]))
        _log_job_event(conn, tenant_id, "hybrid_vector_candidates", {"candidate_k": candidate_k, "got": len(candidates)}, job_id=job_id)
        conn.commit()

    # ---------- resolve chunks uri + load chunk text map ----------
    chunks_uri = req.gcs_chunks_uri
    if not chunks_uri:
        # prefer from candidates join
        for _cid, _dist, _idx, _sec, guri in candidates:
            if guri:
                chunks_uri = guri
                break

    # fallback: latest chunks uri from chunks table
    if not chunks_uri and req.doc_id:
        with _db_conn(settings) as conn:
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
                row = cur.fetchone()
            if row and row[0]:
                chunks_uri = str(row[0])

    chunk_text_map: Dict[str, str] = {}
    if chunks_uri:
        try:
            gcs = _gcs_client(settings)
            b = _gcs_download_bytes(gcs, chunks_uri)
            chunk_text_map = _parse_chunks_jsonl_to_map(b)
        except Exception:
            chunk_text_map = {}

    # ---------- hybrid rerank ----------
    q_toks = _tokens(req.query)
    ents = _entity_terms(req.query)

    alpha = float(req.alpha_vector)
    beta = float(req.beta_keyword)
    if alpha < 0:
        alpha = 0.0
    if beta < 0:
        beta = 0.0
    if alpha + beta <= 0:
        alpha = 1.0
        beta = 0.0
    # normalize weights
    s = alpha + beta
    alpha /= s
    beta /= s

    scored: List[HybridHit] = []
    for cid, dist, cidx, sec, _guri in candidates:
        text = chunk_text_map.get(cid, "")
        vec_score = 1.0 - dist  # cosine similarity approx
        kw_score = _keyword_score(q_toks, text) if text else 0.0
        e_boost = _entity_boost(ents, text) if text else 0.0
        final = (alpha * vec_score) + (beta * kw_score) + e_boost
        # clamp to [0, 1]
        final = max(0.0, min(1.0, final))
        scored.append(
            HybridHit(
                chunk_id=cid,
                score=float(final),
                vector_score=float(vec_score),
                keyword_score=float(kw_score),
                entity_boost=float(e_boost),
                distance=float(dist),
                chunk_index=int(cidx) if cidx is not None else None,
                section_path=str(sec) if sec is not None else None,
                text=text,
            )
        )

    scored.sort(key=lambda x: x.score, reverse=True)

    # ---------- diversification ----------
    if req.diversify:
        picked: List[HybridHit] = []
        used_sections = set()
        used_idxs: List[int] = []

        for h in scored:
            if len(picked) >= top_k:
                break

            if req.one_per_section and h.section_path:
                if h.section_path in used_sections:
                    continue

            if h.chunk_index is not None and used_idxs and int(req.min_chunk_gap) >= 1:
                # avoid chunks too close to already picked ones
                if any(abs(h.chunk_index - ui) <= int(req.min_chunk_gap) for ui in used_idxs):
                    continue

            picked.append(h)
            if h.section_path:
                used_sections.add(h.section_path)
            if h.chunk_index is not None:
                used_idxs.append(h.chunk_index)

        # if diversification dropped too much, backfill
        if len(picked) < top_k:
            existing = {p.chunk_id for p in picked}
            for h in scored:
                if len(picked) >= top_k:
                    break
                if h.chunk_id in existing:
                    continue
                picked.append(h)

        results = picked[:top_k]
    else:
        results = scored[:top_k]

    payload = HybridSearchResponse(
        status="ok",
        kb_id=kb_id,
        query=req.query,
        top_k=top_k,
        doc_id=req.doc_id if req.restrict_to_doc else None,
        gcs_chunks_uri=chunks_uri,
        results=results,
        meta={
            "job_id": job_id,
            "candidate_k": candidate_k,
            "weights": {"alpha_vector": alpha, "beta_keyword": beta},
            "diversify": bool(req.diversify),
            "cache_key": cache_key,
        },
    ).model_dump()

    # ---------- cache save ----------
    if req.use_cache:
        ttl = max(30, min(int(req.cache_ttl_s), 3600))
        expires_at = datetime.now(timezone.utc) + timedelta(seconds=ttl)
        with _db_conn(settings) as conn:
            _ensure_retrieval_cache_table(conn)
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO public.retrieval_cache (cache_key, tenant_id, kb_id, doc_id, expires_at, payload)
                    VALUES (%s, %s::uuid, %s::uuid, %s::uuid, %s, %s::jsonb)
                    ON CONFLICT (cache_key)
                    DO UPDATE SET expires_at=EXCLUDED.expires_at, payload=EXCLUDED.payload
                    """,
                    (cache_key, tenant_id, kb_id, req.doc_id if req.restrict_to_doc else None, expires_at, json.dumps(payload)),
                )
            _log_job_event(conn, tenant_id, "hybrid_search_cached", {"ttl_s": ttl}, job_id=job_id)
            conn.commit()

    with _db_conn(settings) as conn:
        _log_job_event(conn, tenant_id, "hybrid_search_done", {"hits": len(results)}, job_id=job_id)
        conn.commit()

    return HybridSearchResponse(**payload)