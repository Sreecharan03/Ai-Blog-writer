# app/api/v1/chunk.py
from __future__ import annotations

import hashlib
import json
import os
import re
import unicodedata
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import psycopg2
from psycopg2.extras import Json, RealDictCursor
from fastapi import APIRouter, Depends, Header, HTTPException
from jose import JWTError, jwt
from pydantic import BaseModel, Field

from google.cloud import storage
from google.api_core.exceptions import PreconditionFailed


# ---------------------------
# Settings loader (same pattern as ingest.py)
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
# DB
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


def _ensure_chunks_table(conn) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS public.chunks (
              chunk_id uuid PRIMARY KEY,
              tenant_id uuid NOT NULL,
              kb_id uuid NOT NULL,
              doc_id uuid NOT NULL,
              chunk_index integer NOT NULL,
              section_path text,
              start_char integer,
              end_char integer,
              chunk_fingerprint text,
              chunk_chars integer,
              params_hash text NOT NULL,
              input_fingerprint text NOT NULL,
              input_kind text NOT NULL,
              gcs_chunks_uri text NOT NULL,
              gcs_manifest_uri text NOT NULL,
              created_at timestamptz NOT NULL DEFAULT now()
            );
            """
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_chunks_doc ON public.chunks (tenant_id, kb_id, doc_id, created_at DESC);"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_chunks_params ON public.chunks (tenant_id, kb_id, doc_id, params_hash, input_fingerprint);"
        )
        cur.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS uq_chunks_dedup
            ON public.chunks (tenant_id, kb_id, doc_id, params_hash, input_fingerprint, chunk_index);
            """
        )


def _log_job_event(conn, tenant_id: str, event_type: str, detail: Dict[str, Any], job_id: Optional[str]) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO public.job_events (event_id, tenant_id, request_id, job_id, event_type, detail)
            VALUES (%s, %s, %s, %s, %s, %s)
            """,
            (str(uuid.uuid4()), tenant_id, None, job_id, event_type, Json(detail)),
        )


# ---------------------------
# GCS
# ---------------------------
def _norm_prefix(p: str) -> str:
    p = (p or "").strip().replace("\\", "/")
    if not p:
        return ""
    return p if p.endswith("/") else (p + "/")


def _gcs_client(settings: Any) -> storage.Client:
    project_id = _pick(settings, "GCP_PROJECT_ID", "GOOGLE_CLOUD_PROJECT")
    return storage.Client(project=project_id)


def _parse_gs_uri(uri: str) -> Tuple[str, str]:
    if not uri.startswith("gs://"):
        raise ValueError("Not a gs:// URI")
    rest = uri[5:]
    parts = rest.split("/", 1)
    if len(parts) != 2:
        raise ValueError("Invalid gs:// URI")
    return parts[0], parts[1]


def _gcs_download_text(client: storage.Client, uri: str) -> str:
    bucket, obj = _parse_gs_uri(uri)
    b = client.bucket(bucket)
    blob = b.blob(obj)
    data = blob.download_as_bytes()
    return data.decode("utf-8", errors="replace")


def _gcs_upload_text(client: storage.Client, bucket: str, obj: str, text: str, content_type: str = "text/plain; charset=utf-8") -> str:
    b = client.bucket(bucket)
    blob = b.blob(obj)
    try:
        # Create-only (no overwrite => no delete permission needed)
        blob.upload_from_string(text.encode("utf-8"), content_type=content_type, if_generation_match=0)
    except PreconditionFailed:
        # Object already exists => treat as cache hit / reuse
        pass
    return f"gs://{bucket}/{obj}"


def _sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _normalize_text(t: str) -> str:
    t = unicodedata.normalize("NFKC", t)
    t = t.replace("\r\n", "\n").replace("\r", "\n")
    return t


# ---------------------------
# Chunking
# ---------------------------
@dataclass
class Unit:
    start: int
    end: int
    text: str
    section_path: str


_HEADING_MD = re.compile(r"^\s{0,3}(#{1,6})\s+(.+?)\s*$")
_BULLET = re.compile(r"^\s*([-*•]|\d+[.)])\s+\S+")


def _looks_like_heading(line: str) -> Optional[Tuple[int, str]]:
    s = line.strip()
    if not s:
        return None

    m = _HEADING_MD.match(s)
    if m:
        level = len(m.group(1))
        title = m.group(2).strip()
        return level, title

    # heuristics for doc-like headings
    if _BULLET.match(s):
        return None

    if len(s) <= 90 and s.endswith(":"):
        return 2, s[:-1].strip()

    # ALL CAPS-ish line
    if len(s) <= 80 and re.fullmatch(r"[A-Z0-9][A-Z0-9 \-–—]{6,}", s or ""):
        return 2, s.strip()

    # Title Case-ish short line
    if len(s) <= 70 and sum(ch.isalpha() for ch in s) >= 6:
        words = s.split()
        if 2 <= len(words) <= 10:
            titleish = sum(w[:1].isupper() for w in words if w[:1].isalpha()) >= max(2, len(words) // 2)
            if titleish and not s.endswith("."):
                return 3, s.strip()

    return None


def _build_units(text: str) -> List[Unit]:
    units: List[Unit] = []
    stack: List[Tuple[int, str]] = []

    # paragraph iterator with offsets
    for m in re.finditer(r"(?s)(.*?)(\n{2,}|$)", text):
        para = m.group(1)
        if para is None:
            continue
        raw = para
        start = m.start(1)
        end = m.end(1)
        if start == end:
            continue

        stripped = raw.strip()
        if not stripped:
            continue

        # detect headings by first non-empty line
        first_line = stripped.split("\n", 1)[0].strip()
        h = _looks_like_heading(first_line)
        if h:
            level, title = h
            # maintain heading stack
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title))

        section_path = " > ".join([t for _, t in stack]) if stack else ""
        units.append(Unit(start=start, end=end, text=text[start:end], section_path=section_path))

    return units


def _split_long_unit(u: Unit, chunk_size: int, overlap: int) -> List[Unit]:
    # sliding split inside a single big unit, preserving offsets
    out: List[Unit] = []
    s = u.text
    n = len(s)
    if n <= chunk_size:
        return [u]

    step = max(1, chunk_size - max(0, overlap))
    i = 0
    idx = 0
    while i < n:
        j = min(n, i + chunk_size)
        seg = s[i:j]
        out.append(Unit(start=u.start + i, end=u.start + j, text=seg, section_path=u.section_path))
        idx += 1
        if j >= n:
            break
        i += step
    return out


def chunk_text_dynamic(
    text: str,
    chunk_size_chars: int,
    overlap_chars: int,
    max_chunks: int,
) -> List[Dict[str, Any]]:
    text = _normalize_text(text)
    units = _build_units(text)

    # split oversized units
    expanded: List[Unit] = []
    for u in units:
        if (u.end - u.start) > chunk_size_chars:
            expanded.extend(_split_long_unit(u, chunk_size_chars, overlap_chars))
        else:
            expanded.append(u)

    units = expanded
    chunks: List[Dict[str, Any]] = []

    cur: List[Unit] = []
    cur_len = 0

    def flush():
        nonlocal cur, cur_len
        if not cur:
            return
        start = cur[0].start
        end = cur[-1].end
        chunk_txt = text[start:end]
        section_path = cur[0].section_path
        chunks.append(
            {
                "start_char": start,
                "end_char": end,
                "section_path": section_path,
                "text": chunk_txt,
            }
        )

        # build overlap carry by unit tail
        if overlap_chars > 0:
            carry: List[Unit] = []
            carry_len = 0
            for u in reversed(cur):
                carry.insert(0, u)
                carry_len += (u.end - u.start)
                if carry_len >= overlap_chars:
                    break
            cur = carry
            cur_len = sum((u.end - u.start) for u in cur)
        else:
            cur = []
            cur_len = 0

    for u in units:
        u_len = (u.end - u.start)
        if u_len <= 0:
            continue

        # start new chunk if adding exceeds limit
        if cur and (cur_len + u_len) > chunk_size_chars:
            flush()
            if len(chunks) >= max_chunks:
                break

        cur.append(u)
        cur_len += u_len

    if len(chunks) < max_chunks:
        flush()

    # drop empty text chunks
    chunks = [c for c in chunks if c["text"].strip()]
    return chunks


# ---------------------------
# API
# ---------------------------
router = APIRouter(prefix="/api/v1/kb", tags=["chunk"])


class ChunkRequest(BaseModel):
    chunk_size_chars: int = Field(default=2000, ge=300, le=20000)
    overlap_chars: int = Field(default=200, ge=0, le=5000)
    max_chunks: int = Field(default=5000, ge=1, le=200000)
    clean_version: str = Field(default="v1")
    prefer_clean: bool = Field(default=True)


class ChunkResponse(BaseModel):
    status: str
    kb_id: str
    doc_id: str
    chunk_job_id: str
    input_kind: str
    input_fingerprint: str
    params_hash: str
    chunk_count: int
    gcs_chunks_uri: str
    gcs_manifest_uri: str
    stats: Dict[str, Any] = Field(default_factory=dict)


@router.post("/{kb_id}/chunk/{doc_id}", response_model=ChunkResponse)
async def chunk_doc(
    kb_id: str,
    doc_id: str,
    req: ChunkRequest,
    claims: Claims = Depends(require_claims),
):
    # validate UUIDs
    try:
        _ = uuid.UUID(kb_id)
        _ = uuid.UUID(doc_id)
    except Exception:
        raise HTTPException(status_code=400, detail="kb_id/doc_id must be valid UUIDs")

    settings = get_settings()
    tenant_id = claims.tenant_id
    chunk_job_id = str(uuid.uuid4())

    bucket_name = _pick(settings, "GCS_BUCKET_NAME")
    if not bucket_name:
        raise HTTPException(status_code=500, detail="Missing GCS_BUCKET_NAME")

    processed_prefix = _norm_prefix(_pick(settings, "GCS_PROCESSED_PREFIX", default="processed"))

    client = _gcs_client(settings)

    # Decide input text source
    input_kind = "clean"
    input_uri: Optional[str] = None
    if req.prefer_clean:
        with _db_conn(settings) as conn:
            try:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(
                        """
                        SELECT gcs_clean_uri
                        FROM public.preprocess_outputs
                        WHERE tenant_id=%s::uuid AND kb_id=%s::uuid AND doc_id=%s::uuid
                          AND preprocessing_version=%s
                        LIMIT 1
                        """,
                        (tenant_id, kb_id, doc_id, req.clean_version),
                    )
                    row = cur.fetchone()
                if row and row.get("gcs_clean_uri"):
                    input_uri = row["gcs_clean_uri"]
                    input_kind = f"clean_{req.clean_version}"
            except psycopg2.Error:
                # preprocess_outputs may not exist yet; fall back to extracted
                pass

    if not input_uri:
        # fallback to extracted uri from documents table
        with _db_conn(settings) as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT gcs_extracted_uri
                    FROM public.documents
                    WHERE tenant_id=%s::uuid AND kb_id=%s::uuid AND doc_id=%s::uuid
                    LIMIT 1
                    """,
                    (tenant_id, kb_id, doc_id),
                )
                row = cur.fetchone()
        if not row or not row.get("gcs_extracted_uri"):
            raise HTTPException(status_code=404, detail="No clean text found and documents.gcs_extracted_uri is missing")
        input_uri = row["gcs_extracted_uri"]
        input_kind = "extracted"

    text = _gcs_download_text(client, input_uri)
    text = _normalize_text(text)
    if not text.strip():
        raise HTTPException(status_code=422, detail=f"Input text is empty (source={input_kind})")

    input_fingerprint = _sha256_text(text)

    params = req.model_dump()
    params_hash = hashlib.sha256(json.dumps(params, sort_keys=True).encode("utf-8")).hexdigest()

    # idempotency: if already chunked for same input + params, return existing
    with _db_conn(settings) as conn:
        _ensure_chunks_table(conn)
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT gcs_chunks_uri, gcs_manifest_uri, COUNT(*) AS n
                FROM public.chunks
                WHERE tenant_id=%s::uuid AND kb_id=%s::uuid AND doc_id=%s::uuid
                  AND params_hash=%s AND input_fingerprint=%s
                GROUP BY gcs_chunks_uri, gcs_manifest_uri
                ORDER BY n DESC
                LIMIT 1
                """,
                (tenant_id, kb_id, doc_id, params_hash, input_fingerprint),
            )
            hit = cur.fetchone()

        if hit:
            # log + return
            try:
                _log_job_event(conn, tenant_id, "chunk_cache_hit", {"doc_id": doc_id, "params_hash": params_hash}, job_id=chunk_job_id)
            except Exception:
                pass

            return ChunkResponse(
                status="ok",
                kb_id=kb_id,
                doc_id=doc_id,
                chunk_job_id=chunk_job_id,
                input_kind=input_kind,
                input_fingerprint=input_fingerprint,
                params_hash=params_hash,
                chunk_count=int(hit["n"]),
                gcs_chunks_uri=hit["gcs_chunks_uri"],
                gcs_manifest_uri=hit["gcs_manifest_uri"],
                stats={"note": "cache_hit"},
            )

    # chunk now
    chunks = chunk_text_dynamic(
        text=text,
        chunk_size_chars=req.chunk_size_chars,
        overlap_chars=req.overlap_chars,
        max_chunks=req.max_chunks,
    )

    if not chunks:
        raise HTTPException(status_code=422, detail="Chunker produced 0 chunks (check input text)")

    # write artifacts to GCS (versioned, create-only)
    chunk_key = f"{input_fingerprint}_{params_hash}"
    chunks_obj = f"{processed_prefix}{tenant_id}/{kb_id}/{doc_id}/chunks_{req.clean_version}/{chunk_key}.jsonl"
    manifest_obj = f"{processed_prefix}{tenant_id}/{kb_id}/{doc_id}/chunks_{req.clean_version}/{chunk_key}.manifest.json"

    # JSONL
    lines: List[str] = []
    for i, c in enumerate(chunks):
        chunk_id = str(uuid.uuid4())
        payload = {
            "chunk_id": chunk_id,
            "tenant_id": tenant_id,
            "kb_id": kb_id,
            "doc_id": doc_id,
            "chunk_index": i,
            "section_path": c.get("section_path", ""),
            "start_char": c["start_char"],
            "end_char": c["end_char"],
            "text": c["text"],
        }
        lines.append(json.dumps(payload, ensure_ascii=False))

    gcs_chunks_uri = _gcs_upload_text(client, bucket_name, chunks_obj, "\n".join(lines), content_type="application/x-ndjson; charset=utf-8")

    manifest = {
        "chunk_job_id": chunk_job_id,
        "tenant_id": tenant_id,
        "kb_id": kb_id,
        "doc_id": doc_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input_uri": input_uri,
        "input_kind": input_kind,
        "input_fingerprint": input_fingerprint,
        "params": params,
        "params_hash": params_hash,
        "chunk_count": len(chunks),
    }
    gcs_manifest_uri = _gcs_upload_text(client, bucket_name, manifest_obj, json.dumps(manifest, ensure_ascii=False, indent=2), content_type="application/json; charset=utf-8")

    # persist chunk metadata
    with _db_conn(settings) as conn:
        _ensure_chunks_table(conn)
        try:
            _log_job_event(conn, tenant_id, "chunk_started", {"doc_id": doc_id, "input_kind": input_kind}, job_id=chunk_job_id)
        except Exception:
            pass

        with conn.cursor() as cur:
            for i, c in enumerate(chunks):
                txt = c["text"]
                cur.execute(
                    """
                    INSERT INTO public.chunks
                      (chunk_id, tenant_id, kb_id, doc_id, chunk_index, section_path, start_char, end_char,
                       chunk_fingerprint, chunk_chars, params_hash, input_fingerprint, input_kind,
                       gcs_chunks_uri, gcs_manifest_uri)
                    VALUES
                      (%s, %s::uuid, %s::uuid, %s::uuid, %s, %s, %s, %s,
                       %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT DO NOTHING
                    """,
                    (
                        str(uuid.uuid4()),
                        tenant_id,
                        kb_id,
                        doc_id,
                        i,
                        c.get("section_path", ""),
                        int(c["start_char"]),
                        int(c["end_char"]),
                        hashlib.sha256(txt.encode("utf-8")).hexdigest(),
                        len(txt),
                        params_hash,
                        input_fingerprint,
                        input_kind,
                        gcs_chunks_uri,
                        gcs_manifest_uri,
                    ),
                )

        try:
            _log_job_event(conn, tenant_id, "chunks_saved", {"count": len(chunks), "gcs_chunks_uri": gcs_chunks_uri}, job_id=chunk_job_id)
            _log_job_event(conn, tenant_id, "chunk_done", {"doc_id": doc_id}, job_id=chunk_job_id)
        except Exception:
            pass

    return ChunkResponse(
        status="ok",
        kb_id=kb_id,
        doc_id=doc_id,
        chunk_job_id=chunk_job_id,
        input_kind=input_kind,
        input_fingerprint=input_fingerprint,
        params_hash=params_hash,
        chunk_count=len(chunks),
        gcs_chunks_uri=gcs_chunks_uri,
        gcs_manifest_uri=gcs_manifest_uri,
        stats={"input_chars": len(text), "first_chunk_chars": len(chunks[0]["text"])},
    )


if __name__ == "__main__":
    # Real self-test (no fake data):
    # - DB connect
    # - Chunks table create
    settings = get_settings()
    with _db_conn(settings) as conn:
        _ensure_chunks_table(conn)
        with conn.cursor() as cur:
            cur.execute("SELECT 1;")
            print("DB OK:", cur.fetchone()[0])
