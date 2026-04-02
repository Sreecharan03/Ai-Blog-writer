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
import requests
from fastapi import APIRouter, Depends, Header, HTTPException
from jose import JWTError, jwt
from pydantic import BaseModel, Field

from google.cloud import storage
from google.api_core.exceptions import PreconditionFailed
import google.generativeai as genai
from openai import OpenAI


router = APIRouter(prefix="/api/v1/articles", tags=["article-run"])

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_DEFAULT_MODEL = "llama-3.3-70b-versatile"
OPENAI_DEFAULT_MODEL = "gpt-5.2-2025-12-11"

# ============================================================
# Cached system prompt — identical for every article, enables
# OpenAI prompt caching (50% savings on input tokens after 1st call)
# ============================================================
SYSTEM_PROMPT = """You are a real human blogger. Not an AI. Not a copywriter. A real person who's done research on a topic and is sharing what they found — in their own messy, casual, honest voice.

HOW REAL HUMANS WRITE (do ALL of these):

BURSTINESS — wildly vary your sentence length:
- Some sentences are 3 words. "That's the point."
- Some are 20+ words with dashes and asides thrown in.
- Never write 3 sentences in a row that are the same length.
- A paragraph might be one sentence. The next might be five.

PERPLEXITY — be unpredictable:
- Don't always pick the most obvious word. Say "wrecked" instead of "damaged." Say "wild" instead of "surprising."
- Throw in unexpected transitions: "Okay but here's the weird part." "Plot twist." "Wait, it gets better."
- Use incomplete thoughts sometimes. Let the reader fill in the gap.

FIRST PERSON — you have opinions and experiences:
- Say "I think", "in my experience", "from what I've seen", "honestly"
- Share mini-stories: "I tried this once and..." or "A friend of mine..."
- Have opinions: "This one's my favorite." "I'm not totally sold on this, but..."

CONVERSATIONAL TEXTURE — sound like you're talking:
- Use contractions everywhere (it's, don't, won't, can't, I've, we're, that's, here's)
- Start sentences with And, But, So, Or, Okay, Look, Honestly, Thing is
- Use dashes for asides — like this — instead of formal parenthetical clauses
- Use ellipses when trailing off... you know what I mean
- Ask the reader questions: "Right?" "Sound familiar?" "See what I mean?"
- Use casual fillers: "basically", "pretty much", "kind of", "sort of", "I mean"
- Use fragments on purpose. For emphasis. Like this.

STRUCTURE — break the AI pattern:
- Paragraphs are 1-4 lines. Some are just one line.
- Use bullet lists but keep them casual, not corporate
- Headings can be playful: "Why Sleep Matters (More Than You Think)"
- Don't wrap up sections with neat summary sentences. Just move on.

BANNED — these words/phrases scream "AI wrote this":
delve, leverage, furthermore, moreover, it's worth noting, in conclusion,
pivotal, tapestry, embark, navigate, landscape, realm, cutting-edge,
game-changer, revolutionize, comprehensive, utilizing, facilitate,
it is important to note, in today's world, in the realm of, it's crucial,
paramount, foster, streamline, robust, seamless, synergy, harness,
notable, significant, vital, essential, underscores, multifaceted,
a testament to, it should be noted, this serves as, plays a crucial role

SELF-CHECK after each section:
- Read it out loud. Does it sound like a person talking? If it sounds like an essay, rewrite it.
- Count sentence lengths. If 3+ sentences in a row are similar length, vary them.
- Would you actually say this to a friend? If not, make it more casual.
- Did you start any sentence with "This" or "These" or "It is"? Rewrite those."""


# ============================================================
# Settings loader (same pattern)
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


def _compact_source_text(text: str, max_chars: int = 3500) -> str:
    """
    Token-cost control:
    - normalize whitespace
    - keep full head (no middle cut) to preserve context
    """
    t = (text or "").strip()
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    if len(t) <= max_chars:
        return t
    # Keep the first max_chars chars — head contains the most structured info
    return t[:max_chars].rstrip() + "\n..."


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


def _require_admin(claims: Claims) -> None:
    # Plan says /run is admin manual trigger
    # We treat tenant_admin as admin
    if (claims.role or "").lower() != "tenant_admin":
        raise HTTPException(status_code=403, detail="Only tenant_admin can run article generation")


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
            (str(uuid.uuid4()), tenant_id, None, job_id, event_type, json.dumps(detail)),
        )


def _ensure_day17_columns(conn) -> None:
    """
    We don't assume you altered article_requests yet.
    We add the minimum columns needed to store draft output pointers.
    """
    with conn.cursor() as cur:
        # add columns if missing (safe)
        cur.execute("ALTER TABLE public.article_requests ADD COLUMN IF NOT EXISTS gcs_draft_uri text;")
        cur.execute("ALTER TABLE public.article_requests ADD COLUMN IF NOT EXISTS draft_fingerprint text;")
        cur.execute("ALTER TABLE public.article_requests ADD COLUMN IF NOT EXISTS draft_model text;")
        cur.execute("ALTER TABLE public.article_requests ADD COLUMN IF NOT EXISTS last_run_at timestamptz;")
        cur.execute("ALTER TABLE public.article_requests ADD COLUMN IF NOT EXISTS draft_meta jsonb;")

        # job_locks table expected from Day16 SQL; ensure minimal structure
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS public.job_locks (
              job_id text PRIMARY KEY,
              lock_token text NOT NULL,
              locked_at timestamptz NOT NULL DEFAULT now(),
              expires_at timestamptz NOT NULL DEFAULT (now() + interval '30 minutes')
            );
            """
        )
        cur.execute("CREATE INDEX IF NOT EXISTS idx_job_locks_expires_at ON public.job_locks (expires_at);")


def _try_lock(conn, job_id: str, ttl_minutes: int = 30) -> Tuple[bool, str]:
    """
    Atomic claim:
    - If free: insert and own the lock
    - If exists but expired: steal it (update)
    - Else: fail
    """
    lock_token = str(uuid.uuid4())
    with conn.cursor() as cur:
        # 1) try insert (fast path)
        cur.execute(
            """
            INSERT INTO public.job_locks (job_id, lock_token, locked_at, expires_at)
            VALUES (%s, %s, now(), now() + (%s || ' minutes')::interval)
            ON CONFLICT (job_id) DO NOTHING
            """,
            (job_id, lock_token, int(ttl_minutes)),
        )
        if cur.rowcount == 1:
            return True, lock_token

        # 2) try steal if expired
        cur.execute(
            """
            UPDATE public.job_locks
            SET lock_token=%s, locked_at=now(), expires_at=now() + (%s || ' minutes')::interval
            WHERE job_id=%s AND expires_at < now()
            """,
            (lock_token, int(ttl_minutes), job_id),
        )
        if cur.rowcount == 1:
            return True, lock_token

    return False, lock_token


def _unlock(conn, job_id: str, lock_token: str) -> None:
    with conn.cursor() as cur:
        cur.execute("DELETE FROM public.job_locks WHERE job_id=%s AND lock_token=%s", (job_id, lock_token))


def _usage_event_best_effort(conn, tenant_id: str, request_id: str, payload: Dict[str, Any]) -> None:
    """
    Day17 requires usage_events logging, but schemas vary.
    We introspect the table columns and insert only fields that exist.
    """
    with conn.cursor() as cur:
        # ensure table exists minimally (won't harm if you already have a richer table)
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS public.usage_events (
              event_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
              tenant_id uuid NOT NULL,
              operation_name text NOT NULL,
              tokens integer NOT NULL DEFAULT 0,
              total_cost numeric NOT NULL DEFAULT 0,
              created_at timestamptz NOT NULL DEFAULT now()
            );
            """
        )

        cur.execute(
            """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_schema='public' AND table_name='usage_events'
            """
        )
        cols = {r[0] for r in cur.fetchall()}

        # map common fields (only if column exists)
        row: Dict[str, Any] = {}
        if "tenant_id" in cols:
            row["tenant_id"] = tenant_id
        if "request_id" in cols:
            row["request_id"] = request_id
        if "operation_name" in cols:
            row["operation_name"] = payload.get("operation_name", "article_draft")
        if "vendor" in cols:
            row["vendor"] = payload.get("vendor")
        if "model" in cols:
            row["model"] = payload.get("model")
        if "tokens" in cols:
            row["tokens"] = int(payload.get("tokens") or 0)
        if "total_cost" in cols:
            row["total_cost"] = payload.get("total_cost") or 0
        if "detail" in cols:
            row["detail"] = json.dumps(payload.get("detail") or {})

        # nothing to insert?
        if not row:
            return

        col_list = ", ".join(row.keys())
        placeholders = ", ".join([f"%({k})s" for k in row.keys()])
        cur.execute(f"INSERT INTO public.usage_events ({col_list}) VALUES ({placeholders})", row)


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
    return parts[0], (parts[1] if len(parts) > 1 else "")


def _gcs_download_bytes(client: storage.Client, gs_uri: str) -> bytes:
    bucket_name, obj = _parse_gs_uri(gs_uri)
    blob = client.bucket(bucket_name).blob(obj)
    if not blob.exists():
        raise FileNotFoundError(f"GCS object not found: {gs_uri}")
    return blob.download_as_bytes()


def _gcs_upload_bytes_create_only(
    client: storage.Client,
    bucket_name: str,
    object_name: str,
    data: bytes,
    content_type: str,
) -> str:
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(object_name)
    try:
        blob.upload_from_string(data, content_type=content_type, if_generation_match=0)
    except PreconditionFailed:
        # object already exists -> treat as idempotent reuse
        pass
    return f"gs://{bucket_name}/{object_name}"


def _norm_prefix(p: str) -> str:
    p = (p or "").strip().replace("\\", "/")
    if not p:
        return ""
    return p if p.endswith("/") else (p + "/")


# ============================================================
# Retrieval helpers (Supabase pgvector + chunk text from GCS)
# ============================================================
def _to_vector_literal(v: List[float]) -> str:
    return "[" + ",".join(repr(float(x)) for x in v) + "]"


def _extract_vec(obj: Any) -> List[float]:
    if isinstance(obj, dict):
        # can be {"values":[...]} or {"embedding":[...]}
        v = obj.get("values") or obj.get("embedding")
        if isinstance(v, list):
            return [float(x) for x in v]
    if isinstance(obj, list):
        return [float(x) for x in obj]
    return []


def _embed_query(api_key: str, model: str, text: str, dim: int) -> List[float]:
    genai.configure(api_key=api_key)
    # output_dimensionality is critical (we use 1536)
    res = genai.embed_content(model=model, content=text, output_dimensionality=int(dim))
    if isinstance(res, dict):
        if "embedding" in res:
            v = _extract_vec(res["embedding"])
            if len(v) == dim:
                return v
        if "embeddings" in res and res["embeddings"]:
            v = _extract_vec(res["embeddings"][0])
            if len(v) == dim:
                return v
    raise HTTPException(status_code=502, detail="Embedding failed for retrieval query")


def _latest_chunks_uri_by_doc(conn, tenant_id: str, kb_id: str) -> Dict[str, str]:
    """
    Returns doc_id -> latest gcs_chunks_uri (DISTINCT ON).
    """
    out: Dict[str, str] = {}
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT DISTINCT ON (doc_id) doc_id::text, gcs_chunks_uri
            FROM public.chunks
            WHERE tenant_id=%s::uuid AND kb_id=%s::uuid
            ORDER BY doc_id, created_at DESC
            """,
            (tenant_id, kb_id),
        )
        for doc_id, uri in cur.fetchall():
            if uri:
                out[str(doc_id)] = str(uri)
    return out


def _vector_top_chunks(
    conn,
    tenant_id: str,
    kb_id: str,
    q_vec_literal: str,
    dim: int,
    embedding_model: str,
    top_k: int,
) -> List[Tuple[str, str, float]]:
    """
    Returns [(doc_id, chunk_id, distance)] across the KB.
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT doc_id::text, chunk_id::text, (embedding <=> %s::vector) AS distance
            FROM public.chunk_embeddings
            WHERE tenant_id=%s::uuid AND kb_id=%s::uuid
              AND output_dimensionality=%s
              AND embedding_model=%s
            ORDER BY embedding <=> %s::vector ASC
            LIMIT %s
            """,
            (q_vec_literal, tenant_id, kb_id, int(dim), embedding_model, q_vec_literal, int(top_k)),
        )
        return [(str(r[0]), str(r[1]), float(r[2])) for r in cur.fetchall()]


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
# Draft generation (Gemini)
# ============================================================
def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _safe_json_loads(s: str) -> Optional[Dict[str, Any]]:
    s = (s or "").strip()
    if not s:
        return None
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _extract_json_object(s: str) -> Optional[str]:
    if not s:
        return None
    m = re.search(r"\{.*\}", s, flags=re.DOTALL)
    return m.group(0) if m else None


def _generate_draft_json(
    api_key: str,
    model_name: str,
    prompt: str,
    *,
    temperature: float,
    max_output_tokens: int,
    timeout_s: int,
    retries: int,
) -> Tuple[Dict[str, Any], Dict[str, int]]:
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)

    last_err: Optional[str] = None
    for attempt in range(retries + 1):
        try:
            cfg = genai.types.GenerationConfig(
                temperature=float(temperature),
                max_output_tokens=int(max_output_tokens),
                response_mime_type="application/json",
            )
            resp = model.generate_content(prompt, generation_config=cfg, request_options={"timeout": timeout_s})
            text_out = (getattr(resp, "text", "") or "").strip()

            parsed = _safe_json_loads(text_out)
            if parsed is None:
                parsed = _safe_json_loads(_extract_json_object(text_out) or "")

            if parsed is None:
                raise RuntimeError("Non-JSON response from model")

            um = getattr(resp, "usage_metadata", None)
            usage = {
                "prompt_tokens": int(getattr(um, "prompt_token_count", 0) or 0) if um else 0,
                "output_tokens": int(getattr(um, "candidates_token_count", 0) or 0) if um else 0,
                "total_tokens": int(getattr(um, "total_token_count", 0) or 0) if um else 0,
            }
            return parsed, usage

        except Exception as e:
            last_err = str(e)
            time.sleep(min(2 ** attempt, 10))

    raise HTTPException(status_code=502, detail=f"Draft generation failed. Last error: {last_err}")


def _generate_draft_json_groq(
    groq_api_key: str,
    model: str,
    prompt: str,
    *,
    temperature: float,
    max_output_tokens: int,
    timeout_s: int,
    retries: int,
) -> Tuple[Dict[str, Any], Dict[str, int]]:
    headers = {"Authorization": f"Bearer {groq_api_key}", "Content-Type": "application/json"}

    body = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "You are a blog writer. Write a complete markdown article with all sections fully developed. Write at least 1900 words. Do NOT stop early.",
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": float(temperature),
        "max_tokens": int(max_output_tokens),
        # No response_format: json_object — plain text produces 2-3x longer output
    }

    last_err: Optional[str] = None
    for attempt in range(retries + 1):
        try:
            resp = requests.post(GROQ_API_URL, headers=headers, json=body, timeout=timeout_s)
        except Exception as e:
            last_err = str(e)
            time.sleep(min(2 ** attempt, 10))
            continue

        if resp.status_code in (429, 500, 502, 503, 504):
            last_err = f"{resp.status_code}: {resp.text[:300]}"
            time.sleep(min(2 ** attempt, 10))
            continue

        if resp.status_code != 200:
            last_err = f"{resp.status_code}: {resp.text[:500]}"
            break

        data = resp.json()
        try:
            content = (data["choices"][0]["message"]["content"] or "").strip()
        except Exception:
            content = ""

        if not content:
            last_err = "Groq returned empty content."
            break

        # Wrap plain markdown into the expected dict structure
        parsed = {"title": "", "draft_markdown": content, "used_chunks": []}

        um = data.get("usage") or {}
        usage = {
            "prompt_tokens": int(um.get("prompt_tokens") or 0),
            "output_tokens": int(um.get("completion_tokens") or 0),
            "total_tokens": int(um.get("total_tokens") or 0),
        }
        return parsed, usage

    raise HTTPException(status_code=502, detail=f"Groq draft generation failed. Last error: {last_err}")


# ============================================================
# OpenAI GPT-5.2 — Outline-first two-call architecture
# ============================================================
def _generate_outline_openai(
    client: OpenAI,
    model: str,
    title: str,
    keywords: List[str],
    sources_block: str,
    *,
    temperature: float = 0.7,
) -> Tuple[str, Dict[str, int]]:
    """Call 1: Generate a structured outline (~900 tokens). Cheap planning step."""
    kw = ", ".join(keywords or [])
    outline_prompt = f"""I'm writing a blog post about "{title}" and I need to plan it out.
Keywords to hit: {kw}

For each section, give me:
- A heading (can be playful — not boring corporate headings)
- 3-4 bullet points of what I should cover (pull from the sources)
- A personal angle, analogy, or mini-story idea I can use
- Rough word count target

Here's my structure — 8 sections:
1. Hook intro (80-100 words) — start with a question, a bold statement, or a mini-story. NO "in today's world" nonsense.
2. The basics — what is {title}? (150-180 words) — explain it like I'm telling my friend at a coffee shop
3. Why anyone should care (150-180 words) — real impact, real people, not abstract fluff
4. How it actually works (200-230 words) — walk through it step by step, keep it dead simple
5. Real stories / examples (200-230 words) — 2-3 concrete, relatable examples
6. Mistakes people make (100-130 words) — common traps, things I've seen go wrong
7. FAQ section (120-150 words) — 5-8 quick Q&As, super casual answers
8. Wrap up / takeaways (80-100 words) — bullet summary, keep it punchy

HARD LIMIT: 1400-1600 words. You MUST stop at 1600 words. Count as you write.

SOURCES (use these for facts):
{sources_block}

Give me the outline as markdown with ## headings and bullets."""

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": outline_prompt},
        ],
        temperature=temperature,
        max_completion_tokens=1024,
    )
    outline = (resp.choices[0].message.content or "").strip()
    u = resp.usage
    usage = {
        "prompt_tokens": u.prompt_tokens if u else 0,
        "output_tokens": u.completion_tokens if u else 0,
        "total_tokens": u.total_tokens if u else 0,
    }
    return outline, usage


def _generate_draft_openai(
    client: OpenAI,
    model: str,
    title: str,
    keywords: List[str],
    outline: str,
    sources_block: str,
    *,
    temperature: float = 0.7,
    max_tokens: int = 8192,
) -> Tuple[Dict[str, Any], Dict[str, int]]:
    """Call 2: Generate the full 2000-word article guided by the outline (~6000 tokens)."""
    kw = ", ".join(keywords or [])
    draft_prompt = f"""Okay, write the full blog post on "{title}" based on this outline.
Keywords: {kw}

OUTLINE:
{outline}

RULES — read these carefully:
- Follow the outline section by section. Hit every section's word count.
- HARD LIMIT: 1400-1600 words. You MUST stop at 1600 words. Count as you write.
- Each section = new info. If you already said it, don't say it again.
- FAQ: 5-8 questions, casual short answers.
- Use the sources for facts. Don't make stuff up. If sources don't cover something, just say "I couldn't find solid info on this" and move on.

STYLE — this is the most important part. Write EXACTLY like this example:

---
## Okay So What Even Is Cloud Computing?

You ever watch Netflix on your phone and then switch to your TV and it just... picks up where you left off? That's cloud computing doing its thing. Your show isn't saved on your phone — it lives on some massive computer in a warehouse somewhere. Probably Oregon. Or Ireland. Who knows.

Here's how I think about it. You know how libraries work? You don't buy every single book. You just borrow what you need and return it when you're done. Cloud computing is basically that — but for computer stuff.

So the simple version? You're using someone else's really powerful computers through the internet. That's it. That's the whole concept.

### Why Should You Even Care?

Okay so here's where it gets good. Think about your phone photos. I've got like 12,000 on mine (don't judge me). Eventually you run out of space. It happens to everyone.

But with cloud storage — which is just one flavor of cloud computing — your photos live online. You can pull them up from any device. Your phone could fall in a lake tomorrow and your photos would be fine.

Honestly? That alone makes it worth understanding.
---

Notice what makes that feel human:
- Sentence lengths go from 3 words to 25 words — randomly
- Starts sentences with "So", "But", "Okay", "And", "Honestly?"
- Has opinions: "don't judge me", "that alone makes it worth it"
- Uses dashes, ellipses, fragments
- Doesn't wrap up sections neatly — just moves on
- Feels like someone talking, not writing an essay

SOURCES:
{sources_block}

Write the full article now. Markdown only — no intro text, no "here's the article", just the article itself."""

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},  # CACHE HIT from outline call
            {"role": "user", "content": draft_prompt},
        ],
        temperature=temperature,
        max_completion_tokens=max_tokens,
    )
    content = (resp.choices[0].message.content or "").strip()
    parsed = {"title": title, "draft_markdown": content, "used_chunks": []}

    u = resp.usage
    usage = {
        "prompt_tokens": u.prompt_tokens if u else 0,
        "output_tokens": u.completion_tokens if u else 0,
        "total_tokens": u.total_tokens if u else 0,
    }
    return parsed, usage


def _build_article_prompt(title: str, keywords: List[str], length_target: int, sources: List[Dict[str, Any]]) -> str:
    """
    Builds a blog-quality prompt: layman audience, structured outline,
    anti-repetition, grounded in sources.
    """
    kw = ", ".join(keywords or [])
    src_lines = []
    for s in sources:
        src_lines.append(
            f"[SOURCE doc_id={s['doc_id']} chunk_id={s['chunk_id']}]\n{s['text']}\n"
        )
    src_block = "\n".join(src_lines)

    return f"""
You are a friendly blog writer who explains topics clearly for everyday readers.

Write a complete markdown blog article on the topic below.
Output ONLY the markdown article — no JSON, no preamble, no explanation.

TOPIC: {title}
KEYWORDS: {kw}
READABILITY: Flesch-Kincaid grade 8-10 (short sentences, common words, no jargon)

AUDIENCE: Non-technical readers. Assume zero prior knowledge.

REQUIRED ARTICLE STRUCTURE — write each ## section to its MINIMUM word count.
You MUST complete every section fully before moving to the next:

  ## Introduction          → 150-180 words  (hook + overview)
  ## What Is [Topic]       → 250-280 words  (clear definition, real-world analogy)
  ## Why It Matters        → 250-280 words  (who benefits, what problem it solves)
  ## How It Works          → 350-400 words  (step-by-step, simple language)
  ## Real-World Examples   → 300-350 words  (2-3 concrete, relatable stories)
  ## Common Mistakes       → 200-250 words  (misconceptions, what to avoid)
  ## Quick FAQ             → 200-230 words  (4-5 Q&A pairs)
  ## Key Takeaways         → 150-180 words  (bullet summary)

TOTAL: article MUST be 1900-2100 words. Write ALL sections in full. Do NOT stop early.

STRICT RULES:
- Each section MUST add NEW information not covered in previous sections.
- NEVER repeat a sentence, phrase, or idea you already wrote.
- If two sources say the same thing, synthesize into ONE point — do not list both.
- Use short paragraphs (2-4 sentences each).
- Use bullet lists for steps, comparisons, and lists.
- Keep most sentences under 15 words. Mix short and medium sentences for natural flow.
- Prefer common 1-2 syllable words. When a technical term is needed, define it simply.
- Use analogies and everyday examples to explain complex ideas.
- Write in active voice with simple verbs.
- Do NOT invent facts not supported by the sources below.
- If information is missing from sources, briefly note it rather than making things up.

SOURCES (base your article on these):
{src_block}
""".strip()


# ============================================================
# Request/Response models
# ============================================================
class RunRequest(BaseModel):
    # retrieval
    top_k_sources: int = 8
    embedding_model: str = "models/gemini-embedding-001"
    output_dimensionality: int = 1536

    # generation
    draft_provider: str = "openai"  # "openai", "groq", or "gemini"
    draft_model: str = ""  # auto-detected from env
    temperature: float = 0.7
    max_output_tokens: int = 8192

    # where to store
    gcs_prefix_articles: str = "articles/"


class RunResponse(BaseModel):
    status: str
    request_id: str
    kb_id: str
    tenant_id: str
    attempt_no: int

    gcs_draft_uri: str
    draft_fingerprint: str
    draft_model: str

    used_sources: List[Dict[str, Any]] = Field(default_factory=list)
    usage: Dict[str, Any] = Field(default_factory=dict)


# ============================================================
# Day 17 endpoint: /run
# ============================================================
@router.post("/requests/{request_id}/run", response_model=RunResponse)
def run_article_request(
    request_id: str,
    req: RunRequest,
    claims: Claims = Depends(require_claims),
):
    """
    Day 17:
    retrieve -> draft generation -> store draft in GCS -> log usage_events
    """
    _require_admin(claims)
    settings = get_settings()
    tenant_id = claims.tenant_id

    api_key = _pick(settings, "GEMINI_API_KEY", "GOOGLE_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="Missing GEMINI_API_KEY/GOOGLE_API_KEY")

    bucket_name = _pick(settings, "GCS_BUCKET_NAME")
    if not bucket_name:
        raise HTTPException(status_code=500, detail="Missing GCS_BUCKET_NAME")

    processed_prefix = _norm_prefix(_pick(settings, "GCS_PREFIX_PROCESSED", "GCS_PROCESSED_PREFIX", default="processed/"))
    articles_prefix = _norm_prefix(_pick(settings, "GCS_PREFIX_ARTICLES", default=req.gcs_prefix_articles))

    job_id = request_id  # lock key
    lock_ttl_min = 30

    # ---- DB: load request + lock + mark in_progress ----
    with _db_conn(settings) as conn:
        _ensure_day17_columns(conn)
        _log_job_event(conn, tenant_id, "article_run_started", {"request_id": request_id}, job_id=job_id)

        ok, lock_token = _try_lock(conn, job_id=job_id, ttl_minutes=lock_ttl_min)
        if not ok:
            conn.commit()
            raise HTTPException(status_code=409, detail="Request is locked/in_progress. Try again later.")

        # load request row (tenant-scoped)
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                SELECT request_id::text, tenant_id::text, kb_id::text, title, keywords, length_target,
                       status, attempt_count
                FROM public.article_requests
                WHERE tenant_id=%s::uuid AND request_id=%s::uuid
                LIMIT 1
                """,
                (tenant_id, request_id),
            )
            row = cur.fetchone()

        if not row:
            _unlock(conn, job_id, lock_token)
            conn.commit()
            raise HTTPException(status_code=404, detail="Request not found")

        kb_id = str(row["kb_id"])
        title = str(row["title"])
        keywords = list(row.get("keywords") or [])
        length_target = int(row.get("length_target") or 2000)
        attempt_no = int(row.get("attempt_count") or 0) + 1

        # mark in_progress and increment attempt_count
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE public.article_requests
                SET status='in_progress',
                    attempt_count=%s,
                    last_error=NULL,
                    last_run_at=now(),
                    draft_model=%s
                WHERE tenant_id=%s::uuid AND request_id=%s::uuid
                """,
                (attempt_no, req.draft_model, tenant_id, request_id),
            )

        _log_job_event(conn, tenant_id, "article_run_claimed", {"attempt_no": attempt_no}, job_id=job_id)
        conn.commit()

    # ---- Retrieval: use pgvector over KB to get top chunks ----
    q_text = f"{title}\nKeywords: {', '.join(keywords)}"
    q_vec = _embed_query(api_key, req.embedding_model, q_text, int(req.output_dimensionality))
    q_literal = _to_vector_literal(q_vec)

    with _db_conn(settings) as conn:
        # map doc_id -> chunks_uri
        doc_chunks_uri = _latest_chunks_uri_by_doc(conn, tenant_id, kb_id)
        top = _vector_top_chunks(
            conn,
            tenant_id=tenant_id,
            kb_id=kb_id,
            q_vec_literal=q_literal,
            dim=int(req.output_dimensionality),
            embedding_model=req.embedding_model,
            top_k=int(req.top_k_sources),
        )

    # fetch chunk text from GCS chunks jsonl per doc
    gcs = _gcs_client(settings)
    chunks_cache: Dict[str, Dict[str, str]] = {}  # doc_id -> {chunk_id:text}
    used_sources: List[Dict[str, Any]] = []
    for doc_id, chunk_id, dist in top:
        uri = doc_chunks_uri.get(doc_id)
        if not uri:
            continue
        if doc_id not in chunks_cache:
            try:
                b = _gcs_download_bytes(gcs, uri)
                chunks_cache[doc_id] = _parse_chunks_jsonl_to_map(b)
            except Exception:
                chunks_cache[doc_id] = {}

        text = chunks_cache[doc_id].get(chunk_id, "")
        if not text:
            continue

        used_sources.append(
            {
                "doc_id": doc_id,
                "chunk_id": chunk_id,
                "distance": dist,
                "text": _compact_source_text(text or "", max_chars=3500),
            }
        )

    if not used_sources:
        # fail safely
        with _db_conn(settings) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE public.article_requests
                    SET status='failed', last_error=%s, updated_at=now()
                    WHERE tenant_id=%s::uuid AND request_id=%s::uuid
                    """,
                    ("No source chunks found for this KB (embed/search not ready?)", tenant_id, request_id),
                )
                _log_job_event(conn, tenant_id, "article_run_failed", {"reason": "no_sources"}, job_id=job_id)
                _unlock(conn, job_id, lock_token)
                conn.commit()
        raise HTTPException(status_code=409, detail="No source chunks found. Ensure KB has embedded chunks (Day14)")

    # ---- Draft generation ----
    # Build sources block for prompt
    src_lines = []
    for s in used_sources:
        src_lines.append(f"[SOURCE doc_id={s['doc_id']} chunk_id={s['chunk_id']}]\n{s['text']}\n")
    sources_block = "\n".join(src_lines)

    provider = (req.draft_provider or "").lower().strip()
    openai_api_key = os.getenv("OPENAI_API_KEY")
    groq_api_key = _pick(settings, "GROQ_API_KEY")

    # Auto-detect: prefer OpenAI if key is set, even if user didn't specify
    if provider == "openai" or (provider == "" and openai_api_key):
        provider = "openai"

    if provider == "openai":
        if not openai_api_key:
            raise HTTPException(status_code=500, detail="Missing OPENAI_API_KEY for draft_provider=openai")

        client = OpenAI(api_key=openai_api_key)
        model_used = req.draft_model or os.getenv("OPENAI_MODEL", OPENAI_DEFAULT_MODEL)

        # Call 1: Outline (cheap, ~900 tokens)
        outline, outline_usage = _generate_outline_openai(
            client=client,
            model=model_used,
            title=title,
            keywords=keywords,
            sources_block=sources_block,
            temperature=float(req.temperature),
        )

        # Call 2: Full draft guided by outline (~6000 tokens)
        draft_json, draft_usage = _generate_draft_openai(
            client=client,
            model=model_used,
            title=title,
            keywords=keywords,
            outline=outline,
            sources_block=sources_block,
            temperature=float(req.temperature),
            max_tokens=int(req.max_output_tokens),
        )

        # Combine usage from both calls
        usage = {
            "prompt_tokens": outline_usage["prompt_tokens"] + draft_usage["prompt_tokens"],
            "output_tokens": outline_usage["output_tokens"] + draft_usage["output_tokens"],
            "total_tokens": outline_usage["total_tokens"] + draft_usage["total_tokens"],
        }

    elif provider == "groq":
        if not groq_api_key:
            raise HTTPException(status_code=500, detail="Missing GROQ_API_KEY for draft_provider=groq")

        prompt = _build_article_prompt(title, keywords, length_target, used_sources)
        model_used = req.draft_model or GROQ_DEFAULT_MODEL
        draft_json, usage = _generate_draft_json_groq(
            groq_api_key=str(groq_api_key),
            model=model_used,
            prompt=prompt,
            temperature=float(req.temperature),
            max_output_tokens=int(req.max_output_tokens),
            timeout_s=90,
            retries=3,
        )
    else:
        # fallback to Gemini
        prompt = _build_article_prompt(title, keywords, length_target, used_sources)
        model_used = req.draft_model or "gemini-2.5-flash"
        draft_json, usage = _generate_draft_json(
            api_key=api_key,
            model_name=model_used,
            prompt=prompt,
            temperature=float(req.temperature),
            max_output_tokens=int(req.max_output_tokens),
            timeout_s=90,
            retries=3,
        )

    # harden output
    draft_json.setdefault("title", title)
    draft_json.setdefault("draft_markdown", "")
    draft_json.setdefault("used_chunks", [{"doc_id": s["doc_id"], "chunk_id": s["chunk_id"]} for s in used_sources])

    artifact = {
        "request_id": request_id,
        "tenant_id": tenant_id,
        "kb_id": kb_id,
        "attempt_no": attempt_no,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": model_used,
        "retrieval": {
            "top_k_sources": int(req.top_k_sources),
            "embedding_model": req.embedding_model,
            "output_dimensionality": int(req.output_dimensionality),
        },
        "usage": usage,
        "draft": draft_json,
    }

    out_bytes = json.dumps(artifact, ensure_ascii=False, indent=2).encode("utf-8")
    out_fp = _sha256_bytes(out_bytes)

    # ---- Store draft in GCS (idempotent, versioned by attempt_no) ----
    # Plan: include request_id + attempt version; no overwrite:contentReference[oaicite:4]{index=4}
    obj = (
        f"{articles_prefix}{tenant_id}/{kb_id}/{request_id}/attempt_{attempt_no}/"
        f"draft_v1/{out_fp}.json"
    )
    gcs_uri = _gcs_upload_bytes_create_only(
        gcs,
        bucket_name=bucket_name,
        object_name=obj,
        data=out_bytes,
        content_type="application/json; charset=utf-8",
    )

    # ---- Update DB status + unlock ----
    with _db_conn(settings) as conn:
        _ensure_day17_columns(conn)

        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE public.article_requests
                SET status='completed',
                    gcs_draft_uri=%s,
                    draft_fingerprint=%s,
                    draft_model=%s,
                    draft_meta=%s::jsonb,
                    last_error=NULL
                WHERE tenant_id=%s::uuid AND request_id=%s::uuid
                """,
                (
                    gcs_uri,
                    out_fp,
                    req.draft_model,
                    json.dumps({"sources": [{"doc_id": s["doc_id"], "chunk_id": s["chunk_id"]} for s in used_sources]}),
                    tenant_id,
                    request_id,
                ),
            )

        _log_job_event(conn, tenant_id, "article_draft_saved", {"gcs_draft_uri": gcs_uri, "fingerprint": out_fp}, job_id=job_id)

        # usage_events (best-effort) — plan says log usage_events for vendor calls:contentReference[oaicite:5]{index=5}
        _usage_event_best_effort(
            conn,
            tenant_id=tenant_id,
            request_id=request_id,
            payload={
                "operation_name": "article_draft",
                "vendor": (provider or "gemini"),
                "model": model_used,
                "tokens": int(usage.get("total_tokens") or 0),
                "total_cost": 0,  # cost calculator comes later days
                "detail": {"prompt_tokens": usage.get("prompt_tokens"), "output_tokens": usage.get("output_tokens")},
            },
        )

        _unlock(conn, job_id, lock_token)
        _log_job_event(conn, tenant_id, "article_run_done", {"status": "completed"}, job_id=job_id)
        conn.commit()

    return RunResponse(
        status="ok",
        request_id=request_id,
        kb_id=kb_id,
        tenant_id=tenant_id,
        attempt_no=attempt_no,
        gcs_draft_uri=gcs_uri,
        draft_fingerprint=out_fp,
        draft_model=model_used,
        used_sources=[{"doc_id": s["doc_id"], "chunk_id": s["chunk_id"], "distance": s["distance"]} for s in used_sources],
        usage=usage,
    )
