from __future__ import annotations

import hashlib
import json
import os
import re
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import psycopg2
from psycopg2.extras import RealDictCursor, Json
import requests
from fastapi import APIRouter, Depends, Header, HTTPException, Query
from jose import JWTError, jwt
from pydantic import BaseModel, Field

from google.cloud import storage
from google.api_core.exceptions import PreconditionFailed


router = APIRouter(prefix="/api/v1/articles", tags=["article-zerogpt-fix"])

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
ZEROGPT_PASS_THRESHOLD = 20.0   # fakePercentage must be strictly < this


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


def _require_admin(claims: Claims) -> None:
    if (claims.role or "").lower() != "tenant_admin":
        raise HTTPException(status_code=403, detail="Only tenant_admin allowed")


# ============================================================
# DB helpers
# ============================================================
def _db_conn(settings: Any):
    return psycopg2.connect(
        host=_pick(settings, "DB_HOST", "SUPABASE_DB_HOST", "POSTGRES_HOST"),
        port=int(_pick(settings, "DB_PORT", "SUPABASE_DB_PORT", "POSTGRES_PORT", default=5432)),
        dbname=_pick(settings, "DB_NAME", "SUPABASE_DB_NAME", "POSTGRES_DB", default="postgres"),
        user=_pick(settings, "DB_USER", "SUPABASE_DB_USER", "POSTGRES_USER"),
        password=_pick(settings, "DB_PASSWORD", "SUPABASE_DB_PASSWORD", "POSTGRES_PASSWORD"),
        sslmode=_pick(settings, "DB_SSLMODE", "SUPABASE_DB_SSLMODE", "POSTGRES_SSLMODE", default="require"),
        connect_timeout=8,
    )


def _ensure_fix_columns(conn) -> None:
    with conn.cursor() as cur:
        cur.execute("ALTER TABLE public.article_requests ADD COLUMN IF NOT EXISTS zerogpt_score double precision;")
        cur.execute("ALTER TABLE public.article_requests ADD COLUMN IF NOT EXISTS zerogpt_pass boolean;")
        cur.execute("ALTER TABLE public.article_requests ADD COLUMN IF NOT EXISTS zerogpt_fix_attempts integer DEFAULT 0;")
        cur.execute("ALTER TABLE public.article_requests ADD COLUMN IF NOT EXISTS gcs_humanized_uri text;")
        cur.execute("ALTER TABLE public.article_requests ADD COLUMN IF NOT EXISTS humanized_fingerprint text;")
        cur.execute("ALTER TABLE public.article_requests ADD COLUMN IF NOT EXISTS gcs_zerogpt_uri text;")
        cur.execute("ALTER TABLE public.article_requests ADD COLUMN IF NOT EXISTS zerogpt_fingerprint text;")
        cur.execute("ALTER TABLE public.article_requests ADD COLUMN IF NOT EXISTS zerogpt_meta jsonb;")


def _log_job_event(conn, tenant_id: str, event_type: str, detail: Dict[str, Any], request_id: Optional[str] = None) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO public.job_events (event_id, tenant_id, request_id, job_id, event_type, detail)
            VALUES (%s, %s, %s, %s, %s, %s)
            """,
            (str(uuid.uuid4()), tenant_id, request_id, None, event_type, Json(detail)),
        )


# ============================================================
# GCS helpers
# ============================================================
def _gcs_client(settings: Any) -> storage.Client:
    return storage.Client(project=_pick(settings, "GCP_PROJECT_ID", "GOOGLE_CLOUD_PROJECT"))


def _parse_gs_uri(gs_uri: str) -> Tuple[str, str]:
    if not gs_uri.startswith("gs://"):
        raise ValueError(f"Not a gs:// uri: {gs_uri}")
    rest = gs_uri[5:]
    parts = rest.split("/", 1)
    return parts[0], (parts[1] if len(parts) > 1 else "")


def _gcs_download_bytes(client: storage.Client, gs_uri: str) -> bytes:
    bucket, obj = _parse_gs_uri(gs_uri)
    blob = client.bucket(bucket).blob(obj)
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


def _gcs_put_create_only(client: storage.Client, bucket_name: str, obj: str, data: bytes, ctype: str) -> str:
    blob = client.bucket(bucket_name).blob(obj)
    try:
        blob.upload_from_string(data, content_type=ctype, if_generation_match=0)
    except PreconditionFailed:
        pass  # already exists, treat as idempotent
    return f"gs://{bucket_name}/{obj}"


def _norm_prefix(p: str) -> str:
    p = (p or "").strip().replace("\\", "/")
    return p if (not p or p.endswith("/")) else (p + "/")


def _sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


# ============================================================
# Draft markdown extractor (handles both run & qc-fix artifact shapes)
# ============================================================
def _extract_draft_markdown(artifact: Dict[str, Any]) -> str:
    d = artifact.get("draft")
    if isinstance(d, dict):
        return str(d.get("draft_markdown") or d.get("content") or d.get("text") or "")
    return str(artifact.get("draft_markdown") or artifact.get("content") or artifact.get("text") or "")


def _strip_md_noise(text: str) -> str:
    text = text or ""
    text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text


# ============================================================
# ZeroGPT internal call
# ============================================================
def _zerogpt_check(base_url: str, api_key: str, text: str, timeout_s: int = 60) -> Tuple[Optional[float], List[Any], Dict[str, Any]]:
    """
    Returns (score_or_none, sentences_list, raw_response).
    score = fakePercentage (0-100), higher = more AI detected.
    """
    endpoint = os.getenv("ZEROGPT_ENDPOINT_PATH", "/api/detect/detectText")
    if not endpoint.startswith("/"):
        endpoint = "/" + endpoint
    url = (base_url or "").rstrip("/") + endpoint
    headers = {"ApiKey": api_key, "Content-Type": "application/json"}
    try:
        r = requests.post(url, headers=headers, json={"input_text": text}, timeout=(10, timeout_s))
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"ZeroGPT network error: {type(e).__name__}: {str(e)[:200]}")
    if r.status_code != 200:
        raise HTTPException(status_code=502, detail=f"ZeroGPT error {r.status_code}: {r.text[:400]}")
    raw = r.json()
    data = raw.get("data") if isinstance(raw, dict) else None
    score: Optional[float] = None
    sentences: List[Any] = []
    if isinstance(data, dict):
        try:
            score = float(data.get("fakePercentage"))
        except Exception:
            score = None
        # BUG FIX: ZeroGPT returns AI sentences in 'h' field, NOT 'sentences' (confirmed via API test)
        # 'sentences' is always [] — 'h' contains the actual AI-flagged sentence strings
        sentences = data.get("h") or data.get("sentences") or []
        if not isinstance(sentences, list):
            sentences = []
    return score, sentences, raw


# ============================================================
# Extract AI sentence texts from ZeroGPT sentences array
# ============================================================
def _extract_ai_sentences(sentences: List[Any]) -> List[str]:
    """
    ZeroGPT 'sentences' can be:
    - list of strings
    - list of dicts with 'text', 'sentence', or 'content' key
    Returns clean list of sentence strings.
    """
    result: List[str] = []
    for s in sentences:
        if isinstance(s, str) and s.strip():
            result.append(s.strip())
        elif isinstance(s, dict):
            text = s.get("text") or s.get("sentence") or s.get("content") or ""
            if str(text).strip():
                result.append(str(text).strip())
    # Deduplicate while preserving order
    seen: set = set()
    deduped: List[str] = []
    for t in result:
        if t not in seen:
            seen.add(t)
            deduped.append(t)
    return deduped


# ============================================================
# Kimi K2 plain-text humanize call (no JSON mode — produces much longer output)
# ============================================================
def _groq_humanize(
    key: str,
    *,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    max_tokens: int,
    timeout_s: int,
    retries: int,
) -> Tuple[str, Dict[str, int]]:
    """
    Calls Groq in plain text mode (no response_format).
    Returns (rewritten_markdown_text, usage_dict).
    """
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": float(temperature),
        "max_tokens": int(max_tokens),
        # NO response_format — plain text mode, K2 produces 2-3x longer output
    }
    last_err: Optional[str] = None
    for attempt in range(retries + 1):
        try:
            resp = requests.post(GROQ_API_URL, headers=headers, json=body, timeout=timeout_s)
        except Exception as e:
            last_err = f"{type(e).__name__}: {str(e)[:150]}"
            time.sleep(min(2 ** attempt, 8))
            continue
        if resp.status_code in (429, 500, 502, 503, 504):
            last_err = f"{resp.status_code}: {resp.text[:200]}"
            time.sleep(min(2 ** attempt, 8))
            continue
        if resp.status_code != 200:
            last_err = f"{resp.status_code}: {resp.text[:400]}"
            break
        data = resp.json()
        try:
            content = (data["choices"][0]["message"]["content"] or "").strip()
        except Exception:
            content = ""
        if not content:
            last_err = "Groq returned empty content"
            break
        um = data.get("usage") or {}
        usage = {
            "prompt_tokens": int(um.get("prompt_tokens") or 0),
            "output_tokens": int(um.get("completion_tokens") or 0),
            "total_tokens": int(um.get("total_tokens") or 0),
        }
        return content, usage
    raise HTTPException(status_code=502, detail=f"Groq humanize call failed. Last error: {last_err}")


# ============================================================
# OpenAI GPT-5.2 humanize call (preferred when OPENAI_API_KEY is set)
# ============================================================
def _openai_humanize(
    *,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    max_tokens: int,
) -> Tuple[str, Dict[str, int]]:
    """Calls OpenAI GPT-5.2 for humanization. Returns (text, usage)."""
    from openai import OpenAI
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="Missing OPENAI_API_KEY for humanize")
    client = OpenAI(api_key=api_key)
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
        max_completion_tokens=max_tokens,
    )
    content = (resp.choices[0].message.content or "").strip()
    if not content:
        raise HTTPException(status_code=502, detail="OpenAI humanize returned empty content")
    u = resp.usage
    usage = {
        "prompt_tokens": u.prompt_tokens if u else 0,
        "output_tokens": u.completion_tokens if u else 0,
        "total_tokens": u.total_tokens if u else 0,
    }
    return content, usage


def _humanize_llm(
    groq_key: str,
    *,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    max_tokens: int,
    timeout_s: int = 180,
    retries: int = 2,
) -> Tuple[str, Dict[str, int]]:
    """Route humanize calls to OpenAI (preferred) or Groq (fallback)."""
    if os.getenv("OPENAI_API_KEY"):
        openai_model = os.getenv("OPENAI_MODEL", "gpt-5.2-2025-12-11")
        return _openai_humanize(
            model=openai_model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    return _groq_humanize(
        groq_key,
        model=model,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout_s=timeout_s,
        retries=retries,
    )


# ============================================================
# Build humanize prompts
# ============================================================
def _build_humanize_prompts(
    full_markdown: str,
    ai_sentences: List[str],
    current_score: float,
    attempt_num: int,
) -> Tuple[str, str]:
    """
    Returns (system_prompt, user_prompt).
    Uses surgical mode when ai_sentences is non-empty,
    falls back to full-doc rewrite when empty.
    """
    system_prompt = (
        "You are rewriting sentences to beat AI detection. The KEY things AI detectors look for:\n"
        "1. UNIFORM SENTENCE LENGTH — AI writes sentences that are all roughly the same length. "
        "Humans don't. Mix wildly: 3 words, then 22 words, then 8 words.\n"
        "2. PREDICTABLE WORD CHOICES — AI picks the 'safest' word. Humans pick weird, unexpected words. "
        "Say 'wrecked' not 'damaged'. Say 'honestly' not 'in fact'. Say 'wild' not 'surprising'.\n"
        "3. SMOOTH TRANSITIONS — AI connects everything perfectly. Humans jump around. "
        "Use 'Okay but', 'So anyway', 'Here's the thing', 'Plot twist'.\n"
        "4. NO PERSONAL VOICE — AI never says 'I think' or 'in my experience'. Add these everywhere.\n"
        "5. PERFECT GRAMMAR — Humans use fragments. On purpose. Like this. Start with 'And' or 'But'.\n\n"
        "Your job: rewrite flagged text so it has HIGH BURSTINESS (varied sentence lengths) "
        "and HIGH PERPLEXITY (unexpected word choices). Make it sound like a real blogger.\n"
        "Output ONLY the complete article in markdown. No preamble, no explanation."
    )

    ai_clichés = (
        "delve, leverage, it's worth noting, in conclusion, furthermore, "
        "moreover, additionally, in summary, it is important to note, "
        "pivotal, tapestry, embark, navigating, landscape, realm, "
        "cutting-edge, game-changer, comprehensive, utilizing, facilitate, "
        "paramount, foster, streamline, robust, seamless, synergy, harness, "
        "crucial, essential, vital, significant, notable, remarkable"
    )

    # Vary techniques by attempt to avoid repetitive rewrites
    techniques = [
        "Add personal opinions: 'I think...', 'Honestly...', 'In my experience...'",
        "Use sentence fragments: 'Big mistake.' 'Not even close.' 'Worth it.'",
        "Start 5+ sentences with 'And', 'But', 'So', 'Look', 'Okay so'",
        "Add parenthetical asides: (seriously), (no joke), (at least for me), (wild, right?)",
        "Use rhetorical questions: 'Right?', 'Makes sense?', 'Sound familiar?'",
        "Add em-dashes and ellipses: 'stress — and I mean real stress — can wreck you'",
        "Use casual transitions: 'Anyway', 'Moving on', 'Here's the deal', 'Okay but'",
        "Break a long sentence into 2-3 short punchy ones",
    ]
    # Pick different techniques per attempt for variety
    tech_start = (attempt_num - 1) * 3
    picked = [techniques[i % len(techniques)] for i in range(tech_start, tech_start + 4)]
    techniques_block = "\n".join(f"  - {t}" for t in picked)

    if ai_sentences:
        # ── Surgical mode: only AI-detected sentences are rewritten ──
        capped = ai_sentences[:30]
        sentences_block = "\n".join(f'  {i+1}. "{s}"' for i, s in enumerate(capped))
        user_prompt = f"""AI detection score: {current_score:.1f}% (must be < {ZEROGPT_PASS_THRESHOLD}%). Attempt #{attempt_num}.

Rewrite ONLY these AI-flagged sentences. Make each one sound like a real person wrote it — imperfect, casual, with personality.

AI-FLAGGED SENTENCES:
{sentences_block}

EXAMPLES — notice how the GOOD versions feel messy and real:
  BAD:  "It is important to note that exercise can significantly reduce stress levels."
  GOOD: "So here's something I didn't expect — working out actually helps with stress. Like, a lot."

  BAD:  "Research suggests that maintaining social connections is beneficial for mental health."
  GOOD: "Turns out, just texting a friend when you're stressed? That actually does something. Who knew."

  BAD:  "Implementing a consistent sleep schedule can improve overall well-being."
  GOOD: "I started going to bed at the same time every night. Felt weird at first. But honestly... it changed everything."

  BAD:  "These techniques can help individuals manage their stress more effectively."
  GOOD: "Look — these tricks won't fix everything. But they help. They really do."

FOR THIS ATTEMPT, ESPECIALLY USE THESE TECHNIQUES:
{techniques_block}

RULES:
- Rewrite ONLY the flagged sentences. Keep everything else EXACTLY as-is.
- Each rewrite must sound like a different human wrote it — vary the style.
- Use contractions everywhere (it's, don't, won't, can't, I've, that's).
- AVOID: {ai_clichés}
- Keep the same meaning and facts. Don't invent new info.
- CRITICAL: Do NOT add words. Keep the same or fewer words. Replace, don't expand.
- Return the COMPLETE article with replacements applied.

ARTICLE:
{full_markdown}"""

    else:
        # ── Full fallback: humanize the entire document ──
        user_prompt = f"""AI detection score: {current_score:.1f}% (must be < {ZEROGPT_PASS_THRESHOLD}%). Attempt #{attempt_num}.

Rewrite this entire article so it reads like a real human blog — someone with opinions, quirks, and a casual voice. NOT a polished AI essay.

TARGET STYLE — notice the imperfections, personality, and casual flow:
---
Okay so here's the thing about stress. We all deal with it. Every single one of us. And most of the advice out there? It's the same recycled stuff — "just meditate" or "take deep breaths." Cool. Thanks.

But here's what actually helped me. I started small. Like embarrassingly small. I'd take a walk around the block when I felt overwhelmed. That's it. No gym membership, no yoga retreat. Just... walking.

And you know what? It worked. Not perfectly. Not overnight. But slowly, things got a little easier. The tight feeling in my chest would loosen up. My brain would quiet down — at least for a bit.

Look, I'm not a doctor. I'm just someone who's been through it. So take what works for you and leave the rest. Deal?
---

FOR THIS ATTEMPT, ESPECIALLY USE THESE TECHNIQUES:
{techniques_block}

RULES:
- Keep ALL headings, sections, facts, and information in the same order.
- Do NOT remove or invent content. Same info, different voice.
- Write like you're telling a friend about this over coffee.
- Use contractions everywhere. Use fragments. Ask questions.
- Add personal touches: "I think", "honestly", "in my experience", "here's what I found"
- AVOID: {ai_clichés}
- Vary sentence length wildly: some 3 words, some 20 words.
- CRITICAL: Do NOT add words. Keep the same or fewer words. Replace, don't expand.
- Return the COMPLETE rewritten article in markdown.

ARTICLE:
{full_markdown}"""

    return system_prompt, user_prompt


# ============================================================
# Request / Response models
# ============================================================
class ZeroGPTFixRequest(BaseModel):
    model: str = ""  # auto-detected: OpenAI GPT-5.2 if key set, else Groq
    temperature: float = 0.9           # High temp for varied, unpredictable human-like output
    max_output_tokens: int = 8192
    max_attempts: int = 8
    force: bool = False                # Re-run even if already zerogpt_pass=true


class ZeroGPTFixResponse(BaseModel):
    status: str
    request_id: str
    tenant_id: str
    kb_id: str
    attempt_no: int

    # ZeroGPT result
    zerogpt_pass: bool
    zerogpt_pass_threshold: float = ZEROGPT_PASS_THRESHOLD
    initial_score: Optional[float] = None
    final_score: Optional[float] = None
    attempts_used: int

    # Artifacts
    gcs_humanized_uri: str
    humanized_fingerprint: str
    gcs_zerogpt_uri: str
    zerogpt_fingerprint: str

    humanize_mode: str          # "surgical" or "full_fallback"
    usage: Dict[str, Any] = Field(default_factory=dict)
    meta: Dict[str, Any] = Field(default_factory=dict)


# ============================================================
# Endpoint
# ============================================================
@router.post("/requests/{request_id}/zerogpt-fix", response_model=ZeroGPTFixResponse)
def zerogpt_fix(
    request_id: str,
    req: ZeroGPTFixRequest,
    claims: Claims = Depends(require_claims),
):
    """
    Day 22 — ZeroGPT-Fix:
    Humanizes AI-detected content using Kimi K2 until fakePercentage < 10%.
    Uses surgical mode (only AI-flagged sentences) when ZeroGPT returns spans.
    Falls back to full-document rewrite when spans are empty.
    Max 5 attempts with stall detection (stops if score stops improving).
    """
    _require_admin(claims)
    settings = get_settings()
    tenant_id = claims.tenant_id

    groq_key = os.getenv("GROQ_API_KEY")
    if not groq_key:
        raise HTTPException(status_code=500, detail="Missing GROQ_API_KEY")

    zerogpt_api_key = os.getenv("ZEROGPT_API_KEY")
    zerogpt_base_url = os.getenv("ZEROGPT_BASE_URL")
    if not zerogpt_api_key or not zerogpt_base_url:
        raise HTTPException(status_code=500, detail="Missing ZEROGPT_API_KEY or ZEROGPT_BASE_URL")

    bucket_name = _pick(settings, "GCS_BUCKET_NAME")
    if not bucket_name:
        raise HTTPException(status_code=500, detail="Missing GCS_BUCKET_NAME")

    articles_prefix = _norm_prefix(_pick(settings, "GCS_PREFIX_ARTICLES", default="articles/"))
    gcs = _gcs_client(settings)

    # ── Load request row ──────────────────────────────────────
    with _db_conn(settings) as conn:
        _ensure_fix_columns(conn)
        conn.commit()

        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT request_id, tenant_id, kb_id, attempt_count,
                       gcs_draft_uri, draft_fingerprint, title,
                       gcs_qc_uri, qc_fingerprint,
                       gcs_zerogpt_uri, zerogpt_fingerprint,
                       zerogpt_score, zerogpt_pass, zerogpt_meta
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
    title = str(row.get("title") or "")

    # Require draft to exist
    gcs_draft_uri = row.get("gcs_draft_uri")
    draft_fp = row.get("draft_fingerprint")
    if not gcs_draft_uri or not draft_fp:
        raise HTTPException(status_code=409, detail="No draft found. Run /run first.")

    # Require QC to have passed (zerogpt-fix must come after qc-fix)
    gcs_qc_uri = row.get("gcs_qc_uri")
    if not gcs_qc_uri:
        raise HTTPException(status_code=409, detail="QC not run yet. Run /qc then /qc-fix first.")

    try:
        qc_obj = _gcs_download_json(gcs, str(gcs_qc_uri))
        if not qc_obj.get("qc_pass"):
            raise HTTPException(status_code=409, detail="QC failed. Fix draft with /qc-fix before running zerogpt-fix.")
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=502, detail="Failed to read QC artifact from GCS.")

    # Require /zerogpt to have been run first (we need the initial score + sentences)
    gcs_zerogpt_uri = row.get("gcs_zerogpt_uri")
    if not gcs_zerogpt_uri:
        raise HTTPException(status_code=409, detail="ZeroGPT not run yet. Call /zerogpt first.")

    current_zerogpt_pass = row.get("zerogpt_pass")

    # Early exit if already passing (unless force=True)
    if current_zerogpt_pass is True and not req.force:
        current_score = row.get("zerogpt_score")
        return ZeroGPTFixResponse(
            status="ok",
            request_id=request_id,
            tenant_id=tenant_id,
            kb_id=kb_id,
            attempt_no=attempt_no,
            zerogpt_pass=True,
            initial_score=float(current_score) if current_score is not None else None,
            final_score=float(current_score) if current_score is not None else None,
            attempts_used=0,
            gcs_humanized_uri=str(row.get("gcs_humanized_uri") or gcs_draft_uri),
            humanized_fingerprint=str(row.get("humanized_fingerprint") or draft_fp),
            gcs_zerogpt_uri=str(gcs_zerogpt_uri),
            zerogpt_fingerprint=str(row.get("zerogpt_fingerprint") or ""),
            humanize_mode="already_passed",
            meta={"cached": True},
        )

    # ── Download current draft markdown ───────────────────────
    try:
        draft_bytes = _gcs_download_bytes(gcs, str(gcs_draft_uri))
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Failed to download draft from GCS: {str(e)[:200]}")

    try:
        draft_artifact = json.loads(draft_bytes.decode("utf-8", errors="replace"))
    except Exception:
        draft_artifact = {}

    draft_md = _strip_md_noise(_extract_draft_markdown(draft_artifact))
    if not draft_md.strip():
        raise HTTPException(status_code=422, detail="Draft markdown is empty; cannot humanize.")

    # ── Load initial ZeroGPT result (score + sentences) ───────
    try:
        zerogpt_artifact = _gcs_download_json(gcs, str(gcs_zerogpt_uri))
    except Exception:
        zerogpt_artifact = {}

    # Extract initial score and AI sentences from stored zerogpt artifact
    initial_score: Optional[float] = None
    initial_sentences: List[Any] = []
    try:
        zgpt_data = zerogpt_artifact.get("zerogpt", {})
        initial_score = zgpt_data.get("score_fakePercentage")
        if initial_score is not None:
            initial_score = float(initial_score)
        raw_resp = zgpt_data.get("raw") or {}
        data_block = raw_resp.get("data") if isinstance(raw_resp, dict) else None
        if isinstance(data_block, dict):
            # BUG FIX: use 'h' field (AI sentences), not 'sentences' (always empty)
            initial_sentences = data_block.get("h") or data_block.get("sentences") or []
            if not isinstance(initial_sentences, list):
                initial_sentences = []
    except Exception:
        initial_score = None
        initial_sentences = []

    # Fallback: use DB score if artifact parse failed
    if initial_score is None:
        initial_score = row.get("zerogpt_score")
        if initial_score is not None:
            initial_score = float(initial_score)

    if initial_score is None:
        raise HTTPException(status_code=409, detail="Could not read initial ZeroGPT score. Re-run /zerogpt first.")

    # If score already < threshold (score stored but pass flag not set yet)
    if initial_score < ZEROGPT_PASS_THRESHOLD and not req.force:
        # Just update the pass flag and return
        with _db_conn(settings) as conn:
            _ensure_fix_columns(conn)
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE public.article_requests SET zerogpt_pass=%s, updated_at=%s "
                    "WHERE tenant_id=%s::uuid AND request_id=%s::uuid",
                    (True, datetime.now(timezone.utc), tenant_id, request_id),
                )
            conn.commit()
        return ZeroGPTFixResponse(
            status="ok",
            request_id=request_id,
            tenant_id=tenant_id,
            kb_id=kb_id,
            attempt_no=attempt_no,
            zerogpt_pass=True,
            initial_score=initial_score,
            final_score=initial_score,
            attempts_used=0,
            gcs_humanized_uri=str(gcs_draft_uri),
            humanized_fingerprint=str(draft_fp),
            gcs_zerogpt_uri=str(gcs_zerogpt_uri),
            zerogpt_fingerprint=str(row.get("zerogpt_fingerprint") or ""),
            humanize_mode="already_passed",
            meta={"note": "Score was already below threshold, updated pass flag."},
        )

    # ── Humanization loop ─────────────────────────────────────
    max_attempts = max(1, min(int(req.max_attempts), 10))  # hard cap at 10
    max_tokens = max(int(req.max_output_tokens), 4096)

    best_md = draft_md
    best_score: float = initial_score
    best_sentences: List[Any] = initial_sentences
    zerogpt_pass_flag = False
    stall_count = 0
    STALL_THRESHOLD = 1.0   # score must improve by at least 1% to not count as stall
    MAX_STALLS = 3
    surgical_stall_switched = False  # track if we switched from surgical to full

    # Determine humanize mode: surgical if sentences available, fallback if empty
    humanize_mode = "surgical" if _extract_ai_sentences(initial_sentences) else "full_fallback"

    usage_total = {"prompt_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    attempts_used = 0

    # Store the latest zerogpt artifact info (updated each successful check)
    latest_zerogpt_raw: Dict[str, Any] = {}
    latest_zerogpt_score: Optional[float] = best_score
    latest_zerogpt_sentences: List[Any] = best_sentences

    for attempt_idx in range(max_attempts):
        attempts_used = attempt_idx + 1

        # Use best version found so far as input for this attempt
        ai_sentences = _extract_ai_sentences(best_sentences)
        system_p, user_p = _build_humanize_prompts(
            full_markdown=best_md,
            ai_sentences=ai_sentences,
            current_score=best_score,
            attempt_num=attempt_idx + 1,
        )

        # Call LLM (OpenAI preferred, Groq fallback)
        try:
            new_md, usage = _humanize_llm(
                str(groq_key),
                model=req.model,
                system_prompt=system_p,
                user_prompt=user_p,
                temperature=float(req.temperature),
                max_tokens=max_tokens,
                timeout_s=180,
                retries=2,
            )
        except HTTPException:
            # LLM call failed — count as stall, try again next round
            stall_count += 1
            if stall_count >= MAX_STALLS:
                break
            continue

        for k in usage_total:
            usage_total[k] += int(usage.get(k) or 0)

        new_md = _strip_md_noise(new_md)
        if not new_md.strip():
            stall_count += 1
            if stall_count >= MAX_STALLS:
                break
            continue

        # Small delay to respect ZeroGPT rate limits
        time.sleep(1)

        # Re-check with ZeroGPT
        try:
            new_score, new_sentences, new_raw = _zerogpt_check(
                base_url=str(zerogpt_base_url),
                api_key=str(zerogpt_api_key),
                text=new_md,
                timeout_s=60,
            )
        except HTTPException:
            # ZeroGPT call failed — keep best and count as stall
            stall_count += 1
            if stall_count >= MAX_STALLS:
                break
            continue

        # Update latest zerogpt info regardless (for saving at end)
        latest_zerogpt_raw = new_raw
        latest_zerogpt_score = new_score
        latest_zerogpt_sentences = new_sentences

        if new_score is None:
            stall_count += 1
            if stall_count >= MAX_STALLS:
                break
            continue

        # Did this attempt improve on the best?
        improvement = best_score - new_score  # positive = lower AI % = better
        if improvement >= STALL_THRESHOLD:
            best_md = new_md
            best_score = new_score
            best_sentences = new_sentences
            stall_count = 0
        else:
            stall_count += 1

        if best_score < ZEROGPT_PASS_THRESHOLD:
            zerogpt_pass_flag = True
            break

        if stall_count >= MAX_STALLS:
            # If surgical mode stalled, switch to full rewrite and reset stall counter
            if humanize_mode == "surgical" and not surgical_stall_switched:
                humanize_mode = "full_fallback"
                surgical_stall_switched = True
                stall_count = 0
                continue
            break

    # Update mode based on what we actually ended up using
    final_ai_sentences = _extract_ai_sentences(best_sentences)
    if surgical_stall_switched:
        humanize_mode = "surgical_then_full"

    # ── Save humanized draft artifact to GCS ──────────────────
    humanized_artifact = {
        "request_id": request_id,
        "tenant_id": tenant_id,
        "kb_id": kb_id,
        "attempt_no": attempt_no,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": {"provider": "groq", "name": req.model},
        "zerogpt_fix": {
            "initial_score": initial_score,
            "final_score": best_score,
            "attempts_used": attempts_used,
            "zerogpt_pass": zerogpt_pass_flag,
            "humanize_mode": humanize_mode,
            "threshold": ZEROGPT_PASS_THRESHOLD,
        },
        "draft": {"title": title, "draft_markdown": best_md},
    }
    humanized_bytes = json.dumps(humanized_artifact, ensure_ascii=False, indent=2).encode("utf-8")
    humanized_fp = _sha256(humanized_bytes)

    humanized_obj = (
        f"{articles_prefix}{tenant_id}/{kb_id}/{request_id}/attempt_{attempt_no}/"
        f"zerogpt_fix_v1/{humanized_fp}.json"
    )
    gcs_humanized_uri = _gcs_put_create_only(gcs, bucket_name, humanized_obj, humanized_bytes, "application/json; charset=utf-8")

    # ── Save final ZeroGPT result artifact to GCS ─────────────
    zerogpt_result_artifact = {
        "request_id": request_id,
        "tenant_id": tenant_id,
        "kb_id": kb_id,
        "attempt_no": attempt_no,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": "zerogpt_fix",
        "input": {
            "gcs_humanized_uri": gcs_humanized_uri,
            "humanized_fingerprint": humanized_fp,
        },
        "zerogpt": {
            "score_fakePercentage": best_score,
            "zerogpt_pass": zerogpt_pass_flag,
            "raw": latest_zerogpt_raw,
        },
    }
    zerogpt_bytes = json.dumps(zerogpt_result_artifact, ensure_ascii=False, indent=2).encode("utf-8")
    zerogpt_fp = _sha256(zerogpt_bytes)

    zerogpt_obj = (
        f"{articles_prefix}{tenant_id}/{kb_id}/{request_id}/attempt_{attempt_no}/"
        f"zerogpt_fix_v1/zerogpt_{zerogpt_fp}.json"
    )
    gcs_new_zerogpt_uri = _gcs_put_create_only(gcs, bucket_name, zerogpt_obj, zerogpt_bytes, "application/json; charset=utf-8")

    # ── Update DB: replace draft pointer + store zerogpt fix results ──
    with _db_conn(settings) as conn:
        _ensure_fix_columns(conn)
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE public.article_requests
                SET
                    -- Humanized draft replaces the working draft
                    gcs_draft_uri=%s,
                    draft_fingerprint=%s,
                    -- ZeroGPT fix tracking
                    gcs_humanized_uri=%s,
                    humanized_fingerprint=%s,
                    zerogpt_fix_attempts=%s,
                    -- ZeroGPT results from final re-check
                    gcs_zerogpt_uri=%s,
                    zerogpt_fingerprint=%s,
                    zerogpt_score=%s,
                    zerogpt_pass=%s,
                    zerogpt_meta=%s::jsonb,
                    updated_at=%s
                WHERE tenant_id=%s::uuid AND request_id=%s::uuid
                """,
                (
                    gcs_humanized_uri,          # new gcs_draft_uri
                    humanized_fp,               # new draft_fingerprint
                    gcs_humanized_uri,          # gcs_humanized_uri (same, for audit)
                    humanized_fp,               # humanized_fingerprint
                    attempts_used,
                    gcs_new_zerogpt_uri,
                    zerogpt_fp,
                    best_score,
                    zerogpt_pass_flag,
                    json.dumps({
                        "initial_score": initial_score,
                        "final_score": best_score,
                        "attempts_used": attempts_used,
                        "humanize_mode": humanize_mode,
                        "threshold": ZEROGPT_PASS_THRESHOLD,
                    }),
                    datetime.now(timezone.utc),
                    tenant_id,
                    request_id,
                ),
            )
        _log_job_event(
            conn,
            tenant_id,
            "zerogpt_fix_saved",
            {
                "initial_score": initial_score,
                "final_score": best_score,
                "zerogpt_pass": zerogpt_pass_flag,
                "attempts_used": attempts_used,
                "humanize_mode": humanize_mode,
            },
            request_id=request_id,
        )
        conn.commit()

    return ZeroGPTFixResponse(
        status="ok",
        request_id=request_id,
        tenant_id=tenant_id,
        kb_id=kb_id,
        attempt_no=attempt_no,
        zerogpt_pass=zerogpt_pass_flag,
        initial_score=initial_score,
        final_score=best_score,
        attempts_used=attempts_used,
        gcs_humanized_uri=gcs_humanized_uri,
        humanized_fingerprint=humanized_fp,
        gcs_zerogpt_uri=gcs_new_zerogpt_uri,
        zerogpt_fingerprint=zerogpt_fp,
        humanize_mode=humanize_mode,
        usage=usage_total,
        meta={
            "threshold": ZEROGPT_PASS_THRESHOLD,
            "stall_count_at_end": stall_count,
        },
    )
