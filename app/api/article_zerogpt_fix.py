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


def _clean_markdown(text: str, max_sections: int = 16) -> str:
    """
    Post-process generated markdown to fix 4 GPT artifacts before GCS write:
    1. Broken bold in bullets: "- Word:** text" -> "- **Word:** text"
    2. Duplicate FAQ sections: keep first occurrence, drop the rest
    3. Excess sections: trim to max_sections (FAQ preserved even if late)
    4. Duplicate table header rows: remove repeated header after separator
    """
    if not text:
        return text

    # ── Fix 1: Broken bold in bullet items ─────────────────────────────────
    # Catches: "- Innings pacing:** text" (opening ** missing)
    lines = text.split("\n")
    text = "\n".join(
        re.sub(r'^(\s*[-*]\s+)([A-Z][^*\n:]+):\*\*', r'\1**\2:**', line)
        for line in lines
    )

    # ── Fix 2: Deduplicate FAQ sections (keep first) ────────────────────────
    lines = text.split("\n")
    result: list = []
    faq_seen = False
    in_dup_faq = False
    for line in lines:
        if re.match(r"^#{1,3}\s", line):
            if re.search(r"\bfaq\b", line, re.I):
                if faq_seen:
                    in_dup_faq = True
                    continue
                faq_seen = True
                in_dup_faq = False
            else:
                in_dup_faq = False
        if not in_dup_faq:
            result.append(line)
    text = "\n".join(result)

    # ── Fix 3: Trim sections beyond max_sections ────────────────────────────
    lines = text.split("\n")
    sec_pos: list = []  # list of (line_index, is_faq)
    for i, line in enumerate(lines):
        if re.match(r"^##\s", line):
            sec_pos.append((i, bool(re.search(r"\bfaq\b", line, re.I))))

    if len(sec_pos) > max_sections:
        faq_order = next((idx for idx, (_, is_f) in enumerate(sec_pos) if is_f), None)
        if faq_order is not None and faq_order < max_sections:
            # FAQ is within limit  -  simple trim
            lines = lines[: sec_pos[max_sections][0]]
        elif faq_order is not None:
            # FAQ is beyond limit  -  keep first (max-1) + FAQ block
            cutoff = sec_pos[max_sections - 1][0]
            faq_start = sec_pos[faq_order][0]
            faq_end = sec_pos[faq_order + 1][0] if faq_order + 1 < len(sec_pos) else len(lines)
            lines = lines[:cutoff] + [""] + lines[faq_start:faq_end]
        else:
            lines = lines[: sec_pos[max_sections][0]]
        text = "\n".join(lines).rstrip()

    # ── Fix 4: Remove duplicate table header rows ───────────────────────────
    lines = text.split("\n")
    cleaned: list = []
    i = 0
    while i < len(lines):
        line = lines[i]
        is_sep = bool(re.match(r"^\s*\|[-:| ]+\|\s*$", line.strip()))
        if is_sep and cleaned and i + 1 < len(lines):
            header = cleaned[-1].strip()
            if header.startswith("|") and lines[i + 1].strip() == header:
                cleaned.append(line)   # keep separator
                i += 2                 # skip duplicate header row
                continue
        cleaned.append(line)
        i += 1

    return "\n".join(cleaned).strip()


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
        # 'sentences' is always []  -  'h' contains the actual AI-flagged sentence strings
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
# Kimi K2 plain-text humanize call (no JSON mode  -  produces much longer output)
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
        # NO response_format  -  plain text mode, K2 produces 2-3x longer output
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

    Professional editorial mode: reduces AI detection by targeting
    token-level predictability  -  not by adding casual language.
    """
    system_prompt = (
        "You are a professional editorial rewriter. Your task: reduce AI detection scores "
        "while preserving professional tone throughout.\n\n"
        "HOW AI DETECTION WORKS  -  understand this to fix sentences correctly:\n"
        "ZeroGPT measures TOKEN-LEVEL PREDICTABILITY. It flags sentences where every word is "
        "exactly what a language model expects to produce next. A sentence like 'The policy has "
        "been consistently linked to improved outcomes across all measured categories' scores high "
        "because every token is the obvious continuation. A sentence like 'The policy changed one "
        "thing. Outcomes followed. The data confirmed it.' scores low because the split and "
        "abruptness are unexpected.\n\n"
        "PROFESSIONAL PERPLEXITY TECHNIQUES  -  these reduce detection without casual language:\n\n"
        "1. SENTENCE SPLITTING  -  break one smooth long sentence into 2-3 short declaratives:\n"
        "   FLAGGED: 'The team's batting lineup has been consistently strong throughout the season due to depth.'\n"
        "   FIXED:   'The batting lineup has depth. Three players averaging above 40. That cushion changes chase dynamics.'\n\n"
        "2. NEGATION CHAIN  -  state what something is NOT before what it IS:\n"
        "   FLAGGED: 'Cloud computing offers cost savings and operational flexibility for small businesses.'\n"
        "   FIXED:   'Not a guarantee. Not a one-size outcome. A structural advantage  -  when implemented deliberately.'\n\n"
        "3. INVERTED COLON REVEAL  -  setup first, specific finding after the colon:\n"
        "   FLAGGED: 'Social connection significantly reduces stress and improves mental health outcomes.'\n"
        "   FIXED:   'What social connection actually changes: not just mood  -  baseline physiological arousal.'\n\n"
        "4. TEMPORAL CONTRAST  -  anchor abstract claims to specific time points:\n"
        "   FLAGGED: 'Hybrid cloud adoption has been growing steadily among organizations of all sizes.'\n"
        "   FIXED:   'In 2024, hybrid cloud adoption was measured at a modest baseline. By end of 2025, it had risen 35%.'\n\n"
        "5. CONCESSIVE PIVOT  -  acknowledge the expected view, then turn:\n"
        "   FLAGGED: 'Despite security concerns, businesses continue to adopt cloud services at a rapid rate.'\n"
        "   FIXED:   'Security concerns are legitimate. Still, aging hardware and single points of failure carry their own risk register.'\n\n"
        "6. SHORT BLUNT VERDICT  -  follow long analysis with a 3-6 word observation:\n"
        "   FLAGGED: 'This bidirectional relationship between sleep deprivation and stress reactivity creates a compounding negative cycle.'\n"
        "   FIXED:   'Stress damages sleep. Poor sleep sharpens stress the next day. The loop feeds itself.'\n\n"
        "7. EM-DASH INTERRUPTION  -  insert a specific observation mid-sentence:\n"
        "   FLAGGED: 'Cortisol and adrenaline are stress hormones that can negatively affect health when chronically elevated.'\n"
        "   FIXED:   'Cortisol and adrenaline  -  the body's primary threat chemicals  -  become corrosive when they stay elevated for weeks.'\n\n"
        "8. CONTRAST PAIR  -  two short sentences where the second inverts or sharpens the first:\n"
        "   FLAGGED: 'Smaller teams tend to struggle more while larger teams with more resources typically perform better.'\n"
        "   FIXED:   'Resources don't guarantee outcomes. Smaller teams often move faster.'\n\n"
        "PROFESSIONAL STANDARDS  -  never compromise these:\n"
        "- No casual slang: no 'yeah', 'honestly', 'basically', 'kind of', 'I mean', 'plot twist'\n"
        "- No personal anecdotes: no 'I think', 'in my experience', 'a friend told me'\n"
        "- No reader questions: no 'Right?', 'Sound familiar?', 'See what I mean?'\n"
        "- No casual transitions: no 'Okay but', 'Here's the thing', 'So anyway', 'Look'\n"
        "- Professional pivots only: 'However', 'Meanwhile', 'By contrast', 'Still', 'What followed'\n"
        "- Contractions are fine in editorial writing: 'it's', 'don't', 'that's', 'they've'\n\n"
        "WRITING PROHIBITION  -  NEVER write in these styles regardless of article content:\n"
        "- NEVER write kid-simple guides, tiny checklists, beginner step-by-step sections, or child-level analogies.\n"
        "- NEVER use talk-down comparisons: 'like a video game', 'like a school sports day', 'like a cookie jar', 'like pocket money'.\n"
        "- NEVER exceed 12 sections total  -  cut or merge sections if the article has more.\n"
        "- NEVER write below Grade 7 reading level. Short punches are for emphasis, never for simplification.\n\n"
        "ABSOLUTE OUTPUT RULES  -  violations break the article:\n"
        "- NEVER write technique labels in the output text. Labels like 'Concessive pivot:', "
        "'Negation chain:', 'Contrast pair:' are for your reference ONLY. They must NEVER appear "
        "in the article. Apply each technique invisibly  -  the reader should never see a label.\n"
        "- NEVER reproduce example sentences verbatim. The examples above illustrate technique "
        "structure only. Do not copy them word-for-word into the article.\n"
        "- NEVER use first-person source references. Remove any 'I couldn't find...', "
        "'The sources suggest...', 'Based on the provided data...'  -  replace with neutral "
        "editorial phrasing ('Granular breakdowns were not available') or remove entirely.\n\n"
        "CRITICAL RULE: Do NOT add words. Every rewrite must use the SAME OR FEWER words. "
        "Restructure and reorder  -  never expand.\n\n"
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

    # Professional structural techniques  -  rotate per attempt for variety
    # Each technique targets a different sentence pattern that triggers AI detection
    techniques = [
        "Split long smooth sentences into 2-3 short declaratives. No filler, just facts.",
        "Use negation chains: 'Not [X]. Not [Y]. Just [specific Z].'  -  forces unpredictable token sequences.",
        "Use inverted colon reveals: 'What [X] actually [does/means]: [unexpected specific detail].'",
        "Add temporal contrast: 'In [year/period], [X]. By [year/period], [Y].'  -  anchors abstraction to time.",
        "Use concessive pivot: 'Still, [unexpected turn].' or 'By contrast, [sharp observation].'",
        "Add em-dash interruption mid-sentence: '[subject]  -  [specific inserted detail]  -  [predicate].'",
        "Use contrast pairs: '[Short declarative]. [Its inversion or consequence].'  -  two punches.",
        "Replace smooth causal chains ('X because Y, which leads to Z') with a sequence of short observations.",
    ]
    # Pick 4 different techniques per attempt  -  rotate through the list
    tech_start = (attempt_num - 1) * 3
    picked = [techniques[i % len(techniques)] for i in range(tech_start, tech_start + 4)]
    techniques_block = "\n".join(f"  - {t}" for t in picked)

    if ai_sentences:
        # ── Surgical mode: only AI-detected sentences are rewritten ──
        capped = ai_sentences[:30]
        sentences_block = "\n".join(f'  {i+1}. "{s}"' for i, s in enumerate(capped))
        user_prompt = f"""AI detection score: {current_score:.1f}% (target: < {ZEROGPT_PASS_THRESHOLD}%). Attempt #{attempt_num}.

Rewrite ONLY the AI-flagged sentences below using professional perplexity techniques.
Every sentence in the ARTICLE that is NOT flagged must remain EXACTLY as written.

AI-FLAGGED SENTENCES:
{sentences_block}

PROFESSIONAL REWRITE EXAMPLES  -  technique structure only, do NOT copy these verbatim:

  FLAGGED: "The policy has been consistently linked to better outcomes because it addresses core structural issues."
  FIXED:   "The policy changed one variable. Outcomes shifted within a quarter. The link held across all cohorts."

  FLAGGED: "Despite ongoing challenges, organizations continue to invest in digital transformation at a growing rate."
  FIXED:   "Challenges are real. Still, the investment case hasn't weakened  -  downtime at $427 per minute tends to focus priorities."

  FLAGGED: "Teams that maintain strong opening partnerships tend to win more matches than those that do not."
  FIXED:   "When they open well, they average 77. When they don't, 21. That gap explains most of the results table."

  FLAGGED: "The season has seen a significant increase in dropped catches compared to previous years."
  FIXED:   "111 catches dropped in 40 games. Not a blip  -  a systemic pattern, worst since 2020."

  FLAGGED: "Analysts have noted that death-over execution remains the primary differentiator in close contests."
  FIXED:   "Death overs decide it. Not the powerplay. Not the middle overs. The last four  -  every time."

FOR THIS ATTEMPT, PRIORITIZE THESE TECHNIQUES:
{techniques_block}

RULES:
- Rewrite ONLY the numbered flagged sentences. Every other sentence stays EXACTLY as-is.
- Match the professional editorial tone already present in the article  -  look at the surrounding sentences.
- Use contractions where natural (it's, don't, that's, they've)  -  fine in editorial writing.
- AVOID AI clichés: {ai_clichés}
- Keep the same facts and meaning. Do NOT invent new information.
- NEVER write technique labels (Concessive pivot:, Negation chain:, etc.) in the article text.
- NEVER reproduce example sentences verbatim  -  examples show structure, not wording to copy.
- Remove any first-person source notes ('I couldn't find...')  -  use 'Granular data unavailable' or omit.
- BOLD FORMATTING: reduce where excessive. Bold only player names, team names, and the single most critical stat per section. Do NOT bold full analytical clauses or sentences  -  it flags as AI.
- CRITICAL: Do NOT add words. Same or fewer words per rewrite. Restructure, never expand.
- Return the COMPLETE article with replacements applied inline.

ARTICLE:
{full_markdown}"""

    else:
        # ── Full fallback: rewrite the entire document (surgical stalled) ──
        user_prompt = f"""AI detection score: {current_score:.1f}% (target: < {ZEROGPT_PASS_THRESHOLD}%). Attempt #{attempt_num}.

Rewrite this entire article using professional perplexity techniques.
The article should read as published editorial content  -  authoritative, varied, and structurally unpredictable.

TARGET STYLE  -  notice structural variety: short punches, splits, temporal contrast, blunt verdicts:
---
## Week 5: the week the table started to mean something

Four teams moved.

Gujarat Titans chased 204  -  first time in franchise history  -  and went top. Mumbai Indians won their fourth straight, quietly, by turning the Wankhede into a surface SRH's batting had no answer for. RCB lost again at home. SRH's pre-season favouritism is now a historical footnote: six defeats in eight.

What this means for the standings: the top-four shape is clearer, and the teams outside it are running low on margin.

## Dropped catches: 111 in 40 games

That figure deserves attention.

Not because one team is responsible  -  the problem is distributed. But 111 drops across 40 matches is the worst rate since 2020, a season with COVID context that distorts comparison. The 2025 version has no such excuse.

Each drop is a compound-interest mistake. You pay once at the boundary. Then you pay again when the batter reaches 50. Then again when they change the match.

Death overs don't forgive either. LSG held their nerve in Jaipur: Avesh Khan, final over, 9 needed. Two runs the margin. Old-school execution  -  yorkers, wide lines, fielder placement  -  still decides games that technology can't script.
---

FOR THIS ATTEMPT, PRIORITIZE THESE TECHNIQUES:
{techniques_block}

RULES:
- Preserve ALL headings, sections, facts, and information  -  same order, same structure.
- Do NOT remove or invent content. Same information, different construction.
- Vary sentence length throughout: some 3-6 words, some 18-24 words. Never three consecutive sentences of similar length.
- Use contractions where natural (it's, don't, that's). Professional writing allows these.
- AVOID AI clichés: {ai_clichés}
- CRITICAL: Do NOT add words overall. Keep total word count the same or lower.
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
    Day 22  -  ZeroGPT-Fix:
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
        # ZeroGPT score unavailable (rate limit or API error on prior check).
        # Use conservative fallback  -  assume high AI score and run full rewrite.
        initial_score = 100.0
        initial_sentences = []

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
            # LLM call failed  -  count as stall, try again next round
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
            # ZeroGPT call failed  -  keep best and count as stall
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

    # Clean markdown artifacts before writing to GCS  -  fixes section count,
    # duplicate FAQ, broken bold, and duplicate table headers.
    best_md = _clean_markdown(best_md)

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
