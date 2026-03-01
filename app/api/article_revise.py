from __future__ import annotations

import hashlib
import json
import os
import re
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple

import psycopg2
from psycopg2.extras import RealDictCursor, Json
import requests
from fastapi import APIRouter, Depends, Header, HTTPException
from jose import JWTError, jwt
from pydantic import BaseModel, Field
from google.cloud import storage
from google.api_core.exceptions import PreconditionFailed

router = APIRouter(prefix="/api/v1/articles", tags=["article-qc-fix"])


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
        n2 = n.lower()
        if hasattr(settings, n2) and getattr(settings, n2) not in (None, ""):
            return getattr(settings, n2)
        v = os.getenv(n)
        if v not in (None, ""):
            return v
    return default


# -----------------------------
# Auth
# -----------------------------
class Claims(BaseModel):
    tenant_id: str
    user_id: str
    role: str
    exp: int


def require_claims(authorization: str = Header(...)) -> Claims:
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
        raise HTTPException(status_code=401, detail="Token missing claims")
    return Claims(tenant_id=str(tenant_id), user_id=str(user_id), role=str(role), exp=int(exp))


def _require_admin(claims: Claims) -> None:
    if (claims.role or "").lower() != "tenant_admin":
        raise HTTPException(status_code=403, detail="Only tenant_admin allowed")


# -----------------------------
# DB
# -----------------------------
def _db_conn(settings: Any):
    return psycopg2.connect(
        host=_pick(settings, "DB_HOST"),
        port=int(_pick(settings, "DB_PORT", default=5432)),
        dbname=_pick(settings, "DB_NAME", default="postgres"),
        user=_pick(settings, "DB_USER"),
        password=_pick(settings, "DB_PASSWORD"),
        sslmode=_pick(settings, "DB_SSLMODE", default="require"),
        connect_timeout=8,
    )


def _ensure_schema(conn) -> None:
    with conn.cursor() as cur:
        cur.execute("ALTER TABLE public.article_requests ADD COLUMN IF NOT EXISTS last_error_code text;")
        cur.execute("ALTER TABLE public.article_requests ADD COLUMN IF NOT EXISTS last_error_detail jsonb;")
        cur.execute("ALTER TABLE public.article_requests ADD COLUMN IF NOT EXISTS gcs_draft_uri text;")
        cur.execute("ALTER TABLE public.article_requests ADD COLUMN IF NOT EXISTS draft_fingerprint text;")
        cur.execute("ALTER TABLE public.article_requests ADD COLUMN IF NOT EXISTS draft_model text;")
        cur.execute("ALTER TABLE public.article_requests ADD COLUMN IF NOT EXISTS draft_meta jsonb;")
        cur.execute("ALTER TABLE public.article_requests ADD COLUMN IF NOT EXISTS gcs_qc_uri text;")
        cur.execute("ALTER TABLE public.article_requests ADD COLUMN IF NOT EXISTS qc_fingerprint text;")
        cur.execute("ALTER TABLE public.article_requests ADD COLUMN IF NOT EXISTS qc_meta jsonb;")
        cur.execute("ALTER TABLE public.article_requests ADD COLUMN IF NOT EXISTS qc_summary jsonb;")

        cur.execute("ALTER TABLE public.article_requests ADD COLUMN IF NOT EXISTS gcs_zerogpt_uri text;")
        cur.execute("ALTER TABLE public.article_requests ADD COLUMN IF NOT EXISTS zerogpt_fingerprint text;")
        cur.execute("ALTER TABLE public.article_requests ADD COLUMN IF NOT EXISTS zerogpt_score double precision;")
        cur.execute("ALTER TABLE public.article_requests ADD COLUMN IF NOT EXISTS zerogpt_meta jsonb;")

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


def _log_job_event(conn, tenant_id: str, event_type: str, detail: Dict[str, Any], request_id: Optional[str] = None):
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO public.job_events (event_id, tenant_id, request_id, job_id, event_type, detail)
            VALUES (%s,%s,%s,%s,%s,%s)
            """,
            (str(uuid.uuid4()), tenant_id, request_id, None, event_type, Json(detail)),
        )


def _try_lock(conn, job_id: str) -> Tuple[bool, str]:
    tok = str(uuid.uuid4())
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO public.job_locks (job_id, lock_token, locked_at, expires_at)
            VALUES (%s,%s,now(),now()+ interval '30 minutes')
            ON CONFLICT (job_id) DO NOTHING
            """,
            (job_id, tok),
        )
        if cur.rowcount == 1:
            return True, tok
        cur.execute(
            """
            UPDATE public.job_locks
            SET lock_token=%s, locked_at=now(), expires_at=now()+ interval '30 minutes'
            WHERE job_id=%s AND expires_at < now()
            """,
            (tok, job_id),
        )
        if cur.rowcount == 1:
            return True, tok
    return False, tok


def _unlock(conn, job_id: str, tok: str):
    with conn.cursor() as cur:
        cur.execute("DELETE FROM public.job_locks WHERE job_id=%s AND lock_token=%s", (job_id, tok))


# -----------------------------
# GCS
# -----------------------------
def _gcs_client(settings: Any) -> storage.Client:
    return storage.Client(project=_pick(settings, "GCP_PROJECT_ID", "GOOGLE_CLOUD_PROJECT"))


def _parse_gs_uri(gs: str) -> Tuple[str, str]:
    if not gs.startswith("gs://"):
        raise ValueError(gs)
    rest = gs[5:]
    parts = rest.split("/", 1)
    return parts[0], parts[1] if len(parts) > 1 else ""


def _gcs_get_bytes(gcs: storage.Client, gs_uri: str) -> bytes:
    b, o = _parse_gs_uri(gs_uri)
    blob = gcs.bucket(b).blob(o)
    if not blob.exists():
        raise FileNotFoundError(gs_uri)
    return blob.download_as_bytes()


def _gcs_put_create_only(gcs: storage.Client, bucket: str, obj: str, data: bytes, ctype: str) -> str:
    blob = gcs.bucket(bucket).blob(obj)
    try:
        blob.upload_from_string(data, content_type=ctype, if_generation_match=0)
    except PreconditionFailed:
        pass
    return f"gs://{bucket}/{obj}"


def _norm(p: str) -> str:
    p = (p or "").strip().replace("\\", "/")
    return p if (not p or p.endswith("/")) else p + "/"


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


# -----------------------------
# QC metrics
# -----------------------------
_WORD_RE = re.compile(r"[A-Za-z0-9']+")
_SENT_RE = re.compile(r"[.!?]+")


def _syll(word: str) -> int:
    w = re.sub(r"[^a-z]", "", word.lower())
    if not w:
        return 0
    vowels = "aeiouy"
    g = 0
    prev = False
    for ch in w:
        isv = ch in vowels
        if isv and not prev:
            g += 1
        prev = isv
    if w.endswith("e") and g > 1:
        g -= 1
    return max(g, 1)


def _qc(text: str) -> Dict[str, float]:
    words = _WORD_RE.findall(text or "")
    wc = len(words)
    sc = max(1, len(_SENT_RE.findall(text or "")) or 1)
    syll = sum(_syll(w) for w in words) if wc else 0
    wps = wc / sc
    spw = (syll / wc) if wc else 0.0
    flesch = 206.835 - 1.015 * wps - 84.6 * spw
    fk = 0.39 * wps + 11.8 * spw - 15.59
    return {
        "word_count": float(wc),
        "sentence_count": float(sc),
        "syllable_count": float(syll),
        "flesch_reading_ease": float(flesch),
        "flesch_kincaid_grade": float(fk),
    }


def _wc(md: str) -> int:
    return len(_WORD_RE.findall(md or ""))


def _clean_md(md: str) -> str:
    md = md or ""
    md = re.sub(r"```.*?```", "", md, flags=re.DOTALL)
    md = re.sub(r"\n{3,}", "\n\n", md).strip()
    return md


# Word replacements that REDUCE syllables WITHOUT changing meaning.
# Removed entries that made text robotic: "information"->"info", "understand"->"know",
# "example"->"case", "relationship"->"bond", "important"->"key", "because"->"since"
_SIMPLE_REPL = {
    # Genuinely pompous -> simple (big syllable drop, same meaning)
    "approximately": "about",       # 5 syl -> 2
    "utilize": "use",               # 3 -> 1
    "utilization": "use",           # 5 -> 1
    "commence": "start",            # 2 -> 1
    "terminate": "end",             # 3 -> 1
    "facilitate": "help",           # 4 -> 1
    "demonstrate": "show",          # 3 -> 1
    "consequently": "so",           # 4 -> 1
    "therefore": "so",              # 3 -> 1
    "sufficient": "enough",         # 3 -> 2
    "obtain": "get",                # 2 -> 1
    "assist": "help",               # 2 -> 1
    "assistance": "help",           # 3 -> 1
    "beneficial": "helpful",        # 4 -> 2
    "frequently": "often",          # 3 -> 2
    "individuals": "people",        # 5 -> 2
    "numerous": "many",             # 3 -> 2
    "primarily": "mainly",          # 4 -> 2
    "prior": "before",              # 2 -> 2 (clearer)
    "conclude": "end",              # 2 -> 1
    "objective": "goal",            # 3 -> 1
    "various": "many",              # 3 -> 2
}

# Only remove true filler words — "generally" and "mostly" carry meaning
_FILLER_RE = re.compile(r"\b(?:very|really|quite|just|simply|basically|actually)\b", re.IGNORECASE)


def _case_preserve(src: str, repl: str) -> str:
    if src.isupper():
        return repl.upper()
    if src[0].isupper():
        return repl.capitalize()
    return repl


def _simple_replace(text: str) -> str:
    if not text:
        return text
    out = text
    for k, v in _SIMPLE_REPL.items():
        pat = re.compile(rf"\b{k}\b", flags=re.IGNORECASE)

        def _r(m):
            return _case_preserve(m.group(0), v)

        out = pat.sub(_r, out)
    return out


def _split_chunks(text: str, max_words: int) -> list[str]:
    """Split text at natural sentence boundaries only (periods, !, ?).
    Never convert commas or semicolons to periods."""
    s = text.strip()
    if not s:
        return []
    # Split only at real sentence endings — do NOT touch commas/semicolons
    parts = re.split(r"(?<=[.!?])\s+", s)
    chunks: list[str] = []
    for part in parts:
        if not part:
            continue
        words = part.split()
        if len(words) <= max_words:
            chunks.append(part)
        else:
            for i in range(0, len(words), max_words):
                chunk = " ".join(words[i : i + max_words])
                if not chunk.endswith((".", "!", "?")):
                    chunk += "."
                chunks.append(chunk)
    return chunks


def _split_long_sentences(text: str, max_words: int = 20) -> str:
    """Only split sentences that are genuinely too long (>20 words).
    Preserve natural flow — never break at commas."""
    if not text:
        return text
    lines = text.splitlines()
    out_lines = []
    for line in lines:
        if not line:
            out_lines.append(line)
            continue
        if line.lstrip().startswith("#"):
            out_lines.append(line)
            continue
        m = re.match(r"^(\s*(?:[-*]|\d+\.)\s+)(.*)$", line)
        if m:
            prefix, body = m.group(1), m.group(2)
            for chunk in _split_chunks(body, max_words):
                out_lines.append(prefix + chunk)
            continue
        parts = _split_chunks(line, max_words)
        out_lines.append(" ".join(p.strip() for p in parts if p.strip()))
    return "\n".join(out_lines)


def _drop_fillers(text: str) -> str:
    if not text:
        return text
    out = _FILLER_RE.sub("", text)
    out = re.sub(r"\s{2,}", " ", out)
    return out


def _trim_tail_to_wc(text: str, max_wc: int) -> str:
    if not text or max_wc <= 0:
        return text
    if _wc(text) <= max_wc:
        return text
    parts = text.split("\n\n")
    i = len(parts) - 1
    while i >= 0 and _wc("\n\n".join(parts)) > max_wc:
        para = parts[i].strip()
        if not para:
            parts.pop(i)
            i -= 1
            continue
        sentences = re.split(r"(?<=[.!?])\s+", para)
        while len(sentences) > 1 and _wc("\n\n".join(parts[:i] + [" ".join(sentences)] + parts[i + 1 :])) > max_wc:
            sentences.pop()
        new_para = " ".join(s.strip() for s in sentences if s.strip())
        if new_para:
            parts[i] = new_para
        else:
            parts.pop(i)
        i -= 1
    return _clean_md("\n\n".join(parts))


def _post_simplify(text: str, wc_max: Optional[int] = None) -> str:
    out = _split_long_sentences(_simple_replace(text), max_words=20)
    if wc_max is not None and _wc(out) > wc_max:
        out = _drop_fillers(out)
    return _clean_md(out)


# -----------------------------
# Groq JSON
# -----------------------------
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"


def _safe_json(s: str) -> Optional[Dict[str, Any]]:
    s = (s or "").strip()
    if not s:
        return None
    try:
        o = json.loads(s)
        return o if isinstance(o, dict) else None
    except Exception:
        return None


def _extract_obj(s: str) -> Optional[str]:
    m = re.search(r"\{.*\}", s, flags=re.DOTALL)
    return m.group(0) if m else None


def _groq_json(
    key: str,
    *,
    model: str,
    system: str,
    prompt: str,
    temp: float,
    max_tokens: int,
    timeout_s: int,
    retries: int,
) -> Tuple[Dict[str, Any], Dict[str, int]]:
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    body = {
        "model": model,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
        "temperature": float(temp),
        "max_tokens": int(max_tokens),
        "response_format": {"type": "json_object"},
    }
    last = None
    for i in range(retries + 1):
        try:
            r = requests.post(GROQ_API_URL, headers=headers, json=body, timeout=timeout_s)
        except Exception as e:
            last = f"{type(e).__name__}:{str(e)[:150]}"
            time.sleep(min(2**i, 8))
            continue
        if r.status_code in (429, 500, 502, 503, 504):
            last = f"{r.status_code}:{r.text[:150]}"
            time.sleep(min(2**i, 8))
            continue
        if r.status_code != 200:
            last = f"{r.status_code}:{r.text[:300]}"
            break
        data = r.json()
        content = ""
        try:
            content = data["choices"][0]["message"]["content"]
        except Exception:
            content = ""
        parsed = _safe_json(content) or _safe_json(_extract_obj(content) or "")
        if not parsed:
            last = "non_json"
            break
        um = data.get("usage") or {}
        usage = {
            "prompt_tokens": int(um.get("prompt_tokens") or 0),
            "output_tokens": int(um.get("completion_tokens") or 0),
            "total_tokens": int(um.get("total_tokens") or 0),
        }
        return parsed, usage
    raise HTTPException(status_code=502, detail=f"Groq qc-fix failed. Last error: {last}")


# -----------------------------
# API models
# -----------------------------
class QCFixRequest(BaseModel):
    model: str = "llama-3.3-70b-versatile"
    temperature: float = 0.15
    max_output_tokens: int = 9000
    max_passes: int = 8  # deterministic loop needs more than 2


class QCFixResponse(BaseModel):
    status: str
    request_id: str
    tenant_id: str
    kb_id: str
    new_attempt_no: int
    old_draft_fingerprint: str
    new_draft_fingerprint: str
    gcs_new_draft_uri: str
    qc_pass: bool
    qc_metrics: Dict[str, Any]
    thresholds: Dict[str, Any]
    gcs_new_qc_uri: str
    qc_fingerprint: str
    usage: Dict[str, Any] = Field(default_factory=dict)


# -----------------------------
# Endpoint
# -----------------------------
@router.post("/requests/{request_id}/qc-fix", response_model=QCFixResponse)
def qc_fix(request_id: str, req: QCFixRequest, claims: Claims = Depends(require_claims)):
    """
    Day 21 (QC fix) — deterministic:
      1) expand_only until wc in [1950..2050]
      2) trim_only if wc > 2050
      3) simplify_only if FK > 9 (keep wc stable)
    Saves new attempt draft + qc report and updates article_requests pointers.
    """
    _require_admin(claims)
    settings = get_settings()
    tenant_id = claims.tenant_id

    groq_key = os.getenv("GROQ_API_KEY")
    if not groq_key:
        raise HTTPException(status_code=500, detail="Missing GROQ_API_KEY")

    bucket = _pick(settings, "GCS_BUCKET_NAME")
    if not bucket:
        raise HTTPException(status_code=500, detail="Missing GCS_BUCKET_NAME")

    gcs = _gcs_client(settings)
    articles_prefix = _norm(_pick(settings, "GCS_PREFIX_ARTICLES", default="articles/"))

    thresholds = {"wc_min": 1900, "wc_max": 2100, "fk_min": 7.0, "fk_max": 12.0}

    tok = ""
    locked = False

    # --- lock + fetch latest pointers ---
    with _db_conn(settings) as conn:
        _ensure_schema(conn)
        ok, tok = _try_lock(conn, request_id)
        if not ok:
            conn.commit()
            raise HTTPException(status_code=409, detail="Request is locked/in_progress. Try later.")
        locked = True

        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT request_id::text, tenant_id::text, kb_id::text, attempt_count,
                       gcs_draft_uri, draft_fingerprint, title, keywords
                FROM public.article_requests
                WHERE tenant_id=%s::uuid AND request_id=%s::uuid
                """,
                (tenant_id, request_id),
            )
            row = cur.fetchone()

        if not row:
            _unlock(conn, request_id, tok)
            conn.commit()
            raise HTTPException(status_code=404, detail="Request not found")

        kb_id = str(row["kb_id"])
        attempt = int(row.get("attempt_count") or 0)
        new_attempt = attempt + 1
        gcs_draft_uri = str(row["gcs_draft_uri"] or "")
        old_fp = str(row["draft_fingerprint"] or "")
        title = str(row.get("title") or "")
        keywords = list(row.get("keywords") or [])

        if not gcs_draft_uri or not old_fp:
            _unlock(conn, request_id, tok)
            conn.commit()
            raise HTTPException(status_code=409, detail="No draft found yet. Run /run first.")

        _log_job_event(conn, tenant_id, "qc_fix_started", {"from_attempt": attempt, "to_attempt": new_attempt}, request_id=request_id)
        conn.commit()

    try:
        # --- load draft markdown ---
        b = _gcs_get_bytes(gcs, gcs_draft_uri)
        try:
            obj = json.loads(b.decode("utf-8", errors="replace"))
        except Exception:
            obj = {}

        md = ""
        if isinstance(obj, dict):
            d = obj.get("draft")
            if isinstance(d, dict):
                md = str(d.get("draft_markdown") or "")
            else:
                md = str(obj.get("draft_markdown") or obj.get("content") or obj.get("text") or "")

        current = _clean_md(md)
        if not current:
            raise HTTPException(status_code=422, detail="Draft markdown empty; cannot qc-fix.")

        usage_total = {"prompt_tokens": 0, "output_tokens": 0, "total_tokens": 0}

        system = "Return ONLY valid JSON. No markdown. No extra keys."

        max_steps = max(20, int(req.max_passes))
        max_tokens = max(int(req.max_output_tokens), 7000)
        wc_min = thresholds["wc_min"]
        wc_max = thresholds["wc_max"]
        fk_min = thresholds["fk_min"]
        fk_max = thresholds["fk_max"]

        def _penalty(wc_val: int, fk_val: float) -> float:
            wc_pen = 0.0
            if wc_val < wc_min:
                wc_pen = (wc_min - wc_val) / 50.0
            elif wc_val > wc_max:
                wc_pen = (wc_val - wc_max) / 50.0
            fk_pen = 0.0
            if fk_val < fk_min:
                fk_pen = (fk_min - fk_val) * 2.0
            elif fk_val > fk_max:
                fk_pen = (fk_val - fk_max) * 2.0
            return wc_pen + fk_pen

        def _score(text: str) -> Tuple[int, float, float]:
            m = _qc(text)
            wc_val = int(m["word_count"])
            fk_val = float(m["flesch_kincaid_grade"])
            return wc_val, fk_val, _penalty(wc_val, fk_val)

        stall_count = 0

        # --- deterministic state machine ---
        for step in range(max_steps):
            m = _qc(current)
            wc = int(m["word_count"])
            fk = float(m["flesch_kincaid_grade"])
            curr_pen = _penalty(wc, fk)

            if wc_min <= wc <= wc_max and fk_min <= fk <= fk_max:
                break

            if stall_count >= 5:
                break

            # 1) EXPAND (priority)
            if wc < wc_min:
                # add in chunks to avoid model failing to add enough
                desired_total = 2000
                deficit = wc_min - wc
                to_add = desired_total - wc
                if deficit <= 200:
                    to_add = min(200, max(deficit + 30, 80))
                else:
                    # clamp per call so it actually returns
                    to_add = max(260, min(450, to_add))
                # avoid overshooting too much
                if wc + to_add > wc_max + 30:
                    to_add = max(80, wc_max + 30 - wc)

                prompt = {
                    "task": "expand_only",
                    "title": title,
                    "keywords": keywords,
                    "targets": thresholds,
                    "current_metrics": {"word_count": wc, "fk_grade": fk},
                    "hard_rules": [
                        "Do NOT edit or rewrite existing text.",
                        "Append NEW sections only at the end.",
                        "Each new section MUST cover a DIFFERENT aspect not already in the draft.",
                        "Do NOT repeat any heading, topic, or idea already present.",
                        "Use ## headings + bullet lists + short paragraphs (2-4 lines).",
                        "Use simple words and short sentences.",
                        "Target FK <= 8.8.",
                        f"Add at least {to_add} words in this call (minimum).",
                        "Add concrete examples, steps, checklists, or a mini FAQ.",
                        "Keep the topic consistent with the title + keywords.",
                        "Return JSON: { append_markdown: string } only.",
                    ],
                    "draft_markdown_input": current,
                }

                parsed, usage = _groq_json(
                    str(groq_key),
                    model=req.model,
                    system=system,
                    prompt=json.dumps(prompt, ensure_ascii=False),
                    temp=float(req.temperature),
                    max_tokens=max_tokens,
                    timeout_s=180,
                    retries=3,
                )
                for k in usage_total:
                    usage_total[k] += int(usage.get(k) or 0)

                append = _clean_md(str(parsed.get("append_markdown") or parsed.get("content") or parsed.get("text") or ""))
                changed = False
                # guard: accept any append that adds at least 50 words (lowered from 60% of to_add)
                if append and _wc(append) >= 50:
                    candidate = _clean_md(f"{current}\n\n{append}")
                    cand_wc, cand_fk, cand_pen = _score(candidate)
                    if cand_pen < curr_pen - 0.01 or (wc_min <= cand_wc <= wc_max + 30):
                        current = candidate
                        changed = True
                        post = _post_simplify(current, wc_max)
                        if post and post != current:
                            post_wc, post_fk, post_pen = _score(post)
                            curr_wc2, curr_fk2, curr_pen2 = _score(current)
                            if post_pen < curr_pen2 - 0.01:
                                current = post
                else:
                    # still continue loop; next step will attempt expand again
                    _ = None
                if changed:
                    stall_count = 0
                else:
                    stall_count += 1
                continue

            # If slightly over wc_max, trim first to hit range; if FK is very high, only trim when overage is meaningful.
            over = wc - wc_max
            need_trim = wc > wc_max and (over >= 60 or fk <= fk_max + 0.5)

            # 2) TRIM if too long (or FK ok and slightly over)
            if need_trim:
                excess = wc - wc_max
                if excess <= 200:
                    cut = min(max(excess + 30, 120), 250)
                else:
                    cut = min(max(excess + 100, 450), 900)
                target_low = wc_max - 80
                target_high = wc_max + 10

                prompt = {
                    "task": "trim_only",
                    "title": title,
                    "keywords": keywords,
                    "targets": thresholds,
                    "current_metrics": {"word_count": wc, "fk_grade": fk},
                    "hard_rules": [
                        "Remove DUPLICATE or NEAR-DUPLICATE content first.",
                        "Do NOT remove main headings or major sections.",
                        "Preserve the conclusion/summary section.",
                        "Remove repetition, filler, and overly long examples.",
                        f"Cut about {cut} words.",
                        f"Keep total word count between {target_low} and {target_high}.",
                        f"Do NOT go below {target_low} words.",
                        "If you cut too much, add short bullets to reach the minimum.",
                        "While trimming, also simplify long sentences and hard words so FK goes under 9.",
                        "Target FK <= 8.8.",
                        "Return JSON: { draft_markdown: string } only.",
                    ],
                    "draft_markdown_input": current,
                }

                parsed, usage = _groq_json(
                    str(groq_key),
                    model=req.model,
                    system=system,
                    prompt=json.dumps(prompt, ensure_ascii=False),
                    temp=float(req.temperature),
                    max_tokens=max_tokens,
                    timeout_s=180,
                    retries=3,
                )
                for k in usage_total:
                    usage_total[k] += int(usage.get(k) or 0)

                out = _clean_md(str(parsed.get("draft_markdown") or parsed.get("content") or parsed.get("text") or ""))
                trim_changed = False
                if out:
                    cand_wc, cand_fk, cand_pen = _score(out)
                    # Accept ANY trim that actually reduces word count.
                    # Even if it goes below wc_min, next passes can expand back.
                    if cand_wc < wc:
                        current = out
                        wc = cand_wc
                        trim_changed = True
                if trim_changed:
                    stall_count = 0
                    continue

                stall_count += 1
                if fk <= fk_max:
                    continue

            # 3) SIMPLIFY FK (keep WC in range)
            if fk > fk_max:
                wc_low = wc_min
                wc_high = wc_max
                hard_rules = [
                    "Do NOT change headings or section order.",
                    "Do NOT delete ideas. Keep all points.",
                    f"Keep total word count between {wc_low} and {wc_high}.",
                    "Aim for 2000-2030 words.",
                    "If above max, cut sentences until within range.",
                    "If below min, add short bullets to reach the minimum.",
                    "Target FK <= 8.8.",
                    "Replace long/complex words with shorter synonyms.",
                    "Use active voice and simple verbs.",
                    "Keep sentences under 20 words where possible.",
                    "Prefer bullets over long paragraphs.",
                    "PRESERVE the original meaning and structure of each sentence.",
                    "Do NOT rewrite sentences from scratch — only edit for clarity.",
                    "Return JSON: { draft_markdown: string } only.",
                ]
                if fk > fk_max + 2.0:
                    hard_rules += [
                        "Simplify vocabulary but keep sentence structure intact.",
                        "Convert dense paragraphs into bullet lists.",
                        "Use 1-2 syllable words where a simpler synonym exists.",
                    ]

                prompt = {
                    "task": "simplify_only",
                    "title": title,
                    "keywords": keywords,
                    "targets": thresholds,
                    "current_metrics": {"word_count": wc, "fk_grade": fk},
                    "hard_rules": hard_rules,
                    "draft_markdown_input": current,
                }

                parsed, usage = _groq_json(
                    str(groq_key),
                    model=req.model,
                    system=system,
                    prompt=json.dumps(prompt, ensure_ascii=False),
                    temp=float(req.temperature),
                    max_tokens=max_tokens,
                    timeout_s=180,
                    retries=3,
                )
                for k in usage_total:
                    usage_total[k] += int(usage.get(k) or 0)

                out = _clean_md(str(parsed.get("draft_markdown") or parsed.get("content") or parsed.get("text") or ""))
                candidates = []
                if out:
                    candidates.append(out)

                # Only one additional attempt for very high FK (removed 3rd aggressive rewrite)
                if fk > fk_max + 2.0:
                    hard_rules2 = hard_rules + [
                        "Edit each sentence for simpler vocabulary — do NOT rewrite from scratch.",
                        "You may remove minor details to lower reading grade.",
                        "Replace technical terms with plain English equivalents.",
                    ]
                    prompt2 = {
                        "task": "simplify_only",
                        "title": title,
                        "keywords": keywords,
                        "targets": thresholds,
                        "current_metrics": {"word_count": wc, "fk_grade": fk},
                        "hard_rules": hard_rules2,
                        "draft_markdown_input": current,
                    }
                    parsed2, usage2 = _groq_json(
                        str(groq_key),
                        model=req.model,
                        system=system,
                        prompt=json.dumps(prompt2, ensure_ascii=False),
                        temp=float(req.temperature),
                        max_tokens=max_tokens,
                        timeout_s=180,
                        retries=3,
                    )
                    for k in usage_total:
                        usage_total[k] += int(usage2.get(k) or 0)

                    out2 = _clean_md(str(parsed2.get("draft_markdown") or parsed2.get("content") or parsed2.get("text") or ""))
                    if out2:
                        candidates.append(out2)

                # 3rd aggressive fallback REMOVED — it destroyed content quality

                best_text = current
                best_pen = curr_pen
                best_fk = fk
                for cand in candidates:
                    cand_wc, cand_fk, cand_pen = _score(cand)
                    if cand_pen < best_pen - 0.01:
                        best_pen = cand_pen
                        best_text = cand
                        best_fk = cand_fk
                    elif cand_fk < best_fk - 0.3 and (wc_min - 800) <= cand_wc <= (wc_max + 800):
                        best_fk = cand_fk
                        best_text = cand
                if best_text != current:
                    current = best_text
                    stall_count = 0
                else:
                    stall_count += 1

                # deterministic post-simplify (reduce sentence length + simplify words)
                post = _post_simplify(current, wc_max)
                if post and post != current:
                    post_wc, post_fk, post_pen = _score(post)
                    curr_wc2, curr_fk2, curr_pen2 = _score(current)
                    if post_pen < curr_pen2 - 0.01:
                        current = post
                        stall_count = 0
                continue

            # FK too low (rare): slightly raise complexity but keep simple
            if fk < fk_min:
                prompt = {
                    "task": "adjust_grade_up",
                    "title": title,
                    "keywords": keywords,
                    "targets": thresholds,
                    "current_metrics": {"word_count": wc, "fk_grade": fk},
                    "hard_rules": [
                        "Keep content the same, but slightly increase sentence length.",
                        "Do NOT change headings or remove content.",
                        "Keep simple English.",
                        "Return JSON: { draft_markdown: string } only.",
                    ],
                    "draft_markdown_input": current,
                }
                parsed, usage = _groq_json(
                    str(groq_key),
                    model=req.model,
                    system=system,
                    prompt=json.dumps(prompt, ensure_ascii=False),
                    temp=float(req.temperature),
                    max_tokens=max_tokens,
                    timeout_s=180,
                    retries=3,
                )
                for k in usage_total:
                    usage_total[k] += int(usage.get(k) or 0)
                out = _clean_md(str(parsed.get("draft_markdown") or parsed.get("content") or parsed.get("text") or ""))
                if out:
                    current = out
                continue

        # --- final qc ---
        # deterministic tail trim if slightly over word count
        if _wc(current) > thresholds["wc_max"]:
            current = _trim_tail_to_wc(_drop_fillers(current), thresholds["wc_max"])

        final_m = _qc(current)
        wc1 = int(final_m["word_count"])
        fk1 = float(final_m["flesch_kincaid_grade"])
        qc_pass = (thresholds["wc_min"] <= wc1 <= thresholds["wc_max"]) and (thresholds["fk_min"] <= fk1 <= thresholds["fk_max"])

        # --- write new draft artifact ---
        artifact = {
            "request_id": request_id,
            "tenant_id": tenant_id,
            "kb_id": kb_id,
            "attempt_no": new_attempt,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "model": {"provider": "groq", "name": req.model},
            "qc_fix": {"from_draft_fingerprint": old_fp, "thresholds": thresholds, "metrics": final_m, "qc_pass": qc_pass},
            "draft": {"title": title, "draft_markdown": current},
        }
        out_bytes = json.dumps(artifact, ensure_ascii=False, indent=2).encode("utf-8")
        new_fp = _sha(out_bytes)

        draft_obj = f"{articles_prefix}{tenant_id}/{kb_id}/{request_id}/attempt_{new_attempt}/draft_v1/{new_fp}.json"
        gcs_new_draft_uri = _gcs_put_create_only(gcs, bucket, draft_obj, out_bytes, "application/json; charset=utf-8")

        # --- write qc report artifact ---
        qc_report = {
            "request_id": request_id,
            "tenant_id": tenant_id,
            "kb_id": kb_id,
            "attempt_no": new_attempt,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "input": {"gcs_draft_uri": gcs_new_draft_uri, "draft_fingerprint": new_fp},
            "thresholds": {"word_count_min": thresholds["wc_min"], "word_count_max": thresholds["wc_max"], "fk_grade_min": thresholds["fk_min"], "fk_grade_max": thresholds["fk_max"]},
            "qc_pass": qc_pass,
            "metrics": final_m,
            "notes": "Day21 QC-fix deterministic pass",
        }
        qc_bytes = json.dumps(qc_report, ensure_ascii=False, indent=2).encode("utf-8")
        qc_fp = _sha(qc_bytes)
        qc_obj = f"{articles_prefix}{tenant_id}/{kb_id}/{request_id}/attempt_{new_attempt}/qc_v1/{qc_fp}.json"
        gcs_new_qc_uri = _gcs_put_create_only(gcs, bucket, qc_obj, qc_bytes, "application/json; charset=utf-8")

        # --- update supabase pointers (and clear zerogpt) ---
        with _db_conn(settings) as conn:
            _ensure_schema(conn)
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE public.article_requests
                    SET attempt_count=%s,
                        status='completed',
                        gcs_draft_uri=%s,
                        draft_fingerprint=%s,
                        draft_model=%s,
                        draft_meta=%s::jsonb,
                        qc_summary=%s::jsonb,
                        gcs_qc_uri=%s,
                        qc_fingerprint=%s,
                        qc_meta=%s::jsonb,

                        gcs_zerogpt_uri=NULL,
                        zerogpt_fingerprint=NULL,
                        zerogpt_score=NULL,
                        zerogpt_meta=NULL,

                        updated_at=%s
                    WHERE tenant_id=%s::uuid AND request_id=%s::uuid
                    """,
                    (
                        new_attempt,
                        gcs_new_draft_uri,
                        new_fp,
                        req.model,
                        json.dumps({"qc_fix": True, "from": old_fp}),
                        json.dumps({"qc_pass": qc_pass, "word_count": wc1, "fk_grade": fk1}),
                        gcs_new_qc_uri,
                        qc_fp,
                        json.dumps({"attempt_no": new_attempt, "thresholds": thresholds, "metrics": final_m}),
                        datetime.now(timezone.utc),
                        tenant_id,
                        request_id,
                    ),
                )
            _log_job_event(conn, tenant_id, "qc_fix_saved", {"qc_pass": qc_pass, "word_count": wc1, "fk_grade": fk1}, request_id=request_id)
            conn.commit()

        return QCFixResponse(
            status="ok",
            request_id=request_id,
            tenant_id=tenant_id,
            kb_id=kb_id,
            new_attempt_no=new_attempt,
            old_draft_fingerprint=old_fp,
            new_draft_fingerprint=new_fp,
            gcs_new_draft_uri=gcs_new_draft_uri,
            qc_pass=qc_pass,
            qc_metrics=final_m,
            thresholds=thresholds,
            gcs_new_qc_uri=gcs_new_qc_uri,
            qc_fingerprint=qc_fp,
            usage=usage_total,
        )

    finally:
        if locked:
            try:
                with _db_conn(settings) as conn:
                    _unlock(conn, request_id, tok)
                    conn.commit()
            except Exception:
                pass
