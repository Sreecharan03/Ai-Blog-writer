"""
Tone Reviewer Agent.
Runs after SectionWriter on every section.
Fixes three specific problems surgically:
  1. Compliance nudges in body sections (remove — FAQ only)
  2. Cold / cautionary section endings (warm up)
  3. Clinical language that creates distance (soften to plain speech)
Skips the LLM call entirely when none of the patterns are detected.
"""
from __future__ import annotations

import re
from typing import Any, Dict, Optional, Tuple

from openai import OpenAI
from .prompt_engine import BrandContext


# ── Patterns that trigger a review call ───────────────────────────────────────
_NUDGE_RE = re.compile(
    r"talk\s+(?:to|with)\s+a\s+(?:clinician|doctor|specialist|physician|professional)"
    r"|(?:consult|see|visit)\s+a\s+(?:clinician|doctor|specialist|physician|professional|healthcare)"
    r"|bring\s+(?:that|this|the|your)\s+\w+\s+to\s+a\s+(?:clinician|doctor|specialist)"
    r"|screening\s+for\s+(?:sleep\s+)?disorders?"
    r"|(?:seek|get)\s+(?:medical|professional|clinical)\s+(?:help|advice|attention|evaluation)"
    r"|(?:see|visit)\s+a\s+sleep\s+specialist",
    re.IGNORECASE,
)

_COLD_ENDING_RE = re.compile(
    r"(?:consult|talk|speak|visit|see)\s+a\s+\w+"
    r"|if\s+(?:symptoms?|this|these|it)\s+(?:persist|continues?|worsens?|keeps?)"
    r"|bring\s+(?:that|this|the)\s+(?:log|pattern|result|info)",
    re.IGNORECASE,
)


# ── System prompt ──────────────────────────────────────────────────────────────
_SYSTEM = """You are a warm editorial editor. Your job is to make blog sections feel like they were written by a trusted, knowledgeable friend — not a medical brochure.

You make THREE surgical fixes only. You do NOT rewrite entire sections.

FIX 1 — COMPLIANCE NUDGES (body sections only, not FAQ):
If the section contains advice like "consult a clinician", "talk to a doctor", "see a specialist", "bring that to a physician", or "screening for disorders" — remove it. Cut the sentence entirely, or replace it with a warm, forward-looking observation of similar length.

FIX 2 — COLD ENDINGS:
If the last sentence is a warning, disclaimer, or cautionary note — replace it with something that leaves the reader curious, encouraged, or thinking. The ending should feel like a friend saying "try this" not a doctor saying "be careful".
Good ending: "Pick one variable. Change it tonight. See what morning feels like."
Good ending: "That gap is where the interesting work begins."
Bad ending: "If you experience these symptoms regularly, consult a specialist."

FIX 3 — CLINICAL LANGUAGE:
Words like "sleep inertia", "arousals", "NREM Stage 3", "screening for disorders" — if they appear without explanation, add a brief plain-language note in the same sentence, or swap to plain language.
Example: "sleep inertia (that groggy pulled-from-glue feeling)" or just "that groggy pulled-from-glue feeling".

WHAT YOU NEVER DO:
- Never remove specific facts, numbers, or evidence
- Never change headings
- Never change the structure (paragraphs, bullet points stay as-is)
- Never make the section vague — specific is always warmer than general
- Never add new topics or extend the word count significantly

OUTPUT: The corrected section only. No commentary. No explanation of changes."""


_USER = """Review this {role} section. Apply only the three editorial fixes.

IS FAQ: {is_faq}
- If True: skip Fix 1 (compliance nudges allowed in FAQ). Apply Fix 2 and Fix 3 only.
- If False: apply all three fixes.

SECTION:
{section_text}

Output the corrected section only."""


# ── Public function ────────────────────────────────────────────────────────────
def review_section_tone(
    client: OpenAI,
    model: str,
    section_text: str,
    role: str = "body",
    brand_context: Optional[BrandContext] = None,
    temperature: float = 0.35,
    max_tokens: int = 750,
) -> Tuple[str, bool, Dict[str, int]]:
    """
    Run tone review on one section. Returns (text, was_changed, usage).
    Skips the LLM call when no trigger patterns are detected.
    """
    empty_usage: Dict[str, int] = {"prompt_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    is_faq = role == "faq"

    # Fast path: skip LLM if no nudge/cold-ending patterns found in body sections
    if not is_faq and not _NUDGE_RE.search(section_text) and not _COLD_ENDING_RE.search(section_text):
        return section_text, False, empty_usage

    user_prompt = _USER.format(
        role=role,
        is_faq="True" if is_faq else "False",
        section_text=section_text,
    )

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            max_completion_tokens=max_tokens,
        )
        reviewed = (resp.choices[0].message.content or "").strip()
        u = getattr(resp, "usage", None)
        usage: Dict[str, int] = {
            "prompt_tokens": getattr(u, "prompt_tokens", 0) if u else 0,
            "output_tokens": getattr(u, "completion_tokens", 0) if u else 0,
            "total_tokens": getattr(u, "total_tokens", 0) if u else 0,
        }
    except Exception:
        return section_text, False, empty_usage

    # Safety: reject if response is suspiciously short (likely a refusal or error)
    if not reviewed or len(reviewed) < len(section_text) * 0.4:
        return section_text, False, empty_usage

    was_changed = reviewed.strip() != section_text.strip()
    return reviewed, was_changed, usage
