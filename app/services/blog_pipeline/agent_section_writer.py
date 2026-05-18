"""
Section Writer Agent.
One call per section. System prompt is identical across all calls (enables OpenAI caching).
CoT and self-critique are embedded in the prompt — no extra API calls needed.
Max retries: 2 per section, temperature escalates on each retry.
"""
from __future__ import annotations
import re
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI

from .prompts import SECTION_WRITER_SYSTEM, SECTION_WRITER_USER
from .gates_local_qc import run_local_qc, word_count
from .prompt_engine import BrandContext
from .utils import extract_usage, sum_usage

MAX_RETRIES = 2
BASE_TEMPERATURE = 0.75
TEMP_ESCALATION = 0.08  # per retry — more variation on each attempt


def _build_facts_block(facts: List[Dict[str, Any]], assigned_ids: List[str], sparse: bool) -> str:
    if not facts:
        return "- No evidence facts retrieved. Draw on established domain knowledge. Do not invent specific statistics."
    assigned = [f for f in facts if f.get("fact_id") in assigned_ids] if assigned_ids else facts
    if not assigned:
        assigned = facts[:8]
    lines = [
        f"- [{f['fact_id']}] ({f.get('confidence','medium')}) {f['claim']}"
        + (" [VERIFY]" if f.get("confidence") == "low" else "")
        for f in assigned
    ]
    if sparse:
        lines.append("- SPARSE: Evidence is limited. Supplement with general domain knowledge for framing and context. Do not invent specific statistics.")
    return "\n".join(lines)


def _format_instruction(role: str, heading: Optional[str]) -> str:
    if role == "closing":
        return (
            "Plain markdown paragraphs only. No heading. 2-4 sentences maximum. "
            "A strong, forward-looking observation that crystallizes the article's single most important insight — "
            "without summarizing the sections, without starting with 'In conclusion', 'Ultimately', or 'At the end of the day'. "
            "The reader should feel resolved and clear-headed. End on action or truth, not caution."
        )
    if role == "hook":
        return (
            "Plain flowing prose paragraphs ONLY. No heading. No bullet lists. "
            "No bold structural labels such as 'Key takeaways', 'In this article', or 'Table of contents'. "
            "The assembler adds those automatically — your job is the opening paragraphs only."
        )
    if role == "faq":
        h = heading or "FAQ"
        return (
            f"Start output with: ## {h}\n\n"
            "Then write Q&A pairs DIRECTLY — no intro paragraph, no preamble sentence before the first question. "
            "Format each pair as: **Bold question** on its own line, then the answer paragraph (2-4 sentences). "
            "5-7 pairs. No transitional text between pairs."
        )
    if role == "practical":
        return (
            f"{'## ' + heading + chr(10) if heading else ''}"
            "Markdown. One short intro sentence, then bullet list for steps, then 1-2 closing analytical sentences."
        )
    prefix = f"## {heading}\n\n" if heading else ""
    return f"Start output with: {prefix}Then write flowing prose paragraphs."


def _style_guide(content_type: str, audience_level: str, role: str) -> str:
    """Build a short style instruction based on content_type and audience_level."""
    if role in ("hook", "closing"):
        return "Keep it short and specific. 2-3 short paragraphs max. No story longer than one paragraph."
    if content_type in ("comparison", "explainer", "how_to"):
        audience_note = {
            "beginner": "Define every term. Assume the reader has never heard it before. Use ₹/$ examples.",
            "intermediate": "Skip basic definitions. Focus on nuance, edge cases, and practical application.",
            "advanced": "Assume domain knowledge. Focus on mechanisms, data, and non-obvious insights.",
        }.get(audience_level, "Define key terms. Use specific examples.")
        return (
            f"This is a {content_type} article for a {audience_level} audience. "
            f"{audience_note} "
            "Prioritize clarity over style. Definition first, explanation second, example third. "
            "Use tables for comparisons. Use numbered lists for steps."
        )
    return f"This is a {content_type} article. Write clearly and specifically for a {audience_level} reader."


def _prev_tail(prev_section_text: Optional[str], n_words: int = 40) -> str:
    if not prev_section_text:
        return "(this is the opening section)"
    words = prev_section_text.strip().split()
    tail = " ".join(words[-n_words:]) if len(words) > n_words else prev_section_text.strip()
    return f"...{tail}"


def write_section(
    client: OpenAI,
    model: str,
    *,
    title: str,
    section: Dict[str, Any],
    facts: List[Dict[str, Any]],
    sparse: bool = False,
    prev_section_text: Optional[str] = None,
    max_tokens: int = 700,
    brand_context: Optional[BrandContext] = None,
    content_type: str = "explainer",
    audience_level: str = "beginner",
) -> Tuple[str, bool, List[str], Dict[str, int], Dict[str, Any]]:
    """
    Write one blog section with CoT + self-critique + local QC gate.

    Returns:
        text: final section markdown
        passed_qc: whether local QC passed (even if not, best attempt is returned)
        failures: list of QC failure reasons (empty if passed)
        usage: token counts (sum of all attempts)
        meta: attempt count, final temperature, retry reasons
    """
    role = section.get("role", "body")
    heading = section.get("heading")
    target_words = int(section.get("target_words", 250))
    min_words = max(80 if role == "faq" else 150, int(target_words * 0.75))
    assigned_ids = section.get("assigned_fact_ids") or []
    key_term_to_define = section.get("key_term_to_define") or "none"
    writing_intent = section.get("writing_intent", "deliver the key insight for this section")
    opening_constraint = section.get("opening_constraint", "")

    facts_block = _build_facts_block(facts, assigned_ids, sparse)
    prev_tail = _prev_tail(prev_section_text)
    format_instruction = _format_instruction(role, heading)
    sparse_note = (
        "Evidence is sparse. Use general domain knowledge for framing. Mark invented statistics as 'approximately' or omit them."
        if sparse else
        "Evidence coverage is adequate. Prefer Evidence Locker facts for specific claims."
    )

    # Build system prompt: static base + brand voice overlay (tenant-specific)
    voice_block = brand_context.voice_block() if brand_context else ""
    system_content = SECTION_WRITER_SYSTEM + ("\n\n" + voice_block if voice_block else "")

    total_usage: Dict[str, int] = {"prompt_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    best_text = ""
    best_failures: List[str] = ["no attempt made"]
    meta: Dict[str, Any] = {"attempts": 0, "final_temp": BASE_TEMPERATURE, "retry_reasons": []}

    for attempt in range(MAX_RETRIES + 1):
        temperature = BASE_TEMPERATURE + attempt * TEMP_ESCALATION
        meta["attempts"] = attempt + 1
        meta["final_temp"] = temperature

        user_prompt = SECTION_WRITER_USER.format(
            role=role,
            title=title,
            content_type=content_type,
            audience_level=audience_level,
            heading=heading or "(no heading for this section)",
            key_term_to_define=key_term_to_define,
            writing_intent=writing_intent,
            opening_constraint=opening_constraint,
            target_words=target_words,
            min_words=min_words,
            facts_block=facts_block,
            prev_section_tail=prev_tail,
            sparse_note=sparse_note,
            style_guide=_style_guide(content_type, audience_level, role),
            format_instruction=format_instruction,
        )

        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                max_completion_tokens=max_tokens,
            )
            text = (resp.choices[0].message.content or "").strip()
            total_usage = sum_usage(total_usage, extract_usage(resp))
        except Exception as e:
            meta["retry_reasons"].append(f"attempt {attempt+1} API error: {e}")
            continue

        passed, failures = run_local_qc(text, role=role, min_words=min_words, max_words=int(target_words * 1.25))

        # Always keep the best attempt so far
        if not best_text or (not passed and len(failures) < len(best_failures)):
            best_text = text
            best_failures = failures

        if passed:
            return text, True, [], total_usage, meta

        meta["retry_reasons"].append(f"attempt {attempt+1} failed: {failures}")

    # All retries exhausted — return best attempt with failure flags
    return best_text, False, best_failures, total_usage, meta
