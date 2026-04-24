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
        for f in assigned
    ]
    if sparse:
        lines.append("- SPARSE: Evidence is limited. Supplement with general domain knowledge for framing and context. Do not invent specific statistics.")
    return "\n".join(lines)


def _format_instruction(role: str, heading: Optional[str]) -> str:
    if role == "hook":
        return "Plain markdown paragraphs only. No heading line."
    if role == "faq":
        return "## FAQ\n\nMarkdown with **bold question** on its own line, then answer paragraph. 5-7 Q&A pairs. Always start with the heading `## FAQ` on the first line."
    if role == "practical":
        return (
            f"{'## ' + heading + chr(10) if heading else ''}"
            "Markdown. One short intro sentence, then bullet list for steps, then 1-2 closing analytical sentences."
        )
    prefix = f"## {heading}\n\n" if heading else ""
    return f"Start output with: {prefix}Then write flowing prose paragraphs."


def _prev_tail(prev_section_text: Optional[str], n_words: int = 40) -> str:
    if not prev_section_text:
        return "(this is the opening section)"
    words = prev_section_text.strip().split()
    tail = " ".join(words[-n_words:]) if len(words) > n_words else prev_section_text.strip()
    return f"...{tail}"


def _usage(resp: Any) -> Dict[str, int]:
    u = getattr(resp, "usage", None)
    if u is None:
        return {"prompt_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    return {
        "prompt_tokens": getattr(u, "prompt_tokens", 0),
        "output_tokens": getattr(u, "completion_tokens", 0),
        "total_tokens": getattr(u, "total_tokens", 0),
    }


def _sum_usage(*items: Dict[str, int]) -> Dict[str, int]:
    out = {"prompt_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    for d in items:
        for k in out:
            out[k] += d.get(k, 0)
    return out


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
            heading=heading or "(no heading for this section)",
            writing_intent=writing_intent,
            opening_constraint=opening_constraint,
            target_words=target_words,
            min_words=min_words,
            facts_block=facts_block,
            prev_section_tail=prev_tail,
            sparse_note=sparse_note,
            format_instruction=format_instruction,
        )

        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SECTION_WRITER_SYSTEM},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                max_completion_tokens=max_tokens,
            )
            text = (resp.choices[0].message.content or "").strip()
            total_usage = _sum_usage(total_usage, _usage(resp))
        except Exception as e:
            meta["retry_reasons"].append(f"attempt {attempt+1} API error: {e}")
            continue

        passed, failures = run_local_qc(text, role=role, min_words=min_words, max_words=int(target_words * 1.6))

        # Always keep the best attempt so far
        if not best_text or (not passed and len(failures) < len(best_failures)):
            best_text = text
            best_failures = failures

        if passed:
            return text, True, [], total_usage, meta

        meta["retry_reasons"].append(f"attempt {attempt+1} failed: {failures}")

    # All retries exhausted — return best attempt with failure flags
    return best_text, False, best_failures, total_usage, meta
