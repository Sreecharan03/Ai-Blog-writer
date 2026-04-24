"""
Assembler — Phase 3.
No LLM. Deterministic joining, word count check, targeted expand if short.
"""
from __future__ import annotations
import re
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI
from .gates_local_qc import word_count

MIN_WORDS = 1900
MAX_WORDS = 2600
MIN_SECTION_WORDS = 180


def _join(sections: List[Dict[str, Any]]) -> str:
    parts = []
    for s in sections:
        text = (s.get("text") or "").strip()
        if text:
            parts.append(text)
    return "\n\n".join(parts)


def _find_thin_sections(sections: List[Dict[str, Any]], n: int = 2) -> List[int]:
    """Return indices of the n shortest non-FAQ, non-hook sections."""
    scored = [
        (i, word_count(s.get("text") or ""))
        for i, s in enumerate(sections)
        if s.get("role") not in ("faq",) and s.get("text")
    ]
    scored.sort(key=lambda x: x[1])
    return [i for i, _ in scored[:n]]


def _expand_section(
    client: OpenAI,
    model: str,
    section: Dict[str, Any],
    facts_block: str,
    temperature: float = 0.75,
    max_tokens: int = 500,
) -> Tuple[str, Dict[str, int]]:
    """Add 2-3 analytical sentences to a thin section inline."""
    role = section.get("role", "body")
    text = section.get("text", "")

    prompt = f"""Expand this {role} section by adding 2-3 analytical sentences WITHIN the existing paragraphs.
Do NOT add a new section. Do NOT repeat what is already written. Insert sentences that deepen the analysis.

CURRENT SECTION:
{text}

ADDITIONAL EVIDENCE FACTS (use these to deepen, do not invent):
{facts_block}

Rules:
- Add sentences inside existing paragraphs, not at the end
- Each added sentence must be specific (number, name, mechanism) — not generic
- Keep the same heading if present
- No transitional fillers, no paragraph-end summaries
- Output the full expanded section only"""

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a precise editorial writer. Add depth without changing voice. Output only the expanded section."},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            max_completion_tokens=max_tokens,
        )
        expanded = (resp.choices[0].message.content or "").strip()
        u = getattr(resp, "usage", None)
        usage = {
            "prompt_tokens": getattr(u, "prompt_tokens", 0) if u else 0,
            "output_tokens": getattr(u, "completion_tokens", 0) if u else 0,
            "total_tokens": getattr(u, "total_tokens", 0) if u else 0,
        }
        # Only accept if it actually added words
        if expanded and word_count(expanded) > word_count(text) + 20:
            return expanded, usage
    except Exception:
        pass
    return text, {"prompt_tokens": 0, "output_tokens": 0, "total_tokens": 0}


def assemble(
    client: Optional[OpenAI],
    model: Optional[str],
    sections: List[Dict[str, Any]],
    facts: List[Dict[str, Any]],
    target_words: int = 2000,
    max_tokens_per_expand: int = 500,
) -> Tuple[str, int, Dict[str, int]]:
    """
    Join sections, check word count, expand if short (max 2 targeted LLM calls).
    Returns (markdown, final_word_count, usage).
    """
    total_usage: Dict[str, int] = {"prompt_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    facts_block = "\n".join(
        f"- [{f['fact_id']}] {f['claim']}" for f in facts
    ) if facts else "No facts available."

    assembled = _join(sections)
    wc = word_count(assembled)

    if wc < MIN_WORDS and client and model:
        thin_indices = _find_thin_sections(sections, n=2)
        for idx in thin_indices:
            expanded, exp_usage = _expand_section(
                client, model, sections[idx], facts_block,
                max_tokens=max_tokens_per_expand,
            )
            sections[idx]["text"] = expanded
            total_usage = {k: total_usage[k] + exp_usage.get(k, 0) for k in total_usage}

        assembled = _join(sections)
        wc = word_count(assembled)

    return assembled, wc, total_usage


def section_count(markdown: str) -> int:
    return len(re.findall(r"^#{1,3}\s+", markdown, re.MULTILINE))


def has_faq(markdown: str) -> bool:
    return bool(re.search(r"^#{1,3}\s+.*\b(FAQ|Frequently\s+Asked)\b", markdown, re.MULTILINE | re.IGNORECASE))
