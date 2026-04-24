"""
Mini Humanize Agent.
Called ONLY when a section fails the local AI-pattern gate.
Surgical fixes only — does not rewrite the whole section.
Max 1 retry per section.
"""
from __future__ import annotations
from typing import Any, Dict, List, Tuple

from openai import OpenAI
from .prompts import MINI_HUMANIZE_SYSTEM, MINI_HUMANIZE_USER
from .gates_local_qc import run_local_qc

MAX_HUMANIZE_RETRIES = 1


def _usage(resp: Any) -> Dict[str, int]:
    u = getattr(resp, "usage", None)
    if u is None:
        return {"prompt_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    return {
        "prompt_tokens": getattr(u, "prompt_tokens", 0),
        "output_tokens": getattr(u, "completion_tokens", 0),
        "total_tokens": getattr(u, "total_tokens", 0),
    }


def humanize_section(
    client: OpenAI,
    model: str,
    section_text: str,
    failures: List[str],
    role: str = "body",
    temperature: float = 0.85,
    max_tokens: int = 600,
) -> Tuple[str, bool, Dict[str, int]]:
    """
    Attempt surgical humanization of a section.
    Returns (text, improved, usage).
    'improved' means at least one fewer QC failure than before.
    If humanize fails or makes things worse, returns original text.
    """
    total_usage: Dict[str, int] = {"prompt_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    original_failure_count = len(failures)

    problems_block = "\n".join(f"- {f}" for f in failures)

    for attempt in range(MAX_HUMANIZE_RETRIES + 1):
        temp = temperature + attempt * 0.05
        user = MINI_HUMANIZE_USER.format(
            section_text=section_text,
            problems=problems_block,
        )
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": MINI_HUMANIZE_SYSTEM},
                    {"role": "user", "content": user},
                ],
                temperature=temp,
                max_completion_tokens=max_tokens,
            )
            fixed = (resp.choices[0].message.content or "").strip()
            total_usage = {k: total_usage[k] + _usage(resp).get(k, 0) for k in total_usage}
        except Exception:
            break

        if not fixed:
            continue

        passed, new_failures = run_local_qc(fixed, role=role)

        # Accept if it improved (fewer failures) — even if not fully passing
        if len(new_failures) < original_failure_count:
            return fixed, passed, total_usage

    # Humanize made no improvement — return original
    return section_text, False, total_usage
