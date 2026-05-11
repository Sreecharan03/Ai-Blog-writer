"""
Topic Analyst Agent (P1).
Classifies content type, identifies narrative angle, seeds hook and counterargument.
Runs in parallel with Evidence Locker builder.
"""
from __future__ import annotations
from typing import Any, Dict, List, Tuple

from openai import OpenAI
from .prompts import TOPIC_ANALYST_SYSTEM, TOPIC_ANALYST_USER
from .utils import extract_usage, parse_json_response

_DEFAULTS = {
    "content_type": "explainer",
    "audience": "general reader with basic familiarity with the topic",
    "primary_angle": "the evidence-based view on this topic",
    "arc": "narrative",
    "tone": "authoritative",
    "counterargument_seed": "many people already know this and don't need to change",
    "hook_seed": "a specific finding or moment related to this topic",
}


def analyze_topic(
    client: OpenAI,
    model: str,
    title: str,
    keywords: List[str],
    temperature: float = 0.2,
    max_tokens: int = 400,
) -> Tuple[Dict[str, Any], Dict[str, int]]:
    """
    Returns (analysis_dict, usage_dict).
    Falls back to defaults on any failure — never blocks the pipeline.
    """
    kw = ", ".join(keywords or [])
    user = TOPIC_ANALYST_USER.format(title=title, keywords=kw)

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": TOPIC_ANALYST_SYSTEM},
                {"role": "user", "content": user},
            ],
            temperature=temperature,
            max_completion_tokens=max_tokens,
        )
        raw = (resp.choices[0].message.content or "").strip()
        parsed = parse_json_response(raw)
        usage = extract_usage(resp)
    except Exception as e:
        import logging; logging.getLogger("topic_analyst").warning("analyze_topic failed, using defaults: %s", e)
        return dict(_DEFAULTS), {"prompt_tokens": 0, "output_tokens": 0, "total_tokens": 0}

    # merge with defaults for any missing keys
    result = dict(_DEFAULTS)
    result.update({k: v for k, v in parsed.items() if v})
    return result, usage
