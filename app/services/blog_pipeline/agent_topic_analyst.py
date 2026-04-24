"""
Topic Analyst Agent (P1).
Classifies content type, identifies narrative angle, seeds hook and counterargument.
Runs in parallel with Evidence Locker builder.
"""
from __future__ import annotations
import json
from typing import Any, Dict, List, Tuple

from openai import OpenAI
from .prompts import TOPIC_ANALYST_SYSTEM, TOPIC_ANALYST_USER

_DEFAULTS = {
    "content_type": "explainer",
    "audience": "general reader with basic familiarity with the topic",
    "primary_angle": "the evidence-based view on this topic",
    "arc": "narrative",
    "tone": "authoritative",
    "counterargument_seed": "many people already know this and don't need to change",
    "hook_seed": "a specific finding or moment related to this topic",
}


def _parse(raw: str) -> Dict[str, Any]:
    raw = raw.strip()
    # strip ```json fences
    raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        return json.loads(raw)
    except Exception:
        pass
    # try extracting first {...}
    m = __import__("re").search(r"\{[\s\S]+\}", raw)
    if m:
        try:
            return json.loads(m.group())
        except Exception:
            pass
    return {}


def _usage(resp: Any) -> Dict[str, int]:
    u = getattr(resp, "usage", None)
    if u is None:
        return {"prompt_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    return {
        "prompt_tokens": getattr(u, "prompt_tokens", 0),
        "output_tokens": getattr(u, "completion_tokens", 0),
        "total_tokens": getattr(u, "total_tokens", 0),
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
        parsed = _parse(raw)
        usage = _usage(resp)
    except Exception:
        return dict(_DEFAULTS), {"prompt_tokens": 0, "output_tokens": 0, "total_tokens": 0}

    # merge with defaults for any missing keys
    result = dict(_DEFAULTS)
    result.update({k: v for k, v in parsed.items() if v})
    return result, usage
