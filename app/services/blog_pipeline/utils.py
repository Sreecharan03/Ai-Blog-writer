"""
Shared utilities for blog pipeline agents.
Import here instead of copy-pasting across agent files.
"""
from __future__ import annotations
import json
import re
from typing import Any, Dict


def extract_usage(resp: Any) -> Dict[str, int]:
    """Extract token usage from an OpenAI chat completion response."""
    u = getattr(resp, "usage", None)
    if u is None:
        return {"prompt_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    return {
        "prompt_tokens": getattr(u, "prompt_tokens", 0),
        "output_tokens": getattr(u, "completion_tokens", 0),
        "total_tokens": getattr(u, "total_tokens", 0),
    }


def sum_usage(*items: Dict[str, int]) -> Dict[str, int]:
    """Sum multiple usage dicts into one."""
    out = {"prompt_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    for d in items:
        for k in out:
            out[k] += d.get(k, 0)
    return out


def parse_json_response(raw: str) -> Dict[str, Any]:
    """
    Parse JSON from an LLM response, stripping markdown fences if present.
    Falls back to regex extraction of the first {...} block.
    Returns {} on total failure.
    """
    raw = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        return json.loads(raw)
    except Exception:
        pass
    m = re.search(r"\{[\s\S]+\}", raw)
    if m:
        try:
            return json.loads(m.group())
        except Exception:
            pass
    return {}
