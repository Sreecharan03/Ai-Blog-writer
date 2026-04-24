"""
Section Planner Agent (P3).
Waits for Topic Analyst + Evidence Locker, then designs the full section blueprint.
Every other agent depends on its output — it defines the contract.
"""
from __future__ import annotations
import json
import re
from typing import Any, Dict, List, Tuple

from openai import OpenAI
from .prompts import SECTION_PLANNER_SYSTEM, SECTION_PLANNER_USER

_ROLE_TARGET_WORDS = {
    "hook": 185,
    "context": 260,
    "evidence": 290,
    "practical": 270,
    "opinion": 220,
    "counterargument": 240,
    "conclusion": 200,
    "faq": 180,
}

_NARRATIVE_ARC = ["hook", "context", "evidence", "evidence", "opinion", "counterargument", "conclusion", "faq"]
_INSTRUCTIONAL_ARC = ["hook", "context", "evidence", "practical", "practical", "counterargument", "conclusion", "faq"]


def _fallback_plan(arc: str, facts: List[Dict], target_words: int) -> List[Dict[str, Any]]:
    """Returns a safe default plan when the LLM planner fails."""
    roles = _INSTRUCTIONAL_ARC if arc == "instructional" else _NARRATIVE_ARC
    fact_ids = [f["fact_id"] for f in facts]
    per_section = max(1, len(fact_ids) // max(1, len(roles)))
    sections = []
    for i, role in enumerate(roles):
        start = i * per_section
        assigned = fact_ids[start: start + per_section]
        sections.append({
            "index": i,
            "role": role,
            "heading": None if role == "hook" else role.title(),
            "target_words": _ROLE_TARGET_WORDS.get(role, 250),
            "assigned_fact_ids": assigned,
            "writing_intent": f"deliver the key insight for this {role} section",
            "opening_constraint": "",
        })
    return sections


def _parse(raw: str) -> Dict[str, Any]:
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


def _usage(resp: Any) -> Dict[str, int]:
    u = getattr(resp, "usage", None)
    if u is None:
        return {"prompt_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    return {
        "prompt_tokens": getattr(u, "prompt_tokens", 0),
        "output_tokens": getattr(u, "completion_tokens", 0),
        "total_tokens": getattr(u, "total_tokens", 0),
    }


def _facts_summary(facts: List[Dict[str, Any]]) -> str:
    if not facts:
        return "No facts extracted — sparse evidence mode."
    lines = [f"[{f['fact_id']}] ({f.get('category','?')}) {f['claim']}" for f in facts[:24]]
    if len(facts) > 24:
        lines.append(f"... and {len(facts)-24} more facts")
    return "\n".join(lines)


def plan_sections(
    client: OpenAI,
    model: str,
    title: str,
    analysis: Dict[str, Any],
    facts: List[Dict[str, Any]],
    sparse: bool,
    target_words: int = 2000,
    temperature: float = 0.2,
    max_tokens: int = 1200,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """
    Returns (sections_list, usage).
    Falls back to deterministic plan if LLM fails.
    """
    arc = analysis.get("arc", "narrative")
    user = SECTION_PLANNER_USER.format(
        title=title,
        content_type=analysis.get("content_type", "explainer"),
        arc=arc,
        audience=analysis.get("audience", "general reader"),
        primary_angle=analysis.get("primary_angle", ""),
        counterargument_seed=analysis.get("counterargument_seed", ""),
        hook_seed=analysis.get("hook_seed", ""),
        facts_summary=_facts_summary(facts),
        target_words=target_words,
    )

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SECTION_PLANNER_SYSTEM},
                {"role": "user", "content": user},
            ],
            temperature=temperature,
            max_completion_tokens=max_tokens,
        )
        raw = (resp.choices[0].message.content or "").strip()
        parsed = _parse(raw)
        usage = _usage(resp)
    except Exception:
        return _fallback_plan(arc, facts, target_words), {"prompt_tokens": 0, "output_tokens": 0, "total_tokens": 0}

    sections_raw = parsed.get("sections") if isinstance(parsed.get("sections"), list) else []
    if not sections_raw:
        return _fallback_plan(arc, facts, target_words), usage

    # Validate and normalize each section
    valid_fact_ids = {f["fact_id"] for f in facts}
    sections: List[Dict[str, Any]] = []
    for item in sections_raw:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role") or "body").lower()
        assigned = [fid for fid in (item.get("assigned_fact_ids") or []) if fid in valid_fact_ids]
        sections.append({
            "index": len(sections),
            "role": role,
            "heading": item.get("heading"),
            "target_words": int(item.get("target_words") or _ROLE_TARGET_WORDS.get(role, 250)),
            "assigned_fact_ids": assigned,
            "writing_intent": str(item.get("writing_intent") or f"deliver the key insight for this {role} section"),
            "opening_constraint": str(item.get("opening_constraint") or ""),
        })

    if not sections:
        return _fallback_plan(arc, facts, target_words), usage

    return sections, usage
