"""
Section Planner Agent (P3).
Waits for Topic Analyst + Evidence Locker, then designs the full section blueprint.
Every other agent depends on its output — it defines the contract.
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI
from .prompts import SECTION_PLANNER_SYSTEM, SECTION_PLANNER_USER
from .prompt_engine import BrandContext
from .utils import extract_usage, parse_json_response

_ROLE_TARGET_WORDS = {
    "hook": 220,
    "context": 310,
    "evidence": 340,
    "practical": 320,
    "opinion": 260,
    "counterargument": 280,
    "conclusion": 240,
    "faq": 220,
    "closing": 100,
}

_NARRATIVE_ARC = ["hook", "context", "evidence", "evidence", "opinion", "counterargument", "conclusion", "faq", "closing"]
_INSTRUCTIONAL_ARC = ["hook", "context", "evidence", "practical", "practical", "counterargument", "conclusion", "faq", "closing"]

# Meaningful fallback headings used when the LLM planner fails.
# These read as real section titles — not internal role names.
_FALLBACK_HEADINGS: dict[str, str] = {
    "context":         "What You Need to Know First",
    "evidence":        "What the Research Actually Shows",
    "practical":       "How to Put This Into Practice",
    "opinion":         "A Perspective Worth Considering",
    "counterargument": "The Case Against — and Why It Still Matters",
    "conclusion":      "The Bottom Line",
    "faq":             "Questions People Ask",
}


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
            "heading": None if role in ("hook", "closing") else _FALLBACK_HEADINGS.get(role, role.replace("_", " ").title()),
            "target_words": _ROLE_TARGET_WORDS.get(role, 250),
            "assigned_fact_ids": assigned,
            "writing_intent": f"deliver the key insight for this {role} section",
            "opening_constraint": "",
        })
    return sections


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
    max_tokens: int = 1800,
    brand_context: Optional[BrandContext] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """
    Returns (sections_list, usage).
    Falls back to deterministic plan if LLM fails.
    """
    arc = analysis.get("arc", "narrative")

    # Enrich audience signal: prefer brand config over TopicAnalyst inference
    audience_str = analysis.get("audience", "general reader")
    if brand_context:
        brand_audience = brand_context.audience_context()
        if brand_audience:
            audience_str = brand_audience

    user = SECTION_PLANNER_USER.format(
        title=title,
        content_type=analysis.get("content_type", "explainer"),
        arc=arc,
        audience=audience_str,
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
        parsed = parse_json_response(raw)
        usage = extract_usage(resp)
    except Exception as e:
        import logging; logging.getLogger("section_planner").warning("plan_sections LLM failed, using fallback: %s", e)
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

    # Always guarantee a closing section — LLM planners sometimes omit it
    if not any(s["role"] == "closing" for s in sections):
        sections.append({
            "index": len(sections),
            "role": "closing",
            "heading": None,
            "target_words": _ROLE_TARGET_WORDS["closing"],
            "assigned_fact_ids": [],
            "writing_intent": "deliver a single forward-looking observation that crystallizes the article's most important insight",
            "opening_constraint": "",
        })

    return sections, usage
