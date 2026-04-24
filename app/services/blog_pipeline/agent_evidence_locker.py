"""
Evidence Locker Agent (P2).
Extracts structured facts from retrieved source chunks.
Runs in parallel with Topic Analyst.
Falls back to raw text extraction if LLM fails.
"""
from __future__ import annotations
import json
import re
from typing import Any, Dict, List, Tuple

from openai import OpenAI
from .prompts import EVIDENCE_LOCKER_SYSTEM, EVIDENCE_LOCKER_USER

MAX_SOURCE_CHARS = 24000
MAX_FACTS = 36


def _build_sources_block(sources: List[Dict[str, Any]]) -> str:
    lines = []
    total = 0
    for i, s in enumerate(sources, start=1):
        sid = f"S{i}"
        text = (s.get("text") or "").strip()
        if not text:
            continue
        excerpt = text[: MAX_SOURCE_CHARS // max(1, len(sources))]
        total += len(excerpt)
        if total > MAX_SOURCE_CHARS:
            break
        lines.append(f"[{sid}] doc_id={s.get('doc_id','?')}\n{excerpt}\n")
    return "\n".join(lines)


def _fallback_facts(sources: List[Dict[str, Any]], limit: int = 20) -> List[Dict[str, Any]]:
    facts = []
    for i, s in enumerate(sources[:limit], start=1):
        text = (s.get("text") or "").strip()
        if not text:
            continue
        # Split into sentences, take first non-trivial one as a fact
        sents = re.split(r"(?<=[.!?])\s+", text)
        for sent in sents[:3]:
            sent = sent.strip()
            if len(sent.split()) >= 8:
                facts.append({
                    "fact_id": f"F{len(facts)+1}",
                    "source_id": f"S{i}",
                    "claim": sent[:150],
                    "confidence": "low",
                    "category": "finding",
                })
                break
        if len(facts) >= limit:
            break
    return facts


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


def build_evidence_locker(
    client: OpenAI,
    model: str,
    sources: List[Dict[str, Any]],
    max_facts: int = MAX_FACTS,
    temperature: float = 0.1,
    max_tokens: int = 2000,
) -> Tuple[List[Dict[str, Any]], bool, Dict[str, int]]:
    """
    Returns (facts, is_sparse, usage).
    is_sparse=True when fewer than 8 facts could be extracted.
    Never blocks the pipeline — falls back to raw extraction on failure.
    """
    if not sources:
        return [], True, {"prompt_tokens": 0, "output_tokens": 0, "total_tokens": 0}

    sources_block = _build_sources_block(sources)
    user = EVIDENCE_LOCKER_USER.format(sources_block=sources_block, max_facts=max_facts)

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": EVIDENCE_LOCKER_SYSTEM},
                {"role": "user", "content": user},
            ],
            temperature=temperature,
            max_completion_tokens=max_tokens,
        )
        raw = (resp.choices[0].message.content or "").strip()
        parsed = _parse(raw)
        usage = _usage(resp)
    except Exception:
        fallback = _fallback_facts(sources, limit=max_facts)
        return fallback, len(fallback) < 8, {"prompt_tokens": 0, "output_tokens": 0, "total_tokens": 0}

    facts_raw = parsed.get("facts") if isinstance(parsed.get("facts"), list) else []
    sparse_flag = bool(parsed.get("sparse", False))

    valid_source_ids = {f"S{i}" for i in range(1, len(sources) + 1)}
    facts: List[Dict[str, Any]] = []
    for item in facts_raw:
        if not isinstance(item, dict):
            continue
        claim = str(item.get("claim") or "").strip()
        sid = str(item.get("source_id") or "").strip()
        if not claim or sid not in valid_source_ids:
            continue
        facts.append({
            "fact_id": str(item.get("fact_id") or f"F{len(facts)+1}"),
            "source_id": sid,
            "claim": claim,
            "confidence": str(item.get("confidence") or "medium"),
            "category": str(item.get("category") or "finding"),
        })
        if len(facts) >= max_facts:
            break

    if not facts:
        facts = _fallback_facts(sources, limit=max_facts)

    is_sparse = sparse_flag or len(facts) < 8
    return facts, is_sparse, usage
