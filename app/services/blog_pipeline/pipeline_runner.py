"""
Blog Pipeline Orchestrator.
Phases:
  1. Planning   — Topic Analyst + Evidence Locker (parallel) -> Section Planner
  2. Writing    — Section Writer per section (sequential, for narrative continuity)
  3. Assembly   — Deterministic join + targeted expand if short
  4. Final QC   — readability metrics (caller runs ZeroGPT externally)

Returns a structured result dict the API layer can save to GCS.
"""
from __future__ import annotations
import concurrent.futures
import time
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI

from .agent_topic_analyst import analyze_topic
from .agent_evidence_locker import build_evidence_locker
from .agent_section_planner import plan_sections
from .agent_section_writer import write_section
from .agent_mini_humanize import humanize_section
from .assembler import assemble, section_count, has_faq
from .gates_local_qc import word_count, run_local_qc

MAX_SECTION_TOKENS = 750
MAX_ASSEMBLE_EXPAND_TOKENS = 500


def _sum_usage(*items: Dict[str, int]) -> Dict[str, int]:
    out = {"prompt_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    for d in items:
        for k in out:
            out[k] += d.get(k, 0)
    return out


def run_blog_pipeline(
    client: OpenAI,
    model: str,
    *,
    title: str,
    keywords: List[str],
    sources: List[Dict[str, Any]],
    target_words: int = 2000,
) -> Dict[str, Any]:
    """
    Full pipeline. Returns result dict with:
      draft_markdown, word_count, section_count, has_faq,
      section_meta (per-section QC results), usage, warnings
    """
    t0 = time.time()
    warnings: List[str] = []
    all_usage: Dict[str, int] = {"prompt_tokens": 0, "output_tokens": 0, "total_tokens": 0}

    # ── Phase 1A: Topic Analyst + Evidence Locker in parallel ────────────────
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        fut_analysis = pool.submit(analyze_topic, client, model, title, keywords)
        fut_locker = pool.submit(build_evidence_locker, client, model, sources)

        analysis, analysis_usage = fut_analysis.result()
        facts, is_sparse, locker_usage = fut_locker.result()

    all_usage = _sum_usage(all_usage, analysis_usage, locker_usage)

    if is_sparse:
        warnings.append(f"sparse_evidence: only {len(facts)} facts extracted — sections will supplement with general knowledge")

    # ── Phase 1B: Section Planner (needs both above) ─────────────────────────
    sections, planner_usage = plan_sections(
        client, model, title, analysis, facts,
        sparse=is_sparse, target_words=target_words,
    )
    all_usage = _sum_usage(all_usage, planner_usage)

    # ── Phase 2: Section Writing (sequential for narrative continuity) ────────
    written_sections: List[Dict[str, Any]] = []
    section_meta: List[Dict[str, Any]] = []
    prev_text: Optional[str] = None

    for sec in sections:
        role = sec.get("role", "body")

        text, qc_passed, failures, writer_usage, writer_meta = write_section(
            client, model,
            title=title,
            section=sec,
            facts=facts,
            sparse=is_sparse,
            prev_section_text=prev_text,
            max_tokens=MAX_SECTION_TOKENS,
        )
        all_usage = _sum_usage(all_usage, writer_usage)

        humanized = False
        humanize_usage: Dict[str, int] = {"prompt_tokens": 0, "output_tokens": 0, "total_tokens": 0}

        # Run mini-humanize if local AI-pattern gate failed
        if not qc_passed and failures:
            text, humanized, humanize_usage = humanize_section(
                client, model, text, failures, role=role,
            )
            all_usage = _sum_usage(all_usage, humanize_usage)
            if humanized:
                # Re-run QC on humanized version
                qc_passed, failures = run_local_qc(text, role=role)

        if not qc_passed:
            warnings.append(f"section[{role}] did not pass local QC after {writer_meta['attempts']} attempts + humanize: {failures[:2]}")

        written_sections.append({**sec, "text": text})
        prev_text = text

        section_meta.append({
            "index": sec.get("index"),
            "role": role,
            "heading": sec.get("heading"),
            "word_count": word_count(text),
            "qc_passed": qc_passed,
            "humanized": humanized,
            "writer_attempts": writer_meta.get("attempts", 1),
            "failures": failures,
        })

    # ── Phase 3: Assembly ─────────────────────────────────────────────────────
    draft_markdown, final_wc, assemble_usage = assemble(
        client, model, written_sections, facts,
        target_words=target_words,
        max_tokens_per_expand=MAX_ASSEMBLE_EXPAND_TOKENS,
    )
    all_usage = _sum_usage(all_usage, assemble_usage)

    if final_wc < 1900:
        warnings.append(f"final_word_count={final_wc} below 1900 minimum — ZeroGPT fix may inflate this further")
    if final_wc > 2600:
        warnings.append(f"final_word_count={final_wc} above 2600 maximum")

    sc = section_count(draft_markdown)
    faq = has_faq(draft_markdown)
    if not faq:
        warnings.append("no FAQ section detected in assembled article")

    elapsed = round(time.time() - t0, 1)

    return {
        "draft_markdown": draft_markdown,
        "title": title,
        "word_count": final_wc,
        "section_count": sc,
        "has_faq": faq,
        "is_sparse": is_sparse,
        "analysis": analysis,
        "section_plan": [
            {"role": s["role"], "heading": s.get("heading"), "target_words": s.get("target_words")}
            for s in sections
        ],
        "section_meta": section_meta,
        "usage": all_usage,
        "warnings": warnings,
        "elapsed_seconds": elapsed,
    }
