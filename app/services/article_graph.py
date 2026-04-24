"""
LangGraph-based article generation pipeline.

Graph:  crawl -> create_request -> draft -> qc -> [qc_fix] -> zerogpt -> [zerogpt_fix] -> finalize
                                                  ↑ conditional          ↑ conditional

Each node calls existing endpoint functions directly (no HTTP).
HUMANIZATION_CORE (anti-AI detection rules) stays locked throughout.
"""
from __future__ import annotations

import json
import logging
import os
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver

logger = logging.getLogger("article_graph")


# ============================================================
# Pipeline State  -  flows between all nodes
# ============================================================
class PipelineState(TypedDict, total=False):
    # ── Input (set at pipeline start) ──
    tenant_id: str
    user_id: str
    role: str
    kb_id: str
    title: str
    keywords: List[str]
    url: Optional[str]
    skip_crawl: bool

    # ── Crawl config ──
    max_depth: int
    max_pages: int
    respect_robots: bool
    user_agent: str
    auto_pipeline: bool

    # ── Draft config ──
    draft_provider: str
    draft_model: str
    temperature: float
    max_output_tokens: int
    top_k_sources: int
    length_target: int
    rag_grounding_ratio: float
    enable_agentic_orchestration: bool
    expanded_query_count: int
    hybrid_top_k_per_query: int
    predictability_top_n: int
    max_predictability_rewrite_passes: int
    zerogpt_fix_max_attempts: int
    max_quality_retries: int

    # ── Carried between steps ──
    request_id: Optional[str]
    crawl_stats: Optional[Dict[str, Any]]
    draft_uri: Optional[str]
    draft_fingerprint: Optional[str]
    draft_model_used: Optional[str]
    source_analysis: Optional[Dict[str, Any]]
    draft_usage: Optional[Dict[str, int]]

    qc_pass: bool
    qc_metrics: Optional[Dict[str, Any]]
    qc_fix_usage: Optional[Dict[str, int]]

    zerogpt_score: Optional[float]
    zerogpt_pass: bool
    zerogpt_fix_usage: Optional[Dict[str, int]]
    zerogpt_checked_fingerprint: Optional[str]
    quality_retry_count: int
    post_humanization: bool  # set True by zerogpt_fix_node; qc_node uses relaxed thresholds when True

    # ── Final output ──
    article_markdown: Optional[str]
    total_tokens: int
    steps_completed: List[str]
    error: Optional[str]
    current_step: str


# ============================================================
# Shared helpers
# ============================================================
def _get_settings():
    """Import settings lazily to avoid circular imports."""
    try:
        from app.core.config import get_settings
        return get_settings()
    except Exception:
        from app.core.config import settings as _s
        return _s


def _make_claims(state: PipelineState):
    """Build a Claims-compatible object from pipeline state."""
    from pydantic import BaseModel

    class Claims(BaseModel):
        tenant_id: str
        user_id: str
        role: str
        exp: int

    return Claims(
        tenant_id=state["tenant_id"],
        user_id=state.get("user_id", "pipeline"),
        role=state.get("role", "tenant_admin"),
        exp=int(time.time()) + 3600,  # 1 hour from now
    )


def _log_pipeline_event(tenant_id: str, pipeline_id: str, event_type: str, detail: Dict[str, Any]):
    """Log to job_events table for SSE streaming."""
    try:
        import psycopg2
        settings = _get_settings()
        db_params = {}
        for key, env_names in [
            ("host", ["DB_HOST", "SUPABASE_DB_HOST"]),
            ("port", ["DB_PORT", "SUPABASE_DB_PORT"]),
            ("dbname", ["DB_NAME", "SUPABASE_DB_NAME"]),
            ("user", ["DB_USER", "SUPABASE_DB_USER"]),
            ("password", ["DB_PASSWORD", "SUPABASE_DB_PASSWORD"]),
        ]:
            for env in env_names:
                val = getattr(settings, env, None) or os.getenv(env)
                if val:
                    db_params[key] = val
                    break
        db_params["sslmode"] = getattr(settings, "DB_SSLMODE", None) or os.getenv("DB_SSLMODE", "require")

        conn = psycopg2.connect(**db_params)
        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO public.job_events (event_id, tenant_id, job_id, event_type, detail, created_at)
                   VALUES (%s, %s::uuid, %s, %s, %s::jsonb, now())""",
                (str(uuid.uuid4()), tenant_id, pipeline_id, event_type, json.dumps(detail, default=str)),
            )
        conn.commit()
        conn.close()
    except Exception as e:
        logger.warning("Failed to log pipeline event: %s", str(e)[:200])


def _final_qc_thresholds() -> Dict[str, Any]:
    return {
        "word_count_min": 1900,
        "word_count_max": 2500,
        "fk_grade_min": 5.0,
        "fk_grade_max": 12.0,
        "flesch_reading_ease_min": 50.0,
        "flesch_reading_ease_max": 75.0,
        "repetition_ratio_max": 0.15,
        "unique_sections_min": 6,
        "unique_sections_max": 16,
        "faq_section_required": True,
    }


def _post_humanization_qc_thresholds() -> Dict[str, Any]:
    # After zerogpt_fix, humanization legitimately shifts FK grade and FRE.
    # Only enforce structural integrity  -  word count, sections, FAQ, repetition.
    # FK/FRE are intentionally unchecked here to avoid an unescapable loop.
    return {
        "word_count_min": 1900,
        "word_count_max": 2500,
        "fk_grade_min": 0.0,
        "fk_grade_max": 99.0,
        "flesch_reading_ease_min": 0.0,
        "flesch_reading_ease_max": 100.0,
        "repetition_ratio_max": 0.20,
        "unique_sections_min": 6,
        "unique_sections_max": 16,
        "faq_section_required": True,
    }


def _recompute_qc_with_thresholds(markdown: str, thresholds: Dict[str, Any]) -> Dict[str, Any]:
    """Core QC evaluation against an explicit threshold set."""
    from app.api.article_qc import _readability, _has_faq_section

    metrics = _readability(markdown or "")
    wc = float(metrics.get("word_count", 0.0))
    fk = float(metrics.get("flesch_kincaid_grade", 0.0))
    fre = float(metrics.get("flesch_reading_ease", 0.0))
    rep_ratio = float(metrics.get("repetition_ratio", 0.0))
    unique_sec = int(metrics.get("unique_sections", 0))
    has_faq = bool(_has_faq_section(markdown or ""))
    metrics["has_faq_section"] = has_faq

    fre_max = thresholds.get("flesch_reading_ease_max", 100.0)
    sec_max = thresholds.get("unique_sections_max", 9999)
    qc_pass = (
        thresholds["word_count_min"] <= wc <= thresholds["word_count_max"]
        and thresholds["fk_grade_min"] <= fk <= thresholds["fk_grade_max"]
        and thresholds["flesch_reading_ease_min"] <= fre <= fre_max
        and rep_ratio <= thresholds["repetition_ratio_max"]
        and thresholds["unique_sections_min"] <= unique_sec <= sec_max
        and (has_faq or not thresholds.get("faq_section_required", False))
    )
    return {"qc_pass": bool(qc_pass), "qc_metrics": metrics, "qc_thresholds": thresholds}


def _recompute_qc_for_final_markdown(markdown: str) -> Dict[str, Any]:
    return _recompute_qc_with_thresholds(markdown, _final_qc_thresholds())


def _fetch_draft_from_gcs(state: "PipelineState") -> Tuple[str, Optional[Dict[str, Any]]]:
    """Single GCS read returning (draft_markdown, source_analysis). Both default to empty/None."""
    try:
        from google.cloud import storage as gcs_storage
        settings = _get_settings()
        creds_path = getattr(settings, "GOOGLE_APPLICATION_CREDENTIALS", None) or os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        if creds_path:
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = creds_path
        gcs = gcs_storage.Client()
        draft_uri = str(state.get("draft_uri", "") or "")
        if draft_uri.startswith("gs://"):
            rest = draft_uri[5:]
            uri_bucket, _, obj_path = rest.partition("/")
            if uri_bucket and obj_path:
                data = json.loads(gcs.bucket(uri_bucket).blob(obj_path).download_as_text())
                draft = data.get("draft") if isinstance(data, dict) else {}
                if not isinstance(draft, dict):
                    draft = {}
                return draft.get("draft_markdown", ""), draft.get("source_analysis")
    except Exception as e:
        logger.warning("_fetch_draft_from_gcs failed: %s", str(e)[:200])
    return "", None


def _to_int(v: Any, default: int) -> int:
    try:
        return int(v)
    except Exception:
        return int(default)


def _max_quality_retries(state: PipelineState) -> int:
    return max(0, _to_int(state.get("max_quality_retries", 4), 4))


def _quality_retry_count(state: PipelineState) -> int:
    return max(0, _to_int(state.get("quality_retry_count", 0), 0))


def _can_retry_quality(state: PipelineState) -> bool:
    return _quality_retry_count(state) < _max_quality_retries(state)


# ============================================================
# Node 1: CRAWL  -  fetch URL, extract, preprocess, chunk, embed
# ============================================================
def crawl_node(state: PipelineState) -> dict:
    """Crawl a URL and auto-pipeline (preprocess -> chunk -> embed)."""
    if state.get("skip_crawl"):
        logger.info("Pipeline: skip_crawl=True, skipping crawl step")
        return {
            "crawl_stats": {"skipped": True},
            "steps_completed": state.get("steps_completed", []) + ["crawl_skipped"],
            "current_step": "create_request",
        }

    url = state.get("url")
    if not url:
        return {"error": "No URL provided and skip_crawl=False", "current_step": "crawl"}

    from app.api.url_ingest import URLIngestRequest, CrawlJob, _crawl_and_ingest, _normalize_url
    from urllib.parse import urlparse

    seed_norm = _normalize_url(url)
    seed_host = urlparse(seed_norm).netloc.lower()

    payload = URLIngestRequest(
        url=url,
        max_depth=state.get("max_depth", 0),
        max_pages=state.get("max_pages", 1),
        same_host_only=True,
        respect_robots=state.get("respect_robots", False),
        user_agent=state.get("user_agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"),
        auto_pipeline=state.get("auto_pipeline", True),
        wait=True,
    )

    job = CrawlJob(
        tenant_id=state["tenant_id"],
        user_id=state.get("user_id", "pipeline"),
        kb_id=state["kb_id"],
        job_id=str(uuid.uuid4()),
        seed_url=seed_norm,
        seed_host=seed_host,
        scope="tenant_private",
        config=payload,
    )

    stats = _crawl_and_ingest(job)

    return {
        "crawl_stats": stats,
        "steps_completed": state.get("steps_completed", []) + ["crawl"],
        "current_step": "create_request",
    }


# ============================================================
# Node 2: CREATE REQUEST  -  create article_requests row
# ============================================================
def create_request_node(state: PipelineState) -> dict:
    """Create an article request in the DB."""
    if state.get("request_id"):
        # Reuse existing request
        return {
            "steps_completed": state.get("steps_completed", []) + ["create_request_reused"],
            "current_step": "draft",
        }

    from app.api.article_requests import create_article_request, ArticleRequestCreate

    payload = ArticleRequestCreate(
        kb_id=state["kb_id"],
        title=state["title"],
        keywords=state.get("keywords", []),
        length_target=state.get("length_target", 2000),
    )

    claims = _make_claims(state)
    result = create_article_request(payload, claims)
    request_id = result.request.request_id

    return {
        "request_id": request_id,
        "steps_completed": state.get("steps_completed", []) + ["create_request"],
        "current_step": "draft",
    }


# ============================================================
# Node 3: DRAFT  -  Phase 1 analysis + outline + GPT-5.2 draft
# ============================================================
def draft_node(state: PipelineState) -> dict:
    """Generate article draft using the agentic grounded pipeline."""
    from app.api.article_run import run_article_request, RunRequest

    req = RunRequest(
        draft_provider=state.get("draft_provider", "openai"),
        draft_model=state.get("draft_model", ""),
        temperature=state.get("temperature", 0.7),
        max_output_tokens=state.get("max_output_tokens", 8192),
        top_k_sources=state.get("top_k_sources", 8),
        rag_grounding_ratio=state.get("rag_grounding_ratio", 0.95),
        enable_agentic_orchestration=state.get("enable_agentic_orchestration", True),
        expanded_query_count=state.get("expanded_query_count", 4),
        hybrid_top_k_per_query=state.get("hybrid_top_k_per_query", 30),
        predictability_top_n=state.get("predictability_top_n", 14),
        max_predictability_rewrite_passes=state.get("max_predictability_rewrite_passes", 1),
    )

    claims = _make_claims(state)
    result = run_article_request(state["request_id"], req, claims)

    return {
        "draft_uri": result.gcs_draft_uri,
        "draft_fingerprint": result.draft_fingerprint,
        "draft_model_used": result.draft_model,
        "draft_usage": result.usage,
        "zerogpt_checked_fingerprint": None,
        "steps_completed": state.get("steps_completed", []) + ["draft"],
        "current_step": "qc",
    }


# ============================================================
# Node 4: QC  -  readability, word count, FK, FRE, FAQ check
# ============================================================
def qc_node(state: PipelineState) -> dict:
    """Run QC check on the draft.

    Post-humanization path: zerogpt_fix legitimately shifts FK grade and FRE.
    Using full thresholds after humanization creates an unescapable loop. Instead,
    after zerogpt_fix we only enforce structural integrity (word count, sections, FAQ).
    The finalize_node always re-runs full thresholds for the final reported status.
    """
    if state.get("post_humanization"):
        markdown, _ = _fetch_draft_from_gcs(state)
        result = _recompute_qc_with_thresholds(markdown, _post_humanization_qc_thresholds())
        return {
            "qc_pass": result["qc_pass"],
            "qc_metrics": result["qc_metrics"],
            "post_humanization": False,
            "steps_completed": state.get("steps_completed", []) + ["qc_post_humanization"],
            "current_step": "route_qc",
        }

    from app.api.article_qc import get_qc_report
    claims = _make_claims(state)
    result = get_qc_report(state["request_id"], claims, signed_url=False, signed_url_minutes=15)

    return {
        "qc_pass": result.qc_pass,
        "qc_metrics": result.qc_metrics,
        "post_humanization": False,
        "steps_completed": state.get("steps_completed", []) + ["qc"],
        "current_step": "route_qc",
    }


# ============================================================
# Node 5: QC FIX  -  revise draft if QC failed
# ============================================================
def qc_fix_node(state: PipelineState) -> dict:
    """Fix the draft to pass QC thresholds."""
    from app.api.article_revise import qc_fix, QCFixRequest

    req = QCFixRequest(
        model="",
        temperature=0.3,
        max_output_tokens=9000,
        max_passes=8,
    )

    claims = _make_claims(state)
    result = qc_fix(state["request_id"], req, claims)
    next_retry = _quality_retry_count(state) + 1

    return {
        "draft_uri": result.gcs_new_draft_uri,
        "draft_fingerprint": result.new_draft_fingerprint,
        "qc_pass": result.qc_pass,
        "qc_metrics": result.qc_metrics,
        "qc_fix_usage": result.usage,
        "quality_retry_count": next_retry,
        "zerogpt_checked_fingerprint": None,
        "steps_completed": state.get("steps_completed", []) + ["qc_fix"],
        "current_step": "zerogpt",
    }


# ============================================================
# Node 6: ZEROGPT  -  AI detection check
# ============================================================
def zerogpt_node(state: PipelineState) -> dict:
    """Run ZeroGPT AI detection check."""
    from app.api.article_zerogpt import run_zerogpt

    claims = _make_claims(state)
    # Pass explicit values for Query() params (they don't resolve outside FastAPI)
    result = run_zerogpt(state["request_id"], signed_url=False, signed_url_minutes=15, force=False, claims=claims)

    return {
        "zerogpt_score": result.zerogpt_score,
        "zerogpt_pass": result.zerogpt_pass,
        "zerogpt_checked_fingerprint": state.get("draft_fingerprint"),
        "steps_completed": state.get("steps_completed", []) + ["zerogpt"],
        "current_step": "route_zerogpt",
    }


# ============================================================
# Node 7: ZEROGPT FIX  -  surgical humanization
# ============================================================
def zerogpt_fix_node(state: PipelineState) -> dict:
    """Surgically fix AI-detected sentences."""
    from app.api.article_zerogpt_fix import zerogpt_fix, ZeroGPTFixRequest

    req = ZeroGPTFixRequest(
        model="",
        temperature=0.9,
        max_output_tokens=8192,
        max_attempts=state.get("zerogpt_fix_max_attempts", 4),
        force=False,
    )

    claims = _make_claims(state)
    result = zerogpt_fix(state["request_id"], req, claims)
    next_retry = _quality_retry_count(state) + 1

    return {
        "draft_uri": result.gcs_humanized_uri,
        "draft_fingerprint": result.humanized_fingerprint,
        "zerogpt_score": result.final_score,
        "zerogpt_pass": result.zerogpt_pass,
        "zerogpt_fix_usage": result.usage,
        "zerogpt_checked_fingerprint": result.humanized_fingerprint,
        "quality_retry_count": next_retry,
        "post_humanization": True,
        "steps_completed": state.get("steps_completed", []) + ["zerogpt_fix"],
        "current_step": "qc",
    }


# ============================================================
# Node 8: FINALIZE  -  collect results, build summary
# ============================================================
def finalize_node(state: PipelineState) -> dict:
    """Build final result summary with article content and all metrics."""
    # Fetch the final article markdown from GCS
    article_markdown, source_analysis = _fetch_draft_from_gcs(state)
    if source_analysis:
        state["source_analysis"] = source_analysis

    # Final cleanup pass  -  catches anything zerogpt_fix missed
    if article_markdown:
        try:
            from app.api.article_zerogpt_fix import _clean_markdown
            article_markdown = _clean_markdown(article_markdown)
        except Exception as e:
            logger.warning("finalize: _clean_markdown failed: %s", str(e)[:200])

    # Recompute QC against final output to avoid stale pre-fix acceptance.
    qc_passed = bool(state.get("qc_pass", False))
    qc_metrics = state.get("qc_metrics")
    qc_thresholds = None
    if article_markdown:
        try:
            final_qc = _recompute_qc_for_final_markdown(article_markdown)
            qc_passed = bool(final_qc.get("qc_pass", False))
            qc_metrics = final_qc.get("qc_metrics")
            qc_thresholds = final_qc.get("qc_thresholds")
            state["qc_pass"] = qc_passed
            state["qc_metrics"] = qc_metrics
        except Exception as e:
            logger.warning("finalize: failed to recompute final QC: %s", str(e)[:200])

    # Calculate total tokens
    total = 0
    for usage_key in ["draft_usage", "qc_fix_usage", "zerogpt_fix_usage"]:
        usage = state.get(usage_key)
        if isinstance(usage, dict):
            total += int(usage.get("total_tokens", 0))

    # Determine final status
    zerogpt_passed = state.get("zerogpt_pass", False)

    if not qc_passed:
        final_step = "completed_with_warnings"
        error = f"QC did not pass on final output. Metrics: {qc_metrics}"
    elif not zerogpt_passed:
        final_step = "completed_with_warnings"
        error = f"ZeroGPT score {state.get('zerogpt_score', '?')}% did not pass threshold (<20%)"
    else:
        final_step = "completed"
        error = None

    result = {
        "article_markdown": article_markdown,
        "final_word_count": len((article_markdown or "").split()),
        "qc_pass": qc_passed,
        "qc_metrics": qc_metrics,
        "qc_thresholds": qc_thresholds,
        "quality_retry_count": _quality_retry_count(state),
        "max_quality_retries": _max_quality_retries(state),
        "total_tokens": total,
        "steps_completed": state.get("steps_completed", []) + ["finalize"],
        "current_step": final_step,
    }
    if error:
        result["error"] = error
    return result


# ============================================================
# Routing functions  -  conditional edges
# ============================================================
def route_after_qc(state: PipelineState) -> str:
    """
    Route after QC with bounded convergence.
    """
    qc_pass = bool(state.get("qc_pass", False))
    if not qc_pass:
        if _can_retry_quality(state):
            return "qc_fix"
        # Retry budget exhausted. zerogpt endpoint requires QC pass  -  sending there
        # would cause a 409 crash. finalize_node handles completed_with_warnings correctly.
        return "finalize"

    current_fp = str(state.get("draft_fingerprint") or "")
    checked_fp = str(state.get("zerogpt_checked_fingerprint") or "")
    if current_fp and checked_fp == current_fp:
        if bool(state.get("zerogpt_pass", False)):
            return "finalize"
        if _can_retry_quality(state):
            return "zerogpt_fix"
        return "finalize"

    return "zerogpt"


def route_after_zerogpt(state: PipelineState) -> str:
    """
    Route after ZeroGPT with coupled QC + ZeroGPT gate checks.
    """
    qc_pass = bool(state.get("qc_pass", False))
    zerogpt_pass = bool(state.get("zerogpt_pass", False))
    zerogpt_score = state.get("zerogpt_score")

    if qc_pass and zerogpt_pass:
        return "finalize"

    if not _can_retry_quality(state):
        return "finalize"

    if not qc_pass:
        return "qc_fix"

    # If ZeroGPT returned no score (rate limit / API error), zerogpt_fix will
    # crash with 409. Finalize with warning instead of crashing.
    if zerogpt_score is None:
        return "finalize"

    return "zerogpt_fix"


# ============================================================
# Build the graph
# ============================================================
def build_article_pipeline() -> StateGraph:
    """
    Construct the LangGraph StateGraph for the article generation pipeline.

    Graph:
        START -> crawl -> create_request -> draft -> qc
                                                   ↓
                                             route_after_qc
                                              ↙         ↘
                                         qc_fix       zerogpt
                                            ↓  ↖          ↓
                                           qc ←─┘   route_after_zerogpt
                                        (loop up        ↙         ↘
                                      to max_retries) zerogpt_fix  finalize
                                                          ↓           ↓
                                                         qc          END
                                                          ↓
                                                        END
    """
    graph = StateGraph(PipelineState)

    # Add nodes
    graph.add_node("crawl", crawl_node)
    graph.add_node("create_request", create_request_node)
    graph.add_node("draft", draft_node)
    graph.add_node("qc", qc_node)
    graph.add_node("qc_fix", qc_fix_node)
    graph.add_node("zerogpt", zerogpt_node)
    graph.add_node("zerogpt_fix", zerogpt_fix_node)
    graph.add_node("finalize", finalize_node)

    # Add edges  -  linear flow
    graph.add_edge(START, "crawl")
    graph.add_edge("crawl", "create_request")
    graph.add_edge("create_request", "draft")
    graph.add_edge("draft", "qc")

    # Conditional: QC pass -> skip qc_fix
    graph.add_conditional_edges("qc", route_after_qc, {
        "qc_fix": "qc_fix",
        "zerogpt": "zerogpt",
        "zerogpt_fix": "zerogpt_fix",
        "finalize": "finalize",
    })

    # qc_fix loops back to qc to re-verify  -  route_after_qc handles retry vs zerogpt
    graph.add_edge("qc_fix", "qc")

    # Conditional: ZeroGPT pass -> skip zerogpt_fix
    graph.add_conditional_edges("zerogpt", route_after_zerogpt, {
        "qc_fix": "qc_fix",
        "zerogpt_fix": "zerogpt_fix",
        "finalize": "finalize",
    })

    # zerogpt_fix mutates content; re-check QC before deciding completion
    graph.add_edge("zerogpt_fix", "qc")

    # finalize -> END
    graph.add_edge("finalize", END)

    return graph


def compile_pipeline(checkpointer=None):
    """Compile the graph with optional checkpointer."""
    graph = build_article_pipeline()
    if checkpointer is None:
        checkpointer = MemorySaver()
    return graph.compile(checkpointer=checkpointer)
