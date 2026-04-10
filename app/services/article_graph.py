"""
LangGraph-based article generation pipeline.

Graph:  crawl → create_request → draft → qc → [qc_fix] → zerogpt → [zerogpt_fix] → finalize
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
# Pipeline State — flows between all nodes
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


# ============================================================
# Node 1: CRAWL — fetch URL, extract, preprocess, chunk, embed
# ============================================================
def crawl_node(state: PipelineState) -> dict:
    """Crawl a URL and auto-pipeline (preprocess → chunk → embed)."""
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
# Node 2: CREATE REQUEST — create article_requests row
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
# Node 3: DRAFT — Phase 1 analysis + outline + GPT-5.2 draft
# ============================================================
def draft_node(state: PipelineState) -> dict:
    """Generate article draft using GPT-5.2 outline-first architecture."""
    from app.api.article_run import run_article_request, RunRequest

    req = RunRequest(
        draft_provider=state.get("draft_provider", "openai"),
        draft_model=state.get("draft_model", ""),
        temperature=state.get("temperature", 0.7),
        max_output_tokens=state.get("max_output_tokens", 8192),
        top_k_sources=state.get("top_k_sources", 8),
    )

    claims = _make_claims(state)
    result = run_article_request(state["request_id"], req, claims)

    return {
        "draft_uri": result.gcs_draft_uri,
        "draft_fingerprint": result.draft_fingerprint,
        "draft_model_used": result.draft_model,
        "draft_usage": result.usage,
        "steps_completed": state.get("steps_completed", []) + ["draft"],
        "current_step": "qc",
    }


# ============================================================
# Node 4: QC — readability, word count, FK, FRE, FAQ check
# ============================================================
def qc_node(state: PipelineState) -> dict:
    """Run QC check on the draft."""
    from app.api.article_qc import get_qc_report

    claims = _make_claims(state)
    # Pass explicit values for Query() params (they don't resolve outside FastAPI)
    result = get_qc_report(state["request_id"], claims, signed_url=False, signed_url_minutes=15)

    return {
        "qc_pass": result.qc_pass,
        "qc_metrics": result.qc_metrics,
        "steps_completed": state.get("steps_completed", []) + ["qc"],
        "current_step": "route_qc",
    }


# ============================================================
# Node 5: QC FIX — revise draft if QC failed
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

    return {
        "qc_pass": result.qc_pass,
        "qc_metrics": result.qc_metrics,
        "qc_fix_usage": result.usage,
        "steps_completed": state.get("steps_completed", []) + ["qc_fix"],
        "current_step": "zerogpt",
    }


# ============================================================
# Node 6: ZEROGPT — AI detection check
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
        "steps_completed": state.get("steps_completed", []) + ["zerogpt"],
        "current_step": "route_zerogpt",
    }


# ============================================================
# Node 7: ZEROGPT FIX — surgical humanization
# ============================================================
def zerogpt_fix_node(state: PipelineState) -> dict:
    """Surgically fix AI-detected sentences."""
    from app.api.article_zerogpt_fix import zerogpt_fix, ZeroGPTFixRequest

    req = ZeroGPTFixRequest(
        model="",
        temperature=0.9,
        max_output_tokens=8192,
        max_attempts=4,  # cap at 4 to prevent token waste
        force=False,
    )

    claims = _make_claims(state)
    result = zerogpt_fix(state["request_id"], req, claims)

    return {
        "zerogpt_score": result.final_score,
        "zerogpt_pass": result.zerogpt_pass,
        "zerogpt_fix_usage": result.usage,
        "steps_completed": state.get("steps_completed", []) + ["zerogpt_fix"],
        "current_step": "finalize",
    }


# ============================================================
# Node 8: FINALIZE — collect results, build summary
# ============================================================
def finalize_node(state: PipelineState) -> dict:
    """Build final result summary with article content and all metrics."""
    # Fetch the final article markdown from GCS
    article_markdown = ""
    try:
        from google.cloud import storage as gcs_storage
        settings = _get_settings()
        creds_path = getattr(settings, "GOOGLE_APPLICATION_CREDENTIALS", None) or os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        if creds_path:
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = creds_path
        gcs = gcs_storage.Client()
        bucket_name = getattr(settings, "GCS_BUCKET_NAME", None) or os.getenv("GCS_BUCKET_NAME")
        bucket = gcs.bucket(bucket_name)

        # Try humanized version first (zerogpt-fixed), then original draft
        draft_uri = state.get("draft_uri", "")
        # Check if there's a humanized version by looking at zerogpt_fix
        if state.get("zerogpt_fix_usage"):
            # Humanized version exists — look for it
            prefix = draft_uri.replace("draft_v1/", "zerogpt_fix_v1/").rsplit("/", 1)[0] + "/"
            prefix = prefix.replace(f"gs://{bucket_name}/", "")
            blobs = list(bucket.list_blobs(prefix=prefix))
            for b in blobs:
                name = b.name.split("/")[-1]
                if name.endswith(".json") and not name.startswith("zerogpt_"):
                    data = json.loads(b.download_as_text())
                    article_markdown = data.get("draft", {}).get("draft_markdown", "")
                    if article_markdown:
                        break

        if not article_markdown and draft_uri:
            # Use original draft
            obj_path = draft_uri.replace(f"gs://{bucket_name}/", "")
            blob = bucket.blob(obj_path)
            data = json.loads(blob.download_as_text())
            article_markdown = data.get("draft", {}).get("draft_markdown", "")
            source_analysis = data.get("draft", {}).get("source_analysis")
            if source_analysis:
                state["source_analysis"] = source_analysis

    except Exception as e:
        logger.warning("finalize: failed to fetch article markdown: %s", str(e)[:200])

    # Calculate total tokens
    total = 0
    for usage_key in ["draft_usage", "qc_fix_usage", "zerogpt_fix_usage"]:
        usage = state.get(usage_key)
        if isinstance(usage, dict):
            total += int(usage.get("total_tokens", 0))

    # Determine final status
    zerogpt_passed = state.get("zerogpt_pass", False)
    qc_passed = state.get("qc_pass", False)

    if not qc_passed:
        final_step = "completed_with_warnings"
        error = f"QC did not pass. Metrics: {state.get('qc_metrics', {})}"
    elif not zerogpt_passed:
        final_step = "completed_with_warnings"
        error = f"ZeroGPT score {state.get('zerogpt_score', '?')}% did not pass threshold (<20%)"
    else:
        final_step = "completed"
        error = None

    result = {
        "article_markdown": article_markdown,
        "total_tokens": total,
        "steps_completed": state.get("steps_completed", []) + ["finalize"],
        "current_step": final_step,
    }
    if error:
        result["error"] = error
    return result


# ============================================================
# Routing functions — conditional edges
# ============================================================
def route_after_qc(state: PipelineState) -> str:
    """Skip qc_fix if QC already passed."""
    if state.get("qc_pass"):
        return "zerogpt"
    return "qc_fix"


def route_after_zerogpt(state: PipelineState) -> str:
    """Skip zerogpt_fix if ZeroGPT already passed."""
    if state.get("zerogpt_pass"):
        return "finalize"
    return "zerogpt_fix"


# ============================================================
# Build the graph
# ============================================================
def build_article_pipeline() -> StateGraph:
    """
    Construct the LangGraph StateGraph for the article generation pipeline.

    Graph:
        START → crawl → create_request → draft → qc
                                                   ↓
                                             route_after_qc
                                              ↙         ↘
                                         qc_fix       zerogpt
                                            ↓            ↓
                                         zerogpt    route_after_zerogpt
                                                     ↙         ↘
                                              zerogpt_fix    finalize
                                                   ↓            ↓
                                               finalize        END
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

    # Add edges — linear flow
    graph.add_edge(START, "crawl")
    graph.add_edge("crawl", "create_request")
    graph.add_edge("create_request", "draft")
    graph.add_edge("draft", "qc")

    # Conditional: QC pass → skip qc_fix
    graph.add_conditional_edges("qc", route_after_qc, {
        "qc_fix": "qc_fix",
        "zerogpt": "zerogpt",
    })

    # qc_fix always goes to zerogpt
    graph.add_edge("qc_fix", "zerogpt")

    # Conditional: ZeroGPT pass → skip zerogpt_fix
    graph.add_conditional_edges("zerogpt", route_after_zerogpt, {
        "zerogpt_fix": "zerogpt_fix",
        "finalize": "finalize",
    })

    # zerogpt_fix always goes to finalize
    graph.add_edge("zerogpt_fix", "finalize")

    # finalize → END
    graph.add_edge("finalize", END)

    return graph


def compile_pipeline(checkpointer=None):
    """Compile the graph with optional checkpointer."""
    graph = build_article_pipeline()
    if checkpointer is None:
        checkpointer = MemorySaver()
    return graph.compile(checkpointer=checkpointer)
