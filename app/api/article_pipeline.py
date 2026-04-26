"""
Single-endpoint article generation pipeline.

POST /api/v1/articles/pipeline          — start pipeline (returns pipeline_id)
GET  /api/v1/articles/pipeline/{id}     — poll status + get result
PATCH /api/v1/articles/pipeline/{id}/resume — resume from failed step

Orchestrated by LangGraph StateGraph (app/services/article_graph.py).
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import psycopg2
from fastapi import APIRouter, BackgroundTasks, Depends, Header, HTTPException, Query
from pydantic import BaseModel, Field
from jose import JWTError, jwt

logger = logging.getLogger("article_pipeline")

router = APIRouter(prefix="/api/v1/articles", tags=["article-pipeline"])


# ============================================================
# Auth (same pattern as other endpoints)
# ============================================================
class Claims(BaseModel):
    tenant_id: str
    user_id: str
    role: str
    exp: int


def _jwt_secret():
    for k in ("JWT_SECRET_KEY", "JWT_SECRET"):
        v = os.getenv(k)
        if v:
            return v
    try:
        from app.core.config import get_settings
        s = get_settings()
        for k in ("JWT_SECRET_KEY", "JWT_SECRET", "jwt_secret_key"):
            if hasattr(s, k) and getattr(s, k):
                return str(getattr(s, k))
    except Exception:
        pass
    return None


def require_claims(authorization: str = Header(..., description="Bearer <JWT>")) -> Claims:
    secret = _jwt_secret()
    if not secret:
        raise HTTPException(status_code=500, detail="JWT secret not configured")
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing Bearer token")
    token = authorization[7:]
    try:
        payload = jwt.decode(token, secret, algorithms=["HS256"])
        return Claims(
            tenant_id=payload.get("tenant_id", payload.get("sub", "")),
            user_id=payload.get("user_id", payload.get("sub", "")),
            role=payload.get("role", "viewer"),
            exp=int(payload.get("exp", 0)),
        )
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")


# ============================================================
# Settings + DB helpers
# ============================================================
def _get_settings():
    try:
        from app.core.config import get_settings
        return get_settings()
    except Exception:
        from app.core.config import settings as _s
        return _s


def _db_conn(settings):
    params = {}
    for key, names in [
        ("host", ["DB_HOST"]), ("port", ["DB_PORT"]), ("dbname", ["DB_NAME"]),
        ("user", ["DB_USER"]), ("password", ["DB_PASSWORD"]),
    ]:
        for n in names:
            val = getattr(settings, n, None) or os.getenv(n)
            if val:
                params[key] = val
                break
    params["sslmode"] = getattr(settings, "DB_SSLMODE", None) or os.getenv("DB_SSLMODE", "require")
    return psycopg2.connect(**params)


def _ensure_pipeline_table(conn):
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS public.pipeline_runs (
                pipeline_id UUID PRIMARY KEY,
                tenant_id UUID NOT NULL,
                kb_id UUID NOT NULL,
                request_id UUID,
                config JSONB NOT NULL DEFAULT '{}',
                status TEXT NOT NULL DEFAULT 'pending',
                current_step TEXT NOT NULL DEFAULT 'crawl',
                failed_step TEXT,
                error_detail TEXT,
                result_summary JSONB,
                started_at TIMESTAMPTZ,
                completed_at TIMESTAMPTZ,
                created_at TIMESTAMPTZ NOT NULL DEFAULT now()
            );
        """)
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_pipeline_runs_tenant ON public.pipeline_runs (tenant_id, status);"
        )
    conn.commit()


# ============================================================
# Request / Response models
# ============================================================
class PipelineCreateRequest(BaseModel):
    # Source — provide url, urls (up to 8), OR set skip_crawl=True with existing kb_id
    url: Optional[str] = None
    urls: List[str] = Field(default_factory=list)
    skip_crawl: bool = False

    # Article
    kb_id: str
    title: str
    keywords: List[str] = Field(default_factory=list)
    length_target: int = 2000

    # Crawl config (only used if skip_crawl=False)
    max_depth: int = Field(0, ge=0, le=12)
    max_pages: int = Field(1, ge=1, le=100)
    respect_robots: bool = False
    user_agent: str = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"

    # Draft config
    draft_provider: str = "openai"
    draft_model: str = ""
    temperature: float = 0.7
    max_output_tokens: int = 8192
    top_k_sources: int = 8
    rag_grounding_ratio: float = Field(0.95, ge=0.8, le=0.99)
    enable_agentic_orchestration: bool = True
    expanded_query_count: int = Field(4, ge=1, le=6)
    hybrid_top_k_per_query: int = Field(30, ge=5, le=60)
    predictability_top_n: int = Field(14, ge=6, le=30)
    max_predictability_rewrite_passes: int = Field(1, ge=0, le=2)
    zerogpt_fix_max_attempts: int = Field(4, ge=1, le=8)
    max_quality_retries: int = Field(4, ge=0, le=10)

    # Reuse existing
    request_id: Optional[str] = None  # reuse existing article request


class PipelineCreateResponse(BaseModel):
    status: str
    pipeline_id: str
    poll_url: str


class PipelineStatusResponse(BaseModel):
    status: str
    pipeline_id: str
    pipeline_status: str
    current_step: str
    failed_step: Optional[str] = None
    error_detail: Optional[str] = None
    request_id: Optional[str] = None
    steps_completed: Optional[List[str]] = None
    result_summary: Optional[Dict[str, Any]] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None


# ============================================================
# Background pipeline runner
# ============================================================
def _run_pipeline_background(pipeline_id: str, initial_state: dict):
    """Runs the LangGraph pipeline in a background thread."""
    settings = _get_settings()
    # Must be initialised before the try block so the except handler can always
    # reference it, even if the exception fires before the stream loop begins.
    final_state: dict = {}

    # Update status to running
    try:
        conn = _db_conn(settings)
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE public.pipeline_runs SET status='running', started_at=now() WHERE pipeline_id=%s",
                (pipeline_id,),
            )
        conn.commit()
        conn.close()
    except Exception:
        pass

    # Watchdog: if the pipeline hangs (GPT timeout, etc.) mark it timed_out
    # after 15 minutes so the row is never stuck in 'running' forever.
    def _timeout_watchdog():
        logger.error("Pipeline %s exceeded 15-minute timeout — marking timed_out", pipeline_id)
        try:
            conn = _db_conn(settings)
            with conn.cursor() as cur:
                cur.execute(
                    """UPDATE public.pipeline_runs
                       SET status='timed_out', error_detail='Pipeline exceeded 15-minute timeout',
                           completed_at=now()
                       WHERE pipeline_id=%s AND status='running'""",
                    (pipeline_id,),
                )
            conn.commit()
            conn.close()
        except Exception:
            pass

    watchdog = threading.Timer(900, _timeout_watchdog)
    watchdog.daemon = True
    watchdog.start()

    try:
        from app.services.article_graph import compile_pipeline, _log_pipeline_event

        compiled = compile_pipeline()
        config = {"configurable": {"thread_id": pipeline_id}}

        _log_pipeline_event(
            initial_state["tenant_id"], pipeline_id,
            "pipeline_started", {"title": initial_state.get("title", "")}
        )

        # Stream through the graph — each node yields its output
        for step_output in compiled.stream(initial_state, config):
            # step_output is {node_name: {state_updates}}
            for node_name, updates in step_output.items():
                final_state.update(updates)

                # Check for fatal error (not warnings from finalize)
                if updates.get("error") and updates.get("current_step") not in ("completed", "completed_with_warnings"):
                    raise Exception(updates["error"])

                # Log step completion
                _log_pipeline_event(
                    initial_state["tenant_id"], pipeline_id,
                    "pipeline_step_completed",
                    {"step": node_name, "current_step": updates.get("current_step", node_name)},
                )

                # Update DB with current step
                try:
                    conn = _db_conn(settings)
                    with conn.cursor() as cur:
                        cur.execute(
                            """UPDATE public.pipeline_runs
                               SET current_step=%s, request_id=%s::uuid, status='running'
                               WHERE pipeline_id=%s""",
                            (
                                updates.get("current_step", node_name),
                                updates.get("request_id") or final_state.get("request_id"),
                                pipeline_id,
                            ),
                        )
                    conn.commit()
                    conn.close()
                except Exception:
                    pass

        # Pipeline completed — check if fully passed or with warnings
        final_status = final_state.get("current_step", "completed")
        warning = final_state.get("error")  # from finalize node (not a crash — a quality warning)

        result_summary = {
            "article_markdown": final_state.get("article_markdown", ""),
            "word_count": int(final_state.get("final_word_count") or len((final_state.get("article_markdown") or "").split())),
            "qc_pass": final_state.get("qc_pass", False),
            "qc_metrics": final_state.get("qc_metrics"),
            "qc_thresholds": final_state.get("qc_thresholds"),
            "zerogpt_score": final_state.get("zerogpt_score"),
            "zerogpt_pass": final_state.get("zerogpt_pass", False),
            "quality_retry_count": final_state.get("quality_retry_count", 0),
            "max_quality_retries": final_state.get("max_quality_retries"),
            "source_analysis": final_state.get("source_analysis"),
            "total_tokens": final_state.get("total_tokens", 0),
            "steps_completed": final_state.get("steps_completed", []),
            "draft_usage": final_state.get("draft_usage"),
            "qc_fix_usage": final_state.get("qc_fix_usage"),
            "zerogpt_fix_usage": final_state.get("zerogpt_fix_usage"),
            "warning": warning,
        }

        conn = _db_conn(settings)
        with conn.cursor() as cur:
            cur.execute(
                """UPDATE public.pipeline_runs
                   SET status=%s, current_step=%s,
                       request_id=%s::uuid, result_summary=%s::jsonb,
                       completed_at=now()
                   WHERE pipeline_id=%s""",
                (
                    final_status,
                    final_status,
                    final_state.get("request_id"),
                    json.dumps(result_summary, default=str),
                    pipeline_id,
                ),
            )
        conn.commit()
        conn.close()

        watchdog.cancel()
        _log_pipeline_event(
            initial_state["tenant_id"], pipeline_id,
            "pipeline_completed", result_summary,
        )
        logger.info("Pipeline %s completed successfully", pipeline_id)

    except Exception as e:
        watchdog.cancel()
        error_msg = str(e)[:500]
        logger.error("Pipeline %s failed: %s", pipeline_id, error_msg)

        try:
            conn = _db_conn(settings)
            with conn.cursor() as cur:
                cur.execute(
                    """UPDATE public.pipeline_runs
                       SET status='failed', failed_step=%s, error_detail=%s, completed_at=now()
                       WHERE pipeline_id=%s""",
                    (final_state.get("current_step", "unknown"), error_msg, pipeline_id),
                )
            conn.commit()
            conn.close()
        except Exception:
            pass

        _log_pipeline_event(
            initial_state["tenant_id"], pipeline_id,
            "pipeline_failed", {"error": error_msg, "step": final_state.get("current_step", "unknown")},
        )


# ============================================================
# Endpoints
# ============================================================
@router.post("/pipeline", response_model=PipelineCreateResponse, status_code=202)
def start_pipeline(
    req: PipelineCreateRequest,
    background: BackgroundTasks,
    claims: Claims = Depends(require_claims),
):
    """Start a full article generation pipeline. Returns immediately with pipeline_id."""
    # Validate
    if not req.skip_crawl and not req.url and not req.urls:
        raise HTTPException(status_code=400, detail="Provide 'url', 'urls' (list), or set 'skip_crawl': true")
    if req.urls and len(req.urls) > 8:
        raise HTTPException(status_code=400, detail="Maximum 8 URLs allowed in 'urls'")
    if not req.kb_id:
        raise HTTPException(status_code=400, detail="kb_id is required")
    if not req.title or not req.title.strip():
        raise HTTPException(status_code=400, detail="title is required")

    pipeline_id = str(uuid.uuid4())
    settings = _get_settings()

    conn = _db_conn(settings)
    _ensure_pipeline_table(conn)

    # Guard: max 2 active pipelines per tenant to prevent runaway GPT spend
    with conn.cursor() as cur:
        cur.execute(
            """SELECT COUNT(*) FROM public.pipeline_runs
               WHERE tenant_id=%s::uuid AND status IN ('pending', 'running')""",
            (claims.tenant_id,),
        )
        row = cur.fetchone()
        active = row[0] if row else 0

    if active >= 2:
        conn.close()
        raise HTTPException(
            status_code=429,
            detail=f"You already have {active} active pipeline(s) running. Wait for one to complete before starting another.",
        )

    # Create pipeline_runs row
    with conn.cursor() as cur:
        cur.execute(
            """INSERT INTO public.pipeline_runs
                 (pipeline_id, tenant_id, kb_id, config, status, current_step)
               VALUES (%s, %s::uuid, %s::uuid, %s::jsonb, 'pending', 'crawl')""",
            (pipeline_id, claims.tenant_id, req.kb_id, json.dumps(req.model_dump(), default=str)),
        )
    conn.commit()
    conn.close()

    # Build initial state for the graph
    initial_state = {
        "tenant_id": claims.tenant_id,
        "user_id": claims.user_id,
        "role": claims.role,
        "kb_id": req.kb_id,
        "title": req.title,
        "keywords": req.keywords,
        "url": req.url,
        "urls": req.urls,
        "skip_crawl": req.skip_crawl,
        "max_depth": req.max_depth,
        "max_pages": req.max_pages,
        "respect_robots": req.respect_robots,
        "user_agent": req.user_agent,
        "auto_pipeline": True,
        "draft_provider": req.draft_provider,
        "draft_model": req.draft_model,
        "temperature": req.temperature,
        "max_output_tokens": req.max_output_tokens,
        "top_k_sources": req.top_k_sources,
        "rag_grounding_ratio": req.rag_grounding_ratio,
        "enable_agentic_orchestration": req.enable_agentic_orchestration,
        "expanded_query_count": req.expanded_query_count,
        "hybrid_top_k_per_query": req.hybrid_top_k_per_query,
        "predictability_top_n": req.predictability_top_n,
        "max_predictability_rewrite_passes": req.max_predictability_rewrite_passes,
        "zerogpt_fix_max_attempts": req.zerogpt_fix_max_attempts,
        "max_quality_retries": req.max_quality_retries,
        "length_target": req.length_target,
        "request_id": req.request_id,
        "qc_pass": False,
        "zerogpt_pass": False,
        "quality_retry_count": 0,
        "zerogpt_checked_fingerprint": None,
        "total_tokens": 0,
        "steps_completed": [],
        "current_step": "crawl",
    }

    # Run in background
    background.add_task(_run_pipeline_background, pipeline_id, initial_state)

    return PipelineCreateResponse(
        status="accepted",
        pipeline_id=pipeline_id,
        poll_url=f"/api/v1/articles/pipeline/{pipeline_id}",
    )


@router.get("/pipeline/{pipeline_id}", response_model=PipelineStatusResponse)
def get_pipeline_status(
    pipeline_id: str,
    claims: Claims = Depends(require_claims),
):
    """Poll pipeline status. Returns result_summary when completed."""
    settings = _get_settings()
    conn = _db_conn(settings)
    with conn.cursor() as cur:
        cur.execute(
            """SELECT pipeline_id::text, tenant_id::text, kb_id::text, request_id::text,
                      status, current_step, failed_step, error_detail,
                      result_summary, started_at, completed_at
               FROM public.pipeline_runs
               WHERE pipeline_id=%s AND tenant_id=%s::uuid
               LIMIT 1""",
            (pipeline_id, claims.tenant_id),
        )
        row = cur.fetchone()
    conn.close()

    if not row:
        raise HTTPException(status_code=404, detail="Pipeline not found")

    (pid, tid, kb_id, request_id, status, current_step, failed_step,
     error_detail, result_summary, started_at, completed_at) = row

    # Extract steps_completed from result_summary
    steps = None
    if isinstance(result_summary, dict):
        steps = result_summary.get("steps_completed")

    return PipelineStatusResponse(
        status="ok",
        pipeline_id=str(pid),
        pipeline_status=str(status),
        current_step=str(current_step),
        failed_step=str(failed_step) if failed_step else None,
        error_detail=str(error_detail) if error_detail else None,
        request_id=str(request_id) if request_id else None,
        steps_completed=steps,
        result_summary=result_summary if isinstance(result_summary, dict) else None,
        started_at=started_at.isoformat() if started_at else None,
        completed_at=completed_at.isoformat() if completed_at else None,
    )


@router.patch("/pipeline/{pipeline_id}/resume")
def resume_pipeline(
    pipeline_id: str,
    background: BackgroundTasks,
    claims: Claims = Depends(require_claims),
):
    """Resume a failed pipeline from the failed step."""
    settings = _get_settings()
    conn = _db_conn(settings)
    with conn.cursor() as cur:
        cur.execute(
            """SELECT status, failed_step, config
               FROM public.pipeline_runs
               WHERE pipeline_id=%s AND tenant_id=%s::uuid
               LIMIT 1""",
            (pipeline_id, claims.tenant_id),
        )
        row = cur.fetchone()

    if not row:
        conn.close()
        raise HTTPException(status_code=404, detail="Pipeline not found")

    status, failed_step, config = row
    if status != "failed":
        conn.close()
        raise HTTPException(status_code=409, detail=f"Pipeline is '{status}', not 'failed'. Cannot resume.")

    # Reset to pending
    with conn.cursor() as cur:
        cur.execute(
            """UPDATE public.pipeline_runs
               SET status='pending', failed_step=NULL, error_detail=NULL
               WHERE pipeline_id=%s""",
            (pipeline_id,),
        )
    conn.commit()
    conn.close()

    # Rebuild initial state from config
    cfg = config if isinstance(config, dict) else {}
    initial_state = {
        "tenant_id": claims.tenant_id,
        "user_id": claims.user_id,
        "role": claims.role,
        "kb_id": cfg.get("kb_id", ""),
        "title": cfg.get("title", ""),
        "keywords": cfg.get("keywords", []),
        "url": cfg.get("url"),
        "urls": cfg.get("urls", []),
        "skip_crawl": True,  # Always skip crawl on resume
        "auto_pipeline": True,
        "draft_provider": cfg.get("draft_provider", "openai"),
        "draft_model": cfg.get("draft_model", ""),
        "temperature": cfg.get("temperature", 0.7),
        "max_output_tokens": cfg.get("max_output_tokens", 8192),
        "top_k_sources": cfg.get("top_k_sources", 8),
        "rag_grounding_ratio": cfg.get("rag_grounding_ratio", 0.95),
        "enable_agentic_orchestration": cfg.get("enable_agentic_orchestration", True),
        "expanded_query_count": cfg.get("expanded_query_count", 4),
        "hybrid_top_k_per_query": cfg.get("hybrid_top_k_per_query", 30),
        "predictability_top_n": cfg.get("predictability_top_n", 14),
        "max_predictability_rewrite_passes": cfg.get("max_predictability_rewrite_passes", 1),
        "zerogpt_fix_max_attempts": cfg.get("zerogpt_fix_max_attempts", 4),
        "max_quality_retries": cfg.get("max_quality_retries", 4),
        "length_target": cfg.get("length_target", 2000),
        "request_id": cfg.get("request_id"),
        "qc_pass": False,
        "zerogpt_pass": False,
        "quality_retry_count": 0,
        "zerogpt_checked_fingerprint": None,
        "total_tokens": 0,
        "steps_completed": [],
        "current_step": failed_step or "crawl",
    }

    background.add_task(_run_pipeline_background, pipeline_id, initial_state)

    return {
        "status": "resumed",
        "pipeline_id": pipeline_id,
        "resuming_from": failed_step or "crawl",
    }
