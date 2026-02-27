from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from app.api.health import router as health_router
from app.api.db import router as db_router
from app.api.auth import router as auth_router
from app.api.finance import router as finance_router
from app.api.finance_reports import router as finance_reports_router
from app.api.ingest import router as ingest_router
from app.api.kb import router as kb_router
from app.api.kb_docs import router as kb_docs_router
from app.api.url_ingest import router as url_ingest_router
from app.api.ingest_jobs import router as ingest_jobs_router
from app.api.jobs_events import router as jobs_events_router
from app.api.preprocess import router as preprocess_router
from app.api.chunk import router as chunk_router
from app.api.summarize import router as summarize_router
from app.api.embed import router as embed_router
from app.api.search import router as search_router
from app.api.hybrid_search import router as hybrid_search_router
from app.api.article_requests import router as article_requests_router
from app.api.article_run import router as article_run_router
from app.api.article_output import router as article_output_router
from app.api.article_state import router as article_state_router
from app.api.article_qc import router as article_qc_router
from app.api.article_zerogpt import router as article_zerogpt_router
from app.api.article_revise import router as article_revise_router

APP_NAME = "Sighnal Backend"


def create_app() -> FastAPI:
    app = FastAPI(
        title=APP_NAME,
        version="0.1.0",
    )

    # Versioned API
    app.include_router(health_router, prefix="/api/v1")
    app.include_router(db_router, prefix="/api/v1")
    app.include_router(auth_router, prefix="/api/v1")
    app.include_router(finance_router)
    app.include_router(finance_reports_router)
    app.include_router(ingest_router)
    app.include_router(kb_router)
    app.include_router(kb_docs_router)
    app.include_router(url_ingest_router)
    app.include_router(ingest_jobs_router)
    app.include_router(jobs_events_router)
    app.include_router(preprocess_router)
    app.include_router(chunk_router)
    app.include_router(summarize_router)
    app.include_router(embed_router)
    app.include_router(search_router)
    app.include_router(hybrid_search_router)
    app.include_router(article_requests_router)
    app.include_router(article_run_router)
    app.include_router(article_run_router)
    app.include_router(article_output_router)
    app.include_router(article_state_router)
    app.include_router(article_qc_router)
    app.include_router(article_zerogpt_router)
    app.include_router(article_revise_router)   

    # Serve the pipeline tester UI
    static_dir = Path(__file__).parent / "static"
    if static_dir.is_dir():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

        @app.get("/ui", include_in_schema=False)
        def serve_ui():
            return FileResponse(str(static_dir / "index.html"))

    return app


app = create_app()
