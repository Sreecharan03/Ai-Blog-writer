from __future__ import annotations

from fastapi import FastAPI
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



    return app


app = create_app()
