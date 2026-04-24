"""
POST /api/v1/kb/{kb_id}/discover

Takes a topic → calls Tavily → picks authoritative URLs → queues them for ingestion.
Returns discovered URLs immediately; ingestion runs in background.
"""
from __future__ import annotations

import os
import uuid
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import httpx
import psycopg2
import psycopg2.extras
from fastapi import APIRouter, BackgroundTasks, Depends, Header, HTTPException, Path
from jose import JWTError, jwt
from pydantic import BaseModel, Field

router = APIRouter(prefix="/api/v1/kb", tags=["kb-discover"])

# ── trusted domains for article research ────────────────────
_TRUSTED_DOMAINS = {
    # Health
    "healthline.com", "webmd.com", "mayoclinic.org", "nih.gov",
    "pubmed.ncbi.nlm.nih.gov", "medlineplus.gov", "health.harvard.edu",
    "sleepfoundation.org", "cdc.gov", "who.int",
    # Finance
    "investopedia.com", "forbes.com", "bloomberg.com", "wsj.com",
    "ft.com", "cnbc.com", "reuters.com",
    # Tech
    "techcrunch.com", "wired.com", "arstechnica.com", "thenextweb.com",
    # General authoritative
    "wikipedia.org", "britannica.com", "nature.com", "sciencedirect.com",
}

# ── settings & auth (same pattern as other modules) ─────────
try:
    from app.core.config import get_settings  # type: ignore
except Exception:
    from app.core.config import settings as _settings  # type: ignore

    def get_settings():  # type: ignore
        return _settings


def _pick(settings: Any, *names: str, default: Any = None) -> Any:
    for n in names:
        if hasattr(settings, n) and getattr(settings, n) not in (None, ""):
            return getattr(settings, n)
        val = os.getenv(n)
        if val not in (None, ""):
            return val
    return default


class Claims(BaseModel):
    tenant_id: str
    user_id: str
    role: str
    exp: int


def require_claims(authorization: str = Header(..., description="Bearer <JWT>")) -> Claims:
    if not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Missing Bearer token")
    token = authorization.split(" ", 1)[1].strip()
    secret = os.getenv("JWT_SECRET_KEY") or os.getenv("JWT_SECRET") or os.getenv("SECRET_KEY")
    alg = os.getenv("JWT_ALGORITHM", "HS256")
    if not secret:
        raise HTTPException(status_code=500, detail="JWT secret not configured")
    try:
        payload = jwt.decode(token, secret, algorithms=[alg])
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")
    tenant_id = payload.get("tenant_id")
    user_id = payload.get("sub")
    role = payload.get("role")
    exp = payload.get("exp")
    if not all([tenant_id, user_id, role, exp]):
        raise HTTPException(status_code=401, detail="Token missing required claims")
    return Claims(tenant_id=str(tenant_id), user_id=str(user_id), role=str(role), exp=int(exp))


def _db_conn(settings: Any):
    return psycopg2.connect(
        host=_pick(settings, "DB_HOST", "POSTGRES_HOST"),
        port=int(_pick(settings, "DB_PORT", "POSTGRES_PORT", default=5432)),
        dbname=_pick(settings, "DB_NAME", "POSTGRES_DB", default="postgres"),
        user=_pick(settings, "DB_USER", "POSTGRES_USER"),
        password=_pick(settings, "DB_PASSWORD", "POSTGRES_PASSWORD"),
        sslmode=_pick(settings, "DB_SSLMODE", "POSTGRES_SSLMODE", default="require"),
        connect_timeout=8,
    )


# ── request / response models ────────────────────────────────
class DiscoverRequest(BaseModel):
    topic: str = Field(..., min_length=3, max_length=300, description="Research topic to search for")
    max_urls: int = Field(6, ge=1, le=15, description="Max URLs to ingest (after trust filtering)")
    include_domains: Optional[List[str]] = Field(
        default=None,
        description="Extra domains to allow beyond the built-in trusted list",
    )
    search_depth: str = Field("advanced", description="Tavily search depth: basic | advanced")
    days: int = Field(365, ge=1, le=3650, description="Only return results from the last N days")


class DiscoveredURL(BaseModel):
    url: str
    title: str
    domain: str
    score: float
    trusted: bool


class DiscoverResponse(BaseModel):
    status: str
    kb_id: str
    topic: str
    discovered: int
    trusted_count: int
    queued_for_ingestion: int
    urls: List[DiscoveredURL]


# ── Tavily helper ────────────────────────────────────────────
_TAVILY_ENDPOINT = "https://api.tavily.com/search"


def _tavily_search(topic: str, max_results: int, search_depth: str, days: int) -> List[Dict[str, Any]]:
    api_key = os.getenv("TAVILY_API_KEY", "")
    if not api_key:
        raise HTTPException(status_code=500, detail="TAVILY_API_KEY not configured")

    payload = {
        "api_key": api_key,
        "query": topic,
        "search_depth": search_depth,
        "max_results": max_results,
        "days": days,
        "include_answer": False,
        "include_raw_content": False,
        "include_images": False,
    }

    try:
        resp = httpx.post(_TAVILY_ENDPOINT, json=payload, timeout=20)
        resp.raise_for_status()
        return resp.json().get("results", [])
    except httpx.HTTPStatusError as exc:
        raise HTTPException(status_code=502, detail=f"Tavily error {exc.response.status_code}: {exc.response.text[:200]}")
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Tavily request failed: {str(exc)[:200]}")


def _domain(url: str) -> str:
    try:
        host = urlparse(url).netloc.lower()
        return host.removeprefix("www.")
    except Exception:
        return ""


def _is_trusted(domain: str, extra: Optional[List[str]]) -> bool:
    if domain in _TRUSTED_DOMAINS:
        return True
    if extra:
        return any(domain == d.lower().removeprefix("www.") for d in extra)
    return False


# ── background ingestion ─────────────────────────────────────
def _background_ingest(kb_id: str, tenant_id: str, user_id: str, urls: List[str]) -> None:
    """Fire-and-forget: call the existing url_ingest logic for each URL."""
    try:
        from app.api.url_ingest import URLIngestRequest, _crawl_and_ingest, CrawlJob, _normalize_url  # type: ignore
        from urllib.parse import urlparse as _up

        for url in urls:
            try:
                norm = _normalize_url(url)
                if not norm:
                    continue
                req = URLIngestRequest(
                    url=norm,  # type: ignore[arg-type]
                    max_depth=0,       # single page only — Tavily already found the right article
                    max_pages=1,
                    auto_pipeline=True,
                    wait=False,
                )
                job = CrawlJob(
                    tenant_id=tenant_id,
                    user_id=user_id,
                    kb_id=kb_id,
                    job_id=str(uuid.uuid4()),
                    seed_url=norm,
                    seed_host=_up(norm).netloc.lower(),
                    scope="tenant_private",
                    config=req,
                )
                _crawl_and_ingest(job)
            except Exception:
                pass  # one URL failing shouldn't block others
    except Exception:
        pass


# ── endpoint ─────────────────────────────────────────────────
@router.post("/{kb_id}/discover", response_model=DiscoverResponse)
def discover_and_ingest(
    body: DiscoverRequest,
    background: BackgroundTasks,
    kb_id: str = Path(..., description="KB UUID"),
    claims: Claims = Depends(require_claims),
) -> DiscoverResponse:
    """
    Search Tavily for the topic, filter to trusted sources,
    and queue the top URLs for ingestion into the KB.
    """
    try:
        uuid.UUID(kb_id)
    except Exception:
        raise HTTPException(status_code=400, detail="kb_id must be a valid UUID")

    # Fetch more than needed so filtering leaves enough
    raw_results = _tavily_search(
        topic=body.topic,
        max_results=min(body.max_urls * 3, 30),
        search_depth=body.search_depth,
        days=body.days,
    )

    # Score, tag, deduplicate
    seen_urls: set[str] = set()
    all_urls: List[DiscoveredURL] = []
    for r in raw_results:
        url = (r.get("url") or "").strip()
        if not url or url in seen_urls:
            continue
        seen_urls.add(url)
        dom = _domain(url)
        trusted = _is_trusted(dom, body.include_domains)
        all_urls.append(DiscoveredURL(
            url=url,
            title=(r.get("title") or "")[:200],
            domain=dom,
            score=float(r.get("score", 0.0)),
            trusted=trusted,
        ))

    # Prefer trusted; fall back to highest-score untrusted if not enough
    trusted_urls = [u for u in all_urls if u.trusted]
    untrusted_urls = sorted([u for u in all_urls if not u.trusted], key=lambda x: -x.score)

    selected = (trusted_urls + untrusted_urls)[: body.max_urls]
    ingest_urls = [u.url for u in selected]

    # Queue ingestion in background
    if ingest_urls:
        background.add_task(
            _background_ingest,
            kb_id=kb_id,
            tenant_id=claims.tenant_id,
            user_id=claims.user_id,
            urls=ingest_urls,
        )

    return DiscoverResponse(
        status="ok",
        kb_id=kb_id,
        topic=body.topic,
        discovered=len(all_urls),
        trusted_count=len(trusted_urls),
        queued_for_ingestion=len(ingest_urls),
        urls=selected,
    )
