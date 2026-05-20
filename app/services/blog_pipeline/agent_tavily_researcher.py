"""
Tavily Researcher Agent.
Runs before EvidenceLocker to inject real, current domain facts.
Generates targeted search queries from topic analysis, fetches raw content,
and returns results as source dicts compatible with EvidenceLocker input.
"""
from __future__ import annotations
import logging
import os
from typing import Any, Dict, List

logger = logging.getLogger("tavily_researcher")

MAX_RESULTS_PER_QUERY = 4
MAX_QUERIES = 3
MAX_CONTENT_CHARS = 8000  # per result, before passing to EvidenceLocker


def _build_queries(title: str, keywords: List[str], analysis: Dict[str, Any]) -> List[str]:
    """
    Generate 3 targeted search queries from the topic.
    Uses key_terms and content_type from TopicAnalyst output.
    """
    key_terms = analysis.get("key_terms") or []
    content_type = analysis.get("content_type", "explainer")
    primary_angle = analysis.get("primary_angle", "")

    kw_str = " ".join(keywords[:4]) if keywords else ""
    terms_str = " ".join(key_terms[:3]) if key_terms else ""

    queries = []

    # Query 1: core topic with domain context
    q1 = title
    if kw_str:
        q1 = f"{title} {kw_str}"
    queries.append(q1)

    # Query 2: key terms + definitions/explained
    if key_terms:
        queries.append(f"{' '.join(key_terms[:4])} explained India 2025")
    elif keywords:
        queries.append(f"{' '.join(keywords[:3])} explained 2025")

    # Query 3: angle-specific — tax/rules/comparison/calculation based on content type
    if content_type == "comparison":
        queries.append(f"{title} comparison benefits drawbacks India")
    elif content_type == "how_to":
        queries.append(f"how to {kw_str or title} step by step India 2025")
    elif primary_angle:
        queries.append(f"{primary_angle} {kw_str}")
    else:
        queries.append(f"{title} rules tax calculation India 2025")

    return queries[:MAX_QUERIES]


def research_with_tavily(
    title: str,
    keywords: List[str],
    analysis: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    Run Tavily searches and return results as source dicts.

    Each returned dict has:
      text     — raw content (truncated to MAX_CONTENT_CHARS)
      doc_id   — "tavily_{i}"
      chunk_id — "tavily_{i}_0"
      url      — source URL
      title    — page title

    Returns empty list on any failure — never blocks the pipeline.
    """
    api_key = os.environ.get("TAVILY_API_KEY", "")
    if not api_key:
        logger.warning("TAVILY_API_KEY not set — skipping Tavily research")
        return []

    try:
        from tavily import TavilyClient
    except ImportError:
        logger.warning("tavily-python not installed — skipping Tavily research")
        return []

    client = TavilyClient(api_key=api_key)
    queries = _build_queries(title, keywords, analysis)
    sources: List[Dict[str, Any]] = []
    seen_urls: set = set()

    for q in queries:
        try:
            result = client.search(
                q,
                max_results=MAX_RESULTS_PER_QUERY,
                include_raw_content=True,
            )
            for r in result.get("results", []):
                url = r.get("url", "")
                if url in seen_urls:
                    continue
                seen_urls.add(url)

                # Prefer raw_content (full page), fall back to content snippet
                raw = r.get("raw_content") or r.get("content") or ""
                text = raw[:MAX_CONTENT_CHARS].strip()
                if not text:
                    continue

                idx = len(sources) + 1
                sources.append({
                    "text": text,
                    "doc_id": f"tavily_{idx}",
                    "chunk_id": f"tavily_{idx}_0",
                    "url": url,
                    "title": r.get("title", ""),
                    "source": "tavily",
                })
        except Exception as e:
            logger.warning("Tavily query failed [%s]: %s", q[:60], e)
            continue

    logger.info("Tavily research: %d queries → %d sources", len(queries), len(sources))
    return sources
