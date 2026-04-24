"""
Test hybrid retrieval for a given topic/KB — shows exactly what chunks the pipeline receives.
Usage: python scripts/test_retrieval.py
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from app.api.article_run import (
    _agent_retrieve_hybrid_sources,
    _agent_expand_queries,
    _gcs_client,
    get_settings,
    _pick,
)
from app.api.article_run import _db_conn

# ---- Config ----
TENANT_ID  = "5ebc3c3a-5e0e-42a5-a350-76f1b792ac15"
KB_ID      = "69898811-3114-4dce-bebb-a7d2bb205b3d"
TOPIC      = "How to Improve Sleep Quality: What the Science Actually Says"
KEYWORDS   = ["sleep quality", "sleep hygiene", "deep sleep", "sleep deprivation", "better sleep"]
TOP_K      = 20
TOP_K_PER  = 50
EMB_MODEL  = "models/gemini-embedding-001"
EMB_DIM    = 1536

settings   = get_settings()
api_key    = _pick(settings, "GEMINI_API_KEY", "GOOGLE_API_KEY")
openai_key = _pick(settings, "OPENAI_API_KEY")

print(f"Topic : {TOPIC}")
print(f"KB    : {KB_ID}")
print("-" * 60)

# Step 1: expand queries
from openai import OpenAI
client = OpenAI(api_key=openai_key)
model  = _pick(settings, "OPENAI_MODEL", default="gpt-4o-mini")

try:
    result, _ = _agent_expand_queries(
        client=client,
        model=model,
        title=TOPIC,
        keywords=KEYWORDS,
        user_intent={},
        temperature=0.3,
    )
    queries = result.get("queries") or []
    if not queries:
        raise ValueError("empty queries")
except Exception as e:
    print(f"Query expansion failed ({e}), using base queries")
    queries = [TOPIC] + KEYWORDS[:3]

print(f"\nExpanded queries ({len(queries)}):")
for q in queries:
    print(f"  - {q}")

# Step 2: retrieve
gcs  = _gcs_client(settings)
with _db_conn(settings) as conn:
    sources, meta = _agent_retrieve_hybrid_sources(
        conn, gcs,
        tenant_id=TENANT_ID,
        kb_id=KB_ID,
        queries=queries,
        embedding_api_key=api_key,
        embedding_model=EMB_MODEL,
        output_dimensionality=EMB_DIM,
        top_k_final=TOP_K,
        top_k_per_query=TOP_K_PER,
    )

print(f"\nRetrieval meta: {json.dumps(meta, indent=2)}")
print(f"\nRetrieved {len(sources)} chunks:\n")

unique_docs = set()
for i, s in enumerate(sources, 1):
    unique_docs.add(s["doc_id"])
    text_preview = (s.get("text") or "")[:300].replace("\n", " ")
    print(f"[{i:02d}] doc={s['doc_id'][:8]}... chunk={s['chunk_id'][:8]}...")
    print(f"     hybrid={s.get('hybrid_score',0):.4f}  sem={s.get('semantic_score',0):.4f}  bm25={s.get('bm25_score',0):.4f}  dist={s.get('distance',0):.4f}")
    print(f"     words={len((s.get('text') or '').split())}")
    print(f"     text : {text_preview}...")
    print()

print(f"Unique documents: {len(unique_docs)}")
total_words = sum(len((s.get('text') or '').split()) for s in sources)
print(f"Total words in retrieved chunks: {total_words}")
