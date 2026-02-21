# Sighnal Backend — Document Intelligence API

**FastAPI** · **Supabase Postgres** · **Google Cloud Storage** · **Gemini AI**

Multi-tenant pipeline: ingest documents → clean → chunk → embed → search.

---

## System Flow (End-to-End)

```
[Login] → [Create Tenant] → [Create KB] → [Ingest File/URL]
             ↓                                      ↓
      [Check Job Status] ←──────────────── [job_events log]
             ↓
       [Preprocess] → [Chunk] → [Embed] → [Search / Hybrid-Search]
                          ↘ [Summarize]
```

---

## Quick Start

```bash
BASE="http://localhost:8000"
# Run server
uvicorn app.main:app --reload --port 8000
```

---

## STAGE 1 — Health & Readiness

### 1. Health Check
```bash
curl $BASE/api/v1/health
# OUTPUT: {"status":"ok","service":"sighnal-backend","env":"local"}
```

### 2. DB Health
```bash
curl $BASE/api/v1/db/health
# OUTPUT: {"status":"ok","db":"supabase-postgres","select_1":1}
```

### 3. GCS Readiness (confirms bucket + .keep objects)
```bash
curl $BASE/api/v1/ready
# OUTPUT: {"status":"ready","checks":{"gcs":{"ok":true,"detail":{...}}}}
```

---

## STAGE 2 — Auth (JWT)

### 4. Login — Get JWT Token
**Input:** `tenant_id` (UUID), optional `user_id`, `role`
```bash
curl -X POST $BASE/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"tenant_id":"<TENANT_UUID>","role":"tenant_admin"}'
# OUTPUT: {"access_token":"eyJ...","token_type":"bearer","expires_in":43200,
#          "tenant_id":"<UUID>","user_id":"<UUID>","role":"tenant_admin"}
```
> **Save** `access_token` as `TOKEN` for all subsequent calls.

### 5. Verify Token (/me)
```bash
curl $BASE/api/v1/me -H "Authorization: Bearer $TOKEN"
# OUTPUT: {"status":"ok","user":{...},"tenant":{...},"claims":{...}}
```

### 6. Admin Role Check
```bash
curl $BASE/api/v1/admin/ping -H "Authorization: Bearer $TOKEN"
# OUTPUT: {"status":"ok","msg":"admin access granted","tenant_id":"<UUID>"}
```

---

## STAGE 3 — Tenant & DB Setup

### 7. Create Tenant (required before KB)
```bash
curl -X POST $BASE/api/v1/db/tenants \
  -H "Content-Type: application/json" \
  -d '{"name":"MyCompany"}'
# OUTPUT: {"status":"ok","tenant":{"tenant_id":"<UUID>","name":"MyCompany","created_at":"..."}}
```
> **Save** `tenant_id` and use it in `auth/login` to get your TOKEN.

---

## STAGE 4 — Knowledge Base (KB) CRUD

### 8. Create KB
**Input:** `name`, optional `description`, `scope`
```bash
curl -X POST $BASE/api/v1/kb \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"name":"Product Docs","description":"Our product knowledge base","scope":"tenant_private"}'
# OUTPUT: {"status":"ok","kb":{"kb_id":"<UUID>","tenant_id":"<UUID>","name":"Product Docs",...}}
```
> **Save** `kb_id` as `KB_ID`.

### 9. List KBs
```bash
curl "$BASE/api/v1/kb?limit=20&offset=0" -H "Authorization: Bearer $TOKEN"
# OUTPUT: {"status":"ok","tenant_id":"<UUID>","count":1,"kbs":[...]}
```

### 10. Get KB by ID
```bash
curl $BASE/api/v1/kb/$KB_ID -H "Authorization: Bearer $TOKEN"
# OUTPUT: {"status":"ok","kb":{...}}
```

### 11. Delete KB
```bash
curl -X DELETE $BASE/api/v1/kb/$KB_ID -H "Authorization: Bearer $TOKEN"
# OUTPUT: {"status":"ok","deleted":true,"kb_id":"<UUID>"}
```

---

## STAGE 5 — Document Ingestion

### 12. Ingest File (PDF / DOCX / TXT / MD)
**Input:** `kb_id` (path), multipart `file`
```bash
curl -X POST $BASE/api/v1/kb/$KB_ID/ingest/file \
  -H "Authorization: Bearer $TOKEN" \
  -F "file=@/path/to/document.pdf"
# OUTPUT: {"status":"ok","kb_id":"<UUID>","doc_id":"<UUID>",
#          "ingestion_job_id":"<UUID>","fingerprint":"sha256...",
#          "gcs_raw_uri":"gs://...","gcs_extracted_uri":"gs://...",
#          "extraction_ok":true,"extraction_method":"pdf:pdfminer","extracted_chars":12345}
```
> **Save** `doc_id` as `DOC_ID` and `ingestion_job_id` as `JOB_ID`.

### 13. Ingest URL (Web Crawler)
**Input:** `kb_id` (path), JSON body with `url`, crawl config
```bash
curl -X POST $BASE/api/v1/kb/$KB_ID/ingest/url \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"url":"https://docs.example.com/","max_depth":3,"max_pages":50,"wait":false}'
# OUTPUT: {"status":"ok","kb_id":"<UUID>","ingestion_job_id":"<UUID>",
#          "seed_url":"https://docs.example.com/","queued_mode":"background","config":{...}}
```

---

## STAGE 6 — Job Status & Events

### 14. Check Ingest Job Status
```bash
curl "$BASE/api/v1/ingest/jobs/$JOB_ID?include_events=true" \
  -H "Authorization: Bearer $TOKEN"
# OUTPUT: {"status":"ok","job_id":"<UUID>","state":"done","progress":{"current":5,"total":5,"pct":100},
#          "artifacts":{"doc_id":"<UUID>","gcs_raw_uri":"gs://..."},"recent_events":[...]}
```

### 15. List KB Documents
```bash
curl "$BASE/api/v1/kb/$KB_ID/docs?limit=20" -H "Authorization: Bearer $TOKEN"
# OUTPUT: {"status":"ok","kb_id":"<UUID>","count":3,"docs":[...]}
```

### 16. Get Specific Document
```bash
curl $BASE/api/v1/kb/$KB_ID/docs/$DOC_ID -H "Authorization: Bearer $TOKEN"
# OUTPUT: {"status":"ok","doc":{all document metadata}}
```

---

## STAGE 7 — Preprocessing (Text Cleaning)

### 17. Preprocess Document (clean extracted text)
**Input:** `kb_id`, `doc_id` (path), optional JSON body
```bash
curl -X POST $BASE/api/v1/kb/$KB_ID/preprocess/$DOC_ID \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"remove_boilerplate":true,"standardize_bullets":true,"standardize_headings":true}'
# OUTPUT: {"status":"ok","kb_id":"<UUID>","doc_id":"<UUID>","preprocess_job_id":"<UUID>",
#          "preprocessing_version":"v1","clean_fingerprint":"sha256...",
#          "gcs_clean_uri":"gs://...","cleaned_chars":9800,"method":"clean:v1","stats":{...}}
```
> **Save** `preprocess_job_id` as `PREP_JOB_ID`.

### 18. Check Preprocess Job
```bash
curl "$BASE/api/v1/preprocess/jobs/$PREP_JOB_ID?include_events=true" \
  -H "Authorization: Bearer $TOKEN"
# OUTPUT: {"status":"ok","state":"done","progress":{"current":5,"total":5,"pct":100},"artifacts":{...}}
```

---

## STAGE 8 — Chunking

### 19. Chunk Document
**Input:** `kb_id`, `doc_id` (path), JSON body with chunk params
```bash
curl -X POST $BASE/api/v1/kb/$KB_ID/chunk/$DOC_ID \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"chunk_size_chars":2000,"overlap_chars":200,"max_chunks":5000,"prefer_clean":true}'
# OUTPUT: {"status":"ok","kb_id":"<UUID>","doc_id":"<UUID>","chunk_job_id":"<UUID>",
#          "input_kind":"clean_v1","chunk_count":47,
#          "gcs_chunks_uri":"gs://.../chunks_v1/<hash>.jsonl",
#          "gcs_manifest_uri":"gs://.../chunks_v1/<hash>.manifest.json","stats":{...}}
```
> **Save** `gcs_chunks_uri` as `CHUNKS_URI`.

---

## STAGE 9 — Summarize (Gemini/Groq)

### 20. Summarize Document (parallel to embed)
**Input:** `kb_id`, `doc_id` (path), optional JSON body
```bash
curl -X POST $BASE/api/v1/kb/$KB_ID/summarize/$DOC_ID \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"primary_model":"gemini-2.5-flash","fallback_model":"gemini-2.5-pro","max_chunks":200}'
# OUTPUT: {"status":"ok","kb_id":"<UUID>","doc_id":"<UUID>","summarize_job_id":"<UUID>",
#          "chunk_count":47,"gcs_summary_uri":"gs://...","summary_fingerprint":"sha256...",
#          "usage":{"prompt_tokens":4500,"output_tokens":800,"total_tokens":5300},"meta":{...}}
```

---

## STAGE 10 — Embeddings

### 21. Embed Chunks (Gemini text-embedding-001, dim=1536)
**Input:** `kb_id`, `doc_id` (path), optional `gcs_chunks_uri`
```bash
curl -X POST $BASE/api/v1/kb/$KB_ID/embed/$DOC_ID \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"embedding_model":"models/gemini-embedding-001","output_dimensionality":1536,"batch_size":32}'
# OUTPUT: {"status":"ok","kb_id":"<UUID>","doc_id":"<UUID>","embed_job_id":"<UUID>",
#          "chunk_count":47,"embedded_count":47,"embedding_model":"models/gemini-embedding-001",
#          "output_dimensionality":1536,"meta":{"cache_hit":false}}
```

---

## STAGE 11 — Search

### 22. Vector Search (cosine similarity via pgvector)
**Input:** `kb_id` (path), JSON body with `query`
```bash
curl -X POST $BASE/api/v1/kb/$KB_ID/search \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"query":"What are the main features?","doc_id":"<DOC_ID>","top_k":5}'
# OUTPUT: {"status":"ok","kb_id":"<UUID>","query":"...","top_k":5,
#          "results":[{"chunk_id":"<UUID>","score":0.92,"distance":0.08,"text":"..."},...]}
```

### 23. Hybrid Search (Vector + Keyword + Entity + Diversification)
**Input:** `kb_id` (path), JSON body with `query`, weights, diversify options
```bash
curl -X POST $BASE/api/v1/kb/$KB_ID/hybrid-search \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"query":"What are the main features?","doc_id":"<DOC_ID>","top_k":5,
       "alpha_vector":0.75,"beta_keyword":0.25,"diversify":true,"use_cache":true}'
# OUTPUT: {"status":"ok","kb_id":"<UUID>","results":[
#          {"chunk_id":"<UUID>","score":0.89,"vector_score":0.92,
#           "keyword_score":0.67,"entity_boost":0.03,"text":"..."},...],
#          "meta":{"weights":{"alpha_vector":0.75,"beta_keyword":0.25},...}}
```

---

## Database Tables Created

| Table | Purpose |
|---|---|
| `public.tenants_fin` | Multi-tenant registry |
| `public.knowledge_bases` | KB metadata per tenant |
| `public.documents` | Ingested document records |
| `public.job_events` | Audit log for all pipeline steps |
| `public.preprocess_jobs` | Preprocessing job tracking |
| `public.preprocess_outputs` | Clean text artifacts |
| `public.chunks` | Chunk metadata + GCS refs |
| `public.chunk_embeddings` | pgvector embeddings (dim=1536) |
| `public.url_pages` | Web crawl page hierarchy |
| `public.retrieval_cache` | Hybrid search cache (TTL) |
| `public.cache_registry` | File dedup cache across docs |

---

## Progress Status (Completed Stages)

| Stage | Description | Status |
|---|---|---|
| Day 1 | Health + GCS readiness checks | ✅ Done |
| Day 2 | DB health (Supabase Postgres) | ✅ Done |
| Day 3 | JWT Auth (HS256, login/logout/me) | ✅ Done |
| Day 5 | Tenant creation in DB | ✅ Done |
| Day 6 | KB CRUD (create/list/get/delete) | ✅ Done |
| Day 7 | File ingestion + SHA256 cache registry | ✅ Done |
| Day 8 | URL crawl ingestion (BFS, robots, rate-limit) | ✅ Done |
| Day 10 | Ingest job status tracker (from job_events) | ✅ Done |
| Day 11 | Preprocessing: text cleaning + normalization | ✅ Done |
| Day 12 | Dynamic chunking (heading-aware + overlap) | ✅ Done |
| Day 13 | Summarization (Gemini/Groq, per-chunk + doc TOC) | ✅ Done |
| Day 14 | Gemini embeddings + pgvector search | ✅ Done |
| Day 15 | Hybrid search (vector + keyword + entity + cache) | ✅ Done |

**Next:** RAG answer generation, article writer, multi-doc search.

---

## Required .env Variables

```env
GCP_PROJECT_ID=your-gcp-project
GCS_BUCKET_NAME=your-bucket
DB_HOST=your-supabase-host
DB_NAME=postgres
DB_USER=postgres
DB_PASSWORD=your-password
DB_SSLMODE=require
JWT_SECRET_KEY=your-strong-secret-32chars+
GEMINI_API_KEY=your-gemini-key
GROQ_API_KEY=your-groq-key   # optional fallback LLM
```
