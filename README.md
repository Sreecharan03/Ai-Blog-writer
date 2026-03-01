# Sighnal — AI Content Generator Backend

**FastAPI** · **Supabase Postgres** · **Google Cloud Storage** · **Gemini AI** · **Groq (Kimi K2 + Llama 3.3)**

Multi-tenant backend that ingests documents/URLs → preprocesses into a knowledge base → generates ~2000-word SEO blog articles → performs QC (readability + AI detection) → stores everything in GCS.

---

## What Is Sighnal?

Sighnal is a backend-first, multi-tenant AI content system built for the full pipeline:

1. **Ingest** — upload PDFs, DOCX, or crawl URLs into a tenant Knowledge Base
2. **Preprocess** — clean, chunk, summarize, embed (pgvector)
3. **Retrieve** — hybrid vector + keyword + entity search
4. **Generate** — ~2000-word blog articles via Groq LLM (Kimi K2)
5. **QC** — word count, Flesch-Kincaid readability, repetition ratio, section count
6. **Detect** — ZeroGPT AI detection scoring
7. **Revise** — minimal QC-fix loop (expand/trim/simplify) to hit thresholds

All artifacts stored in **Google Cloud Storage (GCS)**. Finance and audit logs in **Supabase**.

---

## Architecture

```
[Client UI] → [FastAPI Backend]
                     ↓
         ┌───────────────────────┐
         │  GCS Content Lake     │  raw/ · processed/ · articles/
         └───────────────────────┘
         ┌───────────────────────┐
         │  Supabase Postgres    │  tenants · KBs · chunks · embeddings
         │  + pgvector           │  job_events · usage_events
         └───────────────────────┘
         ┌───────────────────────┐
         │  Groq API             │  Kimi K2 (draft) · Llama 3.3 (QC-fix)
         │  Gemini API           │  embeddings · summarization
         │  ZeroGPT API          │  AI detection scoring
         └───────────────────────┘
```

---

## Quick Start

```powershell
# Start the server
cd "d:\Hare Krishna_ai_blog"
.venv\Scripts\uvicorn app.main:app --reload --port 8000
```

---

## PowerShell Testing Commands (Full Pipeline)

### Step 1 — Health Checks

```powershell
$BASE = "http://localhost:8000"

# Liveness
Invoke-RestMethod -Uri "$BASE/api/v1/health"

# DB health
Invoke-RestMethod -Uri "$BASE/api/v1/db/health"

# GCS readiness
Invoke-RestMethod -Uri "$BASE/api/v1/ready"
```

---

### Step 2 — Auth (JWT Login)

> **Note:** Dev login uses `tenant_id` directly (no email/password). Query DB for your tenant_id first.

```powershell
# Login with tenant_id
$body = @{
    tenant_id = "5ebc3c3a-5e0e-42a5-a350-76f1b792ac15"
    role      = "tenant_admin"
} | ConvertTo-Json

$loginResp = Invoke-RestMethod -Uri "$BASE/api/v1/auth/login" `
    -Method POST -Body $body -ContentType "application/json"

$TOKEN = $loginResp.access_token
Write-Host "TOKEN: $TOKEN"

# Verify token
Invoke-RestMethod -Uri "$BASE/api/v1/me" `
    -Headers @{Authorization = "Bearer $TOKEN"}
```

---

### Step 3 — Create Tenant (first time only)

```powershell
$body = @{ name = "MyCompany" } | ConvertTo-Json
$t = Invoke-RestMethod -Uri "$BASE/api/v1/db/tenants" `
    -Method POST -Body $body -ContentType "application/json"
$TENANT_ID = $t.tenant.tenant_id
Write-Host "TENANT_ID: $TENANT_ID"
```

---

### Step 4 — Knowledge Base

```powershell
# Create KB
$body = @{
    name        = "Product Docs"
    description = "My knowledge base"
    scope       = "tenant_private"
} | ConvertTo-Json
$kb = Invoke-RestMethod -Uri "$BASE/api/v1/kb" `
    -Method POST -Body $body -ContentType "application/json" `
    -Headers @{Authorization = "Bearer $TOKEN"}
$KB_ID = $kb.kb.kb_id
Write-Host "KB_ID: $KB_ID"

# List KBs
Invoke-RestMethod -Uri "$BASE/api/v1/kb?limit=20" `
    -Headers @{Authorization = "Bearer $TOKEN"}
```

---

### Step 5 — Ingest Document

```powershell
# Ingest a PDF
$ingest = Invoke-RestMethod -Uri "$BASE/api/v1/kb/$KB_ID/ingest/file" `
    -Method POST `
    -Headers @{Authorization = "Bearer $TOKEN"} `
    -Form @{ file = Get-Item "C:\path\to\document.pdf" }

$DOC_ID = $ingest.doc_id
$JOB_ID = $ingest.ingestion_job_id
Write-Host "DOC_ID: $DOC_ID"

# Check job status
Invoke-RestMethod -Uri "$BASE/api/v1/ingest/jobs/$JOB_ID`?include_events=true" `
    -Headers @{Authorization = "Bearer $TOKEN"}
```

---

### Step 6 — Preprocess → Chunk → Embed

```powershell
# Preprocess
$body = @{
    remove_boilerplate   = $true
    standardize_bullets  = $true
    standardize_headings = $true
} | ConvertTo-Json
$prep = Invoke-RestMethod -Uri "$BASE/api/v1/kb/$KB_ID/preprocess/$DOC_ID" `
    -Method POST -Body $body -ContentType "application/json" `
    -Headers @{Authorization = "Bearer $TOKEN"}
$PREP_JOB_ID = $prep.preprocess_job_id

# Check preprocess job
Invoke-RestMethod -Uri "$BASE/api/v1/preprocess/jobs/$PREP_JOB_ID`?include_events=true" `
    -Headers @{Authorization = "Bearer $TOKEN"}

# Chunk
$body = @{
    chunk_size_chars = 2000
    overlap_chars    = 200
    max_chunks       = 5000
    prefer_clean     = $true
} | ConvertTo-Json
$chunk = Invoke-RestMethod -Uri "$BASE/api/v1/kb/$KB_ID/chunk/$DOC_ID" `
    -Method POST -Body $body -ContentType "application/json" `
    -Headers @{Authorization = "Bearer $TOKEN"}

# Embed
$body = @{
    embedding_model       = "models/gemini-embedding-001"
    output_dimensionality = 1536
    batch_size            = 32
} | ConvertTo-Json
Invoke-RestMethod -Uri "$BASE/api/v1/kb/$KB_ID/embed/$DOC_ID" `
    -Method POST -Body $body -ContentType "application/json" `
    -Headers @{Authorization = "Bearer $TOKEN"}
```

---

### Step 7 — Search (Hybrid)

```powershell
$body = @{
    query          = "What are the main features?"
    doc_id         = $DOC_ID
    top_k          = 5
    alpha_vector   = 0.75
    beta_keyword   = 0.25
    diversify      = $true
    use_cache      = $true
} | ConvertTo-Json
Invoke-RestMethod -Uri "$BASE/api/v1/kb/$KB_ID/hybrid-search" `
    -Method POST -Body $body -ContentType "application/json" `
    -Headers @{Authorization = "Bearer $TOKEN"}
```

---

### Step 8 — Article Pipeline (Days 16–21)

```powershell
# 1. Create article request
$body = @{
    kb_id    = $KB_ID
    title    = "Understanding Retry Mechanisms"
    keywords = @("retry", "resilience", "fault tolerance")
} | ConvertTo-Json
$reqs = Invoke-RestMethod -Uri "$BASE/api/v1/articles/requests" `
    -Method POST -Body $body -ContentType "application/json" `
    -Headers @{Authorization = "Bearer $TOKEN"}
$REQUEST_ID = $reqs.request.request_id
Write-Host "REQUEST_ID: $REQUEST_ID"

# 2. List requests (to find existing IDs)
$list = Invoke-RestMethod -Uri "$BASE/api/v1/articles/requests" `
    -Headers @{Authorization = "Bearer $TOKEN"}
$REQUEST_ID = $list.requests[0].request_id

# 3. Run draft generation (~936 words, 8 sections, FK ~3.8)
$run = Invoke-RestMethod -Uri "$BASE/api/v1/articles/requests/$REQUEST_ID/run" `
    -Method POST `
    -Headers @{Authorization = "Bearer $TOKEN"}
$run | ConvertTo-Json -Depth 5

# 4. Check QC (word count + FK + repetition + sections)
$qc = Invoke-RestMethod -Uri "$BASE/api/v1/articles/requests/$REQUEST_ID/qc" `
    -Headers @{Authorization = "Bearer $TOKEN"}
$qc | ConvertTo-Json -Depth 5

# 5. Run QC-fix if needed (expands to 1900-2100 words, ~8000 tokens)
$fix = Invoke-RestMethod -Uri "$BASE/api/v1/articles/requests/$REQUEST_ID/qc-fix" `
    -Method POST `
    -Headers @{Authorization = "Bearer $TOKEN"}
$fix | ConvertTo-Json -Depth 5

# 6. ZeroGPT AI detection
$zgpt = Invoke-RestMethod -Uri "$BASE/api/v1/articles/requests/$REQUEST_ID/zerogpt" `
    -Method POST `
    -Headers @{Authorization = "Bearer $TOKEN"}
$zgpt | ConvertTo-Json -Depth 5

# 7. Get final article (signed GCS URL)
$out = Invoke-RestMethod -Uri "$BASE/api/v1/articles/requests/$REQUEST_ID/output" `
    -Headers @{Authorization = "Bearer $TOKEN"}
$out | ConvertTo-Json -Depth 5

# 8. Save final blog markdown to file
$gcsUrl = $out.signed_url   # or adjust to actual field name
$blog = Invoke-WebRequest -Uri $gcsUrl
$blog.Content | Out-File -FilePath "final_blog.md" -Encoding UTF8
Write-Host "Blog saved to final_blog.md"
```

---

## Changes Made (Sessions 2026-02-27 to 2026-03-01)

### Problem That Was Fixed

The QC-fix loop was running 28+ times and still producing robotic, staccato text like *"A retry helps systems. It fixes failures."* — passing ZeroGPT (0% AI) but completely unreadable as a blog.

### Root Causes Found

| # | Root Cause | Impact |
|---|---|---|
| 1 | `response_format: json_object` in Groq API | Capped all models at ~900-1100 output tokens regardless of max_tokens |
| 2 | `_post_simplify()` converting commas to periods | "retry, fix, recover" → "retry. fix. recover." — staccato text |
| 3 | Word replacement dict: "information"→"info", "understand"→"know" | Bizarre robotic English |
| 4 | Aggressive simplify rules: "no sentence longer than 12 words" | Destroyed article structure |
| 5 | Draft prompt: "expert technical writer" + strict JSON schema | Wrong audience, model parroted sources |
| 6 | No repetition/section checks in QC | 28-iteration loops passing broken content |
| 7 | Duplicate router in `main.py` | Double-registered article routes |

### Files Modified

#### `app/api/article_run.py`
- **Removed** `response_format: json_object` from Groq API body (root cause #1 — JSON mode caps output)
- **Changed** system prompt to: *"Write complete markdown article, at least 1900 words, do NOT stop early"*
- **Changed** prompt from JSON schema to plain markdown with per-section word targets:
  ```
  ## Introduction (150-180 words)
  ## What Is [Topic] (250-280 words)
  ## Why It Matters (250-280 words)
  ## How It Works (350-400 words)
  ## Real-World Examples (300-350 words)
  ## Common Mistakes (200-250 words)
  ## Quick FAQ (200-230 words)
  ## Key Takeaways (150-180 words)
  TOTAL: 1900-2100 words
  ```
- **Changed** parsing to wrap plain text response into `{"title": ..., "draft_markdown": content}`
- **Increased** `max_output_tokens` default: 4096 → 8192
- **Changed** role to: *"You are a friendly blog writer who explains topics clearly for everyday readers"*

#### `app/api/article_revise.py`
- **Removed** comma-to-period conversion in `_split_long_sentences()` (root cause #2)
- **Reduced** `_SIMPLE_REPL` word replacement dict from 46 → 8 safe entries only
- **Removed** "generally", "mostly" from filler list (these carry meaning)
- **Removed** "Rewrite every sentence from scratch" rule from aggressive simplify
- **Removed** "No sentence longer than 12 words" / "Avoid words longer than 8 letters" rules
- **Removed** 3rd aggressive fallback rewrite (was making 3 parallel Groq calls)
- **Changed** `wc_max`: 2050 → 2100
- **Added** stall break: `if stall_count >= 5: break` to prevent infinite loops

#### `app/api/article_qc.py`
- **Added** `repetition_ratio` check — flags if >15% of sentences are near-duplicates
- **Added** `unique_sections` check — counts `#`/`##`/`###` headings, requires ≥4
- **Changed** QC pass/fail to include: `repetition_ratio < 0.15` AND `unique_sections >= 4`
- **Changed** `wc_max`: 2050 → 2100

#### `app/api/article_state.py`
- **Added** `skip_if_qc_pass` guard — if current draft already passes QC, skip heavy simplify

#### `app/main.py`
- **Removed** duplicate `app.include_router(article_run_router)` (was registered twice)

### Results After Fixes

| Metric | Before | After |
|---|---|---|
| QC-fix iterations | 28 (never converged) | 3-5 (converges cleanly) |
| QC-fix token cost | ~66,219 tokens | ~8,129 tokens (8x cheaper) |
| Final word count | 650-780 (never hitting 1900) | 2016 words ✅ |
| FK grade | N/A (broken) | 8.88 ✅ |
| Draft structure | Robotic / staccato | 8 sections, 0% repetition |
| Draft model | Llama 3.3 70B | Kimi K2 (moonshotai/kimi-k2-instruct-0905) |

### Model Config (Current Best)

| Step | Model | Provider | Cost |
|---|---|---|---|
| Draft generation | `moonshotai/kimi-k2-instruct-0905` | Groq | $1.00/M in · $3.00/M out |
| QC-fix rewrites | `llama-3.3-70b-versatile` | Groq | $0.59/M in · $0.79/M out |
| Embeddings | `gemini-embedding-001` | Gemini | standard pricing |
| Summarization | `gemini-2.5-flash` | Gemini | standard pricing |

---

## Progress Tracker

| Day | Deliverable | Status |
|---|---|---|
| Day 1 | Repo setup · GCS buckets · `/health` endpoint | ✅ Done |
| Day 2 | Cloud SQL/Supabase + pgvector · DB schemas · connectivity | ✅ Done |
| Day 3 | Supabase finance/logs tables · usage_events · job_events | ✅ Done |
| Day 4 | FastAPI JWT auth · tenant middleware · role checks · `/me` | ✅ Done |
| Day 5 | Tenant creation · cost calculator · pricebook structure | ✅ Done |
| Day 6 | KB CRUD (create/list/get/delete) · budget structure | ✅ Done |
| Day 7 | File ingestion → GCS raw → fingerprint → DB document row | ✅ Done |
| Day 8 | URL ingestion · BFS crawler · robots.txt · rate-limit | ✅ Done |
| Day 9 | Cache registry · global vs tenant-private scope · cache-hit reuse | ✅ Done |
| Day 10 | Ingestion job tracking · job_events · SSE status streaming | ✅ Done |
| Day 11 | Preprocessing · extraction + cleaning · clean_text in GCS | ✅ Done |
| Day 12 | Dynamic chunker · recursive + sliding overlap · chunk metadata | ✅ Done |
| Day 13 | Gemini summaries + entities · usage token cost logging | ✅ Done |
| Day 14 | Gemini embeddings · pgvector upsert · top_k retrieval | ✅ Done |
| Day 15 | Hybrid retrieval · vector + keyword + entity boost · retrieval cache | ✅ Done |
| Day 16 | Article request queue · locking · FastAPI create/list/get endpoints | ✅ Done |
| Day 17 | Draft generation · Kimi K2 via Groq · plain text mode · GCS storage | ✅ Done |
| Day 18 | Retry loop · attempt management · job state transitions | ✅ Done |
| Day 19 | Local QC · word count + FK + repetition + section checks | ✅ Done |
| Day 20 | ZeroGPT integration · score + span highlights · usage logged | ✅ Done |
| Day 21 | Minimal revision loop · QC-fix (expand/trim/simplify) · 8x cheaper | ✅ Done |
| Day 22 | Stale lock sweeper + recovery · kill worker test · job recovery after TTL | ⏳ Pending |
| Day 23 | Deployment hardening · services · restart policies · structured logs | ⏳ Pending |
| Day 24 | Load testing · DB index tuning · retrieval performance · cache validation | ⏳ Pending |
| Day 25 | Final deployment + runbook · end-to-end demo · sign-off checklist | ⏳ Pending |

**21 of 25 days complete.**

---

## Database Tables

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
| `public.retrieval_cache` | Hybrid search cache (TTL=24h) |
| `public.cache_registry` | File dedup cache across docs |
| `public.article_requests` | Article generation queue |

---

## GCS Folder Layout

```
gs://ai_blog_02/
  raw/<fingerprint>/original.<ext>
  url_snapshots/<url_hash>/<fetched_at>.json
  processed/<fingerprint>/clean_text.json
  processed/<fingerprint>/chunks.json
  processed/<fingerprint>/summaries.json
  articles/<tenant_id>/<request_id>/draft_v1.md
  articles/<tenant_id>/<request_id>/final.md
  articles/<tenant_id>/<request_id>/qc_report.json
```

---

## Required `.env` Variables

```env
# App
APP_NAME=Sighnal
API_HOST=127.0.0.1
API_PORT=8000

# Auth
JWT_SECRET_KEY=your-strong-random-secret-64chars
JWT_ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=720

# GCS
GCP_PROJECT_ID=your-gcp-project-id
GCS_BUCKET_NAME=your-bucket-name
GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json

# Supabase Postgres
DB_HOST=your-supabase-pooler-host
DB_PORT=6543
DB_NAME=postgres
DB_USER=postgres.your-project-ref
DB_PASSWORD=your-db-password
DB_SSLMODE=require

# Gemini
GEMINI_API_KEY=your-gemini-key
GEMINI_MODEL_DRAFT=gemini-2.5-flash
GEMINI_EMBEDDING_MODEL=gemini-embedding-001

# Groq (LLM for article generation + QC-fix)
GROQ_API_KEY=your-groq-key
GROQ_MODEL=moonshotai/kimi-k2-instruct-0905

# ZeroGPT
ZEROGPT_API_KEY=your-zerogpt-key
ZEROGPT_BASE_URL=https://api.zerogpt.com

# QC Thresholds
BLOG_WORDCOUNT_MIN=1900
BLOG_WORDCOUNT_MAX=2100
READABILITY_MIN_GRADE=7.0
READABILITY_MAX_GRADE=12.0
```

---

## QC Thresholds (Current)

| Check | Min | Max | Method |
|---|---|---|---|
| Word count | 1900 | 2100 | Simple split |
| FK grade | 7.0 | 12.0 | textstat |
| Repetition ratio | — | < 15% | Sentence Jaccard |
| Unique sections | ≥ 4 | — | `#`/`##`/`###` heading count |
