# Sighnal — AI Content Generator Backend

**FastAPI** · **Supabase Postgres** · **Google Cloud Storage** · **OpenAI GPT-5.2** · **Gemini AI** · **ZeroGPT**

Multi-tenant backend that ingests documents/URLs → preprocesses into a knowledge base → generates ~2000-2500 word humanized blog articles → performs QC (readability + AI detection) → surgically fixes AI-detected sentences → stores everything in GCS.

---

## What Is Sighnal?

Sighnal is a backend-first, multi-tenant AI content system built for the full pipeline:

1. **Ingest** — upload PDFs, DOCX, or crawl URLs into a tenant Knowledge Base
2. **Preprocess** — clean, chunk, summarize, embed (pgvector)
3. **Retrieve** — hybrid vector + keyword + entity search
4. **Generate** — ~2000-2500 word blog articles via OpenAI GPT-5.2 (outline-first architecture)
5. **QC** — word count, Flesch Reading Ease > 70, FK grade, repetition, section count, FAQ check
6. **Detect** — ZeroGPT AI detection scoring (target < 20%)
7. **Humanize** — surgical zerogpt-fix loop: rephrase only AI-flagged sentences, auto-escalates to full rewrite if needed

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
         │  OpenAI API           │  GPT-5.2 (draft + QC-fix + humanize)
         │  Groq API (fallback)  │  Kimi K2 / Llama 3.3
         │  Gemini API           │  embeddings · summarization
         │  ZeroGPT API          │  AI detection scoring
         └───────────────────────┘
```

---

## Quick Start

```powershell
# Install dependencies
cd "d:\Hare Krishna_ai_blog"
pip install -r requirements.txt

# Start the server
uvicorn app.main:app --reload --port 8000
```

---

## PowerShell Testing Commands (Full Pipeline)

### Step 1 — Login

```powershell
$resp = Invoke-RestMethod -Uri "http://localhost:8000/api/v1/auth/login" `
  -Method POST -ContentType "application/json" `
  -Body '{"tenant_id":"5ebc3c3a-5e0e-42a5-a350-76f1b792ac15","role":"tenant_admin"}'
$TOKEN = $resp.access_token
$headers = @{ Authorization = "Bearer $TOKEN" }
Write-Host "TOKEN ready: $($TOKEN.Substring(0,20))..."
```

---

### Step 2 — Ingest Document

```powershell
# Upload a DOCX/PDF to the Knowledge Base
curl.exe -X POST "http://localhost:8000/api/v1/kb/69898811-3114-4dce-bebb-a7d2bb205b3d/ingest/file" `
  -H "Authorization: Bearer $TOKEN" `
  -F "file=@D:\path\to\document.docx"
# Save the doc_id from the response
```

---

### Step 3 — Preprocess → Chunk → Embed

```powershell
$DOC_ID = "your-doc-id-here"
$KB_ID  = "69898811-3114-4dce-bebb-a7d2bb205b3d"

# Preprocess
curl.exe -X POST "http://localhost:8000/api/v1/kb/$KB_ID/preprocess/$DOC_ID" `
  -H "Authorization: Bearer $TOKEN"

# Chunk
curl.exe -X POST "http://localhost:8000/api/v1/kb/$KB_ID/chunk/$DOC_ID" `
  -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" -d "{}"

# Embed
curl.exe -X POST "http://localhost:8000/api/v1/kb/$KB_ID/embed/$DOC_ID" `
  -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" -d "{}"
```

---

### Step 4 — Create Article Request

```powershell
# Write JSON body to file (avoids PowerShell escaping issues)
'{"kb_id":"69898811-3114-4dce-bebb-a7d2bb205b3d","title":"Your Article Title Here","keywords":["keyword1","keyword2","keyword3"]}' | Out-File -Encoding utf8 body.json

curl.exe -X POST "http://localhost:8000/api/v1/articles/requests" `
  -H "Authorization: Bearer $TOKEN" `
  -H "Content-Type: application/json" -d "@body.json"
# Save the request_id from the response
```

---

### Step 5 — Generate Draft (GPT-5.2, Outline-First)

```powershell
$REQUEST_ID = "your-request-id-here"

# This calls GPT-5.2 twice: outline (~900 tokens) then full draft (~6000 tokens)
curl.exe -X POST "http://localhost:8000/api/v1/articles/requests/$REQUEST_ID/run" `
  -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" -d "{}"
# Takes 30-60 seconds
```

---

### Step 6 — QC Check

```powershell
$qcResp = Invoke-RestMethod -Uri "http://localhost:8000/api/v1/articles/requests/$REQUEST_ID/qc" -Headers $headers
Write-Host "QC Pass: $($qcResp.qc_pass) | Words: $($qcResp.qc_metrics.word_count) | FRE: $($qcResp.qc_metrics.flesch_reading_ease) | FK: $($qcResp.qc_metrics.flesch_kincaid_grade) | FAQ: $($qcResp.qc_metrics.has_faq_section)"
```

---

### Step 7 — QC-Fix (if QC failed)

```powershell
$fixResp = Invoke-RestMethod -Uri "http://localhost:8000/api/v1/articles/requests/$REQUEST_ID/qc-fix" `
  -Method POST -Headers $headers -ContentType "application/json" -Body '{}'
Write-Host "QC Fix: $($fixResp.qc_pass) | Words: $($fixResp.final_word_count)"
```

---

### Step 8 — ZeroGPT AI Detection

```powershell
$zgResp = Invoke-RestMethod -Uri "http://localhost:8000/api/v1/articles/requests/$REQUEST_ID/zerogpt" -Headers $headers
Write-Host "ZeroGPT Score: $($zgResp.zerogpt_score)% | Pass: $($zgResp.zerogpt_pass)"
```

---

### Step 9 — ZeroGPT Fix (Surgical Humanization)

```powershell
# Only needed if ZeroGPT score >= 20%
$zgFixResp = Invoke-RestMethod -Uri "http://localhost:8000/api/v1/articles/requests/$REQUEST_ID/zerogpt-fix" `
  -Method POST -Headers $headers -ContentType "application/json" -Body '{}'
Write-Host "Score: $($zgFixResp.initial_score) -> $($zgFixResp.final_score) | Pass: $($zgFixResp.zerogpt_pass) | Attempts: $($zgFixResp.attempts_used)"
```

---

### Step 10 — Get Final Output

```powershell
$outputResp = Invoke-RestMethod -Uri "http://localhost:8000/api/v1/articles/requests/$REQUEST_ID/output" -Headers $headers
$outputResp | ConvertTo-Json -Depth 5
```

---

## Changes: v2 — GPT-5.2 Humanoid Upgrade (2026-03-04)

### What Changed

Upgraded from Groq (Kimi K2 / Llama 3.3) to **OpenAI GPT-5.2** with advanced anti-AI-detection prompt engineering.

### Key Improvements

| Metric | v1 (Groq) | v2 (GPT-5.2) |
|---|---|---|
| **ZeroGPT Score** | 77.5% (FAIL) | **15.1% (PASS)** |
| **Draft Quality** | Needs 3-5 QC-fix rounds | Passes QC on first draft |
| **ZeroGPT Fix** | Couldn't drop below 70% | **1 attempt** to pass |
| **Total Tokens** | ~35,000 (28 iterations) | **~26,914** (draft + 1 fix) |
| **Architecture** | Single-shot generation | **Outline-first** (2-call) |
| **Writing Style** | Generic AI tone | **Humanoid** (burstiness + perplexity) |

### Anti-AI-Detection Techniques

The prompts use three key techniques that AI detectors look for:

1. **Burstiness** — wildly varying sentence lengths (3 words then 22 words then 8 words). AI writes uniform-length sentences; humans don't.
2. **Perplexity** — unexpected word choices ("wrecked" not "damaged", "wild" not "surprising"). AI picks the safest word; humans pick weird ones.
3. **Personal Voice** — first-person opinions, mini-stories, casual fillers ("I think", "honestly", "in my experience"). AI never does this.

Plus: fragments, dashes, ellipses, rhetorical questions, imperfect grammar, casual transitions.

### Files Modified (v2)

| File | Changes |
|---|---|
| `requirements.txt` | Added `openai` package |
| `app/api/article_run.py` | OpenAI GPT-5.2 + outline-first 2-call architecture + humanoid SYSTEM_PROMPT + anti-AI few-shot examples |
| `app/api/article_qc.py` | Added FRE > 70 check + FAQ section check + updated thresholds (wc 1900-2600, FK 5-12) |
| `app/api/article_revise.py` | `_llm_json()` router (OpenAI preferred, Groq fallback) + humanized expand/simplify prompts |
| `app/api/article_zerogpt.py` | Threshold 10% → 20% |
| `app/api/article_zerogpt_fix.py` | Fixed `h` field bug + `_humanize_llm()` router + aggressive anti-detection prompts with rotating techniques + auto-escalate surgical → full rewrite + temperature 0.9 |
| `.env` | Added `OPENAI_API_KEY`, `OPENAI_MODEL`, `OPENAI_MAX_TOKENS` |

### Outline-First Architecture (Token Efficient)

```
Call 1: Outline (~900 tokens)          → structured plan
Call 2: Full Draft (~6000 tokens)      → guided by outline
                                        ↓
Total draft: ~7,000 tokens (vs ~12,000 single-shot)
Prompt caching: same SYSTEM_PROMPT = 50% savings on input tokens
```

### Surgical ZeroGPT Fix

```
ZeroGPT API returns AI sentences in 'h' field
                    ↓
Rephrase ONLY flagged sentences (not full rewrite)
                    ↓
~600 tokens per fix (vs ~7,000 for full rewrite = 12x cheaper)
                    ↓
If surgical stalls → auto-escalates to full rewrite
```

---

## Changes: v1 — Pipeline Fix (2026-02-27 to 2026-03-01)

### Problem Fixed

The QC-fix loop was running 28+ iterations producing robotic text like *"A retry helps systems. It fixes failures."*

### Root Causes

| # | Root Cause | Fix |
|---|---|---|
| 1 | `response_format: json_object` in Groq capped output at ~900 tokens | Removed JSON mode, plain text |
| 2 | `_post_simplify()` converting commas to periods | Removed comma splitting |
| 3 | Word replacement dict making bizarre substitutions | Reduced to 8 safe entries |
| 4 | "No sentence longer than 12 words" rule | Removed aggressive limits |
| 5 | No repetition/section checks in QC | Added checks |
| 6 | Duplicate router registration | Removed duplicate |

---

## Model Config (Current — v2)

| Step | Model | Provider | Notes |
|---|---|---|---|
| Draft generation | `gpt-5.2-2025-12-11` | OpenAI | Outline-first, 2-call, ~18K tokens |
| QC-fix rewrites | `gpt-5.2-2025-12-11` | OpenAI | Falls back to Groq if no API key |
| ZeroGPT humanize | `gpt-5.2-2025-12-11` | OpenAI | Surgical mode, temp 0.9 |
| Embeddings | `gemini-embedding-001` | Gemini | 1536 dimensions |
| Summarization | `gemini-2.5-flash` | Gemini | Standard pricing |
| AI Detection | ZeroGPT API | ZeroGPT | Threshold < 20% |

---

## QC Thresholds (Current — v2)

| Check | Min | Max | Method |
|---|---|---|---|
| Word count | 1900 | 2600 | Simple split |
| FK grade | 5.0 | 12.0 | textstat |
| Flesch Reading Ease | 70.0 | — | textstat |
| Repetition ratio | — | < 15% | Sentence Jaccard |
| Unique sections | >= 6 | — | Heading count |
| FAQ section | Required | — | Regex for `## FAQ` / `## Frequently Asked` |
| ZeroGPT AI score | — | < 20% | ZeroGPT API |

---

## Token Usage (Typical Run)

| Step | Prompt | Output | Total |
|---|---|---|---|
| Draft (outline + article) | ~14,500 | ~4,300 | **~18,800** |
| ZeroGPT Fix (1 attempt) | ~4,700 | ~3,400 | **~8,100** |
| **Total** | **~19,200** | **~7,700** | **~26,900** |

Cost: ~$0.17 per article (GPT-5.2 pricing)

---

## Progress Tracker

| Day | Deliverable | Status |
|---|---|---|
| Day 1 | Repo setup · GCS buckets · `/health` endpoint | Done |
| Day 2 | Cloud SQL/Supabase + pgvector · DB schemas · connectivity | Done |
| Day 3 | Supabase finance/logs tables · usage_events · job_events | Done |
| Day 4 | FastAPI JWT auth · tenant middleware · role checks · `/me` | Done |
| Day 5 | Tenant creation · cost calculator · pricebook structure | Done |
| Day 6 | KB CRUD (create/list/get/delete) · budget structure | Done |
| Day 7 | File ingestion → GCS raw → fingerprint → DB document row | Done |
| Day 8 | URL ingestion · BFS crawler · robots.txt · rate-limit | Done |
| Day 9 | Cache registry · global vs tenant-private scope · cache-hit reuse | Done |
| Day 10 | Ingestion job tracking · job_events · SSE status streaming | Done |
| Day 11 | Preprocessing · extraction + cleaning · clean_text in GCS | Done |
| Day 12 | Dynamic chunker · recursive + sliding overlap · chunk metadata | Done |
| Day 13 | Gemini summaries + entities · usage token cost logging | Done |
| Day 14 | Gemini embeddings · pgvector upsert · top_k retrieval | Done |
| Day 15 | Hybrid retrieval · vector + keyword + entity boost · retrieval cache | Done |
| Day 16 | Article request queue · locking · FastAPI create/list/get endpoints | Done |
| Day 17 | Draft generation · GPT-5.2 · outline-first · GCS storage | Done |
| Day 18 | Retry loop · attempt management · job state transitions | Done |
| Day 19 | Local QC · word count + FRE + FK + repetition + sections + FAQ | Done |
| Day 20 | ZeroGPT integration · score + surgical fix · humanization loop | Done |
| Day 21 | Full pipeline tested: ZeroGPT 15.1%, FRE 72, 2536 words | Done |
| Day 22 | Stale lock sweeper + recovery | Pending |
| Day 23 | Deployment hardening · services · restart policies · structured logs | Pending |
| Day 24 | Load testing · DB index tuning · retrieval performance | Pending |
| Day 25 | Final deployment + runbook · end-to-end demo · sign-off checklist | Pending |

**21 of 25 days complete.**

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

# OpenAI (GPT-5.2 — primary article LLM)
OPENAI_API_KEY=your-openai-key
OPENAI_MODEL=gpt-5.2-2025-12-11
OPENAI_MAX_TOKENS=16384

# Groq (fallback LLM)
GROQ_API_KEY=your-groq-key

# ZeroGPT
ZEROGPT_API_KEY=your-zerogpt-key
ZEROGPT_BASE_URL=https://api.zerogpt.com
```

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
