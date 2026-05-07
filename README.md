# Sighnal — AI Content Generator Backend

**FastAPI** · **Supabase Postgres** · **Google Cloud Storage** · **OpenAI GPT-5.2** · **Gemini AI** · **ZeroGPT** · **LangGraph**

Multi-tenant backend that ingests documents/URLs → builds a knowledge base → generates ~2000-word humanized blog articles through a 6-agent pipeline → enforces readability + AI-detection QC → stores output in GCS → streams clean Word (.docx) downloads.

---

## What Is Sighnal?

Sighnal is a backend-first, multi-tenant AI content system. Every article runs through five isolated layers:

| Layer | Name | What it does |
|---|---|---|
| **A** | Retrieval | Crawl URLs, extract grounded facts, hybrid RAG retrieval |
| **B** | Brand Voice | Per-tenant persona, tone, audience injected into every writing agent |
| **C** | Article Engine | 6-agent multi-agent pipeline: plan → write → humanize → assemble |
| **D** | QC & Safety | FK / FRE / ZeroGPT gates with automated surgical fix passes |
| **E** | Output | GCS storage → Word document (.docx) streaming download |

---

## Architecture

```
CLIENT (HTTP)
     │
     ▼
┌─────────────────────────────────────────────────────────────┐
│  API Layer  (FastAPI)                                       │
│                                                             │
│  POST /api/v1/articles/run         →  article_run.py        │
│  GET  /api/v1/pipeline/{id}        →  article_run.py        │
│  GET  /api/v1/articles/{id}/download → article_download.py  │
│  GET  /api/v1/config               →  brand_config.py       │
│  PUT  /api/v1/config               →  brand_config.py       │
└──────────────┬──────────────────────────┬───────────────────┘
               │                          │
               │ Load BrandContext         │ GET / PUT brand config
               ▼                          ▼
     ┌─────────────────┐       ┌──────────────────────────┐
     │  Layer B        │       │  public.tenant_brand_    │
     │  PromptEngine   │◄─────►│  configs  (Supabase PG)  │
     │                 │       │                          │
     │  BrandContext:  │       │  persona, tone,          │
     │  • persona      │       │  audience, pain_points,  │
     │  • tone         │       │  reading_level, POV,     │
     │  • audience     │       │  forbidden_phrases,      │
     │  • pain_points  │       │  compliance_note         │
     │  • reading_level│       └──────────────────────────┘
     │  • POV          │
     │  • forbidden    │
     │  • compliance   │
     └────────┬────────┘
              │ brand_context passed into pipeline
              ▼
┌─────────────────────────────────────────────────────────────┐
│  LangGraph Outer Shell  (article_graph.py)                  │
│                                                             │
│  crawl → create_request → [BLOG PIPELINE]                   │
│       → qc → qc_fix? → zerogpt → zerogpt_fix? → finalize   │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  Blog Pipeline  (pipeline_runner.py)                        │
│                                                             │
│  PHASE 1A — Parallel                                        │
│  ┌─────────────────┐   ┌──────────────────────────┐        │
│  │  TopicAnalyst   │   │  EvidenceLocker           │        │
│  │  themes, angles │   │  grounded facts from URLs │        │
│  │  audience raw   │   │  dedup + quality filter   │        │
│  └────────┬────────┘   └─────────────┬─────────────┘        │
│           └─────────────┬────────────┘                      │
│                         ▼                                   │
│  PHASE 1B                                                   │
│  ┌──────────────────────────────────────────────────┐       │
│  │  SectionPlanner                                   │       │
│  │  Plans 6-8 sections with roles:                  │       │
│  │    hook / body / transition / deepdive / faq     │       │
│  │  ← brand_context.audience_context() injected     │       │
│  └──────────────────────────────────────────────────┘       │
│                         │                                   │
│                         ▼                                   │
│  PHASE 2 — Sequential (one section at a time)               │
│  ┌──────────────────────────────────────────────────┐       │
│  │  SectionWriter  (× N sections)                   │       │
│  │  Narrative continuity via prev_section_text      │       │
│  │  system prompt = CORE_LAWS + brand voice_block() │       │
│  │  Local QC gate per section (FK, AI patterns)     │       │
│  └──────────────┬───────────────────────────────────┘       │
│                 │ QC fail?                                   │
│                 ▼                                           │
│  ┌──────────────────────────────────────────────────┐       │
│  │  MiniHumanizer  (conditional)                    │       │
│  │  Strips AI patterns, passive voice, filler       │       │
│  └──────────────────────────────────────────────────┘       │
│                         │                                   │
│                         ▼                                   │
│  PHASE 3 — Assembly (deterministic + 1 optional LLM expand) │
│  ┌──────────────────────────────────────────────────┐       │
│  │  Assembler                                        │       │
│  │  1. Join all sections                             │       │
│  │  2. Expand 2 thin sections if < 1900 words       │       │
│  │  3. Insert above-fold structure:                  │       │
│  │     <!-- META: [hook first sentence] -->         │       │
│  │     [hook paragraphs]                            │       │
│  │     > Key Takeaways  (from section headings)     │       │
│  │     In this article: TOC  (anchor links)         │       │
│  │     [body sections...]                           │       │
│  │     <!-- AUTHOR BIO: [...] -->                   │       │
│  └──────────────────────────────────────────────────┘       │
└──────────────────────┬──────────────────────────────────────┘
                       │ draft_markdown
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  LangGraph Post-Processing  (Layer D)                       │
│                                                             │
│  QC Gate      FK 5-12, FRE 50-75, word count 1900-3000     │
│      └─► QC Fix        conditional LLM rewrite             │
│                                                             │
│  ZeroGPT Gate  AI score < 20%                              │
│      └─► ZeroGPT Fix   surgical sentence-level humanization │
│                                                             │
│  Finalize     re-inject AUTHOR BIO if stripped by fixes     │
│               save JSON artifact to GCS                    │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
         GCS: gs://{bucket}/articles/{request_id}.json
         { draft_markdown, title, word_count, section_meta, ... }
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  Download  (article_download.py + docx_writer.py)           │
│                                                             │
│  Fetch GCS JSON → extract draft_markdown + title            │
│  → markdown_to_docx():                                      │
│     Mojibake cleanup (latin-1, cp1252 artifacts)            │
│     Inline markdown → Word runs (bold, italic, code)        │
│     Heading styles (H1/H2/H3), list styles, HR, blockquote │
│     HTML comments (<!-- -->) stripped silently              │
│  → Stream .docx attachment                                  │
└─────────────────────────────────────────────────────────────┘
```

---

## Quick Start

```powershell
cd "d:\Hare Krishna_ai_blog"
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

---

## API Reference

### Auth

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/v1/auth/register` | Register new user |
| `POST` | `/api/v1/auth/login` | Login — returns JWT |
| `POST` | `/api/v1/auth/forgot-password` | Send OTP email |
| `POST` | `/api/v1/auth/reset-password` | Reset with OTP |
| `GET` | `/api/v1/auth/me` | Current user info |

### Brand Config (Layer B)

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/v1/config` | Get tenant brand config |
| `PUT` | `/api/v1/config` | Save / update brand config |

### Knowledge Base

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/v1/kb` | Create knowledge base |
| `GET` | `/api/v1/kb` | List knowledge bases |
| `GET` | `/api/v1/kb/{kb_id}` | Get KB details |
| `POST` | `/api/v1/kb/{kb_id}/ingest/file` | Upload PDF/DOCX |
| `POST` | `/api/v1/kb/{kb_id}/ingest/url` | Ingest from URL |
| `POST` | `/api/v1/kb/{kb_id}/preprocess/{doc_id}` | Extract + clean text |
| `POST` | `/api/v1/kb/{kb_id}/chunk/{doc_id}` | Chunk text |
| `POST` | `/api/v1/kb/{kb_id}/embed/{doc_id}` | Embed chunks (pgvector) |

### Article Pipeline

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/v1/articles/run` | Start pipeline (async) |
| `GET` | `/api/v1/pipeline/{request_id}` | Poll pipeline status |
| `GET` | `/api/v1/articles/requests/{id}/download` | Download Word (.docx) |

---

## PowerShell Testing Walkthrough

### 1 — Login

```powershell
$resp = Invoke-RestMethod -Uri "http://localhost:8000/api/v1/auth/login" `
  -Method POST -ContentType "application/json" `
  -Body '{"email":"you@example.com","password":"yourpassword"}'
$TOKEN = $resp.access_token
$headers = @{ Authorization = "Bearer $TOKEN" }
```

### 2 — Set Brand Config (Layer B)

```powershell
Invoke-RestMethod -Method PUT -Uri "http://localhost:8000/api/v1/config" `
  -Headers $headers -ContentType "application/json" `
  -Body '{
    "persona": "A knowledgeable but approachable health writer",
    "tone_adjectives": ["clear", "evidence-aware", "non-preachy"],
    "audience_primary": "health-conscious adults aged 28-45",
    "audience_pain_points": ["poor sleep quality", "low energy", "brain fog"],
    "reading_level": "grade 10-12",
    "compliance_note": "Always include a consult-a-professional nudge."
  }'
```

### 3 — Run Pipeline

```powershell
'{"title":"Why You Wake Up Tired Even After 8 Hours","keywords":["sleep quality","sleep stages","deep sleep"],"urls":["https://example.com/sleep-study"]}' `
  | Out-File -Encoding utf8 body_run.json

$run = Invoke-RestMethod -Method POST -Uri "http://localhost:8000/api/v1/articles/run" `
  -Headers $headers -ContentType "application/json" `
  -InFile body_run.json
$REQUEST_ID = $run.request_id
Write-Host "Request ID: $REQUEST_ID"
```

### 4 — Poll Status

```powershell
do {
  $status = Invoke-RestMethod -Uri "http://localhost:8000/api/v1/pipeline/$REQUEST_ID" -Headers $headers
  Write-Host "$([datetime]::Now.ToString('HH:mm:ss'))  status=$($status.status)"
  if ($status.status -notin @("running","pending")) { break }
  Start-Sleep 10
} while ($true)
```

### 5 — Download Word Document

```powershell
$outFile = "outputs\article_$($REQUEST_ID.Substring(0,8)).docx"
Invoke-RestMethod `
  -Uri "http://localhost:8000/api/v1/articles/requests/$REQUEST_ID/download" `
  -Headers $headers -OutFile $outFile
Write-Host "Saved to $outFile"
```

---

## Brand Voice Layer (Layer B)

Every tenant can set a brand config that gets injected into every writing agent for every run.

```json
{
  "persona": "A knowledgeable but approachable health writer",
  "tone_adjectives": ["clear", "evidence-aware", "non-preachy"],
  "audience_primary": "health-conscious adults aged 28-45",
  "audience_pain_points": ["poor sleep quality", "low energy", "brain fog"],
  "reading_level": "grade 10-12",
  "preferred_pov": "second-person",
  "forbidden_phrases": ["it's important to note", "in conclusion"],
  "compliance_note": "Always include a consult-a-professional nudge."
}
```

**Where it gets injected:**

| Agent | How |
|---|---|
| SectionPlanner | `audience_context()` replaces raw TopicAnalyst audience in the user prompt |
| SectionWriter | `voice_block()` appended to system prompt after Core Writing Laws |
| Assembler | Title used for META comment generation |

**What the Assembler adds (deterministic — no LLM):**

```markdown
<!-- META: [first sentence of hook, 150-160 chars] -->

[hook paragraphs]

> **Key takeaways**
> - Section heading 1
> - Section heading 2
> - Section heading 3

**In this article:**
- [Section 1](#anchor)
- [Section 2](#anchor)
...

[body content]

<!-- AUTHOR BIO: [Author name, credentials — fill before publishing] -->
```

---

## QC Thresholds

| Check | Gate | Method |
|---|---|---|
| Word count | 1900 – 3000 | Token split |
| Flesch-Kincaid grade | 5.0 – 12.0 | textstat |
| Flesch Reading Ease | 50 – 75 | textstat |
| ZeroGPT AI score | < 20% | ZeroGPT API |
| Section count | ≥ 6 | Heading regex |
| FAQ section | Required | Heading regex |

---

## Model Config

| Step | Model | Provider |
|---|---|---|
| Draft + QC-fix | `gpt-5.2-2025-12-11` | OpenAI |
| ZeroGPT humanization | `gpt-5.2-2025-12-11` | OpenAI |
| Section expand (assembler) | `gpt-5.2-2025-12-11` | OpenAI |
| Embeddings | `gemini-embedding-001` | Gemini |
| Summarization | `gemini-2.5-flash` | Gemini |
| AI Detection | ZeroGPT API | ZeroGPT |

---

## Database Tables

| Table | Purpose |
|---|---|
| `public.tenants_fin` | Multi-tenant registry |
| `public.tenant_brand_configs` | Per-tenant brand voice config (Layer B) |
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

## Required `.env` Variables

```env
# App
APP_NAME=Sighnal
API_HOST=127.0.0.1
API_PORT=8000
LOG_LEVEL=INFO

# Auth
JWT_SECRET_KEY=your-strong-random-secret-64chars
JWT_ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=720
CORS_ALLOW_ORIGINS=http://localhost:3000

# GCS
GCP_PROJECT_ID=your-gcp-project-id
GCS_BUCKET_NAME=your-bucket-name
GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json
GCS_RAW_PREFIX=raw
GCS_PROCESSED_PREFIX=processed
GCS_ARTICLES_PREFIX=articles

# Supabase Postgres (direct connection — port 5432)
DB_HOST=db.your-project-ref.supabase.co
DB_PORT=5432
DB_NAME=postgres
DB_USER=postgres
DB_PASSWORD=your-db-password
DB_SSLMODE=require

# OpenAI (primary article LLM)
OPENAI_API_KEY=your-openai-key
OPENAI_MODEL=gpt-5.2-2025-12-11
OPENAI_MAX_TOKENS=16384

# Gemini
GEMINI_API_KEY=your-gemini-key
GEMINI_MODEL_DRAFT=gemini-2.5-flash
GEMINI_EMBEDDING_MODEL=gemini-embedding-001

# Groq (fallback LLM)
GROQ_API_KEY=your-groq-key
GROQ_MODEL=llama-3.3-70b-versatile

# Tavily (topic → URL discovery)
TAVILY_API_KEY=your-tavily-key

# ZeroGPT
ZEROGPT_API_KEY=your-zerogpt-key
ZEROGPT_BASE_URL=https://api.zerogpt.com
ZEROGPT_ENDPOINT_PATH=/api/detect/detectText

# QC thresholds
BLOG_WORDCOUNT_MIN=1950
BLOG_WORDCOUNT_MAX=2050
READABILITY_MIN_GRADE=7.0
READABILITY_MAX_GRADE=9.0

# SMTP (Gmail primary / Brevo fallback)
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your-gmail@gmail.com
SMTP_PASS=your-app-password
FRONTEND_URL=http://localhost:3000
```

---

## Key Files

```
app/
├── main.py                                  FastAPI app + router registration
├── api/
│   ├── article_run.py                       POST /articles/run  (pipeline trigger)
│   ├── article_download.py                  GET  /articles/{id}/download
│   ├── brand_config.py                      GET/PUT /config  (Layer B API)
│   └── ...
├── services/
│   ├── article_graph.py                     LangGraph outer shell
│   ├── docx_writer.py                       Shared markdown → Word converter
│   └── blog_pipeline/
│       ├── pipeline_runner.py               Multi-agent orchestrator
│       ├── prompt_engine.py                 Layer B — BrandContext + DB helpers
│       ├── agent_topic_analyst.py           Phase 1A — topic analysis
│       ├── agent_evidence_locker.py         Phase 1A — fact extraction
│       ├── agent_section_planner.py         Phase 1B — section planning
│       ├── agent_section_writer.py          Phase 2  — section writing
│       ├── agent_mini_humanize.py           Phase 2  — conditional humanizer
│       ├── assembler.py                     Phase 3  — join + structure
│       └── gates_local_qc.py               Per-section QC gate
```

---

## Changelog

### v4 — Brand Voice Layer (2026-05-07)
- **Layer B (PromptEngine)** — per-tenant `BrandContext` loaded from DB, injected into SectionPlanner + SectionWriter + Assembler
- **Brand Config API** — `GET/PUT /api/v1/config` → `public.tenant_brand_configs`
- **Above-fold structure** — Assembler now adds META comment, Key Takeaways block, TOC, and AUTHOR BIO placeholder deterministically (no LLM)
- **AUTHOR BIO safety net** — finalize node re-injects if stripped by downstream QC/ZeroGPT fix passes
- **Download endpoint fixes** — SQL `topic` → `title` column; dual GCS JSON shape support (blog vs legacy)
- **docx_writer** — strips `<!-- ... -->` HTML comments so metadata never leaks into Word output

### v3 — Pipeline Hardening (2026-04-26)
- Watchdog timer (15 min max per run)
- NameError fix in finalize_node
- DDL guard on GET poll endpoint
- Concurrent pipeline limit (max 2 per tenant)
- Multi-URL support (up to 8 URLs per run)
- Word document download endpoint + shared `docx_writer` service

### v2 — Multi-Agent Rewrite (2026-04-23)
- Replaced single-shot GPT prompt with 6-agent LangGraph pipeline
- Added EvidenceLocker, SectionPlanner, MiniHumanizer, Assembler agents
- Per-section local QC gate (FK + AI pattern detection)
- 5 pipeline bugs fixed: encoding, critique truncation, grounding ratio, retrieval noise, content dedup

### v1 — GPT-5.2 Humanoid Draft (2026-03-04)
- Upgraded from Groq to OpenAI GPT-5.2
- Outline-first 2-call architecture
- Burstiness + perplexity anti-AI-detection techniques
- ZeroGPT 77.5% → 15.1% on first run
