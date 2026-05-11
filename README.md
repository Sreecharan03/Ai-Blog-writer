# Sighnal — AI Blog Writer

Multi-tenant backend that takes a topic + source URLs, runs them through a 6-agent pipeline, and produces a humanized ~2500-word blog article as a Word (.docx) download.

**Stack:** FastAPI · Supabase Postgres · Google Cloud Storage · OpenAI · LangGraph

---

## How it works

1. You create a Knowledge Base and give it source URLs
2. You start a pipeline with a title + KB ID
3. Six agents run in sequence: analyse topic → extract facts → plan sections → write sections → humanize → assemble
4. QC gates check readability and AI-detection score
5. Download the finished article as a `.docx` file

---

## Setup

**1. Install dependencies**

```powershell
pip install -r requirements.txt
```

**2. Create a `.env` file** (see [Environment Variables](#environment-variables) below)

**3. Start the server**

```powershell
uvicorn app.main:app --reload --port 8000
```

---

## Running the Pipeline (PowerShell)

### Step 1 — Login

```powershell
$r = Invoke-RestMethod -Method POST -Uri "http://localhost:8000/api/v1/auth/login" `
  -ContentType "application/json" `
  -Body '{"email":"you@example.com","password":"yourpassword"}'
$TOKEN = $r.access_token
```

### Step 2 — Create a Knowledge Base

```powershell
$kb = Invoke-RestMethod -Method POST -Uri "http://localhost:8000/api/v1/kb" `
  -Headers @{Authorization="Bearer $TOKEN"} -ContentType "application/json" `
  -Body '{"name":"My KB","description":"Finance articles"}'
$kbId = $kb.kb.kb_id
```

### Step 3 — Start the Pipeline

```powershell
$body = "{`"kb_id`":`"$kbId`",`"title`":`"Your Article Title`",`"urls`":[`"https://source-url.com/article`"],`"length_target`":2600}"
$req = Invoke-RestMethod -Method POST -Uri "http://localhost:8000/api/v1/articles/pipeline" `
  -Headers @{Authorization="Bearer $TOKEN"} -ContentType "application/json" -Body $body
$pipelineId = $req.pipeline_id
Write-Host "Pipeline ID: $pipelineId"
```

### Step 4 — Poll Until Done

```powershell
$status = Invoke-RestMethod -Method GET `
  -Uri "http://localhost:8000/api/v1/articles/pipeline/$pipelineId" `
  -Headers @{Authorization="Bearer $TOKEN"}
Write-Host "Status:" $status.pipeline_status "| Step:" $status.current_step
```

Keep running until `pipeline_status` is `completed` or `completed_with_warnings`.

### Step 5 — Download the Word Document

```powershell
$requestId = $status.request_id
Invoke-RestMethod -Method GET `
  -Uri "http://localhost:8000/api/v1/articles/requests/$requestId/download" `
  -Headers @{Authorization="Bearer $TOKEN"} `
  -OutFile "article.docx"
```

---

## API Endpoints

| Method | Path | What it does |
|---|---|---|
| `POST` | `/api/v1/auth/login` | Login — returns JWT |
| `POST` | `/api/v1/auth/register` | Register new user |
| `POST` | `/api/v1/kb` | Create knowledge base |
| `GET` | `/api/v1/kb` | List knowledge bases |
| `POST` | `/api/v1/kb/{kb_id}/ingest/url` | Add source URL to KB |
| `POST` | `/api/v1/articles/pipeline` | Start article pipeline |
| `GET` | `/api/v1/articles/pipeline/{pipeline_id}` | Poll pipeline status |
| `GET` | `/api/v1/articles/requests/{request_id}/download` | Download `.docx` |
| `GET/PUT` | `/api/v1/config` | Brand voice config |

---

## Environment Variables

```env
# Auth
JWT_SECRET_KEY=your-secret-key
JWT_ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=720

# Database (Supabase)
DB_HOST=db.your-project.supabase.co
DB_PORT=5432
DB_NAME=postgres
DB_USER=postgres
DB_PASSWORD=your-db-password
DB_SSLMODE=require

# Google Cloud Storage
GCP_PROJECT_ID=your-gcp-project
GCS_BUCKET_NAME=your-bucket
GOOGLE_APPLICATION_CREDENTIALS=path/to/service-account.json

# OpenAI
OPENAI_API_KEY=your-openai-key
OPENAI_MODEL=gpt-4o

# Gemini
GEMINI_API_KEY=your-gemini-key
GEMINI_EMBEDDING_MODEL=gemini-embedding-001

# ZeroGPT (AI detection)
ZEROGPT_API_KEY=your-zerogpt-key

# SMTP (for password reset emails)
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your@gmail.com
SMTP_PASS=your-app-password
```

---

## Key Files

```
app/
├── main.py                              FastAPI app entry point
├── api/
│   ├── article_download.py              Download endpoint
│   └── brand_config.py                  Brand voice API
└── services/
    ├── docx_writer.py                   Markdown → Word converter
    └── blog_pipeline/
        ├── pipeline_runner.py           Agent orchestrator
        ├── agent_topic_analyst.py       Analyse topic + angles
        ├── agent_evidence_locker.py     Extract grounded facts
        ├── agent_section_planner.py     Plan article sections
        ├── agent_section_writer.py      Write each section
        ├── agent_mini_humanize.py       Strip AI patterns
        ├── assembler.py                 Join + add TOC/structure
        └── gates_local_qc.py           Readability QC gate

scripts/
    ├── reset_pipelines.py               Force-fail stuck pipelines
    ├── regen_docx.py                    Regenerate .docx from GCS
    └── check_status.py                  Quick pipeline status check
```

---

## Pipeline Status Values

| Status | Meaning |
|---|---|
| `pending` | Queued, not started |
| `running` | In progress |
| `completed` | Done, passed all QC gates |
| `completed_with_warnings` | Done, QC warning (article still usable) |
| `failed` | Error — check `error_detail` |
| `timed_out` | Exceeded 15-minute limit |

> Max 2 active pipelines per tenant at a time. Use `scripts/reset_pipelines.py` to clear stuck ones.
