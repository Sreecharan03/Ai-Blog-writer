# streamlit_app.py
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, List

import requests
import streamlit as st

# Optional .env loading (matches your pattern)
try:
    from dotenv import load_dotenv
    from pathlib import Path

    repo_root = Path(__file__).resolve().parent
    env_path = repo_root / ".env"
    if env_path.exists():
        load_dotenv(dotenv_path=env_path, override=True)
except Exception:
    pass

# GCS download for summary display
try:
    from google.cloud import storage
except Exception:
    storage = None


# -----------------------------
# Helpers
# -----------------------------
def api_url(base: str, path: str) -> str:
    return base.rstrip("/") + path


def safe_json(resp: requests.Response) -> Dict[str, Any]:
    try:
        return resp.json()
    except Exception:
        return {"_raw": resp.text, "_status_code": resp.status_code}


def auth_headers(token: str) -> Dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def parse_gs_uri(gs_uri: str) -> Tuple[str, str]:
    if not gs_uri.startswith("gs://"):
        raise ValueError(f"Not a gs:// uri: {gs_uri}")
    rest = gs_uri[len("gs://") :]
    parts = rest.split("/", 1)
    bucket = parts[0]
    obj = parts[1] if len(parts) > 1 else ""
    return bucket, obj


def gcs_download_json(gs_uri: str, project_id: Optional[str] = None) -> Dict[str, Any]:
    if storage is None:
        raise RuntimeError("google-cloud-storage is not installed. pip install google-cloud-storage")
    bucket, obj = parse_gs_uri(gs_uri)
    client = storage.Client(project=project_id) if project_id else storage.Client()
    blob = client.bucket(bucket).blob(obj)
    if not blob.exists():
        raise FileNotFoundError(f"GCS object not found: {gs_uri}")
    b = blob.download_as_bytes()
    return json.loads(b.decode("utf-8", errors="replace"))


def pretty(obj: Any):
    st.json(obj, expanded=False)


def step_header(title: str, ok: Optional[bool] = None):
    if ok is None:
        st.subheader(title)
    else:
        st.subheader(f"{'✅' if ok else '❌'} {title}")


def require_config(base: str, tenant: str, kb: str, token: str):
    if not base:
        st.error("Missing BASE URL")
        st.stop()
    if not tenant:
        st.error("Missing TENANT UUID")
        st.stop()
    if not kb:
        st.error("Missing KB UUID")
        st.stop()
    if not token:
        st.error("You must Login first to get a token.")
        st.stop()


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Sighnal Backend – Realtime Pipeline Tester", layout="wide")

st.title("Sighnal Backend – Realtime Pipeline Tester")
st.caption("Upload → Ingest → Preprocess → Chunk → Summarize → Embed → Search (+ logs)")

# Persist sidebar values in session_state so they survive rerenders
if "token" not in st.session_state:
    st.session_state["token"] = ""
if "tenant_id" not in st.session_state:
    st.session_state["tenant_id"] = os.getenv("SIGHNAL_TENANT_ID", "")
if "kb_id_cfg" not in st.session_state:
    st.session_state["kb_id_cfg"] = os.getenv("SIGHNAL_KB_ID", "")

# Sidebar config
with st.sidebar:
    st.header("Connection")
    base = st.text_input("BASE URL", value=os.getenv("SIGHNAL_BASE_URL", "http://127.0.0.1:8000"))
    tenant_id = st.text_input("TENANT UUID", key="tenant_id")
    kb_id = st.text_input("KB UUID", key="kb_id_cfg")
    role = st.selectbox("Role", ["tenant_admin", "tenant_user"], index=0)

    st.divider()
    st.header("Token")
    st.text_input("JWT Token (auto after login)", key="token", type="password")

    st.divider()
    st.header("GCS (for summary display)")
    gcp_project_id = st.text_input("GCP_PROJECT_ID (optional)", value=os.getenv("GCP_PROJECT_ID", ""))

    st.divider()
    st.header("Docs / Tips")
    st.write("- Make sure FastAPI is running on BASE URL.")
    st.write("- You already validated Day 14 search works.")
    st.write("- Summary view downloads JSON from gcs_summary_uri via google-cloud-storage.")


tab_pipeline, tab_search, tab_logs, tab_artifacts = st.tabs(["Pipeline", "Search", "Logs", "Artifacts"])

# -----------------------------
# TAB: Pipeline
# -----------------------------
with tab_pipeline:
    st.markdown("### 1) Health + Login")

    c1, c2 = st.columns(2)
    with c1:
        if st.button("Check /health and /ready", use_container_width=True):
            try:
                h = requests.get(api_url(base, "/api/v1/health"), timeout=15)
                r = requests.get(api_url(base, "/api/v1/ready"), timeout=15)
                st.session_state["health"] = safe_json(h)
                st.session_state["ready"] = safe_json(r)
            except Exception as e:
                st.error(f"Health check failed: {e}")

        if "health" in st.session_state:
            step_header("/health response", ok=True)
            pretty(st.session_state["health"])
        if "ready" in st.session_state:
            step_header("/ready response", ok=True)
            pretty(st.session_state["ready"])

    with c2:
        if st.button("Login → Get Token", use_container_width=True):
            try:
                body = {"tenant_id": tenant_id, "role": role}
                resp = requests.post(
                    api_url(base, "/api/v1/auth/login"),
                    json=body,
                    headers={"Content-Type": "application/json"},
                    timeout=20,
                )
                data = safe_json(resp)
                if resp.status_code != 200:
                    st.error(f"Login failed ({resp.status_code}): {data}")
                else:
                    st.session_state["token"] = data.get("access_token", "")
                    # Auto-populate tenant_id from JWT payload
                    try:
                        import base64 as _b64, json as _json
                        _payload = st.session_state["token"].split(".")[1]
                        _payload += "=" * (-len(_payload) % 4)
                        _claims = _json.loads(_b64.urlsafe_b64decode(_payload))
                        if _claims.get("tenant_id") and not st.session_state["tenant_id"]:
                            st.session_state["tenant_id"] = _claims["tenant_id"]
                    except Exception:
                        pass
                    st.success("Token stored in sidebar.")
                    st.session_state["login_resp"] = data
            except Exception as e:
                st.error(f"Login error: {e}")

        if "login_resp" in st.session_state:
            step_header("Login response", ok=True)
            pretty(st.session_state["login_resp"])

    st.divider()
    st.markdown("### 2) Upload file and run pipeline steps")

    uploaded = st.file_uploader("Upload a file to ingest", type=None)
    doc_id_manual = st.text_input("DOC UUID (optional – if you want to run steps on existing doc)", value=st.session_state.get("doc_id", ""))

    # Buttons row
    b1, b2, b3, b4, b5, b6 = st.columns(6)
    run_all = b1.button("Run ALL (2→7)", use_container_width=True)
    run_ingest = b2.button("Ingest", use_container_width=True)
    run_pre = b3.button("Preprocess", use_container_width=True)
    run_chunk = b4.button("Chunk", use_container_width=True)
    run_sum = b5.button("Summarize", use_container_width=True)
    run_embed = b6.button("Embed", use_container_width=True)

    require_config(base, tenant_id, kb_id, st.session_state.get("token", ""))

    token = st.session_state["token"]

    # Helper: get active doc id
    def current_doc_id() -> str:
        if doc_id_manual.strip():
            return doc_id_manual.strip()
        return st.session_state.get("doc_id", "")

    # ---- Ingest ----
    def do_ingest():
        if uploaded is None:
            st.error("Please upload a file first.")
            return
        with st.spinner("Ingesting file..."):
            files = {"file": (uploaded.name, uploaded.getvalue())}
            resp = requests.post(
                api_url(base, f"/api/v1/kb/{kb_id}/ingest/file"),
                headers=auth_headers(token),
                files=files,
                timeout=180,
            )
            data = safe_json(resp)
            st.session_state["ingest_resp"] = data
            if resp.status_code != 200:
                st.error(f"Ingest failed ({resp.status_code}): {data}")
                return
            # store doc_id
            doc_id = data.get("doc_id") or data.get("doc", {}).get("doc_id")
            if doc_id:
                st.session_state["doc_id"] = doc_id
            st.success(f"Ingest OK. doc_id={st.session_state.get('doc_id')}")

    # ---- Preprocess ----
    def do_preprocess():
        doc_id = current_doc_id()
        if not doc_id:
            st.error("No doc_id found. Run Ingest first or paste DOC UUID.")
            return
        with st.spinner("Preprocessing..."):
            resp = requests.post(
                api_url(base, f"/api/v1/kb/{kb_id}/preprocess/{doc_id}"),
                headers={**auth_headers(token), "Content-Type": "application/json"},
                json={},
                timeout=240,
            )
            data = safe_json(resp)
            st.session_state["preprocess_resp"] = data
            if resp.status_code != 200:
                st.error(f"Preprocess failed ({resp.status_code}): {data}")
                return
            st.success("Preprocess OK.")

    # ---- Chunk ----
    def do_chunk():
        doc_id = current_doc_id()
        if not doc_id:
            st.error("No doc_id found. Run Ingest first or paste DOC UUID.")
            return
        with st.spinner("Chunking..."):
            resp = requests.post(
                api_url(base, f"/api/v1/kb/{kb_id}/chunk/{doc_id}"),
                headers={**auth_headers(token), "Content-Type": "application/json"},
                json={},
                timeout=240,
            )
            data = safe_json(resp)
            st.session_state["chunk_resp"] = data
            if resp.status_code != 200:
                st.error(f"Chunk failed ({resp.status_code}): {data}")
                return
            st.success("Chunk OK.")

    # ---- Summarize ----
    def do_summarize():
        doc_id = current_doc_id()
        if not doc_id:
            st.error("No doc_id found. Run Ingest first or paste DOC UUID.")
            return
        with st.spinner("Summarizing... (may take time)"):
            resp = requests.post(
                api_url(base, f"/api/v1/kb/{kb_id}/summarize/{doc_id}"),
                headers={**auth_headers(token), "Content-Type": "application/json"},
                json={},
                timeout=600,
            )
            data = safe_json(resp)
            st.session_state["summarize_resp"] = data
            if resp.status_code != 200:
                st.error(f"Summarize failed ({resp.status_code}): {data}")
                return
            st.success("Summarize OK.")

    # ---- Embed ----
    def do_embed():
        doc_id = current_doc_id()
        if not doc_id:
            st.error("No doc_id found. Run Ingest first or paste DOC UUID.")
            return
        with st.spinner("Embedding chunks..."):
            payload = {"batch_size": 16, "max_chunks": 500, "output_dimensionality": 1536}
            resp = requests.post(
                api_url(base, f"/api/v1/kb/{kb_id}/embed/{doc_id}"),
                headers={**auth_headers(token), "Content-Type": "application/json"},
                json=payload,
                timeout=600,
            )
            data = safe_json(resp)
            st.session_state["embed_resp"] = data
            if resp.status_code != 200:
                st.error(f"Embed failed ({resp.status_code}): {data}")
                return
            st.success(f"Embed OK. embedded_count={data.get('embedded_count')}")

    # Run button logic
    if run_ingest:
        do_ingest()
    if run_pre:
        do_preprocess()
    if run_chunk:
        do_chunk()
    if run_sum:
        do_summarize()
    if run_embed:
        do_embed()

    if run_all:
        # run in correct sequence
        do_ingest()
        do_preprocess()
        do_chunk()
        do_summarize()
        do_embed()

    st.divider()
    st.markdown("### Step outputs (JSON)")

    exp1 = st.expander("Ingest response", expanded=False)
    with exp1:
        pretty(st.session_state.get("ingest_resp", {}))

    exp2 = st.expander("Preprocess response", expanded=False)
    with exp2:
        pretty(st.session_state.get("preprocess_resp", {}))

    exp3 = st.expander("Chunk response", expanded=False)
    with exp3:
        pretty(st.session_state.get("chunk_resp", {}))

    exp4 = st.expander("Summarize response", expanded=False)
    with exp4:
        pretty(st.session_state.get("summarize_resp", {}))

    exp5 = st.expander("Embed response", expanded=False)
    with exp5:
        pretty(st.session_state.get("embed_resp", {}))


# -----------------------------
# TAB: Search
# -----------------------------
with tab_search:
    st.markdown("### Search (returns chunk_id + score + text)")
    require_config(base, tenant_id, kb_id, st.session_state.get("token", ""))
    token = st.session_state["token"]

    doc_id = st.text_input("DOC UUID for restricted search (optional)", value=st.session_state.get("doc_id", ""))
    query = st.text_area("Query", value="what does this document talk about?")
    top_k = st.slider("top_k", 1, 20, 5)

    colA, colB = st.columns(2)
    with colA:
        do_vec_search = st.button("Vector Search (/search)", use_container_width=True)
    with colB:
        do_hybrid = st.button("Hybrid Search (/hybrid-search)", use_container_width=True)

    if do_vec_search:
        body = {
            "query": query,
            "doc_id": doc_id or None,
            "top_k": int(top_k),
            "output_dimensionality": 1536,
        }
        with st.spinner("Searching..."):
            resp = requests.post(
                api_url(base, f"/api/v1/kb/{kb_id}/search"),
                headers={**auth_headers(token), "Content-Type": "application/json"},
                json=body,
                timeout=120,
            )
            data = safe_json(resp)
            st.session_state["search_resp"] = data
            if resp.status_code != 200:
                st.error(f"Search failed ({resp.status_code}): {data}")
            else:
                st.success("Search OK.")

    if do_hybrid:
        body = {
            "query": query,
            "doc_id": doc_id or None,
            "top_k": int(top_k),
            "candidate_k": 40,
            "alpha_vector": 0.75,
            "beta_keyword": 0.25,
            "output_dimensionality": 1536,
            "diversify": True,
            "cache_ttl_s": 300,
        }
        with st.spinner("Hybrid searching..."):
            resp = requests.post(
                api_url(base, f"/api/v1/kb/{kb_id}/hybrid-search"),
                headers={**auth_headers(token), "Content-Type": "application/json"},
                json=body,
                timeout=120,
            )
            data = safe_json(resp)
            st.session_state["hybrid_resp"] = data
            if resp.status_code != 200:
                st.error(f"Hybrid search failed ({resp.status_code}): {data}")
            else:
                st.success("Hybrid Search OK.")

    st.divider()
    st.markdown("### Results")

    if "search_resp" in st.session_state and st.session_state["search_resp"]:
        st.markdown("#### Vector Search response")
        pretty(st.session_state["search_resp"])
        results = st.session_state["search_resp"].get("results", [])
        if results:
            st.markdown("#### Top hits (readable)")
            for i, r in enumerate(results, 1):
                st.markdown(f"**#{i}**  chunk_id=`{r.get('chunk_id')}`  score={r.get('score'):.3f}  distance={r.get('distance'):.3f}")
                st.write(r.get("text", "")[:1500] + ("..." if len(r.get("text", "")) > 1500 else ""))
                st.divider()

    if "hybrid_resp" in st.session_state and st.session_state["hybrid_resp"]:
        st.markdown("#### Hybrid Search response")
        pretty(st.session_state["hybrid_resp"])
        results = st.session_state["hybrid_resp"].get("results", [])
        if results:
            st.markdown("#### Top hits (readable)")
            for i, r in enumerate(results, 1):
                st.markdown(
                    f"**#{i}** chunk_id=`{r.get('chunk_id')}`  "
                    f"score={r.get('score'):.3f}  vec={r.get('vector_score',0):.3f}  "
                    f"kw={r.get('keyword_score',0):.3f}  boost={r.get('entity_boost',0):.3f}"
                )
                st.write(r.get("text", "")[:1500] + ("..." if len(r.get("text", "")) > 1500 else ""))
                st.divider()


# -----------------------------
# TAB: Logs
# -----------------------------
with tab_logs:
    st.markdown("### Logs (job_events) – realtime-ish polling")

    require_config(base, tenant_id, kb_id, st.session_state.get("token", ""))

    limit = st.slider("How many latest events?", 10, 200, 50)
    col1, col2, col3 = st.columns(3)
    refresh = col1.button("Refresh Now", use_container_width=True)
    watch = col2.button("Watch 30s (poll)", use_container_width=True)
    interval = col3.selectbox("Poll interval (seconds)", [1, 2, 3, 5], index=1)

    def fetch_events() -> Dict[str, Any]:
        url = api_url(base, f"/api/v1/db/job-events/latest?tenant_id={tenant_id}&limit={int(limit)}")
        resp = requests.get(url, timeout=20)
        return safe_json(resp)

    placeholder = st.empty()

    def render_events(data: Dict[str, Any]):
        with placeholder.container():
            st.write(f"Fetched at: {time.strftime('%H:%M:%S')}")
            if data.get("status") != "ok":
                st.error(data)
                return
            events = data.get("events", [])
            st.write(f"events: {len(events)}")
            for e in events:
                st.markdown(f"- **{e.get('event_type')}**  `{e.get('created_at')}`  job_id={e.get('job_id')}")
                st.code(json.dumps(e.get("detail", {}), ensure_ascii=False, indent=2)[:1500], language="json")

    if refresh:
        try:
            render_events(fetch_events())
        except Exception as e:
            st.error(f"Log refresh failed: {e}")

    if watch:
        try:
            end = time.time() + 30
            while time.time() < end:
                render_events(fetch_events())
                time.sleep(int(interval))
        except Exception as e:
            st.error(f"Watch failed: {e}")


# -----------------------------
# TAB: Artifacts (Summary viewer)
# -----------------------------
with tab_artifacts:
    st.markdown("### Summary viewer (downloads JSON from GCS)")

    sum_resp = st.session_state.get("summarize_resp", {}) or {}
    gcs_summary_uri = sum_resp.get("gcs_summary_uri", "")

    gcs_uri_input = st.text_input("gcs_summary_uri", value=gcs_summary_uri)

    colx, coly = st.columns([1, 2])
    with colx:
        load_summary = st.button("Load summary JSON from GCS", use_container_width=True)
    with coly:
        st.caption("This needs google-cloud-storage working + GCP auth on the machine running Streamlit.")

    if load_summary:
        if not gcs_uri_input:
            st.error("Provide gcs_summary_uri (run Summarize first or paste URI).")
        else:
            try:
                with st.spinner("Downloading summary JSON from GCS..."):
                    js = gcs_download_json(gcs_uri_input, project_id=gcp_project_id or None)
                st.session_state["summary_json"] = js
                st.success("Loaded summary JSON.")
            except Exception as e:
                st.error(f"Failed to load summary: {e}")

    if "summary_json" in st.session_state:
        js = st.session_state["summary_json"]
        st.markdown("#### Document summary")
        doc_summary = js.get("doc_summary", {})
        pretty(doc_summary)

        st.markdown("#### TOC")
        toc = doc_summary.get("toc", []) if isinstance(doc_summary, dict) else []
        if toc:
            for i, sec in enumerate(toc, 1):
                st.markdown(f"**{i}. {sec.get('section','')}**  (chunks: {', '.join(sec.get('chunk_ids', [])[:6])})")
                st.write(sec.get("summary", ""))

        st.divider()
        st.markdown("#### Chunk summaries (select)")
        chunk_summaries = js.get("chunk_summaries", [])
        if isinstance(chunk_summaries, list) and chunk_summaries:
            options = [f"{i+1}. {cs.get('chunk_id','')}" for i, cs in enumerate(chunk_summaries)]
            idx = st.selectbox("Pick chunk", list(range(len(options))), format_func=lambda i: options[i])
            pretty(chunk_summaries[idx])
        else:
            st.info("No chunk_summaries found in summary JSON.")