'use strict';
/* ==========================================================
   Sighnal Pipeline Tester — app.js
   ========================================================== */

// ─── State ────────────────────────────────────────────────
const state = {
  file: null,
  docId: null,
  stepStatus: { ingest:'idle', preprocess:'idle', chunk:'idle', summarize:'idle', embed:'idle' },
  logWatchTimer: null,
  logWatchEnd: null,
  logWatchInterval: null,
};

// ─── Config (persisted in localStorage) ───────────────────
const CFG_KEYS = ['base','tenant','kb','role','token'];
function cfgGet(k) { return localStorage.getItem('sig_'+k) || ''; }
function cfgSet(k,v) { localStorage.setItem('sig_'+k, v); }

// ─── DOM helpers ──────────────────────────────────────────
const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => [...document.querySelectorAll(sel)];

function el(id) { return document.getElementById(id); }

// ─── Config inputs ────────────────────────────────────────
function loadConfig() {
  el('cfgBase').value   = cfgGet('base')   || 'http://127.0.0.1:8000';
  el('cfgTenant').value = cfgGet('tenant') || '';
  el('cfgKb').value     = cfgGet('kb')     || '';
  el('cfgRole').value   = cfgGet('role')   || 'tenant_admin';
  el('cfgToken').value  = cfgGet('token')  || '';
  updateTokenStatus();
}

function saveConfig() {
  cfgSet('base',   el('cfgBase').value.trim());
  cfgSet('tenant', el('cfgTenant').value.trim());
  cfgSet('kb',     el('cfgKb').value.trim());
  cfgSet('role',   el('cfgRole').value);
  cfgSet('token',  el('cfgToken').value.trim());
  updateTokenStatus();
}

['cfgBase','cfgTenant','cfgKb','cfgRole','cfgToken'].forEach(id => {
  el(id).addEventListener('input', saveConfig);
  el(id).addEventListener('change', saveConfig);
});

function getBase()   { return el('cfgBase').value.trim().replace(/\/$/, ''); }
function getTenant() { return el('cfgTenant').value.trim(); }
function getKb()     { return el('cfgKb').value.trim(); }
function getRole()   { return el('cfgRole').value; }
function getToken()  { return el('cfgToken').value.trim(); }

function authHeaders(extra = {}) {
  return { 'Authorization': 'Bearer ' + getToken(), ...extra };
}

// ─── Token status ─────────────────────────────────────────
function updateTokenStatus() {
  const tok = getToken();
  const statusEl = el('tokenStatus');
  if (!tok) { statusEl.textContent = ''; statusEl.className = 'token-status'; return; }
  try {
    const payload = JSON.parse(atob(tok.split('.')[1].replace(/-/g,'+').replace(/_/g,'/')));
    const exp = payload.exp;
    const now = Math.floor(Date.now()/1000);
    if (exp && now >= exp) {
      statusEl.textContent = '⚠ Token expired';
      statusEl.className = 'token-status err';
    } else {
      const mins = Math.round((exp - now) / 60);
      statusEl.textContent = `✓ Valid · expires in ${mins}m`;
      statusEl.className = 'token-status';
    }
  } catch {
    statusEl.textContent = '✓ Token set';
    statusEl.className = 'token-status';
  }
}

// Token eye toggle
el('tokenEye').addEventListener('click', () => {
  const inp = el('cfgToken');
  inp.type = inp.type === 'password' ? 'text' : 'password';
});

// ─── Auto-extract tenant from JWT ─────────────────────────
function extractTenantFromJwt(token) {
  try {
    const payload = JSON.parse(atob(token.split('.')[1].replace(/-/g,'+').replace(/_/g,'/')));
    return payload.tenant_id || null;
  } catch { return null; }
}

// ─── Validation ───────────────────────────────────────────
function requireConfig(needKb = true) {
  if (!getBase())   { toast('Missing BASE URL', 'error'); return false; }
  if (!getTenant()) { toast('Missing TENANT UUID', 'error'); return false; }
  if (needKb && !getKb()) { toast('Missing KB UUID', 'error'); return false; }
  if (!getToken())  { toast('Login first to get a token', 'error'); return false; }
  return true;
}

function getActiveDocId() {
  return el('docIdManual').value.trim() || state.docId || '';
}

// ─── API fetch wrapper ────────────────────────────────────
async function apiFetch(method, path, { body, isForm = false, timeoutMs = 300000 } = {}) {
  const url = getBase() + path;
  const headers = authHeaders(isForm ? {} : { 'Content-Type': 'application/json' });
  const ctrl = new AbortController();
  const tid = setTimeout(() => ctrl.abort(), timeoutMs);
  try {
    const resp = await fetch(url, {
      method,
      headers,
      body: isForm ? body : (body ? JSON.stringify(body) : undefined),
      signal: ctrl.signal,
    });
    clearTimeout(tid);
    let data;
    try { data = await resp.json(); } catch { data = { _status: resp.status }; }
    return { ok: resp.ok, status: resp.status, data };
  } catch (err) {
    clearTimeout(tid);
    throw err;
  }
}

// ─── Toast system ─────────────────────────────────────────
function toast(msg, type = 'info', duration = 4000) {
  const icons = { success:'✅', error:'❌', info:'ℹ', warning:'⚠' };
  const div = document.createElement('div');
  div.className = `toast ${type}`;
  div.innerHTML = `
    <span class="toast-icon">${icons[type]||'ℹ'}</span>
    <span class="toast-msg">${msg}</span>
    <button class="toast-dismiss">✕</button>
  `;
  div.querySelector('.toast-dismiss').addEventListener('click', () => removeToast(div));
  el('toastContainer').prepend(div);
  if (duration > 0) setTimeout(() => removeToast(div), duration);
  return div;
}
function removeToast(div) {
  div.style.animation = 'slideDown 0.3s ease forwards';
  setTimeout(() => div.remove(), 300);
}

// ─── JSON syntax highlighter ──────────────────────────────
function highlightJson(obj) {
  const str = JSON.stringify(obj, null, 2);
  return str
    .replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;')
    .replace(/("(\\u[a-zA-Z0-9]{4}|\\[^u]|[^\\"])*"(\s*:)?|\b(true|false|null)\b|-?\d+(?:\.\d*)?(?:[eE][+\-]?\d+)?)/g, match => {
      if (/^"/.test(match)) {
        return /:$/.test(match)
          ? `<span class="jk">${match}</span>`
          : `<span class="js">${match}</span>`;
      } else if (/true|false/.test(match)) {
        return `<span class="jb">${match}</span>`;
      } else if (/null/.test(match)) {
        return `<span class="jnull">${match}</span>`;
      }
      return `<span class="jn">${match}</span>`;
    });
}

function renderJsonBlock(obj, label = 'Response') {
  const html = highlightJson(obj);
  return `
    <div class="json-wrap">
      <div class="json-header">
        <span class="json-header-label">${label}</span>
        <div class="json-actions">
          <button class="btn btn-ghost btn-sm" onclick="openModal(${JSON.stringify(label).replace(/</g,'&lt;')}, this.closest('.json-wrap'))">⤢ Expand</button>
          <button class="btn btn-ghost btn-sm" onclick="copyJson(this)">⎘ Copy</button>
        </div>
      </div>
      <pre class="json-block">${html}</pre>
    </div>`;
}

function copyJson(btn) {
  const pre = btn.closest('.json-wrap').querySelector('.json-block');
  navigator.clipboard.writeText(pre.textContent).then(() => {
    btn.textContent = '✓ Copied';
    setTimeout(() => btn.textContent = '⎘ Copy', 1500);
  });
}

function openModal(title, jsonWrap) {
  el('modalTitle').textContent = title;
  el('modalJson').innerHTML = jsonWrap.querySelector('.json-block').innerHTML;
  el('modalOverlay').style.display = 'flex';
}

el('modalClose').addEventListener('click', () => el('modalOverlay').style.display = 'none');
el('modalOverlay').addEventListener('click', e => { if (e.target === el('modalOverlay')) el('modalOverlay').style.display = 'none'; });

// ─── Sidebar ──────────────────────────────────────────────
el('sidebarToggle').addEventListener('click', () => {
  el('sidebar').classList.toggle('collapsed');
});
el('mobileSidebarToggle').addEventListener('click', () => {
  el('sidebar').classList.toggle('mobile-open');
});

// ─── Tabs ─────────────────────────────────────────────────
$$('.tab-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    $$('.tab-btn').forEach(b => b.classList.remove('active'));
    $$('.tab-pane').forEach(p => p.classList.remove('active'));
    btn.classList.add('active');
    el('tab-' + btn.dataset.tab).classList.add('active');
  });
});

// ─── File Upload ──────────────────────────────────────────
const dropZone = el('dropZone');

dropZone.addEventListener('click', e => {
  if (!e.target.closest('.file-remove')) el('fileInput').click();
});
el('fileInput').addEventListener('change', e => {
  if (e.target.files[0]) setFile(e.target.files[0]);
});
dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('drag-over'); });
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
dropZone.addEventListener('drop', e => {
  e.preventDefault();
  dropZone.classList.remove('drag-over');
  if (e.dataTransfer.files[0]) setFile(e.dataTransfer.files[0]);
});
el('fileRemove').addEventListener('click', e => { e.stopPropagation(); clearFile(); });

function setFile(f) {
  state.file = f;
  el('fileName').textContent = f.name;
  el('fileSize').textContent = formatBytes(f.size);
  el('filePreview').style.display = 'flex';
  el('dropZone').querySelector('.drop-zone-inner').style.display = 'none';
}
function clearFile() {
  state.file = null;
  el('filePreview').style.display = 'none';
  el('dropZone').querySelector('.drop-zone-inner').style.display = 'flex';
  el('fileInput').value = '';
}
function formatBytes(b) {
  if (b < 1024) return b + ' B';
  if (b < 1024*1024) return (b/1024).toFixed(1) + ' KB';
  return (b/1024/1024).toFixed(1) + ' MB';
}

// ─── Step state machine ───────────────────────────────────
function setStepState(step, state_) {
  state.stepStatus[step] = state_;
  const circle = el('circle-' + step);
  const card   = el('card-' + step);
  const spin   = el('spin-' + step);
  const dot    = el('navDot-' + step);
  const btn    = el('btn-' + step);

  circle.className = 'step-circle ' + (state_ === 'idle' ? '' : state_);
  card.className   = 'step-card '   + (state_ === 'idle' ? '' : state_);
  dot.className    = 'step-dot '    + (state_ === 'running' ? 'running' : state_ === 'done' ? 'done' : state_ === 'error' ? 'error' : '');

  if (spin) spin.className = 'spinner ' + (state_ === 'running' ? 'active' : '');
  if (btn) btn.disabled = state_ === 'running';
}

function setStepMeta(step, text) {
  const m = el('meta-' + step);
  if (!m) return;
  m.textContent = text;
  m.className = 'step-card-meta ' + (text ? 'visible' : '');
}

function setStepResponse(step, obj) {
  const r = el('resp-' + step);
  if (!r) return;
  r.innerHTML = renderJsonBlock(obj, step + ' response');
  r.className = 'step-response open';
}

// ─── Health + Login ───────────────────────────────────────
el('btnHealth').addEventListener('click', async () => {
  if (!getBase()) { toast('Set BASE URL first', 'warning'); return; }
  el('dotHealth').className = 'health-dot';
  el('dotReady').className  = 'health-dot';
  try {
    const [h, r] = await Promise.all([
      fetch(getBase() + '/api/v1/health', { signal: AbortSignal.timeout(10000) }),
      fetch(getBase() + '/api/v1/ready',  { signal: AbortSignal.timeout(10000) }),
    ]);
    el('dotHealth').className = 'health-dot ' + (h.ok ? 'ok' : 'err');
    el('dotReady').className  = 'health-dot ' + (r.ok ? 'ok' : 'err');
    if (h.ok && r.ok) toast('API is healthy ✓', 'success');
    else toast('API health check failed', 'error');
  } catch (e) {
    el('dotHealth').className = 'health-dot err';
    el('dotReady').className  = 'health-dot err';
    toast('Cannot reach API: ' + e.message, 'error');
  }
});

el('btnLogin').addEventListener('click', async () => {
  if (!getBase())   { toast('Set BASE URL first', 'warning'); return; }
  if (!getTenant()) { toast('Set TENANT UUID first', 'warning'); return; }

  const t = toast('Logging in…', 'info', 0);
  try {
    const resp = await fetch(getBase() + '/api/v1/auth/login', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ tenant_id: getTenant(), role: getRole() }),
      signal: AbortSignal.timeout(20000),
    });
    const data = await resp.json();
    removeToast(t);
    if (!resp.ok) { toast('Login failed: ' + (data.detail || resp.status), 'error'); return; }

    const token = data.access_token || '';
    el('cfgToken').value = token;
    cfgSet('token', token);
    updateTokenStatus();

    // Auto-populate tenant_id from response (or JWT)
    if (data.tenant_id && !getTenant()) {
      el('cfgTenant').value = data.tenant_id;
      cfgSet('tenant', data.tenant_id);
      el('tenantAutoTag').style.display = 'inline';
    }

    toast(`Logged in · role=${data.role} · expires in ${Math.round(data.expires_in/60)}m`, 'success');
  } catch (e) {
    removeToast(t);
    toast('Login error: ' + e.message, 'error');
  }
});

// ─── Pipeline steps ───────────────────────────────────────
async function runIngest() {
  if (!requireConfig()) return;
  if (!state.file) { toast('Upload a file first', 'warning'); return; }
  setStepState('ingest', 'running');
  setStepMeta('ingest', `Uploading ${state.file.name}…`);
  const t = toast('Ingesting file…', 'info', 0);
  try {
    const fd = new FormData();
    fd.append('file', state.file, state.file.name);
    const resp = await fetch(getBase() + `/api/v1/kb/${getKb()}/ingest/file`, {
      method: 'POST',
      headers: authHeaders(),
      body: fd,
      signal: AbortSignal.timeout(180000),
    });
    removeToast(t);
    const data = await resp.json();
    if (!resp.ok) {
      setStepState('ingest', 'error');
      setStepMeta('ingest', `Error ${resp.status}: ${data.detail||''}`);
      setStepResponse('ingest', data);
      toast('Ingest failed: ' + (data.detail || resp.status), 'error');
      return;
    }
    const docId = data.doc_id || data.doc?.doc_id;
    if (docId) {
      state.docId = docId;
      el('activeDocId').textContent = docId;
      el('docIdDisplay').style.display = 'flex';
    }
    setStepState('ingest', 'done');
    setStepMeta('ingest', `doc_id: ${docId} · chars: ${data.extracted_chars ?? '?'} · method: ${data.extraction_method ?? '?'}`);
    setStepResponse('ingest', data);
    toast(`Ingest OK · doc_id=${docId}`, 'success');
  } catch (e) {
    removeToast(t);
    setStepState('ingest', 'error');
    setStepMeta('ingest', e.message);
    toast('Ingest error: ' + e.message, 'error');
  }
}

async function runPreprocess() {
  if (!requireConfig()) return;
  const docId = getActiveDocId();
  if (!docId) { toast('No doc_id — run Ingest first or paste DOC UUID', 'warning'); return; }
  setStepState('preprocess', 'running');
  setStepMeta('preprocess', 'Preprocessing text…');
  const t = toast('Preprocessing…', 'info', 0);
  try {
    const { ok, status, data } = await apiFetch('POST', `/api/v1/kb/${getKb()}/preprocess/${docId}`, { body: {} });
    removeToast(t);
    if (!ok) {
      setStepState('preprocess', 'error');
      setStepMeta('preprocess', `Error ${status}`);
      setStepResponse('preprocess', data);
      toast('Preprocess failed: ' + (data.detail || status), 'error');
      return;
    }
    setStepState('preprocess', 'done');
    setStepMeta('preprocess', `status: ${data.status||'ok'}`);
    setStepResponse('preprocess', data);
    toast('Preprocess OK', 'success');
  } catch(e) {
    removeToast(t);
    setStepState('preprocess', 'error');
    setStepMeta('preprocess', e.message);
    toast('Preprocess error: ' + e.message, 'error');
  }
}

async function runChunk() {
  if (!requireConfig()) return;
  const docId = getActiveDocId();
  if (!docId) { toast('No doc_id — run Ingest first or paste DOC UUID', 'warning'); return; }
  setStepState('chunk', 'running');
  setStepMeta('chunk', 'Chunking document…');
  const t = toast('Chunking…', 'info', 0);
  try {
    const { ok, status, data } = await apiFetch('POST', `/api/v1/kb/${getKb()}/chunk/${docId}`, { body: {} });
    removeToast(t);
    if (!ok) {
      setStepState('chunk', 'error');
      setStepMeta('chunk', `Error ${status}`);
      setStepResponse('chunk', data);
      toast('Chunk failed: ' + (data.detail || status), 'error');
      return;
    }
    const count = data.chunk_count ?? data.chunks_written ?? '?';
    setStepState('chunk', 'done');
    setStepMeta('chunk', `chunks: ${count}`);
    setStepResponse('chunk', data);
    toast(`Chunk OK · ${count} chunks`, 'success');
  } catch(e) {
    removeToast(t);
    setStepState('chunk', 'error');
    setStepMeta('chunk', e.message);
    toast('Chunk error: ' + e.message, 'error');
  }
}

async function runSummarize() {
  if (!requireConfig()) return;
  const docId = getActiveDocId();
  if (!docId) { toast('No doc_id — run Ingest first or paste DOC UUID', 'warning'); return; }
  setStepState('summarize', 'running');
  setStepMeta('summarize', 'Summarizing (may take 1–3 min)…');
  const t = toast('Summarizing… this may take a while', 'info', 0);
  try {
    const { ok, status, data } = await apiFetch('POST', `/api/v1/kb/${getKb()}/summarize/${docId}`, { body: {}, timeoutMs: 600000 });
    removeToast(t);
    if (!ok) {
      setStepState('summarize', 'error');
      setStepMeta('summarize', `Error ${status}`);
      setStepResponse('summarize', data);
      toast('Summarize failed: ' + (data.detail || status), 'error');
      return;
    }
    // Auto-fill GCS URI in artifacts tab
    if (data.gcs_summary_uri) el('gcsUriInput').value = data.gcs_summary_uri;
    setStepState('summarize', 'done');
    setStepMeta('summarize', `gcs_summary_uri: ${data.gcs_summary_uri || '—'}`);
    setStepResponse('summarize', data);
    toast('Summarize OK', 'success');
  } catch(e) {
    removeToast(t);
    setStepState('summarize', 'error');
    setStepMeta('summarize', e.message);
    toast('Summarize error: ' + e.message, 'error');
  }
}

async function runEmbed() {
  if (!requireConfig()) return;
  const docId = getActiveDocId();
  if (!docId) { toast('No doc_id — run Ingest first or paste DOC UUID', 'warning'); return; }
  setStepState('embed', 'running');
  setStepMeta('embed', 'Embedding chunks (batch 16)…');
  const t = toast('Embedding chunks…', 'info', 0);
  try {
    const payload = { batch_size: 16, max_chunks: 500, output_dimensionality: 1536 };
    const { ok, status, data } = await apiFetch('POST', `/api/v1/kb/${getKb()}/embed/${docId}`, { body: payload, timeoutMs: 600000 });
    removeToast(t);
    if (!ok) {
      setStepState('embed', 'error');
      setStepMeta('embed', `Error ${status}`);
      setStepResponse('embed', data);
      toast('Embed failed: ' + (data.detail || status), 'error');
      return;
    }
    const count = data.embedded_count ?? '?';
    setStepState('embed', 'done');
    setStepMeta('embed', `embedded_count: ${count}`);
    setStepResponse('embed', data);
    toast(`Embed OK · ${count} embeddings`, 'success');
  } catch(e) {
    removeToast(t);
    setStepState('embed', 'error');
    setStepMeta('embed', e.message);
    toast('Embed error: ' + e.message, 'error');
  }
}

// Run All
async function runAll() {
  if (!requireConfig()) return;
  if (!state.file) { toast('Upload a file first', 'warning'); return; }
  await runIngest();
  if (state.stepStatus.ingest !== 'done') return;
  await runPreprocess();
  if (state.stepStatus.preprocess !== 'done') return;
  await runChunk();
  if (state.stepStatus.chunk !== 'done') return;
  await runSummarize();
  if (state.stepStatus.summarize !== 'done') return;
  await runEmbed();
}

// Wire buttons
el('btn-ingest').addEventListener('click', runIngest);
el('btn-preprocess').addEventListener('click', runPreprocess);
el('btn-chunk').addEventListener('click', runChunk);
el('btn-summarize').addEventListener('click', runSummarize);
el('btn-embed').addEventListener('click', runEmbed);
el('btnRunAll').addEventListener('click', runAll);

// ─── Search ───────────────────────────────────────────────
el('searchTopK').addEventListener('input', () => {
  el('searchTopKVal').textContent = el('searchTopK').value;
});

function scoreClass(s) {
  if (s >= 0.7) return 'score-high';
  if (s >= 0.4) return 'score-med';
  return 'score-low';
}

function renderResults(results, mode) {
  if (!results || results.length === 0) {
    return '<div class="log-empty">No results returned.</div>';
  }
  return results.map((r, i) => {
    const score = r.score ?? 0;
    const text  = r.text || '';
    const extra = mode === 'hybrid'
      ? ` <span class="score-badge" title="vector">vec ${(r.vector_score||0).toFixed(3)}</span>
         <span class="score-badge" title="keyword">kw ${(r.keyword_score||0).toFixed(3)}</span>`
      : (r.distance != null ? ` <span class="score-badge" title="distance">dist ${r.distance.toFixed(3)}</span>` : '');
    return `
      <div class="result-card">
        <div class="result-header">
          <span class="result-rank">#${i+1}</span>
          <span class="score-badge ${scoreClass(score)}">${score.toFixed(3)}</span>
          ${extra}
          <span class="result-chunk-id">${r.chunk_id || ''}</span>
        </div>
        <div class="result-text" id="rt-${i}">${escapeHtml(text.slice(0, 1500))}${text.length>1500?'…':''}</div>
        ${text.length > 300 ? `<button class="result-expand" onclick="toggleExpand('rt-${i}', this)">Show more</button>` : ''}
      </div>`;
  }).join('');
}

function toggleExpand(id, btn) {
  const el_ = el(id);
  el_.classList.toggle('expanded');
  btn.textContent = el_.classList.contains('expanded') ? 'Show less' : 'Show more';
}

function escapeHtml(s) {
  return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

async function doVectorSearch() {
  if (!requireConfig()) return;
  const spin = el('spin-vecSearch'); spin.className = 'spinner active';
  el('btnVecSearch').disabled = true;
  const t = toast('Vector searching…', 'info', 0);
  try {
    const docId = el('searchDocId').value.trim() || null;
    const body = {
      query: el('searchQuery').value,
      doc_id: docId,
      top_k: parseInt(el('searchTopK').value),
      output_dimensionality: 1536,
    };
    const { ok, status, data } = await apiFetch('POST', `/api/v1/kb/${getKb()}/search`, { body });
    removeToast(t);
    if (!ok) { toast('Search failed: ' + (data.detail||status), 'error'); return; }
    const results = data.results || [];
    el('searchResults').innerHTML =
      `<div class="section-card"><div class="section-title"><span class="section-icon">🧠</span> Vector Search — ${results.length} results</div>${renderResults(results, 'vector')}${renderJsonBlock(data,'Vector Search Response')}</div>`;
    toast(`Vector search OK · ${results.length} hits`, 'success');
  } catch(e) {
    removeToast(t);
    toast('Search error: ' + e.message, 'error');
  } finally {
    spin.className = 'spinner';
    el('btnVecSearch').disabled = false;
  }
}

async function doHybridSearch() {
  if (!requireConfig()) return;
  const spin = el('spin-hybridSearch'); spin.className = 'spinner active';
  el('btnHybridSearch').disabled = true;
  const t = toast('Hybrid searching…', 'info', 0);
  try {
    const docId = el('searchDocId').value.trim() || null;
    const body = {
      query: el('searchQuery').value,
      doc_id: docId,
      top_k: parseInt(el('searchTopK').value),
      candidate_k: parseInt(el('hybridCandidateK').value) || 40,
      alpha_vector: parseFloat(el('hybridAlpha').value) || 0.75,
      beta_keyword: parseFloat(el('hybridBeta').value) || 0.25,
      output_dimensionality: 1536,
      diversify: el('hybridDiversify').value === 'true',
      cache_ttl_s: parseInt(el('hybridTtl').value) || 300,
    };
    const { ok, status, data } = await apiFetch('POST', `/api/v1/kb/${getKb()}/hybrid-search`, { body });
    removeToast(t);
    if (!ok) { toast('Hybrid search failed: ' + (data.detail||status), 'error'); return; }
    const results = data.results || [];
    el('searchResults').innerHTML =
      `<div class="section-card"><div class="section-title"><span class="section-icon">⚡</span> Hybrid Search — ${results.length} results</div>${renderResults(results, 'hybrid')}${renderJsonBlock(data,'Hybrid Search Response')}</div>`;
    toast(`Hybrid search OK · ${results.length} hits`, 'success');
  } catch(e) {
    removeToast(t);
    toast('Hybrid error: ' + e.message, 'error');
  } finally {
    spin.className = 'spinner';
    el('btnHybridSearch').disabled = false;
  }
}

el('btnVecSearch').addEventListener('click', doVectorSearch);
el('btnHybridSearch').addEventListener('click', doHybridSearch);

// ─── Logs ─────────────────────────────────────────────────
el('logLimit').addEventListener('input', () => {
  el('logLimitVal').textContent = el('logLimit').value;
});

function logTypeClass(t) {
  if (/started/i.test(t)) return 'log-type-started';
  if (/done|success|ok/i.test(t)) return 'log-type-done';
  if (/error|fail/i.test(t)) return 'log-type-error';
  return 'log-type-default';
}

async function fetchAndRenderLogs() {
  const tenant = getTenant();
  const limit  = el('logLimit').value;
  if (!getBase() || !tenant) { toast('Set BASE URL and TENANT UUID first', 'warning'); return; }
  try {
    const url = `${getBase()}/api/v1/db/job-events/latest?tenant_id=${encodeURIComponent(tenant)}&limit=${limit}`;
    const resp = await fetch(url, { signal: AbortSignal.timeout(20000) });
    const data = await resp.json();
    renderLogs(data);
  } catch(e) {
    toast('Log fetch failed: ' + e.message, 'error');
  }
}

function renderLogs(data) {
  const feed = el('logFeed');
  if (!data || data.status !== 'ok') {
    feed.innerHTML = `<div class="log-empty">Error: ${JSON.stringify(data)}</div>`;
    return;
  }
  const events = data.events || [];
  if (!events.length) {
    feed.innerHTML = '<div class="log-empty">No events found.</div>';
    return;
  }
  feed.innerHTML = events.map((e, i) => {
    const detail = JSON.stringify(e.detail || {}, null, 2);
    return `
      <div class="log-event">
        <div class="log-event-header">
          <span class="log-type ${logTypeClass(e.event_type||'')}">${e.event_type||'?'}</span>
          <span class="log-time">${e.created_at||''}</span>
          <span class="log-job">job: ${e.job_id||'—'}</span>
        </div>
        <div class="log-detail" id="ld-${i}" onclick="this.classList.toggle('expanded')">${escapeHtml(detail.slice(0,500))}${detail.length>500?'…':''}</div>
      </div>`;
  }).join('');
}

el('btnLogRefresh').addEventListener('click', fetchAndRenderLogs);

el('btnLogWatch').addEventListener('click', () => {
  if (state.logWatchTimer) { stopWatch(); return; }
  startWatch();
});

el('btnLogStop').addEventListener('click', stopWatch);

function startWatch() {
  const durationMs = 30000;
  const intervalSec = parseInt(el('logInterval').value) || 2;
  state.logWatchEnd = Date.now() + durationMs;
  el('btnLogWatch').textContent = '■ Stop';
  el('logWatchBar').style.display = 'flex';

  const tick = async () => {
    const remaining = Math.max(0, state.logWatchEnd - Date.now());
    const pct = (1 - remaining / durationMs) * 100;
    el('logWatchProgress').style.width = pct + '%';
    el('logWatchLabel').textContent = `Watching… ${Math.ceil(remaining/1000)}s`;
    await fetchAndRenderLogs();
    if (remaining <= 0) { stopWatch(); return; }
  };
  tick();
  state.logWatchTimer = setInterval(tick, intervalSec * 1000);
}

function stopWatch() {
  clearInterval(state.logWatchTimer);
  state.logWatchTimer = null;
  el('btnLogWatch').textContent = '▶ Watch (30s)';
  el('logWatchBar').style.display = 'none';
}

// ─── Artifacts ────────────────────────────────────────────
// Just shows the GCS URI — actual download needs server-side proxy
// We display the summarize response JSON that was already captured
el('gcsUriInput').addEventListener('input', () => {
  const uri = el('gcsUriInput').value.trim();
  const display = el('artifactDisplay');
  if (!uri) { display.innerHTML = ''; return; }
  display.innerHTML = `
    <div class="artifact-note">
      GCS URI: <code>${escapeHtml(uri)}</code><br/>
      To view the summary JSON, ensure the Summarize step ran.
      The full summarize response is shown below if available.
    </div>`;
  // Show summarize response if we have it in any open step card
  const preEl = el('resp-summarize');
  if (preEl && preEl.innerHTML) {
    display.innerHTML += preEl.innerHTML;
  }
});

// ─── Init ─────────────────────────────────────────────────
loadConfig();
updateTokenStatus();

// Sync search doc_id with active doc
el('docIdManual').addEventListener('input', () => {
  const val = el('docIdManual').value.trim();
  if (val) {
    el('searchDocId').value = val;
  }
});
