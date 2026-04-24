from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
import uuid
from collections import Counter
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Tuple

import psycopg2
import requests
from fastapi import APIRouter, Depends, Header, HTTPException
from jose import JWTError, jwt
from pydantic import BaseModel, Field

from google.cloud import storage
from google.api_core.exceptions import PreconditionFailed
import google.generativeai as genai
from openai import OpenAI

try:
    from app.services.blog_pipeline.pipeline_runner import run_blog_pipeline as _run_blog_pipeline
    _BLOG_PIPELINE_AVAILABLE = True
except Exception:
    _BLOG_PIPELINE_AVAILABLE = False


router = APIRouter(prefix="/api/v1/articles", tags=["article-run"])

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_DEFAULT_MODEL = "llama-3.3-70b-versatile"
OPENAI_DEFAULT_MODEL = "gpt-5.2-2025-12-11"

# ============================================================
# Cached system prompt  -  identical for every article, enables
# OpenAI prompt caching (50% savings on input tokens after 1st call)
# ============================================================
# ============================================================
# Layer 1: HUMANIZATION CORE  -  NEVER changes. This is what got
# ZeroGPT from 77.5% â†' 10%. Locked across ALL articles.
# ============================================================
HUMANIZATION_CORE = """HOW TO WRITE FOR A PROFESSIONAL EDITORIAL PLATFORM (follow ALL of these  -  these rules OVERRIDE everything):

BURSTINESS  -  vary sentence length (critical rule  -  follow the RATIO):
- MOST sentences (60 - 70%) must be 10 - 18 words: analytical, evidence-based, complete thoughts with a subject, verb, and consequence.
- LONG sentences (18 - 26 words) anchor every paragraph  -  use em-dashes and parenthetical observations to build them out.
- SHORT punches (3 - 6 words) are used sparingly  -  maximum 1 in every 5 sentences  -  for emphasis AFTER a longer analytical sentence. Never open a paragraph with a fragment.
- Never write 3 consecutive sentences of the same length. Count them.
- Short punches always follow a long analytical sentence  -  never the reverse, never two punches in a row.

PERPLEXITY  -  be structurally unpredictable (not casual  -  editorially sharp):
- Use colon reveals: "What this means for recovery: the window closes within 30 minutes."
- Use temporal contrast: "A decade ago, the standard advice was bed rest. The evidence now says the opposite."
- Use concessive pivots: "Still, the numbers don't lie." "By contrast, the data points elsewhere."
- Use inverted syntax: lead with the observation, then the evidence behind it.
- Use domain-crossing lens: borrow vocabulary from adjacent fields to sharpen the description (e.g. economics for policy, physics for biology, engineering for medicine).
- Follow long explanatory sentences with a 3-6 word blunt verdict: "That is the problem." "The data confirms it." "The gap is widening."
- Use contrast pairs: "Isolation magnifies stress. Connection dilutes it." "The intervention costs £40. Hospitalisation costs £4,000."

SEMANTIC DEPTH  -  every paragraph must explain the WHY, not just the WHAT:
- State the fact AND what it means. Not "cortisol rises" but "cortisol rises  -  and when it stays elevated past 48 hours, the hippocampus begins to shrink, impairing memory consolidation."
- Connect data points with causality: "Patients in the high-sleep group averaged 2.1 fewer sick days  -  a gap that compounds across a workforce of thousands."
- Causal chains ('X happened because Y, which means Z') are your most analytically valuable sentences. KEEP them intact. Do not fragment them.
- Every paragraph must contain at least one sentence of 15+ words that explains WHY something happened or WHAT it means for the broader picture.
- Bare facts without context are not analysis. "Sleep deprivation is bad." tells nothing. "Cutting sleep below six hours for two weeks produces cognitive impairment equivalent to 48 hours of total sleep deprivation  -  yet most people feel only moderately tired." carries meaning.

EDITORIAL VOICE  -  authoritative, not personal:
- Write like a senior correspondent who has studied the data and drawn conclusions.
- Occasional editorial judgment is acceptable: "That figure deserves attention." "The timing was not coincidental."
- Contractions are fine in editorial writing: "it's", "don't", "that's", "they've", "won't".
- Use "And" and "But" to open sentences  -  both are standard in quality editorial prose.
- No casual fillers: no "yeah", "honestly", "basically", "kind of", "I mean", "pretty much".
- No personal anecdotes: no "I think", "in my experience", "a friend of mine", "I tried this".
- Avoid casual chat-speak questions: no "Right?", "See what I mean?", "Get it?". Reader-empathetic bridges are acceptable when used sparingly  -  "You may be wondering...", "If you've ever noticed...", "That raises a real question:"  -  maximum once per article, never as a rhetorical filler.
- No casual transitions: no "Okay but", "Plot twist", "Here's the thing", "So anyway", "Look  - ".
- Professional pivots: "However", "Meanwhile", "By contrast", "What followed", "Still".

ARTICLE STRUCTURE  -  adapt to content type (CRITICAL):

For NARRATIVE topics (sports, health, profiles, news, culture, analysis, opinion, explainers):
- HOOK (first 150-200 words, NO heading): Open with a specific moment, tension, or striking fact. No stats dump, no "In this article we will...". Pull the reader in before any section heading appears.
- NARRATIVE BODY (sections 1-6): Each section is a chapter  -  cause, development, consequence. Smooth transitions. The reader feels the story moving forward, not jumping between content blocks.
- INSIGHT LAYER (at least one section): Editorial perspective on why this matters, what it reveals, what the data actually means beyond the numbers.
- CONCLUSION (last section before FAQ): End with a specific observation or forward-looking statement. No "In conclusion".
- FAQ (heading exactly "## FAQ"): 5-7 real reader questions answered concisely.

For INSTRUCTIONAL topics (how-to, tutorial, guide, steps):
- Clear problem/goal intro (1-2 paragraphs, no heading).
- Structured sections with numbered steps or logical phases where appropriate.
- Practical examples and outcomes per section.
- FAQ at the end.

HEADINGS  -  informative and specific, never label-style:
  Good: "The Season That Changed Everything" / "Why He Never Left" / "Step 3: Configure the Environment"
  Bad: "Career Statistics" / "Player Profile Snapshot" / "More Information" / "Additional Details"
  Numbered section headings like "1. Introduction" or "14) FAQ" are NEVER acceptable.

STRUCTURE  -  break the AI pattern with deliberate variety:
- Paragraphs: 1-4 sentences. Some are one sentence for emphasis. Never all the same length.
- Bullet lists: acceptable for concrete data points or step sequences  -  keep tight, 3-6 items max. Never use a bullet list as the only content of a prose section.
- Do NOT wrap up sections with neat summary sentences. State the fact, move on.
- Do NOT start multiple consecutive paragraphs the same way.

SPECIFICITY  -  concrete details beat abstract claims every time:
- Lead with the number or name: "The trial enrolled 4,200 patients. The result was a 34% reduction in relapse."
- Replace vague summaries with evidence: not "several studies found benefits" but "three RCTs with a combined 11,000 participants found a 22% reduction in all-cause mortality."
- Use contrast to show scale: "In 2019, the rate was 1 in 8. By 2024, it was 1 in 5."
- The principle applies to every domain: name the study, name the number, name the person, name the date. Vague claims have no place in analytical writing.

BOLD FORMATTING  -  readability and ZeroGPT both suffer from overuse:
- Bold only: key proper nouns on first mention and ONE critical number or term per section
- Do NOT bold full analytical clauses: not "**the trial showed a 34% reduction**"  -  just "a **34%** reduction"
- Do NOT bold sentences or summary judgments  -  bold is for nouns and numbers, never for conclusions
- Maximum 2-3 bold elements per paragraph; zero bold is perfectly fine for analytical sections
- Excessive bold signals AI generation and visually clutters the article

IMPLICIT AI PATTERNS  -  these make prose feel machine-generated AND weaker. Cutting them improves readability first, reduces AI score second:

TRANSITIONAL FILLER  -  adds zero meaning, wastes the reader's time:
Never use: "That being said", "With that in mind", "Building on this", "Moving forward", "Having said that", "As mentioned earlier", "As we've seen", "In terms of", "With respect to", "It goes without saying", "Needless to say".
Fix: delete the phrase and start directly with the next thought. If the transition needs words, the preceding sentence needs rewriting.

PASSIVE ATTRIBUTION  -  kills credibility and signals AI:
Never: "Research has shown that...", "It has been found that...", "Studies suggest that...", "It is believed that...", "It is widely accepted that..."
Always name the source: "A 2023 meta-analysis of 12 trials found...", "Researchers at Stanford showed...", "A cohort study of 4,200 patients over five years found..."
If no named source exists in the evidence: use ONE precise hedge  -  "The available evidence points to..."  -  then state the finding directly.

PROSE ENUMERATION  -  "First... Second... Third... Finally..." in paragraph text is structural laziness:
Never chain numbered transitions in prose. If you have three connected points, write them as flowing analysis: "The mechanism works in stages  -  the initial cortisol spike triggers X, which compounds into Y, and the cumulative effect over 48 hours is Z."
Reserve numbered lists for actual sequential steps only.

HEDGE STACKING  -  multiple hedges per sentence signal automated caution, not human judgment:
Bad: "This may possibly help to potentially reduce the risk of..."
Good: "The evidence points to a meaningful risk reduction  -  though effect size varies by dosage."
One precise qualifier per claim. Then move on.

FILLER NOUNS  -  vague, unverifiable, domain-specific AI clichés:
Health: holistic, well-being (as a vague noun), empower, optimize your health, lifestyle changes, inflammation (without mechanism), gut health (without specifics).
All domains: world-class, transformative, groundbreaking, unprecedented, forward-thinking, innovative solution, best practices, thought leadership.
Fix: replace with the specific thing. Not "holistic approach"  -  name the actual combination. Not "world-class"  -  give the metric.

PARAGRAPH-END SUMMARIES  -  "This shows why X matters." closes every paragraph like a school essay:
State the fact and its consequence. Then stop. The reader understood. The summary sentence is redundant and reads as AI padding.

READABILITY BALANCE  -  anti-detection rules must never sacrifice clarity:
- Sentence fragments are emphasis tools after a long analytical sentence  -  not randomness generators. Three fragments in a row is worse than one clear flowing sentence.
- Structural unpredictability means varying rhythm, not breaking logic. A causal chain must stay intact even if it produces two sentences of similar length.
- Clarity is the floor. Anti-AI technique layers on top of clarity  -  it never replaces it.
- When a rule conflicts with the reader understanding the point: break the rule, keep the clarity.

WRITING PROHIBITION  -  NEVER write in these styles regardless of evidence quality or source material:
- NEVER write at a child reading level  -  no nursery analogies, no "imagine you are..." framings, no "let's explore together" openings.
- NEVER use talk-down comparisons: "like a video game", "like cleaning your room", "like a cookie jar", "like pocket money".
- NEVER pad with generic beginner explainers when evidence runs thin. Write what the evidence supports; explicitly note where data is missing.
- Bullet lists of factual items (symptoms, warning signs, risk factors, study findings) are acceptable  -  keep them tight (4-8 items) and always follow with at least one analytical sentence explaining what the list means.
- NEVER exceed 12 sections  -  more sections pad word count, they do not add analysis.
- FAQ answers must be at Grade 7+ level  -  concise, factual, no child-level similes.

BANNED WORDS  -  explicit AI detection triggers:
delve, leverage, furthermore, moreover, it's worth noting, in conclusion,
pivotal, tapestry, embark, navigate, landscape, realm, cutting-edge,
game-changer, revolutionize, comprehensive, utilizing, facilitate,
it is important to note, in today's world, in the realm of, it is crucial,
paramount, foster, streamline, robust, seamless, synergy, harness,
notable, significant, vital, essential, underscores, multifaceted,
a testament to, it should be noted, this serves as, plays a crucial role,
that being said, with that in mind, moving forward, building on this,
holistic approach, well-being (standalone noun), empower, transformative, unprecedented

SELF-CHECK after each section:
- Are there 3 consecutive sentences of similar length? Break them up.
- Does every sentence add specific new information? If not, cut it.
- Does this read like published editorial content? If it sounds like a student essay or an AI report, rewrite it.
- Did you use any banned words or implicit patterns above? Fix them.
- Are causal chains intact? ('X because Y, which means Z')  -  KEEP them. Only break up filler, never reasoning.
- Does each paragraph explain WHY, not just WHAT?
- Is any paragraph a summary of what the section already said? Delete it.
- Read the last sentence of each paragraph  -  if it starts with "This shows" or "This means" or "This is why", cut it."""


# ============================================================
# Phase 1: Analyze source material (~500 tokens)
# Returns tone, complexity, content_type, audience_familiarity
# Fully generalized  -  no domain-specific logic
# ============================================================
def _analyze_source(
    client: OpenAI,
    model: str,
    sources_block: str,
    *,
    temperature: float = 0.3,
) -> Tuple[Dict[str, str], Dict[str, int]]:
    """Cheap GPT call to classify source material tone/complexity/type."""
    analysis_prompt = """Read these source excerpts and classify the content.
Return ONLY this JSON  -  no explanation, no markdown, no preamble:

{
  "tone": "casual" or "professional" or "academic" or "conversational" or "journalistic",
  "complexity": "beginner" or "intermediate" or "expert",
  "content_type": "news" or "analysis" or "tutorial" or "opinion" or "explainer" or "report",
  "audience_familiarity": "new_to_topic" or "knows_basics" or "domain_expert",
  "voice_hint": "one sentence  -  who would naturally write about this topic"
}

SOURCE EXCERPTS:
""" + sources_block[:3000]  # cap to save tokens

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You analyze text and return JSON. Nothing else."},
                {"role": "user", "content": analysis_prompt},
            ],
            temperature=temperature,
            max_completion_tokens=256,
        )
        raw = (resp.choices[0].message.content or "").strip()
        # Parse JSON  -  handle markdown code blocks
        raw = re.sub(r"^```json\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
        analysis = json.loads(raw)
        u = resp.usage
        usage = {
            "prompt_tokens": u.prompt_tokens if u else 0,
            "output_tokens": u.completion_tokens if u else 0,
            "total_tokens": u.total_tokens if u else 0,
        }
        return analysis, usage
    except Exception:
        # If analysis fails, return safe defaults  -  never block article generation
        return {
            "tone": "professional",
            "complexity": "intermediate",
            "content_type": "explainer",
            "audience_familiarity": "knows_basics",
            "voice_hint": "someone knowledgeable sharing what they found",
        }, {"prompt_tokens": 0, "output_tokens": 0, "total_tokens": 0}


# ============================================================
# Phase 2: Build dynamic system prompt from analysis
# Layer 1 (humanization) is LOCKED  -  Layer 2 (voice) adapts
# ============================================================
_TONE_MAP = {
    "casual":         "Write with energy and directness  -  punchy sentences, clear observations, no padding.",
    "professional":   "Clear, confident, authoritative  -  editorial quality without being corporate or stiff.",
    "academic":       "Evidence-based and structured, but readable  -  not a textbook, not a press release.",
    "conversational": "Warm and direct, like a knowledgeable correspondent explaining findings to a sharp reader.",
    "journalistic":   "Lead with the most specific fact. Short paragraphs. What happened, why it matters, what's next.",
}

_COMPLEXITY_MAP = {
    "beginner":      "The source material covers basic ground  -  write at Grade 7+ editorial level regardless. Explain domain terms in a single clause; never use child-level analogies or beginner step-by-step guides.",
    "intermediate":  "Reader knows the basics. Use domain terms but keep it accessible.",
    "expert":        "Reader is familiar with the field. Use proper terminology. Skip the 101 stuff.",
}

_CONTENT_TYPE_MAP = {
    "news":      "Lead with the biggest moment. Short paragraphs. What happened, why it matters.",
    "analysis":  "State the thesis early. Build evidence. Draw conclusions.",
    "tutorial":  "Problem â†' steps â†' result. Numbered lists. Make it actionable.",
    "opinion":   "Hot take first. Support with facts. Acknowledge counterpoints.",
    "explainer": "Lead with the most interesting or counter-intuitive finding. Build the evidence behind it. Connect to what it means for the reader. Don't follow a rigid template  -  follow the logic of the story.",
    "report":    "Summary â†' findings â†' implications. Use clear subheadings.",
}

_AUDIENCE_MAP = {
    "new_to_topic":   "Your reader just discovered this topic. Start from scratch, no assumptions.",
    "knows_basics":   "Your reader follows this area casually. Skip the obvious stuff.",
    "domain_expert":  "Your reader is deep in this. Get to the insights fast.",
}


def _build_dynamic_prompt(analysis: Dict[str, str]) -> str:
    """
    Assemble system prompt: LOCKED humanization core + DYNAMIC voice layer.
    The voice layer can NEVER override the humanization rules.
    """
    tone = analysis.get("tone", "professional")
    complexity = analysis.get("complexity", "intermediate")
    content_type = analysis.get("content_type", "explainer")
    audience_fam = analysis.get("audience_familiarity", "knows_basics")

    # 'beginner' classifies the KB source content, not our audience.
    # Always write at Grade 7+ minimum  -  clamp before map lookup.
    if complexity == "beginner":
        complexity = "intermediate"
    voice_hint = analysis.get("voice_hint", "someone knowledgeable sharing what they found")

    voice_section = f"""
VOICE FOR THIS ARTICLE (adapt subject matter knowledge  -  the structural rules above never change):
You are {voice_hint}. You've studied the data and you're sharing what you found.
Tone: {_TONE_MAP.get(tone, _TONE_MAP["professional"])}
Vocabulary: {_COMPLEXITY_MAP.get(complexity, _COMPLEXITY_MAP["intermediate"])}
Structure: {_CONTENT_TYPE_MAP.get(content_type, _CONTENT_TYPE_MAP["explainer"])}
Reader: {_AUDIENCE_MAP.get(audience_fam, _AUDIENCE_MAP["knows_basics"])}

AUDIENCE: Informed adult readers who came looking for real answers. They are not domain experts but they are not beginners  -  they read, they research, they have seen generic content before and they dismiss it fast. Match the depth of the topic: a fintech article gets financial-world readers, a health article gets people who've already Googled the basics, a sports article gets fans who know the game. Be specific, be accurate, respect their intelligence.

CRITICAL: The structural and editorial rules above ALWAYS override voice instructions.
Regardless of tone classification, you STILL use wildly varied sentence lengths,
short declarative punches, colon reveals, temporal contrast, and all structural
anti-detection patterns listed above. The voice layer tells you WHAT to write  -  the
structural rules tell you HOW every sentence is constructed."""

    return HUMANIZATION_CORE + "\n" + voice_section


# Keep backward compatibility  -  static prompt for fallback / Groq path
SYSTEM_PROMPT = _build_dynamic_prompt({
    "tone": "professional",
    "complexity": "intermediate",
    "content_type": "explainer",
    "audience_familiarity": "knows_basics",
    "voice_hint": "someone knowledgeable sharing what they found",
})


# ============================================================
# Agentic draft pipeline (5-agent flow)
# 1) user intent/tone -> 2) query expansion -> 3) hybrid retrieval
# 4) source tone -> 5) grounded compose + self-critique
# ============================================================
def _usage_from_resp(resp: Any) -> Dict[str, int]:
    u = getattr(resp, "usage", None)
    return {
        "prompt_tokens": int(getattr(u, "prompt_tokens", 0) or 0) if u else 0,
        "output_tokens": int(getattr(u, "completion_tokens", 0) or 0) if u else 0,
        "total_tokens": int(getattr(u, "total_tokens", 0) or 0) if u else 0,
    }


def _sum_usage(*items: Dict[str, int]) -> Dict[str, int]:
    out = {"prompt_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    for it in items:
        if not isinstance(it, dict):
            continue
        out["prompt_tokens"] += int(it.get("prompt_tokens") or 0)
        out["output_tokens"] += int(it.get("output_tokens") or 0)
        out["total_tokens"] += int(it.get("total_tokens") or 0)
    return out


def _json_from_model_text(raw: str) -> Optional[Dict[str, Any]]:
    txt = (raw or "").strip()
    if not txt:
        return None
    txt = re.sub(r"^```json\s*", "", txt, flags=re.IGNORECASE)
    txt = re.sub(r"^```\s*", "", txt)
    txt = re.sub(r"\s*```$", "", txt)
    obj = _safe_json_loads(txt)
    if obj is None:
        obj = _safe_json_loads(_extract_json_object(txt) or "")
    return obj if isinstance(obj, dict) else None


def _agent_understand_user_question(
    client: OpenAI,
    model: str,
    title: str,
    keywords: List[str],
    length_target: int,
    *,
    temperature: float = 0.2,
) -> Tuple[Dict[str, Any], Dict[str, int]]:
    prompt = f"""Agent 1 task: understand user goal and writing tone for a blog request.
Return only JSON.

Required JSON schema:
{{
  "intent_summary": "one sentence",
  "user_tone": "professional|journalistic|educational|analytical|opinion",
  "target_reader": "one short phrase",
  "risk_notes": ["array of short strings"],
  "must_include_terms": ["array of strings"]
}}

INPUT:
title: {title}
keywords: {", ".join(keywords or [])}
length_target: {length_target}
"""
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a strict JSON planner. No markdown. No prose outside JSON."},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            max_completion_tokens=256,
        )
        raw = (resp.choices[0].message.content or "").strip()
        parsed = _json_from_model_text(raw) or {}
        if not parsed:
            parsed = {
                "intent_summary": f"Write a grounded article about {title}",
                "user_tone": "professional",
                "target_reader": "informed general readers",
                "risk_notes": ["avoid unsupported claims", "avoid generic filler"],
                "must_include_terms": keywords or [],
            }
        return parsed, _usage_from_resp(resp)
    except Exception:
        return {
            "intent_summary": f"Write a grounded article about {title}",
            "user_tone": "professional",
            "target_reader": "informed general readers",
            "risk_notes": ["avoid unsupported claims", "avoid generic filler"],
            "must_include_terms": keywords or [],
        }, {"prompt_tokens": 0, "output_tokens": 0, "total_tokens": 0}


def _agent_expand_queries(
    client: OpenAI,
    model: str,
    title: str,
    keywords: List[str],
    user_intent: Dict[str, Any],
    *,
    temperature: float = 0.2,
) -> Tuple[Dict[str, Any], Dict[str, int]]:
    prompt = f"""Agent 2 task: create 4 retrieval queries that keep the same semantic intent.
Return only JSON.

Schema:
{{
  "queries": ["q1", "q2", "q3", "q4"],
  "notes": "one sentence"
}}

Constraints:
- Queries must stay grounded to the same topic; no tangents.
- Mix lexical and semantic phrasing for hybrid retrieval.
- Include named entities and measurable terms when possible.

TITLE: {title}
KEYWORDS: {", ".join(keywords or [])}
INTENT: {json.dumps(user_intent, ensure_ascii=False)}
"""
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You output JSON only."},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            max_completion_tokens=256,
        )
        raw = (resp.choices[0].message.content or "").strip()
        parsed = _json_from_model_text(raw) or {}
        queries = parsed.get("queries") if isinstance(parsed.get("queries"), list) else []
        cleaned = []
        seen = set()
        for q in queries:
            qq = str(q or "").strip()
            if not qq:
                continue
            k = qq.lower()
            if k in seen:
                continue
            seen.add(k)
            cleaned.append(qq)
        if not cleaned:
            base = [title, f"{title} {' '.join(keywords[:3])}".strip(), " ".join((keywords or [])[:6])]
            cleaned = [q for q in base if q.strip()]
        return {"queries": cleaned[:4], "notes": str(parsed.get("notes") or "")}, _usage_from_resp(resp)
    except Exception:
        base = [title, f"{title} {' '.join(keywords[:3])}".strip(), " ".join((keywords or [])[:6])]
        cleaned = [q for q in base if q.strip()]
        return {"queries": cleaned[:4], "notes": "fallback query expansion"}, {"prompt_tokens": 0, "output_tokens": 0, "total_tokens": 0}


_LEXICAL_STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "from", "into", "about", "over", "under",
    "are", "was", "were", "been", "have", "has", "had", "will", "would", "could", "should",
    "you", "your", "their", "they", "them", "our", "ours", "his", "her", "its", "a", "an",
    "of", "to", "in", "on", "as", "by", "or", "at", "be", "is", "it", "not", "than", "then",
}


def _lex_tokens(text: str) -> List[str]:
    toks = re.findall(r"[a-z0-9]{2,}", (text or "").lower())
    return [t for t in toks if t not in _LEXICAL_STOPWORDS]


def _normalize(values: List[float]) -> List[float]:
    if not values:
        return []
    lo = min(values)
    hi = max(values)
    if hi <= lo:
        return [1.0 for _ in values]
    return [(v - lo) / (hi - lo) for v in values]


def _bm25_scores(query_tokens: List[str], docs_tokens: List[List[str]]) -> List[float]:
    if not query_tokens or not docs_tokens:
        return [0.0 for _ in docs_tokens]
    n = len(docs_tokens)
    df: Counter = Counter()
    for toks in docs_tokens:
        for t in set(toks):
            df[t] += 1
    avgdl = sum(len(toks) for toks in docs_tokens) / max(1, n)
    k1 = 1.2
    b = 0.75
    q_counts = Counter(query_tokens)

    scores: List[float] = []
    for toks in docs_tokens:
        tf = Counter(toks)
        dl = len(toks)
        s = 0.0
        for term, qf in q_counts.items():
            if term not in tf:
                continue
            term_df = int(df.get(term, 0))
            idf = math.log(1.0 + ((n - term_df + 0.5) / (term_df + 0.5)))
            f = float(tf[term])
            denom = f + k1 * (1.0 - b + b * (dl / max(1.0, avgdl)))
            s += idf * ((f * (k1 + 1.0)) / max(1e-9, denom)) * float(qf)
        scores.append(float(s))
    return scores


def _agent_retrieve_hybrid_sources(
    conn,
    gcs: storage.Client,
    *,
    tenant_id: str,
    kb_id: str,
    queries: List[str],
    embedding_api_key: str,
    embedding_model: str,
    output_dimensionality: int,
    top_k_final: int,
    top_k_per_query: int = 30,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    doc_chunks_uri = _latest_chunks_uri_by_doc(conn, tenant_id, kb_id)
    if not doc_chunks_uri:
        return [], {"query_count": 0, "candidate_count": 0, "strategy": "hybrid_semantic_bm25"}

    candidates: Dict[Tuple[str, str], Dict[str, Any]] = {}
    rrf_k = 60.0
    effective_queries = [q.strip() for q in queries if q and q.strip()]

    for q in effective_queries:
        try:
            q_vec = _embed_query(embedding_api_key, embedding_model, q, int(output_dimensionality))
            q_literal = _to_vector_literal(q_vec)
        except Exception:
            continue

        rows = _vector_top_chunks(
            conn,
            tenant_id=tenant_id,
            kb_id=kb_id,
            q_vec_literal=q_literal,
            dim=int(output_dimensionality),
            embedding_model=embedding_model,
            top_k=int(top_k_per_query),
        )
        for rank, (doc_id, chunk_id, dist) in enumerate(rows, start=1):
            key = (doc_id, chunk_id)
            rec = candidates.get(key)
            if rec is None:
                candidates[key] = {
                    "doc_id": doc_id,
                    "chunk_id": chunk_id,
                    "distance": float(dist),
                    "best_rank": int(rank),
                    "rrf": 1.0 / (rrf_k + float(rank)),
                }
            else:
                rec["distance"] = min(float(rec["distance"]), float(dist))
                rec["best_rank"] = min(int(rec["best_rank"]), int(rank))
                rec["rrf"] = float(rec["rrf"]) + (1.0 / (rrf_k + float(rank)))

    if not candidates:
        return [], {"query_count": len(effective_queries), "candidate_count": 0, "strategy": "hybrid_semantic_bm25"}

    chunks_cache: Dict[str, Dict[str, str]] = {}
    keys = list(candidates.keys())
    texts: List[str] = []
    for doc_id, chunk_id in keys:
        uri = doc_chunks_uri.get(doc_id)
        txt = ""
        if uri:
            if doc_id not in chunks_cache:
                try:
                    chunks_cache[doc_id] = _parse_chunks_jsonl_to_map(_gcs_download_bytes(gcs, uri))
                except Exception:
                    chunks_cache[doc_id] = {}
            txt = chunks_cache[doc_id].get(chunk_id, "")
        candidates[(doc_id, chunk_id)]["text"] = txt
        texts.append(txt or "")

    query_tokens = _lex_tokens(" ".join(effective_queries))
    docs_tokens = [_lex_tokens(t) for t in texts]
    bm25_raw = _bm25_scores(query_tokens, docs_tokens)
    semantic_raw = [1.0 / (1.0 + float(candidates[k]["distance"])) for k in keys]
    rrf_raw = [float(candidates[k]["rrf"]) for k in keys]

    bm25_n = _normalize(bm25_raw)
    sem_n = _normalize(semantic_raw)
    rrf_n = _normalize(rrf_raw)

    results: List[Dict[str, Any]] = []
    for idx, key in enumerate(keys):
        rec = candidates[key]
        txt = str(rec.get("text") or "").strip()
        if not txt:
            continue
        hybrid = 0.55 * sem_n[idx] + 0.35 * bm25_n[idx] + 0.10 * rrf_n[idx]
        results.append(
            {
                "doc_id": rec["doc_id"],
                "chunk_id": rec["chunk_id"],
                "distance": float(rec["distance"]),
                "semantic_score": float(sem_n[idx]),
                "bm25_score": float(bm25_n[idx]),
                "rrf_score": float(rrf_n[idx]),
                "hybrid_score": float(hybrid),
                "text": _compact_source_text(txt, max_chars=3500),
            }
        )

    results.sort(key=lambda x: x.get("hybrid_score", 0.0), reverse=True)
    # Drop low-quality matches: keep only chunks scoring above 50% of the top chunk's score.
    # Always keep at least min_keep chunks so the Evidence Locker has enough facts to write from.
    # min_keep scales with top_k_final: short article (<=8) needs 5, long (>8) needs 8.
    if results:
        min_keep = 8 if int(top_k_final) > 8 else 5
        top_score = results[0].get("hybrid_score", 1.0)
        threshold = max(0.30, top_score * 0.50)
        filtered = [r for r in results if r.get("hybrid_score", 0.0) >= threshold]
        results = filtered if len(filtered) >= min_keep else results[:min_keep]
    final = results[: max(1, int(top_k_final))]
    return final, {
        "query_count": len(effective_queries),
        "candidate_count": len(results),
        "strategy": "hybrid_semantic_bm25_rrf",
        "weights": {"semantic": 0.55, "bm25": 0.35, "rrf": 0.10},
    }


def _agent_analyze_retrieved_tone(
    client: OpenAI,
    model: str,
    sources_block: str,
    *,
    temperature: float = 0.2,
) -> Tuple[Dict[str, Any], Dict[str, int]]:
    prompt = """Agent 4 task: analyze retrieved material tone and structure.
Return only JSON.

Schema:
{
  "source_tone": "professional|journalistic|academic|mixed",
  "formality": "low|medium|high",
  "density": "light|balanced|dense",
  "dominant_format": "reporting|analysis|explainer|mixed",
  "recommended_voice_hint": "one sentence"
}

SOURCE EXCERPTS:
""" + sources_block[:5000]
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You analyze text and return strict JSON only."},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            max_completion_tokens=256,
        )
        raw = (resp.choices[0].message.content or "").strip()
        parsed = _json_from_model_text(raw) or {}
        if not parsed:
            parsed = {
                "source_tone": "professional",
                "formality": "high",
                "density": "balanced",
                "dominant_format": "analysis",
                "recommended_voice_hint": "Write like an editorial analyst grounded in evidence.",
            }
        return parsed, _usage_from_resp(resp)
    except Exception:
        return {
            "source_tone": "professional",
            "formality": "high",
            "density": "balanced",
            "dominant_format": "analysis",
            "recommended_voice_hint": "Write like an editorial analyst grounded in evidence.",
        }, {"prompt_tokens": 0, "output_tokens": 0, "total_tokens": 0}


def _sentence_spans(text: str) -> List[Tuple[int, int, str]]:
    spans: List[Tuple[int, int, str]] = []
    if not text:
        return spans
    pattern = re.compile(r"[^.!?\n]+(?:[.!?]+|$)", re.MULTILINE)
    for m in pattern.finditer(text):
        s = m.group(0).strip()
        if not s:
            continue
        if len(s.split()) < 4:
            continue
        spans.append((m.start(), m.end(), s))
    return spans


def _sanitize_generated_markdown(text: str) -> str:
    t = text or ""
    # Remove source-style inline tags like [S1], [S12] leaking from planning prompts.
    t = re.sub(r"\s*\[S\d+\]", "", t)
    t = re.sub(r"[ \t]+\n", "\n", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def _build_fact_block(facts: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    for f in facts:
        fact_id = str(f.get("fact_id") or "")
        claim = str(f.get("claim") or "")
        source_id = str(f.get("source_id") or "")
        conf = str(f.get("confidence") or "medium")
        if not fact_id or not claim:
            continue
        lines.append(f"- {fact_id} [{source_id}] ({conf}): {claim}")
    return "\n".join(lines)


def _fallback_facts_from_sources(sources: List[Dict[str, Any]], limit: int = 30) -> List[Dict[str, Any]]:
    facts: List[Dict[str, Any]] = []
    idx = 1
    for i, src in enumerate(sources, start=1):
        source_id = f"S{i}"
        txt = str(src.get("text") or "")
        for _, _, sent in _sentence_spans(txt):
            claim = sent.strip()
            if len(claim.split()) < 7:
                continue
            facts.append(
                {
                    "fact_id": f"F{idx}",
                    "source_id": source_id,
                    "claim": claim,
                    "confidence": "medium",
                }
            )
            idx += 1
            if len(facts) >= limit:
                return facts
    return facts


def _agent_build_evidence_locker(
    client: OpenAI,
    model: str,
    sources: List[Dict[str, Any]],
    *,
    max_facts: int = 36,
    temperature: float = 0.1,
) -> Tuple[List[Dict[str, Any]], Dict[str, int], Dict[str, Any]]:
    labeled_sources: List[Dict[str, Any]] = []
    lines: List[str] = []
    for i, s in enumerate(sources, start=1):
        sid = f"S{i}"
        labeled_sources.append(
            {
                "source_id": sid,
                "doc_id": s.get("doc_id"),
                "chunk_id": s.get("chunk_id"),
            }
        )
        lines.append(f"[{sid}] doc_id={s.get('doc_id')} chunk_id={s.get('chunk_id')}\n{s.get('text','')}\n")

    prompt = f"""Build an Evidence Locker from retrieved source excerpts.
Return strict JSON only.

Schema:
{{
  "facts": [
    {{
      "fact_id": "F1",
      "source_id": "S1",
      "claim": "one verifiable claim from that source",
      "confidence": "high|medium|low"
    }}
  ],
  "coverage_notes": "one short sentence"
}}

Rules:
- Extract claims only from provided excerpts.
- No synthesis across sources in one claim.
- No forecasts, no assumptions, no outside knowledge.
- Keep each claim <= 28 words.
- Return up to {int(max_facts)} facts.

RETRIEVED SOURCES:
{''.join(lines)[:24000]}
"""
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a strict evidence extractor. Output JSON only."},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            max_completion_tokens=1800,
        )
        raw = (resp.choices[0].message.content or "").strip()
        parsed = _json_from_model_text(raw) or {}
        facts_raw = parsed.get("facts") if isinstance(parsed.get("facts"), list) else []
        facts: List[Dict[str, Any]] = []
        source_ids = {x["source_id"] for x in labeled_sources}
        for item in facts_raw:
            if not isinstance(item, dict):
                continue
            claim = str(item.get("claim") or "").strip()
            sid = str(item.get("source_id") or "").strip()
            if not claim or sid not in source_ids:
                continue
            facts.append(
                {
                    "fact_id": str(item.get("fact_id") or f"F{len(facts)+1}"),
                    "source_id": sid,
                    "claim": claim,
                    "confidence": str(item.get("confidence") or "medium"),
                }
            )
            if len(facts) >= int(max_facts):
                break
        if not facts:
            facts = _fallback_facts_from_sources(sources, limit=max_facts)
        return facts, _usage_from_resp(resp), {
            "coverage_notes": str(parsed.get("coverage_notes") or ""),
            "source_ids": labeled_sources,
        }
    except Exception:
        return _fallback_facts_from_sources(sources, limit=max_facts), {
            "prompt_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
        }, {"coverage_notes": "fallback facts", "source_ids": labeled_sources}


_RISK_PHRASES = [
    "it is important to note",
    "in conclusion",
    "this highlights",
    "underscores the",
    "plays a crucial role",
    "it should be noted",
]


def _predictability_risk_report(markdown: str, top_n: int = 16) -> Dict[str, Any]:
    spans = _sentence_spans(markdown)
    if not spans:
        return {"average_score": 0.0, "items": []}

    lengths = [len(_lex_tokens(s)) for _, _, s in spans]
    sorted_l = sorted(lengths)
    mid = len(sorted_l) // 2
    median_len = float(sorted_l[mid]) if sorted_l else 0.0
    start_counts: Counter = Counter()
    for _, _, s in spans:
        toks = _lex_tokens(s)
        if toks:
            start_counts[toks[0]] += 1

    items: List[Dict[str, Any]] = []
    for idx, (start, end, sentence) in enumerate(spans, start=1):
        reasons: List[str] = []
        toks = _lex_tokens(sentence)
        n = len(toks)
        score = 0.0
        if n >= 8 and n <= 20:
            score += 0.45
            reasons.append("mid_length_band")
        if abs(n - median_len) <= 2:
            score += 0.35
            reasons.append("uniform_length")
        first = toks[0] if toks else ""
        if first and start_counts[first] >= 3:
            score += 0.45
            reasons.append("repeated_starter")
        low = sentence.lower()
        if any(p in low for p in _RISK_PHRASES):
            score += 0.9
            reasons.append("template_phrase")
        if re.search(r"\[S\d+\]", sentence):
            score += 1.2
            reasons.append("citation_artifact")
        unique_ratio = (len(set(toks)) / n) if n else 1.0
        if n >= 10 and unique_ratio < 0.5:
            score += 0.5
            reasons.append("low_lexical_diversity")
        if n >= 14 and sentence.count(",") == 0 and sentence.count(" - ") == 0 and sentence.count(";") == 0:
            score += 0.25
            reasons.append("flat_clause_shape")
        items.append(
            {
                "index": idx,
                "start": start,
                "end": end,
                "sentence": sentence,
                "score": round(score, 3),
                "reasons": reasons,
            }
        )

    items.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)
    avg = sum(float(i["score"]) for i in items) / max(1, len(items))
    return {"average_score": round(avg, 3), "items": items[: max(1, int(top_n))]}


def _estimate_grounding_ratio(markdown: str, facts: List[Dict[str, Any]]) -> float:
    fact_sets: List[set] = []
    fact_nums: List[set] = []
    for f in facts:
        claim = str(f.get("claim") or "")
        fact_sets.append(set(_lex_tokens(claim)))
        fact_nums.append(set(re.findall(r"\d+(?:\.\d+)?", claim)))

    spans = _sentence_spans(markdown)
    eligible = 0
    grounded = 0
    for _, _, sentence in spans:
        toks = set(_lex_tokens(sentence))
        if len(toks) < 5:
            continue
        eligible += 1
        nums = set(re.findall(r"\d+(?:\.\d+)?", sentence))
        best_overlap = 0.0
        num_match = False
        for fs, fn in zip(fact_sets, fact_nums):
            if not fs:
                continue
            overlap = len(toks & fs) / max(1, len(toks))
            best_overlap = max(best_overlap, overlap)
            if nums and fn and (nums & fn):
                num_match = True
        if best_overlap >= 0.28 or num_match:
            grounded += 1

    if eligible == 0:
        return 0.0
    return grounded / eligible


def _apply_sentence_rewrites(markdown: str, rewrites: List[Tuple[str, str]]) -> str:
    out = markdown
    for old, new in rewrites:
        if not old or not new:
            continue
        old_s = old.strip()
        new_s = new.strip()
        if not old_s or not new_s or old_s == new_s:
            continue
        pos = out.find(old_s)
        if pos >= 0:
            out = out[:pos] + new_s + out[pos + len(old_s):]
    return out


def _agent_rewrite_risky_sentences(
    client: OpenAI,
    model: str,
    markdown: str,
    risk_items: List[Dict[str, Any]],
    fact_block: str,
    *,
    temperature: float = 0.35,
    max_completion_tokens: int = 1200,
) -> Tuple[str, Dict[str, int], Dict[str, Any]]:
    if not risk_items:
        return markdown, {"prompt_tokens": 0, "output_tokens": 0, "total_tokens": 0}, {"rewritten_count": 0}

    numbered = []
    for i, it in enumerate(risk_items, start=1):
        numbered.append(f'{i}. "{it.get("sentence","")}"')

    prompt = f"""Rewrite only the listed sentences.
Return JSON only.

Schema:
{{
  "rewrites": [
    {{
      "item_index": 1,
      "rewrite": "replacement sentence"
    }}
  ]
}}

Rules:
- Keep meaning faithful to evidence facts.
- Do not add outside facts.
- Remove any citation tags like [S1].
- Reduce predictable rhythm (vary structure and cadence).
- Keep each rewrite approximately same length (within +/-25% words).

Evidence facts:
{fact_block[:12000]}

Risk sentences:
{chr(10).join(numbered)}
"""
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a precision editor. Output JSON only."},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            max_completion_tokens=max_completion_tokens,
        )
        raw = (resp.choices[0].message.content or "").strip()
        parsed = _json_from_model_text(raw) or {}
        rw_raw = parsed.get("rewrites") if isinstance(parsed.get("rewrites"), list) else []
        pairs: List[Tuple[str, str]] = []
        for obj in rw_raw:
            if not isinstance(obj, dict):
                continue
            try:
                idx = int(obj.get("item_index"))
            except Exception:
                continue
            if idx < 1 or idx > len(risk_items):
                continue
            old = str(risk_items[idx - 1].get("sentence") or "")
            new = str(obj.get("rewrite") or "")
            if old and new:
                pairs.append((old, new))
        rewritten = _apply_sentence_rewrites(markdown, pairs)
        rewritten = _sanitize_generated_markdown(rewritten)
        return rewritten, _usage_from_resp(resp), {
            "rewritten_count": len(pairs),
            "requested_count": len(risk_items),
        }
    except Exception:
        return markdown, {"prompt_tokens": 0, "output_tokens": 0, "total_tokens": 0}, {"rewritten_count": 0}


def _agent_compose_grounded_article(
    client: OpenAI,
    model: str,
    *,
    title: str,
    keywords: List[str],
    length_target: int,
    sources: List[Dict[str, Any]],
    system_prompt: str,
    user_profile: Dict[str, Any],
    source_tone: Dict[str, Any],
    temperature: float,
    max_tokens: int,
    grounding_ratio: float = 0.95,
    top_risky_sentences: int = 14,
    max_risk_rewrite_passes: int = 1,
) -> Tuple[Dict[str, Any], Dict[str, int], Dict[str, Any]]:
    kw = ", ".join(keywords or [])
    target_words = max(1200, min(2200, int(length_target or 1800)))

    facts, locker_usage, locker_meta = _agent_build_evidence_locker(
        client=client,
        model=model,
        sources=sources,
    )
    facts_block = _build_fact_block(facts)
    if not facts_block:
        facts_block = "- F1 [S1] (low): Specific evidence extraction failed; use only directly observable details."

    few_shot_block = """Few-shot grounding examples:
BAD:
- The sector will grow 40% next year.
GOOD:
- Retrieved evidence does not include a forward growth forecast.

BAD:
- Analysts worldwide agree this is a turning point.
GOOD:
- Retrieved material presents strong indicators, but cross-market consensus is not provided."""

    planning_prompt = f"""Agent 5A task: build a blog-style outline from Evidence Locker facts only.
Think step-by-step privately. Do not reveal your reasoning.
Return markdown outline only.

Topic: {title}
Keywords: {kw}
Target words: {target_words}
User profile: {json.dumps(user_profile, ensure_ascii=False)}
Source tone profile: {json.dumps(source_tone, ensure_ascii=False)}

BLOG STRUCTURE REQUIRED  -  outline must follow this arc:
1. [HOOK  -  no heading, 150-200 words]: opening hook paragraph(s)  -  specific moment or tension. List 2-3 facts from Evidence Locker to anchor it.
2. [Section 1  -  story-driven heading]: First chapter of the narrative. What happened first, or the foundation of the story.
3. [Section 2-4  -  story-driven headings]: Develop the story. Cause -> development -> consequence per section.
4. [Section 5  -  insight/opinion]: Editorial perspective. Why this matters. What it reveals.
5. [Section 6  -  conclusion]: Wrap the story cleanly. No "In conclusion". End with a forward-looking observation.
6. [## FAQ]: 5-7 real reader questions answered concisely.

Rules:
- Every section must reference specific fact_ids from Evidence Locker.
- No external facts. No citations in final article body.
- Create 6-9 sections MAXIMUM. Do NOT create one section per statistic.
- Group related stats into ONE analytical section each covering 150+ words.
- Headings must be story-driven ("The Season That Changed Everything"), never label-style ("Career Statistics", "Player Profile Snapshot").
- Each section must have enough evidence to support 150-200 words of flowing analytical prose.

Evidence Locker:
{facts_block}

{few_shot_block}
"""
    outline_resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": planning_prompt},
        ],
        temperature=max(0.1, float(temperature) * 0.6),
        max_completion_tokens=900,
    )
    outline = (outline_resp.choices[0].message.content or "").strip()
    outline_usage = _usage_from_resp(outline_resp)

    drafting_prompt = f"""Agent 5B task: write the final blog article from Evidence Locker facts.
Think privately. Do not reveal reasoning.
Output markdown only  -  the article, nothing else.

Follow the article structure rules in the system prompt exactly  -  narrative arc for stories, structured arc for tutorials.
Mix facts with meaning: "973 runs" is dead copy; "973 runs that carried RCB to their only final" is alive.
Use specific, informative headings. Never label-style ("Career Statistics", "Player Profile", "More Info").
End with a ## FAQ section.

Hard constraints:
- At least {int(round(grounding_ratio * 100))}% of factual statements must map to Evidence Locker facts.
- No external facts, years, names, metrics, or forecasts unless present in facts.
- Do not output [Sx] or [Fx] citation tags in article text.
- Target total words: {target_words + 100} (minimum {max(1200, target_words - 100)}, maximum {target_words + 300}). Err toward MORE content, not less.
- Write FULLY developed prose  -  minimum 150 words per section. Bullet lists are for data points or step sequences only; a section must never be bullets alone.
- Do NOT create one section per statistic  -  synthesize related facts into analytical narrative sections.
- Vary sentence lengths and avoid repetitive template phrasing.

Outline:
{outline}

Evidence Locker:
{facts_block}

{few_shot_block}
"""
    draft_resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": drafting_prompt},
        ],
        temperature=float(temperature),
        max_completion_tokens=max_tokens,
    )
    draft_text = _sanitize_generated_markdown((draft_resp.choices[0].message.content or "").strip())
    draft_usage = _usage_from_resp(draft_resp)

    critique_prompt = f"""Agent 5C task: revise for strict grounding and clarity.
Think privately. Return JSON only.

Schema:
{{
  "unsupported_claims": ["up to 10 short snippets"],
  "revised_markdown": "full revised markdown"
}}

Rules:
- Rewrite/remove unsupported claims.
- Keep professional tone and readability.
- Remove residual citation artifacts [S1], [F3], etc.
- Keep meaning tied to Evidence Locker only.

Evidence Locker:
{facts_block}

Draft:
{draft_text}
"""
    critique_resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a strict grounding auditor. Output JSON only."},
            {"role": "user", "content": critique_prompt},
        ],
        temperature=0.15,
        max_completion_tokens=max_tokens,
    )
    critique_usage = _usage_from_resp(critique_resp)
    critique_json = _json_from_model_text((critique_resp.choices[0].message.content or "").strip()) or {}

    revised = str(critique_json.get("revised_markdown") or "").strip() or draft_text
    revised = _sanitize_generated_markdown(revised)
    unsupported = critique_json.get("unsupported_claims")
    if not isinstance(unsupported, list):
        unsupported = []

    # Deterministic post-checks before saving.
    local_grounding_ratio = _estimate_grounding_ratio(revised, facts)
    risk_before = _predictability_risk_report(revised, top_n=top_risky_sentences)
    rewrite_usage_total = {"prompt_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    risk_rewrite_meta: Dict[str, Any] = {"passes": 0, "rewritten_count": 0}

    current_text = revised
    current_risk = risk_before
    for _ in range(max(0, int(max_risk_rewrite_passes))):
        items = [x for x in current_risk.get("items", []) if float(x.get("score", 0.0)) >= 1.05]
        if not items:
            break
        current_text, rewrite_usage, rw_meta = _agent_rewrite_risky_sentences(
            client=client,
            model=model,
            markdown=current_text,
            risk_items=items,
            fact_block=facts_block,
            temperature=max(0.25, float(temperature) * 0.65),
            max_completion_tokens=min(1600, max_tokens // 2),
        )
        rewrite_usage_total = _sum_usage(rewrite_usage_total, rewrite_usage)
        risk_rewrite_meta["passes"] = int(risk_rewrite_meta.get("passes", 0)) + 1
        risk_rewrite_meta["rewritten_count"] = int(risk_rewrite_meta.get("rewritten_count", 0)) + int(rw_meta.get("rewritten_count", 0))
        current_risk = _predictability_risk_report(current_text, top_n=top_risky_sentences)

    final_text = _sanitize_generated_markdown(current_text)
    final_grounding_ratio = _estimate_grounding_ratio(final_text, facts)

    draft_json = {
        "title": title,
        "draft_markdown": final_text,
        "used_chunks": [{"doc_id": s.get("doc_id"), "chunk_id": s.get("chunk_id")} for s in sources],
    }
    meta = {
        "outline_markdown": outline,
        "unsupported_claim_count": len(unsupported),
        "unsupported_claims": unsupported[:10],
        "grounding_ratio_estimate_model": float(critique_json.get("grounding_ratio_estimate", 0.0) or 0.0) if isinstance(critique_json, dict) else 0.0,
        "grounding_ratio_estimate_local": final_grounding_ratio,
        "target_grounding_ratio": grounding_ratio,
        "evidence_fact_count": len(facts),
        "evidence_locker": locker_meta,
        "predictability_risk_before": risk_before.get("average_score", 0.0),
        "predictability_risk_after": current_risk.get("average_score", 0.0),
        "predictability_top_items_after": current_risk.get("items", []),
        "risk_rewrite": risk_rewrite_meta,
    }
    return draft_json, _sum_usage(outline_usage, draft_usage, critique_usage, locker_usage, rewrite_usage_total), meta


# ============================================================
# Settings loader (same pattern)
# ============================================================
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
        n_lower = n.lower()
        if hasattr(settings, n_lower) and getattr(settings, n_lower) not in (None, ""):
            return getattr(settings, n_lower)
        env_val = os.getenv(n)
        if env_val not in (None, ""):
            return env_val
    return default


def _compact_source_text(text: str, max_chars: int = 3500) -> str:
    """
    Token-cost control:
    - normalize whitespace
    - keep full head (no middle cut) to preserve context
    """
    t = (text or "").strip()
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    if len(t) <= max_chars:
        return t
    # Keep the first max_chars chars  -  head contains the most structured info
    return t[:max_chars].rstrip() + "\n..."


# ============================================================
# Auth
# ============================================================
class Claims(BaseModel):
    tenant_id: str
    user_id: str
    role: str
    exp: int


def require_claims(
    authorization: str = Header(..., description="Bearer <JWT>"),
) -> Claims:
    if not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Missing Bearer token")

    token = authorization.split(" ", 1)[1].strip()
    secret = os.getenv("JWT_SECRET_KEY") or os.getenv("JWT_SECRET") or os.getenv("SECRET_KEY")
    alg = os.getenv("JWT_ALGORITHM") or "HS256"
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
    if not tenant_id or not user_id or not role or not exp:
        raise HTTPException(status_code=401, detail="Token missing required claims")

    return Claims(tenant_id=str(tenant_id), user_id=str(user_id), role=str(role), exp=int(exp))


def _require_admin(claims: Claims) -> None:
    # Plan says /run is admin manual trigger
    # We treat tenant_admin as admin
    if (claims.role or "").lower() != "tenant_admin":
        raise HTTPException(status_code=403, detail="Only tenant_admin can run article generation")


# ============================================================
# DB helpers (Supabase Postgres via DB_*)
# ============================================================
def _db_conn(settings: Any):
    host = _pick(settings, "DB_HOST", "SUPABASE_DB_HOST", "POSTGRES_HOST")
    port = int(_pick(settings, "DB_PORT", "SUPABASE_DB_PORT", "POSTGRES_PORT", default=5432))
    dbname = _pick(settings, "DB_NAME", "SUPABASE_DB_NAME", "POSTGRES_DB", default="postgres")
    user = _pick(settings, "DB_USER", "SUPABASE_DB_USER", "POSTGRES_USER")
    password = _pick(settings, "DB_PASSWORD", "SUPABASE_DB_PASSWORD", "POSTGRES_PASSWORD")
    sslmode = _pick(settings, "DB_SSLMODE", "SUPABASE_DB_SSLMODE", "POSTGRES_SSLMODE", default="require")

    missing = [k for k, v in [("host", host), ("user", user), ("password", password)] if not v]
    if missing:
        raise RuntimeError(f"Missing DB settings: {', '.join(missing)}")

    return psycopg2.connect(
        host=host,
        port=port,
        dbname=dbname,
        user=user,
        password=password,
        sslmode=sslmode,
        connect_timeout=8,
    )


def _log_job_event(conn, tenant_id: str, event_type: str, detail: Dict[str, Any], job_id: Optional[str] = None) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO public.job_events (event_id, tenant_id, request_id, job_id, event_type, detail)
            VALUES (%s, %s, %s, %s, %s, %s)
            """,
            (str(uuid.uuid4()), tenant_id, None, job_id, event_type, json.dumps(detail)),
        )


def _ensure_day17_columns(conn) -> None:
    """
    We don't assume you altered article_requests yet.
    We add the minimum columns needed to store draft output pointers.
    """
    with conn.cursor() as cur:
        # add columns if missing (safe)
        cur.execute("ALTER TABLE public.article_requests ADD COLUMN IF NOT EXISTS gcs_draft_uri text;")
        cur.execute("ALTER TABLE public.article_requests ADD COLUMN IF NOT EXISTS draft_fingerprint text;")
        cur.execute("ALTER TABLE public.article_requests ADD COLUMN IF NOT EXISTS draft_model text;")
        cur.execute("ALTER TABLE public.article_requests ADD COLUMN IF NOT EXISTS last_run_at timestamptz;")
        cur.execute("ALTER TABLE public.article_requests ADD COLUMN IF NOT EXISTS draft_meta jsonb;")

        # job_locks table expected from Day16 SQL; ensure minimal structure
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS public.job_locks (
              job_id text PRIMARY KEY,
              lock_token text NOT NULL,
              locked_at timestamptz NOT NULL DEFAULT now(),
              expires_at timestamptz NOT NULL DEFAULT (now() + interval '30 minutes')
            );
            """
        )
        cur.execute("CREATE INDEX IF NOT EXISTS idx_job_locks_expires_at ON public.job_locks (expires_at);")


def _try_lock(conn, job_id: str, ttl_minutes: int = 30) -> Tuple[bool, str]:
    """
    Atomic claim:
    - If free: insert and own the lock
    - If exists but expired: steal it (update)
    - Else: fail
    """
    lock_token = str(uuid.uuid4())
    with conn.cursor() as cur:
        # 1) try insert (fast path)
        cur.execute(
            """
            INSERT INTO public.job_locks (job_id, lock_token, locked_at, expires_at)
            VALUES (%s, %s, now(), now() + (%s || ' minutes')::interval)
            ON CONFLICT (job_id) DO NOTHING
            """,
            (job_id, lock_token, int(ttl_minutes)),
        )
        if cur.rowcount == 1:
            return True, lock_token

        # 2) try steal if expired
        cur.execute(
            """
            UPDATE public.job_locks
            SET lock_token=%s, locked_at=now(), expires_at=now() + (%s || ' minutes')::interval
            WHERE job_id=%s AND expires_at < now()
            """,
            (lock_token, int(ttl_minutes), job_id),
        )
        if cur.rowcount == 1:
            return True, lock_token

    return False, lock_token


def _unlock(conn, job_id: str, lock_token: str) -> None:
    with conn.cursor() as cur:
        cur.execute("DELETE FROM public.job_locks WHERE job_id=%s AND lock_token=%s", (job_id, lock_token))


def _usage_event_best_effort(conn, tenant_id: str, request_id: str, payload: Dict[str, Any]) -> None:
    """
    Day17 requires usage_events logging, but schemas vary.
    We introspect the table columns and insert only fields that exist.
    """
    with conn.cursor() as cur:
        # ensure table exists minimally (won't harm if you already have a richer table)
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS public.usage_events (
              event_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
              tenant_id uuid NOT NULL,
              operation_name text NOT NULL,
              tokens integer NOT NULL DEFAULT 0,
              total_cost numeric NOT NULL DEFAULT 0,
              created_at timestamptz NOT NULL DEFAULT now()
            );
            """
        )

        cur.execute(
            """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_schema='public' AND table_name='usage_events'
            """
        )
        cols = {r[0] for r in cur.fetchall()}

        # map common fields (only if column exists)
        row: Dict[str, Any] = {}
        if "tenant_id" in cols:
            row["tenant_id"] = tenant_id
        if "request_id" in cols:
            row["request_id"] = request_id
        if "operation_name" in cols:
            row["operation_name"] = payload.get("operation_name", "article_draft")
        if "vendor" in cols:
            row["vendor"] = payload.get("vendor")
        if "model" in cols:
            row["model"] = payload.get("model")
        if "tokens" in cols:
            row["tokens"] = int(payload.get("tokens") or 0)
        if "total_cost" in cols:
            row["total_cost"] = payload.get("total_cost") or 0
        if "detail" in cols:
            row["detail"] = json.dumps(payload.get("detail") or {})

        # nothing to insert?
        if not row:
            return

        col_list = ", ".join(row.keys())
        placeholders = ", ".join([f"%({k})s" for k in row.keys()])
        cur.execute(f"INSERT INTO public.usage_events ({col_list}) VALUES ({placeholders})", row)


# ============================================================
# GCS helpers
# ============================================================
def _gcs_client(settings: Any) -> storage.Client:
    project_id = _pick(settings, "GCP_PROJECT_ID", "GOOGLE_CLOUD_PROJECT")
    return storage.Client(project=project_id)


def _parse_gs_uri(gs_uri: str) -> Tuple[str, str]:
    if not gs_uri.startswith("gs://"):
        raise ValueError(f"Not a gs:// uri: {gs_uri}")
    rest = gs_uri[len("gs://") :]
    parts = rest.split("/", 1)
    return parts[0], (parts[1] if len(parts) > 1 else "")


def _gcs_download_bytes(client: storage.Client, gs_uri: str) -> bytes:
    bucket_name, obj = _parse_gs_uri(gs_uri)
    blob = client.bucket(bucket_name).blob(obj)
    if not blob.exists():
        raise FileNotFoundError(f"GCS object not found: {gs_uri}")
    return blob.download_as_bytes()


def _gcs_upload_bytes_create_only(
    client: storage.Client,
    bucket_name: str,
    object_name: str,
    data: bytes,
    content_type: str,
) -> str:
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(object_name)
    try:
        blob.upload_from_string(data, content_type=content_type, if_generation_match=0)
    except PreconditionFailed:
        # object already exists -> treat as idempotent reuse
        pass
    return f"gs://{bucket_name}/{object_name}"


def _norm_prefix(p: str) -> str:
    p = (p or "").strip().replace("\\", "/")
    if not p:
        return ""
    return p if p.endswith("/") else (p + "/")


# ============================================================
# Retrieval helpers (Supabase pgvector + chunk text from GCS)
# ============================================================
def _to_vector_literal(v: List[float]) -> str:
    return "[" + ",".join(repr(float(x)) for x in v) + "]"


def _extract_vec(obj: Any) -> List[float]:
    if isinstance(obj, dict):
        # can be {"values":[...]} or {"embedding":[...]}
        v = obj.get("values") or obj.get("embedding")
        if isinstance(v, list):
            return [float(x) for x in v]
    if isinstance(obj, list):
        return [float(x) for x in obj]
    return []


def _embed_query(api_key: str, model: str, text: str, dim: int) -> List[float]:
    genai.configure(api_key=api_key)
    # output_dimensionality is critical (we use 1536)
    res = genai.embed_content(model=model, content=text, output_dimensionality=int(dim))
    if isinstance(res, dict):
        if "embedding" in res:
            v = _extract_vec(res["embedding"])
            if len(v) == dim:
                return v
        if "embeddings" in res and res["embeddings"]:
            v = _extract_vec(res["embeddings"][0])
            if len(v) == dim:
                return v
    raise HTTPException(status_code=502, detail="Embedding failed for retrieval query")


def _latest_chunks_uri_by_doc(conn, tenant_id: str, kb_id: str) -> Dict[str, str]:
    """
    Returns doc_id -> latest gcs_chunks_uri (DISTINCT ON).
    """
    out: Dict[str, str] = {}
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT DISTINCT ON (doc_id) doc_id::text, gcs_chunks_uri
            FROM public.chunks
            WHERE tenant_id=%s::uuid AND kb_id=%s::uuid
            ORDER BY doc_id, created_at DESC
            """,
            (tenant_id, kb_id),
        )
        for doc_id, uri in cur.fetchall():
            if uri:
                out[str(doc_id)] = str(uri)
    return out


def _vector_top_chunks(
    conn,
    tenant_id: str,
    kb_id: str,
    q_vec_literal: str,
    dim: int,
    embedding_model: str,
    top_k: int,
    max_distance: float = 0.50,
) -> List[Tuple[str, str, float]]:
    """
    Returns [(doc_id, chunk_id, distance)] across the KB.
    max_distance: cosine distance ceiling — chunks further than this are off-topic and excluded.
    0.50 means cosine similarity < 0.50, i.e. less than half-similar to the query.
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT doc_id::text, chunk_id::text, (embedding <=> %s::vector) AS distance
            FROM public.chunk_embeddings
            WHERE tenant_id=%s::uuid AND kb_id=%s::uuid
              AND output_dimensionality=%s
              AND embedding_model=%s
              AND (embedding <=> %s::vector) < %s
            ORDER BY embedding <=> %s::vector ASC
            LIMIT %s
            """,
            (q_vec_literal, tenant_id, kb_id, int(dim), embedding_model,
             q_vec_literal, float(max_distance), q_vec_literal, int(top_k)),
        )
        return [(str(r[0]), str(r[1]), float(r[2])) for r in cur.fetchall()]


def _parse_chunks_jsonl_to_map(jsonl_bytes: bytes) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for line in jsonl_bytes.decode("utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        cid = obj.get("chunk_id") or obj.get("id")
        txt = obj.get("text") or obj.get("content")
        if cid and isinstance(txt, str):
            out[str(cid)] = txt
    return out


# ============================================================
# Draft generation (Gemini)
# ============================================================
def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _safe_json_loads(s: str) -> Optional[Dict[str, Any]]:
    s = (s or "").strip()
    if not s:
        return None
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _extract_json_object(s: str) -> Optional[str]:
    if not s:
        return None
    m = re.search(r"\{.*\}", s, flags=re.DOTALL)
    return m.group(0) if m else None


def _generate_draft_json(
    api_key: str,
    model_name: str,
    prompt: str,
    *,
    temperature: float,
    max_output_tokens: int,
    timeout_s: int,
    retries: int,
) -> Tuple[Dict[str, Any], Dict[str, int]]:
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)

    last_err: Optional[str] = None
    for attempt in range(retries + 1):
        try:
            cfg = genai.types.GenerationConfig(
                temperature=float(temperature),
                max_output_tokens=int(max_output_tokens),
                response_mime_type="application/json",
            )
            resp = model.generate_content(prompt, generation_config=cfg, request_options={"timeout": timeout_s})
            text_out = (getattr(resp, "text", "") or "").strip()

            parsed = _safe_json_loads(text_out)
            if parsed is None:
                parsed = _safe_json_loads(_extract_json_object(text_out) or "")

            if parsed is None:
                raise RuntimeError("Non-JSON response from model")

            um = getattr(resp, "usage_metadata", None)
            usage = {
                "prompt_tokens": int(getattr(um, "prompt_token_count", 0) or 0) if um else 0,
                "output_tokens": int(getattr(um, "candidates_token_count", 0) or 0) if um else 0,
                "total_tokens": int(getattr(um, "total_token_count", 0) or 0) if um else 0,
            }
            return parsed, usage

        except Exception as e:
            last_err = str(e)
            time.sleep(min(2 ** attempt, 10))

    raise HTTPException(status_code=502, detail=f"Draft generation failed. Last error: {last_err}")


def _generate_draft_json_groq(
    groq_api_key: str,
    model: str,
    prompt: str,
    *,
    temperature: float,
    max_output_tokens: int,
    timeout_s: int,
    retries: int,
) -> Tuple[Dict[str, Any], Dict[str, int]]:
    headers = {"Authorization": f"Bearer {groq_api_key}", "Content-Type": "application/json"}

    body = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "You are a blog writer. Write a complete markdown article with all sections fully developed. Write at least 1900 words. Do NOT stop early.",
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": float(temperature),
        "max_tokens": int(max_output_tokens),
        # No response_format: json_object  -  plain text produces 2-3x longer output
    }

    last_err: Optional[str] = None
    for attempt in range(retries + 1):
        try:
            resp = requests.post(GROQ_API_URL, headers=headers, json=body, timeout=timeout_s)
        except Exception as e:
            last_err = str(e)
            time.sleep(min(2 ** attempt, 10))
            continue

        if resp.status_code in (429, 500, 502, 503, 504):
            last_err = f"{resp.status_code}: {resp.text[:300]}"
            time.sleep(min(2 ** attempt, 10))
            continue

        if resp.status_code != 200:
            last_err = f"{resp.status_code}: {resp.text[:500]}"
            break

        data = resp.json()
        try:
            content = (data["choices"][0]["message"]["content"] or "").strip()
        except Exception:
            content = ""

        if not content:
            last_err = "Groq returned empty content."
            break

        # Wrap plain markdown into the expected dict structure
        parsed = {"title": "", "draft_markdown": content, "used_chunks": []}

        um = data.get("usage") or {}
        usage = {
            "prompt_tokens": int(um.get("prompt_tokens") or 0),
            "output_tokens": int(um.get("completion_tokens") or 0),
            "total_tokens": int(um.get("total_tokens") or 0),
        }
        return parsed, usage

    raise HTTPException(status_code=502, detail=f"Groq draft generation failed. Last error: {last_err}")


# ============================================================
# OpenAI GPT-5.2  -  Outline-first two-call architecture
# ============================================================
def _generate_outline_openai(
    client: OpenAI,
    model: str,
    title: str,
    keywords: List[str],
    sources_block: str,
    *,
    system_prompt: str = "",
    temperature: float = 0.7,
) -> Tuple[str, Dict[str, int]]:
    """Call 1: Generate a structured outline (~900 tokens). Cheap planning step."""
    sys_prompt = system_prompt or SYSTEM_PROMPT
    kw = ", ".join(keywords or [])
    outline_prompt = f"""I'm writing a blog post about "{title}" and I need to plan it out.
Keywords to hit: {kw}

For each section, give me:
- A heading (can be playful  -  not boring corporate headings)
- 3-4 bullet points of what I should cover (pull from the sources)
- A personal angle, analogy, or mini-story idea I can use
- Rough word count target

Here's my structure  -  8 sections:
1. Hook intro (80-100 words)  -  start with a question, a bold statement, or a mini-story. NO "in today's world" nonsense.
2. The basics  -  what is {title}? (150-180 words)  -  explain like you're telling a friend
3. Why anyone should care (150-180 words)  -  real impact, real people, not abstract fluff
4. How it actually works (200-230 words)  -  walk through it step by step, keep it dead simple
5. Real stories / examples (200-230 words)  -  2-3 concrete, relatable examples
6. Mistakes people make (100-130 words)  -  common traps, things that go wrong
7. FAQ section (120-150 words)  -  5-8 quick Q&As, casual short answers
8. Wrap up / takeaways (80-100 words)  -  bullet summary, keep it punchy

HARD LIMIT: 1600-1800 words. You MUST stop at 1800 words. Count as you write.

SOURCES (use these for facts):
{sources_block}

Give me the outline as markdown with ## headings and bullets."""

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": outline_prompt},
        ],
        temperature=temperature,
        max_completion_tokens=1024,
    )
    outline = (resp.choices[0].message.content or "").strip()
    u = resp.usage
    usage = {
        "prompt_tokens": u.prompt_tokens if u else 0,
        "output_tokens": u.completion_tokens if u else 0,
        "total_tokens": u.total_tokens if u else 0,
    }
    return outline, usage


def _generate_draft_openai(
    client: OpenAI,
    model: str,
    title: str,
    keywords: List[str],
    outline: str,
    sources_block: str,
    *,
    system_prompt: str = "",
    temperature: float = 0.7,
    max_tokens: int = 8192,
) -> Tuple[Dict[str, Any], Dict[str, int]]:
    """Call 2: Generate the full article guided by the outline."""
    sys_prompt = system_prompt or SYSTEM_PROMPT
    kw = ", ".join(keywords or [])
    draft_prompt = f"""Write the full blog post on "{title}" based on this outline.
Keywords: {kw}

OUTLINE:
{outline}

RULES  -  read these carefully:
- Follow the outline section by section. Hit every section's word count.
- HARD LIMIT: 1600-1800 words. You MUST stop at 1800 words. Count as you write.
- Each section = new info. If you already said it, don't say it again.
- FAQ: 5-8 questions, casual short answers.
- Use the sources for facts. Don't make stuff up. If sources don't cover something, just say "I couldn't find solid info on this" and move on.

STYLE  -  notice what makes this example feel human:

---
You ever look at something and think "wait, that can't be right"? That happened to me researching this topic. The numbers don't lie  -  but they definitely surprise you.

Here's how I think about it. You know how sometimes you learn something and it changes how you see everything? That's what happened here. And honestly, I wish someone had explained it to me this way years ago.

So the simple version? It's not as complicated as people make it sound. That's the whole point.
---

Why that works:
- Sentence lengths go from 3 words to 25 words  -  randomly
- Starts sentences with "So", "But", "And", "Honestly?"
- Has opinions and personal reactions
- Uses dashes, ellipses, fragments
- Doesn't wrap up sections neatly  -  just moves on

SOURCES:
{sources_block}

Write the full article now. Markdown only  -  no intro text, no "here's the article", just the article itself."""

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": draft_prompt},
        ],
        temperature=temperature,
        max_completion_tokens=max_tokens,
    )
    content = (resp.choices[0].message.content or "").strip()
    parsed = {"title": title, "draft_markdown": content, "used_chunks": []}

    u = resp.usage
    usage = {
        "prompt_tokens": u.prompt_tokens if u else 0,
        "output_tokens": u.completion_tokens if u else 0,
        "total_tokens": u.total_tokens if u else 0,
    }
    return parsed, usage


def _build_article_prompt(title: str, keywords: List[str], length_target: int, sources: List[Dict[str, Any]]) -> str:
    """
    Fallback prompt (Groq / Gemini path). Uses the same editorial standards as
    the agentic OpenAI pipeline  -  blog arc, no rigid template, Grade 7+ prose.
    """
    kw = ", ".join(keywords or [])
    target = max(1900, min(2300, int(length_target or 2000)))
    src_lines = []
    for s in sources:
        src_lines.append(
            f"[SOURCE doc_id={s['doc_id']} chunk_id={s['chunk_id']}]\n{s['text']}\n"
        )
    src_block = "\n".join(src_lines)

    return f"""Write a complete markdown blog article. Output ONLY the markdown  -  no JSON, no preamble.

TOPIC: {title}
KEYWORDS: {kw}
TARGET LENGTH: {target} words (minimum {target - 100}, do not stop early)

BLOG STRUCTURE  -  follow this arc exactly:
1. Opening hook (150-200 words, NO heading): a specific fact, tension, or counter-intuitive finding that pulls the reader in. No "In this article we will..." openings.
2. 5-7 body sections with specific, informative headings  -  not label-style ("What Is X", "Why It Matters", "How It Works" are banned). Each section minimum 200 words of developed analytical prose.
3. One insight/perspective section: what the evidence means beyond the numbers.
4. Conclusion section: a specific forward-looking observation, no "In conclusion".
5. ## FAQ section: 5-7 real reader questions answered at Grade 7+ level.

WRITING RULES:
- Vary sentence length: most 10-18 words, some long (18-26 words with em-dashes), occasional 3-6 word punches after a long sentence only.
- Every paragraph explains WHY, not just WHAT. State the fact AND its consequence.
- Use colon reveals, temporal contrast, concessive pivots ("Still,", "By contrast,").
- No child-level analogies. No talk-down comparisons. Grade 7+ throughout.
- Do NOT invent facts not present in sources. Note gaps explicitly.
- Banned words: delve, leverage, furthermore, moreover, pivotal, tapestry, embark, navigate, comprehensive, utilize, facilitate, it is important to note, in today's world, it is crucial, paramount, robust, seamless, synergy, notable, significant, vital, underscores, a testament to.

SOURCES (ground all factual claims here):
{src_block}
""".strip()


# ============================================================
# Request/Response models
# ============================================================
class RunRequest(BaseModel):
    # retrieval
    top_k_sources: int = 8
    embedding_model: str = "models/gemini-embedding-001"
    output_dimensionality: int = 1536
    hybrid_top_k_per_query: int = 30
    expanded_query_count: int = 4

    # generation
    draft_provider: str = "openai"  # "openai", "groq", or "gemini"
    draft_model: str = ""  # auto-detected from env
    temperature: float = 0.7
    max_output_tokens: int = 8192
    rag_grounding_ratio: float = 0.60
    enable_agentic_orchestration: bool = True
    use_blog_pipeline: bool = True   # multi-agent section-based pipeline
    predictability_top_n: int = 14
    max_predictability_rewrite_passes: int = 1

    # where to store
    gcs_prefix_articles: str = "articles/"


class RunResponse(BaseModel):
    status: str
    request_id: str
    kb_id: str
    tenant_id: str
    attempt_no: int

    gcs_draft_uri: str
    draft_fingerprint: str
    draft_model: str

    used_sources: List[Dict[str, Any]] = Field(default_factory=list)
    usage: Dict[str, Any] = Field(default_factory=dict)


# ============================================================
# Day 17 endpoint: /run
# ============================================================
@router.post("/requests/{request_id}/run", response_model=RunResponse)
def run_article_request(
    request_id: str,
    req: RunRequest,
    claims: Claims = Depends(require_claims),
):
    """
    Day 17:
    retrieve -> draft generation -> store draft in GCS -> log usage_events
    """
    _require_admin(claims)
    settings = get_settings()
    tenant_id = claims.tenant_id

    api_key = _pick(settings, "GEMINI_API_KEY", "GOOGLE_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="Missing GEMINI_API_KEY/GOOGLE_API_KEY")

    bucket_name = _pick(settings, "GCS_BUCKET_NAME")
    if not bucket_name:
        raise HTTPException(status_code=500, detail="Missing GCS_BUCKET_NAME")

    processed_prefix = _norm_prefix(_pick(settings, "GCS_PREFIX_PROCESSED", "GCS_PROCESSED_PREFIX", default="processed/"))
    articles_prefix = _norm_prefix(_pick(settings, "GCS_PREFIX_ARTICLES", default=req.gcs_prefix_articles))

    job_id = request_id  # lock key
    lock_ttl_min = 30

    # ---- DB: load request + lock + mark in_progress ----
    with _db_conn(settings) as conn:
        _ensure_day17_columns(conn)
        _log_job_event(conn, tenant_id, "article_run_started", {"request_id": request_id}, job_id=job_id)

        ok, lock_token = _try_lock(conn, job_id=job_id, ttl_minutes=lock_ttl_min)
        if not ok:
            conn.commit()
            raise HTTPException(status_code=409, detail="Request is locked/in_progress. Try again later.")

        # load request row (tenant-scoped)
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                SELECT request_id::text, tenant_id::text, kb_id::text, title, keywords, length_target,
                       status, attempt_count
                FROM public.article_requests
                WHERE tenant_id=%s::uuid AND request_id=%s::uuid
                LIMIT 1
                """,
                (tenant_id, request_id),
            )
            row = cur.fetchone()

        if not row:
            _unlock(conn, job_id, lock_token)
            conn.commit()
            raise HTTPException(status_code=404, detail="Request not found")

        kb_id = str(row["kb_id"])
        title = str(row["title"])
        keywords = list(row.get("keywords") or [])
        length_target = int(row.get("length_target") or 2000)
        attempt_no = int(row.get("attempt_count") or 0) + 1

        # mark in_progress and increment attempt_count
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE public.article_requests
                SET status='in_progress',
                    attempt_count=%s,
                    last_error=NULL,
                    last_run_at=now(),
                    draft_model=%s
                WHERE tenant_id=%s::uuid AND request_id=%s::uuid
                """,
                (attempt_no, req.draft_model, tenant_id, request_id),
            )

        _log_job_event(conn, tenant_id, "article_run_claimed", {"attempt_no": attempt_no}, job_id=job_id)
        conn.commit()

    provider = (req.draft_provider or "").lower().strip()
    openai_api_key = os.getenv("OPENAI_API_KEY")
    groq_api_key = _pick(settings, "GROQ_API_KEY")

    # Auto-detect: prefer OpenAI if key is set, even if user didn't specify
    if provider == "openai" or (provider == "" and openai_api_key):
        provider = "openai"

    user_profile: Dict[str, Any] = {
        "intent_summary": f"Write a grounded article about {title}",
        "user_tone": "professional",
        "target_reader": "informed general readers",
        "must_include_terms": keywords or [],
    }
    user_agent_usage = {"prompt_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    query_agent_usage = {"prompt_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    source_tone_usage = {"prompt_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    query_pack: Dict[str, Any] = {"queries": [title] + (keywords[:2] if keywords else [])}

    if req.enable_agentic_orchestration and provider == "openai" and openai_api_key:
        try:
            client_for_planning = OpenAI(api_key=openai_api_key)
            planning_model = req.draft_model or os.getenv("OPENAI_MODEL", OPENAI_DEFAULT_MODEL)
            user_profile, user_agent_usage = _agent_understand_user_question(
                client=client_for_planning,
                model=planning_model,
                title=title,
                keywords=keywords,
                length_target=length_target,
            )
            query_pack, query_agent_usage = _agent_expand_queries(
                client=client_for_planning,
                model=planning_model,
                title=title,
                keywords=keywords,
                user_intent=user_profile,
            )
        except Exception:
            pass

    expanded_queries = query_pack.get("queries") if isinstance(query_pack.get("queries"), list) else []
    expanded_queries = [str(q).strip() for q in expanded_queries if str(q).strip()]
    if not expanded_queries:
        expanded_queries = [title] + [k for k in keywords[:3] if k]
    expanded_queries = expanded_queries[: max(1, int(req.expanded_query_count))]

    # ---- Agent 3 retrieval: hybrid semantic + bm25 + rrf ----
    gcs = _gcs_client(settings)
    with _db_conn(settings) as conn:
        used_sources, retrieval_meta = _agent_retrieve_hybrid_sources(
            conn,
            gcs,
            tenant_id=tenant_id,
            kb_id=kb_id,
            queries=expanded_queries,
            embedding_api_key=api_key,
            embedding_model=req.embedding_model,
            output_dimensionality=int(req.output_dimensionality),
            top_k_final=int(req.top_k_sources),
            top_k_per_query=int(req.hybrid_top_k_per_query),
        )

    if not used_sources:
        # fail safely
        with _db_conn(settings) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE public.article_requests
                    SET status='failed', last_error=%s, updated_at=now()
                    WHERE tenant_id=%s::uuid AND request_id=%s::uuid
                    """,
                    ("No source chunks found for this KB (embed/search not ready?)", tenant_id, request_id),
                )
                _log_job_event(conn, tenant_id, "article_run_failed", {"reason": "no_sources"}, job_id=job_id)
                _unlock(conn, job_id, lock_token)
                conn.commit()
        raise HTTPException(status_code=409, detail="No source chunks found. Ensure KB has embedded chunks (Day14)")

    # Build sources block for prompt / analysis
    src_lines = []
    for s in used_sources:
        src_lines.append(f"[SOURCE doc_id={s['doc_id']} chunk_id={s['chunk_id']}]\n{s['text']}\n")
    sources_block = "\n".join(src_lines)

    if provider == "openai":
        if not openai_api_key:
            raise HTTPException(status_code=500, detail="Missing OPENAI_API_KEY for draft_provider=openai")
        client = OpenAI(api_key=openai_api_key)
        model_used = req.draft_model or os.getenv("OPENAI_MODEL", OPENAI_DEFAULT_MODEL)

        source_tone: Dict[str, Any] = {}
        analysis: Dict[str, Any] = {}
        analysis_usage = {"prompt_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        compose_meta: Dict[str, Any] = {}

        if req.enable_agentic_orchestration:
            # Agent 4: analyze tone from retrieved evidence
            source_tone, source_tone_usage = _agent_analyze_retrieved_tone(
                client=client,
                model=model_used,
                sources_block=sources_block,
            )
            # Keep generic source analysis for compatibility/audit fields.
            analysis, analysis_usage = _analyze_source(
                client=client,
                model=model_used,
                sources_block=sources_block,
            )

            # Blend user and source tone signals into the dynamic voice layer.
            user_tone = str(user_profile.get("user_tone") or "").strip().lower()
            source_tone_label = str(source_tone.get("source_tone") or "").strip().lower()
            merged_analysis = dict(analysis)
            if user_tone in _TONE_MAP:
                merged_analysis["tone"] = user_tone
            elif source_tone_label in _TONE_MAP:
                merged_analysis["tone"] = source_tone_label
            voice_hint = source_tone.get("recommended_voice_hint")
            if voice_hint:
                merged_analysis["voice_hint"] = str(voice_hint)

            dynamic_prompt = _build_dynamic_prompt(merged_analysis)

            if req.use_blog_pipeline and _BLOG_PIPELINE_AVAILABLE:
                # ── Multi-agent section pipeline ──────────────────────────
                blog_result = _run_blog_pipeline(
                    client=client,
                    model=model_used,
                    title=title,
                    keywords=keywords,
                    sources=used_sources,
                    target_words=int(length_target),
                )
                compose_usage = blog_result.get("usage", {})
                compose_meta = {
                    "pipeline": "blog_pipeline_v1",
                    "section_plan": blog_result.get("section_plan", []),
                    "section_meta": blog_result.get("section_meta", []),
                    "warnings": blog_result.get("warnings", []),
                    "is_sparse": blog_result.get("is_sparse", False),
                    "analysis": blog_result.get("analysis", {}),
                    "elapsed_seconds": blog_result.get("elapsed_seconds", 0),
                }
                draft_json = {
                    "title": blog_result.get("title", title),
                    "draft_markdown": blog_result.get("draft_markdown", ""),
                    "used_chunks": [
                        {"doc_id": s.get("doc_id"), "chunk_id": s.get("chunk_id")}
                        for s in used_sources
                    ],
                    "source_analysis": merged_analysis,
                    "agentic_trace": {
                        "user_profile": user_profile,
                        "expanded_queries": expanded_queries,
                        "retrieval_meta": retrieval_meta,
                        "source_tone_profile": source_tone,
                        "compose_meta": compose_meta,
                    },
                }
            else:
                # ── Legacy monolithic pipeline ────────────────────────────
                draft_json, compose_usage, compose_meta = _agent_compose_grounded_article(
                    client=client,
                    model=model_used,
                    title=title,
                    keywords=keywords,
                    length_target=length_target,
                    sources=used_sources,
                    system_prompt=dynamic_prompt,
                    user_profile=user_profile,
                    source_tone=source_tone,
                    temperature=float(req.temperature),
                    max_tokens=int(req.max_output_tokens),
                    grounding_ratio=max(0.8, min(float(req.rag_grounding_ratio), 0.99)),
                    top_risky_sentences=max(6, min(int(req.predictability_top_n), 30)),
                    max_risk_rewrite_passes=max(0, min(int(req.max_predictability_rewrite_passes), 2)),
                )
                draft_json["source_analysis"] = merged_analysis
                draft_json["agentic_trace"] = {
                    "user_profile": user_profile,
                    "expanded_queries": expanded_queries,
                    "retrieval_meta": retrieval_meta,
                    "source_tone_profile": source_tone,
                    "compose_meta": compose_meta,
                }

            usage = _sum_usage(
                user_agent_usage,
                query_agent_usage,
                source_tone_usage,
                analysis_usage,
                compose_usage,
            )
        else:
            # Backward-compatible non-agentic OpenAI path.
            analysis, analysis_usage = _analyze_source(
                client=client,
                model=model_used,
                sources_block=sources_block,
            )
            dynamic_prompt = _build_dynamic_prompt(analysis)
            outline, outline_usage = _generate_outline_openai(
                client=client,
                model=model_used,
                title=title,
                keywords=keywords,
                sources_block=sources_block,
                system_prompt=dynamic_prompt,
                temperature=float(req.temperature),
            )
            draft_json, draft_usage = _generate_draft_openai(
                client=client,
                model=model_used,
                title=title,
                keywords=keywords,
                outline=outline,
                sources_block=sources_block,
                system_prompt=dynamic_prompt,
                temperature=float(req.temperature),
                max_tokens=int(req.max_output_tokens),
            )
            draft_json["source_analysis"] = analysis
            usage = _sum_usage(analysis_usage, outline_usage, draft_usage)

    elif provider == "groq":
        if not groq_api_key:
            raise HTTPException(status_code=500, detail="Missing GROQ_API_KEY for draft_provider=groq")

        prompt = _build_article_prompt(title, keywords, length_target, used_sources)
        model_used = req.draft_model or GROQ_DEFAULT_MODEL
        draft_json, usage = _generate_draft_json_groq(
            groq_api_key=str(groq_api_key),
            model=model_used,
            prompt=prompt,
            temperature=float(req.temperature),
            max_output_tokens=int(req.max_output_tokens),
            timeout_s=90,
            retries=3,
        )
    else:
        # fallback to Gemini
        prompt = _build_article_prompt(title, keywords, length_target, used_sources)
        model_used = req.draft_model or "gemini-2.5-flash"
        draft_json, usage = _generate_draft_json(
            api_key=api_key,
            model_name=model_used,
            prompt=prompt,
            temperature=float(req.temperature),
            max_output_tokens=int(req.max_output_tokens),
            timeout_s=90,
            retries=3,
        )

    # harden output
    draft_json.setdefault("title", title)
    draft_json.setdefault("draft_markdown", "")
    draft_json.setdefault("used_chunks", [{"doc_id": s["doc_id"], "chunk_id": s["chunk_id"]} for s in used_sources])

    artifact = {
        "request_id": request_id,
        "tenant_id": tenant_id,
        "kb_id": kb_id,
        "attempt_no": attempt_no,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": model_used,
        "retrieval": {
            "top_k_sources": int(req.top_k_sources),
            "top_k_per_query": int(req.hybrid_top_k_per_query),
            "expanded_queries": expanded_queries,
            "embedding_model": req.embedding_model,
            "output_dimensionality": int(req.output_dimensionality),
            "meta": retrieval_meta,
        },
        "agentic": {
            "enabled": bool(req.enable_agentic_orchestration),
            "rag_grounding_ratio_target": float(req.rag_grounding_ratio),
            "predictability_top_n": int(req.predictability_top_n),
            "max_predictability_rewrite_passes": int(req.max_predictability_rewrite_passes),
        },
        "usage": usage,
        "draft": draft_json,
    }

    out_bytes = json.dumps(artifact, ensure_ascii=False, indent=2).encode("utf-8")
    out_fp = _sha256_bytes(out_bytes)

    # ---- Store draft in GCS (idempotent, versioned by attempt_no) ----
    # Plan: include request_id + attempt version; no overwrite:contentReference[oaicite:4]{index=4}
    obj = (
        f"{articles_prefix}{tenant_id}/{kb_id}/{request_id}/attempt_{attempt_no}/"
        f"draft_v1/{out_fp}.json"
    )
    gcs_uri = _gcs_upload_bytes_create_only(
        gcs,
        bucket_name=bucket_name,
        object_name=obj,
        data=out_bytes,
        content_type="application/json; charset=utf-8",
    )

    # ---- Update DB status + unlock ----
    with _db_conn(settings) as conn:
        _ensure_day17_columns(conn)

        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE public.article_requests
                SET status='completed',
                    gcs_draft_uri=%s,
                    draft_fingerprint=%s,
                    draft_model=%s,
                    draft_meta=%s::jsonb,
                    last_error=NULL
                WHERE tenant_id=%s::uuid AND request_id=%s::uuid
                """,
                (
                    gcs_uri,
                    out_fp,
                    model_used,
                    json.dumps(
                        {
                            "sources": [{"doc_id": s["doc_id"], "chunk_id": s["chunk_id"]} for s in used_sources],
                            "queries": expanded_queries,
                            "retrieval_meta": retrieval_meta,
                            "agentic": bool(req.enable_agentic_orchestration),
                        }
                    ),
                    tenant_id,
                    request_id,
                ),
            )

        _log_job_event(conn, tenant_id, "article_draft_saved", {"gcs_draft_uri": gcs_uri, "fingerprint": out_fp}, job_id=job_id)

        # usage_events (best-effort)  -  plan says log usage_events for vendor calls:contentReference[oaicite:5]{index=5}
        _usage_event_best_effort(
            conn,
            tenant_id=tenant_id,
            request_id=request_id,
            payload={
                "operation_name": "article_draft",
                "vendor": (provider or "gemini"),
                "model": model_used,
                "tokens": int(usage.get("total_tokens") or 0),
                "total_cost": 0,  # cost calculator comes later days
                "detail": {"prompt_tokens": usage.get("prompt_tokens"), "output_tokens": usage.get("output_tokens")},
            },
        )

        _unlock(conn, job_id, lock_token)
        _log_job_event(conn, tenant_id, "article_run_done", {"status": "completed"}, job_id=job_id)
        conn.commit()

    return RunResponse(
        status="ok",
        request_id=request_id,
        kb_id=kb_id,
        tenant_id=tenant_id,
        attempt_no=attempt_no,
        gcs_draft_uri=gcs_uri,
        draft_fingerprint=out_fp,
        draft_model=model_used,
        used_sources=[{"doc_id": s["doc_id"], "chunk_id": s["chunk_id"], "distance": s["distance"]} for s in used_sources],
        usage=usage,
    )

