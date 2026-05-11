"""
Assembler — Phase 3.
Deterministic joining, word count check, targeted expand if short.
Also adds above-fold structure: META comment, Key Takeaways, TOC, Author Bio.
"""
from __future__ import annotations
import re
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI
from .gates_local_qc import word_count
from .prompt_engine import BrandContext

MIN_WORDS = 2300
MAX_WORDS = 3000
MIN_SECTION_WORDS = 180


def _build_meta_comment(title: str, hook_text: str) -> str:
    """Generate a 150-160 char META comment from the hook's opening sentence."""
    first_sent = ""
    if hook_text:
        m = re.search(r"[.!?]", hook_text)
        first_sent = hook_text[: m.end()].strip() if m else hook_text[:150].strip()
    meta = first_sent or title
    if len(meta) > 157:
        meta = meta[:154] + "..."
    return f"<!-- META: {meta} -->"


def _build_disclaimer_comment(brand_context: Optional["BrandContext"]) -> str:
    """Generate a DISCLAIMER comment if the tenant has set disclaimer text."""
    if not brand_context:
        return ""
    text = (brand_context.disclaimer or "").strip()
    if not text:
        return ""
    return f"<!-- DISCLAIMER: {text} -->"


def _build_key_takeaways(sections: List[Dict[str, Any]]) -> str:
    """Build a Key Takeaways block from section headings (non-hook, non-faq, non-closing).
    Filters out parenthetical/conditional headings that don't read as insights.
    Uses plain bold + bullets (not blockquote) to match CMS rendering expectations.
    """
    headings = [
        s.get("heading")
        for s in sections
        if s.get("heading")
        and s.get("role") not in ("hook", "faq", "closing")
        and len(s.get("heading", "")) <= 90
    ]
    if len(headings) < 2:
        return ""
    picks = headings[:4]
    lines = ["**Key takeaways**", ""]
    for h in picks:
        lines.append(f"- {h}")
    return "\n".join(lines)


def _add_anchors_to_headings(markdown: str) -> str:
    """Add {#anchor-id} to all H2/H3 headings for CMS compatibility (Hugo, Jekyll, etc.)."""
    def make_anchor(text: str) -> str:
        anchor = re.sub(r"[^a-z0-9\s-]", "", text.lower()).strip()
        anchor = re.sub(r"\s+", "-", anchor)
        anchor = re.sub(r"-+", "-", anchor).strip("-")
        return anchor[:60]

    def replace_heading(m: re.Match) -> str:
        hashes = m.group(1)
        text = m.group(2).strip()
        if "{#" in text:
            return m.group(0)
        return f"{hashes} {text} {{#{make_anchor(text)}}}"

    return re.sub(r"^(#{2,3})\s+(.+)$", replace_heading, markdown, flags=re.MULTILINE)


def _build_toc(sections: List[Dict[str, Any]]) -> str:
    """Build an 'In this article' TOC from H2 headings (excludes hook and closing)."""
    headings = [
        s.get("heading")
        for s in sections
        if s.get("heading") and s.get("role") not in ("hook", "closing")
    ]
    if len(headings) < 3:
        return ""
    lines = ["**In this article:**", ""]
    for h in headings:
        anchor = re.sub(r"[^a-z0-9\s-]", "", h.lower()).strip().replace(" ", "-")
        anchor = re.sub(r"-+", "-", anchor).strip("-")
        lines.append(f"- [{h}](#{anchor})")
    return "\n".join(lines)


def _strip_hook_structural(hook_raw: str) -> str:
    """Remove assembler-owned structural blocks that the hook writer must not emit.

    Strips the header line (**In this article...** / **Key takeaways...**) AND
    all immediately following bullet/blank lines belonging to that block.
    The old regex approach only removed the header line, leaving orphaned bullets
    which caused duplicate 'In this article' sections in the rendered output.
    """
    _BLOCK_HEADER = re.compile(
        r'^\*\*(?:In this article|Key takeaways)[:\*]*\*\*\s*$',
        re.IGNORECASE,
    )
    lines = hook_raw.split('\n')
    result: list[str] = []
    i = 0
    while i < len(lines):
        if _BLOCK_HEADER.match(lines[i]):
            i += 1
            # Skip all bullet lines and blank lines that belong to this block
            while i < len(lines) and (
                lines[i].startswith('- ')
                or lines[i].startswith('* ')
                or not lines[i].strip()
            ):
                i += 1
        else:
            result.append(lines[i])
            i += 1
    return '\n'.join(result).rstrip()


def _insert_above_fold(
    assembled: str,
    sections: List[Dict[str, Any]],
    title: str,
    brand_context: Optional["BrandContext"] = None,
) -> str:
    """
    Wrap the assembled markdown with:
      <!-- META: ... -->
      <!-- DISCLAIMER: ... -->   (only if brand_context.disclaimer is set)
      [hook paragraphs]
      Key Takeaways (bold heading + plain bullets)
      In this article TOC
      [rest of article]
      <!-- AUTHOR BIO: ... -->
    """
    author_bio = (
        "<!-- AUTHOR BIO: [Author full name] | "
        "[Credentials / professional designation — e.g. MD, CFP, PhD, RD] | "
        "[Affiliation or institution — fill before publishing] -->"
    )

    # Split at first H2/H3 heading — everything before is the hook
    m = re.search(r"^#{1,3}\s+", assembled, re.MULTILINE)
    if not m:
        hook_section = next((s for s in sections if s.get("role") == "hook"), None)
        hook_text = hook_section.get("text", "") if hook_section else assembled
        meta = _build_meta_comment(title, hook_text)
        disclaimer = _build_disclaimer_comment(brand_context)
        parts = [meta]
        if disclaimer:
            parts.append(disclaimer)
        parts += ["", assembled, "", author_bio]
        return "\n".join(parts)

    # Strip any structural blocks the hook writer may have added (TOC, Key Takeaways).
    # These are assembler-owned — duplicate occurrences ruin the above-fold layout.
    # Line-by-line removal: strips the header line AND all following bullet/blank lines
    # that belong to the block (regex approach left orphaned bullets behind).
    hook_raw = assembled[: m.start()].rstrip()
    hook_part = _strip_hook_structural(hook_raw)
    rest = assembled[m.start():]

    hook_section = next((s for s in sections if s.get("role") == "hook"), None)
    hook_text = hook_section.get("text", "") if hook_section else hook_part

    meta = _build_meta_comment(title, hook_text)
    disclaimer = _build_disclaimer_comment(brand_context)
    takeaways = _build_key_takeaways(sections)
    toc = _build_toc(sections)

    parts = [meta]
    if disclaimer:
        parts.append(disclaimer)
    parts += ["", hook_part]
    if takeaways:
        parts += ["", takeaways]
    if toc:
        parts += ["", toc]
    parts += ["", rest, "", author_bio]

    return "\n".join(parts)


def _join(sections: List[Dict[str, Any]]) -> str:
    parts = []
    for s in sections:
        text = (s.get("text") or "").strip()
        if text:
            parts.append(text)
    return "\n\n".join(parts)


def _find_thin_sections(sections: List[Dict[str, Any]], n: int = 2) -> List[int]:
    """Return indices of the n shortest non-FAQ, non-hook sections."""
    scored = [
        (i, word_count(s.get("text") or ""))
        for i, s in enumerate(sections)
        if s.get("role") not in ("faq",) and s.get("text")
    ]
    scored.sort(key=lambda x: x[1])
    return [i for i, _ in scored[:n]]


def _expand_section(
    client: OpenAI,
    model: str,
    section: Dict[str, Any],
    facts_block: str,
    temperature: float = 0.75,
    max_tokens: int = 500,
) -> Tuple[str, Dict[str, int]]:
    """Add 2-3 analytical sentences to a thin section inline."""
    role = section.get("role", "body")
    text = section.get("text", "")

    prompt = f"""Expand this {role} section by adding 2-3 analytical sentences WITHIN the existing paragraphs.
Do NOT add a new section. Do NOT repeat what is already written. Insert sentences that deepen the analysis.

CURRENT SECTION:
{text}

ADDITIONAL EVIDENCE FACTS (use these to deepen, do not invent):
{facts_block}

Rules:
- Add sentences inside existing paragraphs, not at the end
- Each added sentence must be specific (number, name, mechanism) — not generic
- Keep the same heading if present
- No transitional fillers, no paragraph-end summaries
- Output the full expanded section only"""

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a precise editorial writer. Add depth without changing voice. Output only the expanded section."},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            max_completion_tokens=max_tokens,
        )
        expanded = (resp.choices[0].message.content or "").strip()
        u = getattr(resp, "usage", None)
        usage = {
            "prompt_tokens": getattr(u, "prompt_tokens", 0) if u else 0,
            "output_tokens": getattr(u, "completion_tokens", 0) if u else 0,
            "total_tokens": getattr(u, "total_tokens", 0) if u else 0,
        }
        # Only accept if it actually added words
        if expanded and word_count(expanded) > word_count(text) + 20:
            return expanded, usage
    except Exception:
        pass
    return text, {"prompt_tokens": 0, "output_tokens": 0, "total_tokens": 0}


def _trim_to_budget(sections: List[Dict[str, Any]], budget: int) -> List[Dict[str, Any]]:
    """
    Trim the longest trimmable sections (body roles only) until total word count
    fits within budget. Trims at paragraph boundaries — never mid-sentence.
    hook, faq, and closing sections are left untouched.
    """
    SKIP_ROLES = {"hook", "faq", "closing"}

    def _trim_text(text: str, target: int) -> str:
        """Trim text to target words by dropping trailing paragraphs."""
        paras = [p for p in text.split("\n\n") if p.strip()]
        kept: List[str] = []
        total = 0
        for p in paras:
            pw = word_count(p)
            if total + pw > target and kept:
                break
            kept.append(p)
            total += pw
        return "\n\n".join(kept)

    current = sum(word_count(s.get("text") or "") for s in sections)
    for _ in range(len(sections)):
        if current <= budget:
            break
        # Find the fattest trimmable section
        candidates = [
            (i, word_count(s.get("text") or ""))
            for i, s in enumerate(sections)
            if s.get("role") not in SKIP_ROLES and s.get("text")
        ]
        if not candidates:
            break
        idx, section_wc = max(candidates, key=lambda x: x[1])
        excess = current - budget
        new_target = max(section_wc - excess, MIN_SECTION_WORDS)
        trimmed = _trim_text(sections[idx]["text"], new_target)
        sections[idx] = {**sections[idx], "text": trimmed}
        current = sum(word_count(s.get("text") or "") for s in sections)

    return sections


def assemble(
    client: Optional[OpenAI],
    model: Optional[str],
    sections: List[Dict[str, Any]],
    facts: List[Dict[str, Any]],
    target_words: int = 2000,
    max_tokens_per_expand: int = 500,
    title: str = "",
    brand_context: Optional[BrandContext] = None,
) -> Tuple[str, int, Dict[str, int]]:
    """
    Join sections, check word count, expand if short (max 2 targeted LLM calls).
    Adds META comment, Key Takeaways, TOC, and Author Bio placeholder.
    Returns (markdown, final_word_count, usage).
    """
    total_usage: Dict[str, int] = {"prompt_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    facts_block = "\n".join(
        f"- [{f['fact_id']}] {f['claim']}" for f in facts
    ) if facts else "No facts available."

    assembled = _join(sections)
    wc = word_count(assembled)

    if wc < MIN_WORDS and client and model:
        thin_indices = _find_thin_sections(sections, n=2)
        for idx in thin_indices:
            expanded, exp_usage = _expand_section(
                client, model, sections[idx], facts_block,
                max_tokens=max_tokens_per_expand,
            )
            sections[idx]["text"] = expanded
            total_usage = {k: total_usage[k] + exp_usage.get(k, 0) for k in total_usage}

        assembled = _join(sections)
        wc = word_count(assembled)

    # Trim over-length: if total exceeds MAX_WORDS, shorten the fattest body sections
    if wc > MAX_WORDS:
        sections = _trim_to_budget(sections, budget=MAX_WORDS)
        assembled = _join(sections)
        wc = word_count(assembled)

    # Strip internal pipeline markers — must never appear in published output
    assembled = re.sub(r'\s*\[F\d+\]', '', assembled)
    assembled = re.sub(r'\s*\[VERIFY\]', '', assembled)

    # Add CMS anchor IDs to all H2/H3 headings
    assembled = _add_anchors_to_headings(assembled)

    # Add above-fold structure (no LLM — deterministic)
    assembled = _insert_above_fold(assembled, sections, title or "", brand_context=brand_context)

    return assembled, wc, total_usage


def section_count(markdown: str) -> int:
    return len(re.findall(r"^#{1,3}\s+", markdown, re.MULTILINE))


def has_faq(markdown: str) -> bool:
    if re.search(r"^#{1,3}\s+.*\b(?:FAQ|Frequently\s+Asked|question)\b", markdown, re.MULTILINE | re.IGNORECASE):
        return True
    return len(re.findall(r"^\*\*.+\?\*\*", markdown, re.MULTILINE)) >= 3
