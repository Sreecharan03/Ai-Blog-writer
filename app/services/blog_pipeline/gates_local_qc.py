"""
Local QC gates — no LLM, no API calls, instant.
Run after every section write. Fail fast so retry budget is not wasted.
"""
import re
from typing import Tuple, List

# ── Banned words (AI cliches + implicit pattern words) ──────────────────────
_BANNED = {
    "leverage", "delve", "crucial", "pivotal", "unlock", "transformative",
    "holistic", "empower", "seamlessly", "cutting-edge", "robust", "utilize",
    "comprehensive", "facilitate", "unprecedented", "well-being", "paradigm",
    "synergy", "streamline", "game-changer", "innovative", "scalable",
    "actionable", "impactful", "ecosystem", "disruptive", "navigate",
    "landscape", "journey", "moreover", "furthermore", "additionally",
    "nevertheless", "notwithstanding",
}

# ── Transitional fillers (implicit AI pattern) ───────────────────────────────
_FILLER_PATTERNS = [
    r"\bthat being said\b",
    r"\bwith that in mind\b",
    r"\bbuilding on this\b",
    r"\bmoving forward\b",
    r"\bin light of this\b",
    r"\bit is (important|worth|essential) to note\b",
    r"\bit should be noted\b",
    r"\bin conclusion\b",
    r"\bto summarize\b",
    r"\bin summary\b",
    r"\boverall\b",
    r"\ball in all\b",
    r"\bit goes without saying\b",
    r"\bneedless to say\b",
]

# ── Paragraph-end summary patterns ───────────────────────────────────────────
_PARA_END_PATTERNS = [
    r"this (shows|means|highlights|underscores|demonstrates|reveals|illustrates) (why|how|that|what)",
    r"this is why",
    r"this is how",
    r"that is why",
    r"which (shows|means|highlights|underscores|demonstrates)",
]

# ── Label-style headings ─────────────────────────────────────────────────────
_LABEL_HEADING_PATTERNS = [
    r"^#{1,3}\s+(what is|what are|why (is|are|does|do|it)|how (to|does|do|it|is)|introduction|overview|background|conclusion|summary|key takeaways?|final thoughts?|wrapping up|in conclusion)",
]

# ── Passive attribution ──────────────────────────────────────────────────────
_PASSIVE_ATTR_PATTERNS = [
    r"\b(research|studies|experts?|scientists?|doctors?|analysts?) (has|have) shown\b",
    r"\baccording to (research|studies|experts?)\b",
    r"\bit (has been|is) (shown|proven|found|established|demonstrated) that\b",
]

_WORD_RE = re.compile(r"\b[a-zA-Z'-]+\b")
_SENT_RE = re.compile(r"[^.!?]+[.!?]+")


def _words(text: str) -> List[str]:
    return _WORD_RE.findall(text.lower())


def _sentences(text: str) -> List[str]:
    return [s.strip() for s in _SENT_RE.findall(text) if len(s.strip().split()) >= 3]


def check_word_count(text: str, min_words: int = 180, max_words: int = 450) -> Tuple[bool, str]:
    wc = len(_words(text))
    if wc < min_words:
        return False, f"too short: {wc} words (min {min_words})"
    if wc > max_words:
        return False, f"too long: {wc} words (max {max_words})"
    return True, f"word_count={wc}"


def check_label_heading(text: str) -> Tuple[bool, str]:
    for line in text.splitlines():
        line = line.strip().lower()
        for pat in _LABEL_HEADING_PATTERNS:
            if re.match(pat, line, re.IGNORECASE):
                return False, f"label-style heading detected: '{line[:60]}'"
    return True, "headings ok"


def check_banned_words(text: str) -> Tuple[bool, str]:
    found = [w for w in _words(text) if w in _BANNED]
    if found:
        return False, f"banned words: {list(set(found))}"
    return True, "no banned words"


def check_transitional_fillers(text: str) -> Tuple[bool, str]:
    tl = text.lower()
    hits = [p for p in _FILLER_PATTERNS if re.search(p, tl)]
    if hits:
        return False, f"transitional fillers found: {hits[:3]}"
    return True, "no fillers"


def check_para_end_summaries(text: str) -> Tuple[bool, str]:
    tl = text.lower()
    hits = [p for p in _PARA_END_PATTERNS if re.search(p, tl)]
    if hits:
        return False, f"paragraph-end summary detected"
    return True, "no summaries"


def check_passive_attribution(text: str) -> Tuple[bool, str]:
    tl = text.lower()
    hits = [p for p in _PASSIVE_ATTR_PATTERNS if re.search(p, tl)]
    if hits:
        return False, f"passive attribution: {hits[:2]}"
    return True, "no passive attribution"


def check_sentence_length_variety(text: str) -> Tuple[bool, str]:
    """At least one sentence <= 8 words and one >= 18 words — burstiness check."""
    sents = _sentences(text)
    if len(sents) < 3:
        return True, "too few sentences to check variety"
    lengths = [len(s.split()) for s in sents]
    has_short = any(l <= 8 for l in lengths)
    has_long = any(l >= 18 for l in lengths)
    if not has_short:
        return False, f"no short sentence (<=8 words). lengths={sorted(lengths)}"
    if not has_long:
        return False, f"no long sentence (>=18 words). lengths={sorted(lengths)}"
    return True, f"variety ok: min={min(lengths)} max={max(lengths)}"


def run_local_qc(
    text: str,
    role: str = "body",
    min_words: int = 180,
    max_words: int = 450,
) -> Tuple[bool, List[str]]:
    """
    Run all local gates. Returns (passed, [failure_reasons]).
    FAQ sections get relaxed word count (80-300 words).
    Hook sections skip heading check (no heading by design).
    """
    if role == "faq":
        min_words, max_words = 80, 350

    checks = [
        check_word_count(text, min_words, max_words),
        check_banned_words(text),
        check_transitional_fillers(text),
        check_para_end_summaries(text),
        check_passive_attribution(text),
        check_sentence_length_variety(text),
    ]
    if role != "hook":
        checks.append(check_label_heading(text))

    failures = [msg for ok, msg in checks if not ok]
    return len(failures) == 0, failures


def word_count(text: str) -> int:
    return len(_words(text))
