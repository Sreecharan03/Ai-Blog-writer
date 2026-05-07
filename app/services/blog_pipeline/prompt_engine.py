"""
PromptEngine — Brand Voice Layer (Layer B).

Loads per-tenant brand config from the DB and assembles a BrandContext object
that every writing agent can inject into its prompts. Falls back to sensible
defaults when a tenant hasn't configured their brand yet, so the pipeline
always works even for fresh tenants.

DB table: public.tenant_brand_configs (created on first use via _ensure_table).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, List, Optional


# ─────────────────────────────────────────────────────────────────────────────
# Smart defaults — what a tenant gets before they configure anything
# ─────────────────────────────────────────────────────────────────────────────
_DEFAULT_PERSONA = (
    "A knowledgeable but approachable writer — like a smart friend who happens "
    "to have deep expertise in the subject. Confident. Never preachy."
)
_DEFAULT_TONE = ["clear", "evidence-aware", "direct", "non-preachy"]
_DEFAULT_READING_LEVEL = "grade 10-12"
_DEFAULT_POV = "second-person"


# ─────────────────────────────────────────────────────────────────────────────
# BrandContext dataclass
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class BrandContext:
    persona: str = _DEFAULT_PERSONA
    tone_adjectives: List[str] = field(default_factory=lambda: list(_DEFAULT_TONE))
    forbidden_phrases: List[str] = field(default_factory=list)
    audience_primary: str = ""
    audience_pain_points: List[str] = field(default_factory=list)
    reading_level: str = _DEFAULT_READING_LEVEL
    language: str = "en"
    preferred_pov: str = _DEFAULT_POV
    compliance_note: str = ""

    def is_configured(self) -> bool:
        """True when a tenant has explicitly saved their config."""
        return bool(self.audience_primary)

    def voice_block(self) -> str:
        """
        Returns the brand voice overlay block appended to writer system prompts.
        Empty string when nothing meaningful is configured, so we never inject
        a useless block that just wastes tokens.
        """
        lines: List[str] = [
            "══════════════════════════════════════════════════════",
            "CLIENT BRAND VOICE — apply on top of the Core Writing Laws above",
            "══════════════════════════════════════════════════════",
        ]
        if self.persona:
            lines.append(f"WRITER PERSONA: {self.persona}")
        if self.tone_adjectives:
            lines.append(f"TONE: {', '.join(self.tone_adjectives)}")
        if self.preferred_pov:
            lines.append(f"POINT OF VIEW: {self.preferred_pov}")
        if self.reading_level:
            lines.append(f"READING LEVEL: Target {self.reading_level}. Not simpler, not more academic.")
        if self.forbidden_phrases:
            lines.append(
                f"EXTRA BANNED PHRASES (brand-specific): "
                + ", ".join(self.forbidden_phrases)
            )
        if self.audience_primary:
            lines.append(f"\nAUDIENCE: You are writing for {self.audience_primary}.")
        if self.audience_pain_points:
            pain = "; ".join(self.audience_pain_points)
            lines += [
                f"READER PAIN POINTS: {pain}",
                "Address these problems through the content — not by listing them, "
                "but by writing as if you understand exactly what the reader is struggling with.",
            ]
        if self.compliance_note:
            lines += ["", f"COMPLIANCE NOTE: {self.compliance_note}"]

        lines.append(
            "\nWhen a brand rule conflicts with reader clarity: break the brand rule. Keep the clarity."
        )
        return "\n".join(lines)

    def audience_context(self) -> str:
        """Short audience string for Section Planner user prompt injection."""
        if not self.audience_primary:
            return ""
        parts = [self.audience_primary]
        if self.audience_pain_points:
            parts.append("Pain points: " + "; ".join(self.audience_pain_points))
        return " | ".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# DB helpers
# ─────────────────────────────────────────────────────────────────────────────
_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS public.tenant_brand_configs (
    tenant_id          UUID PRIMARY KEY,
    persona            TEXT        NOT NULL DEFAULT '',
    tone_adjectives    TEXT[]      NOT NULL DEFAULT '{}',
    forbidden_phrases  TEXT[]      NOT NULL DEFAULT '{}',
    audience_primary   TEXT        NOT NULL DEFAULT '',
    audience_pain_points TEXT[]    NOT NULL DEFAULT '{}',
    reading_level      TEXT        NOT NULL DEFAULT 'grade 10-12',
    language           TEXT        NOT NULL DEFAULT 'en',
    preferred_pov      TEXT        NOT NULL DEFAULT 'second-person',
    compliance_note    TEXT        NOT NULL DEFAULT '',
    created_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at         TIMESTAMPTZ NOT NULL DEFAULT now()
);
"""


def _ensure_table(conn: Any) -> None:
    with conn.cursor() as cur:
        cur.execute(_CREATE_TABLE_SQL)
    conn.commit()


def load_brand_context(conn: Any, tenant_id: str) -> BrandContext:
    """
    Load brand config for tenant_id. Creates the table on first call.
    Returns BrandContext with smart defaults if tenant hasn't saved a config yet.
    """
    _ensure_table(conn)
    try:
        import psycopg2.extras  # local import — avoids hard dep at module level
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                "SELECT * FROM public.tenant_brand_configs WHERE tenant_id = %s::uuid",
                (tenant_id,),
            )
            row = cur.fetchone()
    except Exception:
        return BrandContext()

    if not row:
        return BrandContext()

    return BrandContext(
        persona=row.get("persona") or _DEFAULT_PERSONA,
        tone_adjectives=list(row.get("tone_adjectives") or _DEFAULT_TONE),
        forbidden_phrases=list(row.get("forbidden_phrases") or []),
        audience_primary=row.get("audience_primary") or "",
        audience_pain_points=list(row.get("audience_pain_points") or []),
        reading_level=row.get("reading_level") or _DEFAULT_READING_LEVEL,
        language=row.get("language") or "en",
        preferred_pov=row.get("preferred_pov") or _DEFAULT_POV,
        compliance_note=row.get("compliance_note") or "",
    )


def upsert_brand_context(conn: Any, tenant_id: str, ctx: BrandContext) -> None:
    """Save or update brand config for a tenant."""
    _ensure_table(conn)
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO public.tenant_brand_configs
                (tenant_id, persona, tone_adjectives, forbidden_phrases,
                 audience_primary, audience_pain_points, reading_level,
                 language, preferred_pov, compliance_note, updated_at)
            VALUES
                (%s::uuid, %s, %s, %s, %s, %s, %s, %s, %s, %s, now())
            ON CONFLICT (tenant_id) DO UPDATE SET
                persona            = EXCLUDED.persona,
                tone_adjectives    = EXCLUDED.tone_adjectives,
                forbidden_phrases  = EXCLUDED.forbidden_phrases,
                audience_primary   = EXCLUDED.audience_primary,
                audience_pain_points = EXCLUDED.audience_pain_points,
                reading_level      = EXCLUDED.reading_level,
                language           = EXCLUDED.language,
                preferred_pov      = EXCLUDED.preferred_pov,
                compliance_note    = EXCLUDED.compliance_note,
                updated_at         = now()
            """,
            (
                tenant_id,
                ctx.persona,
                ctx.tone_adjectives,
                ctx.forbidden_phrases,
                ctx.audience_primary,
                ctx.audience_pain_points,
                ctx.reading_level,
                ctx.language,
                ctx.preferred_pov,
                ctx.compliance_note,
            ),
        )
    conn.commit()
