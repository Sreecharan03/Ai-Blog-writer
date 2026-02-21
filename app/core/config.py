from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv


def _load_env_once() -> None:
    project_root = Path(__file__).resolve().parents[2]  # folder containing app/
    env_path = project_root / ".env"
    if env_path.exists():
        load_dotenv(dotenv_path=env_path, override=False)
    else:
        load_dotenv(override=False)


@dataclass(frozen=True)
class Settings:
    # Core
    environment: str
    project_id: str
    gcs_bucket_name: str

    # GCS prefixes
    gcs_prefix_raw: str
    gcs_prefix_processed: str
    gcs_prefix_url_snapshots: str
    gcs_prefix_articles: str

    # Supabase Postgres (DB)
    db_host: str
    db_port: int
    db_name: str
    db_user: str
    db_password: str
    db_sslmode: str

    # Optional integrations
    supabase_url: Optional[str]
    supabase_service_role_key: Optional[str]

    gemini_api_key: Optional[str]
    zerogpt_api_key: Optional[str]


def get_settings() -> Settings:
    _load_env_once()

    def _req(key: str) -> str:
        val = os.getenv(key)
        if not val:
            raise RuntimeError(f"Missing required env var: {key}")
        return val

    def _opt(key: str) -> Optional[str]:
        v = os.getenv(key)
        return v if v else None

    return Settings(
        environment=os.getenv("ENVIRONMENT", "local"),
        project_id=_req("GCP_PROJECT_ID"),
        gcs_bucket_name=_req("GCS_BUCKET_NAME"),
        gcs_prefix_raw=os.getenv("GCS_PREFIX_RAW", "raw/"),
        gcs_prefix_processed=os.getenv("GCS_PREFIX_PROCESSED", "processed/"),
        gcs_prefix_url_snapshots=os.getenv("GCS_PREFIX_URL_SNAPSHOTS", "url_snapshots/"),
        gcs_prefix_articles=os.getenv("GCS_PREFIX_ARTICLES", "articles/"),

        # Use DB_* from your .env (Supabase pooler/host details)
        db_host=_req("DB_HOST"),
        db_port=int(os.getenv("DB_PORT", "5432")),
        db_name=_req("DB_NAME"),
        db_user=_req("DB_USER"),
        db_password=_req("DB_PASSWORD"),
        db_sslmode=os.getenv("DB_SSLMODE", "require"),

        supabase_url=_opt("SUPABASE_URL"),
        supabase_service_role_key=_opt("SUPABASE_SERVICE_ROLE_KEY"),
        gemini_api_key=_opt("GEMINI_API_KEY"),
        zerogpt_api_key=_opt("ZEROGPT_API_KEY"),
    )
