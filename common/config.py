"""Centralised settings, loaded from environment / `.env`.

Import `settings` anywhere instead of reading os.environ directly.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(REPO_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Postgres
    postgres_user: str = "gc"
    postgres_password: str = "gc_dev_password"
    postgres_db: str = "greeting_cards"
    postgres_host: str = "localhost"
    postgres_port: int = 5432

    # MinIO / S3
    minio_endpoint: str = "http://localhost:9000"
    minio_root_user: str = "minioadmin"
    minio_root_password: str = "minioadmin"
    minio_bucket: str = "greeting-cards"
    minio_bucket_raw: str = "greeting-cards-raw"

    # Redis
    redis_host: str = "localhost"
    redis_port: int = 6379

    # Scraping
    scraper_user_agent: str = (
        "greeting-cards-research/0.1 (+mailto:gagan708344@gmail.com)"
    )
    scraper_rate_limit_per_sec: float = 1.0
    scraper_cache_dir: Path = REPO_ROOT / ".cache" / "raw_html"
    scraper_raw_html_ttl_days: int = 30

    # LLMs
    anthropic_api_key: str | None = None
    openai_api_key: str | None = None
    llm_provider: str = "anthropic"
    llm_model: str = "claude-sonnet-4-6"

    # HuggingFace
    hf_token: str | None = None

    # Diffusion
    diffusion_backend: str = "flux"
    sdxl_model_id: str = "stabilityai/stable-diffusion-xl-base-1.0"
    sdxl_revision: str | None = None
    flux_model_id: str = "black-forest-labs/FLUX.1-dev"
    flux_fill_model_id: str = "black-forest-labs/FLUX.1-Fill-dev"

    @field_validator("sdxl_revision", "hf_token", "anthropic_api_key", "openai_api_key",
                     "wandb_api_key", "prolific_api_token", "etsy_api_key", mode="before")
    @classmethod
    def _empty_str_to_none(cls, v: str | None) -> str | None:
        """Treat empty strings from .env as None."""
        if isinstance(v, str) and not v.strip():
            return None
        return v

    # W&B
    wandb_api_key: str | None = None
    wandb_project: str = "greeting-cards"
    wandb_entity: str | None = None

    # Prolific
    prolific_api_token: str | None = None

    # Google Forms / Sheets (survey instrument)
    google_service_account_json: Path | None = None  # path to service account JSON key file
    google_forms_id: str | None = None               # form ID from the URL
    google_sheets_id: str | None = None              # linked response spreadsheet ID
    # Pre-filled entry IDs — run survey/instrument/forms_creator.py to discover these
    google_form_entry_participant_id: str | None = None
    google_form_entry_card_key: str | None = None
    google_form_entry_occasion: str | None = None
    google_form_entry_study_id: str | None = None
    google_form_entry_purchase_intent: str | None = None
    google_form_entry_occasion_fit: str | None = None
    google_form_entry_aesthetic: str | None = None
    google_form_entry_emotional_resonance: str | None = None
    google_form_entry_distinctiveness: str | None = None
    google_form_entry_max_price_gbp: str | None = None
    google_form_entry_free_text: str | None = None

    # Etsy Open API v3
    etsy_api_key: str | None = None

    # Misc
    log_level: str = "INFO"
    random_seed: int = 42

    @property
    def postgres_dsn(self) -> str:
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
