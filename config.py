"""
config.py — central configuration for the AI Safety Evaluation Platform.

All settings come from environment variables so nothing is hardcoded.
Copy .env.example to .env and fill in your values, then:
    from dotenv import load_dotenv; load_dotenv()
before importing this module.
"""

from __future__ import annotations

import os
import uuid
from dataclasses import dataclass, field


def _bool(val: str | None, default: bool = False) -> bool:
    """Parse a string env var as a boolean."""
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes")


def _int(val: str | None, default: int = 0) -> int:
    if val is None:
        return default
    try:
        return int(val)
    except ValueError:
        return default


@dataclass
class Config:
    # ── Dataset paths ──────────────────────────────────────────────────────────
    # Path to the Jigsaw train.csv (required)
    dataset_csv_path: str = field(
        default_factory=lambda: os.getenv(
            "DATASET_CSV_PATH",
            r"data/raw/train.csv/train.csv",  # default based on how Kaggle extracted it
        )
    )
    # Path to the Davidson hate-speech CSV (optional — leave blank to skip)
    davidson_csv_path: str = field(
        default_factory=lambda: os.getenv("DAVIDSON_CSV_PATH", "")
    )

    # ── Sampling ───────────────────────────────────────────────────────────────
    # Max rows to load from each dataset. 0 = load all rows.
    # Use a small number (e.g. 500) during development to keep things fast.
    sample_size: int = field(
        default_factory=lambda: _int(os.getenv("SAMPLE_SIZE"), default=5000)
    )

    # ── Database ───────────────────────────────────────────────────────────────
    # Full PostgreSQL connection string.
    # Format: postgresql://user:password@host:port/dbname
    database_url: str = field(
        default_factory=lambda: os.getenv(
            "DATABASE_URL",
            "postgresql://postgres:postgres@localhost:5432/trust_safety",
        )
    )

    # ── LLM API keys (all optional — system falls back to fixtures if absent) ──
    openai_api_key: str = field(
        default_factory=lambda: os.getenv("OPENAI_API_KEY", "")
    )
    gemini_api_key: str = field(
        default_factory=lambda: os.getenv("GEMINI_API_KEY", "")
    )

    # ── Model names ────────────────────────────────────────────────────────────
    openai_model: str = field(
        default_factory=lambda: os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    )
    gemini_model: str = field(
        default_factory=lambda: os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    )

    # ── Fixture mode ───────────────────────────────────────────────────────────
    # When True (or when no API keys are set), the LLM moderators read from
    # pre-recorded fixture files instead of calling any external API.
    # This lets the entire pipeline run with zero API credits.
    use_fixtures: bool = field(
        default_factory=lambda: _bool(os.getenv("USE_FIXTURES"), default=True)
    )

    # Path to the pre-recorded LLM response fixture file
    fixture_path: str = field(
        default_factory=lambda: os.getenv(
            "FIXTURE_PATH", "tests/fixtures/llm_responses.json"
        )
    )

    # ── Pipeline control ───────────────────────────────────────────────────────
    # Set SKIP_INGEST=true to skip loading CSVs (use whatever is already in DB)
    skip_ingest: bool = field(
        default_factory=lambda: _bool(os.getenv("SKIP_INGEST"), default=False)
    )

    # ── Run tracking ───────────────────────────────────────────────────────────
    # A unique ID for this pipeline run. Auto-generated if not set.
    # All predictions + metrics for one run share this ID so you can
    # compare runs side by side in Superset.
    run_id: str = field(
        default_factory=lambda: os.getenv("RUN_ID") or str(uuid.uuid4())
    )

    def is_fixture_mode(self) -> bool:
        """Return True if the pipeline should use fixtures instead of live APIs."""
        return self.use_fixtures or (
            not self.openai_api_key and not self.gemini_api_key
        )

    def __post_init__(self) -> None:
        # If no API keys are present, force fixture mode on so nothing breaks
        if not self.openai_api_key and not self.gemini_api_key:
            object.__setattr__(self, "use_fixtures", True)


def load_config() -> Config:
    """Load config from environment. Call load_dotenv() before this if using .env."""
    return Config()
