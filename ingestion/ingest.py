"""
ingestion/ingest.py — CSV → canonical DataFrame → PostgreSQL ground_truth

Handles two source schemas:

  Jigsaw Toxic Comment Classification
    Columns: id, comment_text, toxic, severe_toxic, obscene, threat,
             insult, identity_hate
    item_id  → str(id)
    text     → comment_text
    All label columns kept as-is (already 0/1)

  Davidson Hate Speech and Offensive Language
    Columns: (unnamed index), count, hate_speech, offensive_language,
             neither, class, tweet
    item_id  → "davidson_<row_index>"
    text     → tweet
    class normalisation (0=hate_speech, 1=offensive, 2=neither):
      toxic      = 1 if class in {0, 1} else 0   (any harmful speech)
      hate_speech = 1 if class == 0 else 0        (Davidson-specific)
    Jigsaw multi-label columns (severe_toxic, obscene, threat, insult,
    identity_hate) → None / NaN — they don't exist in Davidson

Public API
----------
  detect_schema(df)               → "jigsaw" | "davidson"
  normalize_jigsaw(df)            → canonical DataFrame
  normalize_davidson(df)          → canonical DataFrame
  load_ground_truth(cfg, engine)  → rows_loaded: int

The function `load_ground_truth` accepts a Config and a SQLAlchemy Engine so
the caller can inject a test engine (no live DB needed for unit tests).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from sqlalchemy.engine import Engine

    from config import Config

logger = logging.getLogger(__name__)

# ── Column constants ────────────────────────────────────────────────────────

# Canonical ground_truth columns (must match db/schema.sql)
_CANONICAL_COLUMNS: list[str] = [
    "item_id",
    "text",
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate",
    "dataset_source",
    "davidson_class",
]

# Davidson `class` encoding
_DAVIDSON_HATE_SPEECH = 0
_DAVIDSON_OFFENSIVE = 1
_DAVIDSON_NEITHER = 2


# ── Schema detection ────────────────────────────────────────────────────────

def detect_schema(df: pd.DataFrame) -> str:
    """
    Detect which dataset schema a DataFrame comes from.

    Rules:
      - If ``comment_text`` is present  → "jigsaw"
      - If ``tweet`` is present         → "davidson"

    Raises:
        ValueError: if neither key column is found.
    """
    cols = set(df.columns)
    if "comment_text" in cols:
        return "jigsaw"
    if "tweet" in cols:
        return "davidson"
    raise ValueError(
        f"Cannot detect schema — expected 'comment_text' (Jigsaw) or "
        f"'tweet' (Davidson). Found columns: {sorted(cols)}"
    )


# ── Per-schema normalisation ────────────────────────────────────────────────

def normalize_jigsaw(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalise a Jigsaw DataFrame to the canonical ground_truth schema.

    - item_id  : str(id column)
    - text     : comment_text
    - labels   : toxic, severe_toxic, obscene, threat, insult, identity_hate
                 (kept as-is; already 0/1 with possible NaN)
    - dataset_source: "jigsaw"
    """
    out = pd.DataFrame()
    out["item_id"] = df["id"].astype(str)
    out["text"] = df["comment_text"].astype(str)

    # Multi-label columns — may have NaN in the raw data; keep as nullable int
    for label in ("toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"):
        if label in df.columns:
            out[label] = pd.array(df[label], dtype=pd.Int8Dtype())
        else:
            out[label] = pd.array([pd.NA] * len(df), dtype=pd.Int8Dtype())

    out["dataset_source"] = "jigsaw"
    out["davidson_class"] = pd.array([pd.NA] * len(df), dtype=pd.Int8Dtype())
    return out[_CANONICAL_COLUMNS]


def normalize_davidson(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalise a Davidson DataFrame to the canonical ground_truth schema.

    Davidson `class` encoding:
      0 → hate speech    (toxic=1, hate_speech label kept for downstream)
      1 → offensive      (toxic=1)
      2 → neither        (toxic=0)

    Jigsaw-specific multi-label columns are set to NaN because Davidson
    doesn't have them.  ``hate_speech`` is stored in the ``toxic`` column
    (class 0 is the most harmful bucket) so downstream evaluation can score
    it.  The raw ``class`` value is not stored — the canonical schema is
    what persists.

    item_id format: "davidson_<original_index>" for stable IDs across reloads
    (the Davidson CSV has an unnamed numeric index column that we use).
    """
    out = pd.DataFrame()

    # Build stable item IDs from the original row index
    if df.index.name is not None or not isinstance(df.index, pd.RangeIndex):
        out["item_id"] = "davidson_" + df.index.astype(str)
    else:
        # The CSV includes an unnamed leading index column after read_csv
        # (it appears as the DataFrame index when index_col=0 is used, or
        #  as a column named "" / "Unnamed: 0" otherwise)
        if "" in df.columns:
            out["item_id"] = "davidson_" + df[""].astype(str)
        elif "Unnamed: 0" in df.columns:
            out["item_id"] = "davidson_" + df["Unnamed: 0"].astype(str)
        else:
            out["item_id"] = "davidson_" + df.reset_index(drop=True).index.astype(str)

    out["text"] = df["tweet"].astype(str)

    # Binary toxic: class 0 (hate) or 1 (offensive) → 1; class 2 → 0
    cls = df["class"]
    out["toxic"] = pd.array(
        cls.apply(lambda c: 0 if c == _DAVIDSON_NEITHER else 1),
        dtype=pd.Int8Dtype(),
    )

    # Preserve original Davidson class for policy-level analysis:
    # 0 = hate_speech, 1 = offensive_language, 2 = neither
    out["davidson_class"] = pd.array(cls, dtype=pd.Int8Dtype())

    # Jigsaw-specific labels are N/A for Davidson rows
    for label in ("severe_toxic", "obscene", "threat", "insult", "identity_hate"):
        out[label] = pd.array([pd.NA] * len(df), dtype=pd.Int8Dtype())

    out["dataset_source"] = "davidson"
    return out[_CANONICAL_COLUMNS]


# ── Sample cap ──────────────────────────────────────────────────────────────

def _apply_sample_cap(df: pd.DataFrame, sample_size: int) -> pd.DataFrame:
    """
    Return at most ``sample_size`` rows using stratified sampling on the
    ``toxic`` column so the toxic/non-toxic ratio is preserved.

    Falls back to random sampling if ``toxic`` column is not present.
    If ``sample_size`` is 0 or >= len(df), return the full DataFrame.
    """
    if sample_size <= 0 or sample_size >= len(df):
        return df

    # Stratified on toxic column if available
    if "toxic" in df.columns:
        try:
            # Use min of requested fraction or available rows per group
            toxic     = df[df["toxic"] == 1]
            non_toxic = df[df["toxic"] == 0]

            n_toxic     = min(len(toxic),     sample_size // 2)
            n_non_toxic = min(len(non_toxic), sample_size - n_toxic)

            sampled = pd.concat([
                toxic.sample(n=n_toxic,     random_state=42),
                non_toxic.sample(n=n_non_toxic, random_state=42),
            ]).sample(frac=1, random_state=42)  # shuffle

            return sampled.reset_index(drop=True)
        except Exception:
            pass  # fall through to random

    return df.sample(n=sample_size, random_state=42)


# ── Single-file ingestion helper ────────────────────────────────────────────

def ingest_csv(csv_path: str, sample_size: int = 0) -> pd.DataFrame:
    """
    Read a CSV file, detect its schema, normalise it, and apply the sample cap.

    Args:
        csv_path:    Path to the CSV file.
        sample_size: Maximum rows to return (0 = all rows).

    Returns:
        Normalised DataFrame in canonical ground_truth schema.

    Raises:
        FileNotFoundError: if the file doesn't exist.
        ValueError:        if the schema can't be detected.
    """
    logger.info("Reading CSV: %s", csv_path)
    df = pd.read_csv(csv_path)
    logger.info("Read %d rows from %s", len(df), csv_path)

    schema = detect_schema(df)
    logger.info("Detected schema: %s", schema)

    if schema == "jigsaw":
        normalised = normalize_jigsaw(df)
    else:
        normalised = normalize_davidson(df)

    before = len(normalised)
    normalised = _apply_sample_cap(normalised, sample_size)
    after = len(normalised)

    if after < before:
        logger.info("Applied SAMPLE_SIZE cap: %d → %d rows", before, after)

    return normalised


# ── Main entry point ────────────────────────────────────────────────────────

def load_ground_truth(cfg: "Config", engine: "Engine") -> int:
    """
    Full ingestion pipeline: read configured CSV(s), normalise, write to DB.

    Behaviour:
      - Always loads ``cfg.dataset_csv_path`` (Jigsaw by default).
      - If ``cfg.davidson_csv_path`` is set (non-empty), loads and concatenates
        the Davidson dataset as well.
      - The combined DataFrame replaces the ``ground_truth`` table on every run
        (``if_exists="replace"`` per FR-3).
      - ``cfg.sample_size`` is applied independently per dataset so each
        contributes at most ``sample_size`` rows.

    Args:
        cfg:    Loaded Config instance.
        engine: SQLAlchemy Engine pointing at the target database.

    Returns:
        Total number of rows written to ``ground_truth``.

    Raises:
        FileNotFoundError: if a configured CSV path doesn't exist.
        ValueError:        if schema detection fails.
    """
    frames: list[pd.DataFrame] = []

    # ── Primary dataset (Jigsaw by default) ──────────────────────────────────
    if cfg.dataset_csv_path:
        logger.info("Ingesting primary dataset: %s", cfg.dataset_csv_path)
        jigsaw_df = ingest_csv(cfg.dataset_csv_path, cfg.sample_size)
        frames.append(jigsaw_df)

    # ── Secondary dataset (Davidson, optional) ────────────────────────────────
    if cfg.davidson_csv_path:
        logger.info("Ingesting Davidson dataset: %s", cfg.davidson_csv_path)
        davidson_df = ingest_csv(cfg.davidson_csv_path, cfg.sample_size)
        frames.append(davidson_df)

    if not frames:
        logger.warning("No CSV paths configured — ground_truth table not loaded.")
        return 0

    combined = pd.concat(frames, ignore_index=True)
    logger.info("Combined dataset: %d rows total", len(combined))

    # ── Write to PostgreSQL — append new rows only ────────────────────────────
    # We use append + deduplication so existing predictions are preserved.
    # Only rows whose item_id is not already in ground_truth are inserted.
    import sqlalchemy
    with engine.connect() as conn:
        try:
            existing_ids = pd.read_sql(
                sqlalchemy.text("SELECT item_id FROM ground_truth"),
                conn
            )["item_id"].tolist()
        except Exception:
            existing_ids = []

    if existing_ids:
        new_rows = combined[~combined["item_id"].isin(existing_ids)]
        logger.info(
            "Skipping %d already-ingested rows, adding %d new rows",
            len(combined) - len(new_rows), len(new_rows)
        )
        combined = new_rows

    if combined.empty:
        logger.info("No new rows to ingest.")
        return 0

    combined.to_sql(
        name="ground_truth",
        con=engine,
        if_exists="append",
        index=False,
        method="multi",
        chunksize=1000,
    )

    logger.info("Wrote %d rows to ground_truth table.", len(combined))
    return len(combined)
