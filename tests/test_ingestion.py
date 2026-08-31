"""
tests/test_ingestion.py — unit tests for ingestion/ingest.py

Covers:
  - Schema detection for Jigsaw and Davidson formats
  - Null label rows are preserved (not filtered) after ingestion
  - SAMPLE_SIZE cap is respected
  - load_ground_truth writes to DB (SQLite in-memory, no Postgres required)

No real dataset, no network, no API keys needed.
"""

from __future__ import annotations

import io
import os
import pathlib

import pandas as pd
import pytest
from sqlalchemy import create_engine, inspect, text

# Make sure the project root is importable regardless of how pytest is invoked
PROJECT_ROOT = pathlib.Path(__file__).parent.parent
import sys
sys.path.insert(0, str(PROJECT_ROOT))

from ingestion.ingest import (
    detect_schema,
    normalize_davidson,
    normalize_jigsaw,
    ingest_csv,
    load_ground_truth,
    _apply_sample_cap,
)

# ── Fixture paths ────────────────────────────────────────────────────────────

FIXTURE_DIR = PROJECT_ROOT / "tests" / "fixtures"
SAMPLE_GROUND_TRUTH_CSV = FIXTURE_DIR / "sample_ground_truth.csv"

# ── Helpers to build raw-format DataFrames for testing ──────────────────────

def _make_jigsaw_df(n: int = 5) -> pd.DataFrame:
    """Minimal Jigsaw-format DataFrame (raw, pre-normalisation)."""
    _toxic    = [0, 1, 0, 1, 0]
    _severe   = [0, 0, 0, 1, 0]
    _obscene  = [0, 1, 0, 1, 0]
    _threat   = [0, 0, 0, 0, 0]
    _insult   = [0, 1, 0, 1, 0]
    _identity = [0, 0, 0, 0, 0]
    return pd.DataFrame(
        {
            "id": [f"jigsaw_{i}" for i in range(n)],
            "comment_text": [f"Comment text {i}" for i in range(n)],
            "toxic":         [_toxic[i % 5]    for i in range(n)],
            "severe_toxic":  [_severe[i % 5]   for i in range(n)],
            "obscene":       [_obscene[i % 5]  for i in range(n)],
            "threat":        [_threat[i % 5]   for i in range(n)],
            "insult":        [_insult[i % 5]   for i in range(n)],
            "identity_hate": [_identity[i % 5] for i in range(n)],
        }
    )


def _make_davidson_df(n: int = 5) -> pd.DataFrame:
    """Minimal Davidson-format DataFrame (raw, pre-normalisation)."""
    _count  = [3, 4, 2, 5, 1]
    _hate   = [1, 0, 0, 0, 0]
    _off    = [0, 1, 0, 1, 0]
    _neither= [0, 0, 1, 0, 1]
    _class  = [0, 1, 2, 1, 2]
    return pd.DataFrame(
        {
            "count":              [_count[i % 5]   for i in range(n)],
            "hate_speech":        [_hate[i % 5]    for i in range(n)],
            "offensive_language": [_off[i % 5]     for i in range(n)],
            "neither":            [_neither[i % 5] for i in range(n)],
            "class":              [_class[i % 5]   for i in range(n)],
            "tweet": [f"Tweet text {i}" for i in range(n)],
        }
    )


# ── Schema detection tests ───────────────────────────────────────────────────

class TestDetectSchema:
    def test_detects_jigsaw_by_comment_text_column(self):
        df = _make_jigsaw_df()
        assert detect_schema(df) == "jigsaw"

    def test_detects_davidson_by_tweet_column(self):
        df = _make_davidson_df()
        assert detect_schema(df) == "davidson"

    def test_raises_on_unknown_schema(self):
        df = pd.DataFrame({"unknown_column": [1, 2, 3], "another": ["a", "b", "c"]})
        with pytest.raises(ValueError, match="Cannot detect schema"):
            detect_schema(df)

    def test_jigsaw_detection_ignores_extra_columns(self):
        df = _make_jigsaw_df()
        df["extra_col"] = "extra"
        assert detect_schema(df) == "jigsaw"

    def test_davidson_detection_ignores_extra_columns(self):
        df = _make_davidson_df()
        df["extra_col"] = "extra"
        assert detect_schema(df) == "davidson"


# ── Normalisation tests ──────────────────────────────────────────────────────

CANONICAL_COLUMNS = [
    "item_id", "text", "toxic", "severe_toxic", "obscene",
    "threat", "insult", "identity_hate", "dataset_source",
]


class TestNormalizeJigsaw:
    def test_output_has_all_canonical_columns(self):
        out = normalize_jigsaw(_make_jigsaw_df())
        assert list(out.columns) == CANONICAL_COLUMNS

    def test_dataset_source_is_jigsaw(self):
        out = normalize_jigsaw(_make_jigsaw_df())
        assert (out["dataset_source"] == "jigsaw").all()

    def test_item_id_is_stringified(self):
        df = _make_jigsaw_df(3)
        out = normalize_jigsaw(df)
        # All item_ids should be strings
        assert out["item_id"].dtype == object
        assert list(out["item_id"]) == ["jigsaw_0", "jigsaw_1", "jigsaw_2"]

    def test_text_comes_from_comment_text(self):
        df = _make_jigsaw_df(3)
        out = normalize_jigsaw(df)
        assert list(out["text"]) == ["Comment text 0", "Comment text 1", "Comment text 2"]

    def test_null_toxic_label_is_preserved(self):
        """Null labels must be kept — filtering happens at evaluation time (FR-12)."""
        df = _make_jigsaw_df(3)
        df.loc[1, "toxic"] = None   # inject a null
        out = normalize_jigsaw(df)
        assert len(out) == 3
        assert pd.isna(out.loc[1, "toxic"])

    def test_row_count_unchanged(self):
        df = _make_jigsaw_df(5)
        out = normalize_jigsaw(df)
        assert len(out) == 5


class TestNormalizeDavidson:
    def test_output_has_all_canonical_columns(self):
        out = normalize_davidson(_make_davidson_df())
        assert list(out.columns) == CANONICAL_COLUMNS

    def test_dataset_source_is_davidson(self):
        out = normalize_davidson(_make_davidson_df())
        assert (out["dataset_source"] == "davidson").all()

    def test_class_0_and_1_map_to_toxic_1(self):
        """Davidson class 0 (hate) and 1 (offensive) → toxic=1."""
        df = _make_davidson_df(5)
        out = normalize_davidson(df)
        # class 0 → toxic 1
        assert out.loc[0, "toxic"] == 1
        # class 1 → toxic 1
        assert out.loc[1, "toxic"] == 1

    def test_class_2_maps_to_toxic_0(self):
        """Davidson class 2 (neither) → toxic=0."""
        df = _make_davidson_df(5)
        out = normalize_davidson(df)
        # class 2 at index 2 → toxic 0
        assert out.loc[2, "toxic"] == 0

    def test_jigsaw_specific_labels_are_null(self):
        """Davidson rows should have NaN for Jigsaw-only label columns."""
        out = normalize_davidson(_make_davidson_df(3))
        for col in ("severe_toxic", "obscene", "threat", "insult", "identity_hate"):
            assert out[col].isna().all(), f"Expected all NaN for {col}"

    def test_item_id_has_davidson_prefix(self):
        out = normalize_davidson(_make_davidson_df(3))
        for item_id in out["item_id"]:
            assert item_id.startswith("davidson_"), f"Unexpected item_id: {item_id}"

    def test_row_count_unchanged(self):
        df = _make_davidson_df(5)
        out = normalize_davidson(df)
        assert len(out) == 5


# ── Null label preservation tests ────────────────────────────────────────────

class TestNullLabelPreservation:
    """
    Null labels must survive ingestion unchanged (FR-12: filtering at eval time).
    The fixture CSV contains jigsaw_13 which has a null toxic label.
    """

    def test_fixture_csv_has_null_label_row(self):
        df = pd.read_csv(SAMPLE_GROUND_TRUTH_CSV)
        null_rows = df[df["toxic"].isna()]
        assert len(null_rows) >= 1, "Expected at least one null-label row in fixture CSV"

    def test_null_label_preserved_after_normalize_jigsaw(self):
        df = _make_jigsaw_df(3)
        df.loc[1, "toxic"] = None
        out = normalize_jigsaw(df)
        assert pd.isna(out.loc[1, "toxic"])
        assert len(out) == 3  # row not dropped

    def test_null_label_preserved_in_db(self):
        """load_ground_truth must not silently drop null-label rows."""
        # Build a tiny Jigsaw CSV with one null-label row
        raw_data = io.StringIO(
            "id,comment_text,toxic,severe_toxic,obscene,threat,insult,identity_hate\n"
            "row1,Clean comment,0,0,0,0,0,0\n"
            "row2,Toxic comment,1,0,0,0,0,0\n"
            "row3,Null label comment,,0,0,0,0,0\n"
        )
        # Write to a temp CSV
        import tempfile, os
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False, encoding="utf-8"
        ) as f:
            f.write(raw_data.read())
            tmp_path = f.name

        try:
            engine = create_engine("sqlite:///:memory:")
            cfg = _make_config(dataset_csv_path=tmp_path, sample_size=0)
            rows = load_ground_truth(cfg, engine)

            assert rows == 3, f"Expected 3 rows written, got {rows}"

            with engine.connect() as conn:
                result = conn.execute(
                    text("SELECT toxic FROM ground_truth WHERE item_id = 'row3'")
                )
                row = result.fetchone()
            assert row is not None
            assert row[0] is None, "Null toxic label must be NULL in DB, not dropped"
        finally:
            os.unlink(tmp_path)


# ── SAMPLE_SIZE cap tests ────────────────────────────────────────────────────

class TestSampleSizeCap:
    def test_no_cap_when_sample_size_zero(self):
        df = _make_jigsaw_df(10)
        out = _apply_sample_cap(df, sample_size=0)
        assert len(out) == 10

    def test_cap_applied_when_sample_size_less_than_rows(self):
        df = _make_jigsaw_df(10)
        out = _apply_sample_cap(df, sample_size=4)
        assert len(out) == 4

    def test_no_cap_when_sample_size_equals_rows(self):
        df = _make_jigsaw_df(5)
        out = _apply_sample_cap(df, sample_size=5)
        assert len(out) == 5

    def test_no_cap_when_sample_size_exceeds_rows(self):
        df = _make_jigsaw_df(5)
        out = _apply_sample_cap(df, sample_size=100)
        assert len(out) == 5

    def test_sampling_is_reproducible(self):
        """Same seed=42 must always pick same rows."""
        df = _make_jigsaw_df(10)
        out1 = _apply_sample_cap(df, sample_size=4)
        out2 = _apply_sample_cap(df, sample_size=4)
        assert list(out1.index) == list(out2.index)

    def test_ingest_csv_respects_sample_size(self, tmp_path):
        """End-to-end: ingest_csv with sample_size cap limits rows returned."""
        # Write a Jigsaw CSV with 10 rows to a temp file
        df = _make_jigsaw_df(10)
        csv_path = tmp_path / "jigsaw_test.csv"
        df.to_csv(csv_path, index=False)

        result = ingest_csv(str(csv_path), sample_size=3)
        assert len(result) == 3

    def test_ingest_csv_returns_all_rows_when_no_cap(self, tmp_path):
        df = _make_jigsaw_df(10)
        csv_path = tmp_path / "jigsaw_test.csv"
        df.to_csv(csv_path, index=False)

        result = ingest_csv(str(csv_path), sample_size=0)
        assert len(result) == 10


# ── load_ground_truth DB integration tests ───────────────────────────────────

class TestLoadGroundTruth:
    """
    Uses SQLite in-memory engine — no PostgreSQL required.
    """

    def test_writes_rows_to_ground_truth_table(self, tmp_path):
        df = _make_jigsaw_df(5)
        csv_path = tmp_path / "jigsaw.csv"
        df.to_csv(csv_path, index=False)

        engine = create_engine("sqlite:///:memory:")
        cfg = _make_config(dataset_csv_path=str(csv_path))
        rows = load_ground_truth(cfg, engine)

        assert rows == 5
        with engine.connect() as conn:
            count = conn.execute(text("SELECT COUNT(*) FROM ground_truth")).scalar()
        assert count == 5

    def test_ground_truth_table_has_canonical_columns(self, tmp_path):
        df = _make_jigsaw_df(3)
        csv_path = tmp_path / "jigsaw.csv"
        df.to_csv(csv_path, index=False)

        engine = create_engine("sqlite:///:memory:")
        cfg = _make_config(dataset_csv_path=str(csv_path))
        load_ground_truth(cfg, engine)

        inspector = inspect(engine)
        cols = {c["name"] for c in inspector.get_columns("ground_truth")}
        for col in CANONICAL_COLUMNS:
            assert col in cols, f"Column '{col}' missing from ground_truth table"

    def test_combined_jigsaw_and_davidson(self, tmp_path):
        jigsaw_df = _make_jigsaw_df(5)
        jigsaw_csv = tmp_path / "jigsaw.csv"
        jigsaw_df.to_csv(jigsaw_csv, index=False)

        davidson_df = _make_davidson_df(3)
        davidson_csv = tmp_path / "davidson.csv"
        davidson_df.to_csv(davidson_csv, index=False)

        engine = create_engine("sqlite:///:memory:")
        cfg = _make_config(
            dataset_csv_path=str(jigsaw_csv),
            davidson_csv_path=str(davidson_csv),
        )
        rows = load_ground_truth(cfg, engine)

        assert rows == 8  # 5 jigsaw + 3 davidson

    def test_sample_size_cap_applied_per_dataset(self, tmp_path):
        jigsaw_df = _make_jigsaw_df(10)
        jigsaw_csv = tmp_path / "jigsaw.csv"
        jigsaw_df.to_csv(jigsaw_csv, index=False)

        davidson_df = _make_davidson_df(10)
        davidson_csv = tmp_path / "davidson.csv"
        davidson_df.to_csv(davidson_csv, index=False)

        engine = create_engine("sqlite:///:memory:")
        cfg = _make_config(
            dataset_csv_path=str(jigsaw_csv),
            davidson_csv_path=str(davidson_csv),
            sample_size=4,
        )
        rows = load_ground_truth(cfg, engine)

        # Each dataset capped at 4 → total 8
        assert rows == 8

    def test_replaces_existing_data_on_rerun(self, tmp_path):
        """if_exists='replace' means a second run overwrites the table."""
        df = _make_jigsaw_df(5)
        csv_path = tmp_path / "jigsaw.csv"
        df.to_csv(csv_path, index=False)

        engine = create_engine("sqlite:///:memory:")
        cfg = _make_config(dataset_csv_path=str(csv_path))

        load_ground_truth(cfg, engine)  # first run
        rows2 = load_ground_truth(cfg, engine)  # second run

        assert rows2 == 5
        with engine.connect() as conn:
            count = conn.execute(text("SELECT COUNT(*) FROM ground_truth")).scalar()
        assert count == 5  # not 10

    def test_returns_zero_with_no_csv_paths(self):
        engine = create_engine("sqlite:///:memory:")
        cfg = _make_config(dataset_csv_path="", davidson_csv_path="")
        rows = load_ground_truth(cfg, engine)
        assert rows == 0


# ── ingest_csv schema detection integration ───────────────────────────────────

class TestIngestCsvSchemaDetection:
    def test_ingest_jigsaw_csv(self, tmp_path):
        df = _make_jigsaw_df(5)
        csv_path = tmp_path / "jigsaw.csv"
        df.to_csv(csv_path, index=False)

        result = ingest_csv(str(csv_path))
        assert (result["dataset_source"] == "jigsaw").all()
        assert list(result.columns) == CANONICAL_COLUMNS

    def test_ingest_davidson_csv(self, tmp_path):
        df = _make_davidson_df(5)
        csv_path = tmp_path / "davidson.csv"
        df.to_csv(csv_path, index=False)

        result = ingest_csv(str(csv_path))
        assert (result["dataset_source"] == "davidson").all()
        assert list(result.columns) == CANONICAL_COLUMNS

    def test_raises_on_bad_csv_path(self):
        with pytest.raises(Exception):
            ingest_csv("nonexistent/path/file.csv")


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_config(
    dataset_csv_path: str = "",
    davidson_csv_path: str = "",
    sample_size: int = 0,
) -> "Config":
    """Build a minimal Config without touching env vars."""
    from config import Config

    cfg = Config.__new__(Config)
    object.__setattr__(cfg, "dataset_csv_path", dataset_csv_path)
    object.__setattr__(cfg, "davidson_csv_path", davidson_csv_path)
    object.__setattr__(cfg, "sample_size", sample_size)
    object.__setattr__(cfg, "database_url", "sqlite:///:memory:")
    object.__setattr__(cfg, "openai_api_key", "")
    object.__setattr__(cfg, "gemini_api_key", "")
    object.__setattr__(cfg, "openai_model", "gpt-4o-mini")
    object.__setattr__(cfg, "gemini_model", "gemini-2.5-flash")
    object.__setattr__(cfg, "use_fixtures", True)
    object.__setattr__(cfg, "fixture_path", "tests/fixtures/llm_responses.json")
    object.__setattr__(cfg, "skip_ingest", False)
    object.__setattr__(cfg, "run_id", "test-run-id")
    return cfg
