"""
pipeline.py — end-to-end orchestrator for the AI Safety Evaluation Platform.

What it does
------------
1. Load config + connect to DB (or skip if SKIP_INGEST=true)
2. Ingest ground-truth CSV(s) into the ground_truth table
3. For each model × prompt_version combination:
   a. Run moderator over all ground_truth items → write to predictions
   b. Compute evaluate_run per category + "overall" → write to metrics
   c. Run threshold_sweep per category → write to threshold_metrics
   d. Write FP + FN items → error_analysis
4. Print a summary table to stdout

Run modes
---------
    # Full run (fixture mode, no API key needed)
    python pipeline.py

    # Skip CSV ingest (reuse whatever is already in DB)
    SKIP_INGEST=true python pipeline.py

    # Live API mode (set keys in .env)
    cp .env.example .env  # fill in OPENAI_API_KEY / GEMINI_API_KEY
    python pipeline.py

All configuration comes from environment variables (see config.py / .env.example).
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import sqlalchemy

# ── Make the workspace root importable when run directly ────────────────────
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import load_config
from db.db import get_engine, test_connection
from ingestion.ingest import load_ground_truth
from moderation.moderator import (
    RuleBasedModerator,
    HuggingFaceModerator,
    # OmniModerator,      # uncomment when OPENAI_API_KEY is set
    # OpenAIModerator,    # requires OPENAI_API_KEY + credits
    # GeminiModerator,    # requires GEMINI_API_KEY
)
from evaluation.metrics import evaluate_run, threshold_sweep

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pipeline")

# ── Constants ─────────────────────────────────────────────────────────────────

# HuggingFace classifiers don't use prompts — they run once with "default".
# Prompt A/B testing only applies to LLM-based moderators (GPT, Gemini).
PROMPT_VERSIONS = ["default"]

# All Jigsaw label categories + overall + Davidson's binary toxic
CATEGORIES = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]


# ── DB helpers ────────────────────────────────────────────────────────────────

def _write_predictions(predictions_df: pd.DataFrame, engine: Any) -> None:
    """Append a batch of predictions to the predictions table."""
    predictions_df.to_sql(
        "predictions",
        engine,
        if_exists="append",
        index=False,
        method="multi",
        chunksize=500,
    )
    logger.info("Wrote %d rows to predictions", len(predictions_df))


def _write_metrics(metrics_rows: list[dict], run_id: str,
                   prompt_version: str, engine: Any, model_name: str = "") -> None:
    """Append metric rows to the metrics table."""
    if not metrics_rows:
        return
    for row in metrics_rows:
        row["run_id"] = run_id
        row["model_name"] = model_name
    df = pd.DataFrame(metrics_rows)
    df.to_sql("metrics", engine, if_exists="append", index=False,
              method="multi", chunksize=500)
    logger.info("Wrote %d metric rows (model=%s prompt=%s)", len(df), model_name, prompt_version)


def _write_threshold_metrics(rows: list[dict], run_id: str, engine: Any) -> None:
    """Append threshold sweep rows to the threshold_metrics table."""
    if not rows:
        return
    for row in rows:
        row["run_id"] = run_id
    df = pd.DataFrame(rows)
    df.to_sql("threshold_metrics", engine, if_exists="append", index=False,
              method="multi", chunksize=500)
    logger.info("Wrote %d threshold_metrics rows", len(df))


def _write_error_analysis(ground_truth_df: pd.DataFrame,
                           predictions_df: pd.DataFrame,
                           run_id: str,
                           prompt_version: str,
                           engine: Any) -> None:
    """Find FP and FN items and write them to error_analysis."""
    # Filter to this prompt version and drop null decisions
    preds = predictions_df[
        (predictions_df["prompt_version"] == prompt_version) &
        predictions_df["decision"].notna()
    ].copy()

    # Join to ground truth on item_id
    merged = ground_truth_df[["item_id", "text", "toxic"]].merge(
        preds[["item_id", "decision", "confidence"]],
        on="item_id", how="inner"
    )
    merged = merged[merged["toxic"].notna()]

    error_rows = []
    for _, row in merged.iterrows():
        true_label = int(row["toxic"])
        pred_is_toxic = row["decision"] == "toxic"

        if true_label == 0 and pred_is_toxic:
            error_type = "FP"   # flagged as toxic but wasn't
        elif true_label == 1 and not pred_is_toxic:
            error_type = "FN"   # missed actual toxic content
        else:
            continue

        error_rows.append({
            "run_id":          run_id,
            "prompt_version":  prompt_version,
            "item_id":         row["item_id"],
            "text":            row["text"],
            "true_label":      true_label,
            "predicted_label": row["decision"],
            "confidence":      row["confidence"],
            "error_type":      error_type,
        })

    if error_rows:
        pd.DataFrame(error_rows).to_sql(
            "error_analysis", engine, if_exists="append",
            index=False, method="multi", chunksize=500
        )
        logger.info("Wrote %d error_analysis rows (prompt=%s)", len(error_rows), prompt_version)


# ── Summary printer ───────────────────────────────────────────────────────────

def _print_summary(summary_rows: list[dict]) -> None:
    """Print a formatted results table to stdout."""
    if not summary_rows:
        print("\n  (no results to display)\n")
        return

    cols  = ["model", "prompt_version", "category", "precision", "recall", "f1", "fpr", "fnr"]
    width = {c: max(len(c), max(len(str(r.get(c, ""))) for r in summary_rows)) for c in cols}

    divider = "  " + "  ".join("-" * width[c] for c in cols)
    header  = "  " + "  ".join(c.ljust(width[c]) for c in cols)

    print(f"\n{'='*80}")
    print("  PIPELINE RUN SUMMARY")
    print(f"{'='*80}")
    print(header)
    print(divider)
    for row in summary_rows:
        print("  " + "  ".join(str(row.get(c, "")).ljust(width[c]) for c in cols))
    print(f"\n  {len(summary_rows)} result row(s).\n")


# ── Adversarial robustness evaluation ────────────────────────────────────────

# Techniques to test — covers the main real-world evasion patterns
ADVERSARIAL_TECHNIQUES = [
    "original",             # clean baseline
    "spacing",              # f u c k
    "symbol_substitution",  # h@te, k1ll
    "homoglyph",            # Cyrillic/Greek lookalikes
    "zero_width",           # invisible chars inside words
    "repeated_chars",       # haaaaate
    "mixed_case",           # HaTe
    "missing_vowels",       # ht, stpd
    "character_swap",       # ahte
    "misspelling",          # haet, idiiot
]


def _run_adversarial_evaluation(
    ground_truth_df: pd.DataFrame,
    moderators: list,
    run_id: str,
    engine: Any,
    sample_size: int = 100,
) -> None:
    """
    Run adversarial robustness evaluation.

    For each toxic item (up to sample_size), generates adversarial variants
    using each technique, then runs every model on:
      - raw_text   (obfuscated)
      - normalized_text (after normalizer)

    Stores side-by-side results in adversarial_results table so we can
    answer: "Which attack technique causes the biggest recall drop, and
    does normalization recover it?"
    """
    from adversarial.generator import generate_variants
    from adversarial.normalizer import normalize

    # Only test on toxic items — we want to measure recall degradation
    toxic_rows = ground_truth_df[
        ground_truth_df["toxic"].fillna(0).astype(int) == 1
    ].head(sample_size)

    if toxic_rows.empty:
        logger.warning("No toxic rows found for adversarial evaluation.")
        return

    logger.info(
        "Adversarial evaluation: %d toxic items × %d techniques × %d models",
        len(toxic_rows), len(ADVERSARIAL_TECHNIQUES), len(moderators)
    )

    all_adv_rows: list[dict] = []

    for _, gt_row in toxic_rows.iterrows():
        item_id   = str(gt_row["item_id"])
        text      = str(gt_row["text"])
        true_label = int(gt_row["toxic"])

        # Generate all variants for this item
        variants = generate_variants(text, techniques=ADVERSARIAL_TECHNIQUES, seed=42)

        for variant in variants:
            raw_text  = variant["raw_text"]
            norm_text = variant["normalized_text"]
            technique = variant["obfuscation_type"]

            for moderator in moderators:
                model_name = moderator.model_name

                # Run on raw (obfuscated) text
                raw_decision, raw_conf = _moderate_single(
                    moderator, item_id, raw_text
                )
                # Run on normalized text
                norm_decision, norm_conf = _moderate_single(
                    moderator, item_id, norm_text
                )

                raw_correct  = (raw_decision  == "toxic") == bool(true_label)
                norm_correct = (norm_decision == "toxic") == bool(true_label)

                all_adv_rows.append({
                    "run_id":           run_id,
                    "item_id":          item_id,
                    "model_name":       model_name,
                    "prompt_version":   "default",
                    "obfuscation_type": technique,
                    "original_text":    text,
                    "raw_text":         raw_text,
                    "normalized_text":  norm_text,
                    "true_label":       true_label,
                    "raw_decision":     raw_decision,
                    "raw_confidence":   raw_conf,
                    "norm_decision":    norm_decision,
                    "norm_confidence":  norm_conf,
                    "raw_correct":      raw_correct,
                    "norm_correct":     norm_correct,
                    "norm_helped":      (not raw_correct and norm_correct),
                    "norm_hurt":        (raw_correct and not norm_correct),
                })

    if not all_adv_rows:
        return

    df = pd.DataFrame(all_adv_rows)
    df.to_sql(
        "adversarial_results", engine,
        if_exists="append", index=False,
        method="multi", chunksize=500,
    )
    logger.info("Wrote %d adversarial result rows", len(df))

    # Print quick summary
    summary = (
        df.groupby("obfuscation_type")
        .agg(
            raw_recall=("raw_correct",  "mean"),
            norm_recall=("norm_correct", "mean"),
        )
        .assign(delta=lambda x: x["norm_recall"] - x["raw_recall"])
        .sort_values("raw_recall")
    )
    print(f"\n{'='*65}")
    print("  ADVERSARIAL ROBUSTNESS SUMMARY (recall on toxic items)")
    print(f"{'='*65}")
    print(f"  {'technique':<25}  {'raw_recall':>10}  {'norm_recall':>11}  {'delta':>7}")
    print(f"  {'-'*25}  {'-'*10}  {'-'*11}  {'-'*7}")
    for tech, row in summary.iterrows():
        print(
            f"  {tech:<25}  {row['raw_recall']:>10.3f}  "
            f"{row['norm_recall']:>11.3f}  {row['delta']:>+7.3f}"
        )
    print()


def _moderate_single(
    moderator: Any,
    item_id: str,
    text: str,
) -> tuple[str | None, float | None]:
    """
    Run a single text through a moderator, returning (decision, confidence).
    Handles both per-item and batch moderators gracefully.
    """
    try:
        # HuggingFace and Omni override moderate() for batching.
        # For single items we build a one-row DataFrame.
        single_df = pd.DataFrame([{"item_id": item_id, "text": text}])
        result_df = moderator.moderate(single_df, "default")
        if result_df.empty:
            return None, None
        row = result_df.iloc[0]
        return row["decision"], row["confidence"]
    except Exception as exc:
        logger.debug("_moderate_single failed for %s: %s", item_id, exc)
        return None, None


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run_pipeline() -> None:
    """Execute the full pipeline end-to-end."""

    # ── 1. Config + DB connection ────────────────────────────────────────────
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    cfg = load_config()
    logger.info("Run ID: %s", cfg.run_id)
    logger.info("Fixture mode: %s", cfg.is_fixture_mode())

    engine = get_engine(cfg.database_url)

    if not test_connection(cfg.database_url):
        logger.error("Cannot connect to database at %s", cfg.database_url)
        logger.error("Make sure PostgreSQL is running and DATABASE_URL is correct.")
        sys.exit(1)

    # ── 2. Apply schema (idempotent) ─────────────────────────────────────────
    schema_path = _ROOT / "db" / "schema.sql"
    if schema_path.exists():
        logger.info("Applying schema from %s", schema_path)
        schema_sql = schema_path.read_text(encoding="utf-8")
        with engine.begin() as conn:
            # Split on semicolons and run each statement separately
            for stmt in schema_sql.split(";"):
                stmt = stmt.strip()
                if stmt and not stmt.startswith("--"):
                    try:
                        conn.execute(sqlalchemy.text(stmt))
                    except Exception as e:
                        # OR REPLACE VIEW may fail on first run — safe to ignore
                        logger.debug("Schema stmt skipped: %s", e)

    # ── 3. Ingest ────────────────────────────────────────────────────────────
    if cfg.skip_ingest:
        logger.info("SKIP_INGEST=true — loading existing ground_truth from DB")
        with engine.connect() as conn:
            ground_truth_df = pd.read_sql("SELECT * FROM ground_truth", conn)
        logger.info("Loaded %d rows from ground_truth", len(ground_truth_df))
    else:
        rows_loaded = load_ground_truth(cfg, engine)
        logger.info("Ingested %d rows into ground_truth", rows_loaded)
        with engine.connect() as conn:
            ground_truth_df = pd.read_sql("SELECT * FROM ground_truth", conn)

    if ground_truth_df.empty:
        logger.error("ground_truth table is empty — nothing to moderate.")
        sys.exit(1)

    logger.info("Ground truth: %d items", len(ground_truth_df))

    # ── 4. Build moderator instances ─────────────────────────────────────────
    moderators = [
        RuleBasedModerator(cfg, engine),
        HuggingFaceModerator(cfg, engine, hf_model_name="s-nlp/roberta_toxicity_classifier"),
        HuggingFaceModerator(cfg, engine, hf_model_name="Arsive/roberta-toxicity-classifier"),
        HuggingFaceModerator(cfg, engine, hf_model_name="textdetox/xlmr-large-toxicity-classifier-v2"),
    ]
    # Note: OpenAIModerator and GeminiModerator are available in moderator.py
    # but not run by default — add them back when API credits are available.

    summary_rows: list[dict] = []

    # ── 5. Moderate + evaluate per model × prompt_version ───────────────────
    for moderator in moderators:
        model_name = moderator.model_name
        logger.info("── Model: %s ──────────────────────────────", model_name)

        for prompt_version in PROMPT_VERSIONS:
            logger.info("  Prompt version: %s", prompt_version)

            # ── 5a. Run moderation ────────────────────────────────────────
            predictions_df = moderator.moderate(ground_truth_df, prompt_version)
            _write_predictions(predictions_df, engine)

            metrics_rows: list[dict] = []
            threshold_rows: list[dict] = []

            # ── 5b. Evaluate per category ─────────────────────────────────
            for category in CATEGORIES:
                # Skip categories that don't exist in this dataset
                if category not in ground_truth_df.columns:
                    continue
                if ground_truth_df[category].isna().all():
                    continue

                result = evaluate_run(
                    ground_truth_df, predictions_df,
                    category=category, prompt_version=prompt_version
                )
                metrics_rows.append(result)

                # Threshold sweep for this category
                sweep = threshold_sweep(
                    ground_truth_df, predictions_df,
                    category=category, prompt_version=prompt_version
                )
                threshold_rows.extend(sweep)

                # Collect for summary table
                summary_rows.append({
                    "model":          model_name,
                    "prompt_version": prompt_version,
                    "category":       category,
                    "precision":      f"{result['precision']:.3f}",
                    "recall":         f"{result['recall']:.3f}",
                    "f1":             f"{result['f1']:.3f}",
                    "fpr":            f"{result['fpr']:.3f}",
                    "fnr":            f"{result['fnr']:.3f}",
                })

            # ── 5c. Overall (across all categories combined) ──────────────
            # Use "toxic" as the headline overall metric
            if "toxic" in ground_truth_df.columns:
                overall = evaluate_run(
                    ground_truth_df, predictions_df,
                    category="toxic", prompt_version=prompt_version
                )
                overall["category"] = "overall"
                metrics_rows.append(overall)

            # ── 5d. Write metrics + threshold sweep ───────────────────────
            _write_metrics(metrics_rows, cfg.run_id, prompt_version, engine, model_name)
            _write_threshold_metrics(threshold_rows, cfg.run_id, engine)

            # ── 5e. Write error analysis (FP + FN) ───────────────────────
            _write_error_analysis(
                ground_truth_df, predictions_df,
                cfg.run_id, prompt_version, engine
            )

    # ── 6. Adversarial robustness evaluation ─────────────────────────────────
    logger.info("── Adversarial robustness evaluation ──────────────────────")
    _run_adversarial_evaluation(ground_truth_df, moderators, cfg.run_id, engine)

    # ── 7. Print summary ─────────────────────────────────────────────────────
    _print_summary(summary_rows)
    logger.info("Pipeline complete. Run ID: %s", cfg.run_id)


if __name__ == "__main__":
    run_pipeline()
