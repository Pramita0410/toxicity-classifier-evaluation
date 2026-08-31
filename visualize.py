"""
visualize.py — Generate charts from the trust_safety database.

Produces 4 PNG files in the dashboards/ folder:
    1. model_comparison.png     — Precision/Recall/F1 bar chart per model
    2. threshold_curve.png      — Precision-Recall tradeoff curve per model
    3. error_analysis.png       — FP vs FN counts per model/category
    4. category_breakdown.png   — Per-category F1 heatmap across models

Usage:
    python visualize.py                    # uses latest run in DB
    python visualize.py --run-id <uuid>    # specific run

Requirements:
    pip install matplotlib seaborn pandas sqlalchemy psycopg2-binary
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import matplotlib
matplotlib.use("Agg")   # non-interactive backend — works without a display
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np

OUTPUT_DIR = ROOT / "dashboards"
OUTPUT_DIR.mkdir(exist_ok=True)


# ── DB helpers ────────────────────────────────────────────────────────────────

def _get_engine(database_url: str):
    from sqlalchemy import create_engine
    return create_engine(database_url, pool_pre_ping=True)


def _latest_run_id(engine) -> str:
    import sqlalchemy
    with engine.connect() as conn:
        row = conn.execute(
            sqlalchemy.text("SELECT run_id FROM metrics ORDER BY id DESC LIMIT 1")
        ).fetchone()
    if row is None:
        raise RuntimeError("No runs found in the metrics table. Run pipeline.py first.")
    return str(row[0])


def _load_metrics(engine, run_id: str) -> pd.DataFrame:
    import sqlalchemy
    with engine.connect() as conn:
        df = pd.read_sql(
            sqlalchemy.text(
                "SELECT m.*, p.model_name "
                "FROM metrics m "
                "JOIN (SELECT DISTINCT run_id, model_name, prompt_version FROM predictions) p "
                "  ON p.run_id = m.run_id AND p.prompt_version = m.prompt_version "
                "WHERE m.run_id = :run_id"
            ),
            conn, params={"run_id": run_id}
        )
    return df


def _load_threshold_metrics(engine, run_id: str) -> pd.DataFrame:
    import sqlalchemy
    with engine.connect() as conn:
        df = pd.read_sql(
            sqlalchemy.text(
                "SELECT t.*, p.model_name "
                "FROM threshold_metrics t "
                "JOIN (SELECT DISTINCT run_id, model_name, prompt_version FROM predictions) p "
                "  ON p.run_id = t.run_id AND p.prompt_version = t.prompt_version "
                "WHERE t.run_id = :run_id"
            ),
            conn, params={"run_id": run_id}
        )
    return df


def _load_error_analysis(engine, run_id: str) -> pd.DataFrame:
    import sqlalchemy
    with engine.connect() as conn:
        df = pd.read_sql(
            sqlalchemy.text(
                "SELECT * FROM error_analysis WHERE run_id = :run_id"
            ),
            conn, params={"run_id": run_id}
        )
    return df


# ── Chart 1: Model comparison bar chart ──────────────────────────────────────

def plot_model_comparison(metrics_df: pd.DataFrame, run_id: str) -> Path:
    """
    Grouped bar chart: Precision / Recall / F1 for each model+prompt combo
    on the 'toxic' category.
    """
    df = metrics_df[metrics_df["category"] == "toxic"].copy()
    if df.empty:
        print("  [skip] No 'toxic' category rows found for model comparison chart.")
        return None

    # Create label: model + prompt_version
    df["label"] = df["model_name"] + "\n" + df["prompt_version"]
    df = df.sort_values("f1", ascending=False)

    labels    = df["label"].tolist()
    precision = df["precision"].tolist()
    recall    = df["recall"].tolist()
    f1        = df["f1"].tolist()

    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 1.5), 6))
    ax.bar(x - width, precision, width, label="Precision", color="#4C72B0", alpha=0.85)
    ax.bar(x,         recall,    width, label="Recall",    color="#DD8452", alpha=0.85)
    ax.bar(x + width, f1,        width, label="F1",        color="#55A868", alpha=0.85)

    ax.set_xlabel("Model + Prompt Strategy")
    ax.set_ylabel("Score")
    ax.set_title(f"Model Comparison — Toxic Category\nRun: {run_id[:8]}…")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out = OUTPUT_DIR / "model_comparison.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out}")
    return out


# ── Chart 2: Threshold curve (Precision-Recall tradeoff) ─────────────────────

def plot_threshold_curve(threshold_df: pd.DataFrame, run_id: str) -> Path:
    """
    Precision-Recall curve at varying confidence thresholds.
    One line per model+prompt_version, filtered to 'toxic' category.
    """
    df = threshold_df[threshold_df["category"] == "toxic"].copy()
    if df.empty:
        print("  [skip] No threshold_metrics rows found.")
        return None

    fig, ax = plt.subplots(figsize=(9, 6))

    colors = plt.cm.tab10.colors
    df["label"] = df["model_name"] + " / " + df["prompt_version"]
    labels = df["label"].unique()

    for i, label in enumerate(labels):
        subset = df[df["label"] == label].sort_values("threshold")
        ax.plot(
            subset["recall"], subset["precision"],
            marker="o", markersize=3,
            label=label, color=colors[i % len(colors)], linewidth=1.5
        )

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision-Recall Curve by Confidence Threshold\nRun: {run_id[:8]}…")
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(alpha=0.3)

    # Annotate the 0.5 threshold point for each line
    for i, label in enumerate(labels):
        subset = df[df["label"] == label]
        mid = subset.iloc[(subset["threshold"] - 0.5).abs().argsort().iloc[0]]
        ax.annotate(
            f"t=0.5",
            xy=(mid["recall"], mid["precision"]),
            fontsize=6, color=colors[i % len(colors)],
            xytext=(3, 3), textcoords="offset points"
        )

    plt.tight_layout()
    out = OUTPUT_DIR / "threshold_curve.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out}")
    return out


# ── Chart 3: FP vs FN error analysis ─────────────────────────────────────────

def plot_error_analysis(error_df: pd.DataFrame, metrics_df: pd.DataFrame, run_id: str) -> Path:
    """
    Stacked bar: FP (over-moderation) vs FN (missed harm) per model+prompt.
    """
    if error_df.empty:
        print("  [skip] No error_analysis rows found.")
        return None

    # Join with metrics to get model_name
    model_map = (
        metrics_df[["prompt_version", "model_name"]]
        .drop_duplicates()
        .set_index("prompt_version")["model_name"]
        .to_dict()
    )
    error_df = error_df.copy()
    error_df["model_name"] = error_df["prompt_version"].map(model_map).fillna("unknown")
    error_df["label"] = error_df["model_name"] + "\n" + error_df["prompt_version"]

    counts = (
        error_df.groupby(["label", "error_type"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )

    labels = counts["label"].tolist()
    fp = counts.get("FP", pd.Series([0] * len(counts))).tolist()
    fn = counts.get("FN", pd.Series([0] * len(counts))).tolist()

    x = np.arange(len(labels))
    width = 0.4

    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 1.5), 6))
    ax.bar(x, fp, width, label="False Positives (over-moderation)", color="#C44E52", alpha=0.85)
    ax.bar(x, fn, width, bottom=fp, label="False Negatives (missed harm)", color="#8172B2", alpha=0.85)

    ax.set_xlabel("Model + Prompt Strategy")
    ax.set_ylabel("Error Count")
    ax.set_title(f"False Positives vs False Negatives\nRun: {run_id[:8]}…")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out = OUTPUT_DIR / "error_analysis.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out}")
    return out


# ── Chart 4: Category breakdown heatmap ──────────────────────────────────────

def plot_category_breakdown(metrics_df: pd.DataFrame, run_id: str) -> Path:
    """
    Heatmap: F1 score per (model+prompt) row × category column.
    Shows at a glance which harm categories each model handles best/worst.
    """
    CATEGORIES = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

    df = metrics_df[metrics_df["category"].isin(CATEGORIES)].copy()
    if df.empty:
        print("  [skip] No category rows found for heatmap.")
        return None

    df["label"] = df["model_name"] + " / " + df["prompt_version"]

    pivot = df.pivot_table(
        index="label", columns="category", values="f1", aggfunc="mean"
    ).reindex(columns=CATEGORIES)

    # Sort rows by mean F1
    pivot = pivot.loc[pivot.mean(axis=1).sort_values(ascending=False).index]

    fig, ax = plt.subplots(figsize=(10, max(4, len(pivot) * 0.6)))

    im = ax.imshow(pivot.values.astype(float), aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=8)
    ax.set_title(f"F1 Score by Model × Category\nRun: {run_id[:8]}…")

    # Annotate cells
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=7, color="black" if 0.3 < val < 0.8 else "white")

    plt.colorbar(im, ax=ax, label="F1 Score")
    plt.tight_layout()
    out = OUTPUT_DIR / "category_breakdown.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out}")
    return out


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate visualization charts from the trust_safety database."
    )
    parser.add_argument("--run-id", default=None, help="Run ID to visualize (default: latest)")
    args = parser.parse_args()

    # Load config
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    from config import load_config
    cfg = load_config()
    engine = _get_engine(cfg.database_url)

    run_id = args.run_id or _latest_run_id(engine)
    print(f"\nVisualizing run: {run_id}\n")

    print("Loading data from DB...")
    metrics_df   = _load_metrics(engine, run_id)
    threshold_df = _load_threshold_metrics(engine, run_id)
    error_df     = _load_error_analysis(engine, run_id)

    print(f"  metrics rows:    {len(metrics_df)}")
    print(f"  threshold rows:  {len(threshold_df)}")
    print(f"  error rows:      {len(error_df)}")

    if metrics_df.empty:
        print("\nNo data found for this run. Run pipeline.py first.")
        sys.exit(1)

    print("\nGenerating charts...")
    plot_model_comparison(metrics_df, run_id)
    plot_threshold_curve(threshold_df, run_id)
    plot_error_analysis(error_df, metrics_df, run_id)
    plot_category_breakdown(metrics_df, run_id)

    print(f"\nAll charts saved to: {OUTPUT_DIR}/")
    print("Open the PNG files to view results.\n")


if __name__ == "__main__":
    main()
