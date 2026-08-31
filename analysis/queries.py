"""
analysis/queries.py — named SQL analytical queries for the AI Safety Evaluation Platform.

Each query answers a specific research question about the stored results.
Queries run directly against PostgreSQL using the DATABASE_URL from config.py.

Usage (CLI)
-----------
    # List all available queries
    python -m analysis.queries --list

    # Run a specific query
    python -m analysis.queries --query best_prompt_by_metric
    python -m analysis.queries --query adversarial_robustness
    python -m analysis.queries --query threshold_operating_points --run-id <uuid>

    # Run all queries
    python -m analysis.queries --all

Usage (Python)
--------------
    from analysis.queries import run_query, QUERIES
    rows = run_query("best_prompt_by_metric", engine)

Available queries
-----------------
    1. best_prompt_by_metric        Which prompt version achieves best F1/recall per category?
    2. worst_categories             Which harm categories have the lowest F1 scores?
    3. model_comparison             Side-by-side precision/recall/F1 for all models + prompts.
    4. false_positive_samples       Sample FP items with highest confidence (over-moderation).
    5. false_negative_samples       Sample FN items — the missed harm cases.
    6. threshold_operating_points   At what threshold does each model hit 90% recall?
    7. adversarial_robustness       Which obfuscation technique degrades recall most?
    8. normalization_impact         Does normalization help? Delta in accuracy per technique.
    9. multilingual_breakdown       How does each model perform on multilingual adversarial items?
   10. run_summary                  One-line summary of a full pipeline run.
"""

from __future__ import annotations

import argparse
import sys
import textwrap
from typing import Any

#  Query definitions 
# Each entry: {name, question, sql, params_help}

QUERIES: dict[str, dict[str, str]] = {

  # Worst categories 
    "worst_categories": {
        "question": "Which harm categories have the lowest F1 scores? (hardest to moderate)",
        "sql": """
            SELECT
                category,
                ROUND(AVG(f1)::NUMERIC, 4)        AS avg_f1,
                ROUND(AVG(recall)::NUMERIC, 4)    AS avg_recall,
                ROUND(AVG(precision)::NUMERIC, 4) AS avg_precision,
                ROUND(AVG(fnr)::NUMERIC, 4)       AS avg_fnr,
                COUNT(DISTINCT run_id)             AS run_count
            FROM metrics
            GROUP BY category
            ORDER BY avg_f1 ASC;
        """,
        "params_help": "No parameters. Aggregates across all runs.",
    },

    # Model comparison 
    "model_comparison": {
        "question": "Side-by-side precision/recall/F1 for every model + prompt combination.",
        "sql": """
            SELECT
                p.model_name,
                m.prompt_version,
                m.category,
                ROUND(m.precision::NUMERIC, 4) AS precision,
                ROUND(m.recall::NUMERIC, 4)    AS recall,
                ROUND(m.f1::NUMERIC, 4)        AS f1,
                ROUND(m.fpr::NUMERIC, 4)       AS fpr,
                ROUND(m.fnr::NUMERIC, 4)       AS fnr
            FROM metrics m
            JOIN (
                SELECT DISTINCT run_id, model_name, prompt_version
                FROM predictions
            ) p ON p.run_id = m.run_id AND p.prompt_version = m.prompt_version
            WHERE m.run_id = (
                SELECT run_id FROM metrics ORDER BY id DESC LIMIT 1
            )
            ORDER BY m.category, m.f1 DESC;
        """,
        "params_help": "No parameters. Uses the most recent run.",
    },

    # False positive samples 
    "false_positive_samples": {
        "question": "Which items were flagged as toxic but aren't? (highest-confidence FPs — over-moderation)",
        "sql": """
            SELECT
                item_id,
                LEFT(text, 120)         AS text_preview,
                true_label,
                predicted_label,
                ROUND(confidence::NUMERIC, 3) AS confidence,
                prompt_version,
                error_type
            FROM error_analysis
            WHERE error_type = 'FP'
              AND run_id = (
                SELECT run_id FROM error_analysis ORDER BY id DESC LIMIT 1
              )
            ORDER BY confidence DESC
            LIMIT 20;
        """,
        "params_help": "No parameters. Returns top 20 highest-confidence FPs from the latest run.",
    },

    #False negative samples 
    "false_negative_samples": {
        "question": "Which toxic items were MISSED by the moderator? (highest-confidence FNs — missed harm)",
        "sql": """
            SELECT
                item_id,
                LEFT(text, 120)         AS text_preview,
                true_label,
                predicted_label,
                ROUND(confidence::NUMERIC, 3) AS confidence,
                prompt_version,
                error_type
            FROM error_analysis
            WHERE error_type = 'FN'
              AND run_id = (
                SELECT run_id FROM error_analysis ORDER BY id DESC LIMIT 1
              )
            ORDER BY confidence ASC   -- lowest confidence on toxic items = most "confused"
            LIMIT 20;
        """,
        "params_help": "No parameters. Returns top 20 most-confused FN items from the latest run.",
    },

    # Threshold operating points 
    "threshold_operating_points": {
        "question": "At what confidence threshold does each prompt version first reach 90% recall?",
        "sql": """
            WITH ranked AS (
                SELECT
                    prompt_version,
                    category,
                    threshold,
                    ROUND(recall::NUMERIC, 4)    AS recall,
                    ROUND(precision::NUMERIC, 4) AS precision,
                    ROUND(f1::NUMERIC, 4)        AS f1,
                    ROUND(fpr::NUMERIC, 4)       AS fpr,
                    ROW_NUMBER() OVER (
                        PARTITION BY prompt_version, category
                        ORDER BY threshold ASC
                    ) AS rn
                FROM threshold_metrics
                WHERE recall >= 0.90
                  AND run_id = (
                    SELECT run_id FROM threshold_metrics ORDER BY id DESC LIMIT 1
                  )
            )
            SELECT
                prompt_version,
                category,
                threshold   AS min_threshold_for_90pct_recall,
                recall,
                precision,
                f1,
                fpr
            FROM ranked
            WHERE rn = 1
            ORDER BY category, prompt_version;
        """,
        "params_help": "No parameters. Uses most recent threshold sweep run.",
    },

    # Adversarial robustness 
    "adversarial_robustness": {
        "question": "Which obfuscation technique degrades recall the most? (raw vs baseline)",
        "sql": """
            SELECT
                obfuscation_type,
                model_name,
                prompt_version,
                n,
                ROUND(raw_recall::NUMERIC, 4)   AS raw_recall,
                ROUND(norm_recall::NUMERIC, 4)  AS norm_recall,
                ROUND((norm_recall - raw_recall)::NUMERIC, 4) AS recall_delta,
                norm_helped_count,
                norm_hurt_count
            FROM adversarial_summary
            ORDER BY raw_recall ASC;   -- worst raw recall first = highest-impact techniques
        """,
        "params_help": "No parameters. Reads from the adversarial_summary view.",
    },

    # Normalization impact 
    "normalization_impact": {
        "question": "Does normalization actually help? Delta in accuracy per obfuscation technique.",
        "sql": """
            SELECT
                obfuscation_type,
                model_name,
                n,
                ROUND(raw_accuracy::NUMERIC, 4)       AS raw_accuracy,
                ROUND(norm_accuracy::NUMERIC, 4)      AS norm_accuracy,
                ROUND(accuracy_delta::NUMERIC, 4)     AS accuracy_delta,
                CASE
                    WHEN accuracy_delta > 0.05  THEN 'normalization_helps'
                    WHEN accuracy_delta < -0.02 THEN 'normalization_hurts'
                    ELSE                             'no_significant_difference'
                END                                    AS verdict,
                norm_helped_count,
                norm_hurt_count
            FROM adversarial_summary
            ORDER BY accuracy_delta DESC;
        """,
        "params_help": "No parameters. Reads from the adversarial_summary view.",
    },

    # Multilingual breakdown 
    "multilingual_breakdown": {
        "question": "How does each model perform on multilingual and code-switching adversarial items?",
        "sql": """
            SELECT
                obfuscation_type,
                model_name,
                prompt_version,
                COUNT(*)                                                           AS n,
                ROUND(
                    SUM(CASE WHEN raw_correct  THEN 1.0 ELSE 0.0 END)::NUMERIC
                    / NULLIF(COUNT(*), 0), 4
                )                                                                  AS raw_accuracy,
                ROUND(
                    SUM(CASE WHEN norm_correct THEN 1.0 ELSE 0.0 END)::NUMERIC
                    / NULLIF(COUNT(*), 0), 4
                )                                                                  AS norm_accuracy
            FROM adversarial_results
            WHERE obfuscation_type IN ('multilingual', 'code_switching')
            GROUP BY obfuscation_type, model_name, prompt_version
            ORDER BY raw_accuracy ASC;
        """,
        "params_help": "No parameters. Filters to multilingual and code_switching rows.",
    },

    # Run summary 
    "run_summary": {
        "question": "One-line summary of every pipeline run: model, prompts, overall precision/recall/F1.",
        "sql": """
            SELECT
                m.run_id,
                p.model_name,
                m.prompt_version,
                ROUND(m.precision::NUMERIC, 4) AS precision,
                ROUND(m.recall::NUMERIC, 4)    AS recall,
                ROUND(m.f1::NUMERIC, 4)        AS f1,
                COUNT(DISTINCT pred.item_id)    AS items_moderated
            FROM metrics m
            JOIN (
                SELECT DISTINCT run_id, model_name, prompt_version
                FROM predictions
            ) p ON p.run_id = m.run_id AND p.prompt_version = m.prompt_version
            LEFT JOIN predictions pred
                ON pred.run_id = m.run_id AND pred.prompt_version = m.prompt_version
            WHERE m.category = 'toxic'   -- overall toxic label as the headline metric
            GROUP BY m.run_id, p.model_name, m.prompt_version,
                     m.precision, m.recall, m.f1
            ORDER BY m.f1 DESC;
        """,
        "params_help": "No parameters. Shows all runs sorted by F1 descending.",
    },
}


#  Runner 

def run_query(
    name: str,
    engine: Any,
    *,
    params: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """
    Execute a named query against the database and return rows as dicts.

    Parameters
    ----------
    name:
        Key from the QUERIES dict (e.g. "adversarial_robustness").
    engine:
        SQLAlchemy Engine pointing at the target database.
    params:
        Optional dict of bind parameters for parameterized queries.
        Not currently used by the built-in queries but available for
        custom extensions.

    Returns
    -------
    list[dict]
        One dict per result row.

    Raises
    ------
    KeyError:  if *name* is not in QUERIES.
    Exception: any SQLAlchemy / psycopg2 error propagates to the caller.
    """
    import sqlalchemy

    if name not in QUERIES:
        raise KeyError(
            f"Unknown query: {name!r}. Available: {list(QUERIES)}"
        )

    sql = QUERIES[name]["sql"]

    with engine.connect() as conn:
        result = conn.execute(sqlalchemy.text(sql), params or {})
        keys = list(result.keys())
        return [dict(zip(keys, row)) for row in result.fetchall()]


def print_results(name: str, rows: list[dict[str, Any]]) -> None:
    """Pretty-print query results to stdout as an aligned table."""
    entry = QUERIES[name]
    print(f"\n{'='*72}")
    print(f"  Query : {name}")
    print(f"  Q     : {entry['question']}")
    print(f"{'='*72}")

    if not rows:
        print("  (no rows returned)\n")
        return

    # Build column widths
    cols = list(rows[0].keys())
    widths = {c: max(len(str(c)), max(len(str(r.get(c, ""))) for r in rows)) for c in cols}

    header = "  " + "  ".join(str(c).ljust(widths[c]) for c in cols)
    divider = "  " + "  ".join("-" * widths[c] for c in cols)
    print(header)
    print(divider)
    for row in rows:
        print("  " + "  ".join(str(row.get(c, "")).ljust(widths[c]) for c in cols))
    print(f"\n  {len(rows)} row(s) returned.\n")


# ── CLI entry point ───────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run named analytical SQL queries against the trust_safety database.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(
            """
            Examples:
              python -m analysis.queries --list
              python -m analysis.queries --query adversarial_robustness
              python -m analysis.queries --query normalization_impact
              python -m analysis.queries --all
            """
        ),
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--list",  action="store_true", help="List all available queries")
    group.add_argument("--query", metavar="NAME",      help="Run a specific named query")
    group.add_argument("--all",   action="store_true", help="Run all queries in order")
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.list:
        print("\nAvailable queries:\n")
        for name, entry in QUERIES.items():
            print(f"  {name:<35}  {entry['question']}")
        print()
        return

    # Lazy import so the module is importable without a live DB
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    import sys
    import pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

    from config import load_config
    from db.db import get_engine

    cfg = load_config()
    engine = get_engine(cfg.database_url)

    if args.all:
        names = list(QUERIES)
    else:
        names = [args.query]

    for name in names:
        if name not in QUERIES:
            print(f"ERROR: unknown query {name!r}. Use --list to see available queries.")
            sys.exit(1)
        try:
            rows = run_query(name, engine)
            print_results(name, rows)
        except Exception as exc:
            print(f"ERROR running {name!r}: {exc}")
            sys.exit(1)


if __name__ == "__main__":
    main()
