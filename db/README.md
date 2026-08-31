# db

Database layer — schema, connection factory, and Superset dataset queries.

## Files

`schema.sql` — all table definitions. Safe to run multiple times (IF NOT EXISTS).
Run with: `psql -d trust_safety -f db/schema.sql`

`db.py` — SQLAlchemy engine factory. Cached so the same connection pool is
reused across the pipeline. Connection string comes from `DATABASE_URL` in `.env`.

`superset_datasets.sql` — SQL queries for creating virtual datasets in Apache
Superset SQL Lab. Copy each block, run it, save as the named dataset.

## Tables

`ground_truth` — labeled comments from Jigsaw and Davidson.
`predictions` — model decisions, one row per (comment × model).
`metrics` — aggregated F1/precision/recall. Has a known model_name issue — use predictions directly for analysis.
`threshold_metrics` — precision/recall at each confidence threshold. Same issue — use threshold_curve_computed from superset_datasets.sql.
`error_analysis` — every FP and FN with the comment text.
`adversarial_results` — raw vs normalized model performance per attack type.

## Main run ID

`07bc2a8d-8ed7-4c0f-9bff-476f41533f16` — this is the run with 10,885 rows per model.
Use this in all Superset dataset queries.
