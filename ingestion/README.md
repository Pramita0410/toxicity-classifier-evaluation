# ingestion

Reads CSV files and loads them into the `ground_truth` PostgreSQL table.

Supports two dataset formats automatically — Jigsaw (Wikipedia comments) and
Davidson (tweets). Detection is by column name: `comment_text` means Jigsaw,
`tweet` means Davidson.

Sampling is stratified on the `toxic` label so you always get a balanced
mix of toxic and clean rows regardless of sample size.

## Key file

`ingest.py` — the only file here. Entry point is `load_ground_truth(cfg, engine)`.

## To re-ingest without re-running models

```bash
python run_ingest_only.py
```

## To add davidson_class after re-ingest

```bash
python fix_davidson_class.py
```
