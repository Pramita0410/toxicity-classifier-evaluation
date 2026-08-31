"""
run_ingest_only.py — re-ingests ground_truth with davidson_class column.
Does NOT run any models. Existing predictions are untouched.

Usage:
    python run_ingest_only.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from config import load_config
from db.db import get_engine
from ingestion.ingest import load_ground_truth

cfg = load_config()
engine = get_engine(cfg.database_url)

print(f"Re-ingesting ground_truth with davidson_class column...")
print(f"SAMPLE_SIZE: {cfg.sample_size}")
rows = load_ground_truth(cfg, engine)
print(f"Done. {rows} new rows ingested.")
print("Existing predictions are untouched.")
