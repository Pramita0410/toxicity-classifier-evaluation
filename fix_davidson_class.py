"""
fix_davidson_class.py
Updates davidson_class column in ground_truth from the original CSV.
Run once — does not touch predictions or any other table.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv()

import pandas as pd
from sqlalchemy import text
from config import load_config
from db.db import get_engine

cfg = load_config()
print(f"Davidson CSV path: {cfg.davidson_csv_path!r}")

if not cfg.davidson_csv_path:
    print("ERROR: DAVIDSON_CSV_PATH is empty in .env")
    sys.exit(1)

csv_path = Path(cfg.davidson_csv_path)
if not csv_path.exists():
    print(f"ERROR: File not found: {csv_path.absolute()}")
    sys.exit(1)

print(f"Reading {csv_path}...")
df = pd.read_csv(csv_path)
print(f"Read {len(df)} rows. Columns: {list(df.columns)}")

# Build item_id the same way ingest.py does
if "Unnamed: 0" in df.columns:
    df["item_id"] = "davidson_" + df["Unnamed: 0"].astype(str)
elif "" in df.columns:
    df["item_id"] = "davidson_" + df[""].astype(str)
else:
    df["item_id"] = "davidson_" + df.reset_index(drop=True).index.astype(str)

engine = get_engine(cfg.database_url)

print("Updating davidson_class in ground_truth...")
updated = 0
with engine.begin() as conn:
    for _, row in df.iterrows():
        result = conn.execute(
            text("UPDATE ground_truth SET davidson_class = :cls WHERE item_id = :iid"),
            {"cls": int(row["class"]), "iid": row["item_id"]}
        )
        updated += result.rowcount

print(f"Updated {updated} rows.")

# Verify
with engine.connect() as conn:
    counts = conn.execute(text(
        "SELECT davidson_class, COUNT(*) FROM ground_truth "
        "WHERE dataset_source = 'davidson' "
        "GROUP BY davidson_class ORDER BY davidson_class"
    )).fetchall()
    print("\nDavidson class distribution in ground_truth:")
    labels = {0: "hate_speech", 1: "offensive_language", 2: "neither"}
    for cls, cnt in counts:
        print(f"  class {cls} ({labels.get(cls, 'unknown')}): {cnt} rows")
