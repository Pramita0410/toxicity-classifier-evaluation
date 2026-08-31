"""tests/conftest.py — ensures workspace root is on sys.path before any test
in this directory is collected or imported."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]   # scikit_learn_data/
for p in (str(ROOT), str(ROOT / "project")):
    if p not in sys.path:
        sys.path.insert(0, p)
