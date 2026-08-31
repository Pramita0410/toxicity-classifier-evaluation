# Design — AI Safety Evaluation Platform

_Stack: Python 3.11 · pandas · PostgreSQL · Apache Superset_
_Derived from requirements.md — treat that as the source of truth._

---

## 1. Repository Layout

```
trust-safety-analytics/
├── config.py                   # env-var config dataclass
├── ingestion/
│   └── ingest.py               # CSV → canonical DataFrame → Postgres ground_truth
├── moderation/
│   ├── moderator.py            # LLM client + fixture fallback
│   ├── prompts.py              # zero-shot / few-shot prompt builders
│   └── fixtures/
│       └── llm_responses.json  # pre-recorded responses (committed)
├── evaluation/
│   └── metrics.py              # pure-Python confusion matrix + metric math
├── pipeline.py                 # orchestrates ingest → moderate → evaluate
├── db/
│   ├── schema.sql              # CREATE TABLE statements
│   └── db.py                   # SQLAlchemy engine factory
├── dashboards/
│   └── superset_export.json    # Superset dashboard export
├── tests/
│   ├── fixtures/
│   │   ├── sample_ground_truth.csv   # ≤20 synthetic rows, no real toxic text
│   │   └── llm_responses.json        # symlink or copy of moderation/fixtures/
│   ├── test_metrics.py         # Hypothesis property tests (no DB, no network)
│   └── test_ingestion.py       # ingest against sample CSV
├── requirements.txt
├── .env.example
├── .gitignore                  # excludes data/raw/, .env
├── .github/workflows/ci.yml
└── README.md
```

---

## 2. Data Model

### `ground_truth`
| column | type | notes |
|---|---|---|
| item_id | TEXT PK | original dataset ID / row index |
| text | TEXT | comment or tweet |
| toxic | SMALLINT | 0/1, nullable |
| severe_toxic | SMALLINT | Jigsaw only, nullable |
| obscene | SMALLINT | Jigsaw only, nullable |
| threat | SMALLINT | Jigsaw only, nullable |
| insult | SMALLINT | Jigsaw only, nullable |
| identity_hate | SMALLINT | Jigsaw only, nullable |
| dataset_source | TEXT | "jigsaw" or "davidson" |

### `predictions`
| column | type | notes |
|---|---|---|
| id | SERIAL PK | |
| run_id | UUID | groups one full moderation run |
| item_id | TEXT FK → ground_truth | |
| model_name | TEXT | e.g. "gpt-4o-mini" or "fixture" |
| prompt_version | TEXT | "zero_shot" / "few_shot" |
| decision | TEXT | "toxic" / "not_toxic" / null (failure) |
| confidence | FLOAT | [0,1] or null |
| error_msg | TEXT | nullable, populated on per-item failure |

### `metrics`
| column | type | notes |
|---|---|---|
| id | SERIAL PK | |
| run_id | UUID | |
| prompt_version | TEXT | |
| category | TEXT | label name, e.g. "toxic", or "overall" |
| tp | INT | |
| fp | INT | |
| fn | INT | |
| tn | INT | |
| precision | FLOAT | |
| recall | FLOAT | |
| f1 | FLOAT | |
| fpr | FLOAT | false-positive rate |
| fnr | FLOAT | false-negative rate |

### `error_analysis`
| column | type | notes |
|---|---|---|
| id | SERIAL PK | |
| run_id | UUID | |
| prompt_version | TEXT | |
| item_id | TEXT | |
| text | TEXT | |
| true_label | SMALLINT | |
| predicted_label | TEXT | |
| confidence | FLOAT | |
| error_type | TEXT | "FP" or "FN" |

---

## 3. Component Design

### 3.1 Config (`config.py`)
A `dataclass` populated from `os.environ`. Key fields:

```
DATASET_CSV_PATH   str   path to local CSV
SAMPLE_SIZE        int   0 = all rows (default 5000)
DATABASE_URL       str   postgres://user:pass@host/db
LLM_API_KEY        str   optional; absent → fixture mode
LLM_MODEL          str   default "gpt-4o-mini"
USE_FIXTURES       bool  override to force fixtures
RUN_ID             str   UUID, auto-generated if absent
```

### 3.2 Ingestion (`ingestion/ingest.py`)
1. Read CSV with pandas, detect schema (Jigsaw vs Davidson) by column names.
2. Normalise to canonical schema (map Davidson `class` → binary label columns).
3. Apply `SAMPLE_SIZE` cap (random sample, seed=42 for reproducibility).
4. Write to `ground_truth` via `pd.DataFrame.to_sql(..., if_exists="replace")`.

### 3.3 Moderation (`moderation/moderator.py`)

```
ModerationResult(item_id, decision, confidence, error_msg)
```

Two modes selected at construction time:
- **Live mode** (`LLM_API_KEY` set, `USE_FIXTURES=false`): calls the configured LLM API with the
  prompt built by `prompts.py`. Per-item try/except; failures stored with `decision=null`.
- **Fixture mode** (default when no key): loads `llm_responses.json`, keyed by `item_id`. If an
  item is missing from fixtures, returns `decision="not_toxic", confidence=0.5` as a safe default.

`prompts.py` exposes `build_prompt(text, strategy)` for `"zero_shot"` and `"few_shot"`.

### 3.4 Evaluation Engine (`evaluation/metrics.py`)
Pure functions only — no I/O, no imports beyond stdlib + pandas/numpy.

```python
def confusion_counts(y_true, y_pred) -> dict[str, int]:
    # returns {tp, fp, fn, tn}

def compute_metrics(counts) -> dict[str, float]:
    # precision, recall, f1, fpr, fnr
    # zero-denominator → 0.0

def evaluate_run(ground_truth_df, predictions_df, category, prompt_version) -> MetricsRow:
    # filters out null labels, delegates to above two functions
```

### 3.5 Pipeline (`pipeline.py`)
Thin orchestrator:
1. Load config.
2. Run ingestion (unless `SKIP_INGEST=true`).
3. For each `prompt_version` in `["zero_shot", "few_shot"]`:
   a. Run moderator over all `ground_truth` items → write to `predictions`.
   b. Run `evaluate_run` per category + overall → write to `metrics` + `error_analysis`.

### 3.6 Dashboards
Four Superset charts exported to `dashboards/superset_export.json`:
- **Executive KPIs**: big-number tiles for precision, recall, F1, FPR, FNR (latest run).
- **A/B Comparison**: bar chart — metrics by `prompt_version`.
- **Error Analysis**: filterable table from `error_analysis`.
- **Category Breakdown**: grouped bar from `metrics` by `category`.

---

## 4. Evaluation Math (reference)

```
precision  = TP / (TP + FP)         → 0 if TP+FP = 0
recall     = TP / (TP + FN)         → 0 if TP+FN = 0
f1         = 2·P·R / (P + R)        → 0 if P+R = 0
fpr        = FP / (FP + TN)         → 0 if FP+TN = 0
fnr        = FN / (FN + TP)         → 0 if FN+TP = 0
```

Items with `true_label IS NULL` are excluded before any computation.

---

## 5. Testing Strategy

| Layer | Tool | Runs without |
|---|---|---|
| Metric math | Hypothesis (property-based) | DB, network, API key |
| Ingestion | pytest + sample CSV fixture | real dataset, DB optional |
| Moderation | pytest + fixture JSON | API key |
| Pipeline | pytest + sample fixture | real dataset, API key |
| CI | GitHub Actions | all secrets optional |

---

## 6. Key Constraints Reflected

- `data/raw/` in `.gitignore` — dataset never committed.
- All credentials via env vars; `.env.example` documents them.
- `USE_FIXTURES=true` (or absent `LLM_API_KEY`) gives a fully offline run.
- `SAMPLE_SIZE` cap prevents accidentally loading millions of rows.
