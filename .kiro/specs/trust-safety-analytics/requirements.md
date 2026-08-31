# Requirements — AI Safety Evaluation Platform

**Purpose:** Portfolio project demonstrating Trust & Safety engineering skills relevant to Google
Trust & Safety analyst / engineer roles. The platform evaluates LLM-generated content-moderation
decisions against a real labeled toxic-content dataset, surfaces precision/recall trade-offs, runs
prompt A/B experiments, and visualizes results for both technical and non-technical audiences.

---

## 1. Functional Requirements

### 1.1 Dataset Ingestion
- FR-1: Read a local CSV file whose path is configured via environment variable (`DATASET_CSV_PATH`).
  Do NOT download data at runtime.
- FR-2: Support the Jigsaw Toxic Comment Classification dataset schema (columns: `id`, `comment_text`,
  `toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, `identity_hate`) and the Davidson
  hate-speech dataset schema (`tweet`, `class`, `hate_speech`, `offensive_language`, `neither`).
  Normalise both into a canonical schema on ingest.
- FR-3: Load the canonical records into a PostgreSQL `ground_truth` table, replacing previous data
  on each run. A configurable `SAMPLE_SIZE` cap limits rows loaded (default: 5000; 0 = all rows).
- FR-4: Provide a tiny safe sample fixture (≤ 20 synthetic rows, no real toxic text) in
  `tests/fixtures/sample_ground_truth.csv` for CI and unit tests — the real dataset must never be
  committed to the repository.

### 1.2 LLM Moderation Component
- FR-5: Accept a list of text items and return a binary moderation decision (`toxic` / `not_toxic`)
  plus a confidence score [0, 1] per item.
- FR-6: Support at least two prompt strategies: **zero-shot** and **few-shot**. Each strategy is a
  named `prompt_version`.
- FR-7: Store predictions in a PostgreSQL `predictions` table tagged with `model_name`,
  `prompt_version`, and `run_id` (UUID).
- FR-8: **Fixture fallback**: when `LLM_API_KEY` is not set (or `USE_FIXTURES=true`), the component
  reads pre-recorded responses from `tests/fixtures/llm_responses.json` instead of calling any
  external API. The pipeline must be fully runnable with no API key.
- FR-9: Failures for individual items must be isolated — one bad item must not abort the entire
  batch. Failed items are logged and stored with `decision=null`, `confidence=null`.

### 1.3 Evaluation Engine
- FR-10: Compute per-run, per-category, and per-prompt-version confusion matrix counts: TP, FP, FN, TN.
- FR-11: Derive from those counts: Precision, Recall, F1 score, False-Positive Rate (FPR),
  False-Negative Rate (FNR). Zero-denominator cases must return 0, not raise an error.
- FR-12: Exclude any ground-truth item that has no label (null / NaN) from all metric calculations.
- FR-13: Write computed metrics to a PostgreSQL `metrics` table keyed by `run_id`, `category`,
  `prompt_version`.
- FR-14: Write false-positive and false-negative items to a PostgreSQL `error_analysis` table,
  capturing `item_id`, `text`, `true_label`, `predicted_label`, `confidence`, `run_id`,
  `prompt_version`.

### 1.4 Visualization (Apache Superset)
- FR-15: Provide dashboard definitions (JSON export or Python script) for four views:
  1. **Executive KPIs** — overall precision, recall, F1, FPR, FNR for the latest run.
  2. **A/B Comparison** — side-by-side metrics for each `prompt_version`.
  3. **Error Analysis** — browsable table of FP and FN items with text and confidence.
  4. **Category Breakdown** — per-label metrics (toxic, severe_toxic, obscene, threat, insult,
     identity_hate) where applicable.
- FR-16: Dashboard setup must be documented in the README (Superset install, DB connection, import
  steps).

### 1.5 Testing
- FR-17: The evaluation math (FR-10, FR-11, FR-12) must be covered by **property-based tests**
  using [Hypothesis](https://hypothesis.readthedocs.io/). Tests must run with no database
  connection and no API credentials.
- FR-18: CI must run the full Python test suite on every push. Secrets (DB URL, API key) must not
  be required for CI to pass.

---

## 2. Non-Functional Requirements

- NFR-1: **No live API calls in tests.** All tests use fixtures or pure Python only.
- NFR-2: **No real toxic data committed.** `.gitignore` must exclude `data/raw/`.
- NFR-3: Configuration via environment variables only — no hardcoded credentials or paths.
- NFR-4: Python 3.11+. Dependencies pinned in `requirements.txt`.
- NFR-5: The pipeline must be runnable end-to-end on a single developer machine with only Python,
  PostgreSQL, and (optionally) Superset installed.
- NFR-6: Code must be lint-clean (ruff) and type-annotated (mypy, non-strict).

---

## 3. Out of Scope

- Real-time / streaming moderation.
- Multi-tenant or production deployment.
- Fine-tuning any model.
- Any UI beyond Superset dashboards.

---

## 4. Dataset Attribution

- **Jigsaw Toxic Comment Classification Challenge** — Kaggle / Conversation AI.
  License: CC BY 4.0 (comment text from Wikipedia, labels by Jigsaw).
  https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge
- **Hate Speech and Offensive Language** — T. Davidson et al. (2017).
  License: MIT.
  https://github.com/t-davidson/hate-speech-and-offensive-language

Users must comply with the respective dataset licenses and Kaggle competition rules.
