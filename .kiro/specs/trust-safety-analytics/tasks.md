# Tasks — AI Safety Evaluation Platform

Build order: complete each batch fully before moving to the next.
Each batch is small enough to review before proceeding.

---

## Batch A — Project Skeleton + Config
_Goal: runnable project structure with all settings wired through env vars._

- [x] A1. Create folder structure: `ingestion/`, `moderation/`, `evaluation/`, `db/`, `tests/fixtures/`, `dashboards/`
- [x] A2. Write `config.py` — dataclass loaded from env vars:
        - `DATASET_CSV_PATH` (default: `data/raw/train.csv/train.csv`)
        - `DAVIDSON_CSV_PATH` (optional, for second dataset)
        - `SAMPLE_SIZE` (default 5000; 0 = all rows)
        - `DATABASE_URL` (postgres connection string)
        - `OPENAI_API_KEY` (optional)
        - `GEMINI_API_KEY` (optional)
        - `USE_FIXTURES` (bool, default True when no API keys present)
        - `RUN_ID` (UUID, auto-generated if not set)
- [x] A3. Write `.env.example` documenting every env var
- [x] A4. Write `.gitignore` — exclude `data/raw/`, `.env`, `__pycache__`, `*.pyc`
- [x] A5. Write `requirements.txt` with pinned versions:
        pandas, sqlalchemy, psycopg2-binary, openai, google-generativeai,
        hypothesis, pytest, python-dotenv, ruff
- [x] A6. Write `db/schema.sql` — CREATE TABLE for all 4 tables:
        `ground_truth`, `predictions`, `metrics`, `error_analysis`
- [x] A7. Write `db/db.py` — SQLAlchemy engine factory using `DATABASE_URL`

**Pause here — review folder structure and config before writing any logic.**

---

## Batch B — Evaluation Engine + Property Tests
_Goal: the math that grades moderation decisions. No DB, no network — pure Python._
_This is the CORE of the project. Get it right before touching data._

- [x] B1. Write `evaluation/metrics.py`:
        - `confusion_counts(y_true, y_pred) → dict` — TP, FP, FN, TN
        - `compute_metrics(counts) → dict` — precision, recall, F1, FPR, FNR
          (zero-denominator → 0.0, never raises)
        - `evaluate_run(ground_truth_df, predictions_df, category, prompt_version) → dict`
          (filters null labels, calls above two functions)
- [x] B2. Write `tests/test_metrics.py` using Hypothesis:
        - Property: precision + recall are always in [0, 1]
        - Property: F1 is always ≤ max(precision, recall)
        - Property: all-correct predictions → precision=recall=F1=1.0
        - Property: all-wrong predictions → precision=recall=F1=0.0
        - Property: empty y_true after null filter → all metrics = 0.0
        - Property: zero denominator never raises an exception
- [x] B3. Run tests — all must pass before moving on

**Pause here — review math and test results.**

---

## Batch C — Data Ingestion
_Goal: read both CSVs, normalize to one schema, load into PostgreSQL._

- [x] C1. Write `ingestion/ingest.py`:
        - Auto-detect schema: Jigsaw (has `comment_text`) vs Davidson (has `tweet`)
        - Normalize Davidson `class` column → binary label columns
        - Apply `SAMPLE_SIZE` cap (random sample, seed=42)
        - Load into `ground_truth` table via `pd.DataFrame.to_sql(if_exists="replace")`
        - Add `dataset_source` column: "jigsaw" or "davidson"
- [x] C2. Create `tests/fixtures/sample_ground_truth.csv`:
        - 20 synthetic rows, NO real toxic text
        - Covers: all-zero labels, some toxic=1, some multi-label, one null label row
- [x] C3. Write `tests/test_ingestion.py`:
        - Test schema detection works for both formats
        - Test null label rows are preserved in DB (filtering happens at eval time)
        - Test SAMPLE_SIZE cap is respected
        - Uses sample fixture CSV only — no real dataset, no DB required

**Pause here — verify both CSVs load correctly.**

---

## Batch D — LLM Moderation Component
_Goal: three moderators (rule-based, GPT-4o-mini, Gemini 2.5 Flash), each with zero-shot + few-shot prompts, fixture fallback, per-item failure isolation._

- [x] D1. Write `moderation/prompts.py`:
        - `build_prompt(text, strategy)` for `"zero_shot"` and `"few_shot"`
        - Few-shot includes 3 example pairs (one toxic, one borderline, one clean)
        - Keep prompts short and explicit about the binary output format

- [ ] D2. Write `moderation/moderator.py` — base class + three implementations:
        - `RuleBasedModerator` — keyword list approach, no API, always available
          (toxic keywords → decision="toxic", confidence=0.95; else not_toxic, 0.05)
        - `OpenAIModerator` — calls GPT-4o-mini; falls back to fixtures if no key
        - `GeminiModerator` — calls Gemini 2.5 Flash; falls back to fixtures if no key
        - All three: per-item try/except, failed items get decision=null, error logged
        - All three: store results in `predictions` table tagged with model_name + prompt_version + run_id

- [ ] D3. Create `tests/fixtures/llm_responses.json`:
        - Pre-recorded responses keyed by item_id for the 20 sample fixture rows
        - Covers: toxic decisions, not_toxic decisions, one simulated failure

- [ ] D4. Write `tests/test_moderation.py`:
        - Test fixture fallback works with no API key
        - Test per-item failure isolation (one bad item doesn't abort batch)
        - Test all 3 moderators return correct output shape
        - Test both prompt strategies produce valid decisions

**Pause here — run all moderators in fixture mode end-to-end.**

---

## Batch E — Wire Evaluation Over Predictions
_Goal: connect stored predictions → metrics table + error analysis table._

- [ ] E1. Write `pipeline.py` — full orchestrator:
        1. Load config + generate RUN_ID
        2. Run ingestion (unless `SKIP_INGEST=true`)
        3. For each model × prompt_version combination (6 total):
           a. Run moderator → write to `predictions`
           b. Run `evaluate_run` per category + "overall" → write to `metrics`
           c. Write FP and FN items → `error_analysis`
        4. Print summary table to stdout at the end

- [ ] E2. Verify end-to-end run in fixture mode:
        - `USE_FIXTURES=true python pipeline.py`
        - All 4 tables populated
        - No errors

- [ ] E3. Add per-category evaluation for all 6 Jigsaw labels:
        `toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, `identity_hate`
        Plus `overall` and `hate_speech` (Davidson)

**Pause here — inspect DB tables, confirm metrics look right.**

---

## Batch F — Superset Dashboards + README
_Goal: four dashboards visualizing results + full setup documentation._

- [ ] F1. Write `dashboards/superset_setup.py` — Python script that:
        - Creates a Superset database connection to PostgreSQL
        - Creates 4 datasets (one per table)
        - Creates 4 charts:
          1. Executive KPIs — big-number tiles (precision, recall, F1, FPR, FNR)
          2. A/B Comparison — bar chart by model_name + prompt_version
          3. Error Analysis — filterable table (FP/FN items with text + confidence)
          4. Category Breakdown — grouped bar by category
        - Exports dashboard to `dashboards/superset_export.json`

- [ ] F2. Write `README.md`:
        - Project overview (1 paragraph, portfolio-friendly language)
        - Architecture diagram (ASCII)
        - Setup steps: Python env, PostgreSQL, env vars, running the pipeline
        - Superset setup steps
        - Dataset attribution and licensing note (Jigsaw CC0, Davidson MIT)
        - Sample results screenshot placeholder

**Pause here — review README as if you're a Google interviewer seeing it for the first time.**

---

## Batch G — CI Config
_Goal: automated tests on every git push, zero secrets required._

- [ ] G1. Write `.github/workflows/ci.yml`:
        - Trigger: push + pull_request on main
        - Python 3.11
        - Install dependencies from requirements.txt
        - Run: `pytest tests/test_metrics.py tests/test_ingestion.py tests/test_moderation.py`
        - No DB_URL, no API keys — all tests use fixtures
        - Add ruff lint check

- [ ] G2. Verify CI config is valid YAML and all referenced test files exist

---

## Summary — What You'll Have When Done

```
6 moderation strategies compared  (3 models × 2 prompt types)
+  rule-based baseline
= 7-way comparison showing LLMs beat keyword rules, few-shot beats zero-shot

Per-category metrics for 6 harm types (toxic, threat, identity_hate, etc.)
False-negative rate highlighted for severe/egregious categories
Error analysis table for human review simulation
4 Superset dashboards
Property-based test suite (Hypothesis) proving math is correct
CI pipeline — green badge on your GitHub
```

This is what you show Google. Not just "I classified toxic comments" —
"I built the evaluation infrastructure that measures *how well* a classifier works,
across multiple models and harm categories, with a human-review error queue."
