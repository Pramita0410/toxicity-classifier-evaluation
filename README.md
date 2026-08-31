# Trust & Safety Evaluation Platform

Most toxicity classifiers get benchmarked on the same data they trained on and called production-ready. This project questions that.

It runs four content moderation models against two labeled datasets, tests robustness under ten real evasion techniques, and surfaces where models fail — not just where they succeed.

The finding that mattered: a model with F1=0.79 was completely undeployable because it flagged 82% of clean tweets as toxic. Aggregate metrics hid this until we broke down performance by Davidson's original 3-class content label.

---

## Project structure

```
pipeline.py          main orchestrator — run this to do everything
config.py            all settings from .env
ingestion/           CSV → PostgreSQL (Jigsaw + Davidson datasets)
moderation/          4 moderators: rule-based, 2x RoBERTa, XLM-R
evaluation/          confusion matrix math + threshold sweep
adversarial/         evasion attack generator + text normalizer
analysis/            named SQL queries runnable from CLI
db/                  schema, connection factory, Superset dataset queries
tests/               property-based tests (Hypothesis) + integration tests
visualize.py         generates PNG charts from the database
```

---

## Models compared

| Model | Type | F1 | Notes |
|---|---|---|---|
| rule_based | Keywords | 0.24 | Baseline. Catches 15% of toxic content. |
| roberta_toxicity_classifier | RoBERTa binary | 0.87 | Best precision. Fails completely on Unicode attacks. |
| roberta-toxicity-classifier | RoBERTa multi-label | 0.79 | Recall=0.99 but 82% false positive rate on clean tweets. |
| xlmr-large-toxicity-classifier-v2 | XLM-R multilingual | 0.86 | Most consistent across domains. Best for production. |

---

## Dashboard and Charts

<img width="1900" height="945" alt="image" src="https://github.com/user-attachments/assets/8ae67431-b360-4289-9454-a0e07f2485b1" />


<img width="1911" height="844" alt="image" src="https://github.com/user-attachments/assets/bbf78461-1843-4155-a218-a9f31ec6887f" />



<img width="1890" height="832" alt="image" src="https://github.com/user-attachments/assets/26b2b6eb-09b3-4700-ae95-41a1d94e238c" />

<img width="1891" height="858" alt="image" src="https://github.com/user-attachments/assets/7ffc36e8-ed59-4be7-afaf-35ed97c3f22b" />


## Key findings

**Aggregate metrics are misleading.** roberta-multilabel's F1 looked competitive until we checked its flag rate on clean content by Davidson class. It would remove 4 in 5 legitimate tweets.

**Unicode attacks break binary classifiers.** Zero-width characters drop roberta_toxicity_classifier recall from 0.87 to 0.00. Homoglyphs do the same. Models never saw these in training data. Normalization recovers most of the gap.

**Domain shift is real.** roberta_toxicity_classifier drops 17% accuracy from Wikipedia comments to tweets. XLM-R drops 6%. Training data diversity matters more than model size for cross-domain robustness.

**Label noise in Davidson class 1.** Some "offensive language" labels were clearly wrong — a racing tweet labeled as offensive. When reviewing false negatives, some were correct model predictions on bad labels. Evaluation on class 0 (hate speech) only gives a cleaner picture.

---

## Quick start

Requirements: Docker, Python 3.10+, 8GB RAM

```bash
# Start PostgreSQL
docker run -d -p 5432:5432 --name trust-safety-pg \
  -e POSTGRES_PASSWORD=postgres postgres:16

# Install dependencies
python -m venv .venv
source .venv/bin/activate   # or .\.venv\Scripts\Activate.ps1 on Windows
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Configure
cp .env.example .env
# Edit .env: set DATASET_CSV_PATH and DAVIDSON_CSV_PATH

# Run
python pipeline.py
```

---

## Superset dashboards

```bash
docker run -d -p 8088:8088 --name superset \
  -e SUPERSET_SECRET_KEY=changeme apache/superset
docker exec -u root superset pip install psycopg2-binary
docker exec superset superset fab create-admin \
  --username admin --password admin \
  --firstname Admin --lastname User --email admin@local.com
docker exec superset superset db upgrade
docker exec superset superset init

# Connect containers so Superset can reach PostgreSQL
docker network create trust-safety-net
docker network connect trust-safety-net trust-safety-pg
docker network connect trust-safety-net superset
```

Open http://localhost:8088. Add PostgreSQL connection:
`postgresql://postgres:postgres@trust-safety-pg:5432/trust_safety`

Dataset queries are in `db/superset_datasets.sql`.

---

## Tests

```bash
python -m pytest tests/ -v
```

No database or API credentials needed. All tests use fixtures or pure Python.

---

## Datasets

Jigsaw Toxic Comment Classification — Kaggle / CC BY 4.0
Davidson Hate Speech and Offensive Language — GitHub / MIT

Neither dataset is committed to this repo. Download them separately.
