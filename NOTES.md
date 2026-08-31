# Project Notes — Trust & Safety Evaluation Platform

Quick reference for understanding what was built, why, and how to run it again.

---

## What this project does

Evaluates toxicity classifiers against real labeled datasets, tests robustness
against evasion attacks, and visualizes results in Apache Superset.

Core question: **Which model should you deploy, and why?**

---

## The data

### Jigsaw Toxic Comment Classification
- Source: Wikipedia talk page comments
- Labels: toxic, severe_toxic, obscene, threat, insult, identity_hate (multi-label, 0/1)
- 159k rows total — we sample 5000 for evaluation
- Labels made by trained Jigsaw/Google annotators — high quality

### Davidson Hate Speech Dataset  
- Source: Twitter tweets
- Original labels: 0=hate_speech, 1=offensive_language, 2=neither
- We collapsed 0 and 1 into binary toxic=1 for cross-dataset evaluation
- We added davidson_class column to preserve the original 3-class label
- 24k rows total — we sample 5000
- Labels made by crowdworkers — noisier than Jigsaw, especially class 1

**Why two datasets:** To test domain shift. Models trained on Wikipedia comments
degrade when applied to tweets. That gap tells you about real-world robustness.

---

## The models

| DB name | What it is | Key characteristic |
|---|---|---|
| rule_based | 21 hardcoded keywords | Baseline. F1=0.24. Catches 15% of toxic content. |
| roberta_toxicity_classifier | s-nlp/roberta — binary | F1=0.87. High precision (0.91). Conservative. |
| roberta-toxicity-classifier | Arsive/roberta — multi-label | F1=0.79. Recall=0.99. Flags almost everything. 82% FPR on clean tweets. |
| xlmr-large-toxicity-classifier-v2 | textdetox/xlmr — multilingual | F1=0.86. Most consistent across datasets. |

All HuggingFace models run locally — no API key, no cost.

---

## Key findings

**1. roberta-multilabel is undeployable despite high recall**
It catches 99% of toxic content but flags 82% of clean tweets as toxic.
An F1 score of 0.79 hides this catastrophic false positive rate.
Lesson: aggregate metrics are not enough. Always check FPR on clean content.

**2. Domain shift is real**
roberta_toxicity_classifier: Jigsaw accuracy=0.984, Davidson=0.811 (17% drop)
xlmr: Jigsaw=0.898, Davidson=0.837 (6% drop)
XLM-R is more robust because it was trained on multilingual/diverse data.

**3. Transformers fail on Unicode attacks**
Zero-width characters drop roberta_toxicity_classifier recall from 0.87 to 0.00.
Homoglyphs (Cyrillic а instead of Latin a) drop recall to near zero.
Misspellings barely affect performance — models learned from noisy training data.
Normalization recovers Unicode attacks but adds no value for natural language noise.

**4. Normalization is model-dependent**
roberta-multilabel is naturally immune to most attacks (recall stays at 1.0).
roberta_toxicity_classifier requires normalization for Unicode-based attacks.
One-size-fits-all preprocessing is wrong — it depends on which model you deploy.

---

## Database tables

| Table | What it stores |
|---|---|
| ground_truth | Original labeled comments. item_id, text, toxic, all label columns, dataset_source, davidson_class |
| predictions | Model decisions. item_id, model_name, decision, confidence, run_id |
| metrics | Aggregated F1/precision/recall per model (had join issues — use predictions directly) |
| threshold_metrics | Precision/recall at each confidence threshold (had join issues — compute from predictions) |
| error_analysis | Every FP and FN item with text |
| adversarial_results | Side-by-side raw vs normalized model performance per attack technique |

**Important:** metrics and threshold_metrics had a bug where model_name wasn't
stored correctly. Use the SQL in db/superset_datasets.sql to compute metrics
directly from predictions instead.

**run_id for main run:** 07bc2a8d-8ed7-4c0f-9bff-476f41533f16

---

## Superset datasets

All SQL is in db/superset_datasets.sql. Datasets to create in SQL Lab:

| Dataset name | Used for |
|---|---|
| predictions_main | Model comparison bar charts, FP/FN analysis |
| adversarial_main | Robustness heatmaps |
| adversarial_comparison | Raw vs normalized side-by-side heatmap |
| davidson_policy_analysis | Flag rate by content severity (the policy chart) |
| threshold_curve_computed | Recall vs threshold line chart |
| high_confidence_errors | Error browser table |

---

## Superset charts built

| Chart | Dataset | Insight |
|---|---|---|
| Model Error Tradeoff: Wrong Flags vs Missed Harm | predictions_main | FP vs FN per model |
| Model Recall Under Attack (No Normalization) | adversarial_main | Which attacks break which models |
| Normalization Impact by Model and Attack | adversarial_main | Where preprocessing helps |
| Policy-Aware Flag Rate by Model | davidson_policy_analysis | roberta-multilabel 82% FPR on clean tweets |

---

## How to re-run

```powershell
# Start containers
docker start trust-safety-pg superset

# Activate venv
.\.venv\Scripts\Activate.ps1

# Run pipeline (ingest + models + evaluation)
python pipeline.py

# Run ingest only (no model runs)
python run_ingest_only.py

# Fix davidson_class column after re-ingest
python fix_davidson_class.py

# Run tests
python -m pytest tests/ -v

# Generate charts from DB
python visualize.py
```

---

## Known limitations

1. **Davidson class 1 label noise** — "offensive language" includes borderline
   content that many platforms would not act on. Some FN cases are actually
   correct model predictions on noisy labels.

2. **metrics table join bug** — metrics doesn't store model_name directly.
   All metric computations should use predictions + ground_truth directly.

3. **omni-moderation-latest** — ran in fixture mode (fake responses). Delete
   those rows: `DELETE FROM predictions WHERE model_name = 'omni-moderation-latest'`

4. **davidson_class** — original 3-class Davidson label was collapsed to binary
   during initial ingest. Fixed via fix_davidson_class.py. Future work:
   evaluate hate speech (class 0) vs offensive language (class 1) separately.

---

## Files reference

| File | Purpose |
|---|---|
| pipeline.py | Main orchestrator — runs everything end to end |
| config.py | All settings from .env |
| ingestion/ingest.py | CSV → PostgreSQL |
| moderation/moderator.py | All 4 moderator classes |
| evaluation/metrics.py | Confusion matrix math + threshold sweep |
| adversarial/normalizer.py | Cleans obfuscated text |
| adversarial/generator.py | Generates evasion attack variants |
| db/schema.sql | All table definitions |
| db/superset_datasets.sql | All Superset dataset queries |
| analysis/queries.py | Named SQL queries runnable from CLI |
| visualize.py | Generates PNG charts from DB |
| fix_davidson_class.py | One-time fix to add davidson_class column |
| run_ingest_only.py | Re-ingests without running models |
| tests/ | Property tests + integration tests |

---

## What we actually found — the real story

This project started as "run some toxicity models and measure F1." It ended up
finding something more interesting: aggregate metrics hide deployment-breaking
failures.

### The roberta-multilabel problem

roberta-multilabel had the highest recall (0.994) — it catches almost every
toxic comment. On paper it looks like the best model. But when we broke down
its flag rate by Davidson's original 3-class label, the real picture emerged:

- Flag rate on hate speech: 99.5%
- Flag rate on offensive language: 99.5%  
- Flag rate on clean content: **82.5%**

It flags 4 out of 5 clean tweets as toxic. F1 score masked this entirely because
the dataset has more toxic rows than clean ones — the FP rate gets diluted.

No platform could deploy this. It would remove 82% of legitimate user content.

**How we found it:**
We added davidson_class column to preserve Davidson's original 3-class label
(0=hate_speech, 1=offensive_language, 2=neither) which was collapsed to binary
during initial ingestion. One SQL query revealed the problem:

```sql
SELECT
    p.model_name,
    CASE g.davidson_class
        WHEN 0 THEN '0 - Hate Speech'
        WHEN 1 THEN '1 - Offensive Language'
        WHEN 2 THEN '2 - Neither (Clean)'
    END AS content_type,
    ROUND(
        SUM(CASE WHEN p.decision = 'toxic' THEN 1.0 ELSE 0.0 END) / COUNT(*)::numeric
    , 3) AS flag_rate
FROM predictions p
JOIN ground_truth g ON g.item_id = p.item_id
WHERE p.run_id = '07bc2a8d-8ed7-4c0f-9bff-476f41533f16'
  AND g.dataset_source = 'davidson'
  AND g.davidson_class IS NOT NULL
GROUP BY p.model_name, g.davidson_class
ORDER BY p.model_name, g.davidson_class;
```

### The Unicode attack finding

We synthetically generated 10 types of evasion attacks on every toxic comment
(zero-width chars, homoglyphs, spacing, leetspeak, etc.) and ran each model on
the raw obfuscated text vs the normalized version.

Key result:
- Zero-width characters: roberta_toxicity_classifier drops from 0.87 → 0.00 recall
- Same attack: roberta-multilabel stays at 1.00 (already robust, doesn't need normalization)
- Misspellings: no model degrades — transformers learned this from noisy training data

The pattern: models fail on adversarial Unicode (never seen in training data)
but handle natural noise (typos, slang) because real text contains that.

**How we tested it:**
adversarial/generator.py created obfuscated variants of each toxic comment.
adversarial/normalizer.py cleaned them. Both versions were run through each model.
Results stored in adversarial_results table with raw_correct and norm_correct flags.

```sql
-- Which attacks hurt each model the most?
SELECT model_name, obfuscation_type,
    ROUND(AVG(CASE WHEN raw_correct  THEN 1.0 ELSE 0.0 END)::numeric, 3) AS raw_recall,
    ROUND(AVG(CASE WHEN norm_correct THEN 1.0 ELSE 0.0 END)::numeric, 3) AS norm_recall
FROM adversarial_results
WHERE true_label = 1
GROUP BY model_name, obfuscation_type
ORDER BY raw_recall ASC;
```

### The domain shift finding

Models trained on Wikipedia comments degrade on Twitter. This is expected but
the magnitude matters:

- roberta_toxicity_classifier: Jigsaw accuracy 0.984 → Davidson 0.811 (17% drop)
- xlmr-large: Jigsaw 0.898 → Davidson 0.837 (6% drop)

XLM-R was trained on multilingual social media data so it generalizes better.
We included it specifically to test whether training data diversity matters more
than model size for cross-domain robustness. It does.

```sql
-- Domain shift: accuracy per model per dataset
SELECT p.model_name, g.dataset_source,
    ROUND(AVG(CASE WHEN
        (g.toxic=1 AND p.decision='toxic') OR
        (g.toxic=0 AND p.decision='not_toxic')
    THEN 1.0 ELSE 0.0 END)::numeric, 3) AS accuracy
FROM predictions p
JOIN ground_truth g ON g.item_id = p.item_id
WHERE p.run_id = '07bc2a8d-8ed7-4c0f-9bff-476f41533f16'
  AND g.toxic IS NOT NULL
GROUP BY p.model_name, g.dataset_source
ORDER BY p.model_name, g.dataset_source;
```

### The label noise observation

When reviewing false negatives (toxic content the model missed), we found
comments like racing tweets and casual slang labeled as "offensive language"
by Davidson crowdworkers. The model was correct — the label was questionable.

This is a real data quality issue. Davidson class 1 contains genuine label
noise where crowdworker judgment was inconsistent. Evaluating recall on class 1
alone overstates false negatives.

The right evaluation: separate hate speech (class 0) from offensive language
(class 1). Recall on class 0 is the metric that actually matters for policy.

---

## The one-sentence summary for interviews

"I found that a model with F1=0.79 was completely undeployable because it flagged
82% of clean content as toxic — a failure that only appeared when I broke down
performance by the original 3-class content label instead of the collapsed binary."
