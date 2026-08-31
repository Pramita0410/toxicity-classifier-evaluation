-- =============================================================================
-- superset_datasets.sql
-- SQL queries for creating virtual datasets in Apache Superset SQL Lab.
--
-- How to use:
--   1. Open Superset → SQL Lab
--   2. Copy one query block
--   3. Run it, verify it returns data
--   4. Click "Save Dataset" → give it the name in the header comment
--   5. Build charts from that dataset
--
-- Run ID used throughout: 07bc2a8d-8ed7-4c0f-9bff-476f41533f16
-- Replace with your run_id if you re-ran the pipeline.
-- =============================================================================


-- =============================================================================
-- DATASET: predictions_main
-- Use for: model comparison charts, FP/FN analysis, error browser
-- =============================================================================

SELECT
    p.model_name,
    g.dataset_source,
    g.toxic           AS true_label,
    p.decision,
    p.confidence,
    g.item_id,
    LEFT(g.text, 200) AS text_preview,
    CASE
        WHEN g.toxic = 1 AND p.decision = 'not_toxic' THEN 'FN'
        WHEN g.toxic = 0 AND p.decision = 'toxic'     THEN 'FP'
        ELSE 'correct'
    END AS verdict,
    CASE WHEN g.toxic = 1 AND p.decision = 'toxic'     THEN 1 ELSE 0 END AS tp,
    CASE WHEN g.toxic = 0 AND p.decision = 'toxic'     THEN 1 ELSE 0 END AS fp,
    CASE WHEN g.toxic = 1 AND p.decision = 'not_toxic' THEN 1 ELSE 0 END AS fn,
    CASE WHEN g.toxic = 0 AND p.decision = 'not_toxic' THEN 1 ELSE 0 END AS tn,
    CASE WHEN g.toxic = 1 THEN 1 ELSE 0 END AS is_toxic
FROM predictions p
JOIN ground_truth g ON g.item_id = p.item_id
WHERE p.run_id = '07bc2a8d-8ed7-4c0f-9bff-476f41533f16'
  AND g.toxic IS NOT NULL;


-- =============================================================================
-- DATASET: model_pr
-- Use for: Precision-Recall scatter plot (deployment decision chart)
-- One row per model with precision, recall, F1
-- =============================================================================

SELECT
    model_name,
    ROUND(
        SUM(CASE WHEN g.toxic=1 AND p.decision='toxic' THEN 1.0 ELSE 0.0 END) /
        NULLIF(SUM(CASE WHEN p.decision='toxic' THEN 1 ELSE 0 END), 0)::numeric
    , 3) AS precision,
    ROUND(
        SUM(CASE WHEN g.toxic=1 AND p.decision='toxic' THEN 1.0 ELSE 0.0 END) /
        NULLIF(SUM(CASE WHEN g.toxic=1 THEN 1 ELSE 0 END), 0)::numeric
    , 3) AS recall,
    ROUND(
        2.0 *
        (SUM(CASE WHEN g.toxic=1 AND p.decision='toxic' THEN 1.0 ELSE 0.0 END) /
         NULLIF(SUM(CASE WHEN p.decision='toxic' THEN 1 ELSE 0 END), 0)) *
        (SUM(CASE WHEN g.toxic=1 AND p.decision='toxic' THEN 1.0 ELSE 0.0 END) /
         NULLIF(SUM(CASE WHEN g.toxic=1 THEN 1 ELSE 0 END), 0)) /
        NULLIF(
            (SUM(CASE WHEN g.toxic=1 AND p.decision='toxic' THEN 1.0 ELSE 0.0 END) /
             NULLIF(SUM(CASE WHEN p.decision='toxic' THEN 1 ELSE 0 END), 0)) +
            (SUM(CASE WHEN g.toxic=1 AND p.decision='toxic' THEN 1.0 ELSE 0.0 END) /
             NULLIF(SUM(CASE WHEN g.toxic=1 THEN 1 ELSE 0 END), 0))
        , 0)::numeric
    , 3) AS f1,
    SUM(CASE WHEN g.toxic=1 AND p.decision='toxic'     THEN 1 ELSE 0 END) AS tp,
    SUM(CASE WHEN g.toxic=0 AND p.decision='toxic'     THEN 1 ELSE 0 END) AS fp,
    SUM(CASE WHEN g.toxic=1 AND p.decision='not_toxic' THEN 1 ELSE 0 END) AS fn,
    SUM(CASE WHEN g.toxic=0 AND p.decision='not_toxic' THEN 1 ELSE 0 END) AS tn
FROM predictions p
JOIN ground_truth g ON g.item_id = p.item_id
WHERE p.run_id = '07bc2a8d-8ed7-4c0f-9bff-476f41533f16'
  AND g.toxic IS NOT NULL
GROUP BY model_name;


-- =============================================================================
-- DATASET: domain_shift
-- Use for: Domain shift bar chart (jigsaw vs davidson per model)
-- Shows how models degrade on tweets vs Wikipedia comments
-- =============================================================================

SELECT
    p.model_name,
    g.dataset_source,
    COUNT(*)                                                                    AS total,
    ROUND(
        SUM(CASE WHEN g.toxic=1 AND p.decision='toxic'     THEN 1.0 ELSE 0.0 END) /
        NULLIF(SUM(CASE WHEN g.toxic=1 THEN 1 ELSE 0 END), 0)::numeric
    , 3) AS recall,
    ROUND(
        SUM(CASE WHEN g.toxic=1 AND p.decision='toxic'     THEN 1.0 ELSE 0.0 END) /
        NULLIF(SUM(CASE WHEN p.decision='toxic' THEN 1 ELSE 0 END), 0)::numeric
    , 3) AS precision,
    ROUND(
        SUM(CASE WHEN
            (g.toxic=1 AND p.decision='toxic') OR
            (g.toxic=0 AND p.decision='not_toxic')
        THEN 1.0 ELSE 0.0 END) / COUNT(*)::numeric
    , 3) AS accuracy
FROM predictions p
JOIN ground_truth g ON g.item_id = p.item_id
WHERE p.run_id = '07bc2a8d-8ed7-4c0f-9bff-476f41533f16'
  AND g.toxic IS NOT NULL
GROUP BY p.model_name, g.dataset_source
ORDER BY p.model_name, g.dataset_source;


-- =============================================================================
-- DATASET: adversarial_main
-- Use for: Heatmaps showing raw_recall, norm_recall, recall_delta per attack
-- =============================================================================

SELECT
    model_name,
    obfuscation_type,
    ROUND(AVG(CASE WHEN raw_correct  THEN 1.0 ELSE 0.0 END)::numeric, 3) AS raw_recall,
    ROUND(AVG(CASE WHEN norm_correct THEN 1.0 ELSE 0.0 END)::numeric, 3) AS norm_recall,
    ROUND(
        AVG(CASE WHEN norm_correct THEN 1.0 ELSE 0.0 END)::numeric -
        AVG(CASE WHEN raw_correct  THEN 1.0 ELSE 0.0 END)::numeric
    , 3) AS recall_delta,
    SUM(CASE WHEN norm_helped THEN 1 ELSE 0 END) AS times_helped,
    SUM(CASE WHEN norm_hurt   THEN 1 ELSE 0 END) AS times_hurt,
    COUNT(*)                                       AS n
FROM adversarial_results
WHERE true_label = 1
GROUP BY model_name, obfuscation_type
ORDER BY raw_recall ASC;


-- =============================================================================
-- DATASET: adversarial_comparison
-- Use for: Single heatmap comparing raw vs normalized recall side by side
-- Each attack appears twice: "spacing [raw]" and "spacing [norm]"
-- =============================================================================

SELECT
    model_name,
    obfuscation_type || ' [raw]'  AS attack_version,
    ROUND(AVG(CASE WHEN raw_correct  THEN 1.0 ELSE 0.0 END)::numeric, 3) AS recall
FROM adversarial_results
WHERE true_label = 1
GROUP BY model_name, obfuscation_type

UNION ALL

SELECT
    model_name,
    obfuscation_type || ' [norm]' AS attack_version,
    ROUND(AVG(CASE WHEN norm_correct THEN 1.0 ELSE 0.0 END)::numeric, 3) AS recall
FROM adversarial_results
WHERE true_label = 1
GROUP BY model_name, obfuscation_type

ORDER BY attack_version, model_name;


-- =============================================================================
-- DATASET: error_browser
-- Use for: Table showing actual misclassified comments with full text
-- High-confidence errors are the most interesting (model was sure but wrong)
-- =============================================================================

SELECT
    p.model_name,
    LEFT(ea.text, 200)                    AS comment,
    ea.true_label,
    ea.predicted_label,
    ROUND(ea.confidence::numeric, 3)      AS confidence,
    ea.error_type,
    g.dataset_source,
    g.severe_toxic,
    g.obscene,
    g.threat,
    g.insult,
    g.identity_hate
FROM error_analysis ea
JOIN predictions p
    ON  p.item_id      = ea.item_id
    AND p.run_id       = ea.run_id
    AND p.decision     = ea.predicted_label
JOIN ground_truth g ON g.item_id = ea.item_id
WHERE ea.run_id = '07bc2a8d-8ed7-4c0f-9bff-476f41533f16'
ORDER BY ea.confidence DESC;


-- =============================================================================
-- DATASET: threshold_curve
-- Use for: Line chart showing precision/recall at different confidence thresholds
-- Answers: "Where should we set the cutoff?"
-- Computed directly from predictions — no dependency on threshold_metrics table
-- =============================================================================

SELECT
    p.model_name,
    t.threshold,
    ROUND(
        SUM(CASE WHEN g.toxic = 1 AND p.confidence >= t.threshold THEN 1.0 ELSE 0.0 END) /
        NULLIF(SUM(CASE WHEN g.toxic = 1 THEN 1 ELSE 0 END), 0)::numeric
    , 3) AS recall,
    ROUND(
        SUM(CASE WHEN g.toxic = 1 AND p.confidence >= t.threshold THEN 1.0 ELSE 0.0 END) /
        NULLIF(SUM(CASE WHEN p.confidence >= t.threshold THEN 1 ELSE 0 END), 0)::numeric
    , 3) AS precision
FROM predictions p
JOIN ground_truth g ON g.item_id = p.item_id
CROSS JOIN (
    SELECT generate_series * 0.05 AS threshold
    FROM generate_series(2, 19)
) t
WHERE p.run_id = '07bc2a8d-8ed7-4c0f-9bff-476f41533f16'
  AND g.toxic IS NOT NULL
  AND p.confidence IS NOT NULL
GROUP BY p.model_name, t.threshold
ORDER BY p.model_name, t.threshold;


-- =============================================================================
-- DATASET: category_breakdown
-- Use for: Grouped bar showing F1 per harm category per model
-- Shows which harm types are hardest to detect
-- =============================================================================

SELECT
    p.model_name,
    g.dataset_source,
    'toxic'        AS category,
    ROUND(SUM(CASE WHEN g.toxic=1 AND p.decision='toxic' THEN 1.0 ELSE 0.0 END) /
        NULLIF(SUM(CASE WHEN g.toxic IS NOT NULL THEN 1 ELSE 0 END), 0)::numeric, 3) AS recall
FROM predictions p
JOIN ground_truth g ON g.item_id = p.item_id
WHERE p.run_id = '07bc2a8d-8ed7-4c0f-9bff-476f41533f16'
  AND g.toxic IS NOT NULL
GROUP BY p.model_name, g.dataset_source

UNION ALL

SELECT
    p.model_name,
    g.dataset_source,
    'obscene'      AS category,
    ROUND(SUM(CASE WHEN g.obscene=1 AND p.decision='toxic' THEN 1.0 ELSE 0.0 END) /
        NULLIF(SUM(CASE WHEN g.obscene=1 THEN 1 ELSE 0 END), 0)::numeric, 3)
FROM predictions p
JOIN ground_truth g ON g.item_id = p.item_id
WHERE p.run_id = '07bc2a8d-8ed7-4c0f-9bff-476f41533f16'
  AND g.obscene IS NOT NULL
GROUP BY p.model_name, g.dataset_source

UNION ALL

SELECT
    p.model_name,
    g.dataset_source,
    'threat'       AS category,
    ROUND(SUM(CASE WHEN g.threat=1 AND p.decision='toxic' THEN 1.0 ELSE 0.0 END) /
        NULLIF(SUM(CASE WHEN g.threat=1 THEN 1 ELSE 0 END), 0)::numeric, 3)
FROM predictions p
JOIN ground_truth g ON g.item_id = p.item_id
WHERE p.run_id = '07bc2a8d-8ed7-4c0f-9bff-476f41533f16'
  AND g.threat IS NOT NULL
GROUP BY p.model_name, g.dataset_source

UNION ALL

SELECT
    p.model_name,
    g.dataset_source,
    'insult'       AS category,
    ROUND(SUM(CASE WHEN g.insult=1 AND p.decision='toxic' THEN 1.0 ELSE 0.0 END) /
        NULLIF(SUM(CASE WHEN g.insult=1 THEN 1 ELSE 0 END), 0)::numeric, 3)
FROM predictions p
JOIN ground_truth g ON g.item_id = p.item_id
WHERE p.run_id = '07bc2a8d-8ed7-4c0f-9bff-476f41533f16'
  AND g.insult IS NOT NULL
GROUP BY p.model_name, g.dataset_source

UNION ALL

SELECT
    p.model_name,
    g.dataset_source,
    'identity_hate' AS category,
    ROUND(SUM(CASE WHEN g.identity_hate=1 AND p.decision='toxic' THEN 1.0 ELSE 0.0 END) /
        NULLIF(SUM(CASE WHEN g.identity_hate=1 THEN 1 ELSE 0 END), 0)::numeric, 3)
FROM predictions p
JOIN ground_truth g ON g.item_id = p.item_id
WHERE p.run_id = '07bc2a8d-8ed7-4c0f-9bff-476f41533f16'
  AND g.identity_hate IS NOT NULL
GROUP BY p.model_name, g.dataset_source

ORDER BY category, model_name;
