"""
evaluation/metrics.py — pure-Python evaluation math for the AI Safety
Evaluation Platform.

This is the CORE of the project: it grades moderation decisions against
ground-truth labels. It is deliberately dependency-light — no database,
no network, no LLM clients. Just confusion-matrix counting and the metrics
derived from it.

Four public functions:

    confusion_counts(y_true, y_pred) -> dict[str, int]
        Count TP, FP, FN, TN from two aligned sequences of 0/1 labels.

    compute_metrics(counts) -> dict[str, float]
        Turn those counts into precision, recall, F1, FPR, FNR.
        Every zero-denominator case returns 0.0 and never raises.

    evaluate_run(ground_truth_df, predictions_df, category, prompt_version) -> dict
        Join a category column of ground truth with a set of predictions,
        drop rows whose true label is null, then delegate to the two
        functions above. Returns a metrics row ready to insert into the
        `metrics` table.

    threshold_sweep(ground_truth_df, predictions_df, category, prompt_version,
                    thresholds) -> list[dict]
        Sweep confidence thresholds from 0.1 to 0.9 (or a custom list),
        computing precision/recall/F1/FPR/FNR at each operating point.
        Answers: "At what confidence cutoff should we call something toxic?"
        Returns a list of dicts ready to insert into the `threshold_metrics` table.

Reference math (see design.md §4):

    precision = TP / (TP + FP)      -> 0.0 if TP + FP == 0
    recall    = TP / (TP + FN)      -> 0.0 if TP + FN == 0
    f1        = 2PR / (P + R)       -> 0.0 if P + R  == 0
    fpr       = FP / (FP + TN)      -> 0.0 if FP + TN == 0
    fnr       = FN / (FN + TP)      -> 0.0 if FN + TP == 0
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

import pandas as pd

# Default threshold operating points used by threshold_sweep when the caller
# does not supply a custom list.  Covers the full range at 0.05 steps so the
# resulting precision-recall curve is smooth enough to reason about.
DEFAULT_THRESHOLDS: list[float] = [round(t * 0.05, 2) for t in range(2, 20)]
# → [0.10, 0.15, 0.20, ..., 0.90, 0.95]


# ── Label normalisation ─────────────────────────────────────────────────────

_POSITIVE_STRINGS = {"toxic", "positive", "1", "true", "yes"}
_NEGATIVE_STRINGS = {"not_toxic", "nontoxic", "negative", "0", "false", "no"}


def _to_binary(value: Any) -> int:
    if isinstance(value, bool):
        return 1 if value else 0
    if isinstance(value, (int, float)):
        as_int = int(value)
        if as_int in (0, 1):
            return as_int
        raise ValueError(f"Numeric label must be 0 or 1, got: {value!r}")
    if isinstance(value, str):
        token = value.strip().lower()
        if token in _POSITIVE_STRINGS:
            return 1
        if token in _NEGATIVE_STRINGS:
            return 0
        raise ValueError(f"Unrecognised string label: {value!r}")
    raise ValueError(f"Unsupported label type: {type(value).__name__} ({value!r})")


# ── Confusion matrix ────────────────────────────────────────────────────────

def confusion_counts(
    y_true: Sequence[Any] | Iterable[Any],
    y_pred: Sequence[Any] | Iterable[Any],
) -> dict[str, int]:
    true_list = list(y_true)
    pred_list = list(y_pred)
    if len(true_list) != len(pred_list):
        raise ValueError(
            f"y_true and y_pred must be the same length "
            f"({len(true_list)} != {len(pred_list)})"
        )
    tp = fp = fn = tn = 0
    for raw_true, raw_pred in zip(true_list, pred_list):
        t = _to_binary(raw_true)
        p = _to_binary(raw_pred)
        if t == 1 and p == 1:
            tp += 1
        elif t == 0 and p == 1:
            fp += 1
        elif t == 1 and p == 0:
            fn += 1
        else:
            tn += 1
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


# ── Derived metrics ─────────────────────────────────────────────────────────

def _safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def compute_metrics(counts: dict[str, int]) -> dict[str, float]:
    tp = int(counts.get("tp", 0))
    fp = int(counts.get("fp", 0))
    fn = int(counts.get("fn", 0))
    tn = int(counts.get("tn", 0))
    precision = _safe_divide(tp, tp + fp)
    recall = _safe_divide(tp, tp + fn)
    f1 = _safe_divide(2 * precision * recall, precision + recall)
    fpr = _safe_divide(fp, fp + tn)
    fnr = _safe_divide(fn, fn + tp)
    return {"precision": precision, "recall": recall, "f1": f1, "fpr": fpr, "fnr": fnr}


# ── Full run evaluation ─────────────────────────────────────────────────────

def evaluate_run(
    ground_truth_df: pd.DataFrame,
    predictions_df: pd.DataFrame,
    category: str,
    prompt_version: str,
) -> dict[str, Any]:
    base_result: dict[str, Any] = {"category": category, "prompt_version": prompt_version}
    if (
        ground_truth_df is None
        or predictions_df is None
        or ground_truth_df.empty
        or predictions_df.empty
        or category not in ground_truth_df.columns
        or "item_id" not in ground_truth_df.columns
        or "item_id" not in predictions_df.columns
        or "decision" not in predictions_df.columns
    ):
        zero_counts = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
        return {**base_result, **zero_counts, **compute_metrics(zero_counts)}
    preds = predictions_df
    if "prompt_version" in preds.columns:
        preds = preds[preds["prompt_version"] == prompt_version]
    preds = preds[preds["decision"].notna()]
    truth = ground_truth_df[["item_id", category]].copy()
    truth = truth[truth[category].notna()]
    merged = truth.merge(preds[["item_id", "decision"]], on="item_id", how="inner")
    if merged.empty:
        zero_counts = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
        return {**base_result, **zero_counts, **compute_metrics(zero_counts)}
    counts = confusion_counts(merged[category], merged["decision"])
    metrics = compute_metrics(counts)
    return {**base_result, **counts, **metrics}


# ── Threshold sweep ─────────────────────────────────────────────────────────

def threshold_sweep(
    ground_truth_df: pd.DataFrame,
    predictions_df: pd.DataFrame,
    category: str,
    prompt_version: str,
    thresholds: list[float] | None = None,
) -> list[dict[str, Any]]:
    """
    Compute precision/recall/F1/FPR/FNR at each confidence threshold.

    Real content-moderation systems never use a hard 0.5 cutoff.  They tune
    the operating threshold to balance false positives (over-removal) against
    false negatives (missed harm).  This function sweeps a range of thresholds
    and returns one metrics row per point so you can plot the tradeoff curve
    or find the threshold that hits a target recall.

    How it works:
        At each threshold t, a prediction is treated as "toxic" only when
        confidence >= t.  Items whose confidence is null (failed calls)
        are always treated as "not_toxic" at every threshold.

    Args:
        ground_truth_df:  DataFrame with ``item_id`` and the ``category`` column.
        predictions_df:   DataFrame with ``item_id``, ``decision``, ``confidence``,
                          and optionally ``prompt_version``.
        category:         Ground-truth label column to score against (e.g. "toxic").
        prompt_version:   Filters predictions to a single prompt strategy.
        thresholds:       List of floats in (0, 1).  Defaults to
                          DEFAULT_THRESHOLDS (0.10 … 0.95 in 0.05 steps).

    Returns:
        List of dicts, one per threshold:
        {threshold, category, prompt_version, tp, fp, fn, tn,
         precision, recall, f1, fpr, fnr}
        Sorted ascending by threshold.  Empty list on bad/missing inputs.
    """
    if thresholds is None:
        thresholds = DEFAULT_THRESHOLDS

    if (
        ground_truth_df is None
        or predictions_df is None
        or ground_truth_df.empty
        or predictions_df.empty
        or category not in ground_truth_df.columns
        or "item_id" not in ground_truth_df.columns
        or "item_id" not in predictions_df.columns
        or "confidence" not in predictions_df.columns
    ):
        return []

    preds = predictions_df.copy()
    if "prompt_version" in preds.columns:
        preds = preds[preds["prompt_version"] == prompt_version].copy()

    truth = ground_truth_df[["item_id", category]].copy()
    truth = truth[truth[category].notna()]

    if truth.empty or preds.empty:
        return []

    merged = truth.merge(preds[["item_id", "confidence"]], on="item_id", how="inner")

    if merged.empty:
        return []

    results: list[dict[str, Any]] = []

    for t in sorted(thresholds):
        y_pred_binary = (
            merged["confidence"].astype(float).fillna(0.0).ge(t).astype(int).tolist()
        )
        y_true_binary = merged[category].tolist()

        counts = confusion_counts(y_true_binary, y_pred_binary)
        metrics = compute_metrics(counts)

        results.append(
            {
                "threshold": t,
                "category": category,
                "prompt_version": prompt_version,
                **counts,
                **metrics,
            }
        )

    return results
