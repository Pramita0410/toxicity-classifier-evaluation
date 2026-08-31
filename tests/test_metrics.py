"""
tests/test_metrics.py — Hypothesis property-based tests for evaluation/metrics.py
plus threshold_sweep behavioral tests.

No database, no API, no network required.
Validates: FR-10, FR-11, FR-12, FR-17, NFR-1
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.metrics import (
    compute_metrics,
    confusion_counts,
    evaluate_run,
    threshold_sweep,
    DEFAULT_THRESHOLDS,
)


# ── Strategies ────────────────────────────────────────────────────────────────

_LABELS = st.sampled_from([0, 1, "toxic", "not_toxic"])


def _binary(value) -> int:
    if isinstance(value, str):
        return 1 if value.strip().lower() == "toxic" else 0
    return int(value)


def _label_pairs(min_size: int = 0, max_size: int = 50):
    return st.integers(min_value=min_size, max_value=max_size).flatmap(
        lambda n: st.tuples(
            st.lists(_LABELS, min_size=n, max_size=n),
            st.lists(_LABELS, min_size=n, max_size=n),
        )
    )


# ── Property tests ────────────────────────────────────────────────────────────

@settings(max_examples=200)
@given(_label_pairs())
def test_precision_and_recall_in_unit_interval(pairs):
    y_true, y_pred = pairs
    metrics = compute_metrics(confusion_counts(y_true, y_pred))
    assert 0.0 <= metrics["precision"] <= 1.0
    assert 0.0 <= metrics["recall"] <= 1.0


@settings(max_examples=200)
@given(_label_pairs())
def test_f1_never_exceeds_max_of_precision_recall(pairs):
    y_true, y_pred = pairs
    metrics = compute_metrics(confusion_counts(y_true, y_pred))
    upper = max(metrics["precision"], metrics["recall"])
    assert metrics["f1"] <= upper + 1e-9


@settings(max_examples=100)
@given(st.lists(_LABELS, min_size=1, max_size=50))
def test_all_correct_predictions_yield_perfect_scores(labels):
    y_true = ["toxic"] + list(labels)
    y_pred = list(y_true)
    metrics = compute_metrics(confusion_counts(y_true, y_pred))
    assert metrics["precision"] == pytest.approx(1.0)
    assert metrics["recall"] == pytest.approx(1.0)
    assert metrics["f1"] == pytest.approx(1.0)


@settings(max_examples=100)
@given(st.lists(_LABELS, min_size=1, max_size=50))
def test_all_wrong_predictions_yield_zero_scores(labels):
    y_true = list(labels)
    y_pred = [0 if _binary(v) == 1 else 1 for v in y_true]
    metrics = compute_metrics(confusion_counts(y_true, y_pred))
    assert metrics["precision"] == pytest.approx(0.0)
    assert metrics["recall"] == pytest.approx(0.0)
    assert metrics["f1"] == pytest.approx(0.0)


@settings(max_examples=100)
@given(st.integers(min_value=1, max_value=30), st.sampled_from(["toxic", "not_toxic"]))
def test_empty_after_null_filter_yields_zero_metrics(n_rows, prediction):
    ground_truth = pd.DataFrame({"item_id": list(range(n_rows)), "toxic": [None] * n_rows})
    predictions  = pd.DataFrame({"item_id": list(range(n_rows)), "decision": [prediction] * n_rows, "prompt_version": ["v1"] * n_rows})
    result = evaluate_run(ground_truth, predictions, category="toxic", prompt_version="v1")
    for key in ("precision", "recall", "f1", "fpr", "fnr"):
        assert result[key] == pytest.approx(0.0)
    assert result["tp"] == result["fp"] == result["fn"] == result["tn"] == 0


@settings(max_examples=200)
@given(
    st.integers(min_value=0, max_value=1000),
    st.integers(min_value=0, max_value=1000),
    st.integers(min_value=0, max_value=1000),
    st.integers(min_value=0, max_value=1000),
)
def test_zero_denominator_never_raises(tp, fp, fn, tn):
    counts = {"tp": tp, "fp": fp, "fn": fn, "tn": tn}
    metrics = compute_metrics(counts)
    for key in ("precision", "recall", "f1", "fpr", "fnr"):
        value = metrics[key]
        assert isinstance(value, float)
        assert math.isfinite(value)
        assert 0.0 <= value <= 1.0


# ── Threshold sweep tests ─────────────────────────────────────────────────────

class TestThresholdSweep:

    def _make_inputs(self, n: int = 10):
        ground_truth = pd.DataFrame({
            "item_id":  list(range(n)),
            "toxic":    [1, 0] * (n // 2),
        })
        predictions = pd.DataFrame({
            "item_id":        list(range(n)),
            "confidence":     [0.9, 0.1, 0.8, 0.2, 0.95, 0.05, 0.7, 0.3, 0.85, 0.15],
            "prompt_version": ["v1"] * n,
        })
        return ground_truth, predictions

    def test_returns_one_row_per_threshold(self):
        gt, pred = self._make_inputs()
        assert len(threshold_sweep(gt, pred, "toxic", "v1", thresholds=[0.3, 0.5, 0.7])) == 3

    def test_sorted_ascending(self):
        gt, pred = self._make_inputs()
        result = threshold_sweep(gt, pred, "toxic", "v1", thresholds=[0.7, 0.3, 0.5])
        ts = [r["threshold"] for r in result]
        assert ts == sorted(ts)

    def test_required_keys(self):
        gt, pred = self._make_inputs()
        row = threshold_sweep(gt, pred, "toxic", "v1", thresholds=[0.5])[0]
        assert set(row.keys()) == {"threshold", "category", "prompt_version",
                                   "tp", "fp", "fn", "tn",
                                   "precision", "recall", "f1", "fpr", "fnr"}

    def test_low_threshold_higher_recall(self):
        gt, pred = self._make_inputs()
        result = threshold_sweep(gt, pred, "toxic", "v1", thresholds=[0.01, 0.99])
        low  = next(r for r in result if r["threshold"] == 0.01)
        high = next(r for r in result if r["threshold"] == 0.99)
        assert low["recall"] >= high["recall"]

    def test_high_threshold_lower_fpr(self):
        gt, pred = self._make_inputs()
        result = threshold_sweep(gt, pred, "toxic", "v1", thresholds=[0.01, 0.99])
        low  = next(r for r in result if r["threshold"] == 0.01)
        high = next(r for r in result if r["threshold"] == 0.99)
        assert high["fpr"] <= low["fpr"]

    def test_all_metrics_in_unit_interval(self):
        gt, pred = self._make_inputs()
        for row in threshold_sweep(gt, pred, "toxic", "v1"):
            for key in ("precision", "recall", "f1", "fpr", "fnr"):
                assert math.isfinite(row[key])
                assert 0.0 <= row[key] <= 1.0

    def test_empty_inputs_returns_empty(self):
        empty_gt   = pd.DataFrame(columns=["item_id", "toxic"])
        empty_pred = pd.DataFrame(columns=["item_id", "confidence"])
        assert threshold_sweep(empty_gt, empty_pred, "toxic", "v1") == []

    def test_missing_confidence_returns_empty(self):
        gt   = pd.DataFrame({"item_id": [1, 2], "toxic": [1, 0]})
        pred = pd.DataFrame({"item_id": [1, 2], "decision": ["toxic", "not_toxic"]})
        assert threshold_sweep(gt, pred, "toxic", "v1") == []

    def test_default_thresholds_range(self):
        assert min(DEFAULT_THRESHOLDS) <= 0.10
        assert max(DEFAULT_THRESHOLDS) >= 0.90
        assert len(DEFAULT_THRESHOLDS) >= 10

    def test_null_labels_excluded(self):
        gt   = pd.DataFrame({"item_id": [1, 2, 3], "toxic": [1, None, 0]})
        pred = pd.DataFrame({"item_id": [1, 2, 3], "confidence": [0.9, 0.8, 0.1], "prompt_version": ["v1"]*3})
        row  = threshold_sweep(gt, pred, "toxic", "v1", thresholds=[0.5])[0]
        assert row["tp"] + row["fp"] + row["fn"] + row["tn"] == 2
