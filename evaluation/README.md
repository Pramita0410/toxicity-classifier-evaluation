# evaluation

Pure Python evaluation math. No database, no network calls.

## Functions

`confusion_counts(y_true, y_pred)` — counts TP, FP, FN, TN from two label sequences.

`compute_metrics(counts)` — derives precision, recall, F1, FPR, FNR.
Zero denominator always returns 0.0, never raises.

`evaluate_run(ground_truth_df, predictions_df, category, prompt_version)` — joins
truth and predictions, drops null labels, returns a metrics dict.

`threshold_sweep(ground_truth_df, predictions_df, category, prompt_version)` — runs
evaluate_run at each confidence threshold from 0.1 to 0.95. Used to find the
optimal operating point.

## Tests

Property-based tests using Hypothesis are in `tests/test_metrics.py`.
They test mathematical properties: precision always in [0,1], F1 never exceeds
max(P, R), perfect predictions → score of 1.0, etc.

Run with: `python -m pytest tests/test_metrics.py -v`
