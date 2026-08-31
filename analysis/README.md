# analysis

Named SQL queries you can run from the command line to answer specific questions.

## Usage

```bash
# See all available queries
python -m analysis.queries --list

# Run a specific query
python -m analysis.queries --query adversarial_robustness
python -m analysis.queries --query normalization_impact
python -m analysis.queries --query best_prompt_by_metric

# Run all queries
python -m analysis.queries --all
```

## Available queries

`worst_categories` — which harm types are hardest to detect  
`model_comparison` — side-by-side precision/recall/F1 for all models  
`false_positive_samples` — highest-confidence wrong flags  
`false_negative_samples` — highest-confidence missed harm  
`threshold_operating_points` — minimum threshold to hit 90% recall per model  
`adversarial_robustness` — which attack causes biggest recall drop  
`normalization_impact` — does preprocessing actually help per technique  
`multilingual_breakdown` — performance on multilingual adversarial items  
`run_summary` — one-line summary of every pipeline run  
