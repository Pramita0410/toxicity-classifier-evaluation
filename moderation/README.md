# moderation

Four moderators, all with the same interface: accept a DataFrame of comments,
return a DataFrame of decisions.

## Models

`rule_based` — keyword matching, 21 words, no ML. The baseline.

`roberta_toxicity_classifier` — binary RoBERTa, trained on Jigsaw. High precision.
Runs locally via HuggingFace transformers.

`roberta-toxicity-classifier` — multi-label RoBERTa. High recall but 82% false
positive rate on clean tweets. See NOTES.md.

`xlmr-large-toxicity-classifier-v2` — multilingual XLM-R. Most robust across domains.
Large model (~2GB), slow on CPU.

## Fixture mode

When no API key is set, OpenAI and Gemini moderators read from
`tests/fixtures/llm_responses.json` instead of calling the API.
HuggingFace models always run locally — no API needed.

## Adding a new model

Subclass `BaseModerator` in `moderator.py`, implement `_moderate_item()`,
add it to the moderators list in `pipeline.py`.
