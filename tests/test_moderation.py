from __future__ import annotations
import pathlib, sys
import pandas as pd
import pytest

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import Config
from moderation.moderator import RuleBasedModerator, OpenAIModerator, GeminiModerator

FIXTURE_PATH = str(PROJECT_ROOT / "tests" / "fixtures" / "llm_responses.json")
OUTPUT_COLUMNS = ["item_id","decision","confidence","model_name","prompt_version","run_id","error"]
VALID_DECISIONS = {"toxic", "not_toxic"}

def _cfg():
    cfg = Config.__new__(Config)
    for k, v in [("openai_api_key",""),("gemini_api_key",""),("openai_model","gpt-4o-mini"),
                 ("gemini_model","gemini-2.5-flash"),("use_fixtures",True),
                 ("fixture_path",FIXTURE_PATH),("run_id","test-run")]:
        object.__setattr__(cfg, k, v)
    return cfg

def _items(ids, texts=None):
    if texts is None:
        texts = [f"text {i}" for i in ids]
    return pd.DataFrame({"item_id": ids, "text": texts})

ITEMS = _items(["jigsaw_1","jigsaw_3","jigsaw_6","davidson_2","davidson_5"])

# output shape
def test_rule_based_columns(): assert list(RuleBasedModerator(_cfg()).moderate(ITEMS,"zero_shot").columns) == OUTPUT_COLUMNS
def test_openai_columns(): assert list(OpenAIModerator(_cfg()).moderate(ITEMS,"zero_shot").columns) == OUTPUT_COLUMNS
def test_gemini_columns(): assert list(GeminiModerator(_cfg()).moderate(ITEMS,"zero_shot").columns) == OUTPUT_COLUMNS
def test_row_count():
    for C in (RuleBasedModerator, OpenAIModerator, GeminiModerator):
        assert len(C(_cfg()).moderate(ITEMS,"zero_shot")) == len(ITEMS)
def test_run_id(): assert (RuleBasedModerator(_cfg()).moderate(ITEMS,"zero_shot")["run_id"] == "test-run").all()
def test_prompt_version(): assert (RuleBasedModerator(_cfg()).moderate(ITEMS,"few_shot")["prompt_version"] == "few_shot").all()

# valid decisions
@pytest.mark.parametrize("s",["zero_shot","few_shot"])
def test_rule_decisions_valid(s):
    out = RuleBasedModerator(_cfg()).moderate(ITEMS, s)
    assert out["decision"].isin(VALID_DECISIONS).all()

@pytest.mark.parametrize("s",["zero_shot","few_shot"])
def test_openai_fixture_decisions_valid(s):
    out = OpenAIModerator(_cfg()).moderate(ITEMS, s)
    assert out[out["decision"].notna()]["decision"].isin(VALID_DECISIONS).all()

@pytest.mark.parametrize("s",["zero_shot","few_shot"])
def test_gemini_fixture_decisions_valid(s):
    out = GeminiModerator(_cfg()).moderate(ITEMS, s)
    assert out[out["decision"].notna()]["decision"].isin(VALID_DECISIONS).all()

# fixture fallback
def test_openai_reads_correct_fixture():
    out = OpenAIModerator(_cfg()).moderate(_items(["jigsaw_3"]),"zero_shot")
    assert out.iloc[0]["decision"] == "toxic"

def test_gemini_reads_correct_fixture():
    out = GeminiModerator(_cfg()).moderate(_items(["jigsaw_1"]),"zero_shot")
    assert out.iloc[0]["decision"] == "not_toxic"

def test_missing_key_returns_default():
    out = OpenAIModerator(_cfg()).moderate(_items(["unknown_99"]),"zero_shot")
    assert out.iloc[0]["decision"] == "not_toxic"
    assert out.iloc[0]["confidence"] == 0.5
    assert out.iloc[0]["error"] is None

def test_few_shot_vs_zero_shot_differ():
    mod = OpenAIModerator(_cfg())
    zs = mod.moderate(_items(["davidson_5"]),"zero_shot").iloc[0]
    fs = mod.moderate(_items(["davidson_5"]),"few_shot").iloc[0]
    assert zs["decision"] == "toxic"
    assert fs["decision"] == "not_toxic"

# failure isolation
def test_one_bad_item_does_not_abort(monkeypatch):
    mod = RuleBasedModerator(_cfg())
    orig = mod._moderate_item
    def patched(item_id, text, pv):
        if item_id == "jigsaw_3": raise RuntimeError("simulated")
        return orig(item_id, text, pv)
    monkeypatch.setattr(mod, "_moderate_item", patched)
    out = mod.moderate(_items(["jigsaw_1","jigsaw_3","jigsaw_6"]),"zero_shot")
    assert len(out) == 3
    failed = out[out["item_id"]=="jigsaw_3"].iloc[0]
    assert pd.isna(failed["decision"])
    assert "simulated" in str(failed["error"])
    assert out[out["item_id"]!="jigsaw_3"]["decision"].notna().all()

def test_failed_item_null_decision(monkeypatch):
    mod = OpenAIModerator(_cfg())
    monkeypatch.setattr(mod, "_moderate_item", lambda *a: (_ for _ in ()).throw(ValueError("fail")))
    out = mod.moderate(_items(["jigsaw_1"]),"zero_shot")
    assert pd.isna(out.iloc[0]["decision"])
    assert out.iloc[0]["error"] is not None

# rule-based specific
def test_keyword_triggers_toxic():
    mod = RuleBasedModerator(_cfg())
    out = mod.moderate(_items(["x1"],["I want to kill the deadline."]),"zero_shot")
    assert out.iloc[0]["decision"] == "toxic"
    assert out.iloc[0]["confidence"] == 0.95

def test_clean_text_not_toxic():
    mod = RuleBasedModerator(_cfg())
    out = mod.moderate(_items(["x2"],["The weather is nice today."]),"zero_shot")
    assert out.iloc[0]["decision"] == "not_toxic"
    assert out.iloc[0]["confidence"] == 0.05

def test_model_name_rule_based():
    out = RuleBasedModerator(_cfg()).moderate(_items(["x3"]),"zero_shot")
    assert out.iloc[0]["model_name"] == "rule_based"
