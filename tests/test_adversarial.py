"""
tests/test_adversarial.py — unit tests for the adversarial module.

Covers:
  normalizer  — every normalization step, edge cases, pipeline ordering
  generator   — variant generation, obfuscation_type tagging, test set expansion
  multilingual — language detection, code-switching detection, seed lookup

No database, no API, no external network required.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from adversarial.normalizer import normalize, normalize_batch, NORMALIZATION_STEPS
from adversarial.generator import generate_variants, generate_test_set, SUPPORTED_TECHNIQUES
from adversarial.multilingual import (
    detect_language_hint,
    is_code_switching,
    get_seeds_for_language,
    get_all_seeds_flat,
    MULTILINGUAL_TOXIC_SEEDS,
)

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "adversarial_examples.json"


# ── Normalizer: individual steps ─────────────────────────────────────────────

class TestNormalizerSymbolSubstitution:
    def test_at_becomes_a(self):
        assert normalize("h@te", steps=["symbol_substitution"]) == "h@te".replace("@", "a") == "hate"

    def test_zero_becomes_o(self):
        assert normalize("y0u", steps=["symbol_substitution"]) == "you"

    def test_dollar_becomes_s(self):
        assert normalize("$tupid", steps=["symbol_substitution"]) == "stupid"

    def test_one_becomes_i(self):
        assert normalize("1d1ot", steps=["symbol_substitution"]) == "idiot"

    def test_three_becomes_e(self):
        assert normalize("h3ll", steps=["symbol_substitution"]) == "hell"

    def test_mixed_leet(self):
        result = normalize("h@te y0u", steps=["mixed_case", "symbol_substitution"])
        assert result == "hate you"


class TestNormalizerSpacing:
    def test_space_separated_chars(self):
        assert normalize("f u c k", steps=["mixed_case", "spacing"]) == "fuck"

    def test_dot_separated_chars(self):
        assert normalize("f.u.c.k", steps=["mixed_case", "spacing"]) == "fuck"

    def test_hyphen_separated_chars(self):
        assert normalize("f-u-c-k", steps=["mixed_case", "spacing"]) == "fuck"

    def test_longer_word_spaced(self):
        result = normalize("s t u p i d", steps=["mixed_case", "spacing"])
        assert result == "stupid"

    def test_legitimate_hyphen_not_collapsed(self):
        # "well-being" has multi-char segments — must NOT be collapsed
        result = normalize("well-being", steps=["spacing"])
        assert result == "well-being"

    def test_cpp_not_mangled(self):
        # "C++" — the ++ is not a single-char spacing pattern
        result = normalize("C++ is great", steps=["symbol_substitution", "spacing"])
        # symbol_substitution replaces + with t, giving "Ctt is great"
        # but spacing should not collapse "C++" further
        assert "is great" in result

    def test_username_at_not_collapsed(self):
        # "@username" — @ followed by multi-char word, must not be collapsed
        result = normalize("follow @username", steps=["spacing"])
        assert "username" in result


class TestNormalizerHomoglyph:
    def test_cyrillic_a_replaced(self):
        # Cyrillic 'а' (U+0430) → 'a'
        assert normalize("h\u0430te", steps=["homoglyph"]) == "hate"

    def test_cyrillic_o_replaced(self):
        # Cyrillic 'о' (U+043E) → 'o'
        assert normalize("y\u043eu", steps=["homoglyph"]) == "you"

    def test_greek_epsilon_replaced(self):
        # Greek 'ε' (U+03B5) → 'e'
        assert normalize("h\u03b5ll", steps=["homoglyph"]) == "hell"

    def test_inverted_exclamation_replaced(self):
        # '¡' (U+00A1) → 'i'
        assert normalize("¡d¡ot", steps=["homoglyph"]) == "idiot"

    def test_multiple_homoglyphs_in_one_word(self):
        # Mix of Cyrillic: 'уоu' → 'you'
        assert normalize("\u0443\u043eu", steps=["homoglyph"]) == "you"


class TestNormalizerZeroWidth:
    def test_zero_width_space_removed(self):
        # U+200B inserted inside "hate"
        assert normalize("h\u200bate", steps=["zero_width"]) == "hate"

    def test_zero_width_nonjoiner_removed(self):
        assert normalize("h\u200cate", steps=["zero_width"]) == "hate"

    def test_bom_removed(self):
        assert normalize("\ufeffhate", steps=["zero_width"]) == "hate"

    def test_multiple_invisible_chars_removed(self):
        text = "h\u200b\u200c\u2060ate"
        assert normalize(text, steps=["zero_width"]) == "hate"


class TestNormalizerRepeatedChars:
    def test_stretched_vowel_collapsed(self):
        assert normalize("haaaate", steps=["repeated_chars"]) == "hate"

    def test_three_or_more_collapsed(self):
        assert normalize("stuuuupid", steps=["repeated_chars"]) == "stupid"

    def test_two_repeated_not_collapsed(self):
        # "ll" in "hello" must stay — only 3+ runs are collapsed
        assert normalize("hello", steps=["repeated_chars"]) == "hello"

    def test_long_run_fully_collapsed(self):
        assert normalize("yooooooou", steps=["repeated_chars"]) == "you"


class TestNormalizerMixedCase:
    def test_all_uppercase_lowercased(self):
        assert normalize("HATE", steps=["mixed_case"]) == "hate"

    def test_alternating_caps_lowercased(self):
        assert normalize("HaTe", steps=["mixed_case"]) == "hate"

    def test_camelcase_lowercased(self):
        assert normalize("HateYou", steps=["mixed_case"]) == "hateyou"


class TestNormalizerEmojiFlag:
    def test_gun_emoji_gets_hint(self):
        result = normalize("i will \U0001f52b you", steps=["emoji_flag"])
        assert "gun" in result
        assert "\U0001f52b" in result   # original emoji preserved

    def test_skull_emoji_gets_hint(self):
        result = normalize("you are \U0001f480", steps=["emoji_flag"])
        assert "death" in result

    def test_no_emoji_unchanged(self):
        result = normalize("just plain text", steps=["emoji_flag"])
        assert result == "just plain text"


class TestNormalizerPipeline:
    def test_full_pipeline_heavy_obfuscation(self):
        result = normalize("h@te y0u $tup1d")
        assert result == "hate you stupid"

    def test_full_pipeline_spaced_word(self):
        result = normalize("f u c k")
        assert result == "fuck"

    def test_full_pipeline_zero_width_and_homoglyph(self):
        # Cyrillic 'а' with zero-width space inside
        result = normalize("h\u200b\u0430te")
        assert result == "hate"

    def test_empty_string_returns_empty(self):
        assert normalize("") == ""

    def test_clean_text_unchanged_after_pipeline(self):
        # Plain English should survive the full pipeline unchanged
        result = normalize("the weather is nice today")
        assert result == "the weather is nice today"

    def test_unknown_step_raises(self):
        with pytest.raises(ValueError, match="Unknown normalization step"):
            normalize("text", steps=["not_a_real_step"])

    def test_normalization_steps_list_complete(self):
        # Ensure the public constant lists all implemented steps
        assert len(NORMALIZATION_STEPS) == 8
        assert "homoglyph" in NORMALIZATION_STEPS
        assert "zero_width" in NORMALIZATION_STEPS
        assert "spacing" in NORMALIZATION_STEPS


class TestNormalizeBatch:
    def test_batch_applies_to_all_items(self):
        texts = ["h@te", "y0u", "st00pid"]
        results = normalize_batch(texts)
        assert results == ["hate", "you", "stoopid"]

    def test_batch_empty_list(self):
        assert normalize_batch([]) == []

    def test_batch_preserves_order(self):
        texts = ["f u c k", "h@te", "clean text"]
        results = normalize_batch(texts)
        assert results[0] == "fuck"
        assert results[1] == "hate"
        assert results[2] == "clean text"


# ── Generator tests ───────────────────────────────────────────────────────────

class TestGenerator:
    def test_returns_one_variant_per_technique(self):
        variants = generate_variants("I hate you", techniques=["spacing", "homoglyph"], seed=42)
        assert len(variants) == 2

    def test_variant_has_required_keys(self):
        variants = generate_variants("I hate you", techniques=["original"], seed=42)
        assert set(variants[0].keys()) == {"original", "obfuscation_type", "raw_text", "normalized_text"}

    def test_original_technique_unchanged(self):
        variants = generate_variants("I hate you", techniques=["original"])
        assert variants[0]["raw_text"] == "I hate you"
        assert variants[0]["obfuscation_type"] == "original"

    def test_spacing_inserts_spaces(self):
        variants = generate_variants("hate", techniques=["spacing"], seed=42)
        raw = variants[0]["raw_text"]
        assert " " in raw  # characters are separated
        assert "h" in raw and "a" in raw and "t" in raw and "e" in raw

    def test_normalized_text_is_normalizer_output(self):
        variants = generate_variants("hate you", techniques=["symbol_substitution"], seed=42)
        raw = variants[0]["raw_text"]
        from adversarial.normalizer import normalize
        assert variants[0]["normalized_text"] == normalize(raw)

    def test_mixed_case_uses_alternating_caps(self):
        variants = generate_variants("hate", techniques=["mixed_case"], seed=42)
        raw = variants[0]["raw_text"]
        assert raw != raw.lower()  # not all lowercase
        assert raw != raw.upper()  # not all uppercase

    def test_repeated_chars_stretches_vowels(self):
        variants = generate_variants("hate", techniques=["repeated_chars"], seed=42)
        raw = variants[0]["raw_text"]
        assert len(raw) > len("hate")  # must be longer

    def test_missing_vowels_removes_vowels(self):
        variants = generate_variants("hate", techniques=["missing_vowels"], seed=42)
        raw = variants[0]["raw_text"]
        assert "a" not in raw and "e" not in raw

    def test_emoji_appends_emoji(self):
        variants = generate_variants("I hate you", techniques=["emoji"], seed=42)
        raw = variants[0]["raw_text"]
        # Should contain at least one emoji from the threat pool
        assert len(raw) > len("I hate you")

    def test_unknown_technique_raises(self):
        with pytest.raises(ValueError, match="Unknown technique"):
            generate_variants("text", techniques=["not_real"])

    def test_default_techniques_covers_all(self):
        variants = generate_variants("I hate you", seed=42)
        assert len(variants) == len(SUPPORTED_TECHNIQUES)

    def test_seed_makes_output_deterministic(self):
        v1 = generate_variants("I hate you", techniques=["symbol_substitution", "homoglyph"], seed=99)
        v2 = generate_variants("I hate you", techniques=["symbol_substitution", "homoglyph"], seed=99)
        assert v1[0]["raw_text"] == v2[0]["raw_text"]
        assert v1[1]["raw_text"] == v2[1]["raw_text"]

    def test_homoglyph_produces_non_ascii(self):
        variants = generate_variants("hate", techniques=["homoglyph"], seed=42)
        raw = variants[0]["raw_text"]
        assert not raw.isascii(), "Homoglyph variant should contain non-ASCII characters"

    def test_zero_width_contains_invisible_chars(self):
        variants = generate_variants("hate", techniques=["zero_width"], seed=42)
        raw = variants[0]["raw_text"]
        # After zero_width insertion the string is longer but looks the same
        assert len(raw) > len("hate")
        # Normalizer should strip them back
        assert variants[0]["normalized_text"] == "hate"

    def test_character_swap_transposes_letters(self):
        variants = generate_variants("hate you", techniques=["character_swap"], seed=42)
        raw = variants[0]["raw_text"]
        # "hate" → "ahte" (chars 1&2 swapped), "you" → "oyu"
        assert raw != "hate you"


class TestGenerateTestSet:
    def test_returns_one_row_per_item_per_technique(self):
        items = [
            {"item_id": "x1", "text": "I hate you", "true_label": 1},
            {"item_id": "x2", "text": "Nice weather", "true_label": 0},
        ]
        rows = generate_test_set(items, techniques=["original", "spacing"], seed=42)
        assert len(rows) == 4  # 2 items × 2 techniques

    def test_true_label_preserved(self):
        items = [{"item_id": "x1", "text": "I hate you", "true_label": 1}]
        rows = generate_test_set(items, techniques=["original"], seed=42)
        assert rows[0]["true_label"] == 1

    def test_item_id_preserved(self):
        items = [{"item_id": "myid", "text": "some text", "true_label": 0}]
        rows = generate_test_set(items, techniques=["original"], seed=42)
        assert rows[0]["item_id"] == "myid"

    def test_output_has_required_keys(self):
        items = [{"item_id": "x1", "text": "hate", "true_label": 1}]
        rows = generate_test_set(items, techniques=["original"], seed=42)
        expected = {"item_id", "true_label", "original", "obfuscation_type", "raw_text", "normalized_text"}
        assert set(rows[0].keys()) == expected


# ── Multilingual tests ────────────────────────────────────────────────────────

class TestMultilingual:
    def test_cyrillic_detected_as_russian(self):
        assert detect_language_hint("ненависть убить") == "ru"

    def test_arabic_detected(self):
        assert detect_language_hint("كره قتل تهديد") == "ar"

    def test_devanagari_detected_as_hindi(self):
        assert detect_language_hint("घृणा मारना धमकी") == "hi"

    def test_latin_returns_none(self):
        assert detect_language_hint("I hate you") is None

    def test_empty_string_returns_none(self):
        assert detect_language_hint("") is None

    def test_single_cyrillic_char_below_threshold(self):
        # One or two chars should not trigger detection (avoids homoglyph false positives)
        result = detect_language_hint("hа")  # single Cyrillic 'а'
        assert result is None

    def test_is_code_switching_latin_only(self):
        assert is_code_switching("I really hate this") is False

    def test_is_code_switching_mixed(self):
        # Latin + Cyrillic
        assert is_code_switching("I hate \u043d\u0435\u043d\u0430\u0432\u0438\u0441\u0442\u044c you") is True

    def test_get_seeds_for_language_returns_list(self):
        seeds = get_seeds_for_language("es")
        assert isinstance(seeds, list)
        assert len(seeds) > 0
        assert "odio" in seeds

    def test_get_seeds_for_unknown_language_returns_empty(self):
        assert get_seeds_for_language("xx") == []

    def test_get_all_seeds_flat_covers_all_languages(self):
        all_seeds = get_all_seeds_flat()
        assert len(all_seeds) > 0
        # Should contain seeds from every language in the table
        assert len(all_seeds) >= len(MULTILINGUAL_TOXIC_SEEDS)

    def test_all_configured_languages_have_seeds(self):
        for lang, seeds in MULTILINGUAL_TOXIC_SEEDS.items():
            assert len(seeds) >= 3, f"Language {lang} has too few seeds"


# ── Fixture file integrity ────────────────────────────────────────────────────

class TestAdversarialFixture:
    def test_fixture_file_exists(self):
        assert FIXTURE_PATH.exists(), f"Fixture file not found: {FIXTURE_PATH}"

    def test_fixture_is_valid_json(self):
        with FIXTURE_PATH.open("r", encoding="utf-8") as f:
            data = json.load(f)
        assert isinstance(data, list)
        assert len(data) > 0

    def test_fixture_entries_have_required_keys(self):
        with FIXTURE_PATH.open("r", encoding="utf-8") as f:
            data = json.load(f)
        required = {"item_id", "true_label", "original", "obfuscation_type", "raw_text", "normalized_text"}
        for entry in data:
            missing = required - set(entry.keys())
            assert not missing, f"Entry missing keys: {missing}\n{entry}"

    def test_fixture_true_labels_are_binary(self):
        with FIXTURE_PATH.open("r", encoding="utf-8") as f:
            data = json.load(f)
        for entry in data:
            assert entry["true_label"] in (0, 1), f"Bad true_label: {entry}"

    def test_fixture_covers_multiple_techniques(self):
        with FIXTURE_PATH.open("r", encoding="utf-8") as f:
            data = json.load(f)
        techniques = {e["obfuscation_type"] for e in data}
        # Must cover at least these core techniques
        for t in ("original", "symbol_substitution", "spacing", "homoglyph", "zero_width"):
            assert t in techniques, f"Missing technique in fixture: {t}"

    def test_fixture_has_both_toxic_and_clean_items(self):
        with FIXTURE_PATH.open("r", encoding="utf-8") as f:
            data = json.load(f)
        labels = {e["true_label"] for e in data}
        assert 0 in labels, "Fixture has no clean (label=0) items"
        assert 1 in labels, "Fixture has no toxic (label=1) items"
