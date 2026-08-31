# adversarial

Red-teaming toolkit for content moderation robustness testing.

## The idea

Real bad actors don't type `hate` — they type `h@te`, `h a t e`, or insert
invisible Unicode characters inside words. This module generates those variants
and measures whether models still catch them.

## Files

`normalizer.py` — cleans obfuscated text before passing to a model.
Handles: Unicode NFKC normalization, homoglyph substitution (Cyrillic/Greek
lookalikes → Latin), zero-width character removal, mixed case, leetspeak,
spacing/punctuation splits, repeated character collapse, emoji flagging.

`generator.py` — generates obfuscated variants of clean text.
10 techniques: spacing, symbol_substitution, homoglyph, zero_width,
character_swap, repeated_chars, missing_vowels, mixed_case, emoji,
misspelling.

`multilingual.py` — toxic seed words in 14 languages and script detection.

## How results are stored

`adversarial_results` table has one row per (comment × technique × model).
Each row stores both the raw (obfuscated) decision and the normalized decision
so you can compare directly.

`norm_helped = TRUE` means normalization fixed a wrong prediction.
`norm_hurt = TRUE` means normalization broke a correct one (rare).

## Key finding

Models fail on adversarial Unicode (zero-width, homoglyphs) because training
data doesn't contain these. Normalization recovers most of the gap.
Misspellings barely affect anything — models learned from noisy real text.
