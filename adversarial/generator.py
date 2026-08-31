"""
adversarial/generator.py — offensive side of the adversarial module.

Generates obfuscated variants of clean text for robustness testing.
Each variant is tagged with an ``obfuscation_type`` so downstream analysis
can answer: "Which technique most degrades raw moderation recall?"

The generator mirrors the normalizer's technique list so every variant it
produces is something the normalizer is designed to canonicalize.

Techniques
----------
    original            — unchanged text (baseline)
    misspelling         — common deliberate misspellings
    spacing             — f u c k, f.u.c.k, f-u-c-k
    symbol_substitution — h@te, k1ll, sh!t (leetspeak)
    homoglyph           — replace Latin letters with Cyrillic/Greek lookalikes
    zero_width          — insert invisible characters inside words
    character_swap      — transpose adjacent characters (htae → hate reversed)
    repeated_chars      — stretch letters (haaaaate)
    missing_vowels      — drop vowels (h8, stpd)
    mixed_case          — HaTe, sTuPiD
    emoji               — append threat-signal emoji
    multilingual        — include multilingual toxic seeds (see multilingual.py)
    code_switching      — mix English + another language mid-sentence

Public API
----------
    generate_variants(text, techniques=None) -> list[dict]
        Returns a list of {"technique": str, "raw_text": str,
        "normalized_text": str, "original": str} dicts.
        normalized_text is produced by passing raw_text through
        adversarial.normalizer.normalize().

    SUPPORTED_TECHNIQUES : list[str]
        All technique names the generator knows about.
"""

from __future__ import annotations

import random
import re
from typing import Any

from adversarial.normalizer import normalize

# ── Supported technique names ────────────────────────────────────────────────

SUPPORTED_TECHNIQUES: list[str] = [
    "original",
    "misspelling",
    "spacing",
    "symbol_substitution",
    "homoglyph",
    "zero_width",
    "character_swap",
    "repeated_chars",
    "missing_vowels",
    "mixed_case",
    "emoji",
    "multilingual",
    "code_switching",
]

# ── Per-word leet / symbol substitution map (inverse of normalizer) ──────────
# Maps a letter → one or more common evasion substitutions.
# We pick one substitution per letter at random so variants are not identical
# across calls when multiple options exist.

_LEET_SUBSTITUTIONS: dict[str, list[str]] = {
    "a": ["@", "4", "/\\"],
    "b": ["8", "|3"],
    "c": ["(", "¢"],
    "e": ["3", "€", "ε"],  # Greek epsilon
    "g": ["9", "6"],
    "h": ["|-|", "#"],
    "i": ["1", "!", "¡", "ɪ"],
    "k": ["ᴋ"],
    "l": ["|", "ʟ"],
    "o": ["0", "ο", "о"],   # Greek omicron, Cyrillic o
    "s": ["$", "5"],
    "t": ["+", "7"],
    "u": ["ü", "ʊ"],
    "x": ["*", "%", "×"],
    "z": ["2"],
}

# ── Homoglyph substitution map (Latin → lookalike Unicode) ──────────────────
# Inverse of the normalizer's _HOMOGLYPH_MAP. We pick Cyrillic/Greek
# substitutions that are visually indistinguishable in most fonts.

_HOMOGLYPH_SUBSTITUTIONS: dict[str, list[str]] = {
    "a": ["а", "α"],         # Cyrillic а, Greek alpha
    "e": ["е", "ε"],         # Cyrillic е, Greek epsilon
    "i": ["і", "ι", "¡"],   # Cyrillic і, Greek iota, inverted !
    "o": ["о", "ο"],         # Cyrillic о, Greek omicron
    "p": ["р", "ρ"],         # Cyrillic р, Greek rho
    "c": ["с", "ϲ"],         # Cyrillic с
    "x": ["х", "χ"],         # Cyrillic х, Greek chi
    "y": ["у", "γ"],         # Cyrillic у, Greek gamma
    "t": ["т", "τ"],         # Cyrillic т, Greek tau
    "k": ["κ", "ᴋ"],         # Greek kappa, small-cap K
    "n": ["ν", "ɴ"],         # Greek nu, small-cap N
    "h": ["ʜ"],               # small-cap H
    "s": ["ѕ", "ꜱ"],         # Cyrillic DZE, small-cap S
    "b": ["ʙ"],               # small-cap B
    "d": ["ԁ", "ᴅ"],         # Cyrillic KDE, small-cap D
    "v": ["ν", "ᴠ"],         # Greek nu, small-cap V
}

# ── Zero-width characters pool ───────────────────────────────────────────────

_ZERO_WIDTH_POOL: list[str] = [
    "\u200B",  # ZERO WIDTH SPACE
    "\u200C",  # ZERO WIDTH NON-JOINER
    "\u200D",  # ZERO WIDTH JOINER
    "\u2060",  # WORD JOINER
    "\uFEFF",  # BOM / ZERO WIDTH NO-BREAK SPACE
]

# ── Common deliberate misspellings ───────────────────────────────────────────
# Maps a clean word → a list of known evasion misspellings.
# Source: observed in real moderation bypass attempts.

_MISSPELLINGS: dict[str, list[str]] = {
    "hate":       ["haet", "h8", "h8e", "hte", "hayte"],
    "kill":       ["kil", "kiil", "kyll", "k1ll", "keel"],
    "stupid":     ["stoopid", "stooopid", "stewpid", "stpd", "stuped"],
    "idiot":      ["idoit", "iidiot", "idiut", "id10t"],
    "die":        ["dye", "dy", "diee"],
    "ugly":       ["uglee", "ugleh", "uglee"],
    "loser":      ["l0ser", "looser", "lozzer"],
    "trash":      ["tr@sh", "trsh", "trashh"],
    "worthless":  ["worthlss", "w0rthless", "worthles"],
    "attack":     ["attck", "att@ck", "attac"],
    "destroy":    ["destry", "destroi", "destoy"],
    "hurt":       ["hert", "h urt", "hrt"],
    "disgusting": ["discusting", "disgustng", "disgoosting"],
    "horrible":   ["horible", "horribl", "horrble"],
    "terrible":   ["terrble", "terrrible", "terible"],
}

# ── Emoji threat signals ─────────────────────────────────────────────────────

_THREAT_EMOJI: list[str] = ["🔫", "🔪", "💣", "💀", "🖕", "🤬", "🗡", "☠"]

# ── Multilingual toxic seeds (covered more fully in multilingual.py) ─────────
# A small inline set here so the generator doesn't require an import of
# multilingual.py — full seeds are in that module.

_INLINE_MULTILINGUAL: list[tuple[str, str]] = [
    ("odio",       "es"),   # Spanish: I hate
    ("matar",      "es"),   # Spanish: to kill
    ("haine",      "fr"),   # French: hate
    ("tuer",       "fr"),   # French: to kill
    ("hass",       "de"),   # German: hate
    ("töten",      "de"),   # German: to kill
    ("odio",       "it"),   # Italian: I hate
    ("nefret",     "tr"),   # Turkish: hate
    ("ненависть",  "ru"),   # Russian: hate (Cyrillic)
    ("убить",      "ru"),   # Russian: to kill (Cyrillic)
    ("घृणा",      "hi"),   # Hindi: hate (Devanagari)
    ("كره",        "ar"),   # Arabic: hate
    ("憎しみ",     "ja"),   # Japanese: hatred
    ("仇恨",       "zh"),   # Chinese: hatred
]

# ── Code-switching templates ─────────────────────────────────────────────────
# Mix English with another language mid-sentence.

_CODE_SWITCH_TEMPLATES: list[str] = [
    "i {word} you so much",           # filled with multilingual seed
    "you are {word} and i mean it",
    "this is just {word} nothing else",
    "everyone knows you are {word}",
]


# ── Per-technique generators ─────────────────────────────────────────────────

def _gen_original(text: str) -> str:
    return text


def _gen_misspelling(text: str) -> str:
    """Replace known words with deliberate misspellings."""
    words = text.split()
    result = []
    for w in words:
        key = w.lower()
        if key in _MISSPELLINGS:
            result.append(random.choice(_MISSPELLINGS[key]))
        else:
            result.append(w)
    return " ".join(result)


def _gen_spacing(text: str, sep: str = " ") -> str:
    """Insert *sep* between every character of every word."""
    words = text.split()
    spaced = [sep.join(list(w)) for w in words]
    return " ".join(spaced)


def _gen_spacing_punct(text: str) -> str:
    return _gen_spacing(text, sep=".")


def _gen_spacing_hyphen(text: str) -> str:
    return _gen_spacing(text, sep="-")


def _gen_symbol_substitution(text: str) -> str:
    """Replace letters with leetspeak symbols."""
    result = []
    for ch in text.lower():
        if ch in _LEET_SUBSTITUTIONS:
            result.append(random.choice(_LEET_SUBSTITUTIONS[ch]))
        else:
            result.append(ch)
    return "".join(result)


def _gen_homoglyph(text: str) -> str:
    """Replace Latin letters with Unicode lookalikes."""
    result = []
    for ch in text.lower():
        if ch in _HOMOGLYPH_SUBSTITUTIONS:
            result.append(random.choice(_HOMOGLYPH_SUBSTITUTIONS[ch]))
        else:
            result.append(ch)
    return "".join(result)


def _gen_zero_width(text: str) -> str:
    """Insert a zero-width character after every character."""
    zw = random.choice(_ZERO_WIDTH_POOL)
    result = []
    for i, ch in enumerate(text):
        result.append(ch)
        # Only insert inside words (not after spaces)
        if ch != " " and i < len(text) - 1 and text[i + 1] != " ":
            result.append(zw)
    return "".join(result)


def _gen_character_swap(text: str) -> str:
    """Transpose two adjacent characters in each word (Cupertino effect)."""
    words = text.split()
    result = []
    for w in words:
        if len(w) >= 3:
            # Swap chars at position 1 and 2 — detectable but not obvious
            swapped = w[0] + w[2] + w[1] + w[3:]
            result.append(swapped)
        else:
            result.append(w)
    return " ".join(result)


def _gen_repeated_chars(text: str) -> str:
    """Stretch vowels with 3–5 repetitions: hate → haaate."""
    _VOWELS = set("aeiou")
    result = []
    for ch in text:
        if ch.lower() in _VOWELS:
            result.append(ch * random.randint(3, 5))
        else:
            result.append(ch)
    return "".join(result)


def _gen_missing_vowels(text: str) -> str:
    """Drop all vowels: stupid → stpd, hate → ht."""
    _VOWELS = re.compile(r"[aeiouAEIOU]")
    words = text.split()
    result = []
    for w in words:
        stripped = _VOWELS.sub("", w)
        # Keep the word if stripping would leave it empty
        result.append(stripped if stripped else w)
    return " ".join(result)


def _gen_mixed_case(text: str) -> str:
    """Alternate upper/lower per character: HaTe."""
    result = []
    upper = True
    for ch in text:
        if ch.isalpha():
            result.append(ch.upper() if upper else ch.lower())
            upper = not upper
        else:
            result.append(ch)
    return "".join(result)


def _gen_emoji(text: str) -> str:
    """Append a random threat-signal emoji."""
    return text + " " + random.choice(_THREAT_EMOJI)


def _gen_multilingual(text: str) -> str:
    """Replace one word in *text* with a multilingual toxic seed word."""
    seed, lang = random.choice(_INLINE_MULTILINGUAL)
    words = text.split()
    if words:
        idx = random.randrange(len(words))
        words[idx] = seed
    return " ".join(words)


def _gen_code_switching(text: str) -> str:
    """Insert a multilingual toxic word into an English sentence template."""
    seed, _lang = random.choice(_INLINE_MULTILINGUAL)
    template = random.choice(_CODE_SWITCH_TEMPLATES)
    return template.format(word=seed)


# ── Dispatch table ────────────────────────────────────────────────────────────

_TECHNIQUE_FN: dict[str, Any] = {
    "original":            _gen_original,
    "misspelling":         _gen_misspelling,
    "spacing":             _gen_spacing,
    "symbol_substitution": _gen_symbol_substitution,
    "homoglyph":           _gen_homoglyph,
    "zero_width":          _gen_zero_width,
    "character_swap":      _gen_character_swap,
    "repeated_chars":      _gen_repeated_chars,
    "missing_vowels":      _gen_missing_vowels,
    "mixed_case":          _gen_mixed_case,
    "emoji":               _gen_emoji,
    "multilingual":        _gen_multilingual,
    "code_switching":      _gen_code_switching,
}


# ── Public API ────────────────────────────────────────────────────────────────

def generate_variants(
    text: str,
    techniques: list[str] | None = None,
    *,
    seed: int | None = None,
) -> list[dict[str, str]]:
    """
    Generate obfuscated variants of *text*, one per technique.

    This is the OFFENSIVE complement to ``adversarial.normalizer.normalize``.
    Each variant is also normalized so the pipeline can compare:

        raw_text → moderator → raw_decision
        normalized_text → moderator → norm_decision

    and measure: "Did normalization help for this obfuscation type?"

    Parameters
    ----------
    text:
        Clean original text (e.g. "I hate you").
    techniques:
        List of technique names to apply.  Defaults to all techniques in
        ``SUPPORTED_TECHNIQUES``.
    seed:
        Optional random seed for reproducibility.  When set, all random
        choices (leet char selection, emoji pick, etc.) are deterministic.

    Returns
    -------
    list[dict]
        One dict per technique:
        {
            "original":         str,   # unchanged input text
            "obfuscation_type": str,   # technique name
            "raw_text":         str,   # obfuscated variant
            "normalized_text":  str,   # raw_text after normalize()
        }

    Raises
    ------
    ValueError
        If any technique name is not in SUPPORTED_TECHNIQUES.

    Examples
    --------
    >>> variants = generate_variants("I hate you", techniques=["spacing"])
    >>> variants[0]["raw_text"]
    'I h a t e y o u'
    >>> variants[0]["normalized_text"]
    'i hate you'
    """
    if seed is not None:
        random.seed(seed)

    if techniques is None:
        techniques = SUPPORTED_TECHNIQUES

    unknown = [t for t in techniques if t not in _TECHNIQUE_FN]
    if unknown:
        raise ValueError(
            f"Unknown technique(s): {unknown}. "
            f"Valid: {SUPPORTED_TECHNIQUES}"
        )

    results: list[dict[str, str]] = []
    for technique in techniques:
        fn = _TECHNIQUE_FN[technique]
        raw = fn(text)
        normalized = normalize(raw)
        results.append(
            {
                "original":         text,
                "obfuscation_type": technique,
                "raw_text":         raw,
                "normalized_text":  normalized,
            }
        )

    return results


def generate_test_set(
    labeled_items: list[dict[str, Any]],
    techniques: list[str] | None = None,
    *,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """
    Expand a labeled dataset into an adversarial test set.

    For each item in *labeled_items*, generates one variant per technique,
    preserving the original ``true_label`` and ``item_id`` so the adversarial
    results can be joined back to the ground_truth table.

    Parameters
    ----------
    labeled_items:
        List of dicts, each with at least:
        {"item_id": str, "text": str, "true_label": int}
    techniques:
        Passed through to ``generate_variants``.
    seed:
        Random seed for reproducibility (default 42).

    Returns
    -------
    list[dict]
        One dict per (item × technique):
        {
            "item_id":          str,
            "true_label":       int,
            "original":         str,
            "obfuscation_type": str,
            "raw_text":         str,
            "normalized_text":  str,
        }
    """
    random.seed(seed)
    results: list[dict[str, Any]] = []

    for item in labeled_items:
        item_id = item["item_id"]
        text = item["text"]
        true_label = item["true_label"]

        for variant in generate_variants(text, techniques=techniques):
            results.append(
                {
                    "item_id":          item_id,
                    "true_label":       true_label,
                    "original":         variant["original"],
                    "obfuscation_type": variant["obfuscation_type"],
                    "raw_text":         variant["raw_text"],
                    "normalized_text":  variant["normalized_text"],
                }
            )

    return results
