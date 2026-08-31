"""
adversarial/multilingual.py — multilingual toxic seeds and language detection.

Why this matters for T&S
------------------------
Most rule-based and keyword-list moderators are English-only.  LLMs handle
many languages well but degrade on low-resource languages, transliterated
text, and code-switching (mixing languages mid-sentence).  This module
provides:

    1. MULTILINGUAL_TOXIC_SEEDS
       A curated dictionary mapping language codes → lists of harmful seed
       words.  Each seed is a translation/equivalent of core harmful concepts
       (hate, kill, threat, slur) in that language.  Seeds are deliberately
       generic — no real slurs, no graphic content.

    2. TRANSLITERATION_HINTS
       Common romanizations of non-Latin script toxic terms.  E.g. the
       Russian "ubey" (убей, kill) often appears in transliterated form in
       mixed-script comments.

    3. detect_language_hint(text) -> str | None
       Lightweight heuristic that returns a BCP-47 language code if the text
       contains strong signals for a non-English language (e.g. Cyrillic
       characters → "ru", Arabic script → "ar", Devanagari → "hi").
       Returns None for ambiguous/English text.  This is NOT a full language
       detector — use langdetect/fastText for production.  It is fast and
       dependency-free, which is what we need in tests.

    4. is_code_switching(text) -> bool
       Returns True when the text contains tokens from more than one language
       script family, which is a known moderation challenge.
"""

from __future__ import annotations

import re
import unicodedata


# ── Multilingual toxic seed words ────────────────────────────────────────────
# Organized by BCP-47 language code.
# Concepts covered per language:
#   hate / hatred, kill / murder, threat, insult/slur (non-graphic)
#
# NOTE: These are translations of harmful concepts, not graphic slurs.
# They are safe to commit and appropriate for a portfolio project.

MULTILINGUAL_TOXIC_SEEDS: dict[str, list[str]] = {
    # ── Spanish ───────────────────────────────────────────────────────────
    "es": [
        "odio",          # hate
        "matar",         # to kill
        "amenaza",       # threat
        "imbécil",       # insult
        "idiota",        # idiot
        "basura",        # trash
        "inútil",        # worthless
        "destruir",      # destroy
    ],
    # ── French ────────────────────────────────────────────────────────────
    "fr": [
        "haine",         # hatred
        "tuer",          # to kill
        "menace",        # threat
        "idiot",         # idiot
        "déchets",       # trash/waste
        "détruire",      # destroy
        "inutile",       # worthless
    ],
    # ── German ────────────────────────────────────────────────────────────
    "de": [
        "hass",          # hate
        "töten",         # to kill
        "drohung",       # threat
        "idiot",         # idiot
        "müll",          # trash
        "zerstören",     # destroy
        "wertlos",       # worthless
    ],
    # ── Italian ───────────────────────────────────────────────────────────
    "it": [
        "odio",          # hate
        "uccidere",      # to kill
        "minaccia",      # threat
        "idiota",        # idiot
        "spazzatura",    # trash
        "distruggere",   # destroy
        "inutile",       # worthless
    ],
    # ── Portuguese ────────────────────────────────────────────────────────
    "pt": [
        "ódio",          # hate
        "matar",         # to kill
        "ameaça",        # threat
        "idiota",        # idiot
        "lixo",          # trash
        "destruir",      # destroy
        "inútil",        # worthless
    ],
    # ── Turkish ───────────────────────────────────────────────────────────
    "tr": [
        "nefret",        # hate
        "öldürmek",      # to kill
        "tehdit",        # threat
        "aptal",         # stupid
        "çöp",           # trash
        "yok etmek",     # destroy
        "işe yaramaz",   # worthless
    ],
    # ── Russian (Cyrillic) ────────────────────────────────────────────────
    "ru": [
        "ненависть",     # hatred
        "убить",         # to kill
        "угроза",        # threat
        "идиот",         # idiot
        "мусор",         # trash
        "уничтожить",    # destroy
        "бесполезный",   # worthless
    ],
    # ── Arabic ────────────────────────────────────────────────────────────
    "ar": [
        "كره",           # hate
        "قتل",           # kill
        "تهديد",         # threat
        "أحمق",          # idiot
        "قمامة",         # trash
        "تدمير",         # destroy
        "عديم الفائدة",  # worthless
    ],
    # ── Hindi (Devanagari) ────────────────────────────────────────────────
    "hi": [
        "घृणा",          # hatred
        "मारना",         # to kill/hit
        "धमकी",         # threat
        "बेवकूफ",        # idiot
        "कचरा",          # trash
        "नष्ट करना",    # destroy
        "बेकार",         # worthless
    ],
    # ── Japanese ──────────────────────────────────────────────────────────
    "ja": [
        "憎しみ",        # hatred
        "殺す",          # to kill
        "脅迫",          # threat/intimidation
        "馬鹿",          # idiot/fool
        "ゴミ",          # trash
        "破壊する",      # destroy
        "役立たず",      # worthless
    ],
    # ── Chinese (Simplified) ──────────────────────────────────────────────
    "zh": [
        "仇恨",          # hatred
        "杀死",          # to kill
        "威胁",          # threat
        "白痴",          # idiot
        "垃圾",          # trash
        "摧毁",          # destroy
        "没用",          # worthless
    ],
    # ── Korean ────────────────────────────────────────────────────────────
    "ko": [
        "증오",          # hatred
        "죽이다",        # to kill
        "위협",          # threat
        "바보",          # idiot
        "쓰레기",        # trash
        "파괴하다",      # destroy
        "쓸모없는",      # worthless
    ],
}

# ── Transliteration hints ────────────────────────────────────────────────────
# Maps romanized (Latin-script) versions of non-Latin toxic terms to their
# source language.  These appear in comments where users type non-Latin
# languages using the Latin alphabet (common on mobile, in code-switching).

TRANSLITERATION_HINTS: dict[str, str] = {
    # Russian romanizations
    "nenavist":    "ru",   # ненависть (hatred)
    "ubit":        "ru",   # убить (to kill)
    "ugroза":      "ru",   # угроза (threat) — partial
    "musor":       "ru",   # мусор (trash)
    "idiot":       "ru",   # идиот — same in Latin, but used in Cyrillic context
    # Arabic romanizations
    "karaha":      "ar",   # كراهية (hatred)
    "qtl":         "ar",   # قتل (kill) — consonantal transliteration
    "tahdeed":     "ar",   # تهديد (threat)
    # Hindi romanizations
    "ghrina":      "hi",   # घृणा (hatred)
    "marna":       "hi",   # मारना (to hit/kill)
    "dhamki":      "hi",   # धमकी (threat)
    "bewakoof":    "hi",   # बेवकूफ (idiot)
    # Japanese romanizations
    "nikushimi":   "ja",   # 憎しみ (hatred)
    "korosu":      "ja",   # 殺す (to kill)
    "kyohaku":     "ja",   # 脅迫 (threat)
    "baka":        "ja",   # 馬鹿 (idiot) — common even in English internet
    "gomi":        "ja",   # ゴミ (trash)
    # Chinese romanizations (Pinyin)
    "chouhен":     "zh",   # 仇恨 (hatred)
    "shasi":       "zh",   # 杀死 (to kill)
    "weixie":      "zh",   # 威胁 (threat)
    "baichi":      "zh",   # 白痴 (idiot)
    "laji":        "zh",   # 垃圾 (trash)
}

# ── Unicode script range patterns ────────────────────────────────────────────
# Used by detect_language_hint() to identify scripts in a text.

_SCRIPT_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("ru",  re.compile(r"[\u0400-\u04FF]")),       # Cyrillic
    ("ar",  re.compile(r"[\u0600-\u06FF]")),       # Arabic
    ("hi",  re.compile(r"[\u0900-\u097F]")),       # Devanagari
    ("ja",  re.compile(r"[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF]")),  # Japanese
    ("zh",  re.compile(r"[\u4E00-\u9FFF]")),       # CJK Unified Ideographs
    ("ko",  re.compile(r"[\uAC00-\uD7AF]")),       # Hangul
    ("el",  re.compile(r"[\u0370-\u03FF]")),       # Greek
    ("he",  re.compile(r"[\u0590-\u05FF]")),       # Hebrew
    ("th",  re.compile(r"[\u0E00-\u0E7F]")),       # Thai
]

# Minimum character count to trigger a language hint (avoids false positives
# from single homoglyph characters like a stray Greek epsilon)
_SCRIPT_MIN_CHARS = 3


# ── Public API ────────────────────────────────────────────────────────────────

def detect_language_hint(text: str) -> str | None:
    """
    Lightweight script-based language detection.

    Checks for non-Latin script character runs and returns the most likely
    BCP-47 language code.  Returns ``None`` for text that appears to be
    Latin-script only (English, Spanish, French, etc. can't be distinguished
    without a full language model).

    This is NOT a replacement for langdetect or fastText.  It is a fast,
    dependency-free heuristic designed for use in tests and CI pipelines.

    Parameters
    ----------
    text:
        Input text to inspect.

    Returns
    -------
    str | None
        BCP-47 language code (e.g. "ru", "ar", "hi") or None.

    Examples
    --------
    >>> detect_language_hint("I hate you")      # Latin only
    None
    >>> detect_language_hint("ненависть")       # Cyrillic
    'ru'
    >>> detect_language_hint("مرحبا")           # Arabic
    'ar'
    """
    if not text:
        return None

    best_lang: str | None = None
    best_count = 0

    for lang, pattern in _SCRIPT_PATTERNS:
        matches = pattern.findall(text)
        count = len(matches)
        if count >= _SCRIPT_MIN_CHARS and count > best_count:
            best_count = count
            best_lang = lang

    return best_lang


def is_code_switching(text: str) -> bool:
    """
    Return True when the text mixes Latin and at least one non-Latin script.

    Code-switching is a documented moderation challenge: a user might write
    "I really odio this idiot" (Spanish) or "baka お前" (Japanese romaji +
    Japanese script) to confuse automated classifiers.

    Parameters
    ----------
    text:
        Input text.

    Returns
    -------
    bool

    Examples
    --------
    >>> is_code_switching("I really odio this")    # all Latin → False
    False
    >>> is_code_switching("I hate ненависть you")  # Latin + Cyrillic → True
    True
    """
    has_latin = bool(re.search(r"[a-zA-Z]", text))
    has_non_latin = any(pattern.search(text) for _, pattern in _SCRIPT_PATTERNS)
    return has_latin and has_non_latin


def get_seeds_for_language(lang: str) -> list[str]:
    """
    Return the toxic seed word list for a given language code.

    Parameters
    ----------
    lang:
        BCP-47 language code (e.g. "es", "ru", "ja").

    Returns
    -------
    list[str]
        Seed words, or an empty list if the language is not in the table.
    """
    return MULTILINGUAL_TOXIC_SEEDS.get(lang, [])


def get_all_seeds_flat() -> list[str]:
    """
    Return all toxic seed words across all languages as a flat list.

    Useful for building a cross-lingual keyword filter or for seeding
    the adversarial test set with one call.
    """
    return [
        word
        for seeds in MULTILINGUAL_TOXIC_SEEDS.values()
        for word in seeds
    ]
