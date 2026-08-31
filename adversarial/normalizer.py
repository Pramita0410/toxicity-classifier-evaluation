"""
adversarial/normalizer.py — defensive canonicalization of obfuscated text.

The normalizer is the DEFENSIVE side of the adversarial module.  It takes
text that may have been deliberately mangled to evade keyword filters and
returns a canonical form that a downstream moderator (rule-based or LLM) can
reason about correctly.

Techniques handled (in pipeline order):
    1. unicode_nfkc       — Unicode NFKC normalization (collapses compatibility
                            equivalents, decomposes ligatures, normalizes width)
    2. homoglyph          — Replace Cyrillic, Greek, and other look-alike
                            characters with their Latin equivalents
    3. zero_width         — Strip invisible/zero-width Unicode characters
    4. mixed_case         — Lowercase everything
    5. symbol_substitution— Leetspeak / symbol → letter  (@→a, 3→e, 0→o, …)
    6. spacing            — Remove intra-word spaces/punctuation inserted to
                            split tokens (f u c k → fuck, f.u.c.k → fuck)
    7. repeated_chars     — Collapse runs of 3+ identical chars to 2
                            (stuuuupid → stuupid, then stuupid is close enough
                            for edit-distance matching; we collapse to 1 for
                            cleaner downstream matching)
    8. missing_vowels     — NOT re-inserted (we can't reliably invert vowel
                            dropping without a dictionary; instead we flag it)
    9. character_swap     — NOT corrected (transpositions need a dictionary too)
   10. emoji_flag         — Emojis are NOT stripped (LLMs understand them);
                            instead we append a plain-text hint token so
                            rule-based filters can catch obvious weapon/threat
                            emoji clusters.

Public API
----------
    normalize(text, *, steps=None) -> str
        Apply the full normalization pipeline (or a named subset) to *text*.
        Returns the canonical string.

    NORMALIZATION_STEPS : list[str]
        Ordered list of all step names, in the order they run.
"""

from __future__ import annotations

import re
import unicodedata

# ── Ordered step names ──────────────────────────────────────────────────────

NORMALIZATION_STEPS: list[str] = [
    "unicode_nfkc",
    "homoglyph",
    "zero_width",
    "mixed_case",
    "symbol_substitution",
    "spacing",
    "repeated_chars",
    "emoji_flag",
]

# ── Homoglyph map ───────────────────────────────────────────────────────────
# Maps Unicode lookalike characters → ASCII Latin equivalents.
# Sources: Cyrillic confusables, Greek confusables, mathematical alphanumerics,
# fullwidth Latin, common Unicode lookalikes used in real evasion attempts.

_HOMOGLYPH_MAP: dict[str, str] = {
    # ── Cyrillic → Latin ──────────────────────────────────────────────────
    "а": "a",  # U+0430 CYRILLIC SMALL LETTER A
    "А": "A",  # U+0410 CYRILLIC CAPITAL LETTER A
    "е": "e",  # U+0435 CYRILLIC SMALL LETTER IE
    "Е": "E",  # U+0415 CYRILLIC CAPITAL LETTER IE
    "і": "i",  # U+0456 CYRILLIC SMALL LETTER BYELORUSSIAN-UKRAINIAN I
    "І": "I",  # U+0406
    "о": "o",  # U+043E CYRILLIC SMALL LETTER O
    "О": "O",  # U+041E CYRILLIC CAPITAL LETTER O
    "р": "p",  # U+0440 CYRILLIC SMALL LETTER ER (looks like p)
    "Р": "P",  # U+0420
    "с": "c",  # U+0441 CYRILLIC SMALL LETTER ES (looks like c)
    "С": "C",  # U+0421
    "у": "y",  # U+0443 CYRILLIC SMALL LETTER U (looks like y)
    "У": "Y",  # U+0423
    "х": "x",  # U+0445 CYRILLIC SMALL LETTER HA (looks like x)
    "Х": "X",  # U+0425
    "ԁ": "d",  # U+0501 CYRILLIC SMALL LETTER KOMI DE
    "ѕ": "s",  # U+0455 CYRILLIC SMALL LETTER DZE (looks like s in some fonts)
    "ո": "n",  # U+0578 ARMENIAN SMALL LETTER VO
    # ── Greek → Latin ─────────────────────────────────────────────────────
    "α": "a",  # U+03B1 GREEK SMALL LETTER ALPHA
    "Α": "A",  # U+0391
    "β": "b",  # U+03B2 GREEK SMALL LETTER BETA (looks like b in some fonts)
    "ε": "e",  # U+03B5 GREEK SMALL LETTER EPSILON
    "Ε": "E",  # U+0395
    "ι": "i",  # U+03B9 GREEK SMALL LETTER IOTA
    "Ι": "I",  # U+0399
    "κ": "k",  # U+03BA
    "Κ": "K",  # U+039A
    "ν": "v",  # U+03BD GREEK SMALL LETTER NU (looks like v)
    "ο": "o",  # U+03BF GREEK SMALL LETTER OMICRON
    "Ο": "O",  # U+039F
    "ρ": "p",  # U+03C1 GREEK SMALL LETTER RHO (looks like p)
    "Ρ": "P",  # U+03A1
    "τ": "t",  # U+03C4
    "Τ": "T",  # U+03A4
    "υ": "u",  # U+03C5
    "χ": "x",  # U+03C7 GREEK SMALL LETTER CHI
    "Χ": "X",  # U+03A7
    # ── Mathematical / script alphanumerics (common in styled text) ────────
    # These appear in social-media styled fonts ("𝒉𝒂𝒕𝒆", "𝐡𝐚𝐭𝐞", etc.)
    # NFKC normalization handles most of these; we add a few stragglers.
    "ℎ": "h",  # U+210E PLANCK CONSTANT
    "ℓ": "l",  # U+2113 SCRIPT SMALL L
    "℮": "e",  # U+212E ESTIMATED SIGN
    "ℯ": "e",  # U+212F SCRIPT SMALL E
    # ── Fullwidth Latin (used heavily in East-Asian text decoration) ───────
    # NFKC handles the A-Z/a-z range; add punctuation stragglers
    "！": "!",
    "＠": "@",
    "＃": "#",
    "＄": "$",
    "％": "%",
    "＊": "*",
    # ── Other common confusables ──────────────────────────────────────────
    "¡": "i",  # U+00A1 INVERTED EXCLAMATION MARK used as 'i' in "¡d¡ot"
    "ı": "i",  # U+0131 LATIN SMALL LETTER DOTLESS I
    "ʟ": "l",  # U+029C LATIN LETTER SMALL CAPITAL L
    "ʜ": "h",  # U+029C LATIN LETTER SMALL CAPITAL H
    "ᴋ": "k",  # U+1D0B LATIN LETTER SMALL CAPITAL K
    "ᴡ": "w",  # U+1D21 LATIN LETTER SMALL CAPITAL W
    "ʏ": "y",  # U+028F LATIN LETTER SMALL CAPITAL Y
    "ᴅ": "d",  # U+1D05 LATIN LETTER SMALL CAPITAL D
    "ᴠ": "v",  # U+1D20 LATIN LETTER SMALL CAPITAL V
    "ᴢ": "z",  # U+1D22 LATIN LETTER SMALL CAPITAL Z
    "ᴘ": "p",  # U+1D18 LATIN LETTER SMALL CAPITAL P
    "ꜰ": "f",  # U+A730 LATIN LETTER SMALL CAPITAL F
    "ɴ": "n",  # U+0274 LATIN LETTER SMALL CAPITAL N
    "ɢ": "g",  # U+0262 LATIN LETTER SMALL CAPITAL G
    "ꜱ": "s",  # U+A731 LATIN LETTER SMALL CAPITAL S
    "ᴛ": "t",  # U+1D1B LATIN LETTER SMALL CAPITAL T
    "ᴍ": "m",  # U+1D0D LATIN LETTER SMALL CAPITAL M
    "ʀ": "r",  # U+0280 LATIN LETTER SMALL CAPITAL R
    "ʙ": "b",  # U+0299 LATIN LETTER SMALL CAPITAL B
    "ᴄ": "c",  # U+1D04 LATIN LETTER SMALL CAPITAL C
    "ᴏ": "o",  # U+1D0F LATIN LETTER SMALL CAPITAL O
    "ᴜ": "u",  # U+1D1C LATIN LETTER SMALL CAPITAL U
    "ᴀ": "a",  # U+1D00 LATIN LETTER SMALL CAPITAL A
    "ᴇ": "e",  # U+1D07 LATIN LETTER SMALL CAPITAL E
    "ɪ": "i",  # U+026A LATIN LETTER SMALL CAPITAL I
    # ── Digits that look like letters ─────────────────────────────────────
    # (handled below in symbol_substitution too, but useful here for
    #  non-leetspeak contexts like styled numerals)
    "⁰": "0", "¹": "1", "²": "2", "³": "3", "⁴": "4",
    "⁵": "5", "⁶": "6", "⁷": "7", "⁸": "8", "⁹": "9",
}

# Build a translation table for O(1) per-character lookup
_HOMOGLYPH_TABLE: dict[int, str] = {ord(k): v for k, v in _HOMOGLYPH_MAP.items()}

# ── Zero-width / invisible character set ────────────────────────────────────

_ZERO_WIDTH_CHARS: frozenset[int] = frozenset(
    [
        0x200B,  # ZERO WIDTH SPACE
        0x200C,  # ZERO WIDTH NON-JOINER
        0x200D,  # ZERO WIDTH JOINER
        0x200E,  # LEFT-TO-RIGHT MARK
        0x200F,  # RIGHT-TO-LEFT MARK
        0x2060,  # WORD JOINER
        0x2061,  # FUNCTION APPLICATION (invisible)
        0x2062,  # INVISIBLE TIMES
        0x2063,  # INVISIBLE SEPARATOR
        0x2064,  # INVISIBLE PLUS
        0xFEFF,  # ZERO WIDTH NO-BREAK SPACE / BOM
        0x00AD,  # SOFT HYPHEN (renders invisibly in most contexts)
        0x034F,  # COMBINING GRAPHEME JOINER
        0x180E,  # MONGOLIAN VOWEL SEPARATOR (space-like but invisible)
        0x2028,  # LINE SEPARATOR
        0x2029,  # PARAGRAPH SEPARATOR
    ]
)

# ── Symbol-substitution / leetspeak map ─────────────────────────────────────
# Maps symbols/digits commonly used in evasion back to their letter equivalents.
# Order matters for multi-character sequences — check longer keys first.

_LEET_MAP: dict[str, str] = {
    # Multi-character sequences first
    "|-|": "h",
    "|_|": "u",
    "|3": "b",
    "|=": "f",
    "|-": "f",
    "/\\/": "m",
    "\\/\\/": "w",
    "/\\": "a",
    "\\/": "v",
    "(": "c",
    ")": "",    # closing paren — usually noise after 'c'
    # Single characters
    "@": "a",
    "4": "a",
    "8": "b",
    "3": "e",
    "€": "e",
    "6": "g",
    "#": "h",
    "!": "i",
    "1": "i",
    "|": "i",
    "0": "o",
    "9": "g",
    "$": "s",
    "5": "s",
    "+": "t",
    "7": "t",
    "%": "x",
    "2": "z",
    "×": "x",
    "*": "x",
    "^": "a",
    "~": "",    # tilde — usually noise
}

# We'll apply leet substitution character-by-character using the single-char map;
# multi-char sequences are handled by a pre-pass regex replacement.
_LEET_MULTI: list[tuple[re.Pattern[str], str]] = [
    (re.compile(re.escape(src), re.IGNORECASE), dst)
    for src, dst in _LEET_MAP.items()
    if len(src) > 1
]

_LEET_SINGLE: dict[int, str] = {
    ord(src): dst for src, dst in _LEET_MAP.items() if len(src) == 1
}

# ── Emoji → plain-text hint map ─────────────────────────────────────────────
# We do NOT strip emojis — LLMs handle them fine.  For rule-based moderation we
# append a short plain-text token so keyword matching still works.
# Only map emojis that carry strong threat/hate/harm signal when used alone.

_EMOJI_HINTS: dict[str, str] = {
    "🔫": " gun ",
    "🔪": " knife ",
    "💣": " bomb ",
    "💀": " death ",
    "☠": " death ",
    "🖕": " offensive_gesture ",
    "🤬": " angry ",
    "😡": " angry ",
    "🤢": " disgust ",
    "🤮": " disgust ",
    "🗡": " sword ",
    "⚔": " sword ",
    "🧨": " explosive ",
    "☢": " nuclear ",
    "☣": " biohazard ",
    "🪓": " axe ",
    "🏹": " arrow ",
}

# ── Spacing / punctuation insertion patterns ─────────────────────────────────
# Detects runs of "single-char SEP single-char SEP single-char ..." and joins
# them.  Strategy: find maximal runs of (letter)(sep)(letter) and collapse the
# whole run at once, rather than relying on \b which breaks mid-collapse.

# Matches a maximal run of single alphanum characters separated by the same
# kind of intra-word punctuation/space.  Examples:
#   "f u c k"     → group 1 = "f u c k"
#   "f.u.c.k"     → group 1 = "f.u.c.k"
#   "f-u-c-k"     → group 1 = "f-u-c-k"
_SPACED_WORD_RUN = re.compile(
    r"(?<!\w)"                      # not preceded by a word char
    r"([a-z0-9]"                    # first single char
    r"(?:[ .\-_,;:]+[a-z0-9])+)"   # one or more SEP+char pairs
    r"(?!\w)",                      # not followed by a word char
    re.IGNORECASE,
)

# Separator characters to strip when collapsing a matched run
_SEP_CHARS = re.compile(r"[ .\-_,;:]+")

# ── Repeated character pattern ────────────────────────────────────────────────
# Collapses 3+ repeated identical characters to 1: "stuuuupid" → "stupid",
# "haaaate" → "hate".  We keep 1 (not 2) because the collapsed form is the
# canonical word — downstream edit-distance or exact matching then works.

_REPEATED_CHARS = re.compile(r"(.)\1{2,}")


# ── Pipeline step implementations ────────────────────────────────────────────

def _step_unicode_nfkc(text: str) -> str:
    """NFKC normalization: collapse compatibility chars, decompose ligatures,
    normalize fullwidth/halfwidth, remove combining marks where safe."""
    return unicodedata.normalize("NFKC", text)


def _step_homoglyph(text: str) -> str:
    """Replace look-alike Unicode characters with their Latin equivalents."""
    return text.translate(_HOMOGLYPH_TABLE)


def _step_zero_width(text: str) -> str:
    """Strip invisible zero-width and formatting characters."""
    return "".join(ch for ch in text if ord(ch) not in _ZERO_WIDTH_CHARS)


def _step_mixed_case(text: str) -> str:
    """Lowercase everything."""
    return text.lower()


def _step_symbol_substitution(text: str) -> str:
    """Leetspeak / symbol → letter substitution.

    Multi-character sequences are replaced first, then single characters.
    We only apply substitution inside word-like tokens to avoid mangling
    legitimate punctuation in surrounding prose.
    """
    # Multi-char pass
    for pattern, replacement in _LEET_MULTI:
        text = pattern.sub(replacement, text)
    # Single-char pass
    return text.translate(_LEET_SINGLE)


def _step_spacing(text: str) -> str:
    """Collapse intra-word spacing / punctuation splits.

    Detects maximal runs of single characters separated by spaces or
    punctuation (f u c k, f.u.c.k, f-u-c-k) and joins them into one token.
    Uses a single-pass regex over the whole run — no iteration needed.

    Legitimate multi-character words with hyphens (well-being, C++) are NOT
    collapsed because they don't match the single-char-per-segment pattern.
    """
    def _collapse(m: re.Match) -> str:  # type: ignore[type-arg]
        return _SEP_CHARS.sub("", m.group(1))

    return _SPACED_WORD_RUN.sub(_collapse, text)


def _step_repeated_chars(text: str) -> str:
    """Collapse 3+ repeated identical characters to 1: haaaaate → hate."""
    return _REPEATED_CHARS.sub(r"\1", text)


def _step_emoji_flag(text: str) -> str:
    """Append plain-text hint tokens for threat/harm-signal emojis.

    We keep the original emoji intact (LLMs understand them) and add a
    space-padded ASCII token right after each matched emoji so rule-based
    keyword filters can still catch the signal.
    """
    for emoji, hint in _EMOJI_HINTS.items():
        text = text.replace(emoji, emoji + hint)
    return text


# ── Step dispatch table ───────────────────────────────────────────────────────

_STEP_FN: dict[str, "Callable[[str], str]"] = {  # type: ignore[name-defined]
    "unicode_nfkc": _step_unicode_nfkc,
    "homoglyph": _step_homoglyph,
    "zero_width": _step_zero_width,
    "mixed_case": _step_mixed_case,
    "symbol_substitution": _step_symbol_substitution,
    "spacing": _step_spacing,
    "repeated_chars": _step_repeated_chars,
    "emoji_flag": _step_emoji_flag,
}


# ── Public API ───────────────────────────────────────────────────────────────

def normalize(text: str, *, steps: list[str] | None = None) -> str:
    """
    Canonicalize obfuscated text by applying the normalization pipeline.

    Parameters
    ----------
    text:
        Raw input text, potentially containing obfuscation.
    steps:
        Ordered list of step names to apply.  Defaults to
        ``NORMALIZATION_STEPS`` (the full pipeline in the recommended order).
        Pass a subset to apply only specific techniques.

    Returns
    -------
    str
        Canonicalized text suitable for passing to a downstream moderator.

    Examples
    --------
    >>> normalize("h@te y0u")
    'hate you'
    >>> normalize("f u c k")
    'fuck'
    >>> normalize("stuuuupid")
    'stupid'
    >>> normalize("h\u200bate")   # zero-width space inside "hate"
    'hate'
    """
    if steps is None:
        steps = NORMALIZATION_STEPS

    for step_name in steps:
        fn = _STEP_FN.get(step_name)
        if fn is None:
            raise ValueError(
                f"Unknown normalization step: {step_name!r}. "
                f"Valid steps: {list(_STEP_FN)}"
            )
        text = fn(text)

    return text


def normalize_batch(texts: list[str], *, steps: list[str] | None = None) -> list[str]:
    """
    Apply ``normalize`` to every item in *texts*.

    Parameters
    ----------
    texts:
        List of raw input strings.
    steps:
        Passed through to ``normalize``.

    Returns
    -------
    list[str]
        Canonicalized strings in the same order as *texts*.
    """
    return [normalize(t, steps=steps) for t in texts]
