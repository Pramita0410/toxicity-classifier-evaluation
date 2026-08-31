"""
moderation/prompts.py — prompt builders for the LLM moderation component.

Exposes a single public function:
    build_prompt(text: str, strategy: str) -> str

Supported strategies:
    "zero_shot" — plain instruction, no examples
    "few_shot"  — same instruction preceded by 3 labeled examples

Both strategies demand a strict binary output: the model must reply with
exactly "toxic" or "not_toxic" and nothing else.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_INSTRUCTION = (
    "You are a content-moderation classifier. "
    "Decide whether the following text is harmful or not.\n"
    "Reply with exactly one word — either toxic or not_toxic — and nothing else."
)

# Three example pairs used in few-shot prompts:
#   1. Clearly toxic (explicit threat / slur)
#   2. Borderline (aggressive but not overtly harmful)
#   3. Clearly clean (neutral discussion)
_FEW_SHOT_EXAMPLES: list[tuple[str, str]] = [
    (
        "You are a complete idiot and nobody likes you.",
        "toxic",
    ),
    (
        "This is absolutely the worst product I have ever bought.",
        "not_toxic",
    ),
    (
        "The weather today is sunny and warm.",
        "not_toxic",
    ),
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_prompt(text: str, strategy: str) -> str:
    """Return a moderation prompt for *text* using the given *strategy*.

    Parameters
    ----------
    text:
        The user-generated content to classify.
    strategy:
        One of ``"zero_shot"`` or ``"few_shot"``.

    Returns
    -------
    str
        A complete prompt string ready to send to an LLM.

    Raises
    ------
    ValueError
        If *strategy* is not ``"zero_shot"`` or ``"few_shot"``.
    """
    if strategy == "zero_shot":
        return _build_zero_shot(text)
    elif strategy == "few_shot":
        return _build_few_shot(text)
    else:
        raise ValueError(
            f"Unknown prompt strategy {strategy!r}. "
            "Expected 'zero_shot' or 'few_shot'."
        )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _build_zero_shot(text: str) -> str:
    return f"{_INSTRUCTION}\n\nText: {text}"


def _build_few_shot(text: str) -> str:
    example_block = "\n".join(
        f"Text: {example_text}\nLabel: {label}"
        for example_text, label in _FEW_SHOT_EXAMPLES
    )
    return (
        f"{_INSTRUCTION}\n\n"
        f"Examples:\n{example_block}\n\n"
        f"Text: {text}"
    )
