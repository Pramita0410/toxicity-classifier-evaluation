"""
moderation/moderator.py — LLM moderation component.

Public classes
--------------
  BaseModerator          abstract base; all subclasses implement moderate()
  RuleBasedModerator     keyword-list approach, no API required
  OpenAIModerator        GPT-4o-mini via openai library; fixture fallback
  GeminiModerator        Gemini 2.5 Flash via google.generativeai; fixture fallback

All moderators:
  - Accept a DataFrame of ground-truth items and a prompt_version string.
  - Return a DataFrame with columns:
      item_id, decision, confidence, model_name, prompt_version, run_id, error
  - Wrap every per-item call in try/except; failures get decision=None, error logged.
  - In fixture mode (no API key or USE_FIXTURES=True), read pre-recorded responses
    from cfg.fixture_path (JSON keyed by item_id). Missing keys → safe default.
"""

from __future__ import annotations

import json
import logging
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import pandas as pd

if TYPE_CHECKING:
    from sqlalchemy.engine import Engine

    from config import Config

from moderation.prompts import build_prompt

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_VALID_DECISIONS = {"toxic", "not_toxic"}

# Default returned when a fixture entry is missing for an item
_FIXTURE_MISSING_DECISION = "not_toxic"
_FIXTURE_MISSING_CONFIDENCE = 0.5

# Output DataFrame columns (order matters for downstream consumers)
_OUTPUT_COLUMNS = [
    "item_id",
    "decision",
    "confidence",
    "model_name",
    "prompt_version",
    "run_id",
    "error",
]

# Keyword list for rule-based moderation.
# Kept minimal and deliberately non-graphic — no real slurs, no toxicity here.
_TOXIC_KEYWORDS: frozenset[str] = frozenset(
    {
        "kill",
        "die",
        "hate",
        "idiot",
        "stupid",
        "dumb",
        "moron",
        "loser",
        "trash",
        "garbage",
        "worthless",
        "ugly",
        "disgusting",
        "horrible",
        "terrible",
        "awful",
        "threat",
        "attack",
        "destroy",
        "harm",
        "hurt",
    }
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_fixtures(fixture_path: str) -> dict[str, dict]:
    """Load JSON fixture file. Returns empty dict if the file doesn't exist."""
    path = Path(fixture_path)
    if not path.exists():
        logger.warning("Fixture file not found: %s — fixture responses will be empty.", fixture_path)
        return {}
    with path.open("r", encoding="utf-8-sig") as fh:
        data = json.load(fh)
    logger.info("Loaded %d fixture entries from %s", len(data), fixture_path)
    return data


def _parse_llm_response(raw: str) -> tuple[str | None, float | None]:
    """
    Extract (decision, confidence) from a raw LLM reply string.

    The prompt asks for exactly "toxic" or "not_toxic". We normalise by:
      1. Lower-casing and stripping whitespace.
      2. Checking for the exact tokens anywhere in the (possibly messy) reply.
      3. If neither token is found, returning (None, None) to signal failure.

    Confidence is heuristic — 0.9 when the model replied cleanly with one word,
    0.7 when we had to extract it from noisy output.
    """
    cleaned = raw.strip().lower()

    # Clean single-word reply
    if cleaned in _VALID_DECISIONS:
        return cleaned, 0.9

    # Noisy reply — try to find a valid token
    for token in ("not_toxic", "toxic"):   # check not_toxic first (longer match)
        if re.search(r"\b" + token.replace("_", r"[_\s]?") + r"\b", cleaned):
            return token, 0.7

    return None, None


def _get_fixture_response(
    fixtures: dict[str, dict],
    item_id: str,
    prompt_version: str,
) -> tuple[str, float]:
    """
    Look up a pre-recorded fixture response.

    The fixture JSON is expected to have this shape:
        {
          "<item_id>": {
            "zero_shot": {"decision": "toxic", "confidence": 0.9},
            "few_shot":  {"decision": "not_toxic", "confidence": 0.8}
          },
          ...
        }

    Falls back to _FIXTURE_MISSING_* constants when the item/prompt is absent.
    """
    if item_id not in fixtures:
        return _FIXTURE_MISSING_DECISION, _FIXTURE_MISSING_CONFIDENCE

    entry = fixtures[item_id]

    # Support both nested-by-prompt_version and flat (no prompt_version key) formats
    if isinstance(entry, dict) and prompt_version in entry:
        pv_entry = entry[prompt_version]
    else:
        pv_entry = entry

    decision = pv_entry.get("decision", _FIXTURE_MISSING_DECISION)
    confidence = pv_entry.get("confidence", _FIXTURE_MISSING_CONFIDENCE)

    if decision not in _VALID_DECISIONS:
        logger.warning(
            "Fixture entry for item_id=%s has invalid decision=%r; using default.",
            item_id, decision,
        )
        decision = _FIXTURE_MISSING_DECISION

    return decision, float(confidence)


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------

class BaseModerator(ABC):
    """
    Abstract base for all moderation strategies.

    Subclasses must implement ``_moderate_item`` which handles a single text
    and returns ``(decision, confidence)``.  The ``moderate`` method iterates
    the DataFrame, wraps each call in a try/except, and assembles the output.
    """

    def __init__(self, cfg: "Config", engine: Optional["Engine"] = None) -> None:
        self.cfg = cfg
        self.engine = engine
        self._fixtures: dict[str, dict] | None = None  # loaded lazily

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def moderate(
        self,
        items_df: pd.DataFrame,
        prompt_version: str,
    ) -> pd.DataFrame:
        """
        Run moderation over all rows in *items_df*.

        Parameters
        ----------
        items_df:
            DataFrame that must have at least ``item_id`` and ``text`` columns.
            Typically the ``ground_truth`` table loaded into a DataFrame.
        prompt_version:
            One of ``"zero_shot"`` or ``"few_shot"``.

        Returns
        -------
        pd.DataFrame
            One row per input item with columns:
            item_id, decision, confidence, model_name, prompt_version, run_id, error
        """
        records: list[dict] = []

        for _, row in items_df.iterrows():
            item_id = str(row["item_id"])
            text = str(row.get("text", ""))
            try:
                decision, confidence = self._moderate_item(item_id, text, prompt_version)
                records.append(
                    {
                        "item_id": item_id,
                        "decision": decision,
                        "confidence": confidence,
                        "model_name": self.model_name,
                        "prompt_version": prompt_version,
                        "run_id": self.cfg.run_id,
                        "error": None,
                    }
                )
            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "Moderation failed for item_id=%s model=%s: %s",
                    item_id, self.model_name, exc,
                )
                records.append(
                    {
                        "item_id": item_id,
                        "decision": None,
                        "confidence": None,
                        "model_name": self.model_name,
                        "prompt_version": prompt_version,
                        "run_id": self.cfg.run_id,
                        "error": str(exc),
                    }
                )

        return pd.DataFrame(records, columns=_OUTPUT_COLUMNS)

    # ------------------------------------------------------------------
    # Abstract helpers
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Identifier stored in the ``predictions`` table."""

    @abstractmethod
    def _moderate_item(
        self,
        item_id: str,
        text: str,
        prompt_version: str,
    ) -> tuple[str, float]:
        """
        Moderate a single item.

        Returns
        -------
        (decision, confidence)
            decision ∈ {"toxic", "not_toxic"}
            confidence ∈ [0.0, 1.0]

        Raises
        ------
        Any exception — the caller (``moderate``) handles it.
        """

    # ------------------------------------------------------------------
    # Shared fixture helper
    # ------------------------------------------------------------------

    @property
    def _loaded_fixtures(self) -> dict[str, dict]:
        if self._fixtures is None:
            self._fixtures = _load_fixtures(self.cfg.fixture_path)
        return self._fixtures


# ---------------------------------------------------------------------------
# Rule-Based Moderator
# ---------------------------------------------------------------------------

class RuleBasedModerator(BaseModerator):
    """
    Simple keyword-list moderator — no API, always available.

    Decision rule:
      If any toxic keyword appears as a whole word in the lowercased text
      → decision="toxic", confidence=0.95
      Else
      → decision="not_toxic", confidence=0.05

    Prompt strategy is accepted for API consistency but has no effect on
    keyword matching (the same rule applies to both zero_shot and few_shot).
    """

    @property
    def model_name(self) -> str:
        return "rule_based"

    def _moderate_item(
        self,
        item_id: str,
        text: str,
        prompt_version: str,
    ) -> tuple[str, float]:
        lowered = text.lower()
        for keyword in _TOXIC_KEYWORDS:
            # Whole-word match to avoid false positives (e.g. "kill" in "skill")
            if re.search(r"\b" + re.escape(keyword) + r"\b", lowered):
                return "toxic", 0.95
        return "not_toxic", 0.05


# ---------------------------------------------------------------------------
# OpenAI Moderator
# ---------------------------------------------------------------------------

class OpenAIModerator(BaseModerator):
    """
    Calls GPT-4o-mini via the ``openai`` Python library.

    Falls back to fixture responses when:
      - ``cfg.is_fixture_mode()`` returns True, OR
      - the ``openai`` library is not installed.
    """

    @property
    def model_name(self) -> str:
        return self.cfg.openai_model  # default: "gpt-4o-mini"

    def _moderate_item(
        self,
        item_id: str,
        text: str,
        prompt_version: str,
    ) -> tuple[str, float]:
        if self.cfg.is_fixture_mode():
            return _get_fixture_response(self._loaded_fixtures, item_id, prompt_version)

        return self._call_openai(text, prompt_version)

    def _call_openai(self, text: str, prompt_version: str) -> tuple[str, float]:
        try:
            import openai  # noqa: PLC0415  (local import to keep dependency optional)
        except ImportError as exc:
            raise RuntimeError(
                "openai package is not installed. Install it or use fixture mode."
            ) from exc

        client = openai.OpenAI(api_key=self.cfg.openai_api_key)
        prompt = build_prompt(text, prompt_version)

        response = client.chat.completions.create(
            model=self.cfg.openai_model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=10,
            temperature=0,
        )

        raw = response.choices[0].message.content or ""
        decision, confidence = _parse_llm_response(raw)

        if decision is None:
            raise ValueError(
                f"OpenAI returned unrecognised response: {raw!r}"
            )

        return decision, confidence  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Gemini Moderator
# ---------------------------------------------------------------------------

class GeminiModerator(BaseModerator):
    """
    Calls Gemini 2.5 Flash via ``google.generativeai``.

    Falls back to fixture responses when:
      - ``cfg.is_fixture_mode()`` returns True, OR
      - the ``google-generativeai`` library is not installed.
    """

    @property
    def model_name(self) -> str:
        return self.cfg.gemini_model  # default: "gemini-2.5-flash"

    def _moderate_item(
        self,
        item_id: str,
        text: str,
        prompt_version: str,
    ) -> tuple[str, float]:
        if self.cfg.is_fixture_mode():
            return _get_fixture_response(self._loaded_fixtures, item_id, prompt_version)

        return self._call_gemini(text, prompt_version)

    def _call_gemini(self, text: str, prompt_version: str) -> tuple[str, float]:
        try:
            import google.generativeai as genai  # noqa: PLC0415
        except ImportError as exc:
            raise RuntimeError(
                "google-generativeai package is not installed. "
                "Install it or use fixture mode."
            ) from exc

        genai.configure(api_key=self.cfg.gemini_api_key)
        model = genai.GenerativeModel(model_name=self.cfg.gemini_model)

        prompt = build_prompt(text, prompt_version)
        response = model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                max_output_tokens=10,
                temperature=0,
            ),
        )

        raw = response.text if hasattr(response, "text") else ""
        decision, confidence = _parse_llm_response(raw)

        if decision is None:
            raise ValueError(
                f"Gemini returned unrecognised response: {raw!r}"
            )

        return decision, confidence  # type: ignore[return-value]
