"""Optional, non-decisive Part 1 context evidence for the Part 2 pipeline.

This module deliberately contains no detector, provider, risk, VLM, or policy
logic.  It is the single boundary where bounded Part 2 OCR/STT text can be
redacted and sent to the provider-agnostic :class:`AIService`.
"""

from __future__ import annotations

import os
from typing import Any, Callable, Dict, Mapping, Optional

from .contracts import ClassificationRequest
from .pii import redact_pii
from .provider import AIProviderError
from .service import AIService


PART2_AI_CONTEXT_ENABLED_ENV = "PART2_AI_CONTEXT_INTEGRATION_ENABLED"
PART2_AI_CONTEXT_MAX_TEXT_CHARS_ENV = "PART2_AI_CONTEXT_MAX_TEXT_CHARS"
DEFAULT_PART2_AI_CONTEXT_MAX_TEXT_CHARS = 6000

# These labels are application-owned, intentionally fixed, and aligned with
# the existing Part 2 policy categories.  They are supplemental context only.
PART2_MODERATION_CONTEXT_LABELS = (
    "SAFE_CONTEXT",
    "FICTION_OR_NEWS_CONTEXT",
    "HARASSMENT_OR_HATE",
    "CREDIBLE_THREAT_OR_ILLEGAL_ACTIVITY",
    "SEXUAL_CONTENT",
    "SELF_HARM_OR_CHILD_SAFETY",
)

_CONTEXT_INSTRUCTION = (
    "Classify the supplied redacted OCR/transcript evidence for moderation "
    "context. This is supplemental evidence only. Select only applicable "
    "labels; do not make an approval or enforcement decision."
)
_TRUE_VALUES = {"1", "true", "yes", "on"}


def part2_context_integration_enabled(
    environment: Optional[Mapping[str, str]] = None,
) -> bool:
    """Return the explicit opt-in flag; an absent or invalid value is off."""

    source = os.environ if environment is None else environment
    value = str(source.get(PART2_AI_CONTEXT_ENABLED_ENV, "0")).strip().lower()
    return value in _TRUE_VALUES


def _maximum_text_characters(environment: Mapping[str, str]) -> int:
    """Read a bounded text size without allowing a bad setting to escape."""

    configured = environment.get(PART2_AI_CONTEXT_MAX_TEXT_CHARS_ENV)
    if configured is None or not str(configured).strip():
        return DEFAULT_PART2_AI_CONTEXT_MAX_TEXT_CHARS
    try:
        value = int(configured)
    except (TypeError, ValueError):
        return DEFAULT_PART2_AI_CONTEXT_MAX_TEXT_CHARS
    return value if value > 0 else DEFAULT_PART2_AI_CONTEXT_MAX_TEXT_CHARS


def _safe_redaction_metadata(redaction: Any) -> Dict[str, Any]:
    """Expose only redaction counts/categories, never text or matched values."""

    return {
        "redacted": bool(redaction.redacted),
        "categories": [category.value for category in redaction.redacted_categories],
        "counts": dict(redaction.redaction_counts),
    }


def unavailable_part2_context_result(
    reason: str,
    redaction: Optional[Any] = None,
) -> Dict[str, Any]:
    """Return a stable, non-sensitive unavailable result for the report."""

    return {
        "available": False,
        "classification": None,
        "reason": reason,
        "redaction": (
            _safe_redaction_metadata(redaction)
            if redaction is not None
            else {"redacted": False, "categories": [], "counts": {}}
        ),
    }


def classify_part2_moderation_context(
    merged_text: str,
    *,
    service_factory: Callable[[], AIService] = AIService,
    environment: Optional[Mapping[str, str]] = None,
) -> Dict[str, Any]:
    """Return optional PII-safe Part 1 classification evidence.

    Failures are intentionally isolated here.  The result is report evidence
    only; callers must not use it to modify Part 2 risk, VLM routing, policy,
    detector outputs, or the final moderation decision.
    """

    source = os.environ if environment is None else environment
    if not part2_context_integration_enabled(source):
        return unavailable_part2_context_result("integration_disabled")

    try:
        # Redact before truncating so a value cut at the character boundary is
        # never partially passed to an external provider.
        redaction = redact_pii(str(merged_text or ""))
        sanitized_text = redaction.sanitized_text[:_maximum_text_characters(source)]
        if not sanitized_text.strip():
            return unavailable_part2_context_result("no_text_available", redaction)

        result = service_factory().classify_text(
            ClassificationRequest(
                text=sanitized_text,
                labels=PART2_MODERATION_CONTEXT_LABELS,
                instruction=_CONTEXT_INSTRUCTION,
                allow_multiple=True,
                metadata={"source": "part2_optional_context"},
            )
        )
        return {
            "available": True,
            "classification": result.to_dict(),
            "reason": None,
            "redaction": _safe_redaction_metadata(redaction),
        }
    except (AIProviderError, TypeError, ValueError):
        return unavailable_part2_context_result("analysis_unavailable")
    except Exception:
        # The Part 2 request must continue even if provider initialization or
        # a future integration dependency fails unexpectedly.
        return unavailable_part2_context_result("analysis_unavailable")


__all__ = [
    "DEFAULT_PART2_AI_CONTEXT_MAX_TEXT_CHARS",
    "PART2_AI_CONTEXT_ENABLED_ENV",
    "PART2_AI_CONTEXT_MAX_TEXT_CHARS_ENV",
    "PART2_MODERATION_CONTEXT_LABELS",
    "classify_part2_moderation_context",
    "part2_context_integration_enabled",
    "unavailable_part2_context_result",
]
