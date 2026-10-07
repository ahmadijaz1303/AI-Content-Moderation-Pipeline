"""Provider-neutral protection for text sent to external AI providers.

The functions in this module deliberately return only redacted text and safe
summary metadata.  They do not log, retain, or return the sensitive matches.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import re
from typing import Dict, Tuple


class PIICategory(str, Enum):
    """Personally identifiable information categories handled locally."""

    PHONE_NUMBER = "phone_number"
    DATE_OF_BIRTH = "date_of_birth"
    HOME_ADDRESS = "home_address"


@dataclass(frozen=True)
class PIIRedactionResult:
    """Safe output from PII redaction, with no original matches retained."""

    sanitized_text: str
    redacted_categories: Tuple[PIICategory, ...]
    redaction_counts: Dict[str, int]

    @property
    def redacted(self) -> bool:
        """Whether at least one PII value was removed."""

        return bool(self.redacted_categories)

    def to_dict(self) -> Dict[str, object]:
        """Return JSON-safe metadata without sensitive source values."""

        return {
            "sanitized_text": self.sanitized_text,
            "redacted": self.redacted,
            "redacted_categories": [category.value for category in self.redacted_categories],
            "redaction_counts": dict(self.redaction_counts),
        }


# Dates are redacted only when explicit birth-date context is present.  This
# avoids removing ordinary dates such as assignment deadlines or event dates.
_DATE_VALUE = r"(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}[/-]\d{1,2}[/-]\d{1,2}|(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\s+\d{1,2},?\s+\d{4}|\d{1,2}\s+(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\s+\d{4})"
_DATE_OF_BIRTH_PATTERN = re.compile(
    rf"\b(?:date\s+of\s+birth|birth\s*date|dob|born\s+on)\s*(?:is\s*)?[:,-]?\s*{_DATE_VALUE}\b",
    re.IGNORECASE,
)

# A street number plus a recognized street suffix is a reasonably strong home
# address signal.  The narrow suffix list keeps normal prose intact.
_HOME_ADDRESS_PATTERN = re.compile(
    r"\b\d{1,6}\s+(?:[A-Za-z0-9.'-]+\s+){0,5}"
    r"(?:street|st\.?|road|rd\.?|avenue|ave\.?|boulevard|blvd\.?|lane|ln\.?|"
    r"drive|dr\.?|court|ct\.?|way|place|pl\.?|parkway|pkwy\.?|terrace|ter\.?|"
    r"highway|hwy\.?)"
    r"(?:\s*,?\s*(?:apt\.?|apartment|unit|suite|#)\s*[A-Za-z0-9-]+)?\b",
    re.IGNORECASE,
)

# A candidate must contain 10–15 digits before it is treated as a phone number.
# This supports common local/international formats while avoiding short dates
# and ordinary numeric references.
_PHONE_CANDIDATE_PATTERN = re.compile(
    r"(?<!\w)(?:\+?\d{1,3}[\s.-]?)?(?:\(?\d{2,4}\)?[\s.-]?)?"
    r"\d{3,4}(?:[\s.-]?\d{3,4}){1,3}(?!\w)"
)

_PLACEHOLDERS = {
    PIICategory.PHONE_NUMBER: "[REDACTED_PHONE_NUMBER]",
    PIICategory.DATE_OF_BIRTH: "[REDACTED_DATE_OF_BIRTH]",
    PIICategory.HOME_ADDRESS: "[REDACTED_HOME_ADDRESS]",
}


def _replace_pattern(
    text: str, pattern: re.Pattern, category: PIICategory, counts: Dict[str, int]
) -> str:
    """Replace matching PII without preserving its source value."""

    def replacement(_: re.Match) -> str:
        counts[category.value] = counts.get(category.value, 0) + 1
        return _PLACEHOLDERS[category]

    return pattern.sub(replacement, text)


def _replace_phone_numbers(text: str, counts: Dict[str, int]) -> str:
    """Redact only phone candidates with a plausible digit count."""

    def replacement(match: re.Match) -> str:
        digit_count = len(re.sub(r"\D", "", match.group(0)))
        if not 10 <= digit_count <= 15:
            return match.group(0)
        counts[PIICategory.PHONE_NUMBER.value] = (
            counts.get(PIICategory.PHONE_NUMBER.value, 0) + 1
        )
        return _PLACEHOLDERS[PIICategory.PHONE_NUMBER]

    return _PHONE_CANDIDATE_PATTERN.sub(replacement, text)


def redact_pii(text: str) -> PIIRedactionResult:
    """Return provider-safe text with known PII types replaced by placeholders.

    The caller retains responsibility for passing only ``sanitized_text`` to an
    external provider.  This function makes no network calls and emits no logs.
    """

    if not isinstance(text, str):
        raise TypeError("text must be a string.")

    counts: Dict[str, int] = {}
    sanitized_text = _replace_pattern(
        text, _DATE_OF_BIRTH_PATTERN, PIICategory.DATE_OF_BIRTH, counts
    )
    sanitized_text = _replace_pattern(
        sanitized_text, _HOME_ADDRESS_PATTERN, PIICategory.HOME_ADDRESS, counts
    )
    sanitized_text = _replace_phone_numbers(sanitized_text, counts)
    categories = tuple(
        category for category in PIICategory if counts.get(category.value, 0)
    )
    return PIIRedactionResult(
        sanitized_text=sanitized_text,
        redacted_categories=categories,
        redaction_counts=dict(counts),
    )


__all__ = ["PIICategory", "PIIRedactionResult", "redact_pii"]
