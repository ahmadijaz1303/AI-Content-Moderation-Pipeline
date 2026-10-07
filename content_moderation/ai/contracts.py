"""Stable, provider-neutral contracts for the shared AI service layer.

These contracts deliberately use standard-library dataclasses.  The current
project does not use a Pydantic schema layer outside FastAPI, and keeping this
boundary dependency-free allows providers and future features to share it.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, Mapping, Optional, Tuple


def _required_text(value: str, field_name: str) -> str:
    """Return non-empty text or raise a clear contract validation error."""

    normalized = str(value or "").strip()
    if not normalized:
        raise ValueError(f"{field_name} must not be empty.")
    return normalized


def _optional_model(value: Optional[str]) -> Optional[str]:
    """Normalize an optional feature-specific model override."""

    if value is None:
        return None
    return _required_text(value, "model")


def _normalized_labels(labels: Tuple[str, ...]) -> Tuple[str, ...]:
    """Require unique, non-empty labels while preserving their input order."""

    normalized = tuple(_required_text(label, "label") for label in labels)
    if not normalized:
        raise ValueError("labels must contain at least one label.")
    if len(set(normalized)) != len(normalized):
        raise ValueError("labels must not contain duplicates.")
    return normalized


def _confidence(value: float, field_name: str = "confidence") -> float:
    """Validate a normalized confidence score."""

    score = float(value)
    if not 0.0 <= score <= 1.0:
        raise ValueError(f"{field_name} must be between 0.0 and 1.0.")
    return score


class StreamEventType(str, Enum):
    """Provider-neutral lifecycle events for streamed text generation."""

    START = "start"
    DELTA = "delta"
    END = "end"
    ERROR = "error"


@dataclass(frozen=True)
class GenerationMessage:
    """One provider-neutral conversation message."""

    role: str
    content: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "role", _required_text(self.role, "role"))
        object.__setattr__(self, "content", _required_text(self.content, "content"))


@dataclass(frozen=True)
class TextGenerationRequest:
    """Request for regular or Course Helper text generation.

    ``model`` is an optional feature-level override.  Provider selection and
    the configured default model belong to the future shared AI service.
    """

    prompt: str
    system_instruction: Optional[str] = None
    messages: Tuple[GenerationMessage, ...] = ()
    model: Optional[str] = None
    temperature: float = 0.2
    # ``None`` means "do not impose an application-level limit".  The chosen
    # provider/model still enforces its own maximum completion length.
    max_output_tokens: Optional[int] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "prompt", _required_text(self.prompt, "prompt"))
        if self.system_instruction is not None:
            object.__setattr__(
                self,
                "system_instruction",
                _required_text(self.system_instruction, "system_instruction"),
            )
        object.__setattr__(self, "messages", tuple(self.messages))
        if not all(isinstance(message, GenerationMessage) for message in self.messages):
            raise TypeError("messages must contain GenerationMessage values.")
        object.__setattr__(self, "model", _optional_model(self.model))
        if not 0.0 <= float(self.temperature) <= 2.0:
            raise ValueError("temperature must be between 0.0 and 2.0.")
        if self.max_output_tokens is not None and int(self.max_output_tokens) <= 0:
            raise ValueError("max_output_tokens must be greater than zero when set.")
        object.__setattr__(self, "temperature", float(self.temperature))
        if self.max_output_tokens is not None:
            object.__setattr__(self, "max_output_tokens", int(self.max_output_tokens))
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True)
class TextGenerationResult:
    """Stable completed text-generation result without provider SDK objects."""

    text: str
    provider: str
    model: str
    finish_reason: str = "completed"
    request_id: Optional[str] = None
    usage: Mapping[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "text", _required_text(self.text, "text"))
        object.__setattr__(self, "provider", _required_text(self.provider, "provider"))
        object.__setattr__(self, "model", _required_text(self.model, "model"))
        object.__setattr__(
            self, "finish_reason", _required_text(self.finish_reason, "finish_reason")
        )
        if self.request_id is not None:
            object.__setattr__(self, "request_id", _required_text(self.request_id, "request_id"))
        object.__setattr__(self, "usage", {str(key): int(value) for key, value in self.usage.items()})

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable result for a future API layer."""

        return asdict(self)


@dataclass(frozen=True)
class ClassificationRequest:
    """Request for structured text classification against caller-supplied labels."""

    text: str
    labels: Tuple[str, ...]
    instruction: Optional[str] = None
    model: Optional[str] = None
    allow_multiple: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "text", _required_text(self.text, "text"))
        object.__setattr__(self, "labels", _normalized_labels(tuple(self.labels)))
        if self.instruction is not None:
            object.__setattr__(self, "instruction", _required_text(self.instruction, "instruction"))
        object.__setattr__(self, "model", _optional_model(self.model))
        object.__setattr__(self, "allow_multiple", bool(self.allow_multiple))
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True)
class ClassificationLabel:
    """One scored label in a provider-independent classification response."""

    label: str
    confidence: float
    selected: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "label", _required_text(self.label, "label"))
        object.__setattr__(self, "confidence", _confidence(self.confidence))
        object.__setattr__(self, "selected", bool(self.selected))


@dataclass(frozen=True)
class ClassificationResult:
    """Structured classification result suitable for JSON APIs and Course Helper."""

    labels: Tuple[ClassificationLabel, ...]
    provider: str
    model: str
    rationale: Optional[str] = None
    request_id: Optional[str] = None

    def __post_init__(self) -> None:
        normalized = tuple(self.labels)
        if not normalized:
            raise ValueError("labels must contain at least one ClassificationLabel.")
        if not all(isinstance(item, ClassificationLabel) for item in normalized):
            raise TypeError("labels must contain ClassificationLabel values.")
        if len({item.label for item in normalized}) != len(normalized):
            raise ValueError("classification labels must not contain duplicates.")
        object.__setattr__(self, "labels", normalized)
        object.__setattr__(self, "provider", _required_text(self.provider, "provider"))
        object.__setattr__(self, "model", _required_text(self.model, "model"))
        if self.rationale is not None:
            object.__setattr__(self, "rationale", _required_text(self.rationale, "rationale"))
        if self.request_id is not None:
            object.__setattr__(self, "request_id", _required_text(self.request_id, "request_id"))

    @property
    def selected_labels(self) -> Tuple[str, ...]:
        """Return selected labels in their stable provider-result order."""

        return tuple(item.label for item in self.labels if item.selected)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ModerationRequest:
    """Provider-neutral request for an external moderation-capable model.

    This is intentionally separate from Part 2's complete detector report.
    It represents only the content and policy categories supplied to a shared
    AI provider; the existing local Risk Engine and policy layer remain intact.
    """

    content: str
    categories: Tuple[str, ...]
    instruction: Optional[str] = None
    model: Optional[str] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "content", _required_text(self.content, "content"))
        object.__setattr__(self, "categories", _normalized_labels(tuple(self.categories)))
        if self.instruction is not None:
            object.__setattr__(self, "instruction", _required_text(self.instruction, "instruction"))
        object.__setattr__(self, "model", _optional_model(self.model))
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True)
class ModerationFinding:
    """One provider-neutral moderation category finding."""

    category: str
    confidence: float
    severity: str = "none"
    reason: Optional[str] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "category", _required_text(self.category, "category"))
        object.__setattr__(self, "confidence", _confidence(self.confidence))
        object.__setattr__(self, "severity", _required_text(self.severity, "severity"))
        if self.reason is not None:
            object.__setattr__(self, "reason", _required_text(self.reason, "reason"))


@dataclass(frozen=True)
class ModerationResult:
    """Stable moderation decision produced by a future shared AI provider."""

    flagged: bool
    findings: Tuple[ModerationFinding, ...]
    provider: str
    model: str
    decision: str = "APPROVED"
    request_id: Optional[str] = None

    def __post_init__(self) -> None:
        normalized = tuple(self.findings)
        if not all(isinstance(item, ModerationFinding) for item in normalized):
            raise TypeError("findings must contain ModerationFinding values.")
        object.__setattr__(self, "findings", normalized)
        object.__setattr__(self, "flagged", bool(self.flagged))
        object.__setattr__(self, "provider", _required_text(self.provider, "provider"))
        object.__setattr__(self, "model", _required_text(self.model, "model"))
        object.__setattr__(self, "decision", _required_text(self.decision, "decision"))
        if self.request_id is not None:
            object.__setattr__(self, "request_id", _required_text(self.request_id, "request_id"))

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class StreamingTextEvent:
    """One event emitted while a text-generation request is streamed."""

    event: StreamEventType
    sequence: int
    delta: str = ""
    provider: Optional[str] = None
    model: Optional[str] = None
    request_id: Optional[str] = None
    finish_reason: Optional[str] = None
    error_code: Optional[str] = None

    def __post_init__(self) -> None:
        if not isinstance(self.event, StreamEventType):
            object.__setattr__(self, "event", StreamEventType(self.event))
        if int(self.sequence) < 0:
            raise ValueError("sequence must be zero or greater.")
        object.__setattr__(self, "sequence", int(self.sequence))
        object.__setattr__(self, "delta", str(self.delta or ""))
        for field_name in (
            "provider", "model", "request_id", "finish_reason", "error_code",
        ):
            value = getattr(self, field_name)
            if value is not None:
                object.__setattr__(self, field_name, _required_text(value, field_name))
        if self.event is StreamEventType.DELTA and not self.delta:
            raise ValueError("delta events must contain text.")
        if self.event is StreamEventType.ERROR and not self.error_code:
            raise ValueError("error events must include error_code.")

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["event"] = self.event.value
        return payload
