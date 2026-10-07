"""Provider-neutral interface for the future shared AI service.

Concrete adapters will be implemented later.  This module deliberately has
no SDK, credential, or network dependency.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterator

from .contracts import (
    ClassificationRequest,
    ClassificationResult,
    ModerationRequest,
    ModerationResult,
    StreamingTextEvent,
    TextGenerationRequest,
    TextGenerationResult,
)


class AIProviderError(RuntimeError):
    """Safe error raised when an AI provider cannot complete a request."""


class AIProvider(ABC):
    """Common capability contract for any external or local AI provider.

    Provider implementations are responsible for translating these stable
    contracts to their SDK/API formats.  Callers remain provider-neutral.
    """

    @abstractmethod
    def generate_text(self, request: TextGenerationRequest) -> TextGenerationResult:
        """Generate one completed text response."""

        raise NotImplementedError

    @abstractmethod
    def stream_text(self, request: TextGenerationRequest) -> Iterator[StreamingTextEvent]:
        """Yield provider-neutral events for one text-generation stream."""

        raise NotImplementedError

    @abstractmethod
    def classify_text(self, request: ClassificationRequest) -> ClassificationResult:
        """Classify text against the labels supplied in ``request``."""

        raise NotImplementedError

    @abstractmethod
    def moderate_content(self, request: ModerationRequest) -> ModerationResult:
        """Return a provider-neutral content-moderation result."""

        raise NotImplementedError


__all__ = ["AIProvider", "AIProviderError"]
