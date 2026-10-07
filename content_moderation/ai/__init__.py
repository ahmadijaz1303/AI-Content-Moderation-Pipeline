"""Provider-neutral contracts for shared AI features.

Providers and API routes intentionally do not live here yet.  This module is
the stable boundary that future AI helpers, including Course Helper, can use.
"""

from .config import (
    FEATURE_CLASSIFICATION,
    FEATURE_MODERATION,
    FEATURE_STREAMING_TEXT_GENERATION,
    FEATURE_TEXT_GENERATION,
    SUPPORTED_AI_FEATURES,
    FeatureAIConfiguration,
    SharedAIConfiguration,
    load_feature_ai_configuration,
    load_shared_ai_configuration,
)
from .contracts import (
    ClassificationLabel,
    ClassificationRequest,
    ClassificationResult,
    GenerationMessage,
    ModerationFinding,
    ModerationRequest,
    ModerationResult,
    StreamEventType,
    StreamingTextEvent,
    TextGenerationRequest,
    TextGenerationResult,
)
from .pii import PIICategory, PIIRedactionResult, redact_pii
from .provider import AIProvider, AIProviderError
from .service import AIService

__all__ = [
    "FEATURE_CLASSIFICATION",
    "FEATURE_MODERATION",
    "FEATURE_STREAMING_TEXT_GENERATION",
    "FEATURE_TEXT_GENERATION",
    "SUPPORTED_AI_FEATURES",
    "FeatureAIConfiguration",
    "SharedAIConfiguration",
    "load_feature_ai_configuration",
    "load_shared_ai_configuration",
    "ClassificationLabel",
    "ClassificationRequest",
    "ClassificationResult",
    "GenerationMessage",
    "ModerationFinding",
    "ModerationRequest",
    "ModerationResult",
    "StreamEventType",
    "StreamingTextEvent",
    "TextGenerationRequest",
    "TextGenerationResult",
    "PIICategory",
    "PIIRedactionResult",
    "redact_pii",
    "AIProvider",
    "AIProviderError",
    "AIService",
]
