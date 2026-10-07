"""Provider-agnostic shared AI service for future application features."""

from __future__ import annotations

from dataclasses import replace
from typing import Callable, Dict, Iterator, Mapping, Optional

from .config import (
    FEATURE_CLASSIFICATION,
    FEATURE_MODERATION,
    FEATURE_STREAMING_TEXT_GENERATION,
    FEATURE_TEXT_GENERATION,
    FeatureAIConfiguration,
    SharedAIConfiguration,
    load_shared_ai_configuration,
)
from .contracts import (
    ClassificationRequest,
    ClassificationResult,
    ModerationRequest,
    ModerationResult,
    StreamEventType,
    StreamingTextEvent,
    TextGenerationRequest,
    TextGenerationResult,
)
from .provider import AIProvider, AIProviderError
from .providers import GeminiProvider, GroqProvider, OpenRouterProvider


ProviderFactory = Callable[[SharedAIConfiguration], AIProvider]


class AIService:
    """Select and delegate to a configured provider for each AI capability.

    This service deliberately has no provider SDK calls, HTTP payload code, or
    credential handling.  Those concerns remain inside concrete adapters.
    """

    def __init__(
        self,
        configuration: Optional[SharedAIConfiguration] = None,
        provider_factories: Optional[Mapping[str, ProviderFactory]] = None,
    ) -> None:
        self._configuration = configuration or load_shared_ai_configuration()
        self._provider_factories: Dict[str, ProviderFactory] = {
            "gemini": lambda config: GeminiProvider(configuration=config),
            "groq": lambda config: GroqProvider(configuration=config),
            "openrouter": lambda config: OpenRouterProvider(configuration=config),
        }
        if provider_factories:
            self._provider_factories.update(
                {
                    str(name).strip().lower(): factory
                    for name, factory in provider_factories.items()
                }
            )
        self._providers: Dict[str, AIProvider] = {}

    def generate_text(self, request: TextGenerationRequest) -> TextGenerationResult:
        """Delegate text generation to its configured provider."""

        configuration = self._feature_configuration(FEATURE_TEXT_GENERATION)
        last_error: Optional[AIProviderError] = None
        while configuration is not None:
            try:
                return self._provider_for_configuration(
                    FEATURE_TEXT_GENERATION, configuration,
                ).generate_text(request)
            except AIProviderError as error:
                last_error = error
                configuration = configuration.fallback_configuration()
            except Exception:
                last_error = AIProviderError("AI provider request failed.")
                configuration = configuration.fallback_configuration()
        if last_error is not None:
            raise last_error
        raise AIProviderError("AI provider request failed.")

    def stream_text(self, request: TextGenerationRequest) -> Iterator[StreamingTextEvent]:
        """Return a lazy, provider-neutral text-event iterator."""

        configuration = self._feature_configuration(FEATURE_STREAMING_TEXT_GENERATION)
        provider = self._provider_for(FEATURE_STREAMING_TEXT_GENERATION)
        return self._safe_stream(provider, request, configuration)

    def classify_text(self, request: ClassificationRequest) -> ClassificationResult:
        """Delegate classification to its configured provider."""

        provider = self._provider_for(FEATURE_CLASSIFICATION)
        try:
            return provider.classify_text(request)
        except AIProviderError:
            raise
        except Exception:
            raise AIProviderError("AI provider request failed.") from None

    def moderate_content(self, request: ModerationRequest) -> ModerationResult:
        """Delegate generic shared-AI moderation to its configured provider."""

        provider = self._provider_for(FEATURE_MODERATION)
        try:
            return provider.moderate_content(request)
        except AIProviderError:
            raise
        except Exception:
            raise AIProviderError("AI provider request failed.") from None

    def _feature_configuration(self, feature: str) -> FeatureAIConfiguration:
        try:
            return self._configuration.for_feature(feature)
        except (KeyError, ValueError):
            raise AIProviderError("AI provider configuration is invalid.") from None

    def _provider_for(self, feature: str) -> AIProvider:
        configuration = self._feature_configuration(feature)
        return self._provider_for_configuration(feature, configuration)

    def _provider_for_configuration(
        self, feature: str, configuration: FeatureAIConfiguration,
    ) -> AIProvider:
        provider_name = configuration.provider
        provider_key = "|".join((feature, provider_name, configuration.model, configuration.api_key_env))
        if provider_key in self._providers:
            return self._providers[provider_key]

        factory = self._provider_factories.get(provider_name)
        if factory is None:
            raise AIProviderError("Configured AI provider is not supported.")
        try:
            provider = factory(self._configuration_with_feature(feature, configuration))
        except AIProviderError:
            raise
        except Exception:
            raise AIProviderError("AI provider initialization failed.") from None
        if not isinstance(provider, AIProvider):
            raise AIProviderError("AI provider initialization failed.")
        self._providers[provider_key] = provider
        return provider

    def _configuration_with_feature(
        self, feature: str, configuration: FeatureAIConfiguration,
    ) -> SharedAIConfiguration:
        attribute = {
            FEATURE_TEXT_GENERATION: "text_generation",
            FEATURE_STREAMING_TEXT_GENERATION: "streaming_text_generation",
            FEATURE_CLASSIFICATION: "classification",
            FEATURE_MODERATION: "moderation",
        }[feature]
        return replace(self._configuration, **{attribute: configuration})

    def _safe_stream(
        self,
        provider: AIProvider,
        request: TextGenerationRequest,
        configuration: FeatureAIConfiguration,
    ) -> Iterator[StreamingTextEvent]:
        """Preserve lazy streaming while converting unexpected errors safely."""

        sequence = 0
        try:
            for event in provider.stream_text(request):
                if not isinstance(event, StreamingTextEvent):
                    raise TypeError("Provider stream yielded an invalid event.")
                sequence = event.sequence + 1
                yield event
        except Exception:
            yield StreamingTextEvent(
                event=StreamEventType.ERROR,
                sequence=sequence,
                provider=configuration.provider,
                model=configuration.model,
                error_code="provider_error",
            )


__all__ = ["AIService", "ProviderFactory"]
