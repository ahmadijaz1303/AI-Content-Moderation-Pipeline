"""Environment-backed configuration for the future shared AI service.

This module intentionally configures feature selection only.  It does not
instantiate a provider, make network calls, or alter the existing moderation
pipeline configuration.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import os
import re
from typing import Dict, Mapping, Optional

from dotenv import load_dotenv


# Match the existing project pattern: configuration is read from environment
# variables and an optional local .env file.  No secret value is logged,
# serialized, or stored in these settings objects.
load_dotenv()


FEATURE_TEXT_GENERATION = "text_generation"
FEATURE_STREAMING_TEXT_GENERATION = "streaming_text_generation"
FEATURE_CLASSIFICATION = "classification"
FEATURE_MODERATION = "moderation"

SUPPORTED_AI_FEATURES = (
    FEATURE_TEXT_GENERATION,
    FEATURE_STREAMING_TEXT_GENERATION,
    FEATURE_CLASSIFICATION,
    FEATURE_MODERATION,
)


_FEATURE_ENV_PREFIXES = {
    FEATURE_TEXT_GENERATION: "AI_TEXT_GENERATION",
    FEATURE_STREAMING_TEXT_GENERATION: "AI_STREAMING_TEXT",
    FEATURE_CLASSIFICATION: "AI_CLASSIFICATION",
    # Shared AI moderation is intentionally distinct from the existing
    # TEXT_MODERATION_* variables used by the completed Part 2 pipeline.
    FEATURE_MODERATION: "AI_SHARED_MODERATION",
}

_DEFAULT_PROVIDER = "gemini"
_DEFAULT_MODEL = "models/gemini-flash-latest"
_DEFAULT_PROVIDER_API_KEY_ENV = {
    "gemini": "GEMINI_API_KEY",
    "groq": "GROQ_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
}
_ENVIRONMENT_VARIABLE_PATTERN = re.compile(r"^[A-Z_][A-Z0-9_]*$")


def _required_setting(value: Optional[str], name: str) -> str:
    """Normalize one required setting without including sensitive values."""

    normalized = str(value or "").strip()
    if not normalized:
        raise ValueError(f"{name} must not be empty.")
    return normalized


def _environment_value(
    environment: Mapping[str, str], name: str, default: Optional[str] = None
) -> Optional[str]:
    """Read one optional environment value.

    A missing setting receives its documented default.  An explicitly blank
    setting is returned to the contract validator so configuration mistakes
    fail during startup instead of silently selecting an unexpected model.
    """

    value = environment.get(name)
    if value is None:
        return default
    return str(value).strip()


@dataclass(frozen=True)
class FeatureAIConfiguration:
    """Provider and model selection for exactly one shared AI feature.

    ``api_key_env`` identifies the environment variable that a future provider
    adapter may read.  It deliberately does not contain the API key itself.
    """

    feature: str
    provider: str
    model: str
    api_key_env: str
    fallback_provider: Optional[str] = None
    fallback_model: Optional[str] = None
    fallback_api_key_env: Optional[str] = None
    fallback_2_provider: Optional[str] = None
    fallback_2_model: Optional[str] = None
    fallback_2_api_key_env: Optional[str] = None

    def __post_init__(self) -> None:
        if self.feature not in SUPPORTED_AI_FEATURES:
            raise ValueError(f"Unsupported AI feature: {self.feature}")
        object.__setattr__(self, "provider", _required_setting(self.provider, "provider").lower())
        object.__setattr__(self, "model", _required_setting(self.model, "model"))
        api_key_env = _required_setting(self.api_key_env, "api_key_env")
        if not _ENVIRONMENT_VARIABLE_PATTERN.fullmatch(api_key_env):
            raise ValueError("api_key_env must be a valid environment variable name.")
        object.__setattr__(self, "api_key_env", api_key_env)
        fallback_values = (
            self.fallback_provider, self.fallback_model, self.fallback_api_key_env,
        )
        if any(value is not None for value in fallback_values):
            if not all(str(value or "").strip() for value in fallback_values):
                raise ValueError("fallback provider, model, and api_key_env must be configured together.")
            fallback_provider = _required_setting(self.fallback_provider, "fallback_provider").lower()
            fallback_model = _required_setting(self.fallback_model, "fallback_model")
            fallback_api_key_env = _required_setting(self.fallback_api_key_env, "fallback_api_key_env")
            if not _ENVIRONMENT_VARIABLE_PATTERN.fullmatch(fallback_api_key_env):
                raise ValueError("fallback_api_key_env must be a valid environment variable name.")
            object.__setattr__(self, "fallback_provider", fallback_provider)
            object.__setattr__(self, "fallback_model", fallback_model)
            object.__setattr__(self, "fallback_api_key_env", fallback_api_key_env)
        fallback_2_values = (
            self.fallback_2_provider, self.fallback_2_model, self.fallback_2_api_key_env,
        )
        if any(value is not None for value in fallback_2_values):
            if not all(str(value or "").strip() for value in fallback_2_values):
                raise ValueError("fallback_2 provider, model, and api_key_env must be configured together.")
            fallback_2_provider = _required_setting(self.fallback_2_provider, "fallback_2_provider").lower()
            fallback_2_model = _required_setting(self.fallback_2_model, "fallback_2_model")
            fallback_2_api_key_env = _required_setting(self.fallback_2_api_key_env, "fallback_2_api_key_env")
            if not _ENVIRONMENT_VARIABLE_PATTERN.fullmatch(fallback_2_api_key_env):
                raise ValueError("fallback_2_api_key_env must be a valid environment variable name.")
            object.__setattr__(self, "fallback_2_provider", fallback_2_provider)
            object.__setattr__(self, "fallback_2_model", fallback_2_model)
            object.__setattr__(self, "fallback_2_api_key_env", fallback_2_api_key_env)

    def api_key_is_configured(self, environment: Optional[Mapping[str, str]] = None) -> bool:
        """Report credential presence without reading it into a response object."""

        source = os.environ if environment is None else environment
        return bool(str(source.get(self.api_key_env, "")).strip())

    def to_dict(self) -> Dict[str, str]:
        """Return safe, JSON-ready selection metadata without any API key."""

        return asdict(self)

    def fallback_configuration(self) -> Optional["FeatureAIConfiguration"]:
        """Return an independently configured fallback without exposing a key."""

        if self.fallback_provider is None:
            return None
        return FeatureAIConfiguration(
            feature=self.feature,
            provider=self.fallback_provider,
            model=str(self.fallback_model),
            api_key_env=str(self.fallback_api_key_env),
            fallback_provider=self.fallback_2_provider,
            fallback_model=self.fallback_2_model,
            fallback_api_key_env=self.fallback_2_api_key_env,
        )


@dataclass(frozen=True)
class SharedAIConfiguration:
    """Centralized provider/model selection for all shared AI features."""

    text_generation: FeatureAIConfiguration
    streaming_text_generation: FeatureAIConfiguration
    classification: FeatureAIConfiguration
    moderation: FeatureAIConfiguration

    def for_feature(self, feature: str) -> FeatureAIConfiguration:
        """Return one feature selection using the stable feature names."""

        mapping = {
            FEATURE_TEXT_GENERATION: self.text_generation,
            FEATURE_STREAMING_TEXT_GENERATION: self.streaming_text_generation,
            FEATURE_CLASSIFICATION: self.classification,
            FEATURE_MODERATION: self.moderation,
        }
        try:
            return mapping[feature]
        except KeyError as error:
            raise ValueError(f"Unsupported AI feature: {feature}") from error

    def to_dict(self) -> Dict[str, Dict[str, str]]:
        """Return safe feature selections; API keys are never included."""

        return {
            feature: self.for_feature(feature).to_dict()
            for feature in SUPPORTED_AI_FEATURES
        }


def load_feature_ai_configuration(
    feature: str, environment: Optional[Mapping[str, str]] = None
) -> FeatureAIConfiguration:
    """Load one feature's provider/model selection from environment variables.

    The four independent variable groups are:

    - ``AI_TEXT_GENERATION_*``
    - ``AI_STREAMING_TEXT_*``
    - ``AI_CLASSIFICATION_*``
    - ``AI_SHARED_MODERATION_*``

    An unknown provider must explicitly configure ``*_API_KEY_ENV``.  Known
    existing providers use their established credential environment names.
    """

    if feature not in SUPPORTED_AI_FEATURES:
        raise ValueError(f"Unsupported AI feature: {feature}")

    source = os.environ if environment is None else environment
    prefix = _FEATURE_ENV_PREFIXES[feature]
    provider = _required_setting(
        _environment_value(source, f"{prefix}_PROVIDER", _DEFAULT_PROVIDER),
        f"{prefix}_PROVIDER",
    ).lower()
    model = _required_setting(
        _environment_value(source, f"{prefix}_MODEL", _DEFAULT_MODEL),
        f"{prefix}_MODEL",
    )
    default_key_env = _DEFAULT_PROVIDER_API_KEY_ENV.get(provider)
    api_key_env = _environment_value(source, f"{prefix}_API_KEY_ENV", default_key_env)
    if api_key_env is None:
        raise ValueError(
            f"{prefix}_API_KEY_ENV must be configured for provider '{provider}'."
        )
    fallback_provider = _environment_value(source, f"{prefix}_FALLBACK_PROVIDER")
    fallback_model = _environment_value(source, f"{prefix}_FALLBACK_MODEL")
    fallback_api_key_env = _environment_value(source, f"{prefix}_FALLBACK_API_KEY_ENV")
    if fallback_provider:
        fallback_default_key_env = _DEFAULT_PROVIDER_API_KEY_ENV.get(fallback_provider.lower())
        fallback_api_key_env = _environment_value(
            source, f"{prefix}_FALLBACK_API_KEY_ENV", fallback_default_key_env,
        )
    fallback_2_provider = _environment_value(source, f"{prefix}_FALLBACK_2_PROVIDER")
    fallback_2_model = _environment_value(source, f"{prefix}_FALLBACK_2_MODEL")
    fallback_2_api_key_env = _environment_value(source, f"{prefix}_FALLBACK_2_API_KEY_ENV")
    if fallback_2_provider:
        fallback_2_default_key_env = _DEFAULT_PROVIDER_API_KEY_ENV.get(fallback_2_provider.lower())
        fallback_2_api_key_env = _environment_value(
            source, f"{prefix}_FALLBACK_2_API_KEY_ENV", fallback_2_default_key_env,
        )
    return FeatureAIConfiguration(
        feature=feature,
        provider=provider,
        model=model,
        api_key_env=api_key_env,
        fallback_provider=fallback_provider,
        fallback_model=fallback_model,
        fallback_api_key_env=fallback_api_key_env,
        fallback_2_provider=fallback_2_provider,
        fallback_2_model=fallback_2_model,
        fallback_2_api_key_env=fallback_2_api_key_env,
    )


def load_shared_ai_configuration(
    environment: Optional[Mapping[str, str]] = None
) -> SharedAIConfiguration:
    """Load all independent feature selections without creating providers."""

    return SharedAIConfiguration(
        text_generation=load_feature_ai_configuration(
            FEATURE_TEXT_GENERATION, environment
        ),
        streaming_text_generation=load_feature_ai_configuration(
            FEATURE_STREAMING_TEXT_GENERATION, environment
        ),
        classification=load_feature_ai_configuration(
            FEATURE_CLASSIFICATION, environment
        ),
        moderation=load_feature_ai_configuration(FEATURE_MODERATION, environment),
    )


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
]
