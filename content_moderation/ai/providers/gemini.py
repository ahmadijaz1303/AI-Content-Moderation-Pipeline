"""Gemini adapter for the provider-neutral shared AI contracts.

This module is intentionally independent from the completed Part 2 pipeline.
It makes text-only calls for future shared AI features and never exposes Gemini
SDK response objects to callers.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Iterator, Mapping, Optional, Sequence, Tuple

from google import genai
from google.genai import types

from ..config import (
    FEATURE_CLASSIFICATION,
    FEATURE_MODERATION,
    FEATURE_STREAMING_TEXT_GENERATION,
    FEATURE_TEXT_GENERATION,
    FeatureAIConfiguration,
    SharedAIConfiguration,
    load_shared_ai_configuration,
)
from ..contracts import (
    ClassificationLabel,
    ClassificationRequest,
    ClassificationResult,
    ModerationFinding,
    ModerationRequest,
    ModerationResult,
    StreamEventType,
    StreamingTextEvent,
    TextGenerationRequest,
    TextGenerationResult,
)
from ..pii import redact_pii
from ..provider import AIProvider, AIProviderError


_PROVIDER_NAME = "gemini"
_DEFAULT_TIMEOUT_SECONDS = 30

_CLASSIFICATION_SCHEMA = {
    "type": "object",
    "properties": {
        "labels": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "label": {"type": "string"},
                    "confidence": {"type": "number"},
                    "selected": {"type": "boolean"},
                },
                "required": ["label", "confidence", "selected"],
            },
        },
        "rationale": {"type": "string"},
    },
    "required": ["labels"],
}

_MODERATION_SCHEMA = {
    "type": "object",
    "properties": {
        "flagged": {"type": "boolean"},
        "decision": {"type": "string"},
        "findings": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "category": {"type": "string"},
                    "confidence": {"type": "number"},
                    "severity": {"type": "string"},
                    "reason": {"type": "string"},
                },
                "required": ["category", "confidence", "severity"],
            },
        },
    },
    "required": ["flagged", "decision", "findings"],
}


class GeminiProvider(AIProvider):
    """Translate shared AI contracts into Gemini text requests safely."""

    def __init__(
        self,
        configuration: Optional[SharedAIConfiguration] = None,
        environment: Optional[Mapping[str, str]] = None,
        client: Optional[Any] = None,
    ) -> None:
        """Create a provider without making a network request.

        ``client`` exists for controlled dependency injection in tests.  Normal
        runtime construction is lazy and reads the configured credential only
        when a request is made.
        """

        self._environment = os.environ if environment is None else environment
        self._configuration = configuration or load_shared_ai_configuration(
            self._environment
        )
        self._injected_client = client
        self._clients: Dict[str, Any] = {}

    def generate_text(self, request: TextGenerationRequest) -> TextGenerationResult:
        """Generate text and return a stable, SDK-free result."""

        configuration = self._feature_configuration(FEATURE_TEXT_GENERATION)
        try:
            response = self._client_for(configuration).models.generate_content(
                model=self._model_for(request.model, configuration),
                contents=self._generation_contents(request),
                config=self._generation_config(request),
            )
            return TextGenerationResult(
                text=self._response_text(response),
                provider=_PROVIDER_NAME,
                model=self._model_for(request.model, configuration),
                request_id=self._response_identifier(response),
                usage=self._response_usage(response),
            )
        except AIProviderError:
            raise
        except Exception as error:
            raise AIProviderError("AI provider request failed.") from error

    def stream_text(
        self, request: TextGenerationRequest
    ) -> Iterator[StreamingTextEvent]:
        """Yield normalized streaming events without SDK streaming objects."""

        try:
            configuration = self._feature_configuration(
                FEATURE_STREAMING_TEXT_GENERATION
            )
            model = self._model_for(request.model, configuration)
        except Exception:
            yield self._stream_error(sequence=0)
            return

        yield StreamingTextEvent(
            event=StreamEventType.START,
            sequence=0,
            provider=_PROVIDER_NAME,
            model=model,
        )
        sequence = 1
        try:
            stream = self._client_for(configuration).models.generate_content_stream(
                model=model,
                contents=self._generation_contents(request),
                config=self._generation_config(request),
            )
            for response in stream:
                text = self._optional_response_text(response)
                if text:
                    yield StreamingTextEvent(
                        event=StreamEventType.DELTA,
                        sequence=sequence,
                        delta=text,
                        provider=_PROVIDER_NAME,
                        model=model,
                        request_id=self._response_identifier(response),
                    )
                    sequence += 1
        except Exception:
            yield self._stream_error(sequence=sequence, model=model)
            return

        yield StreamingTextEvent(
            event=StreamEventType.END,
            sequence=sequence,
            provider=_PROVIDER_NAME,
            model=model,
            finish_reason="completed",
        )

    def classify_text(self, request: ClassificationRequest) -> ClassificationResult:
        """Return validated structured classification using Gemini JSON output."""

        configuration = self._feature_configuration(FEATURE_CLASSIFICATION)
        try:
            response = self._client_for(configuration).models.generate_content(
                model=self._model_for(request.model, configuration),
                contents=self._classification_prompt(request),
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_json_schema=_CLASSIFICATION_SCHEMA,
                ),
            )
            payload = self._structured_payload(response)
            labels = self._classification_labels(payload, request.labels)
            rationale = payload.get("rationale")
            if rationale is not None and not isinstance(rationale, str):
                raise AIProviderError("AI provider returned invalid structured output.")
            return ClassificationResult(
                labels=labels,
                provider=_PROVIDER_NAME,
                model=self._model_for(request.model, configuration),
                rationale=rationale,
                request_id=self._response_identifier(response),
            )
        except AIProviderError:
            raise
        except Exception as error:
            raise AIProviderError("AI provider request failed.") from error

    def moderate_content(self, request: ModerationRequest) -> ModerationResult:
        """Return validated structured generic moderation output from Gemini."""

        configuration = self._feature_configuration(FEATURE_MODERATION)
        try:
            response = self._client_for(configuration).models.generate_content(
                model=self._model_for(request.model, configuration),
                contents=self._moderation_prompt(request),
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_json_schema=_MODERATION_SCHEMA,
                ),
            )
            payload = self._structured_payload(response)
            flagged = payload.get("flagged")
            decision = payload.get("decision")
            if not isinstance(flagged, bool) or not isinstance(decision, str):
                raise AIProviderError("AI provider returned invalid structured output.")
            findings = self._moderation_findings(payload, request.categories)
            return ModerationResult(
                flagged=flagged,
                findings=findings,
                provider=_PROVIDER_NAME,
                model=self._model_for(request.model, configuration),
                decision=decision,
                request_id=self._response_identifier(response),
            )
        except AIProviderError:
            raise
        except Exception as error:
            raise AIProviderError("AI provider request failed.") from error

    def _feature_configuration(self, feature: str) -> FeatureAIConfiguration:
        configuration = self._configuration.for_feature(feature)
        if configuration.provider != _PROVIDER_NAME:
            raise AIProviderError("Configured AI provider does not match this adapter.")
        return configuration

    def _client_for(self, configuration: FeatureAIConfiguration) -> Any:
        if self._injected_client is not None:
            return self._injected_client
        if configuration.api_key_env in self._clients:
            return self._clients[configuration.api_key_env]

        api_key = str(self._environment.get(configuration.api_key_env, "")).strip()
        if not api_key:
            raise AIProviderError("AI provider credentials are not configured.")
        timeout_seconds = int(
            self._environment.get("GEMINI_TIMEOUT_SECONDS", _DEFAULT_TIMEOUT_SECONDS)
        )
        if timeout_seconds <= 0:
            raise AIProviderError("Gemini timeout configuration is invalid.")
        client = genai.Client(
            api_key=api_key,
            # google-genai's timeout field is milliseconds; configuration uses
            # seconds so it remains readable and consistent with Part 2.
            http_options=types.HttpOptions(timeout=timeout_seconds * 1000),
        )
        self._clients[configuration.api_key_env] = client
        return client

    @staticmethod
    def _model_for(model_override: Optional[str], configuration: FeatureAIConfiguration) -> str:
        return model_override or configuration.model

    @staticmethod
    def _safe_text(text: Optional[str]) -> str:
        return redact_pii(text or "").sanitized_text

    def _generation_contents(self, request: TextGenerationRequest) -> Sequence[Dict[str, Any]]:
        contents = []
        for message in request.messages:
            role = "model" if message.role.casefold() == "assistant" else "user"
            contents.append(
                {"role": role, "parts": [{"text": self._safe_text(message.content)}]}
            )
        contents.append({"role": "user", "parts": [{"text": self._safe_text(request.prompt)}]})
        return contents

    def _generation_config(self, request: TextGenerationRequest) -> types.GenerateContentConfig:
        system_instruction = (
            self._safe_text(request.system_instruction)
            if request.system_instruction is not None
            else None
        )
        config_values = {
            "system_instruction": system_instruction,
            "temperature": request.temperature,
        }
        # Leave completion length to Gemini unless the caller intentionally
        # requests a cap for a specific feature.
        if request.max_output_tokens is not None:
            config_values["max_output_tokens"] = request.max_output_tokens
        return types.GenerateContentConfig(**config_values)

    def _classification_prompt(self, request: ClassificationRequest) -> str:
        instruction = self._safe_text(request.instruction)
        safe_text = self._safe_text(request.text)
        labels = ", ".join(request.labels)
        return (
            "Classify the supplied text using only the allowed labels. "
            f"Allowed labels: {labels}. Allow multiple labels: {request.allow_multiple}. "
            "Return the requested JSON structure only. "
            f"Instruction: {instruction or 'None'}. Text: {safe_text}"
        )

    def _moderation_prompt(self, request: ModerationRequest) -> str:
        instruction = self._safe_text(request.instruction)
        safe_content = self._safe_text(request.content)
        categories = ", ".join(request.categories)
        return (
            "Moderate the supplied text using only the allowed categories. "
            f"Allowed categories: {categories}. Return the requested JSON structure only. "
            f"Instruction: {instruction or 'None'}. Content: {safe_content}"
        )

    @staticmethod
    def _optional_response_text(response: Any) -> str:
        text = getattr(response, "text", "")
        if not isinstance(text, str) or not text.strip():
            return ""
        return text

    def _response_text(self, response: Any) -> str:
        text = self._optional_response_text(response)
        if not text:
            raise AIProviderError("AI provider returned an empty text response.")
        return text

    @staticmethod
    def _response_identifier(response: Any) -> Optional[str]:
        value = getattr(response, "response_id", None)
        return value.strip() if isinstance(value, str) and value.strip() else None

    @staticmethod
    def _response_usage(response: Any) -> Dict[str, int]:
        usage = getattr(response, "usage_metadata", None)
        values = {
            "prompt_tokens": getattr(usage, "prompt_token_count", None),
            "completion_tokens": getattr(usage, "candidates_token_count", None),
            "total_tokens": getattr(usage, "total_token_count", None),
        }
        return {name: int(value) for name, value in values.items() if value is not None}

    def _structured_payload(self, response: Any) -> Dict[str, Any]:
        payload = getattr(response, "parsed", None)
        if payload is None:
            try:
                payload = json.loads(self._response_text(response))
            except (TypeError, ValueError, json.JSONDecodeError) as error:
                raise AIProviderError("AI provider returned invalid structured output.") from error
        if not isinstance(payload, dict):
            raise AIProviderError("AI provider returned invalid structured output.")
        return payload

    def _classification_labels(
        self, payload: Dict[str, Any], allowed_labels: Tuple[str, ...]
    ) -> Tuple[ClassificationLabel, ...]:
        raw_labels = payload.get("labels")
        if not isinstance(raw_labels, list) or not raw_labels:
            raise AIProviderError("AI provider returned invalid structured output.")
        labels = []
        seen = set()
        for item in raw_labels:
            if not isinstance(item, dict):
                raise AIProviderError("AI provider returned invalid structured output.")
            label = item.get("label")
            confidence = item.get("confidence")
            selected = item.get("selected")
            if (
                not isinstance(label, str)
                or label not in allowed_labels
                or label in seen
                or not isinstance(selected, bool)
            ):
                raise AIProviderError("AI provider returned invalid structured output.")
            try:
                labels.append(
                    ClassificationLabel(
                        label=label,
                        confidence=confidence,
                        selected=selected,
                    )
                )
            except (TypeError, ValueError) as error:
                raise AIProviderError("AI provider returned invalid structured output.") from error
            seen.add(label)
        return tuple(labels)

    def _moderation_findings(
        self, payload: Dict[str, Any], allowed_categories: Tuple[str, ...]
    ) -> Tuple[ModerationFinding, ...]:
        raw_findings = payload.get("findings")
        if not isinstance(raw_findings, list):
            raise AIProviderError("AI provider returned invalid structured output.")
        findings = []
        for item in raw_findings:
            if not isinstance(item, dict) or item.get("category") not in allowed_categories:
                raise AIProviderError("AI provider returned invalid structured output.")
            try:
                findings.append(
                    ModerationFinding(
                        category=item["category"],
                        confidence=item["confidence"],
                        severity=item["severity"],
                        reason=item.get("reason"),
                    )
                )
            except (KeyError, TypeError, ValueError) as error:
                raise AIProviderError("AI provider returned invalid structured output.") from error
        return tuple(findings)

    @staticmethod
    def _stream_error(sequence: int, model: Optional[str] = None) -> StreamingTextEvent:
        return StreamingTextEvent(
            event=StreamEventType.ERROR,
            sequence=sequence,
            provider=_PROVIDER_NAME,
            model=model,
            error_code="provider_error",
        )


__all__ = ["GeminiProvider"]
