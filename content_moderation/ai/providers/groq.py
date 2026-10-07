"""Groq adapter for the provider-neutral shared AI contracts.

This adapter is intentionally independent from the completed Part 2
moderation pipeline.  It redacts text before every outbound provider call and
does not expose Groq SDK objects beyond this module.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Iterator, Mapping, Optional, Sequence, Tuple

from groq import Groq

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


_PROVIDER_NAME = "groq"
_DEFAULT_TIMEOUT_SECONDS = 30


class GroqProvider(AIProvider):
    """Translate shared AI contracts into official Groq SDK calls."""

    def __init__(
        self,
        configuration: Optional[SharedAIConfiguration] = None,
        environment: Optional[Mapping[str, str]] = None,
        client: Optional[Any] = None,
        timeout_seconds: Optional[int] = None,
    ) -> None:
        self._environment = os.environ if environment is None else environment
        self._configuration = configuration or load_shared_ai_configuration(self._environment)
        self._client_override = client
        configured_timeout = self._environment.get("GROQ_TIMEOUT_SECONDS", _DEFAULT_TIMEOUT_SECONDS)
        self._timeout_seconds = int(timeout_seconds if timeout_seconds is not None else configured_timeout)
        self._clients: Dict[str, Any] = {}

    def generate_text(self, request: TextGenerationRequest) -> TextGenerationResult:
        configuration = self._feature_configuration(FEATURE_TEXT_GENERATION)
        model = request.model or configuration.model
        try:
            response = self._client_for(configuration).chat.completions.create(
                model=model,
                messages=self._generation_messages(request),
                temperature=request.temperature,
                max_completion_tokens=request.max_output_tokens,
            )
            text = self._content(response)
            return TextGenerationResult(
                text=text,
                provider=_PROVIDER_NAME,
                model=model,
                request_id=self._identifier(response),
                usage=self._usage(response),
            )
        except AIProviderError:
            raise
        except Exception:
            raise AIProviderError("AI provider request failed.") from None

    def stream_text(self, request: TextGenerationRequest) -> Iterator[StreamingTextEvent]:
        try:
            configuration = self._feature_configuration(FEATURE_STREAMING_TEXT_GENERATION)
            model = request.model or configuration.model
        except Exception:
            yield self._stream_error(0)
            return
        yield StreamingTextEvent(StreamEventType.START, 0, provider=_PROVIDER_NAME, model=model)
        sequence = 1
        try:
            stream = self._client_for(configuration).chat.completions.create(
                model=model,
                messages=self._generation_messages(request),
                temperature=request.temperature,
                max_completion_tokens=request.max_output_tokens,
                stream=True,
            )
            for chunk in stream:
                content = self._chunk_content(chunk)
                if content:
                    yield StreamingTextEvent(
                        StreamEventType.DELTA, sequence, delta=content,
                        provider=_PROVIDER_NAME, model=model,
                        request_id=self._identifier(chunk),
                    )
                    sequence += 1
        except Exception:
            yield self._stream_error(sequence, model)
            return
        yield StreamingTextEvent(StreamEventType.END, sequence, provider=_PROVIDER_NAME,
                                 model=model, finish_reason="completed")

    def classify_text(self, request: ClassificationRequest) -> ClassificationResult:
        configuration = self._feature_configuration(FEATURE_CLASSIFICATION)
        model = request.model or configuration.model
        try:
            response = self._client_for(configuration).chat.completions.create(
                model=model,
                messages=self._classification_messages(request),
                response_format={"type": "json_object"},
            )
            body = self._structured_body(response)
            labels = self._classification_labels(body, request.labels)
            rationale = body.get("rationale")
            if rationale is not None and not isinstance(rationale, str):
                raise AIProviderError("AI provider returned invalid structured output.")
            return ClassificationResult(labels=labels, provider=_PROVIDER_NAME, model=model,
                                        rationale=rationale, request_id=self._identifier(response))
        except AIProviderError:
            raise
        except Exception:
            raise AIProviderError("AI provider request failed.") from None

    def moderate_content(self, request: ModerationRequest) -> ModerationResult:
        configuration = self._feature_configuration(FEATURE_MODERATION)
        model = request.model or configuration.model
        try:
            response = self._client_for(configuration).chat.completions.create(
                model=model,
                messages=self._moderation_messages(request),
                response_format={"type": "json_object"},
            )
            body = self._structured_body(response)
            if not isinstance(body.get("flagged"), bool) or not isinstance(body.get("decision"), str):
                raise AIProviderError("AI provider returned invalid structured output.")
            return ModerationResult(
                flagged=body["flagged"], findings=self._moderation_findings(body, request.categories),
                provider=_PROVIDER_NAME, model=model, decision=body["decision"],
                request_id=self._identifier(response),
            )
        except AIProviderError:
            raise
        except Exception:
            raise AIProviderError("AI provider request failed.") from None

    def _feature_configuration(self, feature: str) -> FeatureAIConfiguration:
        configuration = self._configuration.for_feature(feature)
        if configuration.provider != _PROVIDER_NAME:
            raise AIProviderError("Configured AI provider does not match this adapter.")
        return configuration

    def _client_for(self, configuration: FeatureAIConfiguration) -> Any:
        api_key = str(self._environment.get(configuration.api_key_env, "")).strip()
        if not api_key:
            raise AIProviderError("AI provider credentials are not configured.")
        if self._client_override is not None:
            return self._client_override
        if configuration.api_key_env not in self._clients:
            self._clients[configuration.api_key_env] = Groq(api_key=api_key, timeout=self._timeout_seconds)
        return self._clients[configuration.api_key_env]

    @staticmethod
    def _safe_text(value: Optional[str]) -> str:
        return redact_pii(value or "").sanitized_text

    def _generation_messages(self, request: TextGenerationRequest) -> Sequence[Dict[str, str]]:
        messages = []
        if request.system_instruction:
            messages.append({"role": "system", "content": self._safe_text(request.system_instruction)})
        for message in request.messages:
            role = message.role.casefold()
            messages.append({"role": role if role in {"system", "user", "assistant"} else "user",
                             "content": self._safe_text(message.content)})
        messages.append({"role": "user", "content": self._safe_text(request.prompt)})
        return messages

    def _classification_messages(self, request: ClassificationRequest) -> Sequence[Dict[str, str]]:
        return [
            {"role": "system", "content": "Return JSON only with labels and optional rationale. Each label has label, confidence, selected. Use only allowed labels."},
            {"role": "user", "content": "Allowed labels: %s. Allow multiple: %s. Instruction: %s. Text: %s" % (", ".join(request.labels), request.allow_multiple, self._safe_text(request.instruction), self._safe_text(request.text))},
        ]

    def _moderation_messages(self, request: ModerationRequest) -> Sequence[Dict[str, str]]:
        return [
            {"role": "system", "content": "Return JSON only with flagged, decision, findings. Each finding has category, confidence, severity and optional reason. Use only allowed categories."},
            {"role": "user", "content": "Allowed categories: %s. Instruction: %s. Content: %s" % (", ".join(request.categories), self._safe_text(request.instruction), self._safe_text(request.content))},
        ]

    @staticmethod
    def _content(response: Any) -> str:
        try:
            value = response.choices[0].message.content
        except (AttributeError, IndexError, TypeError) as error:
            raise AIProviderError("AI provider returned an invalid response.") from error
        if not isinstance(value, str) or not value.strip():
            raise AIProviderError("AI provider returned an empty text response.")
        return value

    @staticmethod
    def _chunk_content(chunk: Any) -> str:
        try:
            value = chunk.choices[0].delta.content
        except (AttributeError, IndexError, TypeError):
            return ""
        return value if isinstance(value, str) else ""

    def _structured_body(self, response: Any) -> Dict[str, Any]:
        try:
            result = json.loads(self._content(response))
        except (TypeError, ValueError, json.JSONDecodeError) as error:
            raise AIProviderError("AI provider returned invalid structured output.") from error
        if not isinstance(result, dict):
            raise AIProviderError("AI provider returned invalid structured output.")
        return result

    @staticmethod
    def _identifier(response: Any) -> Optional[str]:
        value = getattr(response, "id", None)
        return value.strip() if isinstance(value, str) and value.strip() else None

    @staticmethod
    def _usage(response: Any) -> Dict[str, int]:
        usage = getattr(response, "usage", None)
        names = ("prompt_tokens", "completion_tokens", "total_tokens")
        try:
            return {name: int(getattr(usage, name)) for name in names if getattr(usage, name, None) is not None}
        except (TypeError, ValueError) as error:
            raise AIProviderError("AI provider returned an invalid response.") from error

    @staticmethod
    def _classification_labels(body: Dict[str, Any], allowed: Tuple[str, ...]) -> Tuple[ClassificationLabel, ...]:
        raw = body.get("labels")
        if not isinstance(raw, list) or not raw:
            raise AIProviderError("AI provider returned invalid structured output.")
        labels = []
        for item in raw:
            if not isinstance(item, dict) or item.get("label") not in allowed or not isinstance(item.get("selected"), bool):
                raise AIProviderError("AI provider returned invalid structured output.")
            try:
                labels.append(ClassificationLabel(item["label"], item["confidence"], item["selected"]))
            except (KeyError, TypeError, ValueError) as error:
                raise AIProviderError("AI provider returned invalid structured output.") from error
        return tuple(labels)

    @staticmethod
    def _moderation_findings(body: Dict[str, Any], allowed: Tuple[str, ...]) -> Tuple[ModerationFinding, ...]:
        raw = body.get("findings")
        if not isinstance(raw, list):
            raise AIProviderError("AI provider returned invalid structured output.")
        findings = []
        for item in raw:
            if not isinstance(item, dict) or item.get("category") not in allowed:
                raise AIProviderError("AI provider returned invalid structured output.")
            try:
                findings.append(ModerationFinding(item["category"], item["confidence"], item["severity"], item.get("reason")))
            except (KeyError, TypeError, ValueError) as error:
                raise AIProviderError("AI provider returned invalid structured output.") from error
        return tuple(findings)

    @staticmethod
    def _stream_error(sequence: int, model: Optional[str] = None) -> StreamingTextEvent:
        return StreamingTextEvent(StreamEventType.ERROR, sequence, provider=_PROVIDER_NAME,
                                  model=model, error_code="provider_error")


__all__ = ["GroqProvider"]
