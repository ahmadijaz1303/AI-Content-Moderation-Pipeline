"""OpenRouter adapter for the provider-neutral shared AI contracts.

This adapter is separate from the completed Part 2 vision fallback.  It makes
text-only OpenAI-compatible chat-completions requests for future shared AI
features and returns only the project's provider-neutral contracts.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Iterator, Mapping, Optional, Sequence, Tuple

import requests

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


_PROVIDER_NAME = "openrouter"
_DEFAULT_URL = "https://openrouter.ai/api/v1/chat/completions"


class OpenRouterProvider(AIProvider):
    """Translate shared AI contracts into OpenRouter chat-completions calls."""

    def __init__(
        self,
        configuration: Optional[SharedAIConfiguration] = None,
        environment: Optional[Mapping[str, str]] = None,
        http_client: Optional[Any] = None,
        timeout_seconds: int = 60,
    ) -> None:
        """Create a provider without making a network request.

        ``http_client`` supports controlled mock injection in tests.  Normal
        runtime use relies on the existing ``requests`` dependency.
        """

        self._environment = os.environ if environment is None else environment
        self._configuration = configuration or load_shared_ai_configuration(
            self._environment
        )
        self._http_client = http_client or requests
        self._timeout_seconds = int(timeout_seconds)
        self._url = str(self._environment.get("OPENROUTER_URL", _DEFAULT_URL)).strip()
        if not self._url:
            raise ValueError("OPENROUTER_URL must not be empty.")

    def generate_text(self, request: TextGenerationRequest) -> TextGenerationResult:
        """Generate text and return a stable, SDK-free result."""

        configuration = self._feature_configuration(FEATURE_TEXT_GENERATION)
        model = self._model_for(request.model, configuration)
        try:
            payload = self._chat_payload(
                model=model,
                messages=self._generation_messages(request),
                temperature=request.temperature,
                max_tokens=request.max_output_tokens,
            )
            response = self._post(configuration, payload)
            body = self._json_body(response)
            return TextGenerationResult(
                text=self._message_content(body),
                provider=_PROVIDER_NAME,
                model=model,
                request_id=self._response_identifier(body, response),
                usage=self._response_usage(body),
            )
        except AIProviderError:
            raise
        except Exception:
            raise AIProviderError("AI provider request failed.") from None

    def stream_text(
        self, request: TextGenerationRequest
    ) -> Iterator[StreamingTextEvent]:
        """Yield normalized Server-Sent Event chunks from OpenRouter."""

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
            payload = self._chat_payload(
                model=model,
                messages=self._generation_messages(request),
                temperature=request.temperature,
                max_tokens=request.max_output_tokens,
                stream=True,
            )
            response = self._post(configuration, payload, stream=True)
            for payload in self._stream_payloads(response):
                text = self._stream_delta(payload)
                if text:
                    yield StreamingTextEvent(
                        event=StreamEventType.DELTA,
                        sequence=sequence,
                        delta=text,
                        provider=_PROVIDER_NAME,
                        model=model,
                        request_id=self._response_identifier(payload, response),
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
        """Return validated structured classification using JSON-object output."""

        configuration = self._feature_configuration(FEATURE_CLASSIFICATION)
        model = self._model_for(request.model, configuration)
        try:
            payload = self._chat_payload(
                model=model,
                messages=self._classification_messages(request),
                response_format={"type": "json_object"},
            )
            response = self._post(configuration, payload)
            response_body = self._json_body(response)
            body = self._structured_content(response_body)
            labels = self._classification_labels(body, request.labels)
            rationale = body.get("rationale")
            if rationale is not None and not isinstance(rationale, str):
                raise AIProviderError("AI provider returned invalid structured output.")
            return ClassificationResult(
                labels=labels,
                provider=_PROVIDER_NAME,
                model=model,
                rationale=rationale,
                request_id=self._response_identifier(response_body, response),
            )
        except AIProviderError:
            raise
        except Exception:
            raise AIProviderError("AI provider request failed.") from None

    def moderate_content(self, request: ModerationRequest) -> ModerationResult:
        """Return validated structured generic moderation output."""

        configuration = self._feature_configuration(FEATURE_MODERATION)
        model = self._model_for(request.model, configuration)
        try:
            payload = self._chat_payload(
                model=model,
                messages=self._moderation_messages(request),
                response_format={"type": "json_object"},
            )
            response = self._post(configuration, payload)
            response_body = self._json_body(response)
            body = self._structured_content(response_body)
            flagged = body.get("flagged")
            decision = body.get("decision")
            if not isinstance(flagged, bool) or not isinstance(decision, str):
                raise AIProviderError("AI provider returned invalid structured output.")
            return ModerationResult(
                flagged=flagged,
                findings=self._moderation_findings(body, request.categories),
                provider=_PROVIDER_NAME,
                model=model,
                decision=decision,
                request_id=self._response_identifier(response_body, response),
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

    def _post(
        self,
        configuration: FeatureAIConfiguration,
        payload: Dict[str, Any],
        stream: bool = False,
    ) -> Any:
        api_key = str(self._environment.get(configuration.api_key_env, "")).strip()
        if not api_key:
            raise AIProviderError("AI provider credentials are not configured.")
        response = self._http_client.post(
            self._url,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=self._timeout_seconds,
            stream=stream,
        )
        response.raise_for_status()
        return response

    @staticmethod
    def _model_for(model_override: Optional[str], configuration: FeatureAIConfiguration) -> str:
        return model_override or configuration.model

    @staticmethod
    def _safe_text(text: Optional[str]) -> str:
        return redact_pii(text or "").sanitized_text

    def _generation_messages(self, request: TextGenerationRequest) -> Sequence[Dict[str, str]]:
        messages = []
        if request.system_instruction is not None:
            messages.append(
                {"role": "system", "content": self._safe_text(request.system_instruction)}
            )
        for message in request.messages:
            role = message.role.casefold()
            if role not in {"system", "user", "assistant"}:
                role = "user"
            messages.append({"role": role, "content": self._safe_text(message.content)})
        messages.append({"role": "user", "content": self._safe_text(request.prompt)})
        return messages

    def _classification_messages(self, request: ClassificationRequest) -> Sequence[Dict[str, str]]:
        instruction = self._safe_text(request.instruction)
        safe_text = self._safe_text(request.text)
        labels = ", ".join(request.labels)
        return [
            {
                "role": "system",
                "content": (
                    "Return one JSON object with labels and optional rationale. "
                    "Each labels item must contain label, confidence, and selected. "
                    "Use only the allowed labels and do not add prose outside JSON."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Allowed labels: {labels}. Allow multiple labels: {request.allow_multiple}. "
                    f"Instruction: {instruction or 'None'}. Text: {safe_text}"
                ),
            },
        ]

    def _moderation_messages(self, request: ModerationRequest) -> Sequence[Dict[str, str]]:
        instruction = self._safe_text(request.instruction)
        safe_content = self._safe_text(request.content)
        categories = ", ".join(request.categories)
        return [
            {
                "role": "system",
                "content": (
                    "Return one JSON object with flagged, decision, and findings. "
                    "Each finding must contain category, confidence, severity, and optional reason. "
                    "Use only the allowed categories and do not add prose outside JSON."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Allowed categories: {categories}. "
                    f"Instruction: {instruction or 'None'}. Content: {safe_content}"
                ),
            },
        ]

    @staticmethod
    def _chat_payload(
        model: str,
        messages: Sequence[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        response_format: Optional[Dict[str, str]] = None,
        stream: bool = False,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"model": model, "messages": list(messages)}
        if temperature is not None:
            payload["temperature"] = temperature
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if response_format is not None:
            payload["response_format"] = response_format
        if stream:
            payload["stream"] = True
        return payload

    @staticmethod
    def _json_body(response: Any) -> Dict[str, Any]:
        try:
            body = response.json()
        except (TypeError, ValueError) as error:
            raise AIProviderError("AI provider returned an invalid response.") from error
        if not isinstance(body, dict):
            raise AIProviderError("AI provider returned an invalid response.")
        return body

    def _message_content(self, body: Dict[str, Any]) -> str:
        try:
            content = body["choices"][0]["message"]["content"]
        except (IndexError, KeyError, TypeError) as error:
            raise AIProviderError("AI provider returned an invalid response.") from error
        if not isinstance(content, str) or not content.strip():
            raise AIProviderError("AI provider returned an empty text response.")
        return content

    def _structured_body(self, response: Any) -> Dict[str, Any]:
        return self._structured_content(self._json_body(response))

    def _structured_content(self, response_body: Dict[str, Any]) -> Dict[str, Any]:
        try:
            content = response_body["choices"][0]["message"]["content"]
            body = json.loads(content)
        except (IndexError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise AIProviderError("AI provider returned invalid structured output.") from error
        if not isinstance(body, dict):
            raise AIProviderError("AI provider returned invalid structured output.")
        return body

    @staticmethod
    def _response_identifier(body: Dict[str, Any], response: Any) -> Optional[str]:
        value = body.get("id")
        if isinstance(value, str) and value.strip():
            return value.strip()
        headers = getattr(response, "headers", {})
        value = headers.get("x-request-id") if hasattr(headers, "get") else None
        return value.strip() if isinstance(value, str) and value.strip() else None

    @staticmethod
    def _response_usage(body: Dict[str, Any]) -> Dict[str, int]:
        usage = body.get("usage")
        if not isinstance(usage, dict):
            return {}
        values = {
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "total_tokens": usage.get("total_tokens"),
        }
        try:
            return {name: int(value) for name, value in values.items() if value is not None}
        except (TypeError, ValueError) as error:
            raise AIProviderError("AI provider returned an invalid response.") from error

    def _classification_labels(
        self, body: Dict[str, Any], allowed_labels: Tuple[str, ...]
    ) -> Tuple[ClassificationLabel, ...]:
        raw_labels = body.get("labels")
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
        self, body: Dict[str, Any], allowed_categories: Tuple[str, ...]
    ) -> Tuple[ModerationFinding, ...]:
        raw_findings = body.get("findings")
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
    def _stream_payloads(response: Any) -> Iterator[Dict[str, Any]]:
        for raw_line in response.iter_lines(decode_unicode=True):
            if isinstance(raw_line, bytes):
                raw_line = raw_line.decode("utf-8", errors="replace")
            if not isinstance(raw_line, str) or not raw_line.startswith("data:"):
                continue
            data = raw_line[5:].strip()
            if data == "[DONE]":
                return
            try:
                payload = json.loads(data)
            except (TypeError, ValueError, json.JSONDecodeError) as error:
                raise AIProviderError("AI provider returned an invalid stream.") from error
            if not isinstance(payload, dict):
                raise AIProviderError("AI provider returned an invalid stream.")
            yield payload

    @staticmethod
    def _stream_delta(payload: Dict[str, Any]) -> str:
        try:
            content = payload["choices"][0]["delta"].get("content", "")
        except (IndexError, KeyError, TypeError) as error:
            raise AIProviderError("AI provider returned an invalid stream.") from error
        if not isinstance(content, str) or not content.strip():
            return ""
        return content

    @staticmethod
    def _stream_error(sequence: int, model: Optional[str] = None) -> StreamingTextEvent:
        return StreamingTextEvent(
            event=StreamEventType.ERROR,
            sequence=sequence,
            provider=_PROVIDER_NAME,
            model=model,
            error_code="provider_error",
        )


__all__ = ["OpenRouterProvider"]
