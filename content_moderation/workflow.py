"""Backend workflow around the canonical AI moderation pipeline.

The review queue is process-local and non-durable. It is intended only for the
current integration phase and must be replaced by the company's persistent
queue or database before multi-worker or multi-instance deployment.
"""

from collections import deque
from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import threading
import uuid

from .pipeline import (
    MediaValidationError,
    run_production_moderation,
    validate_media_file,
)


BACKEND_MODERATION_STATUSES = {
    "APPROVED",
    "PENDING_REVIEW",
    "AUTO_ADMIN_REVIEW",
}

BACKEND_LONG_VIDEO_SECONDS = float(
    os.getenv("BACKEND_LONG_VIDEO_SECONDS", "20")
)
ADMIN_REVIEW_QUEUE_MAX_SIZE = int(
    os.getenv("ADMIN_REVIEW_QUEUE_MAX_SIZE", "1000")
)
ADMIN_PAYLOAD_MAX_OCR_CHARS = int(
    os.getenv("ADMIN_PAYLOAD_MAX_OCR_CHARS", "2000")
)
ADMIN_PAYLOAD_MAX_TRANSCRIPT_CHARS = int(
    os.getenv("ADMIN_PAYLOAD_MAX_TRANSCRIPT_CHARS", "4000")
)
ADMIN_PAYLOAD_MAX_REASON_CHARS = int(
    os.getenv("ADMIN_PAYLOAD_MAX_REASON_CHARS", "1000")
)
ADMIN_PAYLOAD_MAX_EVIDENCE_ITEMS = int(
    os.getenv("ADMIN_PAYLOAD_MAX_EVIDENCE_ITEMS", "50")
)
ADMIN_PAYLOAD_MAX_DETECTIONS = int(
    os.getenv("ADMIN_PAYLOAD_MAX_DETECTIONS", "25")
)

if BACKEND_LONG_VIDEO_SECONDS <= 0:
    raise ValueError("BACKEND_LONG_VIDEO_SECONDS must be greater than zero.")
for _name, _value in (
    ("ADMIN_REVIEW_QUEUE_MAX_SIZE", ADMIN_REVIEW_QUEUE_MAX_SIZE),
    ("ADMIN_PAYLOAD_MAX_OCR_CHARS", ADMIN_PAYLOAD_MAX_OCR_CHARS),
    ("ADMIN_PAYLOAD_MAX_TRANSCRIPT_CHARS", ADMIN_PAYLOAD_MAX_TRANSCRIPT_CHARS),
    ("ADMIN_PAYLOAD_MAX_REASON_CHARS", ADMIN_PAYLOAD_MAX_REASON_CHARS),
    ("ADMIN_PAYLOAD_MAX_EVIDENCE_ITEMS", ADMIN_PAYLOAD_MAX_EVIDENCE_ITEMS),
    ("ADMIN_PAYLOAD_MAX_DETECTIONS", ADMIN_PAYLOAD_MAX_DETECTIONS),
):
    if _value <= 0:
        raise ValueError(f"{_name} must be greater than zero.")


class ModerationWorkflowError(RuntimeError):
    """Base error for operational moderation failures."""


class ModerationProcessingError(ModerationWorkflowError):
    """The AI pipeline failed before producing a safe backend decision."""


class AdminQueueError(ModerationWorkflowError):
    """Base error for review queue failures."""


class AdminQueueFullError(AdminQueueError):
    """The bounded review queue cannot accept another item."""


class DuplicateReviewError(AdminQueueError):
    """A review ID is already present in the queue."""


class InMemoryAdminReviewQueue:
    """Thread-safe, bounded, process-local queue for the integration phase."""

    def __init__(self, max_size):
        if int(max_size) <= 0:
            raise ValueError("Queue maximum size must be greater than zero.")
        self._max_size = int(max_size)
        self._items = deque()
        self._review_ids = set()
        self._lock = threading.RLock()

    @property
    def max_size(self):
        return self._max_size

    def enqueue(self, payload):
        item = deepcopy(payload)
        review_id = item.get("review_id")
        if not review_id:
            raise AdminQueueError("Review payload requires a review_id.")
        with self._lock:
            if review_id in self._review_ids:
                raise DuplicateReviewError(f"Duplicate review_id: {review_id}")
            if len(self._items) >= self._max_size:
                raise AdminQueueFullError("The admin review queue is full.")
            self._items.append(item)
            self._review_ids.add(review_id)
        return deepcopy(item)

    def get(self, review_id):
        with self._lock:
            for item in self._items:
                if item.get("review_id") == review_id:
                    return deepcopy(item)
        return None

    def list(self, limit=100):
        bounded_limit = int(limit)
        if bounded_limit <= 0:
            raise ValueError("Queue list limit must be greater than zero.")
        with self._lock:
            return deepcopy(list(self._items)[:bounded_limit])

    def size(self):
        with self._lock:
            return len(self._items)

    def clear(self):
        with self._lock:
            self._items.clear()
            self._review_ids.clear()


admin_review_queue = InMemoryAdminReviewQueue(ADMIN_REVIEW_QUEUE_MAX_SIZE)

# First-integration safety: detector/model objects execute one request at a
# time in this process. Scale later with isolated workers after replacing the
# process-local queue with durable shared infrastructure.
AI_EXECUTION_LOCK = threading.RLock()


_SENSITIVE_KEY_PARTS = (
    "api_key", "authorization", "base64", "raw_bytes", "media_bytes",
    "file_path", "filepath", "frame_path", "analyzed_files",
    "temporary", "temp_path", "model_path", "environment",
)


def _safe_original_filename(original_filename):
    if not original_filename:
        return None
    return Path(str(original_filename).replace("\\", "/")).name[:255] or None


def _is_sensitive_key(key):
    normalized = str(key).casefold()
    return any(part in normalized for part in _SENSITIVE_KEY_PARTS)


def _sanitize_value(value, key=None, list_limit=ADMIN_PAYLOAD_MAX_EVIDENCE_ITEMS):
    if key is not None and _is_sensitive_key(key):
        return None
    if isinstance(value, bytes):
        return None
    if isinstance(value, Path):
        return None
    if isinstance(value, dict):
        return {
            str(item_key): _sanitize_value(item_value, item_key, list_limit)
            for item_key, item_value in value.items()
            if not _is_sensitive_key(item_key)
        }
    if isinstance(value, (list, tuple, set)):
        return [
            _sanitize_value(item, None, list_limit)
            for item in list(value)[:list_limit]
        ]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def sanitize_validation_result(validation_result):
    """Return validation evidence without any server-local file path."""

    safe = _sanitize_value(validation_result)
    if isinstance(safe, dict):
        safe.pop("file_path", None)
    return safe


def sanitize_moderation_report(moderation_report):
    """Create a bounded, JSON-safe copy without paths, bytes, or secrets."""

    return _sanitize_value(deepcopy(moderation_report))


def _bounded_text(value, limit):
    text = str(value or "")
    return {
        "text": text[:limit],
        "original_length": len(text),
        "truncated": len(text) > limit,
    }


def _bounded_list(value, limit):
    items = list(value or [])
    safe_items = _sanitize_value(items[:limit], list_limit=limit)
    return {
        "items": safe_items,
        "original_count": len(items),
        "truncated": len(items) > limit,
    }


def _compact_detector(detector_result, detection_limit=None):
    if not isinstance(detector_result, dict):
        return None
    compact = {
        key: detector_result.get(key)
        for key in (
            "detected", "confirmed", "triggered", "confidence", "count", "category",
            "severity", "status", "reason", "language", "duration_seconds",
        )
        if key in detector_result
    }
    if "detections" in detector_result:
        compact["detections"] = _bounded_list(
            detector_result.get("detections"),
            detection_limit or ADMIN_PAYLOAD_MAX_DETECTIONS,
        )
    for list_name in ("keywords", "violations", "ignored_classes"):
        if list_name in detector_result:
            compact[list_name] = _bounded_list(
                detector_result.get(list_name),
                ADMIN_PAYLOAD_MAX_EVIDENCE_ITEMS,
            )
    return _sanitize_value(compact)


def build_admin_review_payload(backend_result, moderation_report=None):
    """Build one bounded, JSON-serializable payload for human review."""

    report = moderation_report if isinstance(moderation_report, dict) else {}
    detectors = report.get("detectors", {})
    temporal = report.get("temporal_fusion", {})
    policy = report.get("policy_result", {})
    risk = report.get("risk", {})
    routing = report.get("routing", {})
    media = deepcopy(backend_result.get("media", {}))

    ocr = detectors.get("ocr", {})
    stt = detectors.get("stt", {})
    keyword = detectors.get("keyword", {})
    vlm = detectors.get("vlm", detectors.get("gemini", {}))

    payload = {
        "review_id": backend_result.get("review_id"),
        "request_id": backend_result.get("request_id"),
        "status": backend_result.get("status"),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "review_reason": _bounded_text(
            backend_result.get("reason"), ADMIN_PAYLOAD_MAX_REASON_CHARS
        ),
        "media": _sanitize_value(media),
        "internal_decision": backend_result.get("internal_decision"),
        "ai_processing_skipped": bool(
            backend_result.get("ai_processing_skipped", False)
        ),
        "ai_skip_reason": backend_result.get("ai_skip_reason"),
        "policy": {
            "label": policy.get("policy", report.get("policy_label")),
            "severity": policy.get("severity"),
            "confidence": policy.get("confidence", report.get("policy_confidence")),
            "reason": _bounded_text(
                policy.get("reason", report.get("policy_reason")),
                ADMIN_PAYLOAD_MAX_REASON_CHARS,
            ) if report else None,
        } if report else None,
        "risk": {
            "score": risk.get("risk_score"),
            "breakdown": _sanitize_value(
                risk.get("score_breakdown", risk.get("breakdown"))
            ),
        } if report else None,
        "vlm_routing": _sanitize_value(routing) if report else None,
        "escalation_reasons": _bounded_list(
            routing.get("escalation_reasons", report.get("escalation_reasons", [])),
            ADMIN_PAYLOAD_MAX_EVIDENCE_ITEMS,
        ) if report else None,
        "temporal_fusion": {
            "summary": _sanitize_value(temporal.get("summary")),
            "confirmed_events": _bounded_list(
                temporal.get("confirmed_events"), ADMIN_PAYLOAD_MAX_EVIDENCE_ITEMS
            ),
            "uncertain_events": _bounded_list(
                temporal.get("uncertain_events"), ADMIN_PAYLOAD_MAX_EVIDENCE_ITEMS
            ),
            "suppressed_events": _bounded_list(
                temporal.get("suppressed_events"), ADMIN_PAYLOAD_MAX_EVIDENCE_ITEMS
            ),
        } if report else None,
        "sampling_metadata": _sanitize_value(
            report.get("media", {}).get("sampling")
        ) if report else None,
        "detectors": {
            "ocr": {
                **(_compact_detector(ocr) or {}),
                "text": _bounded_text(ocr.get("text"), ADMIN_PAYLOAD_MAX_OCR_CHARS),
            },
            "keyword": _compact_detector(keyword),
            "nsfw": _compact_detector(detectors.get("nsfw")),
            "weapon": _compact_detector(detectors.get("weapon")),
            "violence": _compact_detector(detectors.get("violence")),
            "stt": {
                **(_compact_detector(stt) or {}),
                "transcript": _bounded_text(
                    stt.get("transcript"), ADMIN_PAYLOAD_MAX_TRANSCRIPT_CHARS
                ),
            },
            "vlm": _compact_detector(vlm),
        } if report else None,
    }
    sanitized = _sanitize_value(payload)
    json.dumps(sanitized)
    return sanitized


def _safe_media_metadata(validation, original_filename, content_type):
    return {
        "original_filename": _safe_original_filename(original_filename),
        "content_type": str(content_type)[:255] if content_type else None,
        "media_type": validation.get("media_type"),
        "size_bytes": int(validation.get("size_bytes") or 0),
        "duration_seconds": float(validation.get("duration_seconds") or 0.0),
        "long_video_threshold_seconds": BACKEND_LONG_VIDEO_SECONDS,
    }


def _map_internal_decision(internal_decision):
    if internal_decision in {"APPROVED", "APPROVED_WITH_LOG"}:
        return "APPROVED"
    if internal_decision == "PENDING_REVIEW":
        return "PENDING_REVIEW"
    return "PENDING_REVIEW"


def _enqueue_required_review(backend_result, moderation_report):
    payload = build_admin_review_payload(backend_result, moderation_report)
    admin_review_queue.enqueue(payload)
    backend_result["queued_for_admin_review"] = True


def moderate_for_backend(
    file_path,
    original_filename=None,
    content_type=None,
    request_id=None,
):
    """Return the stable application contract using one canonical AI invocation."""

    resolved_request_id = str(request_id or uuid.uuid4())
    validation = validate_media_file(file_path)
    if not validation.get("valid"):
        raise MediaValidationError(validation)

    media = _safe_media_metadata(validation, original_filename, content_type)
    is_long_video = (
        validation.get("media_type") == "video"
        and float(validation.get("duration_seconds") or 0.0)
        > BACKEND_LONG_VIDEO_SECONDS
    )

    if is_long_video:
        reason = "Video duration exceeds the automatic AI moderation limit."
        result = {
            "request_id": resolved_request_id,
            "status": "AUTO_ADMIN_REVIEW",
            "internal_decision": None,
            "reason": reason,
            "review_id": str(uuid.uuid4()),
            "queued_for_admin_review": False,
            "ai_processing_skipped": True,
            "ai_skip_reason": reason,
            "media": media,
            "moderation_report": None,
        }
        _enqueue_required_review(result, None)
        return result

    try:
        with AI_EXECUTION_LOCK:
            report = run_production_moderation(file_path)
    except MediaValidationError:
        raise
    except Exception as error:
        raise ModerationProcessingError(
            "The AI moderation pipeline failed closed."
        ) from error

    internal_decision = report.get("final_decision")
    status = _map_internal_decision(internal_decision)
    result = {
        "request_id": resolved_request_id,
        "status": status,
        "internal_decision": internal_decision,
        "reason": str(report.get("reason") or "Human review is required."),
        "review_id": str(uuid.uuid4()) if status != "APPROVED" else None,
        "queued_for_admin_review": False,
        "ai_processing_skipped": False,
        "ai_skip_reason": None,
        "media": media,
        "moderation_report": sanitize_moderation_report(report),
    }
    if status != "APPROVED":
        _enqueue_required_review(result, report)
    return result


__all__ = [
    "BACKEND_MODERATION_STATUSES",
    "BACKEND_LONG_VIDEO_SECONDS",
    "ModerationWorkflowError",
    "ModerationProcessingError",
    "AdminQueueError",
    "AdminQueueFullError",
    "DuplicateReviewError",
    "InMemoryAdminReviewQueue",
    "admin_review_queue",
    "build_admin_review_payload",
    "moderate_for_backend",
    "sanitize_validation_result",
    "sanitize_moderation_report",
]
