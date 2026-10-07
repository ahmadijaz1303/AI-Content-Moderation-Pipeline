"""Canonical production AI content-moderation pipeline.

Extracted once from check.ipynb. The notebook imports this module and
contains only development, validation, demonstration, and benchmark code.
"""
import base64
import hashlib
import json
import mimetypes
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
import unicodedata
from pathlib import Path

import cv2
import numpy as np
import onnx
import onnxruntime as ort
import pytesseract
import requests
import torch
from dotenv import load_dotenv
from google import genai
from google.genai import types
from nudenet import NudeDetector
from PIL import Image

from .ai.part2_integration import (
    classify_part2_moderation_context,
    part2_context_integration_enabled,
)

# Preserve Pillow's decoder before Ultralytics adds optional HEIF behavior.
PIL_IMAGE_OPEN = Image.open

from ultralytics import YOLO


# Environment must be loaded before module-level configuration.
load_dotenv()


MODERATION_MODEL_DIRECTORY = Path(
    os.getenv("MODERATION_MODEL_DIRECTORY") or "models"
)
MODERATION_TEMP_DIRECTORY = Path(
    os.getenv("MODERATION_TEMP_DIRECTORY") or tempfile.gettempdir()
)
MODERATION_FRAME_DIRECTORY = Path(
    os.getenv("MODERATION_FRAME_DIRECTORY")
    or str(MODERATION_TEMP_DIRECTORY / "moderation_frames")
)

for configured_directory in (
    MODERATION_MODEL_DIRECTORY,
    MODERATION_TEMP_DIRECTORY,
    MODERATION_FRAME_DIRECTORY
):
    configured_directory.mkdir(parents=True, exist_ok=True)


configured_tesseract_path = os.getenv("TESSERACT_CMD", "").strip()
windows_tesseract_path = Path(
    "C:/Program Files/Tesseract-OCR/tesseract.exe"
)

if configured_tesseract_path:
    resolved_tesseract_path = configured_tesseract_path
elif shutil.which("tesseract"):
    resolved_tesseract_path = shutil.which("tesseract")
elif windows_tesseract_path.is_file():
    resolved_tesseract_path = str(windows_tesseract_path)
else:
    resolved_tesseract_path = None

if resolved_tesseract_path:
    pytesseract.pytesseract.tesseract_cmd = resolved_tesseract_path
    print(f"Tesseract ready: {resolved_tesseract_path}")
else:
    print(
        "Tesseract was not found. Set TESSERACT_CMD before processing "
        "media that requires OCR."
    )


MODEL = os.getenv("GEMINI_VISION_MODEL", "models/gemini-flash-latest")
GEMINI_TIMEOUT_SECONDS = int(os.getenv("GEMINI_TIMEOUT_SECONDS", "30"))
_GEMINI_CLIENT = None
_GEMINI_CLIENT_LOCK = threading.RLock()


def _get_gemini_client():
    """Create the Gemini client once, only when VLM inference is required."""

    global _GEMINI_CLIENT
    if _GEMINI_CLIENT is None:
        with _GEMINI_CLIENT_LOCK:
            if _GEMINI_CLIENT is None:
                api_key = os.getenv("GEMINI_API_KEY")
                if not api_key:
                    raise ValueError("GEMINI_API_KEY is not configured.")
                _GEMINI_CLIENT = genai.Client(
                    api_key=api_key,
                    # google-genai expects HTTP timeouts in milliseconds; the
                    # environment setting is intentionally expressed in seconds.
                    http_options=types.HttpOptions(
                        timeout=GEMINI_TIMEOUT_SECONDS * 1000
                    ),
                )
    return _GEMINI_CLIENT

# Extracted from original notebook cell 5.
def ask_gemini(file_path, prompt):
    """Call Gemini, retry temporary errors, and immediately expose quota errors."""

    mime_type = mimetypes.guess_type(file_path)[0]

    with open(file_path, "rb") as file:
        file_data = file.read()

    for attempt in range(3):

        try:
            response = _get_gemini_client().models.generate_content(
                model=MODEL,
                contents=[
                    types.Part.from_bytes(
                        data=file_data,
                        mime_type=mime_type
                    ),
                    prompt
                ],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json"
                )
            )

            return response.text

        except Exception as error:
            error_text = str(error)

            # Retrying cannot fix an exhausted quota. Raise immediately so
            # ask_moderation_model() switches to OpenRouter without delay.
            is_quota_error = (
                "429" in error_text
                or "RESOURCE_EXHAUSTED" in error_text
                or "quota" in error_text.lower()
            )

            if is_quota_error:
                print("Gemini quota or rate limit reached.")
                raise

            print(f"Gemini attempt {attempt + 1} failed.")
            print(error)

            if attempt < 2:
                print("Retrying Gemini in 3 seconds...\n")
                time.sleep(3)
            else:
                raise


# Extracted from original notebook cell 6.
# OpenRouter is used only after Gemini becomes unavailable.

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_URL = os.getenv(
    "OPENROUTER_URL", "https://openrouter.ai/api/v1/chat/completions"
)
OPENROUTER_VISION_MODEL = os.getenv(
    "OPENROUTER_VISION_MODEL", "openrouter/free"
)

# OpenRouter's free router selects an available vision-capable model.
# This value lasts only for the current notebook kernel session.
# It prevents repeated Gemini calls after its quota has been exhausted.
gemini_available = True
last_moderation_model = None


def ask_openrouter(file_path, prompt):
    """Send a local image to OpenRouter's free vision-model router."""

    global last_moderation_model

    if not OPENROUTER_API_KEY:
        raise ValueError(
            "OPENROUTER_API_KEY is missing. Add it to the .env file."
        )

    mime_type = mimetypes.guess_type(file_path)[0] or "image/jpeg"

    if not mime_type.startswith("image/"):
        raise ValueError(
            "OpenRouter fallback expects an image or extracted video frame."
        )

    with open(file_path, "rb") as file:
        image_base64 = base64.b64encode(file.read()).decode("utf-8")

    image_url = f"data:{mime_type};base64,{image_base64}"

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": OPENROUTER_VISION_MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": image_url}
                    }
                ]
            }
        ],
        "response_format": {"type": "json_object"},
        "temperature": 0,
        "max_tokens": 300
    }

    for attempt in range(3):
        try:
            print(
                f"Trying OpenRouter free vision router "
                f"(attempt {attempt + 1}/3)"
            )

            api_response = requests.post(
                OPENROUTER_URL,
                headers=headers,
                json=payload,
                timeout=60
            )

            # A 429 can be temporary, so retry it instead of disabling
            # OpenRouter for every remaining video frame.
            if api_response.status_code == 429:
                if attempt < 2:
                    retry_after = api_response.headers.get("Retry-After", "5")

                    try:
                        wait_seconds = min(float(retry_after), 10)
                    except ValueError:
                        wait_seconds = 5

                    print(
                        "OpenRouter rate limit reached. "
                        f"Retrying in {wait_seconds:g} seconds..."
                    )
                    time.sleep(wait_seconds)
                    continue

            api_response.raise_for_status()

            response_data = api_response.json()
            response_text = response_data["choices"][0]["message"]["content"]

            if not response_text:
                raise ValueError("OpenRouter returned an empty response.")

            last_moderation_model = "OpenRouter (free router)"
            print("OpenRouter free vision model used.")
            return response_text

        except (requests.RequestException, KeyError, ValueError) as error:
            if attempt < 2:
                print("OpenRouter request failed:", error)
                print("Retrying OpenRouter in 3 seconds...")
                time.sleep(3)
            else:
                raise RuntimeError(
                    f"OpenRouter failed after 3 attempts: {error}"
                ) from error

    raise RuntimeError("OpenRouter rate limit continued after 3 attempts.")

def ask_moderation_model(file_path, prompt):
    """Use Gemini first, then keep using OpenRouter if Gemini is unavailable."""

    global gemini_available
    global last_moderation_model

    if gemini_available:
        try:
            response_text = ask_gemini(file_path, prompt)
            last_moderation_model = "Gemini"
            print("Moderation model used: Gemini")
            return response_text

        except Exception as gemini_error:
            # Avoid repeating predictable failures for every video frame.
            gemini_available = False

            print("\nGemini is unavailable for the rest of this run.")
            print("Reason:", gemini_error)
            print("Switching to OpenRouter fallback...\n")

    else:
        print("Gemini previously failed; using OpenRouter directly.")

    return ask_openrouter(file_path, prompt)


# Extracted from original notebook cell 8.
_NUDENET_DETECTOR = None
_NUDENET_LOCK = threading.RLock()


def _get_nudenet_detector():
    """Initialize NudeNet once, only when NSFW inference is required."""

    global _NUDENET_DETECTOR
    if _NUDENET_DETECTOR is None:
        with _NUDENET_LOCK:
            if _NUDENET_DETECTOR is None:
                _NUDENET_DETECTOR = NudeDetector()
    return _NUDENET_DETECTOR

# Extracted from original notebook cell 9.
def extract_text(image_path):

    image = PIL_IMAGE_OPEN(image_path)

    text = pytesseract.image_to_string(image)

    return text.strip()

# Extracted from original notebook cell 10.
MODERATION_RULES = {
    "Drugs": {
        "keywords": ["weed", "cocaine", "meth", "heroin", "marijuana"],
        "action": "PENDING_REVIEW"
    },
    "Adult": {
        "keywords": ["porn", "onlyfans", "escort", "sex"],
        "action": "PENDING_REVIEW"
    },
    "Violence": {
        "keywords": ["kill", "bomb", "terrorist", "murder"],
        "action": "PENDING_REVIEW"
    },
    "Sensitive Information": {
        "keywords": [
            "password", "credit card", "social security number",
            "bank account", "private key"
        ],
        "action": "PENDING_REVIEW"
    }
}


# Extracted from original notebook cell 11.
def normalize_keyword_text(text):
    """Normalize Unicode and whitespace without changing policy phrases."""

    normalized = unicodedata.normalize("NFKC", str(text)).casefold()
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip()


def keyword_phrase_pattern(phrase):
    """Compile one exact word or phrase with safe token boundaries."""

    normalized_phrase = normalize_keyword_text(phrase)
    phrase_tokens = normalized_phrase.split()
    phrase_body = r"\s+".join(re.escape(token) for token in phrase_tokens)
    return re.compile(rf"(?<!\w){phrase_body}(?!\w)", re.IGNORECASE)


def find_configured_phrases(text, phrases):
    """Return configured words/phrases found without substring collisions."""

    normalized_text = normalize_keyword_text(text)
    return [
        phrase
        for phrase in phrases
        if keyword_phrase_pattern(phrase).search(normalized_text)
    ]


def keyword_filter(text):
    """Return the established violation list using production-safe matching."""

    violations = []
    for category, rule in MODERATION_RULES.items():
        matched = find_configured_phrases(text, rule["keywords"])
        if matched:
            violations.append({
                "category": category,
                "matched": matched
            })
    return violations


# Extracted from original notebook cell 13.
NSFW_THRESHOLD = float(os.getenv("NSFW_THRESHOLD", "0.65"))
WEAPON_THRESHOLD = float(os.getenv("WEAPON_THRESHOLD", "0.50"))
VIOLENCE_THRESHOLD = float(os.getenv("VIOLENCE_THRESHOLD", "0.80"))


def validate_detector_calibration():
    """Fail early when environment-based calibration is invalid."""

    confidence_values = {
        "NSFW_THRESHOLD": NSFW_THRESHOLD,
        "WEAPON_THRESHOLD": WEAPON_THRESHOLD,
        "VIOLENCE_THRESHOLD": VIOLENCE_THRESHOLD
    }
    for name, value in confidence_values.items():
        if not 0 <= value <= 1:
            raise ValueError(f"{name} must be between 0 and 1.")
def summarize_frame_evidence(
    frame_results,
    threshold,
    is_video
):
    """Return raw frame statistics without making a confirmation decision."""

    confidences = [
        float(result.get("confidence", 0.0)) for result in frame_results
    ]
    positive_flags = [confidence >= threshold for confidence in confidences]
    positive_frames = sum(positive_flags)
    analyzed_frames = len(frame_results)
    positive_ratio = (
        positive_frames / analyzed_frames if analyzed_frames else 0.0
    )

    maximum_consecutive = 0
    current_consecutive = 0
    for positive in positive_flags:
        current_consecutive = current_consecutive + 1 if positive else 0
        maximum_consecutive = max(maximum_consecutive, current_consecutive)

    highest_confidence = max(confidences, default=0.0)
    return {
        "threshold": threshold,
        "analyzed_frames": analyzed_frames,
        "positive_frames": positive_frames,
        "positive_frame_ratio": round(positive_ratio, 4),
        "max_consecutive_positive_frames": maximum_consecutive,
        "highest_confidence": round(highest_confidence, 4)
    }


validate_detector_calibration()

# Extracted from original notebook cell 15.
# These terms are independent from MODERATION_RULES and can later come
# from a database or a dedicated abusive-language model.
OCR_ABUSIVE_TERMS = (
    "idiot",
    "stupid",
    "moron",
    "loser",
    "hate you",
    "go die"
)


def find_terms_in_text(text, terms):
    """Return configured terms found in text, without duplicates."""

    normalized_text = text.lower()
    return [term for term in terms if term in normalized_text]


def get_structured_ocr_result(image_path):
    """Run the existing OCR function and return structured information."""

    text = extract_text(image_path)
    abusive_terms = find_terms_in_text(text, OCR_ABUSIVE_TERMS)

    return {
        "detected": bool(text),
        "text": text,
        "abusive_language_detected": bool(abusive_terms),
        "abusive_terms": abusive_terms
    }


def get_structured_keyword_result(ocr_result):
    """Adapt the existing keyword filter to one structured result."""

    violations = keyword_filter(ocr_result.get("text", ""))
    matched_keywords = []

    for violation in violations:
        for keyword in violation.get("matched", []):
            if keyword not in matched_keywords:
                matched_keywords.append(keyword)

    return {
        "triggered": bool(matched_keywords),
        "keywords": matched_keywords,
        "violations": violations
    }


def get_structured_gemini_result(model_result=None):
    """Normalize an existing parsed Gemini result without another API call."""

    if not isinstance(model_result, dict):
        return {
            "status": "NOT_RUN",
            "reason": "Gemini did not return a parsed result."
        }

    return {
        "status": model_result.get("status", "PENDING_REVIEW"),
        "reason": model_result.get("reason", "No reason provided."),
        "violations": model_result.get("violations", [])
    }


# Extracted from original notebook cell 17.
DEFAULT_PROHIBITED_NSFW_CLASSES = {
    "FEMALE_BREAST_EXPOSED",
    "FEMALE_GENITALIA_EXPOSED",
    "MALE_GENITALIA_EXPOSED",
    "ANUS_EXPOSED",
    "BUTTOCKS_EXPOSED"
}


def configured_prohibited_nsfw_classes():
    """Return normalized prohibited classes from configuration."""

    configured_classes = os.getenv("PROHIBITED_NSFW_CLASSES", "").strip()
    if not configured_classes:
        return set(DEFAULT_PROHIBITED_NSFW_CLASSES)

    return {
        class_name.strip().upper()
        for class_name in configured_classes.split(",")
        if class_name.strip()
    }


PROHIBITED_NSFW_CLASSES = configured_prohibited_nsfw_classes()


def nudenet_class_name(detection):
    """Normalize NudeNet class fields across supported versions."""

    return str(
        detection.get("class")
        or detection.get("label")
        or "UNKNOWN"
    ).strip().upper()


def filter_nudenet_detections(raw_detections, threshold=NSFW_THRESHOLD):
    """Separate policy violations from safe or low-confidence classes."""

    filtered_detections = []
    ignored_detections = []

    for raw_detection in raw_detections:
        detection = dict(raw_detection)
        class_name = nudenet_class_name(detection)
        confidence = float(detection.get("score", 0.0))
        detection["normalized_class"] = class_name

        if class_name not in PROHIBITED_NSFW_CLASSES:
            ignored_detections.append({
                **detection,
                "ignored_reason": "safe class (not prohibited by policy)"
            })
        elif confidence < threshold:
            ignored_detections.append({
                **detection,
                "ignored_reason": (
                    f"below NSFW threshold ({confidence:.4f} < "
                    f"{threshold:.4f})"
                )
            })
        else:
            filtered_detections.append(detection)

    ignored_classes = sorted({
        detection["normalized_class"]
        for detection in ignored_detections
    })
    ignored_reasons = [
        {
            "class": detection["normalized_class"],
            "reason": detection["ignored_reason"]
        }
        for detection in ignored_detections
    ]
    confidence = max(
        (float(item.get("score", 0.0)) for item in filtered_detections),
        default=0.0
    )

    return {
        "confidence": round(confidence, 4),
        "raw_detections": list(raw_detections),
        "filtered_detections": filtered_detections,
        "ignored_detections": ignored_detections,
        "ignored_classes": ignored_classes,
        "ignored_reasons": ignored_reasons,
        "threshold": threshold,
        "count": len(filtered_detections),
        "detections": filtered_detections
    }


def get_production_nsfw_result(image_path, threshold=NSFW_THRESHOLD):
    """Run NudeNet once, preserve raw output, then apply policy filtering."""

    raw_detections = _get_nudenet_detector().detect(image_path)
    return filter_nudenet_detections(raw_detections, threshold=threshold)


# Preserve the established interface consumed by orchestration and Risk Engine.
get_structured_nsfw_result = get_production_nsfw_result

# Extracted from original notebook cell 18.
# Keeping weights in one configuration makes future tuning straightforward.
RISK_WEIGHTS = {
    "keyword_trigger": 30,
    "nsfw": 80,
    "weapon": 40,
    "violence": 50,
    "ocr_abusive_language": 25,
    "gemini_pending_review": 20
}


def risk_decision_from_score(risk_score):
    """Map a numeric risk score to a moderation decision."""

    if risk_score < 30:
        return "APPROVED"

    if risk_score < 60:
        return "APPROVED_WITH_LOG"

    return "PENDING_REVIEW"


def calculate_risk(
    ocr_result,
    keyword_result,
    nsfw_result,
    weapon_result,
    gemini_result=None,
    violence_result=None
):
    """Combine structured detector outputs into one explainable score."""

    gemini_result = gemini_result or {}
    violence_result = violence_result or {"detected": False}

    triggered_rules = {
        "keyword_trigger": bool(keyword_result.get("triggered", False)),
        "nsfw": bool(nsfw_result.get("detected", False)),
        "weapon": bool(weapon_result.get("detected", False)),
        "violence": bool(violence_result.get("detected", False)),
        "ocr_abusive_language": bool(
            ocr_result.get("abusive_language_detected", False)
        ),
        "gemini_pending_review": (
            gemini_result.get("status") == "PENDING_REVIEW"
        )
    }

    score_breakdown = {
        rule_name: RISK_WEIGHTS[rule_name]
        for rule_name, triggered in triggered_rules.items()
        if triggered
    }

    risk_score = min(sum(score_breakdown.values()), 100)

    return {
        "risk_score": risk_score,
        "decision": risk_decision_from_score(risk_score),
        "score_breakdown": score_breakdown,
        "triggered_rules": [
            rule_name
            for rule_name, triggered in triggered_rules.items()
            if triggered
        ]
    }

# Extracted from original notebook cell 19.
def build_generic_vlm_result(model_result):
    """Attach provider/model metadata without changing the VLM decision schema."""

    result = dict(model_result or {})
    status = result.get("status", "NOT_RUN")
    provider = result.get("provider")
    model_name = result.get("model")

    provider_failure = (
        "moderation_provider_failure"
        in result.get("violations", [])
    )
    if provider_failure:
        provider = "unavailable"
        model_name = None
    elif status == "NOT_RUN":
        provider = provider or "not_run"
        model_name = model_name or None
    elif not provider:
        active_provider = str(
            globals().get("last_moderation_model") or "unknown"
        )
        if active_provider.lower().startswith("gemini"):
            provider = "gemini"
            model_name = model_name or globals().get("MODEL")
        elif active_provider.lower().startswith("openrouter"):
            provider = "openrouter"
            model_name = model_name or globals().get(
                "OPENROUTER_VISION_MODEL", "openrouter/free"
            )
        else:
            provider = active_provider.lower()

    result["provider"] = provider
    result["model"] = model_name
    return result


def build_moderation_report(
    ocr_result,
    keyword_result,
    nsfw_result,
    weapon_result,
    gemini_result,
    risk_result,
    violence_result=None,
    stt_result=None,
    escalated_to_vlm=False,
    escalation_reasons=None,
    merged_text_result=None,
    text_moderation_result=None
):
    """Return one structured record for a dashboard or database."""

    escalation_reasons = escalation_reasons or []
    merged_text_result = merged_text_result or {
        "text": "", "language": "unknown",
        "source_lengths": {"ocr": 0, "stt": 0, "merged": 0}
    }
    text_moderation_result = (
        text_moderation_result or empty_text_moderation_result()
    )
    vlm_result = build_generic_vlm_result(gemini_result)
    final_decision = (
        vlm_result.get("status", "PENDING_REVIEW")
        if escalated_to_vlm
        else risk_result["decision"]
    )

    return {
        "detectors": {
            "ocr": ocr_result,
            "keyword": keyword_result,
            "nsfw": nsfw_result,
            "weapon": weapon_result,
            "violence": violence_result or {"detected": False},
            "stt": stt_result or empty_stt_result(),
            "vlm": vlm_result,
            # Backward compatibility for dashboards and benchmark cells.
            "gemini": vlm_result
        },
        "merged_text": merged_text_result["text"],
        "text_moderation": text_moderation_result,
        "category": text_moderation_result.get("category", "safe"),
        "severity": text_moderation_result.get("severity", "none"),
        "confidence": text_moderation_result.get("confidence", 0.0),
        "violations": text_moderation_result.get("violations", []),
        "text_moderation_reason": text_moderation_result.get("reason", ""),
        "language": merged_text_result.get("language", "unknown"),
        "source_lengths": merged_text_result.get("source_lengths", {}),
        "risk": risk_result,
        "routing": {
            "escalated_to_vlm": escalated_to_vlm,
            "escalation_reasons": escalation_reasons,
            "vlm_decision": vlm_result.get("status", "NOT_RUN"),
            "vlm_provider": vlm_result.get("provider"),
            "vlm_model": vlm_result.get("model")
        },
        "final_decision": final_decision,
        "reason": vlm_result.get("reason", "No reason provided.")
    }


def print_compact_moderation_report(moderation_report):
    """Print a compact human-readable view of a structured report."""

    detectors = moderation_report["detectors"]
    ocr_failed = detectors["ocr"].get("abusive_language_detected", False)
    vlm_result = detectors.get("vlm", detectors.get("gemini", {}))
    vlm_status = vlm_result.get("status", "NOT_RUN")
    routing = moderation_report["routing"]
    stt_result = detectors["stt"]
    text_moderation_result = moderation_report.get("text_moderation", {})

    display_statuses = {
        "OCR": "FAIL" if ocr_failed else "PASS",
        "Keyword": "FAIL" if detectors["keyword"].get("triggered") else "PASS",
        "NSFW": "FAIL" if detectors["nsfw"].get("detected") else "PASS",
        "Weapon": "FAIL" if detectors["weapon"].get("detected") else "PASS",
        "Violence": "FAIL" if detectors["violence"].get("detected") else "PASS",
        "STT": "PASS" if stt_result.get("transcript") else "SKIPPED",
        "VLM": "REVIEW" if vlm_status == "PENDING_REVIEW" else vlm_status
    }

    print("=" * 29)
    for detector_name, display_status in display_statuses.items():
        print(f"{detector_name:.<17} {display_status}")
    print()
    print(f"Risk Score ..... {moderation_report['risk']['risk_score']}")
    print(f"Language.......... {stt_result.get('language', 'unknown')}")
    print(f"Transcript Length. {len(stt_result.get('transcript', ''))} chars")
    print(f"Text Category .... {text_moderation_result.get('category', 'safe')}")
    print(f"Text Severity .... {text_moderation_result.get('severity', 'none')}")
    print(
        "Escalated to VLM ..... "
        + ("YES" if routing["escalated_to_vlm"] else "NO")
    )
    print(f"VLM Provider .... {routing.get('vlm_provider')}")
    print(f"VLM Model ....... {routing.get('vlm_model')}")
    print(f"VLM Decision .... {routing['vlm_decision']}")
    print(f"Reason .... {moderation_report['reason']}")
    print("\nFINAL DECISION:")
    print(moderation_report["final_decision"])
    print("=" * 29)


def print_structured_detector_debug(moderation_report):
    """Print complete structured detector evidence before compact output."""

    detector_names = (
        "ocr",
        "keyword",
        "nsfw",
        "weapon",
        "violence",
        "stt",
        "vlm"
    )
    for detector_name in detector_names:
        print("-" * 36)
        print(f"{detector_name.upper()} RESULT")
        print("-" * 36)
        print(json.dumps(
            moderation_report["detectors"][detector_name],
            indent=2,
            default=str
        ))
    if "temporal_fusion" in moderation_report:
        print("-" * 36)
        print("TEMPORAL FUSION RESULT")
        print("-" * 36)
        print(json.dumps(
            moderation_report["temporal_fusion"], indent=2, default=str
        ))


def print_video_detector_statistics(moderation_report):
    """Print calibrated frame statistics for video moderation."""

    if moderation_report.get("media", {}).get("media_type") != "video":
        return

    for detector_name in ("nsfw", "weapon", "violence"):
        result = moderation_report["detectors"][detector_name]
        print()
        print(detector_name.title())
        print(
            "Positive Frames: "
            f"{result.get('positive_frames', 0)} / "
            f"{result.get('analyzed_frames', 0)}"
        )
        print(
            "Highest Confidence: "
            f"{result.get('highest_confidence', 0.0):.2f}"
        )
        print(f"Threshold: {result.get('threshold', 0.0):.2f}")
        confirmed = "YES" if result.get("detected", False) else "NO"
        print(f"Confirmed: {confirmed}")

# Extracted from original notebook cell 21.
# Default lightweight checkpoint: YOLOv8n trained for pistols and knives.
# Set WEAPON_MODEL_PATH in .env to use your approved .onnx weights later
# without changing the moderation or Risk Engine cells.
DEFAULT_WEAPON_MODEL_URL = (
    "https://huggingface.co/Hadi959/weapon-detection-yolov8/resolve/"
    "1c7397a7f9268ed60611650cdad936e306a3dc04/best.onnx"
)
DEFAULT_WEAPON_MODEL_SHA256 = (
    "e72be0b7574fd8364f562eaf5e5b2d8c08915410a67a8a2e5a982a29ac0659a1"
)
DEFAULT_DEPLOYMENT_MODEL_SHA256 = (
    "1262fbff7bc6e383a8d31fcee108bfd4319b24a0e818c53ba014649a76f839d8"
)
custom_weapon_model_path = os.getenv("WEAPON_MODEL_PATH")
WEAPON_MODEL_PATH = Path(
    custom_weapon_model_path
    or MODERATION_MODEL_DIRECTORY / "weapon_yolov8n_opset21.onnx"
)
WEAPON_IMAGE_SIZE = 640
WEAPON_DEFAULT_THRESHOLD = WEAPON_THRESHOLD

_production_weapon_model = None
_production_weapon_device = None


def calculate_file_sha256(file_path):
    """Calculate a file checksum without loading the whole file into RAM."""

    sha256 = hashlib.sha256()

    with open(file_path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            sha256.update(chunk)

    return sha256.hexdigest()


def ensure_default_weapon_weights():
    """Download and verify the pinned default weights only when required."""

    if custom_weapon_model_path:
        if WEAPON_MODEL_PATH.suffix.lower() != ".onnx":
            raise ValueError("WEAPON_MODEL_PATH must point to an ONNX model.")

        if not WEAPON_MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Custom weapon model not found: {WEAPON_MODEL_PATH}"
            )
        return WEAPON_MODEL_PATH

    if WEAPON_MODEL_PATH.exists():
        current_hash = calculate_file_sha256(WEAPON_MODEL_PATH)
        if current_hash == DEFAULT_DEPLOYMENT_MODEL_SHA256:
            return WEAPON_MODEL_PATH

        raise ValueError(
            "The cached weapon model checksum is invalid. "
            "Remove only that file and run this cell again."
        )

    WEAPON_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    source_path = WEAPON_MODEL_PATH.with_suffix(".source.onnx")
    temporary_path = WEAPON_MODEL_PATH.with_suffix(".download")

    try:
        with requests.get(
            DEFAULT_WEAPON_MODEL_URL,
            stream=True,
            timeout=60
        ) as response:
            response.raise_for_status()

            with open(source_path, "wb") as file:
                for chunk in response.iter_content(1024 * 1024):
                    if chunk:
                        file.write(chunk)

        downloaded_hash = calculate_file_sha256(source_path)
        if downloaded_hash != DEFAULT_WEAPON_MODEL_SHA256:
            raise ValueError("Downloaded weapon model checksum is invalid.")

        # The public export uses experimental opset 22. Convert it to the
        # officially supported opset 21 for broad ONNX Runtime compatibility.
        source_model = onnx.load(str(source_path))
        deployment_model = onnx.version_converter.convert_version(
            source_model,
            21
        )
        onnx.save(deployment_model, str(temporary_path))

        deployment_hash = calculate_file_sha256(temporary_path)
        if deployment_hash != DEFAULT_DEPLOYMENT_MODEL_SHA256:
            raise ValueError("Converted weapon model checksum is invalid.")

        temporary_path.replace(WEAPON_MODEL_PATH)

    finally:
        if temporary_path.exists():
            temporary_path.unlink()
        if source_path.exists():
            source_path.unlink()

    return WEAPON_MODEL_PATH


def load_production_weapon_detector():
    """Load the YOLO detector once and keep it in memory."""

    global _production_weapon_model, _production_weapon_device

    if _production_weapon_model is None:
        weights_path = ensure_default_weapon_weights()
        cuda_provider_available = (
            "CUDAExecutionProvider" in ort.get_available_providers()
        )
        _production_weapon_device = (
            0
            if torch.cuda.is_available() and cuda_provider_available
            else "cpu"
        )
        _production_weapon_model = YOLO(str(weights_path), task="detect")

        device_name = "CUDA" if _production_weapon_device == 0 else "CPU"
        print(f"Production YOLO weapon detector ready on {device_name}.")

    return _production_weapon_model, _production_weapon_device

# Extracted from original notebook cell 22.
def normalize_weapon_label(raw_label):
    """Normalize model labels while preserving meaningful class detail."""

    label = str(raw_label).strip().lower().replace("_", " ")
    label_aliases = {
        "guns": "gun",
        "pistols": "handgun",
        "pistol": "handgun"
    }
    return label_aliases.get(label, label)


def detect_weapons(
    image_path,
    confidence_threshold=WEAPON_DEFAULT_THRESHOLD
):
    """Detect weapons with YOLO and return the Risk Engine contract."""

    if not 0 <= confidence_threshold <= 1:
        raise ValueError("confidence_threshold must be between 0 and 1.")

    if not Path(image_path).is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    model, device = load_production_weapon_detector()
    prediction = model.predict(
        source=str(image_path),
        conf=confidence_threshold,
        imgsz=WEAPON_IMAGE_SIZE,
        device=device,
        verbose=False
    )[0]

    detections = []

    if prediction.boxes is not None:
        boxes = prediction.boxes.xyxy.detach().cpu().tolist()
        confidence_scores = prediction.boxes.conf.detach().cpu().tolist()
        class_indexes = prediction.boxes.cls.detach().cpu().tolist()

        for box, confidence, class_index in zip(
            boxes,
            confidence_scores,
            class_indexes
        ):
            raw_label = model.names[int(class_index)]
            detections.append({
                "label": normalize_weapon_label(raw_label),
                "confidence": round(float(confidence), 4),
                "box": [round(float(value), 2) for value in box]
            })

    detections.sort(
        key=lambda detection: detection["confidence"],
        reverse=True
    )
    highest_confidence = (
        detections[0]["confidence"] if detections else 0.0
    )

    return {
        "confidence": highest_confidence,
        "threshold": confidence_threshold,
        "count": len(detections),
        "detections": detections
    }


def get_structured_weapon_result(
    image_path,
    threshold=WEAPON_DEFAULT_THRESHOLD
):
    """Keep the existing Risk Engine interface backed by production YOLO."""

    return detect_weapons(
        image_path,
        confidence_threshold=threshold
    )


def get_production_weapon_model_path():
    """Return the verified ONNX path for ONNX Runtime or TensorRT."""

    return str(ensure_default_weapon_weights())

# Extracted from original notebook cell 24.
# Override this in .env when your own fine-tuned ONNX weights are ready.
DEFAULT_VIOLENCE_MODEL_URL = (
    "https://raw.githubusercontent.com/banitalebi/"
    "MobileNetV3-Violence-Detection/main/mobilenetv3_model.onnx"
)
DEFAULT_VIOLENCE_MODEL_SHA256 = (
    "0117f87d93c13720b369afa66eb20c2bbf737bad927c5a5ba3580b8cfda35e35"
)
custom_violence_model_path = os.getenv("VIOLENCE_MODEL_PATH")
VIOLENCE_MODEL_PATH = Path(
    custom_violence_model_path
    or MODERATION_MODEL_DIRECTORY / "violence_mobilenetv3.onnx"
)
VIOLENCE_INPUT_SIZE = (224, 224)
VIOLENCE_DEFAULT_THRESHOLD = VIOLENCE_THRESHOLD

_violence_session = None
_violence_input_name = None
_violence_provider = None


def ensure_violence_model():
    """Download and verify default ONNX weights only when required."""

    if VIOLENCE_MODEL_PATH.suffix.lower() != ".onnx":
        raise ValueError("VIOLENCE_MODEL_PATH must point to an ONNX model.")

    if custom_violence_model_path:
        if not VIOLENCE_MODEL_PATH.is_file():
            raise FileNotFoundError(
                f"Custom violence model not found: {VIOLENCE_MODEL_PATH}"
            )
        return VIOLENCE_MODEL_PATH

    if VIOLENCE_MODEL_PATH.is_file():
        if (
            calculate_file_sha256(VIOLENCE_MODEL_PATH)
            == DEFAULT_VIOLENCE_MODEL_SHA256
        ):
            return VIOLENCE_MODEL_PATH

        raise ValueError("Cached violence model checksum is invalid.")

    VIOLENCE_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = VIOLENCE_MODEL_PATH.with_suffix(".download")

    try:
        with requests.get(
            DEFAULT_VIOLENCE_MODEL_URL,
            stream=True,
            timeout=60
        ) as response:
            response.raise_for_status()

            with open(temporary_path, "wb") as file:
                for chunk in response.iter_content(1024 * 1024):
                    if chunk:
                        file.write(chunk)

        if (
            calculate_file_sha256(temporary_path)
            != DEFAULT_VIOLENCE_MODEL_SHA256
        ):
            raise ValueError("Downloaded violence model checksum is invalid.")

        temporary_path.replace(VIOLENCE_MODEL_PATH)

    finally:
        if temporary_path.exists():
            temporary_path.unlink()

    return VIOLENCE_MODEL_PATH


def select_violence_providers():
    """Prefer TensorRT/CUDA when installed and otherwise use CPU."""

    available_providers = ort.get_available_providers()

    if "TensorrtExecutionProvider" in available_providers:
        return [
            "TensorrtExecutionProvider",
            "CUDAExecutionProvider",
            "CPUExecutionProvider"
        ]

    if "CUDAExecutionProvider" in available_providers:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]

    return ["CPUExecutionProvider"]


def load_violence_detector():
    """Load one ONNX session and keep it in memory for future frames."""

    global _violence_session, _violence_input_name, _violence_provider

    if _violence_session is None:
        model_path = ensure_violence_model()
        providers = select_violence_providers()
        _violence_session = ort.InferenceSession(
            str(model_path),
            providers=providers
        )
        _violence_input_name = _violence_session.get_inputs()[0].name
        _violence_provider = _violence_session.get_providers()[0]
        print(f"Violence detector ready with {_violence_provider}.")

    return _violence_session, _violence_input_name

# Extracted from original notebook cell 25.
def preprocess_violence_frame(image_path):
    """Prepare one frame using the model's documented preprocessing."""

    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    image = cv2.resize(image, VIOLENCE_INPUT_SIZE)
    image = image.astype(np.float32) / 255.0
    return np.expand_dims(image, axis=0)


def violence_category_from_context(weapon_result=None):
    """Use existing weapon evidence to refine a violent-event category."""

    weapon_result = weapon_result or {}
    weapon_labels = {
        detection.get("label", "").lower()
        for detection in weapon_result.get("detections", [])
    }

    if weapon_labels.intersection({"handgun", "gun", "rifle", "shotgun"}):
        return "shooting"

    if "knife" in weapon_labels:
        return "stabbing"

    return "physical_fight"


def violence_severity(confidence, category):
    """Map model confidence and event context to an initial severity."""

    if category in {"shooting", "stabbing"} or confidence >= 0.95:
        return "high"

    if confidence >= VIOLENCE_DEFAULT_THRESHOLD:
        return "medium"

    return "none"


def classify_violence_frame(
    image_path,
    confidence_threshold=VIOLENCE_DEFAULT_THRESHOLD,
    weapon_result=None
):
    """Classify one image or extracted video frame."""

    if not 0 <= confidence_threshold <= 1:
        raise ValueError("confidence_threshold must be between 0 and 1.")

    session, input_name = load_violence_detector()
    model_input = preprocess_violence_frame(image_path)
    model_output = session.run(None, {input_name: model_input})[0]
    probabilities = np.asarray(model_output).reshape(-1)

    if probabilities.size != 2:
        raise ValueError("Violence model must return two class scores.")

    # The default model returns [non_violence, violence] probabilities.
    violence_confidence = float(probabilities[1])
    category = violence_category_from_context(weapon_result)

    return {
        "confidence": round(violence_confidence, 4),
        "threshold": confidence_threshold,
        "category": category,
        "severity": violence_severity(violence_confidence, category)
    }


def aggregate_violence_results(frame_results, is_video=False):
    """Aggregate raw violence evidence without confirming an event."""

    statistics = summarize_frame_evidence(
        frame_results,
        threshold=VIOLENCE_THRESHOLD,
        is_video=is_video
    )
    highest_result = max(
        frame_results,
        key=lambda result: result.get("confidence", 0.0),
        default={}
    )
    return {
        "confidence": statistics["highest_confidence"],
        "category": highest_result.get("category"),
        "severity": highest_result.get("severity", "none"),
        "frame_results": frame_results,
        **statistics
    }


def detect_violence(
    file_path,
    frame_paths=None,
    confidence_threshold=VIOLENCE_DEFAULT_THRESHOLD,
    weapon_result=None
):
    """Detect violence in an image or aggregate extracted video frames."""

    video_extensions = (".mp4", ".avi", ".mov", ".mkv", ".webm")
    is_video_input = str(file_path).lower().endswith(video_extensions)

    if is_video_input:
        frames_to_analyze = frame_paths or extract_frames(file_path)
    else:
        frames_to_analyze = [file_path]

    frame_results = [
        classify_violence_frame(
            frame_path,
            confidence_threshold=confidence_threshold,
            weapon_result=weapon_result
        )
        for frame_path in frames_to_analyze
    ]

    return aggregate_violence_results(
        frame_results,
        is_video=is_video_input
    )


def get_structured_violence_result(
    file_path,
    frame_paths=None,
    weapon_result=None
):
    """Expose the structure consumed by the existing Risk Engine."""

    return detect_violence(
        file_path,
        frame_paths=frame_paths,
        weapon_result=weapon_result
    )


# Extracted from original notebook cell 27.
import importlib.util

if importlib.util.find_spec("faster_whisper") is None:
    raise ImportError(
        "faster-whisper is required. Install project dependencies before "
        "starting the notebook kernel."
    )

print("faster-whisper dependency ready")


# Extracted from original notebook cell 28.
import math
import warnings


STT_MODEL_SIZE = os.getenv("STT_MODEL_SIZE", "small")
STT_AUDIO_EXTENSIONS = (".wav", ".mp3", ".m4a", ".flac", ".ogg")
STT_VIDEO_EXTENSIONS = (".mp4", ".mov", ".avi", ".mkv", ".webm")
STT_SUPPORTED_EXTENSIONS = STT_AUDIO_EXTENSIONS + STT_VIDEO_EXTENSIONS

_STT_MODEL = None
_STT_MODEL_CONFIGURATION = None


def empty_stt_result():
    """Return the stable STT schema used for skipped or failed inputs."""

    return {
        "transcript": "",
        "language": "unknown",
        "confidence": 0.0,
        "duration_seconds": 0.0,
        "segments": []
    }


def get_stt_model():
    """Load faster-whisper once and reuse it for the kernel lifetime."""

    global _STT_MODEL
    global _STT_MODEL_CONFIGURATION

    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"
    configuration = (STT_MODEL_SIZE, device, compute_type)

    if _STT_MODEL is None or _STT_MODEL_CONFIGURATION != configuration:
        from faster_whisper import WhisperModel

        _STT_MODEL = WhisperModel(
            STT_MODEL_SIZE,
            device=device,
            compute_type=compute_type
        )
        _STT_MODEL_CONFIGURATION = configuration
        print(
            f"STT model '{STT_MODEL_SIZE}' ready on {device.upper()} "
            f"with {compute_type}."
        )

    return _STT_MODEL


def extract_audio_for_stt(video_path):
    """Extract mono 16 kHz audio, or safely fall back to direct decoding."""

    ffmpeg_path = shutil.which("ffmpeg")
    if not ffmpeg_path:
        warnings.warn(
            "ffmpeg is unavailable; faster-whisper will decode the video "
            "directly when its bundled decoder supports the format.",
            RuntimeWarning
        )
        return str(video_path), None

    temporary_audio = tempfile.NamedTemporaryFile(
        suffix=".wav",
        delete=False,
        dir=MODERATION_TEMP_DIRECTORY
    )
    temporary_audio.close()

    try:
        subprocess.run(
            [
                ffmpeg_path,
                "-y",
                "-i",
                str(video_path),
                "-vn",
                "-ac",
                "1",
                "-ar",
                "16000",
                temporary_audio.name
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        return temporary_audio.name, temporary_audio.name
    except Exception:
        Path(temporary_audio.name).unlink(missing_ok=True)
        warnings.warn(
            "ffmpeg audio extraction failed; attempting direct video decoding.",
            RuntimeWarning
        )
        return str(video_path), None


def transcribe_media(file_path):
    """Return structured local transcription without raising pipeline errors."""

    media_path = Path(file_path)
    if media_path.suffix.lower() not in STT_SUPPORTED_EXTENSIONS:
        return empty_stt_result()

    transcription_path = str(media_path)
    temporary_audio_path = None

    try:
        if media_path.suffix.lower() in STT_VIDEO_EXTENSIONS:
            transcription_path, temporary_audio_path = extract_audio_for_stt(
                media_path
            )

        model = get_stt_model()
        segment_iterator, transcription_info = model.transcribe(
            transcription_path,
            beam_size=5,
            vad_filter=True
        )

        segments = []
        confidence_values = []
        for segment in segment_iterator:
            text = segment.text.strip()
            segments.append({
                "start": round(float(segment.start), 3),
                "end": round(float(segment.end), 3),
                "text": text
            })
            confidence_values.append(
                max(0.0, min(1.0, math.exp(float(segment.avg_logprob))))
            )

        transcript = " ".join(
            segment["text"] for segment in segments if segment["text"]
        ).strip()
        duration = float(
            getattr(transcription_info, "duration", 0.0) or 0.0
        )

        return {
            "transcript": transcript,
            "language": getattr(
                transcription_info,
                "language",
                "unknown"
            ) or "unknown",
            "confidence": round(
                sum(confidence_values) / len(confidence_values),
                4
            ) if confidence_values else 0.0,
            "duration_seconds": round(duration, 3),
            "segments": segments
        }
    except Exception as error:
        warnings.warn(f"Speech-to-text failed safely: {error}", RuntimeWarning)
        return empty_stt_result()
    finally:
        if temporary_audio_path:
            Path(temporary_audio_path).unlink(missing_ok=True)


# Extracted from original notebook cell 30.
TEXT_MODERATION_MODEL = os.getenv("TEXT_MODERATION_MODEL", "openai/gpt-4.1-mini")
TEXT_MODERATION_TIMEOUT = float(os.getenv("TEXT_MODERATION_TIMEOUT", "30"))
TEXT_MODERATION_TEMPERATURE = float(os.getenv("TEXT_MODERATION_TEMPERATURE", "0"))
TEXT_MODERATION_MAX_TOKENS = int(os.getenv("TEXT_MODERATION_MAX_TOKENS", "350"))
TEXT_MODERATION_RETRIES = int(os.getenv("TEXT_MODERATION_RETRIES", "2"))
TEXT_MODERATION_CONFIDENCE_THRESHOLD = float(
    os.getenv("TEXT_MODERATION_CONFIDENCE_THRESHOLD", "0.70")
)
TEXT_MODERATION_CATEGORIES = {
    "SAFE", "THREAT", "VIOLENCE", "HARASSMENT", "BULLYING",
    "HATE_SPEECH", "EXTREMISM", "TERRORISM", "SCAM",
    "BLACKMAIL", "EXTORTION", "ILLEGAL_ACTIVITY", "DRUG_SALES",
    "SEXUAL_SOLICITATION", "CHILD_SAFETY", "SELF_HARM",
    "SUICIDE_ENCOURAGEMENT", "PERSONAL_INFORMATION_REQUEST",
    "PROFANITY", "TOXICITY", "UNAVAILABLE"
}
DEFAULT_TEXT_MODERATION_WEIGHTS = {
    "THREAT": 50, "VIOLENCE": 45, "HARASSMENT": 30,
    "BULLYING": 30, "HATE_SPEECH": 40, "EXTREMISM": 60,
    "TERRORISM": 60, "SCAM": 35, "BLACKMAIL": 45,
    "EXTORTION": 45, "ILLEGAL_ACTIVITY": 40, "DRUG_SALES": 45,
    "SEXUAL_SOLICITATION": 45, "CHILD_SAFETY": 70,
    "SELF_HARM": 50, "SUICIDE_ENCOURAGEMENT": 60,
    "PERSONAL_INFORMATION_REQUEST": 30, "PROFANITY": 10,
    "TOXICITY": 20
}


def configured_text_moderation_weights():
    """Load category weights from one optional JSON environment value."""

    configured = os.getenv("TEXT_MODERATION_WEIGHTS", "").strip()
    if not configured:
        return dict(DEFAULT_TEXT_MODERATION_WEIGHTS)
    parsed = json.loads(configured)
    return {str(name).upper(): int(value) for name, value in parsed.items()}


TEXT_MODERATION_WEIGHTS = configured_text_moderation_weights()


def merge_extracted_text(ocr_result, stt_result):
    """Merge non-empty OCR and transcript evidence without changing either result."""

    ocr_text = str(ocr_result.get("text", "")).strip()
    transcript = str(stt_result.get("transcript", "")).strip()
    merged_text = "\n".join(part for part in (ocr_text, transcript) if part)
    return {
        "text": merged_text,
        "language": stt_result.get("language", "unknown"),
        "source_lengths": {
            "ocr": len(ocr_text),
            "stt": len(transcript),
            "merged": len(merged_text)
        }
    }


def empty_text_moderation_result(reason="No text was available for moderation."):
    """Return the stable semantic moderation schema for empty input."""

    return {
        "detected": False, "category": "safe", "severity": "none",
        "confidence": 1.0, "reason": reason, "violations": [],
        "available": True
    }


def normalize_text_moderation_result(model_result):
    """Validate and normalize the OpenRouter JSON response."""

    category = str(model_result.get("category", "SAFE")).upper()
    if category not in TEXT_MODERATION_CATEGORIES:
        category = "UNAVAILABLE"
    confidence = max(0.0, min(1.0, float(model_result.get("confidence", 0.0))))
    detected = bool(model_result.get("detected", category not in {"SAFE", "UNAVAILABLE"}))
    detected = detected and confidence >= TEXT_MODERATION_CONFIDENCE_THRESHOLD
    violations = [str(item).upper() for item in model_result.get("violations", [])]
    return {
        "detected": detected,
        "category": category.lower(),
        "severity": str(model_result.get("severity", "none")).lower(),
        "confidence": round(confidence, 4),
        "reason": str(model_result.get("reason", "No reason provided.")),
        "violations": violations if detected else [],
        "available": category != "UNAVAILABLE"
    }


def moderate_text_semantically(merged_text):
    """Moderate merged text through OpenRouter with deterministic JSON output."""

    if not str(merged_text).strip():
        return empty_text_moderation_result()
    if not OPENROUTER_API_KEY:
        return {
            "detected": False, "category": "unavailable",
            "severity": "none", "confidence": 0.0,
            "reason": "Semantic text moderation provider is unavailable.",
            "violations": [], "available": False
        }

    system_prompt = (
        "You are a production content-moderation classifier. Analyze intent and context, "
        "not keyword presence alone. Return one JSON object only with keys detected, "
        "category, severity, confidence, reason, violations. Category must be one of: "
        + ", ".join(sorted(TEXT_MODERATION_CATEGORIES - {"UNAVAILABLE"}))
        + ". Severity must be none, low, medium, high, or critical."
    )
    payload = {
        "model": TEXT_MODERATION_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": str(merged_text)}
        ],
        "response_format": {"type": "json_object"},
        "temperature": TEXT_MODERATION_TEMPERATURE,
        "max_tokens": TEXT_MODERATION_MAX_TOKENS,
        "seed": 0
    }
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    for attempt in range(TEXT_MODERATION_RETRIES + 1):
        try:
            response = requests.post(
                OPENROUTER_URL, headers=headers, json=payload,
                timeout=TEXT_MODERATION_TIMEOUT
            )
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"]
            return normalize_text_moderation_result(json.loads(content))
        except (requests.RequestException, KeyError, ValueError, TypeError, json.JSONDecodeError):
            if attempt < TEXT_MODERATION_RETRIES:
                time.sleep(min(2 ** attempt, 4))
    return {
        "detected": False, "category": "unavailable",
        "severity": "none", "confidence": 0.0,
        "reason": "Semantic text moderation provider is unavailable.",
        "violations": [], "available": False
    }


def apply_text_moderation_risk(risk_result, text_moderation_result):
    """Adapt semantic moderation to the existing Risk Engine result contract."""

    adapted = dict(risk_result)
    category = str(text_moderation_result.get("category", "safe")).upper()
    weight = TEXT_MODERATION_WEIGHTS.get(category, 0) if text_moderation_result.get("detected") else 0
    breakdown = dict(adapted.get("score_breakdown", {}))
    triggered_rules = list(adapted.get("triggered_rules", []))
    if weight:
        rule_name = f"semantic_text_{category.lower()}"
        breakdown[rule_name] = weight
        if rule_name not in triggered_rules:
            triggered_rules.append(rule_name)
    score = min(sum(breakdown.values()), 100)
    adapted.update({
        "risk_score": score, "decision": risk_decision_from_score(score),
        "score_breakdown": breakdown, "triggered_rules": triggered_rules
    })
    return adapted


# Extracted from original notebook cell 33.
VLM_RISK_THRESHOLD = 60
VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".webm")
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
AUDIO_EXTENSIONS = STT_AUDIO_EXTENSIONS

MEDIA_MAX_IMAGE_BYTES = int(
    os.getenv("MEDIA_MAX_IMAGE_BYTES", str(25 * 1024 * 1024))
)
MEDIA_MAX_VIDEO_BYTES = int(
    os.getenv("MEDIA_MAX_VIDEO_BYTES", str(2 * 1024 * 1024 * 1024))
)
MEDIA_MAX_AUDIO_BYTES = int(
    os.getenv("MEDIA_MAX_AUDIO_BYTES", str(250 * 1024 * 1024))
)
MEDIA_MAX_VIDEO_DURATION_SECONDS = float(
    os.getenv("MEDIA_MAX_VIDEO_DURATION_SECONDS", "600")
)
MEDIA_ALLOWED_VIDEO_CODECS = {
    codec.strip().lower()
    for codec in os.getenv("MEDIA_ALLOWED_VIDEO_CODECS", "").split(",")
    if codec.strip()
}


class MediaValidationError(ValueError):
    """Expose structured rejection evidence to a future API layer."""

    def __init__(self, validation_result):
        self.validation_result = validation_result
        reasons = "; ".join(validation_result.get("errors", []))
        super().__init__(f"Media validation failed: {reasons}")


def detect_file_signature(file_path):
    """Identify supported media containers using stable leading bytes."""

    with open(file_path, "rb") as media_file:
        header = media_file.read(32)

    if header.startswith(b"\xff\xd8\xff"):
        return "jpeg"
    if header.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    if header.startswith(b"BM"):
        return "bmp"
    if header.startswith(b"RIFF") and header[8:12] == b"WEBP":
        return "webp"
    if header.startswith(b"RIFF") and header[8:12] == b"AVI ":
        return "avi"
    if len(header) >= 8 and header[4:8] == b"ftyp":
        return "iso_bmff"
    if header.startswith(b"\x1aE\xdf\xa3"):
        return "ebml"
    if header.startswith(b"RIFF") and header[8:12] == b"WAVE":
        return "wav"
    if header.startswith(b"fLaC"):
        return "flac"
    if header.startswith(b"OggS"):
        return "ogg"
    if header.startswith(b"ID3") or (
        len(header) >= 2 and header[0] == 0xFF and header[1] & 0xE0 == 0xE0
    ):
        return "mp3"
    return "unknown"


def decoded_fourcc(capture):
    """Return a readable OpenCV codec code when the container exposes one."""

    fourcc_value = int(capture.get(cv2.CAP_PROP_FOURCC))
    if not fourcc_value:
        return "unknown"
    return "".join(
        chr((fourcc_value >> (8 * index)) & 0xFF)
        for index in range(4)
    ).strip().lower() or "unknown"


def validate_media_file(file_path):
    """Return structured validation before any AI detector is called."""

    path = Path(file_path).expanduser()
    extension = path.suffix.lower()
    result = {
        "valid": False,
        "file_path": str(path),
        "media_type": "unsupported",
        "extension": extension,
        "mime_type": None,
        "signature": None,
        "size_bytes": 0,
        "duration_seconds": None,
        "codec": None,
        "errors": [],
        "warnings": []
    }

    if not path.is_file():
        result["errors"].append("file does not exist")
        return result

    image_extensions = set(IMAGE_EXTENSIONS)
    video_extensions = set(VIDEO_EXTENSIONS)
    audio_extensions = set(AUDIO_EXTENSIONS)
    if extension in image_extensions:
        media_type = "image"
        maximum_bytes = MEDIA_MAX_IMAGE_BYTES
    elif extension in video_extensions:
        media_type = "video"
        maximum_bytes = MEDIA_MAX_VIDEO_BYTES
    elif extension in audio_extensions:
        media_type = "audio"
        maximum_bytes = MEDIA_MAX_AUDIO_BYTES
    else:
        result["errors"].append("unsupported file extension")
        return result

    result["media_type"] = media_type
    result["size_bytes"] = path.stat().st_size
    result["mime_type"] = mimetypes.guess_type(path.name)[0]
    result["signature"] = detect_file_signature(path)

    if result["size_bytes"] <= 0:
        result["errors"].append("file is empty")
    elif result["size_bytes"] > maximum_bytes:
        result["errors"].append(
            f"file exceeds configured {media_type} size limit"
        )

    mime_type = result["mime_type"] or ""
    if media_type == "image" and not mime_type.startswith("image/"):
        result["errors"].append("extension does not resolve to an image MIME type")
    elif media_type == "video" and not mime_type.startswith("video/"):
        result["errors"].append("extension does not resolve to a video MIME type")
    elif media_type == "audio" and not mime_type.startswith("audio/"):
        result["errors"].append("extension does not resolve to an audio MIME type")

    expected_image_signatures = {
        ".jpg": "jpeg", ".jpeg": "jpeg", ".png": "png",
        ".bmp": "bmp", ".webp": "webp"
    }
    expected_video_signatures = {
        ".mp4": {"iso_bmff"}, ".mov": {"iso_bmff"},
        ".avi": {"avi"}, ".mkv": {"ebml"}, ".webm": {"ebml"}
    }
    expected_audio_signatures = {
        ".wav": {"wav"}, ".mp3": {"mp3"}, ".flac": {"flac"},
        ".ogg": {"ogg"}, ".m4a": {"iso_bmff"}
    }

    if media_type == "image":
        if result["signature"] != expected_image_signatures[extension]:
            result["errors"].append("image signature does not match extension")
        try:
            with PIL_IMAGE_OPEN(path) as image:
                image.verify()
            with PIL_IMAGE_OPEN(path) as image:
                image.load()
            if cv2.imread(str(path)) is None:
                result["errors"].append("OpenCV could not decode image")
        except Exception as error:
            result["errors"].append(
                f"image decoder rejected file: {type(error).__name__}"
            )

    elif media_type == "video":
        if result["signature"] not in expected_video_signatures[extension]:
            result["errors"].append("video signature does not match extension")
        capture = cv2.VideoCapture(str(path))
        try:
            if not capture.isOpened():
                result["errors"].append("OpenCV could not open video")
            else:
                frame_available, first_frame = capture.read()
                if not frame_available or first_frame is None:
                    result["errors"].append("video has no readable first frame")
                frames_per_second = float(capture.get(cv2.CAP_PROP_FPS))
                total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
                if frames_per_second <= 0 or total_frames <= 0:
                    result["errors"].append("video duration metadata is unreadable")
                else:
                    duration = total_frames / frames_per_second
                    result["duration_seconds"] = round(duration, 3)
                    if duration > MEDIA_MAX_VIDEO_DURATION_SECONDS:
                        result["errors"].append(
                            "video exceeds configured maximum duration"
                        )
                result["codec"] = decoded_fourcc(capture)
                if (
                    MEDIA_ALLOWED_VIDEO_CODECS
                    and result["codec"] not in MEDIA_ALLOWED_VIDEO_CODECS
                ):
                    result["errors"].append(
                        f"unsupported video codec: {result['codec']}"
                    )
        finally:
            capture.release()

    else:
        if result["signature"] not in expected_audio_signatures[extension]:
            result["errors"].append("audio signature does not match extension")
        if not shutil.which("ffprobe"):
            result["warnings"].append(
                "ffprobe unavailable; audio decoding will be validated by STT"
            )

    result["valid"] = not result["errors"]
    return result


def require_valid_media(file_path):
    """Return validation evidence or reject before detector execution."""

    validation_result = validate_media_file(file_path)
    if not validation_result["valid"]:
        raise MediaValidationError(validation_result)
    return validation_result


def select_media_file():
    """Return a configured file or use the legacy local upload dialog."""

    test_file = os.getenv("MODERATION_TEST_FILE")
    if test_file:
        if not Path(test_file).is_file():
            raise FileNotFoundError(
                f"MODERATION_TEST_FILE does not exist: {test_file}"
            )
        return str(Path(test_file))

    automated = os.getenv("MODERATION_AUTOMATED", "0") == "1"
    if automated:
        raise RuntimeError(
            "Automated execution requires MODERATION_TEST_FILE."
        )

    try:
        from tkinter import Tk
        from tkinter.filedialog import askopenfilename
    except ImportError as error:
        raise RuntimeError(
            "The local file dialog is unavailable. Set MODERATION_TEST_FILE."
        ) from error

    root = Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    try:
        selected_file = askopenfilename(
            title="Select Image, Video, or Audio",
            filetypes=[
                (
                    "Media Files",
                    "*.jpg *.jpeg *.png *.bmp *.webp "
                    "*.mp4 *.avi *.mov *.mkv *.webm "
                    "*.wav *.mp3 *.m4a *.flac *.ogg"
                ),
                ("All Files", "*.*")
            ]
        )
    finally:
        root.destroy()

    if not selected_file:
        raise ValueError("No media file was selected.")
    return selected_file


def detect_media_type(file_path):
    """Return image/video type from the selected file extension."""

    lower_path = str(file_path).lower()
    if lower_path.endswith(VIDEO_EXTENSIONS):
        return "video"
    if lower_path.endswith(AUDIO_EXTENSIONS):
        return "audio"
    if lower_path.endswith(IMAGE_EXTENSIONS):
        return "image"
    raise ValueError(f"Unsupported media type: {file_path}")


def prepare_media(file_path, media_type):
    """Return the image or representative frames used by local detectors."""

    if media_type == "image":
        return [file_path]
    if media_type == "audio":
        return []

    frame_paths = extract_frames(
        file_path,
        sample_every_seconds=2,
        max_frames=8
    )
    if not frame_paths:
        raise ValueError("No frames could be extracted from the video.")
    return frame_paths


def aggregate_ocr(frame_results):
    """Aggregate structured OCR across image/video frames."""

    combined_text = "\n".join(
        result["text"] for result in frame_results if result["text"]
    )
    abusive_terms = sorted({
        term
        for result in frame_results
        for term in result["abusive_terms"]
    })
    return {
        "detected": bool(combined_text),
        "text": combined_text,
        "abusive_language_detected": bool(abusive_terms),
        "abusive_terms": abusive_terms
    }


def aggregate_weapons(frame_results, is_video=False):
    """Aggregate raw weapon frame evidence without confirmation."""

    detections = [
        detection
        for result in frame_results
        for detection in result["detections"]
    ]
    statistics = summarize_frame_evidence(
        frame_results,
        threshold=WEAPON_THRESHOLD,
        is_video=is_video
    )
    return {
        "confidence": statistics["highest_confidence"],
        "count": len(detections),
        "detections": detections,
        "frame_results": frame_results,
        **statistics
    }


def parse_model_json(response_text):
    """Parse a JSON-only VLM response and fail closed on invalid output."""

    cleaned_response = response_text.strip()
    if cleaned_response.startswith("```json"):
        cleaned_response = cleaned_response[7:]
    elif cleaned_response.startswith("```"):
        cleaned_response = cleaned_response[3:]
    if cleaned_response.endswith("```"):
        cleaned_response = cleaned_response[:-3]

    result = json.loads(cleaned_response.strip())
    status = result.get("status")
    if status not in {"APPROVED", "PENDING_REVIEW"}:
        raise ValueError(f"Invalid moderation status: {status}")
    return get_structured_gemini_result(result)


def build_moderation_prompt(
    ocr_result,
    nsfw_result,
    weapon_result,
    violence_result
):
    """Build one consistent VLM prompt using local detector context."""

    return f"""
You are a strict production content moderation system.

Analyze the complete visual scene.

Local detector context:
- OCR text: {ocr_result["text"] or "No text detected"}
- NSFW detected: {nsfw_result["detected"]}
- Weapon detected: {weapon_result["detected"]}
- Violence detected: {violence_result["detected"]}

Return PENDING_REVIEW for weapons, violence, blood, injury, nudity,
explicit sexual content, illegal drugs, hate symbols, extremist content,
or credible threats. Treat uncertain realistic weapons as review-worthy.

Return only one JSON object:
{{
    "status": "APPROVED or PENDING_REVIEW",
    "reason": "Short evidence-based reason",
    "violations": ["specific policy categories"]
}}
"""


# Extracted from original notebook cell 34.
def aggregate_production_nsfw(frame_results, is_video=False):
    """Aggregate only policy-filtered NSFW evidence across frames."""

    raw_detections = [
        detection
        for result in frame_results
        for detection in result.get("raw_detections", [])
    ]
    filtered_detections = [
        detection
        for result in frame_results
        for detection in result.get("filtered_detections", [])
    ]
    ignored_detections = [
        detection
        for result in frame_results
        for detection in result.get("ignored_detections", [])
    ]
    statistics = summarize_frame_evidence(
        frame_results,
        threshold=NSFW_THRESHOLD,
        is_video=is_video
    )
    ignored_classes = sorted({
        detection.get("normalized_class", "UNKNOWN")
        for detection in ignored_detections
    })

    return {
        "confidence": statistics["highest_confidence"],
        "raw_detections": raw_detections,
        "filtered_detections": filtered_detections,
        "ignored_detections": ignored_detections,
        "ignored_classes": ignored_classes,
        "ignored_reasons": [
            {
                "class": detection.get("normalized_class", "UNKNOWN"),
                "reason": detection.get("ignored_reason", "ignored")
            }
            for detection in ignored_detections
        ],
        "count": len(filtered_detections),
        "detections": filtered_detections,
        "frame_results": frame_results,
        **statistics
    }


# Replace aggregation binding without changing its public call signature.
aggregate_nsfw = aggregate_production_nsfw

# Extracted from original notebook cell 38.
VIDEO_SHORT_DURATION_SECONDS = float(
    os.getenv("VIDEO_SHORT_DURATION_SECONDS", "10")
)
VIDEO_MEDIUM_DURATION_SECONDS = float(
    os.getenv("VIDEO_MEDIUM_DURATION_SECONDS", "30")
)
VIDEO_LONG_DURATION_SECONDS = float(
    os.getenv("VIDEO_LONG_DURATION_SECONDS", "60")
)
VIDEO_SHORT_FRAME_INTERVAL = int(
    os.getenv("VIDEO_SHORT_FRAME_INTERVAL", "3")
)
VIDEO_MEDIUM_FRAME_INTERVAL = int(
    os.getenv("VIDEO_MEDIUM_FRAME_INTERVAL", "5")
)
VIDEO_LONG_FRAME_INTERVAL = int(
    os.getenv("VIDEO_LONG_FRAME_INTERVAL", "10")
)
VIDEO_MAX_FRAME_BUDGET = int(
    os.getenv("VIDEO_MAX_FRAME_BUDGET", "120")
)

ADAPTIVE_VIDEO_SAMPLING_METADATA = {}


def validate_video_sampling_configuration():
    """Validate duration boundaries, intervals, and frame budget."""

    durations = (
        VIDEO_SHORT_DURATION_SECONDS,
        VIDEO_MEDIUM_DURATION_SECONDS,
        VIDEO_LONG_DURATION_SECONDS
    )
    if not 0 < durations[0] < durations[1] < durations[2]:
        raise ValueError("Video duration thresholds must be increasing.")

    intervals = (
        VIDEO_SHORT_FRAME_INTERVAL,
        VIDEO_MEDIUM_FRAME_INTERVAL,
        VIDEO_LONG_FRAME_INTERVAL,
        VIDEO_MAX_FRAME_BUDGET
    )
    if any(value < 1 for value in intervals):
        raise ValueError("Video sampling intervals and budget must be positive.")


def select_adaptive_sampling_policy(video_duration, total_frames):
    """Return a frame interval and descriptive strategy for a video."""

    if video_duration < VIDEO_SHORT_DURATION_SECONDS:
        frame_interval = VIDEO_SHORT_FRAME_INTERVAL
        strategy = f"adaptive_every_{frame_interval}_frames"
    elif video_duration < VIDEO_MEDIUM_DURATION_SECONDS:
        frame_interval = VIDEO_MEDIUM_FRAME_INTERVAL
        strategy = f"adaptive_every_{frame_interval}_frames"
    elif video_duration <= VIDEO_LONG_DURATION_SECONDS:
        frame_interval = VIDEO_LONG_FRAME_INTERVAL
        strategy = f"adaptive_every_{frame_interval}_frames"
    else:
        budget_interval = math.ceil(total_frames / VIDEO_MAX_FRAME_BUDGET)
        frame_interval = max(VIDEO_LONG_FRAME_INTERVAL, budget_interval)
        strategy = f"adaptive_budget_every_{frame_interval}_frames"

    estimated_frames = math.ceil(total_frames / frame_interval)
    return {
        "frame_interval": frame_interval,
        "sampling_strategy": strategy,
        "estimated_sampled_frames": estimated_frames
    }


def extract_frames_adaptive(
    video_path,
    output_folder=None,
    sample_every_seconds=None,
    max_frames=None
):
    """Extract duration-aware frames and cache structured sampling metadata."""

    video_path = str(video_path)
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    try:
        frames_per_second = float(capture.get(cv2.CAP_PROP_FPS))
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if frames_per_second <= 0 or total_frames <= 0:
            raise ValueError("Could not read video FPS or frame count.")

        video_duration = total_frames / frames_per_second
        policy = select_adaptive_sampling_policy(
            video_duration,
            total_frames
        )
        frame_interval = policy["frame_interval"]

        path_fingerprint = hashlib.sha256(
            str(Path(video_path).resolve()).encode("utf-8")
        ).hexdigest()[:12]
        frame_root = Path(output_folder or MODERATION_FRAME_DIRECTORY)
        frame_directory = frame_root / (
            f"{Path(video_path).stem}_adaptive_{path_fingerprint}"
        )
        frame_directory.mkdir(parents=True, exist_ok=True)

        sampled_paths = []
        frame_index = 0
        while True:
            frame_available, frame = capture.read()
            if not frame_available:
                break

            if frame_index % frame_interval == 0:
                frame_path = frame_directory / f"frame_{frame_index:08d}.jpg"
                if not cv2.imwrite(str(frame_path), frame):
                    raise ValueError(f"Could not write sampled frame: {frame_path}")
                sampled_paths.append(str(frame_path))
            frame_index += 1
    finally:
        capture.release()

    if not sampled_paths:
        raise ValueError("Adaptive sampling did not produce any frames.")

    metadata = {
        "video_duration": round(video_duration, 3),
        "total_frames": total_frames,
        "sampled_frames": len(sampled_paths),
        "sampling_strategy": policy["sampling_strategy"],
        "frame_interval": frame_interval,
        "frames_per_second": round(frames_per_second, 3),
        "maximum_frame_budget": VIDEO_MAX_FRAME_BUDGET
    }
    ADAPTIVE_VIDEO_SAMPLING_METADATA[str(Path(video_path).resolve())] = metadata
    return sampled_paths


validate_video_sampling_configuration()

# Preserve the interface used by preparation, detectors, and VLM routing.
extract_frames = extract_frames_adaptive

# Extracted from original notebook cell 43.
TEMPORAL_MIN_EVENT_FRAMES = int(
    os.getenv("TEMPORAL_MIN_EVENT_FRAMES", "4")
)
TEMPORAL_MIN_EVENT_DURATION = float(
    os.getenv("TEMPORAL_MIN_EVENT_DURATION", "1.0")
)
TEMPORAL_MIN_AVERAGE_CONFIDENCE = float(
    os.getenv("TEMPORAL_MIN_AVERAGE_CONFIDENCE", "0.80")
)
TEMPORAL_PEAK_CONFIRMATION_CONFIDENCE = float(
    os.getenv("TEMPORAL_PEAK_CONFIRMATION_CONFIDENCE", "0.95")
)
TEMPORAL_UNCERTAIN_MIN_CONFIDENCE = float(
    os.getenv("TEMPORAL_UNCERTAIN_MIN_CONFIDENCE", "0.60")
)
TEMPORAL_AVERAGE_CONFIDENCE_MIN_FRAMES = int(
    os.getenv("TEMPORAL_AVERAGE_CONFIDENCE_MIN_FRAMES", "3")
)
TEMPORAL_PEAK_CONFIDENCE_MIN_FRAMES = int(
    os.getenv("TEMPORAL_PEAK_CONFIDENCE_MIN_FRAMES", "2")
)
TEMPORAL_MIN_POSITIVE_FRAME_RATIO = float(
    os.getenv("TEMPORAL_MIN_POSITIVE_FRAME_RATIO", "0.02")
)
TEMPORAL_CONFIDENCE_BOOST_PER_SUPPORT = float(
    os.getenv("TEMPORAL_CONFIDENCE_BOOST_PER_SUPPORT", "0.05")
)
IMAGE_CONFIRM_THRESHOLD = float(
    os.getenv("IMAGE_CONFIRM_THRESHOLD", "0.85")
)
IMAGE_UNCERTAIN_THRESHOLD = float(
    os.getenv("IMAGE_UNCERTAIN_THRESHOLD", "0.60")
)
TEMPORAL_THREAT_KEYWORDS = tuple(
    term.strip().lower()
    for term in os.getenv(
        "TEMPORAL_THREAT_KEYWORDS",
        "kill,shoot,stab,murder,hurt,bomb"
    ).split(",")
    if term.strip()
)


def validate_temporal_policy_configuration():
    """Fail early when image decision thresholds are invalid."""

    if not (
        0.0 <= IMAGE_UNCERTAIN_THRESHOLD
        < IMAGE_CONFIRM_THRESHOLD
        <= 1.0
    ):
        raise ValueError(
            "Require 0 <= IMAGE_UNCERTAIN_THRESHOLD < "
            "IMAGE_CONFIRM_THRESHOLD <= 1."
        )


validate_temporal_policy_configuration()


def frame_timestamp(frame_path, frame_position, sampling_metadata):
    """Return a sampled frame timestamp using its original frame index."""

    frames_per_second = float(
        sampling_metadata.get("frames_per_second", 0.0) or 0.0
    )
    frame_interval = int(sampling_metadata.get("frame_interval", 1) or 1)
    original_index = frame_position * frame_interval

    if frame_path:
        final_token = Path(frame_path).stem.rsplit("_", 1)[-1]
        if final_token.isdigit():
            original_index = int(final_token)

    if frames_per_second <= 0:
        return float(frame_position)
    return original_index / frames_per_second


def transcript_contains_threat(stt_result):
    """Return threat cues for detector agreement, not a moderation verdict."""

    transcript = stt_result.get("transcript", "").lower()
    return [term for term in TEMPORAL_THREAT_KEYWORDS if term in transcript]


def timed_threat_windows(stt_result):
    """Return timestamped STT segments containing configured threat cues."""

    windows = []
    for segment in stt_result.get("segments", []):
        text = segment.get("text", "").lower()
        terms = [term for term in TEMPORAL_THREAT_KEYWORDS if term in text]
        if terms:
            windows.append({
                "start_time": float(segment.get("start", 0.0)),
                "end_time": float(segment.get("end", 0.0)),
                "terms": terms,
                "source": "stt_threat"
            })
    return windows


def timed_ocr_windows(ocr_result, frame_paths, sampling_metadata):
    """Return timestamped OCR frames containing threat cues."""

    windows = []
    frame_interval = int(sampling_metadata.get("frame_interval", 1) or 1)
    frames_per_second = float(
        sampling_metadata.get("frames_per_second", 0.0) or 0.0
    )
    sample_duration = (
        frame_interval / frames_per_second if frames_per_second > 0 else 0.0
    )
    for position, result in enumerate(ocr_result.get("frame_results", [])):
        text = result.get("text", "").lower()
        terms = [term for term in TEMPORAL_THREAT_KEYWORDS if term in text]
        if not terms:
            continue
        start_time = frame_timestamp(
            frame_paths[position] if position < len(frame_paths) else None,
            position,
            sampling_metadata
        )
        windows.append({
            "start_time": start_time,
            "end_time": start_time + sample_duration,
            "terms": terms,
            "source": "ocr_threat"
        })
    return windows


def group_positive_frame_events(
    event_type,
    detector_result,
    frame_paths,
    sampling_metadata,
    media_type
):
    """Group consecutive positive sampled frames into candidate events."""

    frame_results = detector_result.get("frame_results", [])
    threshold = float(detector_result.get("threshold", 0.0) or 0.0)

    if media_type == "image":
        confidence = float(detector_result.get("confidence", 0.0))
        if confidence > 0.0:
            return [{
                "type": event_type,
                "start_time": 0.0,
                "end_time": 0.0,
                "duration": 0.0,
                "frame_count": 1,
                "average_confidence": confidence,
                "peak_confidence": confidence,
                "confidence": confidence,
                "supporting_detectors": [event_type],
                "evidence_scope": "image"
            }]
        return []

    candidates = []
    current_group = []
    for position, result in enumerate(frame_results):
        confidence = float(result.get("confidence", 0.0))
        if confidence >= threshold and threshold > 0:
            current_group.append((position, confidence))
        elif current_group:
            candidates.append(current_group)
            current_group = []
    if current_group:
        candidates.append(current_group)

    frame_interval = int(sampling_metadata.get("frame_interval", 1) or 1)
    frames_per_second = float(
        sampling_metadata.get("frames_per_second", 0.0) or 0.0
    )
    sample_duration = (
        frame_interval / frames_per_second if frames_per_second > 0 else 0.0
    )
    events = []
    for group in candidates:
        positions = [item[0] for item in group]
        confidences = [item[1] for item in group]
        start_position = positions[0]
        end_position = positions[-1]
        start_time = frame_timestamp(
            frame_paths[start_position] if start_position < len(frame_paths) else None,
            start_position,
            sampling_metadata
        )
        final_frame_time = frame_timestamp(
            frame_paths[end_position] if end_position < len(frame_paths) else None,
            end_position,
            sampling_metadata
        )
        end_time = final_frame_time + sample_duration
        frame_count = len(group)
        events.append({
            "type": event_type,
            "start_time": round(start_time, 3),
            "end_time": round(end_time, 3),
            "duration": round(max(0.0, end_time - start_time), 3),
            "frame_count": frame_count,
            "average_confidence": round(sum(confidences) / frame_count, 4),
            "peak_confidence": round(max(confidences), 4),
            "confidence": round(sum(confidences) / frame_count, 4),
            "supporting_detectors": [event_type],
            "evidence_scope": "video"
        })
    return events


def temporal_events_overlap(first_event, second_event):
    """Return whether two detector events overlap in time."""

    return (
        first_event["start_time"] <= second_event["end_time"]
        and second_event["start_time"] <= first_event["end_time"]
    )


def event_severity(event, threat_detected):
    """Estimate event severity from detector and transcript agreement."""

    support = set(event["supporting_detectors"])
    if (
        {"weapon", "violence"}.issubset(support)
        and "stt_threat" in support
    ):
        return "critical"
    if {"weapon", "violence"}.issubset(support):
        return "high"
    return "medium"


def fuse_temporal_events(
    ocr_result,
    keyword_result,
    stt_result,
    nsfw_result,
    weapon_result,
    violence_result,
    media_type,
    frame_paths,
    sampling_metadata,
    risk_weights
):
    """Classify evidence as confirmed, uncertain, or suppressed events."""

    detector_results = {
        "nsfw": nsfw_result,
        "weapon": weapon_result,
        "violence": violence_result
    }
    all_events = []
    for detector_name, detector_result in detector_results.items():
        all_events.extend(group_positive_frame_events(
            detector_name,
            detector_result,
            frame_paths,
            sampling_metadata,
            media_type
        ))

    threat_terms = transcript_contains_threat(stt_result)
    threat_detected = bool(threat_terms)
    stt_threat_windows = timed_threat_windows(stt_result)
    ocr_threat_windows = timed_ocr_windows(
        ocr_result,
        frame_paths,
        sampling_metadata
    )

    for event in all_events:
        analyzed_frames = int(
            detector_results[event["type"]].get("analyzed_frames", 0)
            or len(detector_results[event["type"]].get("frame_results", []))
            or 1
        )
        event["positive_frame_ratio"] = round(
            event["frame_count"] / analyzed_frames,
            4
        )

        for other_event in all_events:
            if event is other_event or event["type"] == other_event["type"]:
                continue
            if temporal_events_overlap(event, other_event):
                other_type = other_event["type"]
                if other_type not in event["supporting_detectors"]:
                    event["supporting_detectors"].append(other_type)

        for threat_window in stt_threat_windows + ocr_threat_windows:
            if temporal_events_overlap(event, threat_window):
                source = threat_window["source"]
                if source not in event["supporting_detectors"]:
                    event["supporting_detectors"].append(source)

        if media_type == "image":
            confidence = event["peak_confidence"]
            event["duration_seconds"] = 0.0
            if confidence >= IMAGE_CONFIRM_THRESHOLD:
                reason = "image confidence reached confirmation threshold"
                event["decision"] = "CONFIRMED"
                event["confirmation_reasons"] = [
                    "image_confidence_threshold_reached"
                ]
                event["suppression_reason"] = None
                event["uncertainty_reason"] = None
            elif confidence >= IMAGE_UNCERTAIN_THRESHOLD:
                reason = "image confidence requires VLM review"
                event["decision"] = "UNCERTAIN"
                event["confirmation_reasons"] = []
                event["suppression_reason"] = None
                event["uncertainty_reason"] = reason
            else:
                reason = "image confidence below uncertainty threshold"
                event["decision"] = "SUPPRESSED"
                event["confirmation_reasons"] = []
                event["suppression_reason"] = reason
                event["uncertainty_reason"] = None
            event["reason"] = reason
            continue

        sustained_frames = event["frame_count"] >= TEMPORAL_MIN_EVENT_FRAMES
        sustained_duration = event["duration"] >= TEMPORAL_MIN_EVENT_DURATION
        strong_average = (
            event["average_confidence"] >= TEMPORAL_MIN_AVERAGE_CONFIDENCE
            and event["frame_count"] >= TEMPORAL_AVERAGE_CONFIDENCE_MIN_FRAMES
        )
        strong_peak = (
            event["peak_confidence"] >= TEMPORAL_PEAK_CONFIRMATION_CONFIDENCE
            and event["frame_count"] >= TEMPORAL_PEAK_CONFIDENCE_MIN_FRAMES
        )
        support = set(event["supporting_detectors"])
        weapon_violence_overlap = (
            event["type"] in {"weapon", "violence"}
            and {"weapon", "violence"}.issubset(support)
        )
        weapon_threat_overlap = (
            event["type"] == "weapon" and "stt_threat" in support
        )
        confirmation_reasons = []
        if sustained_frames:
            confirmation_reasons.append("minimum_consecutive_frames_reached")
        if sustained_duration:
            confirmation_reasons.append("minimum_event_duration_reached")
        if strong_average:
            confirmation_reasons.append("strong_average_confidence")
        if strong_peak:
            confirmation_reasons.append("strong_peak_confidence")
        if weapon_violence_overlap:
            confirmation_reasons.append("weapon_violence_overlap")
        if weapon_threat_overlap:
            confirmation_reasons.append("weapon_timed_threat_overlap")

        event["duration_seconds"] = event["duration"]
        event["confirmation_reasons"] = confirmation_reasons
        if confirmation_reasons:
            event["decision"] = "CONFIRMED"
            event["reason"] = "; ".join(confirmation_reasons)
            event["suppression_reason"] = None
            event["uncertainty_reason"] = None
        elif event["peak_confidence"] >= TEMPORAL_UNCERTAIN_MIN_CONFIDENCE:
            uncertainty_reasons = []
            if event["frame_count"] < TEMPORAL_MIN_EVENT_FRAMES:
                uncertainty_reasons.append("limited consecutive evidence")
            if event["duration"] < TEMPORAL_MIN_EVENT_DURATION:
                uncertainty_reasons.append("short transient event")
            if len(event["supporting_detectors"]) == 1:
                uncertainty_reasons.append("single detector evidence")
            event["decision"] = "UNCERTAIN"
            event["reason"] = "; ".join(uncertainty_reasons)
            event["suppression_reason"] = None
            event["uncertainty_reason"] = event["reason"]
        else:
            event["decision"] = "SUPPRESSED"
            event["reason"] = "isolated low-confidence detector evidence"
            event["suppression_reason"] = event["reason"]
            event["uncertainty_reason"] = None

    confirmed_events = [
        event for event in all_events if event["decision"] == "CONFIRMED"
    ]
    uncertain_events = [
        event for event in all_events if event["decision"] == "UNCERTAIN"
    ]
    suppressed_events = [
        event for event in all_events if event["decision"] == "SUPPRESSED"
    ]

    for event in confirmed_events + uncertain_events:
        additional_support = max(0, len(event["supporting_detectors"]) - 1)
        event["confidence"] = round(min(
            1.0,
            event["confidence"]
            + additional_support * TEMPORAL_CONFIDENCE_BOOST_PER_SUPPORT
        ), 4)
        event["severity"] = event_severity(event, threat_detected)

    agreement = [
        {
            "event_type": event["type"],
            "supporting_detectors": event["supporting_detectors"]
        }
        for event in confirmed_events + uncertain_events
        if len(event["supporting_detectors"]) > 1
    ]
    disagreement = [
        {
            "event_type": event["type"],
            "reason": event["reason"],
            "decision": event["decision"]
        }
        for event in uncertain_events + suppressed_events
    ]
    confirmed_types = sorted({event["type"] for event in confirmed_events})
    overall_confidence = max(
        (
            event["confidence"]
            for event in confirmed_events + uncertain_events
        ),
        default=0.0
    )

    if confirmed_events:
        event_descriptions = [
            f"{event['type']} for {event['duration']:.2f}s "
            f"({event['severity']})"
            for event in confirmed_events
        ]
        summary = "Confirmed " + "; ".join(event_descriptions) + "."
    elif uncertain_events:
        summary = "Visual evidence requires VLM reasoning: " + "; ".join(
            f"{event['type']} ({event['reason']})"
            for event in uncertain_events
        ) + "."
    elif suppressed_events:
        summary = "Only isolated or unstable detector evidence was suppressed."
    elif threat_detected:
        summary = "Transcript threat cue found; no visual temporal event confirmed."
    else:
        summary = "No temporal moderation events confirmed."

    timeline = sorted(
        [dict(event) for event in all_events],
        key=lambda event: event["start_time"]
    )

    return {
        "media_type": media_type,
        "decision_policy": (
            "image_confidence"
            if media_type == "image"
            else "video_temporal"
        ),
        "confirmed_events": confirmed_events,
        "uncertain_events": uncertain_events,
        "suppressed_events": suppressed_events,
        "event_counts": {
            "confirmed": len(confirmed_events),
            "uncertain": len(uncertain_events),
            "suppressed": len(suppressed_events)
        },
        "detector_agreement": agreement,
        "detector_disagreement": disagreement,
        "timeline": timeline,
        "overall_confidence": round(overall_confidence, 4),
        "confidence_summary": {
            "overall": round(overall_confidence, 4),
            "confirmed_peak": round(max(
                (event["confidence"] for event in confirmed_events),
                default=0.0
            ), 4),
            "uncertain_peak": round(max(
                (event["confidence"] for event in uncertain_events),
                default=0.0
            ), 4)
        },
        "confirmed_detector_list": confirmed_types,
        "threat_transcript_detected": threat_detected,
        "threat_terms": threat_terms,
        "risk_weights": dict(risk_weights),
        "ocr_detected": bool(ocr_result.get("detected", False)),
        "keyword_triggered": bool(keyword_result.get("triggered", False)),
        "sampling_metadata": dict(sampling_metadata),
        "summary": summary
    }


def adapt_detectors_to_confirmed_events(
    nsfw_result,
    weapon_result,
    violence_result,
    temporal_result
):
    """Expose only confirmed temporal event types to risk routing."""

    confirmed_events = temporal_result["confirmed_events"]
    adapted_results = {}
    for detector_name, detector_result in {
        "nsfw": nsfw_result,
        "weapon": weapon_result,
        "violence": violence_result
    }.items():
        matching_events = [
            event for event in confirmed_events
            if event["type"] == detector_name
        ]
        adapted_result = dict(detector_result)
        adapted_result["raw_positive_frames"] = int(
            detector_result.get("positive_frames", 0)
        )
        adapted_result["detected"] = bool(matching_events)
        adapted_result["temporal_events"] = matching_events
        if matching_events:
            adapted_result["confidence"] = max(
                event["confidence"] for event in matching_events
            )
        elif media_type_is_video := bool(
            detector_result.get("analyzed_frames", 0) > 1
        ):
            adapted_result["confidence"] = 0.0
            adapted_result["suppressed_by_temporal_fusion"] = media_type_is_video
        adapted_results[detector_name] = adapted_result
    return adapted_results


# Extracted from original notebook cell 44.
def build_temporal_vlm_prompt(
    ocr_result,
    nsfw_result,
    weapon_result,
    violence_result,
    temporal_result,
    keyword_result,
    stt_result,
    risk_result,
    merged_text_result,
    text_moderation_result
):
    """Build the established VLM prompt with temporal context."""

    base_prompt = build_moderation_prompt(
        ocr_result,
        nsfw_result,
        weapon_result,
        violence_result
    )
    temporal_context = json.dumps({
        "confirmed_events": temporal_result["confirmed_events"],
        "uncertain_events": temporal_result["uncertain_events"],
        "suppressed_events": temporal_result["suppressed_events"],
        "detector_agreement": temporal_result["detector_agreement"],
        "detector_disagreement": temporal_result["detector_disagreement"],
        "timeline": temporal_result["timeline"],
        "overall_confidence": temporal_result["overall_confidence"],
        "ocr": ocr_result,
        "transcript": stt_result.get("transcript", ""),
        "keyword_result": keyword_result,
        "merged_text": merged_text_result.get("text", ""),
        "text_moderation": text_moderation_result,
        "detector_confidences": {
            "nsfw": nsfw_result.get("confidence", 0.0),
            "weapon": weapon_result.get("confidence", 0.0),
            "violence": violence_result.get("confidence", 0.0)
        },
        "risk_score": risk_result.get("risk_score", 0),
        "summary": temporal_result["summary"]
    }, indent=2)
    return base_prompt + "\n\nTemporal fusion context:\n" + temporal_context


def moderate_media_with_temporal_vlm(
    media_files,
    ocr_result,
    nsfw_result,
    weapon_result,
    violence_result,
    temporal_result,
    keyword_result,
    stt_result,
    risk_result,
    merged_text_result,
    text_moderation_result
):
    """Run the existing provider router with temporal event context."""

    prompt = build_temporal_vlm_prompt(
        ocr_result,
        nsfw_result,
        weapon_result,
        violence_result,
        temporal_result,
        keyword_result,
        stt_result,
        risk_result,
        merged_text_result,
        text_moderation_result
    )
    approved_result = {
        "status": "APPROVED",
        "reason": "No VLM policy violation detected.",
        "violations": []
    }
    for media_path in media_files:
        try:
            current_result = parse_model_json(
                ask_moderation_model(media_path, prompt)
            )
        except Exception:
            return {
                "status": "PENDING_REVIEW",
                "reason": "High-risk content requires manual review.",
                "violations": ["moderation_provider_failure"]
            }
        if current_result["status"] == "PENDING_REVIEW":
            return current_result
        approved_result = current_result
    return approved_result


def run_production_moderation(file_path):
    """Validate media and return the complete final policy decision."""

    media_validation = require_valid_media(file_path)
    media_type = detect_media_type(file_path)
    media_files = prepare_media(file_path, media_type)

    if media_type == "audio":
        ocr_result = aggregate_ocr([])
    else:
        ocr_frame_results = [
            get_structured_ocr_result(path) for path in media_files
        ]
        ocr_result = aggregate_ocr(ocr_frame_results)
        ocr_result["frame_results"] = ocr_frame_results

    stt_result = transcribe_media(file_path)
    merged_text_result = merge_extracted_text(ocr_result, stt_result)
    merged_text_input = dict(ocr_result)
    merged_text_input["text"] = merged_text_result["text"]
    keyword_result = get_structured_keyword_result(merged_text_input)
    text_moderation_result = moderate_text_semantically(
        merged_text_result["text"]
    )

    if media_type == "audio":
        nsfw_result = aggregate_nsfw([], is_video=False)
        weapon_result = aggregate_weapons([], is_video=False)
        violence_result = {
            "confidence": 0.0,
            "category": "none",
            "severity": "none",
            "frame_results": []
        }
    else:
        nsfw_result = aggregate_nsfw([
            get_structured_nsfw_result(path, threshold=NSFW_THRESHOLD)
            for path in media_files
        ], is_video=media_type == "video")
        weapon_result = aggregate_weapons([
            get_structured_weapon_result(path, threshold=WEAPON_THRESHOLD)
            for path in media_files
        ], is_video=media_type == "video")
        violence_result = get_structured_violence_result(
            file_path,
            frame_paths=media_files if media_type == "video" else None,
            weapon_result=weapon_result
        )

    sampling_metadata = ADAPTIVE_VIDEO_SAMPLING_METADATA.get(
        str(Path(file_path).resolve()),
        {
            "video_duration": 0.0,
            "total_frames": len(media_files),
            "sampled_frames": len(media_files),
            "sampling_strategy": "not_applicable",
            "frame_interval": 1,
            "frames_per_second": 0.0
        }
    )
    temporal_result = fuse_temporal_events(
        ocr_result,
        keyword_result,
        stt_result,
        nsfw_result,
        weapon_result,
        violence_result,
        media_type,
        media_files,
        sampling_metadata,
        RISK_WEIGHTS
    )
    adapted = adapt_detectors_to_confirmed_events(
        nsfw_result,
        weapon_result,
        violence_result,
        temporal_result
    )
    nsfw_result = adapted["nsfw"]
    weapon_result = adapted["weapon"]
    violence_result = adapted["violence"]

    vlm_result = {
        "status": "NOT_RUN",
        "reason": "VLM analysis was not required.",
        "violations": []
    }
    risk_result = calculate_risk(
        ocr_result,
        keyword_result,
        nsfw_result,
        weapon_result,
        gemini_result=vlm_result,
        violence_result=violence_result
    )
    risk_result = apply_text_moderation_risk(
        risk_result, text_moderation_result
    )
    escalation_reasons = [
        f"confirmed_{event_type}_event"
        for event_type in temporal_result["confirmed_detector_list"]
    ]
    if temporal_result["uncertain_events"]:
        escalation_reasons.append("uncertain_temporal_event")
    if not text_moderation_result.get("available", True):
        escalation_reasons.append("text_moderation_unavailable")
    if (
        text_moderation_result.get("detected")
        and text_moderation_result.get("severity") in {"high", "critical"}
    ):
        escalation_reasons.append("high_severity_semantic_text")
    if risk_result["risk_score"] >= VLM_RISK_THRESHOLD:
        escalation_reasons.append("risk_threshold_reached")
    escalation_reasons = list(dict.fromkeys(escalation_reasons))
    escalated_to_vlm = bool(escalation_reasons)

    # Part 1 is strictly optional supplemental context.  It is evaluated only
    # after the existing Part 2 escalation candidate is established, and its
    # result never feeds risk, VLM routing, policy precedence, or final decision.
    part1_ai_result = None
    if escalated_to_vlm and part2_context_integration_enabled():
        part1_ai_result = classify_part2_moderation_context(
            merged_text_result["text"]
        )

    if escalated_to_vlm and media_files:
        vlm_result = moderate_media_with_temporal_vlm(
            media_files,
            ocr_result,
            nsfw_result,
            weapon_result,
            violence_result,
            temporal_result,
            keyword_result,
            stt_result,
            risk_result,
            merged_text_result,
            text_moderation_result
        )
    elif escalated_to_vlm:
        vlm_result = {
            "status": "PENDING_REVIEW",
            "reason": "High-risk content requires manual review.",
            "violations": ["no_visual_media_for_vlm"]
        }

    report = build_moderation_report(
        ocr_result=ocr_result,
        keyword_result=keyword_result,
        nsfw_result=nsfw_result,
        weapon_result=weapon_result,
        violence_result=violence_result,
        stt_result=stt_result,
        gemini_result=vlm_result,
        risk_result=risk_result,
        escalated_to_vlm=escalated_to_vlm,
        escalation_reasons=escalation_reasons,
        merged_text_result=merged_text_result,
        text_moderation_result=text_moderation_result
    )
    if part1_ai_result is not None:
        report["part1_ai"] = part1_ai_result
    report["temporal_fusion"] = temporal_result
    report["media"] = {
        "file_path": file_path,
        "media_type": media_type,
        "analyzed_files": media_files,
        "sampling": dict(sampling_metadata)
    }
    report["media_validation"] = media_validation
    report = apply_policy_reasoning(report)
    report["reason"] = report["policy_reason"]
    return report


# Backward-compatible name; both names resolve to the same function body.
run_moderation_with_temporal_fusion = run_production_moderation

# Extracted from original notebook cell 53.
POLICY_LABELS = {
    "REAL_WORLD_THREAT",
    "HARASSMENT",
    "HATE_SPEECH",
    "SCAM",
    "SELF_HARM",
    "SUICIDE",
    "SUICIDE_ENCOURAGEMENT",
    "SEXUAL_CONTENT",
    "SEXUAL_EXPLOITATION",
    "CHILD_SAFETY",
    "TERRORISM",
    "ILLEGAL_ACTIVITY",
    "CREDIBLE_ILLEGAL_ACTIVITY",
    "MOVIE_DIALOG",
    "GAMEPLAY",
    "ROLEPLAY",
    "LYRICS",
    "QUOTE",
    "EDUCATIONAL",
    "NEWS",
    "COMEDY",
    "SAFE"
}

POLICY_ALWAYS_REVIEW = {
    "REAL_WORLD_THREAT",
    "TERRORISM",
    "CHILD_SAFETY",
    "SUICIDE_ENCOURAGEMENT",
    "CREDIBLE_ILLEGAL_ACTIVITY",
    "SEXUAL_EXPLOITATION"
}

POLICY_CONTEXT_APPROVAL = {
    "MOVIE_DIALOG",
    "GAMEPLAY",
    "ROLEPLAY",
    "LYRICS",
    "NEWS",
    "QUOTE",
    "EDUCATIONAL",
    "COMEDY"
}


def policy_text_from_report(moderation_report):
    """Combine existing textual evidence without running a new detector."""

    detectors = moderation_report.get("detectors", {})
    merged_text = moderation_report.get("merged_text", "")
    if isinstance(merged_text, dict):
        merged_text = merged_text.get("text", "")
    ocr_text = detectors.get("ocr", {}).get("text", "")
    transcript = detectors.get("stt", {}).get("transcript", "")
    keywords = " ".join(detectors.get("keyword", {}).get("keywords", []))
    vlm_result = detectors.get("vlm", detectors.get("gemini", {}))
    vlm_reason = vlm_result.get("reason", "")
    text_moderation = moderation_report.get("text_moderation", {})
    context_hint = moderation_report.get("policy_context_hint", "")
    return " ".join((
        str(merged_text),
        ocr_text,
        transcript,
        keywords,
        vlm_reason,
        json.dumps(text_moderation, default=str),
        context_hint
    )).strip().lower()


def contains_any(text, phrases):
    """Return exact configured words/phrases found in normalized text."""

    return find_configured_phrases(text, phrases)


def detect_policy_context(text):
    """Classify explicit contextual markers after safety evidence is collected."""

    context_rules = (
        ("GAMEPLAY", "fiction", (
            "gameplay", "video game", "call of duty", "fortnite",
            "gta", "gaming stream", "game"
        )),
        ("MOVIE_DIALOG", "fiction", (
            "movie dialogue", "movie scene", "film scene", "in this movie",
            "this movie", "the movie", "screenplay", "scripted", "movie",
            "film", "acting", "drama", "fictional"
        )),
        ("ROLEPLAY", "fiction", (
            "roleplay", "role play", "in character", "fictional role"
        )),
        ("LYRICS", "artistic", (
            "song lyrics", "lyrics", "in this song", "chorus",
            "verse from a song"
        )),
        ("QUOTE", "quoted", (
            "quote", "quoted from", "famous quote", "the character says"
        )),
        ("NEWS", "news", (
            "breaking news", "news report", "reported that",
            "according to police", "journalist", "news coverage"
        )),
        ("EDUCATIONAL", "educational", (
            "educational", "for education", "history lesson",
            "safety training", "documentary", "awareness campaign"
        )),
        ("COMEDY", "fiction", (
            "comedy", "satire", "parody", "stand-up", "just a joke"
        ))
    )
    for policy, context, markers in context_rules:
        matched = contains_any(text, markers)
        if matched:
            return policy, context, matched
    return None, None, []


def strong_detector_disagreement(moderation_report):
    """Return every confirmed detector event that context cannot override."""

    temporal = moderation_report.get("temporal_fusion", {})
    return [
        event
        for event in temporal.get("confirmed_events", [])
        if event.get("type") in {"weapon", "violence", "nsfw"}
    ]


def make_policy_result(decision, policy, severity, confidence, context, reason):
    """Build and validate the strict policy result schema."""

    if decision not in {"APPROVED", "PENDING_REVIEW"}:
        raise ValueError(f"Invalid policy decision: {decision}")
    if policy not in POLICY_LABELS:
        raise ValueError(f"Invalid policy label: {policy}")
    if severity not in {"NONE", "LOW", "MEDIUM", "HIGH", "CRITICAL"}:
        raise ValueError(f"Invalid policy severity: {severity}")
    return {
        "decision": decision,
        "policy": policy,
        "severity": severity,
        "confidence": round(max(0.0, min(1.0, confidence)), 4),
        "context": context,
        "reason": reason
    }


def reason_about_policy(moderation_report):
    """Apply contextual policy rules after temporal fusion and VLM."""

    text = policy_text_from_report(moderation_report)
    context_policy, context, context_markers = detect_policy_context(text)
    strong_events = strong_detector_disagreement(moderation_report)
    detectors = moderation_report.get("detectors", {})
    vlm_result = detectors.get("vlm", detectors.get("gemini", {}))
    vlm_status = vlm_result.get("status", "NOT_RUN")
    text_moderation = moderation_report.get("text_moderation", {})
    high_severity_semantic = (
        text_moderation.get("detected", False)
        and str(text_moderation.get("severity", "none")).lower()
        in {"high", "critical"}
    )

    suicide_encouragement = contains_any(text, (
        "kill yourself", "you should die", "go kill yourself"
    ))
    self_harm = contains_any(text, (
        "i want to die", "i will hurt myself", "self harm",
        "end my life", "suicidal"
    ))
    child_safety = contains_any(text, (
        "sexual content involving a child", "minor exploitation",
        "child sexual abuse", "csam"
    ))
    sexual_exploitation = contains_any(text, (
        "sexual exploitation", "forced prostitution", "sex trafficking"
    ))
    terrorism = contains_any(text, (
        "terrorist attack", "join the terrorist", "build a bomb to attack",
        "bomb the building"
    ))
    credible_illegal = contains_any(text, (
        "bomb manual", "how to build a bomb", "sell illegal drugs",
        "buy illegal weapons", "commit the robbery"
    ))
    credible_threat = contains_any(text, (
        "kill you tomorrow", "at your house", "shoot everyone",
        "going to shoot", "i will find you and kill", "bomb your"
    ))
    generic_threat = contains_any(text, (
        "i will kill you", "i'm going to shoot", "i am going to shoot"
    ))
    harassment = contains_any(text, (
        "i hate you", "you are an idiot", "you are stupid",
        "you are a loser"
    ))
    scam = contains_any(text, (
        "send me your password", "guaranteed investment",
        "claim your prize", "wire the money"
    ))

    if child_safety:
        return make_policy_result(
            "PENDING_REVIEW", "CHILD_SAFETY", "CRITICAL", 0.99,
            "real_world", "Potential child-safety exploitation content."
        )
    if sexual_exploitation:
        return make_policy_result(
            "PENDING_REVIEW", "SEXUAL_EXPLOITATION", "CRITICAL", 0.98,
            "real_world", "Potential sexual exploitation requires review."
        )
    if suicide_encouragement:
        return make_policy_result(
            "PENDING_REVIEW", "SUICIDE_ENCOURAGEMENT", "HIGH", 0.98,
            "real_world", "Language encourages suicide or death."
        )
    if self_harm:
        return make_policy_result(
            "PENDING_REVIEW", "SELF_HARM", "HIGH", 0.94,
            "real_world", "High-severity self-harm language requires review."
        )
    if terrorism:
        return make_policy_result(
            "PENDING_REVIEW", "TERRORISM", "CRITICAL", 0.97,
            "real_world", "Credible terrorism-related intent or instruction."
        )
    if credible_illegal:
        return make_policy_result(
            "PENDING_REVIEW", "CREDIBLE_ILLEGAL_ACTIVITY", "HIGH", 0.92,
            "real_world", "Potentially actionable illegal activity or instruction."
        )

    # Production precedence (highest to lowest):
    # 1. Mandatory safety policies above.
    # 2. Confirmed detector events, high-severity semantic evidence, or VLM review.
    # 3. Benign contextual indicators such as movie/game/news/education.
    # 4. Lower-severity text policies and finally SAFE.
    # Context words are deliberately evaluated only after all safety evidence.
    strong_event_types = {event.get("type") for event in strong_events}
    if "nsfw" in strong_event_types:
        return make_policy_result(
            "PENDING_REVIEW", "SEXUAL_CONTENT", "HIGH", 0.95,
            "safety_evidence", "Confirmed NSFW evidence requires human review."
        )
    if strong_event_types.intersection({"weapon", "violence"}):
        return make_policy_result(
            "PENDING_REVIEW", "REAL_WORLD_THREAT", "HIGH", 0.95,
            "safety_evidence",
            "Confirmed weapon or violence evidence requires human review."
        )
    if high_severity_semantic:
        return make_policy_result(
            "PENDING_REVIEW", "REAL_WORLD_THREAT", "HIGH", 0.95,
            "semantic_evidence",
            "High-severity semantic moderation evidence requires human review."
        )
    if vlm_status == "PENDING_REVIEW":
        return make_policy_result(
            "PENDING_REVIEW", "REAL_WORLD_THREAT", "HIGH", 0.95,
            "vlm_evidence", "The Vision Language Model requires human review."
        )

    if context_policy:
        serious_event_types = strong_event_types
        safety_conflicts = sorted(serious_event_types)
        if high_severity_semantic:
            safety_conflicts.append("high_severity_semantic_text")
        if vlm_status == "PENDING_REVIEW":
            safety_conflicts.append("vlm_pending_review")

        if safety_conflicts:
            if "nsfw" in serious_event_types:
                conflict_policy = "SEXUAL_CONTENT"
            else:
                conflict_policy = "REAL_WORLD_THREAT"
            return make_policy_result(
                "PENDING_REVIEW", conflict_policy, "HIGH", 0.95,
                "context_conflict",
                "Context cannot override safety evidence: "
                + ", ".join(safety_conflicts) + "."
            )
        return make_policy_result(
            "APPROVED", context_policy, "NONE", 0.95, context,
            f"Content is explicitly contextualized as {context.replace('_', ' ')}."
        )

    if credible_threat or generic_threat:
        confidence = 0.98 if credible_threat else 0.86
        return make_policy_result(
            "PENDING_REVIEW", "REAL_WORLD_THREAT", "HIGH", confidence,
            "real_world", "Direct threat lacks a clear fictional or quoted context."
        )
    if scam:
        return make_policy_result(
            "APPROVED", "SCAM", "MEDIUM", 0.84, "real_world",
            "Possible scam language detected; current policy does not auto-review it."
        )
    if harassment:
        return make_policy_result(
            "APPROVED", "HARASSMENT", "LOW", 0.88, "real_world",
            "Low-severity interpersonal harassment without a credible threat."
        )

    if strong_event_types.intersection({"weapon", "violence"}) or vlm_status == "PENDING_REVIEW":
        return make_policy_result(
            "PENDING_REVIEW", "REAL_WORLD_THREAT", "HIGH", 0.82,
            "uncertain", "Strong detector or VLM concern lacks enough policy context."
        )

    return make_policy_result(
        "APPROVED", "SAFE", "NONE", 0.99, "safe",
        "No review-mandatory policy evidence was identified."
    )


def apply_policy_reasoning(moderation_report):
    """Attach policy fields and make policy the final report decision."""

    policy_result = reason_about_policy(moderation_report)
    moderation_report["policy_result"] = policy_result
    moderation_report["policy_label"] = policy_result["policy"]
    moderation_report["context"] = policy_result["context"]
    moderation_report["policy_confidence"] = policy_result["confidence"]
    moderation_report["policy_reason"] = policy_result["reason"]
    moderation_report.setdefault(
        "pre_policy_decision", moderation_report.get("final_decision")
    )
    moderation_report["final_decision"] = policy_result["decision"]
    return moderation_report


# Intentional notebook compatibility surface. Production integrations should
# normally use run_production_moderation rather than individual stages.
__all__ = [
    "AUDIO_EXTENSIONS",
    "IMAGE_EXTENSIONS",
    "VIDEO_EXTENSIONS",
    "MODERATION_TEMP_DIRECTORY",
    "PIL_IMAGE_OPEN",
    "RISK_WEIGHTS",
    "VIDEO_MAX_FRAME_BUDGET",
    "MediaValidationError",
    "adapt_detectors_to_confirmed_events",
    "apply_policy_reasoning",
    "apply_text_moderation_risk",
    "ask_moderation_model",
    "build_generic_vlm_result",
    "build_moderation_report",
    "calculate_file_sha256",
    "calculate_risk",
    "detect_media_type",
    "detect_weapons",
    "filter_nudenet_detections",
    "fuse_temporal_events",
    "get_structured_nsfw_result",
    "get_structured_ocr_result",
    "get_structured_violence_result",
    "get_structured_weapon_result",
    "keyword_filter",
    "moderate_text_semantically",
    "print_compact_moderation_report",
    "print_structured_detector_debug",
    "print_video_detector_statistics",
    "reason_about_policy",
    "run_moderation_with_temporal_fusion",
    "run_production_moderation",
    "select_adaptive_sampling_policy",
    "select_media_file",
    "transcribe_media",
    "validate_media_file",
]
