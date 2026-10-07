"""HTTP entry point for the reusable content moderation pipeline."""

from __future__ import annotations

import os
import tempfile
import uuid
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile

from content_moderation.pipeline import MediaValidationError
from content_moderation.workflow import (
    AdminQueueError,
    ModerationProcessingError,
    moderate_for_backend,
    sanitize_validation_result,
)


app = FastAPI(title="AI Content Moderation Pipeline", version="1.0.0")
MAX_UPLOAD_BYTES = int(os.getenv("API_MAX_UPLOAD_BYTES", str(2 * 1024 * 1024 * 1024)))
CHUNK_BYTES = int(os.getenv("API_UPLOAD_CHUNK_BYTES", str(1024 * 1024)))
TEMP_DIRECTORY = Path(os.getenv("API_TEMP_DIRECTORY") or tempfile.gettempdir())
TEMP_DIRECTORY.mkdir(parents=True, exist_ok=True)


@app.post("/moderate")
def moderate_upload(file: UploadFile = File(...)):
    """Moderate one image, video, or audio upload and delete the temporary file."""

    request_id = str(uuid.uuid4())
    filename = Path(str(file.filename or "upload").replace("\\", "/")).name
    temporary_path = None
    try:
        suffix = Path(filename).suffix.lower() or ".upload"
        with tempfile.NamedTemporaryFile(
            mode="wb", suffix=suffix, prefix="moderation_", dir=TEMP_DIRECTORY,
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            total = 0
            while True:
                chunk = file.file.read(CHUNK_BYTES)
                if not chunk:
                    break
                total += len(chunk)
                if total > MAX_UPLOAD_BYTES:
                    raise HTTPException(status_code=413, detail={"error": "Upload exceeds size limit."})
                temporary.write(chunk)
        return moderate_for_backend(
            str(temporary_path), original_filename=filename,
            content_type=file.content_type, request_id=request_id,
        )
    except HTTPException:
        raise
    except MediaValidationError as error:
        raise HTTPException(
            status_code=422,
            detail={"error": "Media validation failed.", "validation": sanitize_validation_result(error.validation_result)},
        ) from None
    except AdminQueueError:
        raise HTTPException(status_code=503, detail={"error": "Review queue unavailable."}) from None
    except ModerationProcessingError:
        raise HTTPException(status_code=500, detail={"error": "Moderation failed closed."}) from None
    except Exception:
        raise HTTPException(status_code=500, detail={"error": "Internal moderation error."}) from None
    finally:
        try:
            file.file.close()
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
