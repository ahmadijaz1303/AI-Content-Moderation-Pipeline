from unittest.mock import patch

from fastapi.testclient import TestClient

import app


def test_moderate_route_delegates_one_upload_to_workflow():
    with patch.object(app, "moderate_for_backend", return_value={"status": "APPROVED"}) as moderate:
        response = TestClient(app.app).post(
            "/moderate", files={"file": ("sample.jpg", b"image-data", "image/jpeg")},
        )

    assert response.status_code == 200
    assert response.json()["status"] == "APPROVED"
    assert moderate.call_count == 1
