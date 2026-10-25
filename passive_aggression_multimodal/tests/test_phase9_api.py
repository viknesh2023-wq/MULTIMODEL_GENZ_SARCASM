"""
tests/test_phase9_api.py

Automated test suite verifying Phase 9 FastAPI Inference Pipeline:
- Health and metadata endpoints
- Text-only sarcasm prediction
- Multimodal sarcasm prediction with image upload
- Modality fallback when image is missing
- Corrupt image rejection and query fallback policy
- HTTP 503 response on passive-aggression endpoint
- Input validation (empty text, whitespace, length limits, mime types)
- Clean error responses without internal stack traces
"""

from __future__ import annotations

import io
from PIL import Image
import pytest
from fastapi.testclient import TestClient

from src.api.main import app


@pytest.fixture(scope="module")
def client():
    """Initializes FastAPI test client within lifespan context."""
    with TestClient(app) as test_client:
        yield test_client


def create_test_image_bytes(format="JPEG", color="blue", size=(100, 100)) -> bytes:
    """Helper to generate valid test image bytes in-memory."""
    img = Image.new("RGB", size, color=color)
    buf = io.BytesIO()
    img.save(buf, format=format)
    return buf.getvalue()


class TestPhase9SystemEndpoints:
    """Tests for /health and /metadata endpoints."""

    def test_health_endpoint(self, client: TestClient):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] in ["healthy", "degraded"]
        assert "text_tfidf_baseline" in data["loaded_models"]
        assert data["model_status"]["text_tfidf_baseline"] == "ready"
        assert data["passive_aggression_benchmark_status"] == "BLOCKED_PENDING_HUMAN_ANNOTATION"
        assert "timestamp" in data

    def test_metadata_endpoint(self, client: TestClient):
        response = client.get("/metadata")
        assert response.status_code == 200
        data = response.json()
        assert "Context-Aware Multimodal" in data["project_name"]
        assert data["api_version"] == "1.0.0"
        assert "text_tfidf_baseline" in data["available_models"]
        assert "gated_multimodal_fusion" in data["available_models"]
        assert data["tasks_supported"]["text_tfidf_baseline"] == "sarcasm_detection"
        assert data["expected_labels"] == ["non_sarcastic", "sarcastic"]
        assert data["passive_aggression_benchmark_gate"] == "BLOCKED_PENDING_HUMAN_ANNOTATION"
        assert len(data["known_limitations"]) >= 3


class TestPhase9TextSarcasmPrediction:
    """Tests for POST /predict/sarcasm/text endpoint."""

    def test_successful_text_prediction(self, client: TestClient):
        payload = {"text": "Oh wonderful, another delay on my commute home!"}
        response = client.post("/predict/sarcasm/text", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["task"] == "sarcasm_detection"
        assert data["model_id"] == "text_tfidf_baseline"
        assert data["predicted_label"] in ["sarcastic", "non_sarcastic"]
        assert "class_scores" in data
        assert "non_sarcastic" in data["class_scores"]
        assert "sarcastic" in data["class_scores"]
        assert abs(data["class_scores"]["non_sarcastic"] + data["class_scores"]["sarcastic"] - 1.0) < 0.01
        assert data["modality_used"] == ["text"]
        assert data["fallback_applied"] is False
        assert len(data["warnings"]) >= 1

    def test_empty_text_rejection(self, client: TestClient):
        payload = {"text": ""}
        response = client.post("/predict/sarcasm/text", json=payload)
        assert response.status_code == 422
        data = response.json()
        assert "error" in data or "detail" in data

    def test_whitespace_only_text_rejection(self, client: TestClient):
        payload = {"text": "    \n\t   "}
        response = client.post("/predict/sarcasm/text", json=payload)
        assert response.status_code == 422

    def test_excessive_length_text_rejection(self, client: TestClient):
        payload = {"text": "word " * 1500}  # ~7500 chars > 5000 ceiling
        response = client.post("/predict/sarcasm/text", json=payload)
        assert response.status_code == 422

    def test_invalid_json_rejection(self, client: TestClient):
        response = client.post(
            "/predict/sarcasm/text",
            content="invalid-not-json",
            headers={"Content-Type": "application/json"},
        )
        assert response.status_code == 422


class TestPhase9MultimodalPredictionAndFallback:
    """Tests for POST /predict/sarcasm/multimodal and fallback policies."""

    def test_multimodal_with_valid_image(self, client: TestClient):
        img_bytes = create_test_image_bytes(format="JPEG", color="green")
        response = client.post(
            "/predict/sarcasm/multimodal",
            data={"text": "Look at this amazing scenery today."},
            files={"image": ("scenery.jpg", img_bytes, "image/jpeg")},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["task"] == "sarcasm_detection"
        assert data["model_id"] == "gated_multimodal_fusion"
        assert data["modality_used"] == ["text", "image"]
        assert data["fallback_applied"] is False
        assert data["predicted_label"] in ["sarcastic", "non_sarcastic"]
        assert "class_scores" in data

    def test_multimodal_missing_image_triggers_text_fallback(self, client: TestClient):
        response = client.post(
            "/predict/sarcasm/multimodal",
            data={"text": "Just another comment with no image attached."},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["task"] == "sarcasm_detection"
        assert data["model_id"] == "text_tfidf_baseline"
        assert data["modality_used"] == ["text"]
        assert data["fallback_applied"] is True
        assert any("fallback" in w.lower() for w in data["warnings"])

    def test_multimodal_corrupt_image_returns_422_by_default(self, client: TestClient):
        corrupt_bytes = b"this is not a valid image payload at all"
        response = client.post(
            "/predict/sarcasm/multimodal",
            data={"text": "Comment with corrupted image."},
            files={"image": ("corrupted.jpg", corrupt_bytes, "image/jpeg")},
        )
        assert response.status_code == 422
        data = response.json()
        assert "detail" in data

    def test_multimodal_corrupt_image_fallback_when_enabled(self, client: TestClient):
        corrupt_bytes = b"corrupted image content"
        response = client.post(
            "/predict/sarcasm/multimodal?fallback_on_invalid_image=true",
            data={"text": "Comment with corrupted image fallback enabled."},
            files={"image": ("corrupted.jpg", corrupt_bytes, "image/jpeg")},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["modality_used"] == ["text"]
        assert data["fallback_applied"] is True
        assert any("routed through text-only fallback" in w for w in data["warnings"])

    def test_multimodal_unsupported_image_mime(self, client: TestClient):
        fake_pdf_bytes = b"%PDF-1.4 fake pdf"
        response = client.post(
            "/predict/sarcasm/multimodal",
            data={"text": "Comment with pdf."},
            files={"image": ("document.pdf", fake_pdf_bytes, "application/pdf")},
        )
        assert response.status_code == 422

    def test_multimodal_empty_text_rejection(self, client: TestClient):
        img_bytes = create_test_image_bytes()
        response = client.post(
            "/predict/sarcasm/multimodal",
            data={"text": "   "},
            files={"image": ("test.jpg", img_bytes, "image/jpeg")},
        )
        assert response.status_code == 422


class TestPhase9PassiveAggressionGovernance:
    """Tests for strict governance boundary on passive-aggression endpoint."""

    def test_passive_aggression_endpoint_returns_503(self, client: TestClient):
        response = client.post("/predict/passive-aggression")
        assert response.status_code == 503
        data = response.json()
        assert data["status"] == "unavailable"
        assert data["task"] == "passive_aggression_detection"
        assert data["benchmark_status"] == "BLOCKED_PENDING_HUMAN_ANNOTATION"
        assert "Auxiliary sarcasm models cannot be substituted" in data["message"]

    def test_no_sarcasm_to_pa_conversion(self, client: TestClient):
        # Verify text prediction response does NOT contain passive aggression labels
        response = client.post("/predict/sarcasm/text", json={"text": "Thanks for nothing!"})
        assert response.status_code == 200
        data = response.json()
        assert data["task"] == "sarcasm_detection"
        assert data["predicted_label"] in ["sarcastic", "non_sarcastic"]
        assert "passive_aggressive" not in data["class_scores"]
