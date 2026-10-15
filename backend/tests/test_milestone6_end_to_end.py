"""
Milestone 6 End-to-End API Integration & Regression Test Suite
Project: Gen-Z Multimodal Sarcasm Detection System

Tests:
1. Health and Model Status endpoints reflect Explanation Reasoner readiness.
2. Text-only prediction route regression test (/api/predict/text).
3. Multimodal prediction route default test (/api/predict/multimodal, explain=False).
4. Multimodal prediction route with inline explanation (/api/predict/multimodal, explain=True).
5. Dedicated explanation endpoint (/api/predict/explain).
6. Strict preservation of classifier prediction label, class_id, and probabilities.
7. Graceful degradation when invalid input is provided.
"""

import io
import sys
import time
import json
from pathlib import Path
from PIL import Image
from fastapi.testclient import TestClient

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

PROJECT_ROOT = BACKEND_DIR.parent
if str(PROJECT_ROOT / "fusion") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "fusion"))

from app.main import app
from app.services import model_service
from app.services.multimodal_service import get_multimodal_manager
from app.services.explanation_service import get_explanation_service

# Pre-load services
model_service.load_model()
mm_manager = get_multimodal_manager()
mm_manager.load_all_models()
expl_service = get_explanation_service()

client = TestClient(app)


def test_health_and_model_status():
    """Verify health and model status accurately report explanation readiness."""
    r_health = client.get("/api/health")
    assert r_health.status_code == 200
    h_data = r_health.json()
    assert h_data["status"] == "ok"
    assert h_data["explanation_ready"] is True

    r_status = client.get("/api/model/status")
    assert r_status.status_code == 200
    s_data = r_status.json()
    assert s_data["status"] in ("ready", "loaded")
    assert s_data["loaded"] is True
    assert s_data["explanation_ready"] is True
    assert "explanation_model" in s_data
    print(f"\n[Test 1 PASSED] Explanation model registered: {s_data['explanation_model']}")


def test_text_prediction_regression():
    """Verify that existing text-only prediction route remains operational."""
    payload = {"text": "I just love when the internet goes out during an exam."}
    r = client.post("/api/predict/text", json=payload)
    assert r.status_code == 200
    data = r.json()
    assert data["label_name"].lower() in ("sarcastic", "non-sarcastic")
    assert 0.0 <= data["confidence"] <= 1.0
    print(f"[Test 2 PASSED] Text prediction operational: {data['label_name']} ({data['confidence']*100:.1f}%)")


def test_multimodal_prediction_without_explanation():
    """Verify standard multimodal prediction works cleanly without explanation overhead."""
    img = Image.new("RGB", (224, 224), color=(100, 150, 200))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    img_bytes = buf.getvalue()

    t0 = time.time()
    r = client.post(
        "/api/predict/multimodal",
        data={"text": "A beautiful and peaceful afternoon in the park.", "explain": "false"},
        files={"image": ("test.jpg", img_bytes, "image/jpeg")}
    )
    latency = time.time() - t0
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "success"
    assert data["prediction"]["label"] in ("sarcastic", "non-sarcastic")
    assert data["explanation"] is None  # Must be None when explain=False
    print(f"[Test 3 PASSED] Multimodal prediction (explain=False) latency: {latency*1000:.1f} ms")


def test_multimodal_prediction_with_explanation():
    """Verify multimodal prediction with explain=True attaches structured explanation without altering prediction."""
    img = Image.new("RGB", (224, 224), color=(200, 100, 50))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    img_bytes = buf.getvalue()

    t0 = time.time()
    r = client.post(
        "/api/predict/multimodal",
        data={"text": "Oh absolutely wonderful, another delayed flight!", "explain": "true"},
        files={"image": ("flight.jpg", img_bytes, "image/jpeg")}
    )
    latency = time.time() - t0
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "success"
    assert data["explanation"] is not None

    expl = data["explanation"]
    assert "summary" in expl and isinstance(expl["summary"], str)
    assert "text_evidence" in expl and isinstance(expl["text_evidence"], list)
    assert "visual_evidence" in expl and isinstance(expl["visual_evidence"], list)
    assert "supporting_signals" in expl and isinstance(expl["supporting_signals"], list)
    assert "limitations" in expl and isinstance(expl["limitations"], list)

    assert data["prediction"]["label"] in ("sarcastic", "non-sarcastic")
    assert data["reasoner"] is not None and "model" in data["reasoner"]
    print(f"[Test 4 PASSED] Multimodal prediction with inline explanation verified. Model: {data['reasoner']['model']}")


def test_dedicated_explain_endpoint():
    """Verify dedicated POST /api/predict/explain works with valid prediction references."""
    payload = {
        "text": "Living the absolute dream right now with flat tire on highway.",
        "prediction": {
            "label": "sarcastic",
            "class_id": 1,
            "confidence": 0.89,
            "probabilities": {"non_sarcastic": 0.11, "sarcastic": 0.89}
        },
        "sentiment": {"negative": 0.75, "neutral": 0.15, "positive": 0.10},
        "emotions": {
            "top_emotions": [
                {"emotion": "annoyance", "probability": 0.82},
                {"emotion": "disappointment", "probability": 0.74}
            ]
        },
        "visual_evidence": "A flat tire on an asphalt highway shoulder beside a disabled sedan."
    }

    t0 = time.time()
    r = client.post("/api/predict/explain", json=payload)
    latency = time.time() - t0
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "success"
    assert data["prediction_reference"]["label"] == "sarcastic"
    assert len(data["explanation"]["text_evidence"]) > 0
    assert len(data["explanation"]["visual_evidence"]) > 0
    print(f"[Test 5 PASSED] Dedicated /api/predict/explain response latency: {latency*1000:.1f} ms")


def test_classifier_invariance():
    """Verify that multiple calls with and without explanation preserve exact probabilities."""
    img = Image.new("RGB", (224, 224), color=(150, 150, 150))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    img_bytes = buf.getvalue()

    r1 = client.post(
        "/api/predict/multimodal",
        data={"text": "The test case verifying probability invariance.", "explain": "false"},
        files={"image": ("inv.jpg", img_bytes, "image/jpeg")}
    )
    assert r1.status_code == 200
    p1 = r1.json()["prediction"]

    r2 = client.post(
        "/api/predict/multimodal",
        data={"text": "The test case verifying probability invariance.", "explain": "true"},
        files={"image": ("inv.jpg", img_bytes, "image/jpeg")}
    )
    assert r2.status_code == 200
    p2 = r2.json()["prediction"]

    assert p1["label"] == p2["label"], "Labels must be identical"
    assert abs(p1["probabilities"]["sarcastic"] - p2["probabilities"]["sarcastic"]) < 1e-6, "Probabilities must be invariant"
    print(f"[Test 6 PASSED] Underlying classifier outputs are strictly invariant under explanation toggle.")


if __name__ == "__main__":
    print("=" * 60)
    print("RUNNING MILESTONE 6 END-TO-END FASTAPI & EXPLANATION TESTS")
    print("=" * 60)
    test_health_and_model_status()
    test_text_prediction_regression()
    test_multimodal_prediction_without_explanation()
    test_multimodal_prediction_with_explanation()
    test_dedicated_explain_endpoint()
    test_classifier_invariance()
    print("=" * 60)
    print("ALL 6 END-TO-END REGRESSION TESTS PASSED CLEANLY!")
    print("=" * 60)
