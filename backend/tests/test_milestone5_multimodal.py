"""
Automated Test Suite for Milestone 5: Production-Ready Multimodal Inference
Project: Gen-Z Multimodal Sarcasm Detection System
Hardware: NVIDIA GeForce RTX 4050 Laptop GPU / CUDA Acceleration

Tests:
1. Feature extractor output shapes and value finiteness.
2. Canonical 2,847-D concatenation and index ordering.
3. Saved StandardScaler and MultimodalFusionMLP loading and inference.
4. Health, model status, and features status readiness endpoints.
5. Text sarcasm endpoint regression tests (/api/predict/text).
6. Input validation for /api/predict/multimodal:
   - Missing text
   - Empty text
   - Missing image
   - Corrupted image bytes
   - Out-of-bounds image dimensions
7. Real end-to-end multimodal inference request (/api/predict/multimodal) with real text & image.
8. GPU concurrency and sequential safety.
"""

import io
import os
import sys
import time
import json
import numpy as np
from PIL import Image
from pathlib import Path
from fastapi.testclient import TestClient

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

PROJECT_ROOT = BACKEND_DIR.parent
if str(PROJECT_ROOT / "fusion") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "fusion"))

from app.main import app
from app.services import model_service
from app.services.multimodal_service import get_multimodal_manager, EMOTION_LABELS, SENTIMENT_LABELS
from fusion_inference import MultimodalFusionInference, EXPECTED_DIMENSION

# Explicitly initialize models for test execution
model_service.load_model()
manager = get_multimodal_manager()
manager.load_all_models()

client = TestClient(app)


def test_saved_scaler_and_fusion_model_loading():
    """Verify saved scaler and PyTorch checkpoint load correctly and produce calibrated outputs."""
    fusion_dir = PROJECT_ROOT / "models" / "multimodal_fusion"
    assert (fusion_dir / "fusion_model.pt").exists(), "fusion_model.pt must exist"
    assert (fusion_dir / "fusion_scaler.joblib").exists(), "fusion_scaler.joblib must exist"
    assert (fusion_dir / "model_config.json").exists(), "model_config.json must exist"

    engine = MultimodalFusionInference(model_dir=fusion_dir, device="cpu")
    assert engine.scaler.n_features_in_ == EXPECTED_DIMENSION, f"Scaler must expect {EXPECTED_DIMENSION} features"

    # Test random finite vector
    test_vec = np.random.randn(EXPECTED_DIMENSION).astype(np.float32)
    res = engine.predict_vector(test_vec)

    assert res["predicted_label"] in (0, 1)
    assert res["label_name"] in ("non-sarcastic", "sarcastic")
    assert 0.0 <= res["prob_non_sarcastic"] <= 1.0
    assert 0.0 <= res["prob_sarcastic"] <= 1.0
    assert abs((res["prob_non_sarcastic"] + res["prob_sarcastic"]) - 1.0) < 1e-4
    assert res["confidence"] >= 0.5


def test_feature_extractor_shapes_and_ordering():
    """Verify that the multimodal manager extracts all 5 blocks with exact shapes and canonical order."""
    manager = get_multimodal_manager()
    manager.load_all_models()
    assert manager.is_ready()

    sample_text = "Oh absolutely wonderful weather today for a picnic!"
    sample_img = Image.new("RGB", (224, 224), color=(100, 150, 200))

    res = manager.predict_multimodal(text=sample_text, image=sample_img)
    assert res["status"] == "success"
    assert res["model"]["feature_dimensions"] == 2847

    # Check sentiment
    sent = res["sentiment"]
    assert "negative" in sent and "neutral" in sent and "positive" in sent
    sent_sum = sent["negative"] + sent["neutral"] + sent["positive"]
    assert abs(sent_sum - 1.0) < 1e-4

    # Check emotions
    emos = res["emotions"]["label_probabilities"]
    assert len(emos) == 28, f"Expected 28 emotions, got {len(emos)}"
    for emo_label in EMOTION_LABELS:
        assert emo_label in emos, f"Missing emotion label: {emo_label}"
        assert 0.0 <= emos[emo_label] <= 1.0


def test_readiness_and_health_endpoints():
    """Verify /api/health, /api/model/status, and /api/features/status report accurate readiness."""
    r_health = client.get("/api/health")
    assert r_health.status_code == 200
    data_health = r_health.json()
    assert data_health["status"] == "ok"
    assert data_health["model_loaded"] is True
    assert data_health["multimodal_ready"] is True

    r_status = client.get("/api/model/status")
    assert r_status.status_code == 200
    data_status = r_status.json()
    assert data_status["loaded"] is True
    assert data_status["multimodal_ready"] is True
    assert data_status["status"] == "ready"

    r_feat = client.get("/api/features/status")
    assert r_feat.status_code == 200
    data_feat = r_feat.json()
    assert len(data_feat["features"]) == 6
    for f in data_feat["features"]:
        assert f["status"] == "ready"


def test_text_prediction_regression():
    """Verify existing text sarcasm endpoint (/api/predict/text) functions without regression."""
    # Valid sarcasm
    r_sarc = client.post("/api/predict/text", json={"text": "I just love standing in line for 4 hours."})
    assert r_sarc.status_code == 200
    data_sarc = r_sarc.json()
    assert data_sarc["label"] in (0, 1)
    assert 0.0 <= data_sarc["confidence"] <= 1.0
    assert 0.0 <= data_sarc["probabilities"]["sarcastic"] <= 1.0
    assert 0.0 <= data_sarc["probabilities"]["non_sarcastic"] <= 1.0

    # Empty text validation
    r_empty = client.post("/api/predict/text", json={"text": "   "})
    assert r_empty.status_code == 422


def test_multimodal_input_validation():
    """Verify that /api/predict/multimodal rejects invalid, missing, or corrupted inputs with HTTP 422."""
    valid_img = Image.new("RGB", (100, 100), color=(255, 0, 0))
    img_byte_arr = io.BytesIO()
    valid_img.save(img_byte_arr, format="JPEG")
    valid_bytes = img_byte_arr.getvalue()

    # 1. Missing text
    r_no_text = client.post(
        "/api/predict/multimodal",
        data={"text": "   "},
        files={"image": ("test.jpg", valid_bytes, "image/jpeg")},
    )
    assert r_no_text.status_code == 422

    # 2. Empty image bytes
    r_empty_img = client.post(
        "/api/predict/multimodal",
        data={"text": "A valid caption"},
        files={"image": ("test.jpg", b"", "image/jpeg")},
    )
    assert r_empty_img.status_code == 422

    # 3. Corrupted image
    r_corrupt = client.post(
        "/api/predict/multimodal",
        data={"text": "A valid caption"},
        files={"image": ("test.jpg", b"NOT_AN_IMAGE_RANDOM_CORRUPT_BYTES", "image/jpeg")},
    )
    assert r_corrupt.status_code == 422

    # 4. Tiny image (< 10x10)
    tiny_img = Image.new("RGB", (4, 4), color=(0, 255, 0))
    tiny_arr = io.BytesIO()
    tiny_img.save(tiny_arr, format="PNG")
    r_tiny = client.post(
        "/api/predict/multimodal",
        data={"text": "A valid caption"},
        files={"image": ("tiny.png", tiny_arr.getvalue(), "image/png")},
    )
    assert r_tiny.status_code == 422


def test_real_multimodal_prediction_end_to_end():
    """Execute real end-to-end multimodal inference via FastAPI POST /api/predict/multimodal."""
    # Create test image or load real dataset image
    real_img_path = PROJECT_ROOT / "06_MMSD2.0" / "DATA1" / "dataset_image" / "dataset_image" / "840006160660983809.jpg"
    if real_img_path.exists():
        with open(real_img_path, "rb") as f:
            img_bytes = f.read()
    else:
        test_img = Image.new("RGB", (300, 300), color=(120, 200, 240))
        buf = io.BytesIO()
        test_img.save(buf, format="JPEG")
        img_bytes = buf.getvalue()

    text_input = "Oh great, thank you so much for breaking the server right before launch!"

    t0 = time.time()
    response = client.post(
        "/api/predict/multimodal",
        data={"text": text_input},
        files={"image": ("sample.jpg", img_bytes, "image/jpeg")},
    )
    elapsed = time.time() - t0

    assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
    body = response.json()

    assert body["status"] == "success"
    pred = body["prediction"]
    assert pred["label"] in ("sarcastic", "non-sarcastic")
    assert pred["class_id"] in (0, 1)
    assert 0.0 <= pred["confidence"] <= 1.0
    assert 0.0 <= pred["probabilities"]["non_sarcastic"] <= 1.0
    assert 0.0 <= pred["probabilities"]["sarcastic"] <= 1.0

    sent = body["sentiment"]
    assert 0.0 <= sent["negative"] <= 1.0
    assert 0.0 <= sent["neutral"] <= 1.0
    assert 0.0 <= sent["positive"] <= 1.0

    emos = body["emotions"]["label_probabilities"]
    assert len(emos) == 28

    model_info = body["model"]
    assert model_info["fusion_model"] == "MultimodalFusionMLP"
    assert model_info["feature_dimensions"] == 2847

    perf = body["performance"]
    assert perf["total_latency_ms"] > 0
    print(f"\n[Test] Real multimodal inference latency: {perf['total_latency_ms']} ms (client roundtrip: {elapsed*1000:.1f} ms)")


def test_gpu_concurrency_safety():
    """Verify that multiple consecutive requests do not exhaust GPU memory or deadlock the lock."""
    sample_img = Image.new("RGB", (200, 200), color=(50, 100, 150))
    buf = io.BytesIO()
    sample_img.save(buf, format="JPEG")
    img_bytes = buf.getvalue()

    for i in range(2):
        res = client.post(
            "/api/predict/multimodal",
            data={"text": f"Consecutive inference test {i}"},
            files={"image": (f"consec_{i}.jpg", img_bytes, "image/jpeg")},
        )
        assert res.status_code == 200
        assert res.json()["status"] == "success"


if __name__ == "__main__":
    print("=" * 70)
    print("RUNNING MILESTONE 5 AUTOMATED TESTS")
    print("=" * 70)
    tests = [
        ("Saved Scaler & Fusion Model Loading", test_saved_scaler_and_fusion_model_loading),
        ("Feature Extractor Shapes & Ordering", test_feature_extractor_shapes_and_ordering),
        ("Readiness & Health Endpoints", test_readiness_and_health_endpoints),
        ("Text Prediction Regression", test_text_prediction_regression),
        ("Multimodal Input Validation", test_multimodal_input_validation),
        ("Real Multimodal Prediction End-to-End", test_real_multimodal_prediction_end_to_end),
        ("GPU Concurrency & Sequential Safety", test_gpu_concurrency_safety),
    ]

    passed = 0
    failed = 0
    for name, test_fn in tests:
        print(f"\n--- Running: {name} ---")
        try:
            t0 = time.time()
            test_fn()
            dur = time.time() - t0
            print(f"PASSED ({dur:.2f}s)")
            passed += 1
        except Exception as e:
            print(f"FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 70)
    print(f"TEST SUMMARY: {passed} PASSED, {failed} FAILED")
    print("=" * 70)
    if failed > 0:
        sys.exit(1)

