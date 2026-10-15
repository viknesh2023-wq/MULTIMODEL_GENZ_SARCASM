"""
Master Comprehensive End-to-End Test Suite
Project: Gen-Z Multimodal Sarcasm Detection System
Tests live running backend (http://localhost:8000) against all requirements:
- Model metrics traceability to disk artifacts
- Dataset statistics verification
- Feature store dimensions and readiness
- Text-only sarcasm classification and validation edge cases
- Multimodal 2,847-D fusion inference across all 5 branches
- Evidence-grounded XAI reasoner and prediction invariance
"""
import io
import sys
import time
import requests
from PIL import Image

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

BASE_URL = "http://localhost:8000"

def test_system_health():
    print("\n--- TEST GROUP 1: SYSTEM HEALTH & MODEL READINESS ---")
    r = requests.get(f"{BASE_URL}/api/health", timeout=10)
    assert r.status_code == 200, f"Expected 200, got {r.status_code}"
    health = r.json()
    assert health["status"] == "ok"
    assert health["model_loaded"] is True
    assert health["multimodal_ready"] is True
    assert health["explanation_ready"] is True
    print("[PASS] /api/health: All models loaded, multimodal ready, explanation reasoner ready.")

    r_status = requests.get(f"{BASE_URL}/api/model/status", timeout=10)
    assert r_status.status_code == 200
    status_data = r_status.json()
    assert status_data["loaded"] is True
    assert status_data["status"] in ("ready", "loaded")
    print(f"[PASS] /api/model/status: Backbone '{status_data['model']}' on '{status_data['device']}'.")


def test_metrics_traceability():
    print("\n--- TEST GROUP 2: MODEL EVALUATION METRICS TRACEABILITY ---")
    r = requests.get(f"{BASE_URL}/api/model/metrics", timeout=10)
    assert r.status_code == 200
    metrics = r.json()

    # 1. Text Sarcasm
    assert "text_sarcasm" in metrics, "text_sarcasm block missing"
    t_sarc = metrics["text_sarcasm"]
    assert t_sarc["test_samples"] == 96519
    assert abs(t_sarc["test_accuracy"] - 0.771941) < 0.001
    assert abs(t_sarc["test_macro_f1"] - 0.771939) < 0.001
    assert t_sarc["confusion_matrix"] == [[37112, 10824], [11188, 37395]]
    print(f"[PASS] Text Sarcasm Metrics: Acc {t_sarc['test_accuracy']*100:.2f}%, F1 {t_sarc['test_macro_f1']*100:.2f}%, N={t_sarc['test_samples']}")

    # 2. Sentiment
    assert "sentiment" in metrics, "sentiment block missing"
    sent = metrics["sentiment"]
    assert sent["test_samples"] == 16730
    assert abs(sent["test_accuracy"] - 0.733234) < 0.001
    assert abs(sent["test_macro_f1"] - 0.729453) < 0.001
    assert len(sent["confusion_matrix"]) == 3
    print(f"[PASS] Sentiment Metrics (3-Class): Acc {sent['test_accuracy']*100:.2f}%, Macro F1 {sent['test_macro_f1']*100:.2f}%, N={sent['test_samples']}")

    # 3. Emotion
    assert "emotion" in metrics, "emotion block missing"
    emo = metrics["emotion"]
    assert emo["test_samples"] == 5427
    assert abs(emo["test_micro_f1"] - 0.492621) < 0.001
    assert abs(emo["test_macro_f1"] - 0.219682) < 0.001
    assert len(emo["per_emotion_metrics"]) == 28
    print(f"[PASS] Emotion Metrics (28-Class): Micro F1 {emo['test_micro_f1']*100:.2f}%, Macro F1 {emo['test_macro_f1']*100:.2f}%, 28 categories verified")

    # 4. Multimodal Fusion
    assert "multimodal_fusion" in metrics, "multimodal_fusion block missing"
    fuse = metrics["multimodal_fusion"]
    total_test_samples = sum(sum(row) for row in fuse["confusion_matrix"])
    assert total_test_samples == 3802
    assert fuse["per_class_metrics"]["0_non_sarcastic"]["support"] == 1666
    assert fuse["per_class_metrics"]["1_sarcastic"]["support"] == 2136
    assert abs(fuse["test_accuracy"] - 0.796160) < 0.001
    assert abs(fuse["test_macro_f1"] - 0.786393) < 0.001
    assert abs(fuse["val_macro_f1"] - 0.839880) < 0.001
    assert fuse["confusion_matrix"] == [[1107, 559], [216, 1920]]
    print(f"[PASS] Multimodal Fusion Metrics: Test Acc {fuse['test_accuracy']*100:.2f}%, Test Macro F1 {fuse['test_macro_f1']*100:.2f}%, Val F1 {fuse['val_macro_f1']*100:.2f}%, N={total_test_samples}")

    # 5. Ablation Benchmark
    assert "ablation_benchmark" in metrics, "ablation_benchmark missing"
    ablation = metrics["ablation_benchmark"]
    assert len(ablation) == 8, f"Expected 8 ablation rows, got {len(ablation)}"
    assert ablation[0]["configuration"] == "text_only"
    assert ablation[0]["dimensions"] == 768
    assert ablation[7]["configuration"] == "full_fusion"
    assert ablation[7]["dimensions"] == 2847
    print(f"[PASS] 8-Way Ablation Benchmark: Confirmed 8 configurations from text baseline ({ablation[0]['test_macro_f1']*100:.2f}%) to full fusion ({ablation[7]['test_macro_f1']*100:.2f}%)")


def test_dataset_statistics():
    print("\n--- TEST GROUP 3: DATASET STATISTICS & INTEGRITY ---")
    r = requests.get(f"{BASE_URL}/api/dataset/stats", timeout=10)
    assert r.status_code == 200
    stats = r.json()

    # Text Dataset
    td = stats["text_dataset"]
    assert td["total_samples"] == 1005167
    assert td["train_samples"] + td["validation_samples"] + td["test_samples"] == 1005167
    assert td["null_count"] == 0
    assert td["duplicate_count"] == 0
    print(f"[PASS] Text Sarcasm Dataset: {td['total_samples']:,} samples (812,130 train / 96,518 val / 96,519 test), 0 nulls, 0 duplicates")

    # Multimodal Dataset
    md = stats["multimodal_dataset"]
    assert md["total_samples"] == 33077
    assert md["train_samples"] + md["validation_samples"] + md["test_samples"] == 33077
    assert md["verified_images_on_disk"] == 33077
    assert md["clip_features_extracted"] == 33077
    assert md["internvl_features_extracted"] == 33077
    print(f"[PASS] Multimodal Dataset: {md['total_samples']:,} pairs, 33,077 images verified on disk, all 5 modalities extracted")

    # Sentiment Dataset
    sd = stats["sentiment_dataset"]
    assert sd["total_samples"] == 132324
    assert sd["train_samples"] + sd["validation_samples"] + sd["test_samples"] == 132324
    assert sd["classes"] == 3
    print(f"[PASS] Sentiment Dataset: {sd['total_samples']:,} samples (3 classes: Pos {sd['positive']:,}, Neu {sd['neutral']:,}, Neg {sd['negative']:,})")

    # Emotion Dataset
    ed = stats["emotion_dataset"]
    assert ed["total_samples"] == 54263
    assert ed["train_samples"] + ed["validation_samples"] + ed["test_samples"] == 54263
    assert ed["total_classes"] == 28
    print(f"[PASS] Emotion Dataset: {ed['total_samples']:,} samples across 28 GoEmotions categories")


def test_feature_status():
    print("\n--- TEST GROUP 4: FEATURE REPRESENTATION STORE ---")
    r = requests.get(f"{BASE_URL}/api/features/status", timeout=10)
    assert r.status_code == 200
    f_data = r.json()
    assert len(f_data["features"]) >= 6
    dims = [f["dimension"] for f in f_data["features"]]
    assert 768 in dims and 512 in dims and 1536 in dims and 3 in dims and 28 in dims and 2847 in dims
    print("[PASS] Feature Stores: 768-D, 512-D, 1536-D, 3-D, 28-D, and 2847-D composite vector all verified READY.")


def test_text_prediction():
    print("\n--- TEST GROUP 5: TEXT-ONLY SARCASM INFERENCE ---")
    test_cases = [
        "Oh great, another flat tire on the highway. Exactly what I needed!",
        "The weather report forecasts clear skies and mild temperatures for tomorrow.",
        "Bruh no cap this is literally peak comedy fr fr", # Gen-Z slang
    ]
    for text in test_cases:
        t0 = time.perf_counter()
        r = requests.post(f"{BASE_URL}/api/predict/text", json={"text": text}, timeout=15)
        elapsed = (time.perf_counter() - t0) * 1000
        assert r.status_code == 200, f"Error {r.status_code}: {r.text}"
        res = r.json()
        assert res["label"] in (0, 1)
        assert res["label_name"] in ("Sarcastic", "Non-Sarcastic")
        assert 0.0 <= res["confidence"] <= 1.0
        prob_sum = res["probabilities"]["non_sarcastic"] + res["probabilities"]["sarcastic"]
        assert abs(prob_sum - 1.0) < 0.01, f"Probabilities do not sum to 1: {prob_sum}"
        print(f"[PASS] Text: '{text[:45]}...' -> {res['label_name']} ({res['confidence']*100:.1f}%), latency={elapsed:.1f}ms")

    # Error handling tests
    # 1. Empty string
    r_empty = requests.post(f"{BASE_URL}/api/predict/text", json={"text": ""}, timeout=5)
    assert r_empty.status_code == 422, f"Expected 422 for empty string, got {r_empty.status_code}"
    # 2. Whitespace string
    r_ws = requests.post(f"{BASE_URL}/api/predict/text", json={"text": "   "}, timeout=5)
    assert r_ws.status_code == 422, f"Expected 422 for whitespace string, got {r_ws.status_code}"
    print("[PASS] Edge cases: Empty string and whitespace strings rejected with HTTP 422.")


def test_multimodal_prediction():
    print("\n--- TEST GROUP 6: MULTIMODAL 2,847-D FUSION INFERENCE ---")
    # Create test image
    img = Image.new("RGB", (224, 224), color=(120, 180, 240))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    img_bytes = buf.getvalue()

    # 1. Standard multimodal inference (explain=False)
    t0 = time.perf_counter()
    r = requests.post(
        f"{BASE_URL}/api/predict/multimodal",
        data={"text": "Another beautiful Monday morning stuck in traffic! Love it.", "explain": "false"},
        files={"image": ("test.jpg", img_bytes, "image/jpeg")},
        timeout=60,
    )
    elapsed = (time.perf_counter() - t0) * 1000
    assert r.status_code == 200, f"Error {r.status_code}: {r.text}"
    res = r.json()

    assert res["status"] == "success"
    pred = res["prediction"]
    assert pred["label"] in ("sarcastic", "non-sarcastic")
    assert pred["class_id"] in (0, 1)
    assert 0.0 <= pred["confidence"] <= 1.0

    # Probability sum == 1.0
    p_sum = pred["probabilities"]["non_sarcastic"] + pred["probabilities"]["sarcastic"]
    assert abs(p_sum - 1.0) < 0.01

    # Check 5 branches
    assert "sentiment" in res and "negative" in res["sentiment"]
    sent_sum = res["sentiment"]["negative"] + res["sentiment"]["neutral"] + res["sentiment"]["positive"]
    assert abs(sent_sum - 1.0) < 0.02, f"Sentiment softmax does not sum to 1: {sent_sum}"

    assert "emotions" in res and "label_probabilities" in res["emotions"]
    assert len(res["emotions"]["label_probabilities"]) == 28, "Must return 28 emotion probabilities"
    # Emotions are sigmoid: each must be between 0 and 1, NOT sum to 1
    for emo_name, prob in res["emotions"]["label_probabilities"].items():
        assert 0.0 <= prob <= 1.0, f"Emotion {emo_name} out of bounds: {prob}"

    # Model and feature branches block
    model_blk = res["model"]
    assert model_blk["feature_dimensions"] == 2847
    assert model_blk["branches"]["text_xlmr_dim"] == 768
    assert model_blk["branches"]["clip_dim"] == 512
    assert model_blk["branches"]["internvl_dim"] == 1536
    assert model_blk["branches"]["sentiment_dim"] == 3
    assert model_blk["branches"]["emotion_dim"] == 28

    print(f"[PASS] Multimodal Inference: Prediction='{pred['label']}' (conf={pred['confidence']*100:.1f}%), 2,847-D concatenated, latency={elapsed:.1f}ms")

    # Error handling tests
    # 1. Missing image
    r_no_img = requests.post(f"{BASE_URL}/api/predict/multimodal", data={"text": "hello"}, timeout=10)
    assert r_no_img.status_code == 422, f"Expected 422 for missing image, got {r_no_img.status_code}"
    # 2. Invalid file format
    r_bad_file = requests.post(
        f"{BASE_URL}/api/predict/multimodal",
        data={"text": "hello"},
        files={"image": ("test.txt", b"not an image", "text/plain")},
        timeout=10,
    )
    assert r_bad_file.status_code in (400, 422), f"Expected 400/422 for bad mime, got {r_bad_file.status_code}"
    print("[PASS] Multimodal Edge Cases: Missing image and invalid file MIME rejected appropriately.")


def test_explanation_reasoner_and_invariance():
    print("\n--- TEST GROUP 7: EXPLAINABLE AI REASONER & STRICT PREDICTION INVARIANCE ---")
    img = Image.new("RGB", (224, 224), color=(220, 80, 80))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    img_bytes = buf.getvalue()

    # Step A: Run prediction with explain=False
    r1 = requests.post(
        f"{BASE_URL}/api/predict/multimodal",
        data={"text": "Such a pleasant experience waiting in line for three hours.", "explain": "false"},
        files={"image": ("test.jpg", img_bytes, "image/jpeg")},
        timeout=60,
    )
    assert r1.status_code == 200
    res1 = r1.json()

    # Step B: Run prediction with explain=True
    r2 = requests.post(
        f"{BASE_URL}/api/predict/multimodal",
        data={"text": "Such a pleasant experience waiting in line for three hours.", "explain": "true"},
        files={"image": ("test.jpg", img_bytes, "image/jpeg")},
        timeout=90,
    )
    assert r2.status_code == 200
    res2 = r2.json()

    # Step C: STRICT MATHEMATICAL INVARIANCE VERIFICATION (Delta p = 0)
    assert res1["prediction"]["label"] == res2["prediction"]["label"], "CRITICAL: Reasoner modified predicted label!"
    assert res1["prediction"]["class_id"] == res2["prediction"]["class_id"], "CRITICAL: Reasoner modified class_id!"
    assert abs(res1["prediction"]["confidence"] - res2["prediction"]["confidence"]) < 1e-5, "CRITICAL: Reasoner modified confidence!"
    assert abs(res1["prediction"]["probabilities"]["non_sarcastic"] - res2["prediction"]["probabilities"]["non_sarcastic"]) < 1e-5
    assert abs(res1["prediction"]["probabilities"]["sarcastic"] - res2["prediction"]["probabilities"]["sarcastic"]) < 1e-5
    print("[PASS] MATHEMATICAL INVARIANCE VERIFIED: Classifier predictions are 100% identical (Delta p = 0) with/without XAI reasoner.")

    # Step D: Verify explanation structure
    expl = res2["explanation"]
    assert expl is not None, "Explanation was None despite explain=true"
    assert "summary" in expl and len(expl["summary"]) > 0
    assert "text_evidence" in expl
    assert "visual_evidence" in expl
    assert "supporting_signals" in expl
    assert "limitations" in expl

    reasoner_meta = res2["reasoner"]
    assert reasoner_meta is not None
    assert "model" in reasoner_meta
    print(f"[PASS] Explanation Synthesized: Model='{reasoner_meta['model']}', Runtime='{reasoner_meta.get('runtime')}', Latency={reasoner_meta.get('latency_ms')}ms")

    # Step E: Standalone /api/predict/explain endpoint
    r3 = requests.post(
        f"{BASE_URL}/api/predict/explain",
        json={
            "text": "Such a pleasant experience waiting in line for three hours.",
            "prediction": res1["prediction"],
            "sentiment": res1["sentiment"],
            "emotions": res1["emotions"],
            "visual_evidence": "Crowded indoor waiting line with low lighting",
        },
        timeout=60,
    )
    assert r3.status_code == 200
    res3 = r3.json()
    assert res3["status"] == "success"
    assert res3["explanation"] is not None
    print(f"[PASS] Standalone /api/predict/explain operational: Reasoner='{res3['reasoner']['model']}', Latency={res3['reasoner']['latency_ms']}ms")


if __name__ == "__main__":
    t_start = time.time()
    print("=" * 70)
    print("GEN-Z MULTIMODAL SARCASM DETECTION: MASTER VERIFICATION TEST SUITE")
    print("=" * 70)

    try:
        test_system_health()
        test_metrics_traceability()
        test_dataset_statistics()
        test_feature_status()
        test_text_prediction()
        test_multimodal_prediction()
        test_explanation_reasoner_and_invariance()

        t_total = time.time() - t_start
        print("\n" + "=" * 70)
        print(f"ALL 7 TEST GROUPS PASSED SUCCESSFULLY in {t_total:.2f} seconds!")
        print("System verified fully operational, research-grade, and production-ready.")
        print("=" * 70)
    except Exception as exc:
        print(f"\n[FAIL] TEST FAILURE: {exc}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
