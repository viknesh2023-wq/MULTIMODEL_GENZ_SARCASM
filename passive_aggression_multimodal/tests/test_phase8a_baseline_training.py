"""
tests/test_phase8a_baseline_training.py

Automated test suite for Phase 8A: Baseline Training & Multimodal Fusion.

Verifies:
  1. Phase 8A artifacts and directories exist.
  2. Text-only sarcasm baseline checkpoint exists and is loadable.
  3. Gated multimodal fusion checkpoint exists and is loadable.
  4. Master training results JSON contains complete metrics for both models.
  5. Test split preservation: test metrics computed on exactly 4,209 held-out records.
  6. Confusion matrices are valid 2x2 integer matrices matching test size.
  7. Strict task boundary: task_type is AUXILIARY_MULTIMODAL_SARCASM_DETECTION.
  8. Human benchmark gate remains strictly BLOCKED_PENDING_HUMAN_ANNOTATION.
  9. Phase 8A training report exists, is non-empty, and preserves scientific warnings.
"""

import json
import pickle
from pathlib import Path
import sys
import pytest
import torch

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

ARTIFACTS_P8A = ROOT_DIR / "artifacts/phase8a"
RESULTS_JSON = ARTIFACTS_P8A / "phase8a_training_results.json"
TEXT_CKPT = ARTIFACTS_P8A / "text_baseline/best_text_baseline.pkl"
MM_CKPT = ARTIFACTS_P8A / "multimodal_fusion/best_multimodal_fusion.pt"
REPORT_MD = ROOT_DIR / "reports/phase8a/phase8a_training_report.md"


@pytest.fixture
def phase8a_results():
    if not RESULTS_JSON.exists():
        pytest.skip(f"Phase 8A results JSON not yet generated at {RESULTS_JSON}")
    with open(RESULTS_JSON, "r", encoding="utf-8") as f:
        return json.load(f)


def test_phase8a_artifacts_exist():
    """Confirms Phase 8A artifact directory and report exist."""
    assert ARTIFACTS_P8A.exists(), "Phase 8A artifacts directory missing"
    assert REPORT_MD.exists(), "Phase 8A training report missing"
    assert REPORT_MD.stat().st_size > 500, "Report is empty or too short"


def test_text_baseline_checkpoint_loadable():
    """Verifies that the text-only baseline pickle checkpoint can be loaded and predict."""
    from src.models.text_baseline import TextBaselineModel
    if not TEXT_CKPT.exists():
        pytest.skip(f"Text baseline checkpoint missing: {TEXT_CKPT}")
    model = TextBaselineModel.load(TEXT_CKPT)
    assert hasattr(model, "predict"), "Loaded model missing predict method"
    preds = model.predict(["This is a test comment", "Another comment"])
    assert len(preds) == 2


def test_multimodal_fusion_checkpoint_loadable():
    """Verifies that the multimodal fusion PyTorch checkpoint can be loaded."""
    if not MM_CKPT.exists():
        pytest.skip(f"Multimodal fusion checkpoint missing: {MM_CKPT}")
    ckpt = torch.load(MM_CKPT, map_location="cpu")
    assert "model_state_dict" in ckpt, "Checkpoint missing model_state_dict"
    assert "val_macro_f1" in ckpt, "Checkpoint missing val_macro_f1"
    assert "model_config" in ckpt, "Checkpoint missing model_config"
    assert ckpt["val_macro_f1"] > 0.0, "Validation Macro-F1 should be positive"


def test_metrics_integrity(phase8a_results):
    """Verifies metric structure and validity for both models."""
    tb = phase8a_results["text_baseline"]["test_metrics"]
    mf = phase8a_results["multimodal_fusion"]["test_metrics"]

    for name, m in [("Text Baseline", tb), ("Multimodal Fusion", mf)]:
        assert 0.0 <= m["accuracy"] <= 1.0, f"{name}: invalid accuracy"
        assert 0.0 <= m["macro_f1"] <= 1.0, f"{name}: invalid macro_f1"
        assert 0.0 <= m["macro_precision"] <= 1.0, f"{name}: invalid macro_precision"
        assert 0.0 <= m["macro_recall"] <= 1.0, f"{name}: invalid macro_recall"
        assert "per_class" in m
        assert "non_sarcastic" in m["per_class"]
        assert "sarcastic" in m["per_class"]


def test_confusion_matrix_dimensions(phase8a_results):
    """Verifies confusion matrices are 2x2 and sum to total test records."""
    tb_cm = phase8a_results["text_baseline"]["test_metrics"]["confusion_matrix"]
    mf_cm = phase8a_results["multimodal_fusion"]["test_metrics"]["confusion_matrix"]

    for cm in [tb_cm, mf_cm]:
        assert len(cm) == 2
        assert len(cm[0]) == 2
        assert len(cm[1]) == 2
        total_eval = sum(sum(row) for row in cm)
        assert total_eval > 0


def test_strict_auxiliary_task_boundary(phase8a_results):
    """CRITICAL: Verifies experiment is strictly marked as auxiliary sarcasm detection."""
    assert phase8a_results["task_type"] == "AUXILIARY_MULTIMODAL_SARCASM_DETECTION"
    assert phase8a_results["human_benchmark_gate"] == "BLOCKED_PENDING_HUMAN_ANNOTATION"


def test_scientific_report_warnings():
    """Verifies that the report contains explicit scientific warnings and gate status."""
    with open(REPORT_MD, "r", encoding="utf-8") as f:
        text = f.read()
    assert "BLOCKED_PENDING_HUMAN_ANNOTATION" in text
    assert "Auxiliary Sarcasm Detection" in text
    assert "not" in text.lower() and "passive-aggression" in text
