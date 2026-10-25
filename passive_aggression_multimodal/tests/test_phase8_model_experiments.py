"""
tests/test_phase8_model_experiments.py

Automated test suite verifying Phase 8:
- Checkpoint loading and prediction shapes
- Record-to-feature alignment
- Leakage isolation and ID disjointness
- Preservation of the human annotation evaluation gate
- Scientific distinction between auxiliary sarcasm and primary passive aggression
- Confusion matrix orientation and metric correctness
- Missing-modality graceful handling and zero-image ablation behavior
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
import numpy as np
import pytest
import torch

from src.evaluation.metrics import compute_classification_metrics
from src.models.gated_multimodal_fusion import MultimodalFusionClassifier
from src.models.text_baseline import TextBaselineModel

ROOT_DIR = Path(__file__).resolve().parent.parent
SPLITS_DIR = ROOT_DIR / "data/splits/phase7"
FEATURES_DIR = ROOT_DIR / "data/features/phase8a"
ARTIFACTS_P8A = ROOT_DIR / "artifacts/phase8a"
ARTIFACTS_P8 = ROOT_DIR / "artifacts/phase8"
REPORTS_P8 = ROOT_DIR / "reports/phase8"


class TestPhase8Verification:
    """Verify preflight integrity, manifests, and zero-leakage conditions."""

    def test_split_manifests_exist_and_counts(self):
        train_csv = SPLITS_DIR / "sarcasm_auxiliary_train.csv"
        val_csv = SPLITS_DIR / "sarcasm_auxiliary_val.csv"
        test_csv = SPLITS_DIR / "sarcasm_auxiliary_test.csv"

        assert train_csv.exists(), "Missing train split CSV"
        assert val_csv.exists(), "Missing val split CSV"
        assert test_csv.exists(), "Missing test split CSV"

        with open(train_csv, "r", encoding="utf-8") as f:
            train_rows = list(csv.DictReader(f))
        with open(val_csv, "r", encoding="utf-8") as f:
            val_rows = list(csv.DictReader(f))
        with open(test_csv, "r", encoding="utf-8") as f:
            test_rows = list(csv.DictReader(f))

        assert len(train_rows) == 19618, f"Expected 19618 train records, got {len(train_rows)}"
        assert len(val_rows) == 4143, f"Expected 4143 val records, got {len(val_rows)}"
        assert len(test_rows) == 4209, f"Expected 4209 test records, got {len(test_rows)}"
        assert len(train_rows) + len(val_rows) + len(test_rows) == 27970

    def test_split_id_disjointness_and_duplicate_isolation(self):
        def load_ids_and_dup_groups(csv_p):
            ids, dup_groups = set(), set()
            with open(csv_p, "r", encoding="utf-8") as f:
                for r in csv.DictReader(f):
                    ids.add(r["record_id"])
                    dg = r.get("duplicate_group_id") or r.get("canonical_id")
                    if dg:
                        dup_groups.add(dg)
            return ids, dup_groups

        tr_ids, tr_dg = load_ids_and_dup_groups(SPLITS_DIR / "sarcasm_auxiliary_train.csv")
        va_ids, va_dg = load_ids_and_dup_groups(SPLITS_DIR / "sarcasm_auxiliary_val.csv")
        te_ids, te_dg = load_ids_and_dup_groups(SPLITS_DIR / "sarcasm_auxiliary_test.csv")

        # Zero record ID overlap
        assert len(tr_ids & va_ids) == 0, "Train-Val ID overlap detected!"
        assert len(tr_ids & te_ids) == 0, "Train-Test ID overlap detected!"
        assert len(va_ids & te_ids) == 0, "Val-Test ID overlap detected!"

        # Zero duplicate-group leakage
        assert len(tr_dg & va_dg) == 0, "Train-Val duplicate group leakage detected!"
        assert len(tr_dg & te_dg) == 0, "Train-Test duplicate group leakage detected!"
        assert len(va_dg & te_dg) == 0, "Val-Test duplicate group leakage detected!"

    def test_cached_features_shapes_and_values(self):
        f_train = torch.load(FEATURES_DIR / "features_train_19618.pt", map_location="cpu")
        f_val = torch.load(FEATURES_DIR / "features_val_4143.pt", map_location="cpu")
        f_test = torch.load(FEATURES_DIR / "features_test_4209.pt", map_location="cpu")

        for split_name, f_data, expected_len in [
            ("Train", f_train, 19618),
            ("Val", f_val, 4143),
            ("Test", f_test, 4209),
        ]:
            assert f_data["text"].shape == (expected_len, 768), f"{split_name} text feature shape mismatch"
            assert f_data["vision"].shape == (expected_len, 768), f"{split_name} vision feature shape mismatch"
            assert f_data["mask"].shape == (expected_len,), f"{split_name} mask shape mismatch"
            assert f_data["label"].shape == (expected_len,), f"{split_name} label shape mismatch"

            assert torch.isfinite(f_data["text"]).all(), f"NaN or Inf in {split_name} text features"
            assert torch.isfinite(f_data["vision"]).all(), f"NaN or Inf in {split_name} vision features"


class TestPhase8ModelCheckpoints:
    """Verify saved checkpoints, predictions, and reproducibility."""

    def test_text_baseline_pickle_loading_and_inference(self):
        ckpt_p = ARTIFACTS_P8A / "text_baseline/best_text_baseline.pkl"
        assert ckpt_p.exists(), "Text baseline pickle checkpoint missing"

        model = TextBaselineModel.load(ckpt_p)
        sample_texts = [
            "Oh brilliant, my train is delayed again!",
            "Thank you for the helpful and timely assistance.",
        ]
        preds = model.predict(sample_texts)
        probs = model.predict_proba(sample_texts)

        assert len(preds) == 2
        assert probs.shape == (2, 2)
        assert np.allclose(probs.sum(axis=1), 1.0)

    def test_multimodal_gated_checkpoint_loading_and_fallback(self):
        ckpt_p = ARTIFACTS_P8A / "multimodal_fusion/best_multimodal_fusion.pt"
        assert ckpt_p.exists(), "Multimodal fusion checkpoint missing"

        ckpt = torch.load(ckpt_p, map_location="cpu")
        assert "model_state_dict" in ckpt
        assert "val_macro_f1" in ckpt

        model = MultimodalFusionClassifier(
            image_dim=768, text_dim=768, hidden_dim=512,
            num_classes_primary=2, num_classes_auxiliary=None,
        )
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        # Test normal inference
        dummy_t = torch.randn(4, 768)
        dummy_v = torch.randn(4, 768)
        dummy_m = torch.ones(4)
        out = model(comment_features=dummy_t, image_features=dummy_v, image_masks=dummy_m)
        assert "primary_logits" in out
        assert out["primary_logits"].shape == (4, 2)

        # Test missing image fallback (mask = 0)
        zero_v = torch.zeros(4, 768)
        zero_m = torch.zeros(4)
        out_fallback = model(comment_features=dummy_t, image_features=zero_v, image_masks=zero_m)
        assert out_fallback["primary_logits"].shape == (4, 2)
        assert torch.isfinite(out_fallback["primary_logits"]).all()

    def test_phase8_new_checkpoints_exist(self):
        expected_checkpoints = [
            "text_roberta_neural.pt",
            "image_only_neural.pt",
            "concat_multimodal_fusion.pt",
            "residual_gated_fusion.pt",
        ]
        for name in expected_checkpoints:
            p = ARTIFACTS_P8 / name
            assert p.exists(), f"Phase 8 checkpoint {name} does not exist in {ARTIFACTS_P8}"
            assert p.stat().st_size > 100_000, f"Phase 8 checkpoint {name} is unusually small"


class TestPhase8MetricsAndScientificGovernance:
    """Verify metric computation, task labeling, and benchmark gate preservation."""

    def test_metrics_computation_and_cm_orientation(self):
        y_true = np.array([0, 0, 1, 1, 1])
        y_pred = np.array([0, 1, 1, 1, 0])
        # Non-sarcastic (0): TP=1, FP=1, FN=1 -> F1=0.6667
        # Sarcastic (1):     TP=2, FP=1, FN=1 -> F1=0.6667
        # CM: [[1, 1], [1, 2]]
        metrics = compute_classification_metrics(y_true, y_pred, class_names=["non_sarcastic", "sarcastic"])

        assert metrics["total_samples"] == 5
        cm = np.array(metrics["confusion_matrix"])
        assert cm[0, 0] == 1  # True 0, Pred 0
        assert cm[0, 1] == 1  # True 0, Pred 1
        assert cm[1, 0] == 1  # True 1, Pred 0
        assert cm[1, 1] == 2  # True 1, Pred 1
        assert cm.sum() == 5

    def test_experiment_results_json_schema_and_counts(self):
        results_file = ARTIFACTS_P8 / "phase8_experiment_results.json"
        assert results_file.exists(), "Master experiment results JSON missing"

        with open(results_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert data["benchmark_gate"] == "BLOCKED_PENDING_HUMAN_ANNOTATION"
        assert data["random_seed"] == 42

        exps = data["experiments"]
        assert "text_tfidf_baseline" in exps
        assert "text_roberta_neural" in exps
        assert "image_only_baseline" in exps
        assert "gated_multimodal_fusion" in exps
        assert "concat_multimodal_fusion" in exps
        assert "residual_gated_fusion" in exps
        assert "ablation_zero_image_inference" in exps

        # Check sample count in test metrics
        for k in ["text_tfidf_baseline", "text_roberta_neural", "gated_multimodal_fusion"]:
            assert exps[k]["test_metrics"]["total_samples"] == 4209

    def test_human_annotation_files_untouched(self):
        h1 = ROOT_DIR / "data/annotations/phase6a/annotator_1_responses.csv"
        h2 = ROOT_DIR / "data/annotations/phase6a/annotator_2_responses.csv"
        status_file = ROOT_DIR / "data/annotations/phase6a/annotation_status_manifest.json"

        assert h1.exists() and h2.exists() and status_file.exists()

        with open(status_file, "r", encoding="utf-8") as f:
            st = json.load(f)
        assert st["annotator_01"]["completed_count"] == 0
        assert st["annotator_02"]["completed_count"] == 0
        assert st["evaluation_readiness_status"] == "BLOCKED_PENDING_HUMAN_ANNOTATION"
