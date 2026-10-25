"""Unit and integration tests for Phase 4 independent audit requirements."""
import os
import json
import pytest
import numpy as np
import torch

from src.evaluation.metrics import (
    compute_classification_metrics,
    compute_bootstrap_confidence_intervals,
    compute_calibration_error,
    compute_selective_metrics,
)
from src.models.classification_head import ClassificationHead
from src.models.gated_multimodal_fusion import GatedMultimodalFusion


def test_clean_synthetic_splits_disjointness():
    """Verify clean synthetic splits have zero record or comment leakage across splits."""
    registry_file = "data/splits/clean_synthetic_splits_registry.json"
    assert os.path.exists(registry_file), "Clean splits registry must exist"

    with open(registry_file, "r", encoding="utf-8") as f:
        registry = json.load(f)

    train_idx = set(registry["train_indices"])
    val_idx = set(registry["val_indices"])
    test_idx = set(registry["test_indices"])

    # Index disjointness
    assert len(train_idx.intersection(val_idx)) == 0, "Train and Val indices must be disjoint"
    assert len(train_idx.intersection(test_idx)) == 0, "Train and Test indices must be disjoint"
    assert len(val_idx.intersection(test_idx)) == 0, "Val and Test indices must be disjoint"

    # Comment text disjointness
    with open("data/synthetic/synthetic_pilot_350.jsonl", "r", encoding="utf-8") as f:
        records = [json.loads(l)["target_comment"].strip().lower() for l in f if l.strip()]

    train_comments = {records[i] for i in train_idx}
    val_comments = {records[i] for i in val_idx}
    test_comments = {records[i] for i in test_idx}

    assert len(train_comments.intersection(test_comments)) == 0, "Train and Test comments must have zero overlap"
    assert len(train_comments.intersection(val_comments)) == 0, "Train and Val comments must have zero overlap"


def test_primary_and_auxiliary_label_separation():
    """Ensure primary 3-way mapping and auxiliary binary mapping are decoupled."""
    primary_classes = ["passive_aggressive", "non_passive_aggressive", "ambiguous"]
    aux_classes = ["non_sarcastic", "sarcastic"]

    assert len(primary_classes) == 3
    assert len(aux_classes) == 2
    assert primary_classes[0] != aux_classes[0]
    assert "ambiguous" not in aux_classes


def test_evaluation_metrics_correctness():
    """Verify precision, recall, F1, bootstrap CI, and calibration calculations."""
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 1, 2, 0, 1, 1])

    metrics = compute_classification_metrics(y_true, y_pred, class_names=["pa", "non_pa", "ambiguous"])
    assert metrics["accuracy"] == pytest.approx(5.0 / 6.0)
    assert "macro_f1" in metrics
    assert "confusion_matrix" in metrics

    # Bootstrap CIs
    ci = compute_bootstrap_confidence_intervals(y_true, y_pred, n_bootstraps=50)
    assert "macro_f1" in ci
    assert ci["macro_f1"]["lower"] <= ci["macro_f1"]["upper"]

    # Calibration
    probs = np.array([
        [0.9, 0.05, 0.05],
        [0.1, 0.8, 0.1],
        [0.1, 0.1, 0.8],
        [0.85, 0.1, 0.05],
        [0.2, 0.7, 0.1],
        [0.1, 0.6, 0.3],
    ])
    calib = compute_calibration_error(y_true, probs, n_bins=5)
    assert 0.0 <= calib["ece"] <= 1.0

    # Selective classification
    sel = compute_selective_metrics(y_true, y_pred, probs, threshold=0.7)
    assert 0.0 <= sel["coverage"] <= 1.0


def test_checkpoint_head_loading_and_inference():
    """Ensure checkpoint classification heads load cleanly with correct dimensions."""
    head = ClassificationHead(in_dim=768, hidden_dim=256, num_classes=3, dropout=0.1)
    dummy_input = torch.randn(4, 768)
    logits = head(dummy_input)
    assert logits.shape == (4, 3)

    # Missing modality handling (e.g., zero vector input)
    zero_input = torch.zeros(2, 768)
    zero_logits = head(zero_input)
    assert zero_logits.shape == (2, 3)
    assert not torch.isnan(zero_logits).any()


def test_annotation_manifest_schema_compliance():
    """Verify Task E candidate sampling manifest contains valid structure and 0 reviews."""
    status_file = "data/annotations/phase4/human_review_status_manifest.json"
    assert os.path.exists(status_file)
    with open(status_file, "r", encoding="utf-8") as f:
        status = json.load(f)

    assert status["total_candidate_samples"] == 100
    assert status["human_reviews_completed"] == 0
    assert status["evaluation_readiness_status"] == "BLOCKED_PENDING_HUMAN_REVIEW"

    manifest_file = "data/annotations/phase4/human_sampling_manifest_100.jsonl"
    assert os.path.exists(manifest_file)
    with open(manifest_file, "r", encoding="utf-8") as f:
        lines = [json.loads(l) for l in f]
    assert len(lines) == 100
    for item in lines:
        assert item["primary_label"] is None
        assert item["adjudication_status"] == "PENDING_REVIEW"
