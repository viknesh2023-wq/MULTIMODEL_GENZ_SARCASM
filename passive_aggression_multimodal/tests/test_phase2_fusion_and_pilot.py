"""Tests for Phase 2 feature extraction interfaces, fusion classifier, and annotation pilot."""
from __future__ import annotations

import json
from pathlib import Path
import pytest
import torch

from src.preprocessing.feature_extraction import VisionFeatureExtractor, TextFeatureExtractor
from src.preprocessing.fusion_interface import PassiveAggressionMultimodalModel, validate_fusion_shapes

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_feature_extractors_dimensions_and_stubs():
    # Vision extractor interface
    v_extractor = VisionFeatureExtractor(output_dim=768, offline_test_mode=True)
    v_out = v_extractor.extract_batch([None, None])
    assert v_out["embeddings"].shape == (2, 768)
    assert v_out["modality_mask"].shape == (2,)
    assert not v_out["modality_mask"][0]  # Both were None

    # Text extractor interface
    t_extractor = TextFeatureExtractor(output_dim=768, offline_test_mode=True)
    t_out = t_extractor.extract_batch(["Hello world", "Passive aggressive remark"])
    assert t_out["embeddings"].shape == (2, 768)
    assert t_out["output_dim"] == 768


def test_fusion_model_forward_pass_and_shapes():
    shapes_res = validate_fusion_shapes(batch_size=4)
    assert shapes_res["status"] == "VALIDATED"
    assert shapes_res["pa_logits_shape"] == [4, 3]
    assert shapes_res["sarcasm_logits_shape"] == [4, 3]
    assert shapes_res["subtype_logits_shape"] == [4, 7]
    assert shapes_res["fused_dim"] == [4, 512]
    assert shapes_res["gate_weights_shape"] == [4, 4]


def test_annotation_pilot_records_integrity():
    pilot_path = PROJECT_ROOT / "data" / "processed" / "annotation_pilot_150.jsonl"
    queue_path = PROJECT_ROOT / "data" / "processed" / "human_review_queue_150.json"

    assert pilot_path.exists(), "annotation_pilot_150.jsonl must exist"
    assert queue_path.exists(), "human_review_queue_150.json must exist"

    records = []
    with open(pilot_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    assert len(records) == 150, f"Expected exactly 150 records, got {len(records)}"

    for r in records:
        # Check integrity of governance fields
        assert r["synthetic_status"] == "REAL", "No synthetic data allowed in real annotation pilot"
        assert r["human_verification_status"] == "AWAITING_HUMAN_REVIEW", "Gold standard must be pending"
        assert r["human_verified_label"] is None, "Human verified label must be None prior to human review"
        assert r["independent_sarcasm_label"] in ["sarcastic", "non_sarcastic", "ambiguous"]
        assert r["provisional_pa_label"] in ["passive_aggressive", "non_passive_aggressive", "ambiguous"]
        assert r["vision_annotator_output"]["execution_status"] == "INTERFACE_STUB_PENDING_VLM_WEIGHTS"
