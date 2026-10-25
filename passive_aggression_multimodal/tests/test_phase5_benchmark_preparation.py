"""Comprehensive Unit and Integration Tests for Phase 5 Benchmark Preparation."""
import json
import os
import tempfile
import pytest
import numpy as np

from src.annotation.quality_analysis import (
    compute_inter_annotator_agreement,
    generate_adjudication_file,
    validate_annotation_row,
)
from src.evaluation.benchmark_evaluator import RealWorldBenchmarkEvaluator
from src.evaluation.human_split_generator import HumanSplitGenerator, compute_sha256_of_file


def test_annotation_schema_validation():
    """Verify that annotation validation catches invalid labels, missing subtypes, and bad ranges."""
    # 1. Valid PA row
    valid_pa = {
        "record_id": "test_001",
        "primary_label": "passive_aggressive",
        "subtype": "fake_politeness",
        "image_role": "HELPFUL",
        "context_sufficiency": "SUFFICIENT",
        "confidence": "4",
    }
    is_valid, errors = validate_annotation_row(valid_pa)
    assert is_valid is True
    assert len(errors) == 0

    # 2. Invalid: PA without subtype
    invalid_no_sub = {
        "record_id": "test_002",
        "primary_label": "passive_aggressive",
        "subtype": "",
    }
    is_valid, errors = validate_annotation_row(invalid_no_sub)
    assert is_valid is False
    assert any("Subtype is required" in e for e in errors)

    # 3. Invalid: unrecognized primary label
    invalid_label = {
        "record_id": "test_003",
        "primary_label": "sarcastic",  # Auxiliary label, illegal as primary PA label
    }
    is_valid, errors = validate_annotation_row(invalid_label)
    assert is_valid is False
    assert any("Invalid primary_label" in e for e in errors)

    # 4. Invalid confidence range
    invalid_conf = {
        "record_id": "test_004",
        "primary_label": "non_passive_aggressive",
        "confidence": "9",
    }
    is_valid, errors = validate_annotation_row(invalid_conf)
    assert is_valid is False
    assert any("out of range" in e for e in errors)


def test_annotator_agreement_calculation():
    """Verify Cohen's kappa and raw agreement computation across matching and disputing annotators."""
    # Mock identical annotations (kappa = 1.0)
    recs1 = {
        f"r{i}": {
            "record_id": f"r{i}",
            "primary_label": "passive_aggressive" if i % 2 == 0 else "non_passive_aggressive",
            "subtype": "fake_politeness" if i % 2 == 0 else "",
            "confidence": "5",
        }
        for i in range(20)
    }
    recs2 = {k: dict(v) for k, v in recs1.items()}

    res_perfect = compute_inter_annotator_agreement(recs1, recs2, target_kappa=0.70)
    assert res_perfect["status"] == "COMPUTED"
    assert res_perfect["raw_agreement_percent"] == 100.0
    assert res_perfect["cohens_kappa"] == 1.0
    assert res_perfect["unanimous_count"] == 20
    assert res_perfect["disputed_count"] == 0
    assert res_perfect["target_achieved"] is True

    # Mock empty / pending annotations (should report awaiting state)
    empty1 = {f"r{i}": {"record_id": f"r{i}", "primary_label": ""} for i in range(10)}
    empty2 = {f"r{i}": {"record_id": f"r{i}", "primary_label": ""} for i in range(10)}
    res_empty = compute_inter_annotator_agreement(empty1, empty2)
    assert res_empty["status"] == "AWAITING_HUMAN_REVIEWS"
    assert res_empty["cohens_kappa"] is None


def test_adjudication_file_generation(tmp_path):
    """Verify that adjudication queue correctly flags unanimous matches and disputes."""
    recs1 = {
        "r1": {"record_id": "r1", "primary_label": "passive_aggressive", "subtype": "fake_politeness"},
        "r2": {"record_id": "r2", "primary_label": "passive_aggressive", "subtype": "implied_blame"},
    }
    recs2 = {
        "r1": {"record_id": "r1", "primary_label": "passive_aggressive", "subtype": "fake_politeness"},
        "r2": {"record_id": "r2", "primary_label": "non_passive_aggressive", "subtype": ""},  # Dispute
    }
    out_csv = str(tmp_path / "adjudication_test.csv")
    count = generate_adjudication_file(recs1, recs2, out_csv)
    assert count == 2

    import csv
    with open(out_csv, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    assert rows[0]["agreement_type"] == "UNANIMOUS_AGREEMENT"
    assert rows[0]["adjudicated_primary_label"] == "passive_aggressive"
    assert rows[1]["agreement_type"] == "DISPUTE_REQUIRES_ADJUDICATION"
    assert rows[1]["adjudicated_primary_label"] == ""


def test_human_split_generator_threshold_blocking(tmp_path):
    """Verify that split generator blocks if verified records are below threshold."""
    gen = HumanSplitGenerator(min_required_records=30)
    empty_file = str(tmp_path / "empty.jsonl")
    with open(empty_file, "w", encoding="utf-8") as f:
        pass  # 0 records

    res = gen.generate_splits(empty_file, str(tmp_path / "splits"))
    assert res["status"] == "SPLIT_GENERATION_BLOCKED"
    assert res["verified_records_available"] == 0
    assert res["minimum_required_threshold"] == 30


def test_group_leakage_prevention_in_split_generator(tmp_path):
    """Verify that conversation clusters are never split across train and test."""
    mock_records = []
    # 40 records across 4 conversations (10 per conversation)
    for c in range(4):
        for i in range(10):
            mock_records.append({
                "record_id": f"rec_{c}_{i}",
                "conversation_id": f"conv_thread_{c}",
                "primary_label": "passive_aggressive" if i % 2 == 0 else "non_passive_aggressive",
                "adjudication_status": "RESOLVED_UNANIMOUS",
                "comment_text": f"Comment {c} {i}",
            })

    input_file = str(tmp_path / "mock_adjudicated.jsonl")
    with open(input_file, "w", encoding="utf-8") as f:
        for r in mock_records:
            f.write(json.dumps(r) + "\n")

    gen = HumanSplitGenerator(min_required_records=20, train_ratio=0.5, val_ratio=0.25, test_ratio=0.25, seed=42)
    split_dir = str(tmp_path / "splits_out")
    res = gen.generate_splits(input_file, split_dir)
    assert res["status"] == "SPLITS_GENERATED"

    # Verify conversation ID disjointness across train, val, test
    train_convs = set()
    with open(res["manifests"]["train"]["path"], "r", encoding="utf-8") as f:
        for line in f:
            train_convs.add(json.loads(line)["conversation_id"])

    test_convs = set()
    with open(res["manifests"]["test"]["path"], "r", encoding="utf-8") as f:
        for line in f:
            test_convs.add(json.loads(line)["conversation_id"])

    assert len(train_convs.intersection(test_convs)) == 0, "Conversation thread must NOT leak between train and test!"


def test_benchmark_evaluator_interface(tmp_path):
    """Verify that RealWorldBenchmarkEvaluator evaluates records, computes CIs, and exports predictions."""
    evaluator = RealWorldBenchmarkEvaluator()

    mock_test = [
        {"record_id": f"t_{i}", "primary_label": "passive_aggressive", "image_available": True}
        for i in range(10)
    ] + [
        {"record_id": f"t_{i+10}", "primary_label": "non_passive_aggressive", "image_available": False}
        for i in range(10)
    ]

    # Mock predictions and probabilities
    y_pred = [0] * 10 + [1] * 10
    y_prob = [[0.9, 0.05, 0.05]] * 10 + [[0.1, 0.85, 0.05]] * 10

    pred_out = str(tmp_path / "eval_preds.jsonl")
    res = evaluator.evaluate_predictions(
        records=mock_test,
        y_pred=y_pred,
        y_prob=y_prob,
        model_id="test_gated_fusion",
        split_id="human_gold_test",
        output_prediction_file=pred_out,
        n_bootstraps=50,
    )

    assert res["total_evaluated_records"] == 20
    assert res["metrics"]["accuracy"] == 1.0
    assert "bootstrap_ci_95" in res
    assert "calibration" in res
    assert os.path.exists(pred_out)
