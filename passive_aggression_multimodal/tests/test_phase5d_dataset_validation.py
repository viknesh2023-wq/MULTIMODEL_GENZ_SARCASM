"""
Automated Integrity, Leakage, and Provenance Tests for Phase 5D Dataset Validation.
"""
import csv
import json
import os
import re
import pytest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PHASE5D_DIR = os.path.join(PROJECT_ROOT, "data", "annotations", "phase5d")


def normalize_text(text: str) -> str:
    """Normalize text for near-duplicate and leakage checks."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return " ".join(text.split())


def test_phase5d_artifacts_existence():
    """Verify all 5 Phase 5D data artifacts exist on disk."""
    candidates_path = os.path.join(PHASE5D_DIR, "silver_dataset_candidates.jsonl")
    verified_path = os.path.join(PHASE5D_DIR, "silver_dataset_verified.jsonl")
    excluded_path = os.path.join(PHASE5D_DIR, "excluded_records.csv")
    negative_audit_path = os.path.join(PHASE5D_DIR, "negative_candidate_audit.csv")
    manifest_path = os.path.join(PHASE5D_DIR, "dataset_manifest.json")

    assert os.path.exists(candidates_path), "Missing silver_dataset_candidates.jsonl"
    assert os.path.exists(verified_path), "Missing silver_dataset_verified.jsonl"
    assert os.path.exists(excluded_path), "Missing excluded_records.csv"
    assert os.path.exists(negative_audit_path), "Missing negative_candidate_audit.csv"
    assert os.path.exists(manifest_path), "Missing dataset_manifest.json"


def test_manifest_counts_and_partition_consistency():
    """Verify candidate pool partitions exactly into verified and excluded subsets."""
    candidates_path = os.path.join(PHASE5D_DIR, "silver_dataset_candidates.jsonl")
    verified_path = os.path.join(PHASE5D_DIR, "silver_dataset_verified.jsonl")
    excluded_path = os.path.join(PHASE5D_DIR, "excluded_records.csv")
    manifest_path = os.path.join(PHASE5D_DIR, "dataset_manifest.json")

    with open(candidates_path, "r", encoding="utf-8") as f:
        candidates = [json.loads(l) for l in f if l.strip()]
    with open(verified_path, "r", encoding="utf-8") as f:
        verified = [json.loads(l) for l in f if l.strip()]
    with open(excluded_path, "r", encoding="utf-8") as f:
        excluded = list(csv.DictReader(f))
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    assert len(candidates) == 100, f"Expected 100 candidates, got {len(candidates)}"
    assert len(verified) == 19, f"Expected 19 verified records, got {len(verified)}"
    assert len(excluded) == 81, f"Expected 81 excluded records, got {len(excluded)}"

    # Conservation: 19 + 81 = 100
    assert len(verified) + len(excluded) == len(candidates)

    # Manifest reflection
    assert manifest["candidate_pool_count"] == 100
    assert manifest["verified_subset_count"] == 19
    assert manifest["excluded_record_count"] == 81
    assert manifest["verified_class_distribution"]["passive_aggressive"] == 18
    assert manifest["verified_class_distribution"]["non_passive_aggressive"] == 1
    assert manifest["verified_class_distribution"]["ambiguous"] == 0


def test_no_duplicate_ids_or_near_duplicate_leakage():
    """Verify zero duplicate IDs and zero near-duplicate normalized text collisions."""
    verified_path = os.path.join(PHASE5D_DIR, "silver_dataset_verified.jsonl")
    with open(verified_path, "r", encoding="utf-8") as f:
        records = [json.loads(l) for l in f if l.strip()]

    ids = [r["record_id"] for r in records]
    assert len(ids) == len(set(ids)), "Duplicate record ID found in verified subset!"

    normalized_texts = [normalize_text(r["comment_text"]) for r in records]
    assert len(normalized_texts) == len(set(normalized_texts)), "Duplicate normalized text found in verified subset!"


def test_immutable_provenance_and_tagging():
    """Verify every verified record contains the required TRAINED_ON_LLM_SILVER_DATA metadata."""
    verified_path = os.path.join(PHASE5D_DIR, "silver_dataset_verified.jsonl")
    with open(verified_path, "r", encoding="utf-8") as f:
        records = [json.loads(l) for l in f if l.strip()]

    for r in records:
        prov = r.get("provenance_metadata", {})
        assert prov.get("provenance_tag") == "TRAINED_ON_LLM_SILVER_DATA", f"Missing provenance tag in {r['record_id']}"
        assert prov.get("model_a") == "llama3.2:latest"
        assert prov.get("model_b") == "qwen2.5:1.5b"
        assert prov.get("prompt_version") == "v1.0.0"
        assert prov.get("epistemic_status") == "SILVER_WEAK_SUPERVISION_ONLY"
        assert r["audit_decision"] in ("VERIFIED_SILVER_POSITIVE", "VERIFIED_SILVER_NEGATIVE")


def test_modality_honesty_in_verified_subset():
    """Verify text-only execution flag and no visual hallucination in verified records."""
    verified_path = os.path.join(PHASE5D_DIR, "silver_dataset_verified.jsonl")
    with open(verified_path, "r", encoding="utf-8") as f:
        records = [json.loads(l) for l in f if l.strip()]

    for r in records:
        assert r["modality_used"] == "TEXT_ONLY", f"Invalid modality in {r['record_id']}"
        if r["image_available"]:
            assert r["image_role"] == "UNASSESSED_VISION_MODEL_ABSENT", f"Invalid image_role in {r['record_id']}"
        else:
            assert r["image_role"] == "ABSENT"


def test_unverified_negatives_isolation():
    """Verify that unverified negative candidates are never admitted into verified subset."""
    verified_path = os.path.join(PHASE5D_DIR, "silver_dataset_verified.jsonl")
    negative_audit_path = os.path.join(PHASE5D_DIR, "negative_candidate_audit.csv")

    with open(verified_path, "r", encoding="utf-8") as f:
        verified_ids = {json.loads(l)["record_id"] for l in f if l.strip()}

    with open(negative_audit_path, "r", encoding="utf-8") as f:
        neg_candidates = list(csv.DictReader(f))

    for neg in neg_candidates:
        if neg["is_verified_non_pa"] != "Yes":
            assert neg["record_id"] not in verified_ids, (
                f"Unverified negative candidate {neg['record_id']} leaked into verified subset!"
            )


def test_human_ground_truth_isolation():
    """Verify human status manifest remains untouched and evaluation remains gated."""
    human_status_path = os.path.join(PROJECT_ROOT, "data", "annotations", "phase5", "human_review_status_manifest.json")
    with open(human_status_path, "r", encoding="utf-8") as f:
        status_data = json.load(f)

    assert status_data["annotator_01"]["completed_count"] == 0
    assert status_data["annotator_02"]["completed_count"] == 0
    assert status_data["evaluation_readiness_status"] == "BLOCKED_PENDING_HUMAN_ANNOTATION"
