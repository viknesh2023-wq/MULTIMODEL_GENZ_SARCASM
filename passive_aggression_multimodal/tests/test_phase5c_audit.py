"""
Automated Integrity and Completeness Tests for Phase 5C Deep Audit.
"""
import csv
import json
import os
import pytest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TABLES_DIR = os.path.join(PROJECT_ROOT, "reports", "phase5c", "tables")


def test_table1_label_distribution():
    """Verify model_label_distribution.csv counts and exact percentages."""
    csv_path = os.path.join(TABLES_DIR, "model_label_distribution.csv")
    assert os.path.exists(csv_path), "model_label_distribution.csv missing"

    with open(csv_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 6, f"Expected 6 rows, got {len(rows)}"

    # Check 100-record Llama 3.2
    llama_100 = next(r for r in rows if r["dataset_scope"] == "100-record manifest" and r["model_name"] == "llama3.2:latest")
    assert int(llama_100["total_evaluated"]) == 100
    assert int(llama_100["passive_aggressive_count"]) == 96
    assert int(llama_100["non_passive_aggressive_count"]) == 4
    assert int(llama_100["ambiguous_count"]) == 0

    # Check 100-record Qwen 2.5
    qwen_100 = next(r for r in rows if r["dataset_scope"] == "100-record manifest" and r["model_name"] == "qwen2.5:1.5b")
    assert int(qwen_100["total_evaluated"]) == 100
    assert int(qwen_100["passive_aggressive_count"]) == 68
    assert int(qwen_100["non_passive_aggressive_count"]) == 28
    assert int(qwen_100["ambiguous_count"]) == 4


def test_table2_disagreement_audit_completeness():
    """Verify disagreement_audit_32.csv has exactly 32 records and required fields."""
    csv_path = os.path.join(TABLES_DIR, "disagreement_audit_32.csv")
    assert os.path.exists(csv_path), "disagreement_audit_32.csv missing"

    with open(csv_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 32, f"Expected 32 disagreements, got {len(rows)}"

    for r in rows:
        assert r["record_id"]
        assert r["comment_text"]
        assert r["model_a_label"] != r["model_b_label"]
        assert r["image_actually_analyzed"] == "No (UNASSESSED_VISION_MODEL_ABSENT)"
        assert r["disagreement_category"] in (
            "Factual report or news",
            "Sincere praise",
            "Direct aggression or direct argument",
            "Situational venting",
            "Sarcasm without passive aggression",
            "Friendly teasing",
            "Missing context",
            "Genuine passive-aggression candidate",
            "Other or unresolved",
        )


def test_table3_llama_non_pa_4_audit():
    """Verify llama_non_passive_aggressive_audit_4.csv contains the exact 4 records."""
    csv_path = os.path.join(TABLES_DIR, "llama_non_passive_aggressive_audit_4.csv")
    assert os.path.exists(csv_path), "llama_non_passive_aggressive_audit_4.csv missing"

    with open(csv_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 4, f"Expected 4 non-PA records, got {len(rows)}"
    record_ids = {r["record_id"] for r in rows}
    expected_ids = {"sarcnet_test_00053", "mmsd_train_16580", "mmsd_validation_02047", "sarcnet_test_00188"}
    assert record_ids == expected_ids, f"Expected {expected_ids}, got {record_ids}"

    # Verify both models agreed on sarcnet_test_00053
    sarcnet_rec = next(r for r in rows if r["record_id"] == "sarcnet_test_00053")
    assert sarcnet_rec["both_models_agreed"] == "Yes"


def test_table4_agreement_audit_68_completeness():
    """Verify agreement_audit_68.csv has exactly 68 records and correct consensus counts."""
    csv_path = os.path.join(TABLES_DIR, "agreement_audit_68.csv")
    assert os.path.exists(csv_path), "agreement_audit_68.csv missing"

    with open(csv_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 68, f"Expected 68 agreements, got {len(rows)}"

    pa_agreed = sum(1 for r in rows if r["agreed_primary_label"] == "passive_aggressive")
    non_pa_agreed = sum(1 for r in rows if r["agreed_primary_label"] == "non_passive_aggressive")
    assert pa_agreed == 67
    assert non_pa_agreed == 1


def test_table5_prompt_v1_v2_comparison():
    """Verify prompt_v1_v2_comparison.csv contains all 12 diagnostic records."""
    csv_path = os.path.join(TABLES_DIR, "prompt_v1_v2_comparison.csv")
    assert os.path.exists(csv_path), "prompt_v1_v2_comparison.csv missing"

    with open(csv_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 12, f"Expected 12 records, got {len(rows)}"
    assert all("v2_llama_rationale" in r and "v2_qwen_rationale" in r for r in rows)


def test_table6_and_7_integrity():
    """Verify modality coverage and annotation failure tables."""
    t6 = os.path.join(TABLES_DIR, "modality_coverage.csv")
    t7 = os.path.join(TABLES_DIR, "annotation_failure_summary.csv")
    assert os.path.exists(t6)
    assert os.path.exists(t7)

    with open(t6, "r", encoding="utf-8") as f:
        rows6 = list(csv.DictReader(f))
    total_row = next(r for r in rows6 if r["dataset_name"] == "TOTAL / OVERALL")
    assert int(total_row["total_records"]) == 100
    assert int(total_row["genuine_multimodal_vision_count"]) == 0
    assert int(total_row["image_file_exists"]) == 60


def test_human_ground_truth_status_preserved():
    """Verify human review manifest remains strictly blocked at zero reviews."""
    status_path = os.path.join(PROJECT_ROOT, "data", "annotations", "phase5", "human_review_status_manifest.json")
    with open(status_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    assert manifest["annotator_01"]["completed_count"] == 0
    assert manifest["annotator_02"]["completed_count"] == 0
    assert manifest["evaluation_readiness_status"] == "BLOCKED_PENDING_HUMAN_ANNOTATION"
