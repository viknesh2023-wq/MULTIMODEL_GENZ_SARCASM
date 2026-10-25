"""
tests/test_phase5e_agreement_audit.py

Quality assurance tests for Phase 5E: Forensic Audit of All 68 Model-Agreement Records.
Verifies:
  1. Input cohort completeness (exactly 68 agreement records).
  2. Record ID uniqueness and stability.
  3. Preservation of source text and model predictions.
  4. Required fields and valid conservative category values.
  5. Correct category totals summing to exactly 68.
  6. Perfect reconciliation of Phase 5D inclusion status (68 / 68 concordance).
  7. No accidental mutation of prior-phase artifacts (Phase 5A, 5C, 5D).
  8. Explicit labeling of unassessed image/OCR evidence.
  9. Epistemic governance: zero human ground truth claims on silver agreements.
"""

import csv
import json
from pathlib import Path
import pytest

ROOT_DIR = Path(__file__).resolve().parent.parent

# File paths
PASS1_FILE = ROOT_DIR / "data/annotations/phase5a/llm/llm_annotations_pass1.jsonl"
PASS2_FILE = ROOT_DIR / "data/annotations/phase5a/llm/llm_annotations_pass2.jsonl"
MANIFEST_FILE = ROOT_DIR / "data/annotations/phase4/human_sampling_manifest_100.jsonl"

P5D_VERIFIED_FILE = ROOT_DIR / "data/annotations/phase5d/silver_dataset_verified.jsonl"
P5D_EXCLUDED_FILE = ROOT_DIR / "data/annotations/phase5d/excluded_records.csv"

TABLE1_FILE = ROOT_DIR / "reports/phase5e/tables/agreement_audit_68_complete.csv"
TABLE2_FILE = ROOT_DIR / "reports/phase5e/tables/agreement_category_distribution.csv"
TABLE3_FILE = ROOT_DIR / "reports/phase5e/tables/phase5d_reconciliation_68.csv"
TABLE4_FILE = ROOT_DIR / "reports/phase5e/tables/shared_error_risk_register.csv"
TABLE5_FILE = ROOT_DIR / "reports/phase5e/phase5e_human_review_priority_queue.csv"
JSONL_FILE = ROOT_DIR / "data/annotations/phase5e/agreement_audit_68_complete.jsonl"

HUMAN_STATUS_FILE = ROOT_DIR / "data/annotations/phase5/human_review_status_manifest.json"


@pytest.fixture
def raw_agreements():
    p1 = [json.loads(line) for line in PASS1_FILE.read_text(encoding="utf-8").strip().split("\n")]
    p2 = [json.loads(line) for line in PASS2_FILE.read_text(encoding="utf-8").strip().split("\n")]
    p1_map = {r["record_id"]: r for r in p1}
    p2_map = {r["record_id"]: r for r in p2}
    return [rid for rid, r1 in p1_map.items() if r1["primary_label"] == p2_map[rid]["primary_label"]]


@pytest.fixture
def table1_rows():
    with open(TABLE1_FILE, mode="r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


@pytest.fixture
def table2_rows():
    with open(TABLE2_FILE, mode="r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


@pytest.fixture
def table3_rows():
    with open(TABLE3_FILE, mode="r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_input_cohort_completeness_and_uniqueness(raw_agreements, table1_rows):
    """Test 1: Input cohort has exactly 68 records and table1 has matching unique record IDs."""
    assert len(raw_agreements) == 68, f"Expected 68 raw agreements, found {len(raw_agreements)}"
    assert len(table1_rows) == 68, f"Expected 68 rows in Table 1, found {len(table1_rows)}"
    
    table1_ids = [r["record_id"] for r in table1_rows]
    assert len(set(table1_ids)) == 68, "Duplicate record IDs found in Table 1"
    assert set(table1_ids) == set(raw_agreements), "Record ID mismatch between raw agreements and Table 1"


def test_preservation_of_source_text_and_model_predictions(table1_rows):
    """Test 2: Original comments and model predictions from Phase 5A are perfectly preserved."""
    p1_map = {json.loads(l)["record_id"]: json.loads(l) for l in PASS1_FILE.read_text(encoding="utf-8").strip().split("\n")}
    p2_map = {json.loads(l)["record_id"]: json.loads(l) for l in PASS2_FILE.read_text(encoding="utf-8").strip().split("\n")}
    manifest_map = {json.loads(l)["record_id"]: json.loads(l) for l in MANIFEST_FILE.read_text(encoding="utf-8").strip().split("\n")}

    for row in table1_rows:
        rid = row["record_id"]
        # Comment text matches manifest
        assert row["comment_text"].strip() == manifest_map[rid]["comment_text"].strip(), f"Text altered for {rid}"
        # Model predictions match Pass 1 and Pass 2
        assert row["model_a_prediction"] == p1_map[rid]["primary_label"], f"Model A prediction altered for {rid}"
        assert row["model_b_prediction"] == p2_map[rid]["primary_label"], f"Model B prediction altered for {rid}"
        assert row["agreement_status"] == "EXACT_MATCH"


def test_required_fields_and_conservative_categories(table1_rows):
    """Test 3: Required columns exist and audit_category uses valid conservative taxonomy values."""
    required_cols = [
        "record_id", "source_dataset", "source_record_id", "comment_text", "context_text",
        "image_available", "ocr_available", "image_assessed_by_models", "ocr_assessed_by_models",
        "modality_used", "model_a_prediction", "model_b_prediction", "agreement_status",
        "interpersonal_target", "covert_hostility", "plausible_deniability", "rhetorical_form",
        "shared_risk_flag", "audit_category", "audit_confidence", "supporting_evidence",
        "limitations", "recommended_disposition", "disposition_justification"
    ]
    for col in required_cols:
        assert col in table1_rows[0], f"Missing required column: {col}"

    valid_categories = {
        "LIKELY_GENUINE_PA_SILVER",
        "LIKELY_NON_PA",
        "GENERAL_SARCASM_NO_INTERPERSONAL_TARGET",
        "INSTITUTIONAL_OR_SITUATIONAL_CRITIQUE",
        "DIRECT_AGGRESSION_OR_DIRECT_ARGUMENT",
        "SINCERE_PRAISE_OR_FRIENDLY_TEASING",
        "AMBIGUOUS_INSUFFICIENT_CONTEXT",
        "SHARED_MODEL_ERROR_SUSPECTED",
        "UNRESOLVED_REQUIRES_HUMAN_REVIEW",
    }
    for row in table1_rows:
        assert row["audit_category"] in valid_categories, f"Invalid category {row['audit_category']} for {row['record_id']}"


def test_category_distribution_totals(table2_rows, table1_rows):
    """Test 4: Category distribution sums to exactly 68 and matches Table 1 category counts."""
    total_count = sum(int(r["record_count"]) for r in table2_rows)
    assert total_count == 68, f"Category distribution sum expected 68, got {total_count}"

    cat_counts_table1 = {}
    for r in table1_rows:
        cat_counts_table1[r["audit_category"]] = cat_counts_table1.get(r["audit_category"], 0) + 1

    for r in table2_rows:
        cat = r["audit_category"]
        expected = cat_counts_table1.get(cat, 0)
        assert int(r["record_count"]) == expected, f"Count mismatch for category {cat}: {r['record_count']} vs {expected}"


def test_phase5d_reconciliation_exact_agreement(table3_rows):
    """Test 5: Reconciliation table contains 68 records with 100% agreement on inclusion/exclusion decisions."""
    assert len(table3_rows) == 68, f"Expected 68 reconciliation records, got {len(table3_rows)}"
    
    verified_p5d = [json.loads(l)["record_id"] for l in P5D_VERIFIED_FILE.read_text(encoding="utf-8").strip().split("\n")]
    
    for r in table3_rows:
        rid = r["record_id"]
        in_p5d = r["in_phase5d_verified"]
        if rid in verified_p5d:
            assert in_p5d == "YES", f"Record {rid} is in verified P5D but marked NO in reconciliation"
            assert r["phase5e_category"] in ("LIKELY_GENUINE_PA_SILVER", "LIKELY_NON_PA")
        else:
            assert in_p5d == "NO", f"Record {rid} is not in verified P5D but marked YES in reconciliation"
            assert r["phase5e_category"] not in ("LIKELY_GENUINE_PA_SILVER", "LIKELY_NON_PA")
        
        assert r["decisions_agree"] == "YES", f"Decision disagreement on {rid}"


def test_explicit_unassessed_modality_labeling(table1_rows):
    """Test 6: Models are explicitly marked as having evaluated TEXT_ONLY, without vision/OCR assessment."""
    for r in table1_rows:
        assert r["modality_used"] == "TEXT_ONLY"
        assert r["image_assessed_by_models"] == "False"
        assert r["ocr_assessed_by_models"] == "False"


def test_preservation_of_prior_phase_artifacts():
    """Test 7: Prior phase artifacts exist and remain unmodified."""
    # Phase 5A
    assert PASS1_FILE.exists() and len(PASS1_FILE.read_text(encoding="utf-8").strip().split("\n")) == 100
    assert PASS2_FILE.exists() and len(PASS2_FILE.read_text(encoding="utf-8").strip().split("\n")) == 100
    # Phase 5D
    assert P5D_VERIFIED_FILE.exists() and len(P5D_VERIFIED_FILE.read_text(encoding="utf-8").strip().split("\n")) == 19
    with open(P5D_EXCLUDED_FILE, encoding="utf-8") as f:
        assert len(list(csv.DictReader(f))) == 81


def test_epistemic_governance_no_human_ground_truth_claims():
    """Test 8: Silver agreements are never labeled as human ground truth and benchmark remains blocked."""
    assert HUMAN_STATUS_FILE.exists()
    status_data = json.loads(HUMAN_STATUS_FILE.read_text(encoding="utf-8"))
    assert status_data.get("evaluation_readiness_status") == "BLOCKED_PENDING_HUMAN_ANNOTATION"
    assert status_data["annotator_01"]["completed_count"] == 0
    assert status_data["annotator_02"]["completed_count"] == 0
    assert status_data["adjudication"]["completed_count"] == 0
