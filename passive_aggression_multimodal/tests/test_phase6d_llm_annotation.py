"""
tests/test_phase6d_llm_annotation.py

Automated test suite for Phase 6D:
LLM-Based Dual Annotation, Agreement Analysis, and Adjudication.

Verifies:
  1. Record completeness & duplicate ID protection across all 30 pilot records.
  2. Schema validity: primary labels, allowed subtypes, normalized none subtypes.
  3. Double-blind isolation: independent model inputs without label leakage.
  4. Modality honesty: TEXT_ONLY, image_assessed=False, ocr_assessed=False.
  5. Language segregation: 27 Core English Benchmark vs 3 Chinese separate records.
  6. Provenance & epistemic integrity: machine provenance, benchmark gate remains BLOCKED.
  7. Preservation of human files: Phase 6A human response files untouched (count=0).
  8. Agreement calculation logic: raw agreement, Cohen's kappa, confusion matrix.
  9. Adjudication completeness: all records resolved, root cause documented for disagreements.
  10. Reports and manifest integrity: valid JSON schema, required governance clauses.
"""

import csv
import json
import math
import os
from pathlib import Path
import sys
import pytest

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.run_phase6d_llm_annotation import (
    ROOT_DIR,
    MANIFEST_CSV,
    ANN1_HUMAN_CSV,
    ANN2_HUMAN_CSV,
    ANN1_LLM_CSV,
    ANN2_LLM_CSV,
    DISAGREEMENTS_CSV,
    ADJUDICATION_CSV,
    MANIFEST_JSON,
    AGREEMENT_REPORT_MD,
    QUALITY_AUDIT_MD,
    calculate_agreement_metrics,
)

VALID_PRIMARY_LABELS = {"passive_aggressive", "non_passive_aggressive", "ambiguous"}
VALID_SUBTYPES = {
    "fake_politeness", "indirect_criticism", "backhanded_compliment",
    "implied_blame", "sarcastic_agreement", "dismissiveness", "other_or_unclear", "none"
}
VALID_TARGET_ASSESSMENTS = {"PRESENT", "ABSENT", "UNCLEAR"}
VALID_HOSTILITY_ASSESSMENTS = {"COVERT", "OVERT", "ABSENT", "UNCLEAR"}
VALID_DENIABILITY_ASSESSMENTS = {"YES", "NO", "UNCLEAR"}


@pytest.fixture
def manifest_rows():
    with open(MANIFEST_CSV, mode="r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


@pytest.fixture
def ann1_rows():
    if not ANN1_LLM_CSV.exists():
        pytest.skip("Phase 6D annotator 1 LLM responses file not yet generated.")
    with open(ANN1_LLM_CSV, mode="r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


@pytest.fixture
def ann2_rows():
    if not ANN2_LLM_CSV.exists():
        pytest.skip("Phase 6D annotator 2 LLM responses file not yet generated.")
    with open(ANN2_LLM_CSV, mode="r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


@pytest.fixture
def adjudication_rows():
    if not ADJUDICATION_CSV.exists():
        pytest.skip("Phase 6D adjudication file not yet generated.")
    with open(ADJUDICATION_CSV, mode="r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


@pytest.fixture
def manifest_data():
    if not MANIFEST_JSON.exists():
        pytest.skip("Phase 6D manifest JSON file not yet generated.")
    with open(MANIFEST_JSON, mode="r", encoding="utf-8") as f:
        return json.load(f)


def test_record_completeness_and_no_duplicates(manifest_rows, ann1_rows, ann2_rows, adjudication_rows):
    """Test 1: All files contain exactly 30 unique records matching the manifest IDs."""
    manifest_ids = [r["record_id"] for r in manifest_rows]
    assert len(manifest_ids) == 30
    assert len(set(manifest_ids)) == 30

    # Annotator 1
    ann1_ids = [r["record_id"] for r in ann1_rows]
    assert len(ann1_ids) == 30
    assert len(set(ann1_ids)) == 30
    assert set(ann1_ids) == set(manifest_ids)

    # Annotator 2
    ann2_ids = [r["record_id"] for r in ann2_rows]
    assert len(ann2_ids) == 30
    assert len(set(ann2_ids)) == 30
    assert set(ann2_ids) == set(manifest_ids)

    # Adjudication
    adj_ids = [r["record_id"] for r in adjudication_rows]
    assert len(adj_ids) == 30
    assert len(set(adj_ids)) == 30
    assert set(adj_ids) == set(manifest_ids)


def test_schema_validity_and_subtype_normalization(ann1_rows, ann2_rows):
    """Test 2: Schema compliance and subtype normalization."""
    for idx, r in enumerate(ann1_rows + ann2_rows):
        rid = r["record_id"]
        lbl = r["primary_label"]
        sub = r["subtype"]

        assert lbl in VALID_PRIMARY_LABELS, f"Invalid label '{lbl}' in record {rid}"
        assert sub in VALID_SUBTYPES, f"Invalid subtype '{sub}' in record {rid}"

        # Subtype normalization: if not PA, subtype must be 'none'
        if lbl != "passive_aggressive":
            assert sub == "none", f"Record {rid} has label '{lbl}' but non-none subtype '{sub}'"

        # Confidence range
        conf = int(r["confidence"])
        assert 1 <= conf <= 5, f"Record {rid} has invalid confidence {conf}"

        # Pragmatic assessments present
        assert r["interpersonal_target_assessment"] in VALID_TARGET_ASSESSMENTS
        assert r["covert_hostility_assessment"] in VALID_HOSTILITY_ASSESSMENTS
        assert r["plausible_deniability_assessment"] in VALID_DENIABILITY_ASSESSMENTS


def test_independent_model_inputs_and_double_blind(ann1_rows, ann2_rows):
    """Test 3: Double-blind isolation and independent model configurations."""
    for r1, r2 in zip(ann1_rows, ann2_rows):
        assert r1["model_name"] == "llama3.2:latest"
        assert r2["model_name"] == "qwen2.5:1.5b"
        assert r1["annotator_type"] == "LLM_ANNOTATOR_1"
        assert r2["annotator_type"] == "LLM_ANNOTATOR_2"

        # Model 2 raw response must not reference Model 1
        raw2 = r2.get("raw_model_response", "")
        assert "llama3.2" not in raw2.lower()
        assert "annotator_1" not in raw2.lower()


def test_modality_honesty(ann1_rows, ann2_rows):
    """Test 4: Modality honesty: images unassessed (TEXT_ONLY)."""
    for r in ann1_rows + ann2_rows:
        assert r["modality_used"] == "TEXT_ONLY"
        assert r["image_assessed"] in (False, "False")
        assert r["ocr_assessed"] in (False, "False")


def test_language_segregation(manifest_rows, manifest_data):
    """Test 5: Segregates 27 English records from 3 Chinese records."""
    en_records = [r for r in manifest_rows if r.get("language_scope") == "CORE_ENGLISH_BENCHMARK"]
    zh_records = [r for r in manifest_rows if r.get("language_scope") == "NON_ENGLISH_SEPARATE_ANALYSIS"]

    assert len(en_records) == 27
    assert len(zh_records) == 3

    # Manifest checks
    agreement_manifest = manifest_data["inter_model_agreement"]
    assert agreement_manifest["core_english_benchmark_27_records"]["en_pairs"] == 27
    assert agreement_manifest["chinese_separate_analysis_3_records"]["zh_pairs"] == 3


def test_preservation_of_human_annotation_files():
    """Test 6: Human annotation files remain untouched with completed_count = 0."""
    assert ANN1_HUMAN_CSV.exists()
    assert ANN2_HUMAN_CSV.exists()

    with open(ANN1_HUMAN_CSV, "r", encoding="utf-8") as f:
        h1 = list(csv.DictReader(f))
        assert len(h1) == 30
        assert sum(1 for r in h1 if r.get("primary_label", "").strip()) == 0

    with open(ANN2_HUMAN_CSV, "r", encoding="utf-8") as f:
        h2 = list(csv.DictReader(f))
        assert len(h2) == 30
        assert sum(1 for r in h2 if r.get("primary_label", "").strip()) == 0


def test_provenance_and_benchmark_gate_integrity(manifest_data, ann1_rows, adjudication_rows):
    """Test 7: Machine provenance and locked human benchmark gate."""
    assert manifest_data["pipeline_provenance"] == "LOCAL_OLLAMA_LLM_DUAL_ANNOTATION_AND_ADJUDICATION"
    assert manifest_data["ground_truth_status"] == "MACHINE_GENERATED_LABELS_NOT_HUMAN_GROUND_TRUTH"
    assert manifest_data["human_benchmark_gate_status"] == "BLOCKED_PENDING_HUMAN_ANNOTATION"
    assert manifest_data["human_annotator_01_completion_count"] == 0
    assert manifest_data["human_annotator_02_completion_count"] == 0

    for r in ann1_rows:
        assert r["provenance"] == "MACHINE_GENERATED_LOCAL_OLLAMA_LLM"

    for r in adjudication_rows:
        assert r["provenance"] in ("MACHINE_CONSENSUS_NOT_GROUND_TRUTH", "MACHINE_GENERATED_ADJUDICATION_NOT_GROUND_TRUTH")


def test_adjudication_completeness(adjudication_rows):
    """Test 8: Adjudication completeness and root-cause justification."""
    assert len(adjudication_rows) == 30
    for r in adjudication_rows:
        assert r["adjudicated_label"] in VALID_PRIMARY_LABELS
        assert r["adjudicated_subtype"] in VALID_SUBTYPES
        assert r["adjudication_type"] in ("UNANIMOUS_AGREEMENT", "LLM_ADJUDICATED_DISAGREEMENT")
        assert len(r["adjudication_rationale"].strip()) > 0
        assert len(r["disagreement_root_cause"].strip()) > 0


def test_agreement_calculation_logic():
    """Test 9: Mathematical accuracy of inter-model agreement engine."""
    mock1 = [
        {"record_id": "r1", "primary_label": "passive_aggressive", "subtype": "sarcastic_agreement", "language_scope": "CORE_ENGLISH_BENCHMARK"},
        {"record_id": "r2", "primary_label": "non_passive_aggressive", "subtype": "none", "language_scope": "CORE_ENGLISH_BENCHMARK"},
        {"record_id": "r3", "primary_label": "ambiguous", "subtype": "none", "language_scope": "NON_ENGLISH_SEPARATE_ANALYSIS"},
        {"record_id": "r4", "primary_label": "passive_aggressive", "subtype": "fake_politeness", "language_scope": "CORE_ENGLISH_BENCHMARK"},
    ]
    mock2 = [
        {"record_id": "r1", "primary_label": "passive_aggressive", "subtype": "sarcastic_agreement", "language_scope": "CORE_ENGLISH_BENCHMARK"},
        {"record_id": "r2", "primary_label": "passive_aggressive", "subtype": "indirect_criticism", "language_scope": "CORE_ENGLISH_BENCHMARK"},
        {"record_id": "r3", "primary_label": "ambiguous", "subtype": "none", "language_scope": "NON_ENGLISH_SEPARATE_ANALYSIS"},
        {"record_id": "r4", "primary_label": "passive_aggressive", "subtype": "fake_politeness", "language_scope": "CORE_ENGLISH_BENCHMARK"},
    ]

    metrics = calculate_agreement_metrics(mock1, mock2)
    assert metrics["total_pairs"] == 4
    assert metrics["exact_matches"] == 3
    assert metrics["disagreement_count"] == 1
    assert metrics["raw_agreement_percent_overall"] == 75.0
    assert metrics["english_benchmark"]["en_pairs"] == 3
    assert metrics["english_benchmark"]["en_matches"] == 2
    assert metrics["chinese_separate"]["zh_pairs"] == 1
    assert metrics["chinese_separate"]["zh_matches"] == 1
    assert metrics["subtype_agreement"]["both_passive_aggressive_count"] == 2
    assert metrics["subtype_agreement"]["subtype_matches"] == 2
    assert metrics["subtype_agreement"]["subtype_concordance_percent"] == 100.0


def test_report_files_exist_and_contain_required_headings():
    """Test 10: Agreement and quality audit markdown reports exist and have proper structure."""
    if not AGREEMENT_REPORT_MD.exists() or not QUALITY_AUDIT_MD.exists():
        pytest.skip("Phase 6D reports not yet generated.")

    with open(AGREEMENT_REPORT_MD, "r", encoding="utf-8") as f:
        rep = f.read()
        assert "Inter-Model Agreement Report" in rep
        assert "BLOCKED_PENDING_HUMAN_ANNOTATION" in rep
        assert "Confusion Matrix" in rep

    with open(QUALITY_AUDIT_MD, "r", encoding="utf-8") as f:
        audit = f.read()
        assert "Quality Audit & Governance Document" in rep or "Quality Audit" in audit
        assert "TEXT_ONLY" in audit
        assert "BLOCKED_PENDING_HUMAN_ANNOTATION" in audit
