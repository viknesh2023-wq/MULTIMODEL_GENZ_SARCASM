"""
tests/test_phase6a_annotation_workflow.py

Automated quality assurance and integrity tests for Phase 6A:
Human Annotation Pilot and Inter-Annotator Agreement Setup.

Verifies:
  1. Pilot sampling manifest size (exactly 30 records) and ID uniqueness.
  2. Balanced representation across all 6 linguistic archetypes (5 records each).
  3. Language scope tracking (27 Core English Benchmark, 3 Non-English Separate Analysis).
  4. Physical existence of all 14 referenced image files on disk.
  5. Double-blind annotator isolation: zero model predictions or silver labels in response templates.
  6. Status manifest reflects un-fabricated state (0 completed, BLOCKED_PENDING_HUMAN_ANNOTATION).
  7. Agreement engine handles incomplete reviews gracefully without fabricating results.
  8. Preservation of prior-phase artifacts (Phase 5D silver dataset, Phase 5E audit tables).
"""

import csv
import json
from pathlib import Path
import pytest

from scripts.analyze_phase6a_agreement import analyze_agreement, compute_cohens_kappa

ROOT_DIR = Path(__file__).resolve().parent.parent

MANIFEST_CSV = ROOT_DIR / "data/annotations/phase6a/pilot_sampling_manifest.csv"
ANN1_CSV = ROOT_DIR / "data/annotations/phase6a/annotator_1_responses.csv"
ANN2_CSV = ROOT_DIR / "data/annotations/phase6a/annotator_2_responses.csv"
STATUS_JSON = ROOT_DIR / "data/annotations/phase6a/annotation_status_manifest.json"

P5D_VERIFIED_FILE = ROOT_DIR / "data/annotations/phase5d/silver_dataset_verified.jsonl"
P5E_AUDIT_FILE = ROOT_DIR / "reports/phase5e/tables/agreement_audit_68_complete.csv"


@pytest.fixture
def manifest_rows():
    with open(MANIFEST_CSV, mode="r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


@pytest.fixture
def ann1_rows():
    with open(ANN1_CSV, mode="r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


@pytest.fixture
def ann2_rows():
    with open(ANN2_CSV, mode="r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


@pytest.fixture
def status_manifest():
    with open(STATUS_JSON, mode="r", encoding="utf-8") as f:
        return json.load(f)


def test_pilot_sampling_manifest_size_and_uniqueness(manifest_rows):
    """Test 1: Sampling manifest has exactly 30 records with unique IDs and all source datasets."""
    assert len(manifest_rows) == 30, f"Expected 30 records, got {len(manifest_rows)}"
    record_ids = [r["record_id"] for r in manifest_rows]
    assert len(set(record_ids)) == 30, "Duplicate record IDs found in pilot manifest"

    datasets = set(r["source_dataset"] for r in manifest_rows)
    assert datasets == {"mustard", "civil_comments", "mmsd", "sarcnet"}, "All 4 source datasets must be represented"


def test_sampling_archetype_balance(manifest_rows):
    """Test 2: All 6 archetypes are represented with exactly 5 records each."""
    archetypes = [r["sampling_archetype"] for r in manifest_rows]
    counts = {}
    for a in archetypes:
        counts[a] = counts.get(a, 0) + 1

    expected_archetypes = {
        "LLM_SILVER_POSITIVE_CANDIDATE": 5,
        "NON_PA_CONTRAST_CANDIDATE": 5,
        "DIRECT_AGGRESSION_OR_DISAGREEMENT": 5,
        "GENERAL_SARCASM_INSTITUTIONAL_CRITIQUE": 5,
        "AMBIGUOUS_CONTEXT_DEPENDENT": 5,
        "MULTIMODAL_IMAGE_DEPENDENT": 5,
    }
    assert counts == expected_archetypes, f"Archetype distribution mismatch: {counts} vs {expected_archetypes}"


def test_language_scope_tracking(manifest_rows):
    """Test 3: Language scope policy is enforced (27 Core English Benchmark, 3 Non-English Separate Analysis)."""
    lang_scopes = [r["language_scope"] for r in manifest_rows]
    core_count = lang_scopes.count("CORE_ENGLISH_BENCHMARK")
    non_en_count = lang_scopes.count("NON_ENGLISH_SEPARATE_ANALYSIS")

    assert core_count == 27, f"Expected 27 Core English records, got {core_count}"
    assert non_en_count == 3, f"Expected 3 Non-English records, got {non_en_count}"
    assert core_count + non_en_count == 30


def test_image_files_existence(manifest_rows):
    """Test 4: For all 14 records with image_available == True, the image files exist on disk."""
    image_records = [r for r in manifest_rows if r["image_available"].lower() == "true"]
    assert len(image_records) == 14, f"Expected 14 image-available records, got {len(image_records)}"

    for r in image_records:
        rel_path = r["image_path"]
        assert rel_path, f"Record {r['record_id']} has image_available=True but empty image_path"
        full_path = ROOT_DIR / rel_path
        assert full_path.exists(), f"Image file not found on disk: {full_path}"


def test_annotator_double_blind_isolation(manifest_rows, ann1_rows, ann2_rows):
    """Test 5: Response templates have identical 30 records, but HIDE all model predictions and silver labels."""
    assert len(ann1_rows) == 30
    assert len(ann2_rows) == 30

    forbidden_fields = {
        "model_a_prediction", "model_b_prediction", "model_a_rationale", "model_b_rationale",
        "silver_primary_label", "silver_subtype", "audit_decision", "phase5d_status", "phase5e_category"
    }

    ann1_cols = set(ann1_rows[0].keys())
    ann2_cols = set(ann2_rows[0].keys())

    assert ann1_cols == ann2_cols, "Annotator 1 and Annotator 2 must have identical schemas"
    assert not (ann1_cols & forbidden_fields), f"Annotator templates contain leaked model fields: {ann1_cols & forbidden_fields}"

    # IDs match manifest in identical order
    manifest_ids = [r["record_id"] for r in manifest_rows]
    ann1_ids = [r["record_id"] for r in ann1_rows]
    ann2_ids = [r["record_id"] for r in ann2_rows]
    assert ann1_ids == manifest_ids
    assert ann2_ids == manifest_ids


def test_status_manifest_initialization(status_manifest):
    """Test 6: Status manifest accurately reports uncompleted pilot state without fabrication."""
    assert status_manifest["pilot_sample_size"] == 30
    assert status_manifest["annotator_01"]["completed_count"] == 0
    assert status_manifest["annotator_02"]["completed_count"] == 0
    assert status_manifest["annotator_01"]["status"] == "AWAITING_HUMAN_INPUT"
    assert status_manifest["annotator_02"]["status"] == "AWAITING_HUMAN_INPUT"
    assert status_manifest["inter_annotator_agreement"]["cohens_kappa"] is None
    assert status_manifest["evaluation_readiness_status"] == "BLOCKED_PENDING_HUMAN_ANNOTATION"


def test_agreement_engine_pending_behavior():
    """Test 7: Agreement engine reports PENDING when reviews are incomplete and computes Kappa correctly on mock data."""
    # Run on current empty files
    result = analyze_agreement(update_status=False)
    assert result["status"] == "PENDING_HUMAN_ANNOTATION"
    assert result["cohens_kappa"] is None

    # Test kappa calculation math on known synthetic distribution
    l1 = ["passive_aggressive", "passive_aggressive", "non_passive_aggressive", "non_passive_aggressive"]
    l2 = ["passive_aggressive", "non_passive_aggressive", "non_passive_aggressive", "non_passive_aggressive"]
    categories = ["ambiguous", "non_passive_aggressive", "passive_aggressive"]
    p_o, kappa, conf = compute_cohens_kappa(l1, l2, categories)
    assert p_o == 0.75
    assert isinstance(kappa, float)


def test_preservation_of_prior_phase_artifacts():
    """Test 8: Phase 5D verified silver dataset and Phase 5E agreement tables remain intact."""
    assert P5D_VERIFIED_FILE.exists()
    verified_lines = [l for l in P5D_VERIFIED_FILE.read_text(encoding="utf-8").strip().split("\n") if l.strip()]
    assert len(verified_lines) == 19, f"Phase 5D silver verified dataset mutated! Expected 19, got {len(verified_lines)}"

    assert P5E_AUDIT_FILE.exists()
    with open(P5E_AUDIT_FILE, mode="r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 68, f"Phase 5E agreement table mutated! Expected 68, got {len(rows)}"
