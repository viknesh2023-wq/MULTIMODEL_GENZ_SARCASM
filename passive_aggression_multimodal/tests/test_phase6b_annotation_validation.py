"""
tests/test_phase6b_annotation_validation.py

Automated quality assurance and validation tests for Phase 6B:
Human Annotation Execution Validation and Agreement Analysis.

Verifies:
  1. Validation engine runs successfully and generates valid validation JSON.
  2. Missing labels remain missing (unpopulated fields are never silently defaulted).
  3. Detection of duplicate record IDs.
  4. Detection of invalid primary labels or subtypes.
  5. Incomplete annotator submissions reported honestly (0 completed, 30 pending).
  6. Language scope segregation (27 Core English Benchmark, 3 Non-English Separate Analysis).
  7. Image review metadata validation for image-bearing records.
  8. Inter-annotator agreement calculation correctness on valid mock completed data.
  9. Agreement report is NOT generated when paired human labels are pending.
  10. Preservation of prior-phase artifacts (Phase 5D silver dataset, Phase 6A manifest).
"""

import csv
import json
from pathlib import Path
import pytest

from scripts.validate_phase6b_annotations import (
    run_validation,
    validate_annotator_file,
    compute_inter_annotator_metrics,
    VALID_PRIMARY_LABELS,
    VALID_SUBTYPES,
    ROOT_DIR,
    MANIFEST_CSV,
    ANN1_CSV,
    ANN2_CSV,
    VALIDATION_RESULTS_JSON,
    STATUS_REPORT_MD,
    AGREEMENT_REPORT_MD,
)

P5D_VERIFIED_FILE = ROOT_DIR / "data/annotations/phase5d/silver_dataset_verified.jsonl"


@pytest.fixture
def manifest_rows():
    with open(MANIFEST_CSV, mode="r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


@pytest.fixture
def validation_results():
    with open(VALIDATION_RESULTS_JSON, mode="r", encoding="utf-8") as f:
        return json.load(f)


def test_validation_engine_runs_successfully():
    """Test 1: Validation engine executes cleanly and outputs valid results structure."""
    results = run_validation(update_files=False)
    assert results["pilot_size"] == 30
    assert results["annotator_01"]["valid"] is True
    assert results["annotator_02"]["valid"] is True
    assert results["paired_completed_count"] == 0
    assert results["agreement_status"] == "AWAITING_HUMAN_INPUT"
    assert results["evaluation_readiness_status"] == "BLOCKED_PENDING_HUMAN_ANNOTATION"


def test_missing_labels_not_converted():
    """Test 2: Empty labels in response templates remain empty and are not silently defaulted."""
    with open(ANN1_CSV, mode="r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    for r in rows:
        label = (r.get("primary_label") or "").strip()
        assert label == "", f"Label unexpectedly non-empty for record {r['record_id']}: '{label}'"
        assert r["review_status"] == "PENDING_REVIEW"


def test_duplicate_id_detection(manifest_rows):
    """Test 3: Validator flags duplicate record IDs if present in an annotator file."""
    # Create mock rows with a duplicate ID
    mock_rows = [dict(manifest_rows[0]), dict(manifest_rows[0])]
    mock_path = ROOT_DIR / "scratch/test_duplicate_mock.csv"
    mock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(mock_path, mode="w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(manifest_rows[0].keys()) + ["primary_label", "subtype", "annotator_id", "review_status"])
        writer.writeheader()
        writer.writerows(mock_rows)

    is_valid, errors, comp, pend, vrows = validate_annotator_file(mock_path, manifest_rows, "annotator_01")
    assert is_valid is False
    assert any("Duplicate record_id" in err for err in errors)
    if mock_path.exists():
        mock_path.unlink()


def test_invalid_category_detection(manifest_rows):
    """Test 4: Validator flags invalid primary labels or subtypes."""
    mock_row = dict(manifest_rows[0])
    mock_row["primary_label"] = "somewhat_toxic"
    mock_row["subtype"] = "invalid_subtype_name"
    mock_row["annotator_id"] = "annotator_01"

    mock_path = ROOT_DIR / "scratch/test_invalid_cat_mock.csv"
    mock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(mock_path, mode="w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(mock_row.keys()))
        writer.writeheader()
        writer.writerow(mock_row)

    is_valid, errors, comp, pend, vrows = validate_annotator_file(mock_path, manifest_rows, "annotator_01")
    assert is_valid is False
    assert any("Invalid primary_label 'somewhat_toxic'" in err for err in errors)
    assert any("Invalid subtype 'invalid_subtype_name'" in err for err in errors)
    if mock_path.exists():
        mock_path.unlink()


def test_incomplete_submissions_reported_honestly(validation_results):
    """Test 5: Incomplete submissions report 0 completed, 30 pending, and no fabricated kappa."""
    assert validation_results["annotator_01"]["completed_count"] == 0
    assert validation_results["annotator_01"]["pending_count"] == 30
    assert validation_results["annotator_02"]["completed_count"] == 0
    assert validation_results["annotator_02"]["pending_count"] == 30
    assert validation_results["inter_annotator_agreement"]["status"] == "INSUFFICIENT_PAIRED_LABELS"
    assert validation_results["inter_annotator_agreement"]["cohens_kappa"] is None


def test_language_scope_segregation(manifest_rows):
    """Test 6: 27 Core English records vs 3 Non-English Separate Analysis records are properly tracked."""
    core = [r for r in manifest_rows if r["language_scope"] == "CORE_ENGLISH_BENCHMARK"]
    non_en = [r for r in manifest_rows if r["language_scope"] == "NON_ENGLISH_SEPARATE_ANALYSIS"]
    assert len(core) == 27
    assert len(non_en) == 3


def test_image_review_metadata_integrity(manifest_rows):
    """Test 7: Validator verifies image review metadata fields."""
    mock_row = dict(manifest_rows[0])
    mock_row["annotator_id"] = "annotator_01"
    mock_row["image_reviewed"] = "INVALID_CHOICE"
    mock_row["image_role"] = "SUPER_ROLE"

    mock_path = ROOT_DIR / "scratch/test_img_meta_mock.csv"
    mock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(mock_path, mode="w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(mock_row.keys()))
        writer.writeheader()
        writer.writerow(mock_row)

    is_valid, errors, comp, pend, vrows = validate_annotator_file(mock_path, manifest_rows, "annotator_01")
    assert is_valid is False
    assert any("Invalid image_reviewed 'INVALID_CHOICE'" in err for err in errors)
    assert any("Invalid image_role 'SUPER_ROLE'" in err for err in errors)
    if mock_path.exists():
        mock_path.unlink()


def test_agreement_calculation_on_mock_completed_data():
    """Test 8: compute_inter_annotator_metrics computes exact metrics on completed paired data."""
    mock_rows1 = [
        {"record_id": f"rec_{i}", "primary_label": "passive_aggressive"} for i in range(10)
    ]
    mock_rows2 = [
        {"record_id": f"rec_{i}", "primary_label": "passive_aggressive" if i < 8 else "non_passive_aggressive"} for i in range(10)
    ]

    metrics = compute_inter_annotator_metrics(mock_rows1, mock_rows2)
    assert metrics["status"] == "CALCULATED"
    assert metrics["paired_count"] == 10
    assert metrics["raw_agreement_percent"] == 80.0
    assert isinstance(metrics["cohens_kappa"], float)
    assert "passive_aggressive->passive_aggressive" in metrics["confusion_matrix"]


def test_agreement_report_not_generated_when_incomplete():
    """Test 9: phase6b_agreement_report.md is NOT created when paired human labels are pending."""
    run_validation(update_files=True)
    assert not AGREEMENT_REPORT_MD.exists(), "Agreement report must NOT exist when reviews are incomplete"
    assert STATUS_REPORT_MD.exists(), "Status report must exist"


def test_prior_phase_artifacts_preserved():
    """Test 10: Phase 5D verified silver dataset and Phase 6A sampling manifest remain unmodified."""
    assert P5D_VERIFIED_FILE.exists()
    verified_lines = [l for l in P5D_VERIFIED_FILE.read_text(encoding="utf-8").strip().split("\n") if l.strip()]
    assert len(verified_lines) == 19

    assert MANIFEST_CSV.exists()
    with open(MANIFEST_CSV, mode="r", encoding="utf-8") as f:
        assert len(list(csv.DictReader(f))) == 30
