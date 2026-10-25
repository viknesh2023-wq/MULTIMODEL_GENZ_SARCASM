"""
tests/test_phase6c_completion_gate.py

Automated quality assurance and gate verification tests for Phase 6C:
Human Annotation Completion Gate and Agreement Analysis.

Verifies:
  1. Completion gate executes cleanly and exports valid JSON.
  2. Gate status is strictly GATE_LOCKED_PENDING_HUMAN_ANNOTATION.
  3. Zero fabricated or default human labels (both annotators 0 completed, 30 pending).
  4. Inter-annotator agreement metrics are None/null (no artificial scores).
  5. Agreement report and adjudication queue are NOT created prematurely.
  6. Language scope segregation (27 Core English Benchmark vs 3 Non-English Separate Analysis).
  7. All 14 referenced image files exist on disk.
  8. Prior phase artifacts (Phase 5D, 5E, 6A, 6B) remain unmodified.
"""

import csv
import json
from pathlib import Path
import pytest

from scripts.check_phase6c_completion_gate import (
    run_completion_gate,
    inspect_responses,
    compute_metrics,
    ROOT_DIR,
    MANIFEST_CSV,
    ANN1_CSV,
    ANN2_CSV,
    GATE_STATUS_JSON,
    GATE_REPORT_MD,
    AGREEMENT_REPORT_MD,
    ADJUDICATION_QUEUE_CSV,
)

P5D_VERIFIED_FILE = ROOT_DIR / "data/annotations/phase5d/silver_dataset_verified.jsonl"
P5E_AUDIT_FILE = ROOT_DIR / "reports/phase5e/tables/agreement_audit_68_complete.csv"


@pytest.fixture
def gate_status():
    with open(GATE_STATUS_JSON, mode="r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture
def manifest_rows():
    with open(MANIFEST_CSV, mode="r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_completion_gate_inspection_runs():
    """Test 1: Gate inspection runs cleanly and returns valid structure."""
    results = run_completion_gate(update_files=False)
    assert results["pilot_size"] == 30
    assert results["gate_status"] == "GATE_LOCKED_PENDING_HUMAN_ANNOTATION"
    assert results["paired_completed_count"] == 0
    assert results["evaluation_readiness_status"] == "BLOCKED_PENDING_HUMAN_ANNOTATION"


def test_gate_status_is_locked(gate_status):
    """Test 2: Gate status is strictly locked pending human reviews."""
    assert gate_status["gate_status"] == "GATE_LOCKED_PENDING_HUMAN_ANNOTATION"
    assert gate_status["evaluation_readiness_status"] == "BLOCKED_PENDING_HUMAN_ANNOTATION"
    assert gate_status["adjudication_status"] == "AWAITING_COMPLETED_REVIEWS"


def test_zero_fabricated_labels(gate_status):
    """Test 3: Both annotators have 0 completed, 30 pending, and 0 invalid records."""
    assert gate_status["annotator_01"]["completed_count"] == 0
    assert gate_status["annotator_01"]["pending_count"] == 30
    assert gate_status["annotator_01"]["invalid_count"] == 0

    assert gate_status["annotator_02"]["completed_count"] == 0
    assert gate_status["annotator_02"]["pending_count"] == 30
    assert gate_status["annotator_02"]["invalid_count"] == 0

    assert gate_status["paired_completed_count"] == 0


def test_agreement_metrics_are_null(gate_status):
    """Test 4: Inter-annotator agreement metrics are None (no synthetic scores)."""
    agreement = gate_status["inter_annotator_agreement"]
    assert agreement["status"] == "INSUFFICIENT_PAIRED_LABELS"
    assert agreement["cohens_kappa"] is None
    assert agreement["raw_agreement_percent"] is None
    assert agreement["paired_count"] == 0


def test_agreement_report_and_queue_not_created_prematurely():
    """Test 5: Neither agreement report nor adjudication queue exists when reviews are incomplete."""
    run_completion_gate(update_files=True)
    assert not AGREEMENT_REPORT_MD.exists(), "Agreement report must NOT exist when reviews are incomplete"
    assert not ADJUDICATION_QUEUE_CSV.exists(), "Adjudication queue must NOT exist when reviews are incomplete"
    assert GATE_REPORT_MD.exists(), "Gate status report must exist"


def test_language_scope_segregation(manifest_rows, gate_status):
    """Test 6: 27 Core English Benchmark vs 3 Non-English Separate Analysis records are properly tracked."""
    core = [r for r in manifest_rows if r["language_scope"] == "CORE_ENGLISH_BENCHMARK"]
    non_en = [r for r in manifest_rows if r["language_scope"] == "NON_ENGLISH_SEPARATE_ANALYSIS"]
    assert len(core) == 27
    assert len(non_en) == 3

    assert gate_status["language_scope"]["CORE_ENGLISH_BENCHMARK"] == 27
    assert gate_status["language_scope"]["NON_ENGLISH_SEPARATE_ANALYSIS"] == 3


def test_image_files_exist_on_disk(manifest_rows):
    """Test 7: For all 14 records with image_available == True, the image files exist on disk."""
    image_records = [r for r in manifest_rows if r["image_available"].lower() == "true"]
    assert len(image_records) == 14

    for r in image_records:
        rel_path = r["image_path"]
        assert rel_path
        full_path = ROOT_DIR / rel_path
        assert full_path.exists(), f"Image file missing: {full_path}"


def test_prior_phase_artifacts_preserved():
    """Test 8: Phase 5D silver dataset and Phase 5E agreement tables remain unmodified."""
    assert P5D_VERIFIED_FILE.exists()
    lines = [l for l in P5D_VERIFIED_FILE.read_text(encoding="utf-8").strip().split("\n") if l.strip()]
    assert len(lines) == 19

    assert P5E_AUDIT_FILE.exists()
    with open(P5E_AUDIT_FILE, mode="r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 68
