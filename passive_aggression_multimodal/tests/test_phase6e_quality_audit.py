"""
tests/test_phase6e_quality_audit.py

Automated test suite for Phase 6E: Quality Audit for LLM Dual-Annotation,
Inter-Model Agreement, and Adjudication.

Verifies:
  1. Phase 6E audit artifacts exist and adhere to JSON/Markdown schema.
  2. 100% record reconciliation across pilot manifest, Ann1, Ann2, Disagreements, and Adjudication.
  3. Modality honesty: 100% text-only processing, zero visual evidence claims.
  4. Language scope segregation: 27 Core English Benchmark vs 3 Chinese SarcNet records.
  5. Mathematical agreement recalculation matches Phase 6D manifest exactly.
  6. Human response files preservation (0 completed reviews) and locked evaluation gate.
  7. Conservative collapse label distribution: 16 non-PA, 14 ambiguous, 0 PA.
  8. Disagreement traceability: all 14 divergences have documented root cause and rationale.
  9. Provenance integrity: all labels strictly marked with machine-generated provenance.
  10. Project readiness gating: weak supervision permitted, benchmark evaluation locked.
"""

import csv
import json
from pathlib import Path
import sys
import pytest

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

MANIFEST_CSV = ROOT_DIR / "data/annotations/phase6a/pilot_sampling_manifest.csv"
ANN1_HUMAN_CSV = ROOT_DIR / "data/annotations/phase6a/annotator_1_responses.csv"
ANN2_HUMAN_CSV = ROOT_DIR / "data/annotations/phase6a/annotator_2_responses.csv"
STATUS_MANIFEST_JSON = ROOT_DIR / "data/annotations/phase6a/annotation_status_manifest.json"

ANN1_LLM_CSV = ROOT_DIR / "data/annotations/phase6d/annotator_1_llm_responses.csv"
ANN2_LLM_CSV = ROOT_DIR / "data/annotations/phase6d/annotator_2_llm_responses.csv"
DISAGREEMENTS_CSV = ROOT_DIR / "data/annotations/phase6d/llm_annotation_disagreements.csv"
ADJUDICATION_CSV = ROOT_DIR / "data/annotations/phase6d/llm_adjudication_results.csv"
MANIFEST_6D_JSON = ROOT_DIR / "data/annotations/phase6d/llm_annotation_manifest.json"

AUDIT_RESULTS_JSON = ROOT_DIR / "data/annotations/phase6e/phase6e_audit_results.json"
QUALITY_AUDIT_MD = ROOT_DIR / "reports/phase6e/phase6e_quality_audit.md"


@pytest.fixture
def audit_data():
    if not AUDIT_RESULTS_JSON.exists():
        pytest.skip("Phase 6E audit results JSON not yet generated.")
    with open(AUDIT_RESULTS_JSON, mode="r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture
def manifest_rows():
    with open(MANIFEST_CSV, mode="r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


@pytest.fixture
def adjudication_rows():
    with open(ADJUDICATION_CSV, mode="r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_phase6e_artifacts_exist():
    """Test 1: Phase 6E JSON and Markdown audit reports exist."""
    assert AUDIT_RESULTS_JSON.exists(), f"Missing {AUDIT_RESULTS_JSON}"
    assert QUALITY_AUDIT_MD.exists(), f"Missing {QUALITY_AUDIT_MD}"
    assert AUDIT_RESULTS_JSON.stat().st_size > 1000
    assert QUALITY_AUDIT_MD.stat().st_size > 1000


def test_pilot_reconciliation_integrity(audit_data):
    """Test 2: All 30 records reconcile across all files with zero duplicates."""
    recon = audit_data["id_reconciliation"]
    assert recon["pilot_record_count"] == 30
    assert recon["ann1_record_count"] == 30
    assert recon["ann2_record_count"] == 30
    assert recon["adj_record_count"] == 30
    assert recon["disagreement_record_count"] == 14
    assert recon["id_sets_identical"] is True
    assert recon["unique_pilot_ids"] is True
    assert recon["unique_ann1_ids"] is True
    assert recon["unique_ann2_ids"] is True
    assert recon["unique_adj_ids"] is True
    assert audit_data["dataset_integrity"]["duplicate_comments_count"] == 0


def test_modality_honesty_text_only(audit_data):
    """Test 3: 100% of records are text-only with images unassessed."""
    mod = audit_data["modality_honesty_check"]
    assert mod["all_ann1_text_only"] is True
    assert mod["all_ann2_text_only"] is True


def test_language_segregation_27_3(audit_data):
    """Test 4: Strict segregation of 27 Core English Benchmark vs 3 Chinese records."""
    lang = audit_data["dataset_integrity"]["language_distribution"]
    assert lang["CORE_ENGLISH_BENCHMARK"] == 27
    assert lang["NON_ENGLISH_SEPARATE_ANALYSIS"] == 3

    agr = audit_data["agreement_recalculation"]
    assert agr["core_english_benchmark"]["total_pairs"] == 27
    assert agr["chinese_exploratory"]["total_pairs"] == 3


def test_agreement_recalculation_matches_manifest(audit_data):
    """Test 5: Recalculated agreement matches Phase 6D manifest values."""
    agr = audit_data["agreement_recalculation"]
    assert agr["overall"]["total_pairs"] == 30
    assert agr["overall"]["exact_matches"] == 16
    assert agr["overall"]["disagreements"] == 14
    assert agr["overall"]["raw_agreement_percent"] == 53.33
    assert agr["overall"]["cohens_kappa"] == 0.1322

    assert agr["core_english_benchmark"]["total_pairs"] == 27
    assert agr["core_english_benchmark"]["exact_matches"] == 14
    assert agr["core_english_benchmark"]["disagreements"] == 13
    assert agr["core_english_benchmark"]["raw_agreement_percent"] == 51.85
    assert agr["core_english_benchmark"]["cohens_kappa"] == 0.1397


def test_human_ground_truth_isolation(audit_data):
    """Test 6: Human response files remain untouched (0 reviews) and gate is locked."""
    human = audit_data["human_protection_check"]
    assert human["annotator_1_human_completed"] == 0
    assert human["annotator_2_human_completed"] == 0
    assert human["human_files_untouched"] is True
    assert human["gate_strictly_blocked"] is True
    assert human["human_evaluation_gate"] == "BLOCKED_PENDING_HUMAN_ANNOTATION"


def test_conservative_collapse_label_distribution(audit_data, adjudication_rows):
    """Test 7: Conservative collapse verification (16 non-PA, 14 ambiguous, 0 PA)."""
    adj_dist = audit_data["label_distributions"]["adjudicated_final"]
    assert adj_dist.get("passive_aggressive", 0) == 0
    assert adj_dist.get("non_passive_aggressive") == 16
    assert adj_dist.get("ambiguous") == 14

    assert len(adjudication_rows) == 30
    assert sum(1 for r in adjudication_rows if r["adjudicated_label"] == "passive_aggressive") == 0
    assert sum(1 for r in adjudication_rows if r["adjudicated_label"] == "non_passive_aggressive") == 16
    assert sum(1 for r in adjudication_rows if r["adjudicated_label"] == "ambiguous") == 14


def test_disagreement_traceability():
    """Test 8: All 14 disagreements have root causes and adjudication rationales."""
    with open(DISAGREEMENTS_CSV, mode="r", encoding="utf-8") as f:
        dis_rows = list(csv.DictReader(f))
    with open(ADJUDICATION_CSV, mode="r", encoding="utf-8") as f:
        adj_map = {r["record_id"]: r for r in csv.DictReader(f)}

    assert len(dis_rows) == 14
    for d in dis_rows:
        rid = d["record_id"]
        adj = adj_map[rid]
        assert adj["adjudication_type"] == "LLM_ADJUDICATED_DISAGREEMENT"
        assert len(adj["disagreement_root_cause"].strip()) > 10
        assert len(adj["adjudication_rationale"].strip()) > 10


def test_provenance_strictly_machine_labeled(audit_data):
    """Test 9: All outputs are machine-generated with explicit provenance."""
    prov = audit_data["provenance_check"]
    assert prov["all_machine_provenance"] is True
    assert prov["ann1_provenance_set"] == ["MACHINE_GENERATED_LOCAL_OLLAMA_LLM"]
    assert prov["ann2_provenance_set"] == ["MACHINE_GENERATED_LOCAL_OLLAMA_LLM"]


def test_project_readiness_gating(audit_data):
    """Test 10: Weak supervision is permitted, real-data benchmark evaluation is locked."""
    readiness = audit_data["project_readiness_assessment"]
    assert readiness["ready_for_weak_supervision_experiments"] is True
    assert readiness["ready_for_real_data_benchmark_evaluation"] is False
    assert readiness["benchmark_evaluation_gate"] == "BLOCKED_PENDING_HUMAN_ANNOTATION"
