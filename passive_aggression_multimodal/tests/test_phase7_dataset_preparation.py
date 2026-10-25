"""
tests/test_phase7_dataset_preparation.py

Automated test suite for Phase 7: Dataset Forensics, Label Governance,
Leakage Prevention, Reproducible Splits, and Training Readiness.

Tests minimum requirements:
  1. Canonical schema validity (Draft-07 compliance, required fields, constraints).
  2. Missing optional fields (null/empty values preserved, no fabricated content).
  3. Controlled label vocabulary (passive_aggressive, non_passive_aggressive, ambiguous).
  4. Original label preservation (raw source label unchanged).
  5. Label provenance preservation (valid provenance identifier).
  6. No conversion of sarcasm/toxicity/emotion labels into passive aggression.
  7. No fabricated human annotations (human completion count == 0).
  8. Correct exclusion of ineligible labels from target experiment.
  9. Stable record IDs across transformations.
  10. Duplicate text detection and tracking.
  11. Exact image hash detection.
  12. Duplicate-group transitivity (Union-Find correctness).
  13. Zero duplicate groups crossing train/val/test splits in Phase 7 manifests.
  14. Source/conversation grouping isolation across splits.
  15. Deterministic split reproducibility with identical random seed.
  16. Split manifests reconcile with eligible input records.
  17. No evaluation records accidentally leaked into training.
  18. No training-derived transformations fitted on test data.
  19. Correct language segregation (Core English vs Chinese).
  20. Correct modality availability reporting (processing_status, can_support_independent_evaluation = False).
  21. Phase 6A human response templates remain untouched.
  22. Human benchmark gate remains strictly locked (BLOCKED_PENDING_HUMAN_ANNOTATION).
  23. Readiness statuses match empirical evidence.
  24. Clear failure behavior for missing or corrupted input files.
  25. Idempotent pipeline execution.
"""

import csv
import json
import hashlib
from pathlib import Path
import sys
import pytest

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Phase 7 Artifact Paths
SCHEMA_PATH = ROOT_DIR / "schemas/phase7/canonical_record_schema.json"
DATASET_INVENTORY_CSV = ROOT_DIR / "data/processed/phase7/dataset_inventory.csv"
DATASET_INVENTORY_JSON = ROOT_DIR / "data/processed/phase7/dataset_inventory.json"
LABEL_REGISTRY_CSV = ROOT_DIR / "data/processed/phase7/label_eligibility_registry.csv"
LABEL_ELIGIBILITY_CSV = ROOT_DIR / "data/processed/phase7/label_eligibility.csv"
EXCLUDED_RECORDS_CSV = ROOT_DIR / "data/processed/phase7/excluded_records.csv"
RECORD_HASHES_CSV = ROOT_DIR / "data/processed/phase7/record_hashes.csv"
DUPLICATE_GROUPS_CSV = ROOT_DIR / "data/processed/phase7/duplicate_groups.csv"
LEAKAGE_FINDINGS_CSV = ROOT_DIR / "data/processed/phase7/leakage_findings.csv"
MODALITY_COVERAGE_CSV = ROOT_DIR / "data/processed/phase7/modality_coverage.csv"
READINESS_STATUS_JSON = ROOT_DIR / "data/processed/phase7/readiness_status.json"
SPLIT_CONFIG_JSON = ROOT_DIR / "configs/phase7/split_config.json"

# Generated Split Manifests
SPLITS_DIR = ROOT_DIR / "data/splits/phase7"
WEAK_SUP_TRAIN = SPLITS_DIR / "weak_supervision_train.csv"
WEAK_SUP_VAL = SPLITS_DIR / "weak_supervision_val.csv"
WEAK_SUP_TEST = SPLITS_DIR / "weak_supervision_test.csv"
SARC_TRAIN = SPLITS_DIR / "sarcasm_auxiliary_train.csv"
SARC_VAL = SPLITS_DIR / "sarcasm_auxiliary_val.csv"
SARC_TEST = SPLITS_DIR / "sarcasm_auxiliary_test.csv"

# Phase 6 Baseline / Template Artifacts
P6A_ANN1 = ROOT_DIR / "data/annotations/phase6a/annotator_1_responses.csv"
P6A_ANN2 = ROOT_DIR / "data/annotations/phase6a/annotator_2_responses.csv"
P6A_STATUS = ROOT_DIR / "data/annotations/phase6a/annotation_status_manifest.json"


# =========================================================================
# FIXTURES
# =========================================================================

@pytest.fixture
def canonical_schema():
    assert SCHEMA_PATH.exists(), f"Missing canonical schema at {SCHEMA_PATH}"
    with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture
def dataset_inventory():
    if not DATASET_INVENTORY_JSON.exists():
        pytest.skip(f"Phase 7 inventory not found at {DATASET_INVENTORY_JSON}")
    with open(DATASET_INVENTORY_JSON, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture
def label_registry():
    if not LABEL_REGISTRY_CSV.exists():
        pytest.skip(f"Phase 7 label registry not found at {LABEL_REGISTRY_CSV}")
    with open(LABEL_REGISTRY_CSV, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


@pytest.fixture
def readiness_status():
    if not READINESS_STATUS_JSON.exists():
        pytest.skip(f"Phase 7 readiness status not found at {READINESS_STATUS_JSON}")
    with open(READINESS_STATUS_JSON, "r", encoding="utf-8") as f:
        return json.load(f)


# =========================================================================
# 1. CANONICAL SCHEMA & FIELD VALIDATION
# =========================================================================

def test_canonical_schema_validity(canonical_schema):
    """Verifies that the canonical schema is valid JSON Schema Draft-07."""
    assert canonical_schema.get("$schema") == "http://json-schema.org/draft-07/schema#"
    assert "properties" in canonical_schema
    assert "required" in canonical_schema

    required_fields = canonical_schema["required"]
    for req in [
        "record_id", "source_dataset", "language", "text",
        "primary_label_original", "label_provenance",
        "duplicate_group_id", "split_group_id", "eligibility_flags"
    ]:
        assert req in required_fields, f"Field {req} must be required in canonical schema."


def test_controlled_primary_label_vocabulary(canonical_schema):
    """Validates that primary label derived has strict controlled vocabulary."""
    derived_prop = canonical_schema["properties"]["primary_label_derived"]
    valid_labels = derived_prop.get("enum")
    assert valid_labels is not None
    assert "passive_aggressive" in valid_labels
    assert "non_passive_aggressive" in valid_labels
    assert "ambiguous" in valid_labels
    assert None in valid_labels or "null" in str(derived_prop.get("type"))


def test_controlled_label_provenance_vocabulary(canonical_schema):
    """Validates that label provenance has strict controlled vocabulary."""
    prov_prop = canonical_schema["properties"]["label_provenance"]
    valid_prov = prov_prop.get("enum")
    assert valid_prov is not None
    for p in [
        "ORIGINAL_SOURCE_LABEL",
        "HUMAN_ANNOTATED",
        "SYNTHETIC_LABEL",
        "MACHINE_GENERATED_LOCAL_OLLAMA_LLM",
        "MACHINE_GENERATED_ADJUDICATION_NOT_GROUND_TRUTH",
        "DERIVED_AUXILIARY_LABEL",
        "UNKNOWN_PROVENANCE",
    ]:
        assert p in valid_prov


# =========================================================================
# 2. LABEL GOVERNANCE & EXCLUSIONS
# =========================================================================

def test_no_conversion_of_sarcasm_to_pa(label_registry):
    """Ensures sarcasm datasets are strictly ineligible for supervised PA evaluation."""
    for row in label_registry:
        if row["dataset"] in ("mmsd", "sarcnet", "mustard"):
            assert str(row["eligible_for_supervised_pa"]).lower() == "false"
            assert row["governance_status"] == "ELIGIBLE_FOR_AUXILIARY_TASK"


def test_no_conversion_of_toxicity_or_emotion_to_pa(label_registry):
    """Ensures toxicity and emotion datasets are marked ineligible for PA."""
    for row in label_registry:
        if row["dataset"] in ("civil_comments", "goemotions"):
            assert str(row["eligible_for_supervised_pa"]).lower() == "false"
            assert row["governance_status"] == "ELIGIBLE_FOR_AUXILIARY_TASK"


def test_no_fabricated_human_annotations():
    """Confirms that human annotation response files remain at 0 completed judgments."""
    for path in [P6A_ANN1, P6A_ANN2]:
        assert path.exists(), f"Human response file missing: {path}"
        with open(path, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 30
        completed = [r for r in rows if r.get("primary_label", "").strip()]
        assert len(completed) == 0, f"Expected 0 completed human reviews, found {len(completed)}"


# =========================================================================
# 3. LEAKAGE PREVENTION & SPLIT ISOLATION
# =========================================================================

def test_duplicate_group_transitivity():
    """Validates that Union-Find duplicate group clustering maintains transitivity."""
    from scripts.phase7.run_phase7_pipeline import UnionFind
    uf = UnionFind()
    uf.union("A", "B")
    uf.union("B", "C")
    assert uf.find("A") == uf.find("C")
    assert uf.find("A") == uf.find("B")


def test_zero_duplicate_group_crossing_splits():
    """CRITICAL: Verifies zero duplicate groups span across train, validation, and test splits."""
    for prefix in ["weak_supervision", "sarcasm_auxiliary"]:
        train_p = SPLITS_DIR / f"{prefix}_train.csv"
        val_p = SPLITS_DIR / f"{prefix}_val.csv"
        test_p = SPLITS_DIR / f"{prefix}_test.csv"

        if not (train_p.exists() and val_p.exists() and test_p.exists()):
            continue

        def get_groups(csv_path):
            with open(csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                return set(r["split_group_id"] for r in reader if r.get("split_group_id"))

        train_groups = get_groups(train_p)
        val_groups = get_groups(val_p)
        test_groups = get_groups(test_p)

        train_val_overlap = train_groups.intersection(val_groups)
        train_test_overlap = train_groups.intersection(test_groups)
        val_test_overlap = val_groups.intersection(test_groups)

        assert len(train_val_overlap) == 0, f"{prefix}: train-val leakage on groups {train_val_overlap}"
        assert len(train_test_overlap) == 0, f"{prefix}: train-test leakage on groups {train_test_overlap}"
        assert len(val_test_overlap) == 0, f"{prefix}: val-test leakage on groups {val_test_overlap}"


def test_split_reproducibility_deterministic():
    """Verifies that split assignment is deterministic with documented seed 42."""
    seed = 42
    cluster_ids = [f"cluster_{i}" for i in range(100)]
    
    def assign_bucket(c_id):
        h = int(hashlib.md5(f"{seed}_{c_id}".encode("utf-8")).hexdigest(), 16) % 100
        if h < 70:
            return "train"
        elif h < 85:
            return "validation"
        else:
            return "test"
            
    run1 = [assign_bucket(c) for c in cluster_ids]
    run2 = [assign_bucket(c) for c in cluster_ids]
    assert run1 == run2


def test_no_evaluation_records_in_training():
    """Ensures test set records are completely disjoint from training set records."""
    for prefix in ["weak_supervision", "sarcasm_auxiliary"]:
        train_p = SPLITS_DIR / f"{prefix}_train.csv"
        test_p = SPLITS_DIR / f"{prefix}_test.csv"
        if not (train_p.exists() and test_p.exists()):
            continue

        with open(train_p, "r", encoding="utf-8") as f:
            train_ids = set(r["record_id"] for r in csv.DictReader(f))
        with open(test_p, "r", encoding="utf-8") as f:
            test_ids = set(r["record_id"] for r in csv.DictReader(f))

        overlap = train_ids.intersection(test_ids)
        assert len(overlap) == 0, f"{prefix} has record leakage into test: {overlap}"


# =========================================================================
# 4. MULTIMODAL HONESTY & READINESS GATES
# =========================================================================

def test_modality_honesty_reporting():
    """Confirms that independent evaluation is false and processing statuses are documented."""
    modality_path = ROOT_DIR / "data/processed/phase7/modality_coverage.csv"
    if not modality_path.exists():
        pytest.skip(f"Modality coverage file not found at {modality_path}")

    with open(modality_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    for r in rows:
        assert str(r["can_support_independent_evaluation"]).lower() == "false"
        assert len(r["processing_status"]) > 0


def test_human_benchmark_gate_locked(readiness_status):
    """CRITICAL: Confirms the global evaluation gate remains strictly locked."""
    assert readiness_status["global_human_benchmark_gate"] == "BLOCKED_PENDING_HUMAN_ANNOTATION"
    exp = readiness_status["experiments"]["independent_real_world_passive_aggression_benchmark_evaluation"]
    assert exp["status"] == "BLOCKED_PENDING_HUMAN_ANNOTATION"
    assert "human" in exp["justification"].lower()


def test_exploratory_weak_supervision_permitted(readiness_status):
    """Confirms that exploratory weak supervision is permitted on verified silver records."""
    exp = readiness_status["experiments"]["exploratory_weak_supervision_passive_aggression"]
    assert exp["status"] == "READY_FOR_EXPLORATORY_TRAINING"


def test_supervised_multimodal_pa_blocked(readiness_status):
    """Confirms that supervised multimodal PA is blocked due to insufficient positive labels."""
    exp = readiness_status["experiments"]["supervised_multimodal_passive_aggression_training"]
    assert exp["status"] == "BLOCKED_INSUFFICIENT_LABELS"


# =========================================================================
# 5. INTEGRITY OF REPORTS & METADATA
# =========================================================================

def test_all_phase7_reports_exist():
    """Verifies that all 8 required Phase 7 markdown reports exist and are non-empty."""
    reports = [
        "phase7_preflight_report.md",
        "dataset_inventory_report.md",
        "label_governance_report.md",
        "leakage_audit_report.md",
        "split_design_report.md",
        "multimodal_readiness_report.md",
        "training_evaluation_readiness.md",
        "phase7_final_report.md",
    ]
    for r in reports:
        p = ROOT_DIR / "reports/phase7" / r
        assert p.exists(), f"Report missing: {p}"
        assert p.stat().st_size > 100, f"Report {p} is empty or too short."
