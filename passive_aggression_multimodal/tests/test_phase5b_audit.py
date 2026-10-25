"""
Tests for Phase 5B: LLM Silver-Label Quality Audit and Prompt Calibration.
"""
import csv
import json
import os
import pytest

from scripts.test_prompt_v2_subset import (
    PROMPT_V2_SYSTEM,
    SCHEMA_V2,
    build_v2_user_prompt,
)


def test_phase5a_artifacts_existence_and_alignment():
    """Verify Phase 5A outputs exist, have 100 records, and exact record ID alignment."""
    p1_path = "data/annotations/phase5a/llm/llm_annotations_pass1.jsonl"
    p2_path = "data/annotations/phase5a/llm/llm_annotations_pass2.jsonl"
    dis_path = "data/annotations/phase5a/llm/llm_annotation_disagreements.csv"
    manifest_path = "data/annotations/phase5a/llm/llm_annotation_run_manifest.json"
    sampling_path = "data/annotations/phase4/human_sampling_manifest_100.jsonl"

    assert os.path.exists(p1_path), "Pass 1 file missing"
    assert os.path.exists(p2_path), "Pass 2 file missing"
    assert os.path.exists(dis_path), "Disagreements CSV missing"
    assert os.path.exists(manifest_path), "Run manifest missing"
    assert os.path.exists(sampling_path), "Sampling manifest missing"

    with open(sampling_path, "r", encoding="utf-8") as f:
        sampling_ids = [json.loads(line)["record_id"] for line in f if line.strip()]

    with open(p1_path, "r", encoding="utf-8") as f:
        p1_records = [json.loads(line) for line in f if line.strip()]

    with open(p2_path, "r", encoding="utf-8") as f:
        p2_records = [json.loads(line) for line in f if line.strip()]

    assert len(p1_records) == 100, f"Expected 100 P1 records, got {len(p1_records)}"
    assert len(p2_records) == 100, f"Expected 100 P2 records, got {len(p2_records)}"

    p1_ids = [r["record_id"] for r in p1_records]
    p2_ids = [r["record_id"] for r in p2_records]

    assert p1_ids == sampling_ids, "P1 record IDs do not match sampling manifest"
    assert p2_ids == sampling_ids, "P2 record IDs do not match sampling manifest"


def test_disagreements_csv_integrity():
    """Verify disagreements CSV has exactly 32 records and valid cross-model divergence."""
    dis_path = "data/annotations/phase5a/llm/llm_annotation_disagreements.csv"
    with open(dis_path, "r", encoding="utf-8") as f:
        disagreements = list(csv.DictReader(f))

    assert len(disagreements) == 32, f"Expected 32 disagreements, got {len(disagreements)}"

    for row in disagreements:
        assert row["pass1_label"] != row["pass2_label"], (
            f"Record {row['record_id']} has identical labels ({row['pass1_label']}) in disagreement file!"
        )
        assert row["pass1_label"] in ("passive_aggressive", "non_passive_aggressive", "ambiguous")
        assert row["pass2_label"] in ("passive_aggressive", "non_passive_aggressive", "ambiguous")


def test_human_ground_truth_isolation():
    """Verify human review manifest remains untouched at zero completed reviews."""
    status_path = "data/annotations/phase5/human_review_status_manifest.json"
    with open(status_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["annotator_01"]["completed_count"] == 0
    assert data["annotator_02"]["completed_count"] == 0
    assert data["evaluation_readiness_status"] == "BLOCKED_PENDING_HUMAN_ANNOTATION"


def test_prompt_v2_construction_and_schema():
    """Verify Prompt v2 user prompt builds with contrast reminders and schema validity."""
    sample_rec = {
        "record_id": "test_rec_999",
        "source_dataset": "mmsd",
        "comment_text": "Great job, genius!",
        "context_text": "Broken window",
        "image_available": True,
    }
    user_prompt = build_v2_user_prompt(sample_rec)
    assert "test_rec_999" in user_prompt
    assert "Great job, genius!" in user_prompt
    assert "Broken window" in user_prompt
    assert "News headlines, factual statements, direct insults, and sincere praise are NOT passive-aggressive." in user_prompt

    # Schema checks
    assert "brief_rationale" in SCHEMA_V2["required"]
    assert "evidence_span" in SCHEMA_V2["required"]
    assert "primary_label" in SCHEMA_V2["required"]
    assert "subtype" in SCHEMA_V2["required"]
    assert SCHEMA_V2["properties"]["primary_label"]["enum"] == [
        "non_passive_aggressive", "ambiguous", "passive_aggressive"
    ]


def test_prompt_v2_diagnostic_subset_results():
    """Verify prompt v2 diagnostic results exist and confirm shift in overprediction."""
    subset_file = "data/annotations/phase5b/prompt_v2_subset_test.json"
    assert os.path.exists(subset_file), "Prompt v2 subset comparison file missing"

    with open(subset_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert len(data) == 12, f"Expected 12 records in diagnostic subset, got {len(data)}"

    # Check that sports news (mmsd_test_02285) correctly shifted from PA in v1 to non-PA in v2 for Llama
    sports_rec = next(r for r in data if r["record_id"] == "mmsd_test_02285")
    assert "passive_aggressive" in sports_rec["v1_llama"]
    assert "non_passive_aggressive" in sports_rec["v2_llama"]
