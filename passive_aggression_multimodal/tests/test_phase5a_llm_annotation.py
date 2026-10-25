"""Robustness and Integrity Tests for Phase 5A Local LLM Annotation Pipeline."""
import csv
import json
import os
import tempfile
import pytest

from scripts.annotate_with_local_ollama import (
    ANNOTATION_JSON_SCHEMA,
    annotate_record,
    build_user_prompt,
    compute_consistency,
    query_ollama,
    run_pass,
)


def test_user_prompt_construction():
    """Verify prompt correctly grounds record text, context, and modality notes."""
    rec = {
        "record_id": "test_rec_001",
        "source_dataset": "mmsd",
        "comment_text": "Great parking job, genius!",
        "context_text": "Parking lot photo",
        "image_available": True,
    }
    prompt = build_user_prompt(rec)
    assert "test_rec_001" in prompt
    assert "Great parking job, genius!" in prompt
    assert "Parking lot photo" in prompt
    assert "Image pixels unassessed" in prompt


def test_annotate_record_dry_run():
    """Verify schema-validated record output formatting in dry-run mode."""
    rec = {
        "record_id": "test_002",
        "source_dataset": "sarcnet",
        "comment_text": "Thank you for sharing your unique effort.",
        "image_available": True,
        "context_text": "",
    }
    annotated, err = annotate_record(rec, model="mock_model", dry_run=True)
    assert err is None
    assert annotated is not None
    assert annotated["record_id"] == "test_002"
    assert annotated["annotation_source"] == "LLM"
    assert annotated["annotation_status"] == "LLM_PROVISIONAL"
    assert annotated["modality_used"] == "TEXT_ONLY"
    assert annotated["image_role"] == "UNASSESSED_VISION_MODEL_ABSENT"
    assert annotated["primary_label"] in ("passive_aggressive", "non_passive_aggressive", "ambiguous")


def test_ollama_unavailable_handling():
    """Verify clean exception when Ollama endpoint is down or unreachable."""
    with pytest.raises(RuntimeError) as exc_info:
        query_ollama(
            model="non_existent_model",
            user_prompt="Hello",
            endpoint="http://localhost:99999/api/chat",  # Invalid port
            timeout=1,
            dry_run=False,
        )
    assert "Ollama inference error" in str(exc_info.value)


def test_resumable_run_skips_completed(tmp_path):
    """Verify pipeline skips already annotated records without duplicate writes."""
    out_file = str(tmp_path / "test_pass1.jsonl")
    fail_file = str(tmp_path / "test_fail.jsonl")

    recs = [
        {"record_id": f"rec_{i}", "source_dataset": "mmsd", "comment_text": f"Comment {i}"}
        for i in range(5)
    ]

    # Pre-populate record 0 and 1
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(json.dumps({"record_id": "rec_0", "primary_label": "ambiguous"}) + "\n")
        f.write(json.dumps({"record_id": "rec_1", "primary_label": "non_passive_aggressive"}) + "\n")

    # Run pass with resume=True
    completed = run_pass(
        records=recs,
        output_path=out_file,
        failure_path=fail_file,
        model="mock_model",
        pass_name="TEST_PASS",
        resume=True,
        dry_run=True,
    )

    # Should have 5 records total in output file (2 pre-existing + 3 newly written)
    with open(out_file, "r", encoding="utf-8") as f:
        lines = [json.loads(l) for l in f if l.strip()]

    assert len(lines) == 5
    ids = [item["record_id"] for item in lines]
    assert len(set(ids)) == 5, "Duplicate record IDs found in output!"


def test_consistency_calculation(tmp_path):
    """Verify consistency computation between Pass 1 and Pass 2."""
    pass1 = [
        {"record_id": "r1", "comment_text": "Text 1", "primary_label": "passive_aggressive", "subtype": "fake_politeness"},
        {"record_id": "r2", "comment_text": "Text 2", "primary_label": "non_passive_aggressive", "subtype": "none"},
        {"record_id": "r3", "comment_text": "Text 3", "primary_label": "ambiguous", "subtype": "none"},
    ]
    # Pass 2 agrees on r1 and r2, disagrees on r3
    pass2 = [
        {"record_id": "r1", "comment_text": "Text 1", "primary_label": "passive_aggressive", "subtype": "fake_politeness"},
        {"record_id": "r2", "comment_text": "Text 2", "primary_label": "non_passive_aggressive", "subtype": "none"},
        {"record_id": "r3", "comment_text": "Text 3", "primary_label": "non_passive_aggressive", "subtype": "none"},
    ]

    disagree_csv = str(tmp_path / "disagreements.csv")
    res = compute_consistency(pass1, pass2, disagree_csv)

    assert res["paired_records_count"] == 3
    assert res["agreement_count"] == 2
    assert res["disagreement_count"] == 1
    assert res["raw_consistency_percent"] == pytest.approx(66.67, abs=0.01)
    assert os.path.exists(disagree_csv)

    with open(disagree_csv, "r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))
    assert len(reader) == 1
    assert reader[0]["record_id"] == "r3"


def test_human_annotation_files_unmodified():
    """Verify that human response CSVs and status manifest are never overwritten or altered by LLM pipeline."""
    h1_path = "data/annotations/phase5/annotator_1_responses.csv"
    h2_path = "data/annotations/phase5/annotator_2_responses.csv"
    status_path = "data/annotations/phase5/human_review_status_manifest.json"

    assert os.path.exists(h1_path)
    assert os.path.exists(h2_path)
    assert os.path.exists(status_path)

    with open(status_path, "r", encoding="utf-8") as f:
        status_data = json.load(f)

    # Human reviews MUST remain strictly at 0
    assert status_data["annotator_01"]["completed_count"] == 0
    assert status_data["annotator_02"]["completed_count"] == 0
    assert status_data["evaluation_readiness_status"] == "BLOCKED_PENDING_HUMAN_ANNOTATION"
