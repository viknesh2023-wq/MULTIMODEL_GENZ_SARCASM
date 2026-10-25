"""Unit tests for the AI-assisted annotation pipeline skeleton."""
import sys
from pathlib import Path
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.annotation.annotation_pipeline import (
    AdjudicationPipeline,
    RuleAndHeuristicTextAnnotator,
    VisionLanguageAnnotator,
)
from src.utilities.schema import (
    AnnotationStatus,
    ImageAvailabilityStatus,
    SyntheticStatus,
    UnifiedRecord,
)


def test_text_annotator_detects_fake_politeness():
    annotator = RuleAndHeuristicTextAnnotator()
    rec = UnifiedRecord(
        example_id="test_fp_01",
        source_dataset="unit_test",
        source_record_id="rec_01",
        target_comment="With all due respect, your analysis lacks basic rigor.",
    )
    pred = annotator.annotate(rec)
    assert pred.primary_label == "passive_aggressive"
    assert pred.subtype == "fake_politeness"
    assert pred.confidence >= 0.8


def test_text_annotator_detects_dismissiveness():
    annotator = RuleAndHeuristicTextAnnotator()
    rec = UnifiedRecord(
        example_id="test_dis_01",
        source_dataset="unit_test",
        source_record_id="rec_02",
        target_comment="Must be nice to have unlimited vacation time.",
    )
    pred = annotator.annotate(rec)
    assert pred.primary_label == "passive_aggressive"
    assert pred.subtype == "dismissiveness"


def test_adjudication_pipeline_preserves_provenance():
    pipeline = AdjudicationPipeline()
    rec = UnifiedRecord(
        example_id="test_adj_01",
        source_dataset="mmsd",
        source_record_id="img_123",
        original_label=1,
        image_availability_status=ImageAvailabilityStatus.AVAILABLE,
        image_path="data/images/mmsd/sample.jpg",
        target_comment="Per my previous email, you forgot the attachments.",
        annotation_status=AnnotationStatus.RAW,
        synthetic_status=SyntheticStatus.REAL,
    )
    res = pipeline.process_record(rec)
    assert res.example_id == "test_adj_01"
    assert res.original_source_label == 1
    assert res.text_only_prediction is not None
    assert res.vlm_prediction is not None
    assert res.human_verification_status == "AWAITING_HUMAN_REVIEW"
    assert res.agreement_status in ["UNANIMOUS_AGREEMENT", "DISAGREEMENT", "UNIMODAL_ONLY"]
