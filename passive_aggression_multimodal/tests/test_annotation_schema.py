"""Unit tests for the UnifiedRecord schema and label policies."""
import sys
from pathlib import Path
import pytest
from pydantic import ValidationError

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utilities.schema import (
    AnnotationStatus,
    ImageAvailabilityStatus,
    PASubtype,
    PrimaryPALabel,
    SyntheticStatus,
    UnifiedRecord,
)


def test_valid_unified_record():
    rec = UnifiedRecord(
        example_id="test_001",
        source_dataset="test_source",
        source_record_id="rec_123",
        target_comment="With all due respect, your presentation was fascinating.",
        passive_aggression_label=PrimaryPALabel.PASSIVE_AGGRESSIVE.value,
        passive_aggression_subtype=PASubtype.FAKE_POLITENESS.value,
        annotation_status=AnnotationStatus.RAW,
        synthetic_status=SyntheticStatus.REAL,
    )
    assert rec.example_id == "test_001"
    assert rec.passive_aggression_label == "passive_aggressive"
    assert rec.passive_aggression_subtype == "fake_politeness"
    assert rec.target_comment == "With all due respect, your presentation was fascinating."


def test_empty_target_comment_rejected():
    with pytest.raises(ValidationError):
        UnifiedRecord(
            example_id="test_002",
            source_dataset="test_source",
            source_record_id="rec_124",
            target_comment="   ",  # whitespace only
        )


def test_invalid_primary_label_rejected():
    with pytest.raises(ValidationError):
        UnifiedRecord(
            example_id="test_003",
            source_dataset="test_source",
            source_record_id="rec_125",
            target_comment="A normal comment",
            passive_aggression_label="definitely_angry",  # invalid label
        )


def test_invalid_subtype_rejected():
    with pytest.raises(ValidationError):
        UnifiedRecord(
            example_id="test_004",
            source_dataset="test_source",
            source_record_id="rec_126",
            target_comment="A normal comment",
            passive_aggression_label="passive_aggressive",
            passive_aggression_subtype="random_invented_subtype",  # invalid
        )


def test_json_roundtrip():
    rec = UnifiedRecord(
        example_id="test_005",
        source_dataset="mmsd",
        source_record_id="862902619928506372",
        target_comment="Great weather today.",
        image_availability_status=ImageAvailabilityStatus.AVAILABLE,
        annotation_status=AnnotationStatus.PROVISIONAL_AI,
        synthetic_status=SyntheticStatus.REAL,
    )
    d = rec.to_jsonl_dict()
    assert d["example_id"] == "test_005"
    assert d["image_availability_status"] == "AVAILABLE"
    rec2 = UnifiedRecord.model_validate(d)
    assert rec2.example_id == rec.example_id
