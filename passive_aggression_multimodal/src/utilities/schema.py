"""Unified metadata schema for multimodal passive aggression detection."""
from __future__ import annotations

import json
from enum import Enum
from typing import Any, List, Optional
from pydantic import BaseModel, Field, field_validator


class ImageAvailabilityStatus(str, Enum):
    AVAILABLE = "AVAILABLE"
    MISSING = "MISSING"
    NOT_APPLICABLE = "NOT_APPLICABLE"
    CORRUPT = "CORRUPT"


class AnnotationStatus(str, Enum):
    RAW = "RAW"
    PROVISIONAL_AI = "PROVISIONAL_AI"
    HUMAN_VERIFIED = "HUMAN_VERIFIED"
    SYNTHETIC_PILOT = "SYNTHETIC_PILOT"


class SyntheticStatus(str, Enum):
    REAL = "REAL"
    SYNTHETIC_UNVERIFIED = "SYNTHETIC_UNVERIFIED"
    SYNTHETIC_VERIFIED = "SYNTHETIC_VERIFIED"


class PrimaryPALabel(str, Enum):
    PASSIVE_AGGRESSIVE = "passive_aggressive"
    NON_PASSIVE_AGGRESSIVE = "non_passive_aggressive"
    AMBIGUOUS = "ambiguous"


class PASubtype(str, Enum):
    FAKE_POLITENESS = "fake_politeness"
    INDIRECT_CRITICISM = "indirect_criticism"
    BACKHANDED_COMPLIMENT = "backhanded_compliment"
    IMPLIED_BLAME = "implied_blame"
    SARCASTIC_AGREEMENT = "sarcastic_agreement"
    DISMISSIVENESS = "dismissiveness"
    OTHER_OR_UNCLEAR = "other_or_unclear"


class UnifiedRecord(BaseModel):
    """Unified record representation preserving original provenance while supporting multimodal PA analysis."""
    example_id: str = Field(..., description="Globally unique identifier: dataset_prefix_id")
    source_dataset: str = Field(..., description="Name of source dataset, e.g. mmsd, sarcnet, goemotions")
    source_record_id: str = Field(..., description="Original record identifier in source dataset")
    source_url_or_reference: Optional[str] = Field(default=None, description="Repository or paper reference")
    dataset_version: Optional[str] = Field(default="1.0.0", description="Version of source dataset")
    source_license: Optional[str] = Field(default=None, description="License for text/dataset")
    asset_license_status: Optional[str] = Field(default=None, description="License for media assets")

    post_id: Optional[str] = Field(default=None, description="Social media post id if distinct from record")
    conversation_id: Optional[str] = Field(default=None, description="Thread or conversation cluster id")

    image_path: Optional[str] = Field(default=None, description="Relative path to verified image file")
    image_sha256: Optional[str] = Field(default=None, description="SHA256 checksum of image file")
    image_availability_status: ImageAvailabilityStatus = Field(
        default=ImageAvailabilityStatus.NOT_APPLICABLE,
        description="Availability status of the image asset"
    )

    caption: Optional[str] = Field(default=None, description="Post text or author's caption")
    image_ocr_text: Optional[str] = Field(default=None, description="OCR text detected in image")
    target_comment: str = Field(..., description="The comment being evaluated for passive aggression")
    parent_comment: Optional[str] = Field(default=None, description="Immediate reply-to parent text")
    conversation_context: Optional[List[str]] = Field(default=None, description="Chronological dialogue context")

    original_label: Optional[Any] = Field(default=None, description="Intact label from source dataset")
    original_label_scheme: Optional[str] = Field(default=None, description="Explanation of source label format")

    passive_aggression_label: Optional[str] = Field(
        default=None,
        description="Primary label: passive_aggressive, non_passive_aggressive, ambiguous"
    )
    passive_aggression_subtype: Optional[str] = Field(
        default=None,
        description="Secondary subtype label"
    )
    sarcasm_label: Optional[str] = Field(
        default=None,
        description="Independent auxiliary label: sarcastic, non_sarcastic, ambiguous, unknown"
    )
    image_needed_for_interpretation: Optional[bool] = Field(
        default=None,
        description="True if image context changes or enables PA interpretation"
    )

    annotation_confidence: Optional[float] = Field(default=None, description="Confidence score [0.0 - 1.0]")
    annotation_status: AnnotationStatus = Field(
        default=AnnotationStatus.RAW,
        description="Stage in annotation lifecycle"
    )
    synthetic_status: SyntheticStatus = Field(
        default=SyntheticStatus.REAL,
        description="Real vs synthetic provenance"
    )
    train_validation_test_split: Optional[str] = Field(default=None, description="train, validation, test, or null")

    @field_validator("target_comment")
    @classmethod
    def validate_target_comment(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("target_comment must be a non-empty string")
        return v.strip()

    @field_validator("passive_aggression_label")
    @classmethod
    def validate_pa_label(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and v not in [e.value for e in PrimaryPALabel]:
            raise ValueError(f"Invalid primary label: {v}. Must be one of {[e.value for e in PrimaryPALabel]}")
        return v

    @field_validator("passive_aggression_subtype")
    @classmethod
    def validate_pa_subtype(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and v not in [e.value for e in PASubtype]:
            raise ValueError(f"Invalid subtype: {v}. Must be one of {[e.value for e in PASubtype]}")
        return v

    @field_validator("sarcasm_label")
    @classmethod
    def validate_sarcasm_label(cls, v: Optional[str]) -> Optional[str]:
        valid_sarcasm = ["sarcastic", "non_sarcastic", "ambiguous", "unknown"]
        if v is not None and v not in valid_sarcasm:
            raise ValueError(f"Invalid sarcasm label: {v}. Must be one of {valid_sarcasm}")
        return v

    def to_jsonl_dict(self) -> dict:
        return self.model_dump(mode="json")
