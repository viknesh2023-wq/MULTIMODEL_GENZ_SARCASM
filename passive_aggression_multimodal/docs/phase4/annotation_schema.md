# Phase 4 Human Annotation Schema Specification

**Project Title:** Context-Aware Multimodal Passive-Aggression Detection in Social Media Using Image–Comment Fusion  
**Artifact:** `docs/phase4/annotation_schema.md`  
**Date:** October 21, 2026  
**Version:** 4.0.0  

---

## 1. Overview

This document specifies the exact data contracts, types, valid values, and validation rules for human-annotated passive-aggression benchmarks in social media.

All annotations must adhere strictly to this schema. Automated validation ensures that no invalid labels, incomplete judgments, or missing rationale enter the gold-standard evaluation pipeline.

---

## 2. Core Label Definitions & Values

### 2.1 Primary Labels (`primary_label`)
- `passive_aggressive` (Integer mapping: `0`): Interpersonal hostility, blame, criticism, or resentment delivered covertly with plausible deniability.
- `non_passive_aggressive` (Integer mapping: `1`): Neutral statements, sincere compliments, direct feedback, friendly banter, or venting about external circumstances.
- `ambiguous` (Integer mapping: `2`): Insufficient context or genuinely unverifiable intent.

### 2.2 Subtype Labels (`subtype`)
Applicable when `primary_label == "passive_aggressive"`; optional or `null` otherwise:
- `fake_politeness`: Exaggerated formal politeness or etiquette masking hostility.
- `indirect_criticism`: Implicit fault-finding without direct confrontation.
- `backhanded_compliment`: Diminishing praise or qualifying compliments.
- `implied_blame`: Shifting responsibility indirectly.
- `sarcastic_agreement`: Feigned compliance or agreement to ridicule.
- `dismissiveness`: Stonewalling, minimization, or contemptuous brevity.
- `other_or_unclear`: Other indirect aggressive forms.

### 2.3 Multimodal Role Flags
- `image_role`:
  - `ESSENTIAL`: The image is required to detect the passive aggression (e.g., text is polite, image shows failure/infraction).
  - `HELPFUL`: The image reinforces or clarifies the text, but text alone has indicators.
  - `IRRELEVANT`: The image provides no interpersonal signal.
  - `ABSENT`: No image is associated with the record.
- `context_sufficiency`:
  - `SUFFICIENT`: Surrounding conversational turns or post caption provide clear context.
  - `PARTIAL`: Some context exists, but ambiguity remains.
  - `INSUFFICIENT`: Isolated comment with no contextual grounding.

---

## 3. Pydantic Model Specification

```python
from datetime import datetime
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field, field_validator


class PrimaryLabelEnum(str, Enum):
    PASSIVE_AGGRESSIVE = "passive_aggressive"
    NON_PASSIVE_AGGRESSIVE = "non_passive_aggressive"
    AMBIGUOUS = "ambiguous"


class SubtypeEnum(str, Enum):
    FAKE_POLITENESS = "fake_politeness"
    INDIRECT_CRITICISM = "indirect_criticism"
    BACKHANDED_COMPLIMENT = "backhanded_compliment"
    IMPLIED_BLAME = "implied_blame"
    SARCASTIC_AGREEMENT = "sarcastic_agreement"
    DISMISSIVENESS = "dismissiveness"
    OTHER_OR_UNCLEAR = "other_or_unclear"


class ImageRoleEnum(str, Enum):
    ESSENTIAL = "ESSENTIAL"
    HELPFUL = "HELPFUL"
    IRRELEVANT = "IRRELEVANT"
    ABSENT = "ABSENT"


class ContextSufficiencyEnum(str, Enum):
    SUFFICIENT = "SUFFICIENT"
    PARTIAL = "PARTIAL"
    INSUFFICIENT = "INSUFFICIENT"


class AdjudicationStatusEnum(str, Enum):
    PENDING_REVIEW = "PENDING_REVIEW"
    SINGLE_ANNOTATED = "SINGLE_ANNOTATED"
    AGREEMENT_REACHED = "AGREEMENT_REACHED"
    ADJUDICATED = "ADJUDICATED"


class HumanAnnotationRecord(BaseModel):
    record_id: str = Field(..., description="Unique immutable record identifier")
    source_dataset: str = Field(..., description="Name of source dataset (e.g., MMSD, SarcNet, MUStARD)")
    comment_text: str = Field(..., min_length=1, description="Original social media text under review")
    image_path: Optional[str] = Field(None, description="Path to image if multimodal")
    context_text: Optional[str] = Field(None, description="Conversational context or post caption")
    
    primary_label: PrimaryLabelEnum = Field(..., description="Primary 3-way classification")
    subtype: Optional[SubtypeEnum] = Field(None, description="Subtype taxonomy if passive aggressive")
    image_role: ImageRoleEnum = Field(ImageRoleEnum.IRRELEVANT, description="Role of visual modality")
    context_sufficiency: ContextSufficiencyEnum = Field(..., description="Sufficiency of conversational context")
    
    confidence: int = Field(..., ge=1, le=5, description="Annotator subjective confidence (1=very low, 5=certain)")
    evidence_span: Optional[str] = Field(None, description="Exact phrase or clue indicating the label")
    annotator_id: str = Field(..., description="Pseudonymous annotator identifier")
    annotation_timestamp: datetime = Field(default_factory=datetime.utcnow, description="UTC timestamp of annotation")
    adjudication_status: AdjudicationStatusEnum = Field(
        AdjudicationStatusEnum.PENDING_REVIEW, description="Workflow stage"
    )

    @field_validator("subtype")
    def validate_subtype_consistency(cls, v, values):
        primary = values.data.get("primary_label")
        if primary == PrimaryLabelEnum.PASSIVE_AGGRESSIVE and v is None:
            raise ValueError("subtype is required when primary_label is passive_aggressive")
        return v
```

---

## 4. JSON Schema Representation

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "HumanAnnotationRecord",
  "type": "object",
  "properties": {
    "record_id": { "type": "string" },
    "source_dataset": { "type": "string" },
    "comment_text": { "type": "string", "minLength": 1 },
    "image_path": { "type": ["string", "null"] },
    "context_text": { "type": ["string", "null"] },
    "primary_label": { 
      "type": "string", 
      "enum": ["passive_aggressive", "non_passive_aggressive", "ambiguous"] 
    },
    "subtype": { 
      "type": ["string", "null"],
      "enum": [
        "fake_politeness", "indirect_criticism", "backhanded_compliment", 
        "implied_blame", "sarcastic_agreement", "dismissiveness", 
        "other_or_unclear", null
      ]
    },
    "image_role": { 
      "type": "string", 
      "enum": ["ESSENTIAL", "HELPFUL", "IRRELEVANT", "ABSENT"] 
    },
    "context_sufficiency": { 
      "type": "string", 
      "enum": ["SUFFICIENT", "PARTIAL", "INSUFFICIENT"] 
    },
    "confidence": { "type": "integer", "minimum": 1, "maximum": 5 },
    "evidence_span": { "type": ["string", "null"] },
    "annotator_id": { "type": "string" },
    "annotation_timestamp": { "type": "string", "format": "date-time" },
    "adjudication_status": { 
      "type": "string", 
      "enum": ["PENDING_REVIEW", "SINGLE_ANNOTATED", "AGREEMENT_REACHED", "ADJUDICATED"] 
    }
  },
  "required": [
    "record_id", "source_dataset", "comment_text", 
    "primary_label", "context_sufficiency", "confidence", 
    "annotator_id", "adjudication_status"
  ]
}
```

---

## 5. CSV Template Field Specifications

For annotators working with spreadsheet applications, the CSV template columns must be ordered as follows:

| Column Header | Required | Accepted Values / Format | Description |
|---|---|---|---|
| `record_id` | Yes | String | Unique record ID from manifest |
| `source_dataset` | Yes | String | MMSD, SarcNet, MUStARD, Civil Comments |
| `comment_text` | Yes | String | Text to be evaluated |
| `image_available` | Yes | `TRUE` / `FALSE` | Visual modality presence |
| `context_text` | No | String | Conversation history or caption |
| `primary_label` | Yes | `passive_aggressive`, `non_passive_aggressive`, `ambiguous` | Target classification |
| `subtype` | Conditional | One of 7 subtypes (required if PA) | Passive aggression category |
| `image_role` | Yes | `ESSENTIAL`, `HELPFUL`, `IRRELEVANT`, `ABSENT` | Multimodal contribution |
| `context_sufficiency`| Yes | `SUFFICIENT`, `PARTIAL`, `INSUFFICIENT` | Adequacy of contextual turns |
| `confidence` | Yes | 1 to 5 integer | 1 = guess, 5 = absolute certainty |
| `evidence_span` | No | Text snippet | Trigger phrase or visual contradiction |
| `annotator_id` | Yes | String | e.g., `annotator_01` |
| `notes` | No | String | Freeform rationale or observations |
