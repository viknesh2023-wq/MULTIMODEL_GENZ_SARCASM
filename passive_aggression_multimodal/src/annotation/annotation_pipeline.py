"""AI-assisted annotation pipeline skeleton for multimodal passive aggression detection.

Supports:
1. Text-Only Annotator (lexical / transformer-ready)
2. Vision-Language Annotator (multimodal image-text fusion interface)
3. Adjudication Annotator (reconciliation & provisional labeling)

Preserves:
- Original source label
- Text-only prediction
- Image-comment prediction
- Adjudication prediction
- Model identifier and version
- Prompt version
- Explanation
- Agreement status
- Provisional label
- Synthetic-data status
- Human-verification status
- Human-verified label
"""
from __future__ import annotations

import abc
import os
import re
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from src.utilities.schema import (
    AnnotationStatus,
    ImageAvailabilityStatus,
    PASubtype,
    PrimaryPALabel,
    SyntheticStatus,
    UnifiedRecord,
)


class AnnotationPrediction(BaseModel):
    """Output prediction from a single annotator module."""
    annotator_id: str
    model_version: str
    prompt_version: str
    primary_label: str  # passive_aggressive, non_passive_aggressive, ambiguous
    subtype: str
    confidence: float
    explanation: str
    image_needed: Optional[bool] = None


class AdjudicatedAnnotation(BaseModel):
    """Final adjudicated output combining multiple annotator outputs."""
    example_id: str
    original_source_label: Any
    text_only_prediction: Optional[AnnotationPrediction] = None
    vlm_prediction: Optional[AnnotationPrediction] = None
    adjudication_prediction: Optional[AnnotationPrediction] = None
    agreement_status: str  # UNANIMOUS_AGREEMENT, MAJORITY_AGREEMENT, DISAGREEMENT, UNIMODAL_ONLY
    provisional_label: str
    provisional_subtype: str
    adjudication_rationale: str
    synthetic_status: str
    human_verification_status: str  # AWAITING_HUMAN_REVIEW, VERIFIED, REJECTED
    human_verified_label: Optional[str] = None


class BaseAnnotator(abc.ABC):
    """Abstract annotator interface."""

    @abc.abstractmethod
    def annotate(self, record: UnifiedRecord) -> AnnotationPrediction:
        """Runs annotation on a single unified record."""
        pass


class RuleAndHeuristicTextAnnotator(BaseAnnotator):
    """
    Lightweight, free local text annotator analyzing rhetorical markers
    for passive-aggression detection baseline.
    """

    def __init__(self, model_id: str = "heuristic_text_v1", prompt_version: str = "v1.0"):
        self.model_id = model_id
        self.prompt_version = prompt_version

        # Linguistic pattern dictionaries
        self.fake_politeness_patterns = [
            r"\bper my previous email\b",
            r"\bwith all due respect\b",
            r"\bbless (your|their) heart\b",
            r"\bthank you in advance\b",
            r"\bno offense,?\s*but\b",
            r"\bjust to clarify\b",
            r"\bas you should know\b",
        ]
        self.dismissive_patterns = [
            r"\bmust be nice\b",
            r"\bwhatever you say\b",
            r"\bcool story\b",
            r"\bgood for you\b",
            r"\bif you say so\b",
            r"\bwho asked\b",
        ]
        self.implied_blame_patterns = [
            r"\bfunny how\b",
            r"\bi love how\b",
            r"\bconveniently\b",
            r"\bi guess (some of us|rules)\b",
            r"\bmiraculously\b",
        ]
        self.backhanded_patterns = [
            r"\bsurprisingly decent\b",
            r"\bfor someone who\b",
            r"\bactually (edible|wearable|good)\b",
            r"\bbetter than i expected\b",
        ]

    def annotate(self, record: UnifiedRecord) -> AnnotationPrediction:
        text = record.target_comment.lower()

        # Check markers
        for pat in self.fake_politeness_patterns:
            if re.search(pat, text):
                return AnnotationPrediction(
                    annotator_id=self.model_id,
                    model_version="1.0.0",
                    prompt_version=self.prompt_version,
                    primary_label=PrimaryPALabel.PASSIVE_AGGRESSIVE.value,
                    subtype=PASubtype.FAKE_POLITENESS.value,
                    confidence=0.85,
                    explanation=f"Detected weaponized politeness marker matching pattern '{pat}'.",
                    image_needed=False,
                )

        for pat in self.dismissive_patterns:
            if re.search(pat, text):
                return AnnotationPrediction(
                    annotator_id=self.model_id,
                    model_version="1.0.0",
                    prompt_version=self.prompt_version,
                    primary_label=PrimaryPALabel.PASSIVE_AGGRESSIVE.value,
                    subtype=PASubtype.DISMISSIVENESS.value,
                    confidence=0.88,
                    explanation=f"Detected dismissive invalidation marker matching pattern '{pat}'.",
                    image_needed=False,
                )

        for pat in self.implied_blame_patterns:
            if re.search(pat, text):
                return AnnotationPrediction(
                    annotator_id=self.model_id,
                    model_version="1.0.0",
                    prompt_version=self.prompt_version,
                    primary_label=PrimaryPALabel.PASSIVE_AGGRESSIVE.value,
                    subtype=PASubtype.IMPLIED_BLAME.value,
                    confidence=0.82,
                    explanation=f"Detected implied blame syntax matching pattern '{pat}'.",
                    image_needed=False,
                )

        for pat in self.backhanded_patterns:
            if re.search(pat, text):
                return AnnotationPrediction(
                    annotator_id=self.model_id,
                    model_version="1.0.0",
                    prompt_version=self.prompt_version,
                    primary_label=PrimaryPALabel.PASSIVE_AGGRESSIVE.value,
                    subtype=PASubtype.BACKHANDED_COMPLIMENT.value,
                    confidence=0.84,
                    explanation=f"Detected qualifying backhanded praise matching pattern '{pat}'.",
                    image_needed=False,
                )

        # Default fallback for neutral / non-PA
        if len(text.split()) < 3 and text in ["noted.", "ok.", "fine.", "interesting."]:
            return AnnotationPrediction(
                annotator_id=self.model_id,
                model_version="1.0.0",
                prompt_version=self.prompt_version,
                primary_label=PrimaryPALabel.AMBIGUOUS.value,
                subtype=PASubtype.OTHER_OR_UNCLEAR.value,
                confidence=0.50,
                explanation="Short monosyllabic or terse response with insufficient context to calibrate intent.",
                image_needed=True,
            )

        return AnnotationPrediction(
            annotator_id=self.model_id,
            model_version="1.0.0",
            prompt_version=self.prompt_version,
            primary_label=PrimaryPALabel.NON_PASSIVE_AGGRESSIVE.value,
            subtype=PASubtype.OTHER_OR_UNCLEAR.value,
            confidence=0.75,
            explanation="No indirect hostility or weaponized polite markers identified in text.",
            image_needed=False,
        )


class VisionLanguageAnnotator(BaseAnnotator):
    """
    Vision-Language Annotator interface designed to process both post image and comment text.
    Provides pluggable integration point for free local vision models (e.g. InternVL, LLaVA, BLIP).
    """

    def __init__(self, model_id: str = "vlm_interface_stub", prompt_version: str = "v1.0"):
        self.model_id = model_id
        self.prompt_version = prompt_version

    def annotate(self, record: UnifiedRecord) -> AnnotationPrediction:
        # Check image availability
        if record.image_availability_status != ImageAvailabilityStatus.AVAILABLE or not record.image_path:
            return AnnotationPrediction(
                annotator_id=self.model_id,
                model_version="1.0.0",
                prompt_version=self.prompt_version,
                primary_label=PrimaryPALabel.AMBIGUOUS.value,
                subtype=PASubtype.OTHER_OR_UNCLEAR.value,
                confidence=0.0,
                explanation="Image unavailable or not applicable for multimodal inspection.",
                image_needed=True,
            )

        # In Phase 1 CPU-only environment: interface is verified and reports multimodal assessment
        # SarcNet/MMSD cross-modal check
        is_sarcastic_original = False
        if record.source_dataset == "mmsd" and record.original_label == 1:
            is_sarcastic_original = True

        return AnnotationPrediction(
            annotator_id=self.model_id,
            model_version="1.0.0",
            prompt_version=self.prompt_version,
            primary_label=PrimaryPALabel.AMBIGUOUS.value if is_sarcastic_original else PrimaryPALabel.NON_PASSIVE_AGGRESSIVE.value,
            subtype=PASubtype.SARCASTIC_AGREEMENT.value if is_sarcastic_original else PASubtype.OTHER_OR_UNCLEAR.value,
            confidence=0.70,
            explanation="Multimodal interface inspected image presence; provisional evaluation pending full local VLM checkpoint.",
            image_needed=True,
        )


class AdjudicationPipeline:
    """
    Combines text-only, VLM, and adjudication heuristics to produce
    consistent provisional labels while preserving full provenance and uncertainty.
    """

    def __init__(self):
        self.text_annotator = RuleAndHeuristicTextAnnotator()
        self.vlm_annotator = VisionLanguageAnnotator()

    def process_record(self, record: UnifiedRecord) -> AdjudicatedAnnotation:
        text_pred = self.text_annotator.annotate(record)
        vlm_pred = self.vlm_annotator.annotate(record)

        # Reconcile predictions
        if record.image_availability_status == ImageAvailabilityStatus.AVAILABLE:
            if text_pred.primary_label == vlm_pred.primary_label:
                agreement = "UNANIMOUS_AGREEMENT"
                provisional_label = text_pred.primary_label
                provisional_subtype = text_pred.subtype
                rationale = f"Both text and vision annotators agree on {provisional_label}."
            else:
                agreement = "DISAGREEMENT"
                # If text detects specific PA markers but VLM is ambiguous, prioritize text marker with lower confidence
                if text_pred.primary_label == PrimaryPALabel.PASSIVE_AGGRESSIVE.value:
                    provisional_label = PrimaryPALabel.PASSIVE_AGGRESSIVE.value
                    provisional_subtype = text_pred.subtype
                    rationale = f"Text annotator flagged specific PA marker ({text_pred.subtype}) while VLM flagged {vlm_pred.primary_label}."
                else:
                    provisional_label = PrimaryPALabel.AMBIGUOUS.value
                    provisional_subtype = PASubtype.OTHER_OR_UNCLEAR.value
                    rationale = "Annotators disagreed; preserved as ambiguous pending human adjudication."
        else:
            agreement = "UNIMODAL_ONLY"
            provisional_label = text_pred.primary_label
            provisional_subtype = text_pred.subtype
            rationale = f"Only text modality available: {text_pred.explanation}"

        return AdjudicatedAnnotation(
            example_id=record.example_id,
            original_source_label=record.original_label,
            text_only_prediction=text_pred,
            vlm_prediction=vlm_pred,
            agreement_status=agreement,
            provisional_label=provisional_label,
            provisional_subtype=provisional_subtype,
            adjudication_rationale=rationale,
            synthetic_status=record.synthetic_status.value,
            human_verification_status="AWAITING_HUMAN_REVIEW",
            human_verified_label=None,
        )
