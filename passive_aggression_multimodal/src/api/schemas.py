"""
src/api/schemas.py

Pydantic schemas and contract definitions for FastAPI inference pipeline.
Strictly separates auxiliary sarcasm detection from primary passive-aggression classification.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator


class HealthResponse(BaseModel):
    """Health check response schema."""
    status: str = Field(..., description="Overall service status ('healthy' or 'degraded')")
    loaded_models: List[str] = Field(default_factory=list, description="List of initialized model names")
    model_status: Dict[str, str] = Field(default_factory=dict, description="Detailed readiness status per model")
    device: str = Field(..., description="Device used for tensor operations ('cuda' or 'cpu')")
    passive_aggression_benchmark_status: str = Field(
        default="BLOCKED_PENDING_HUMAN_ANNOTATION",
        description="Strict governance benchmark gate status"
    )
    timestamp: str = Field(..., description="ISO 8601 UTC timestamp")


class MetadataResponse(BaseModel):
    """API metadata and capability specification."""
    project_name: str
    api_version: str
    available_models: List[str]
    tasks_supported: Dict[str, str]
    supported_modalities: List[str]
    expected_labels: List[str]
    known_limitations: List[str]
    passive_aggression_benchmark_gate: str


class TextPredictionRequest(BaseModel):
    """Request payload for text-only sarcasm prediction."""
    text: str = Field(..., description="Social media comment or text input (non-empty)")

    @field_validator("text")
    @classmethod
    def validate_text(cls, v: str) -> str:
        if not isinstance(v, str):
            raise ValueError("Input text must be a string.")
        cleaned = v.strip()
        if not cleaned:
            raise ValueError("Input text cannot be empty or whitespace only.")
        if len(cleaned) > 5000:
            raise ValueError("Input text exceeds maximum allowed length of 5,000 characters.")
        return cleaned


class SarcasmPredictionResponse(BaseModel):
    """Structured response for sarcasm prediction."""
    task: str = Field(default="sarcasm_detection", description="Task identifier")
    model_id: str = Field(..., description="Identifier of the model that produced the prediction")
    predicted_label: str = Field(..., description="Predicted class label ('sarcastic' or 'non_sarcastic')")
    class_scores: Dict[str, float] = Field(..., description="Model class posterior probabilities")
    modality_used: List[str] = Field(..., description="List of modalities utilized ('text', 'image')")
    fallback_applied: bool = Field(default=False, description="True if missing/corrupt image triggered text fallback")
    model_version: str = Field(..., description="Checkpoint or model version string")
    latency_ms: float = Field(..., description="Inference latency in milliseconds")
    warnings: List[str] = Field(default_factory=list, description="Scientific disclaimers and route warnings")


class PassiveAggressionUnavailableResponse(BaseModel):
    """503 Service Unavailable payload for passive aggression endpoint."""
    status: str = Field(default="unavailable")
    task: str = Field(default="passive_aggression_detection")
    detail: str = Field(
        default="Validated passive-aggression inference is currently unavailable because no adequate, independently evaluated PA model exists."
    )
    benchmark_status: str = Field(default="BLOCKED_PENDING_HUMAN_ANNOTATION")
    message: str = Field(
        default="Auxiliary sarcasm models cannot be substituted for passive-aggression detection. Operational definition requires interpersonal hostility, resentment, blame, or criticism with plausible deniability."
    )
    human_annotation_requirement: str = Field(
        default="Requires completion and adjudication of the Phase 6A 30-record human annotation pilot."
    )


class PassiveAggressionExperimentalPredictionResponse(BaseModel):
    """Structured response for exploratory silver-label passive aggression prediction."""
    task: str = Field(default="passive_aggression_detection")
    model_id: str = Field(default="pa_silver_tfidf_v1")
    predicted_label: str = Field(..., description="Predicted label ('passive_aggressive' or 'non_passive_aggressive')")
    class_scores: Dict[str, float] = Field(..., description="Estimated class probabilities")
    label_provenance: str = Field(default="LLM_SILVER")
    evaluation_status: str = Field(default="LLM_SILVER_DIAGNOSTIC_EVALUATION_AVAILABLE")
    latency_ms: float = Field(..., description="Inference latency in milliseconds")
    warnings: List[str] = Field(
        default_factory=lambda: [
            "EXPERIMENTAL MODEL: Trained exclusively on automated LLM silver labels.",
            "This model has NOT been validated against independent human ground truth.",
            "The primary human benchmark remains BLOCKED_PENDING_HUMAN_ANNOTATION."
        ]
    )


class ErrorResponse(BaseModel):
    """Standard error response payload."""
    error: str
    detail: str

