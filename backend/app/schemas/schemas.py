"""
Pydantic schemas for API request/response validation.
"""
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List


# ── Request schemas ──────────────────────────────────────────────────────────

class TextPredictRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Input text to classify")


class MultimodalPredictRequest(BaseModel):
    text: Optional[str] = Field(None, description="Optional text input")
    image_base64: Optional[str] = Field(None, description="Base64 encoded image")


# ── Response schemas ─────────────────────────────────────────────────────────

class ProbabilitySet(BaseModel):
    non_sarcastic: float
    sarcastic: float


class TextPredictResponse(BaseModel):
    label: int
    label_name: str
    confidence: float
    probabilities: ProbabilitySet
    model: str = "XLM-R Base"
    device: str


class MultimodalNotReadyResponse(BaseModel):
    status: str = "not_ready"
    message: str = "Multimodal fusion model is currently under development."
    feature_status: Dict[str, str] = {
        "clip_features": "ready",
        "internvl_features": "ready",
        "sentiment_model": "ready",
        "emotion_model": "ready",
        "fusion_classifier": "ready",
    }


class MultimodalPredictionBlock(BaseModel):
    label: str
    class_id: int
    probabilities: ProbabilitySet
    confidence: float


class SentimentBlock(BaseModel):
    negative: float
    neutral: float
    positive: float


class EmotionBlock(BaseModel):
    label_probabilities: Optional[Dict[str, float]] = None
    top_emotions: Optional[List[Dict[str, Any]]] = None


class MultimodalModelBlock(BaseModel):
    fusion_model: str
    feature_dimensions: int
    branches: Optional[Dict[str, int]] = None


class MultimodalPerformanceBlock(BaseModel):
    total_latency_ms: float
    latency_breakdown_ms: Optional[Dict[str, float]] = None


# ── Explanation schemas ──────────────────────────────────────────────────────

class ExplanationBlock(BaseModel):
    summary: str
    text_evidence: List[str]
    visual_evidence: List[str]
    supporting_signals: List[str]
    limitations: List[str]


class ExplanationReasonerMeta(BaseModel):
    model: str
    runtime: str
    latency_ms: Optional[float] = None


class ExplanationRequest(BaseModel):
    text: str
    prediction: MultimodalPredictionBlock
    sentiment: SentimentBlock
    emotions: EmotionBlock
    visual_evidence: Optional[str] = None


class ExplanationResponse(BaseModel):
    status: str = "success"
    explanation: Optional[ExplanationBlock] = None
    prediction_reference: MultimodalPredictionBlock
    reasoner: ExplanationReasonerMeta
    error: Optional[str] = None


class MultimodalPredictResponse(BaseModel):
    status: str = "success"
    prediction: MultimodalPredictionBlock
    sentiment: SentimentBlock
    emotions: EmotionBlock
    model: MultimodalModelBlock
    performance: MultimodalPerformanceBlock
    explanation: Optional[ExplanationBlock] = None
    reasoner: Optional[ExplanationReasonerMeta] = None


class ModelStatusResponse(BaseModel):
    model: str
    status: str
    device: str
    model_path: str
    loaded: bool
    multimodal_ready: Optional[bool] = False
    explanation_ready: Optional[bool] = False
    explanation_model: Optional[str] = None
    vram_allocated_mb: Optional[float] = None


class HealthResponse(BaseModel):
    status: str
    version: str
    model_loaded: bool
    multimodal_ready: Optional[bool] = False
    explanation_ready: Optional[bool] = False


class MetricsResponse(BaseModel):
    test_accuracy: float
    test_macro_precision: float
    test_macro_recall: float
    test_macro_f1: float
    confusion_matrix: list
    train_samples: int
    validation_samples: int
    test_samples: int
    epochs: int
    learning_rate: float
    model_name: str
    text_sarcasm: Optional[Dict[str, Any]] = None
    sentiment: Optional[Dict[str, Any]] = None
    emotion: Optional[Dict[str, Any]] = None
    multimodal_fusion: Optional[Dict[str, Any]] = None
    ablation_benchmark: Optional[List[Dict[str, Any]]] = None


class DatasetStatsResponse(BaseModel):
    text_dataset: Dict[str, Any]
    multimodal_dataset: Dict[str, Any]
    sentiment_dataset: Optional[Dict[str, Any]] = None
    emotion_dataset: Optional[Dict[str, Any]] = None


class FeatureStatusResponse(BaseModel):
    features: list
