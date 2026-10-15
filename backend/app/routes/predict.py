"""
Prediction routes.

POST /api/predict/text       — Real XLM-R inference (model loaded at startup)
POST /api/predict/multimodal — Real 5-branch multimodal feature extraction + 2,847-D fusion MLP
POST /api/predict/explain    — Standalone local Explainable AI reasoner
"""
import io
import logging
from typing import Optional
from fastapi import APIRouter, HTTPException, status, File, UploadFile, Form
from PIL import Image

from app.schemas.schemas import (
    TextPredictRequest,
    TextPredictResponse,
    ProbabilitySet,
    MultimodalPredictResponse,
    MultimodalPredictionBlock,
    SentimentBlock,
    EmotionBlock,
    MultimodalModelBlock,
    MultimodalPerformanceBlock,
    ExplanationBlock,
    ExplanationReasonerMeta,
    ExplanationRequest,
    ExplanationResponse,
)
from app.services import model_service
from app.services.multimodal_service import get_multimodal_manager
from app.services.explanation_service import get_explanation_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/predict", tags=["Prediction"])

MAX_IMAGE_BYTES = 25 * 1024 * 1024  # 25 MB
ALLOWED_MIME_TYPES = {
    "image/jpeg",
    "image/png",
    "image/webp",
    "image/bmp",
    "image/jpg",
}


@router.post("/text", response_model=TextPredictResponse)
async def predict_text(request: TextPredictRequest):
    """
    Classify a text input as sarcastic or non-sarcastic using
    the trained XLM-R model. Returns real softmax probabilities.
    """
    text = request.text.strip()
    if not text:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Text must not be empty.",
        )

    if not model_service.is_loaded():
        err = model_service.get_load_error()
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Model not loaded. Error: {err}",
        )

    try:
        result = model_service.predict_text(text)
        return TextPredictResponse(
            label=result["label"],
            label_name=result["label_name"],
            confidence=result["confidence"],
            probabilities=ProbabilitySet(
                non_sarcastic=result["probabilities"]["non_sarcastic"],
                sarcastic=result["probabilities"]["sarcastic"],
            ),
            device=result["device"],
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        )


@router.post("/multimodal", response_model=MultimodalPredictResponse)
async def predict_multimodal(
    text: str = Form(..., description="Post or caption text to analyze"),
    image: UploadFile = File(..., description="Image file (JPEG, PNG, WEBP)"),
    explain: bool = Form(False, description="Optionally synthesize structured explanation"),
):
    """
    Classify text + image multimodal input.
    Executes real inference across all five feature branches:
    - XLM-R contextual text representation (768-D)
    - OpenAI CLIP visual representation (512-D)
    - InternVL3-2B visual representation (1,536-D)
    - XLM-R 3-class sentiment probabilities (3-D)
    - XLM-R 28-class emotion probabilities (28-D)
    - Saved 2,847-D MultimodalFusionMLP classifier
    Optionally calls the local reasoner if explain=True.
    """
    # 1. Text Validation
    cleaned_text = text.strip() if text else ""
    if not cleaned_text:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Text input cannot be empty or whitespace.",
        )

    # 2. Image File Validation
    if not image or not image.filename:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Image file must be provided.",
        )

    # 3. Read image bytes
    try:
        contents = await image.read()
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to read uploaded image: {exc}",
        )

    if len(contents) == 0:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Uploaded image file is empty (0 bytes).",
        )

    if len(contents) > MAX_IMAGE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Image exceeds maximum allowable size of {MAX_IMAGE_BYTES // (1024*1024)} MB.",
        )

    # 4. Image Decoding & Integrity Check
    try:
        img_buffer = io.BytesIO(contents)
        with Image.open(img_buffer) as probe:
            probe.verify()

        img_buffer.seek(0)
        pil_image = Image.open(img_buffer).convert("RGB")

    except Exception as exc:
        logger.warning(f"Image decode failed: {exc}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Corrupted or unsupported image file. Could not decode into RGB image. ({exc})",
        )

    w, h = pil_image.size
    if w < 10 or h < 10:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Image dimensions ({w}x{h}) are too small. Minimum dimension is 10x10.",
        )
    if w > 10000 or h > 10000:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Image dimensions ({w}x{h}) exceed maximum allowable size (10,000x10,000).",
        )

    # 5. Check Multimodal Manager Readiness
    manager = get_multimodal_manager()
    if not manager.is_ready():
        try:
            manager.load_all_models()
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Multimodal inference pipeline failed to load: {exc}",
            )

    # 6. Run Inference
    try:
        result = manager.predict_multimodal(text=cleaned_text, image=pil_image)
    except Exception as exc:
        logger.error(f"Multimodal inference execution failed: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Multimodal inference error: {exc}",
        )

    # 7. Optional Explanation Stage
    if explain:
        try:
            expl_service = get_explanation_service()
            vis_desc = expl_service.generate_visual_description(pil_image)
            expl_res = expl_service.generate_explanation(
                text=cleaned_text,
                prediction=result["prediction"],
                sentiment=result["sentiment"],
                emotions=result["emotions"],
                visual_evidence=vis_desc,
                image_metadata={"dimensions": f"{w}x{h}"},
            )
            result["explanation"] = expl_res.get("explanation")
            result["reasoner"] = expl_res.get("reasoner")
        except Exception as exc:
            logger.warning(f"Explanation generation failed: {exc}. Returning prediction without explanation.")
            result["explanation"] = None
            result["reasoner"] = None

    return MultimodalPredictResponse(**result)


@router.post("/explain", response_model=ExplanationResponse)
async def explain_prediction(request: ExplanationRequest):
    """
    Dedicated endpoint to synthesize an evidence-grounded natural-language explanation
    from an existing, verified prediction response.
    Never alters the underlying classification verdict or probabilities.
    """
    cleaned_text = request.text.strip()
    if not cleaned_text:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Text cannot be empty.",
        )

    try:
        expl_service = get_explanation_service()
        res = expl_service.generate_explanation(
            text=cleaned_text,
            prediction=request.prediction.dict(),
            sentiment=request.sentiment.dict(),
            emotions=request.emotions.dict(),
            visual_evidence=request.visual_evidence,
        )
        return ExplanationResponse(**res)
    except Exception as exc:
        logger.error(f"Explanation service error: {exc}", exc_info=True)
        return ExplanationResponse(
            status="unavailable",
            explanation=None,
            prediction_reference=request.prediction,
            reasoner={"model": "None", "runtime": "local_cpu"},
            error=str(exc),
        )
