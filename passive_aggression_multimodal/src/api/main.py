"""
src/api/main.py

FastAPI Application Entry Point for Context-Aware Multimodal Inference.
Exposes endpoints for sarcasm prediction, modality fallback, health checks,
and enforces strict governance gating for passive-aggression classification.
"""

from __future__ import annotations

import logging
import sys
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse

from src.api.schemas import (
    ErrorResponse,
    HealthResponse,
    MetadataResponse,
    PassiveAggressionExperimentalPredictionResponse,
    PassiveAggressionUnavailableResponse,
    SarcasmPredictionResponse,
    TextPredictionRequest,
)
from src.api.service import ModelInferenceService

DEMO_HTML_PATH = Path(__file__).resolve().parent / "demo.html"

# Setup privacy-conscious logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("Phase9-API")

inference_service: Optional[ModelInferenceService] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan context manager loading models once at startup."""
    global inference_service
    logger.info("Starting Phase 9 FastAPI Application...")
    inference_service = ModelInferenceService()
    try:
        inference_service.initialize_models()
    except Exception as e:
        logger.error("Error during model startup: %s", e)
    app.state.service = inference_service
    yield
    logger.info("Shutting down Phase 9 FastAPI Application...")


app = FastAPI(
    title="Context-Aware Multimodal Social Media Inference API",
    version="1.0.0",
    description=(
        "Production inference pipeline serving text-only and multimodal sarcasm detection models "
        "with explicit modality fallback. Preserves strict governance boundaries for passive-aggression benchmark."
    ),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def privacy_conscious_access_log(request: Request, call_next):
    """Logs request metadata without logging full comment texts or raw images."""
    t0 = time.perf_counter()
    response = await call_next(request)
    duration_ms = (time.perf_counter() - t0) * 1000.0
    logger.info(
        "METHOD=%s PATH=%s STATUS=%d DURATION=%.2fms",
        request.method,
        request.url.path,
        response.status_code,
        duration_ms,
    )
    return response


# =========================================================================
# EXCEPTION HANDLERS
# =========================================================================

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Custom handler for pydantic input validation failures."""
    details = exc.errors()
    msg = "; ".join([f"{err.get('loc')}: {err.get('msg')}" for err in details])
    logger.warning("Input validation error on %s: %s", request.url.path, msg)
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"error": "ValidationError", "detail": msg},
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Catches unhandled errors without leaking stack traces to clients."""
    logger.error("Unhandled error on %s: %s", request.url.path, exc, exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"error": "InternalServerError", "detail": "An internal inference error occurred."},
    )


# =========================================================================
# ENDPOINTS
# =========================================================================

@app.get("/health", response_model=HealthResponse, tags=["System"])
def get_health(request: Request):
    """
    Returns service health, loaded models, device, and benchmark status.
    A model is only reported as ready if it passed its startup smoke test.
    """
    svc: ModelInferenceService = getattr(request.app.state, "service", inference_service)
    if svc is None:
        return HealthResponse(
            status="uninitialized",
            loaded_models=[],
            model_status={},
            device="unknown",
            passive_aggression_benchmark_status="BLOCKED_PENDING_HUMAN_ANNOTATION",
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    is_healthy = len(svc.loaded_models) > 0
    return HealthResponse(
        status="healthy" if is_healthy else "degraded",
        loaded_models=svc.loaded_models,
        model_status=svc.model_status,
        device=str(svc.device),
        passive_aggression_benchmark_status="BLOCKED_PENDING_HUMAN_ANNOTATION",
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


@app.get("/metadata", response_model=MetadataResponse, tags=["System"])
def get_metadata():
    """
    Returns full API capability metadata, supported tasks, label schemas,
    and explicit disclosures of model boundaries and limitations.
    """
    return MetadataResponse(
        project_name="Context-Aware Multimodal Passive-Aggression Detection in Social Media Using Image–Comment Fusion",
        api_version="1.0.0",
        available_models=["text_tfidf_baseline", "gated_multimodal_fusion"],
        tasks_supported={
            "text_tfidf_baseline": "sarcasm_detection",
            "gated_multimodal_fusion": "sarcasm_detection",
            "passive_aggression": "UNAVAILABLE_PENDING_HUMAN_ANNOTATION",
        },
        supported_modalities=["text", "image"],
        expected_labels=["non_sarcastic", "sarcastic"],
        known_limitations=[
            "Auxiliary Sarcasm Only: Served models detect sarcasm patterns from social media data; they do NOT detect passive aggression.",
            "Text Baseline: Statistical word+char n-gram model; lacks deep syntactic reasoning.",
            "Multimodal Gated Fusion: In offline execution, visual features are low-level spatial statistics; learned gate heavily relies on the text modality.",
            "Primary Benchmark Status: Real-world passive-aggression evaluation remains blocked until the 30-record Phase 6A human pilot is completed.",
        ],
        passive_aggression_benchmark_gate="BLOCKED_PENDING_HUMAN_ANNOTATION",
    )


@app.post(
    "/predict/sarcasm/text",
    response_model=SarcasmPredictionResponse,
    responses={422: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
    tags=["Sarcasm Inference"],
)
def predict_sarcasm_text(payload: TextPredictionRequest, request: Request):
    """
    Stateless text-only sarcasm prediction using the verified Phase 8A TF-IDF baseline.
    Rejects empty or whitespace-only inputs with HTTP 422.
    """
    svc: ModelInferenceService = getattr(request.app.state, "service", inference_service)
    if svc is None or svc.text_baseline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Text baseline model is not initialized.",
        )

    try:
        res = svc.predict_sarcasm_text(payload.text)
        return SarcasmPredictionResponse(**res)
    except Exception as e:
        logger.error("Text prediction failed: %s", e)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@app.post(
    "/predict/sarcasm/multimodal",
    response_model=SarcasmPredictionResponse,
    responses={422: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
    tags=["Sarcasm Inference"],
)
async def predict_sarcasm_multimodal(
    request: Request,
    text: str = Form(..., description="Comment text (required non-empty string)"),
    image: Optional[UploadFile] = File(None, description="Optional image upload (JPEG/PNG/WebP)"),
    fallback_on_invalid_image: bool = Query(
        False,
        description="If True, routes unreadable/corrupted images to text fallback instead of returning HTTP 422.",
    ),
):
    """
    Multimodal sarcasm prediction combining comment text and image.
    Behavior:
      - Valid image provided -> Gated multimodal fusion prediction.
      - Image omitted/empty -> Seamless text-only baseline fallback (explicitly reported in modality_used and warnings).
      - Corrupt image -> HTTP 422 error by default, or text fallback if fallback_on_invalid_image=True.
    """
    svc: ModelInferenceService = getattr(request.app.state, "service", inference_service)
    if svc is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Inference service is not initialized.",
        )

    # 1. Validate text input
    if not text or not text.strip():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Input text cannot be empty or whitespace only.",
        )
    if len(text.strip()) > 5000:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Input text exceeds maximum length of 5,000 characters.",
        )

    # 2. Read optional image bytes
    image_bytes = None
    if image is not None:
        # Check content type if supplied
        if image.content_type and image.content_type not in ["image/jpeg", "image/png", "image/webp", "application/octet-stream"]:
            if not fallback_on_invalid_image:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail=f"Unsupported image content-type '{image.content_type}'. Supported: image/jpeg, image/png, image/webp.",
                )
        image_bytes = await image.read()

    # 3. Execute inference
    try:
        res = svc.predict_sarcasm_multimodal(
            text=text.strip(),
            image_bytes=image_bytes,
            fallback_on_invalid_image=fallback_on_invalid_image,
        )
        return SarcasmPredictionResponse(**res)
    except ValueError as val_err:
        logger.warning("Multimodal image validation error: %s", val_err)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(val_err),
        )
    except Exception as e:
        logger.error("Multimodal prediction failed: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal multimodal inference failed.",
        )


@app.post(
    "/predict/passive-aggression",
    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
    response_model=PassiveAggressionUnavailableResponse,
    tags=["Governance & Research Gating"],
)
def predict_passive_aggression():
    """
    Strict Governance Boundary: Returns HTTP 503 Service Unavailable.
    Discloses that passive-aggression classification is locked pending independent human annotation.
    """
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content=PassiveAggressionUnavailableResponse().model_dump(),
    )


@app.post(
    "/predict/passive-aggression/experimental",
    status_code=status.HTTP_200_OK,
    response_model=PassiveAggressionExperimentalPredictionResponse,
    tags=["Exploratory Research Gating"],
)
def predict_passive_aggression_experimental(
    request: Request,
    body: TextPredictionRequest,
):
    """
    Exploratory Silver-Label Passive Aggression Prediction.
    Exposes exploratory model trained exclusively on accepted LLM-annotated silver labels.
    Explicitly labeled with provenance LLM_SILVER and unvalidated status.
    """
    svc: ModelInferenceService = getattr(request.app.state, "service", inference_service)
    if svc is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Inference service is not initialized.",
        )
    try:
        res = svc.predict_passive_aggression_experimental(body.text)
        return PassiveAggressionExperimentalPredictionResponse(**res)
    except RuntimeError as r_err:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(r_err),
        )
    except Exception as e:
        logger.error("Experimental PA prediction failed: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal experimental prediction failed.",
        )


# =========================================================================
# INTERACTIVE DEMO DASHBOARD & API ALIASES
# =========================================================================

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
@app.get("/demo", response_class=HTMLResponse, tags=["Demo Dashboard"])
def get_demo_dashboard():
    """Serves the interactive single-page demo dashboard."""
    if DEMO_HTML_PATH.exists():
        return HTMLResponse(content=DEMO_HTML_PATH.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>Context-Aware Multimodal Inference API</h1><p>Demo UI file missing. Visit <a href='/docs'>/docs</a>.</p>")


# Convenience aliases for frontend compatibility
app.add_api_route("/api/health", get_health, methods=["GET"], include_in_schema=False)
app.add_api_route("/api/metadata", get_metadata, methods=["GET"], include_in_schema=False)
app.add_api_route("/api/model/status", get_health, methods=["GET"], include_in_schema=False)
app.add_api_route("/api/predict/text", predict_sarcasm_text, methods=["POST"], include_in_schema=False)
app.add_api_route("/api/predict/multimodal", predict_sarcasm_multimodal, methods=["POST"], include_in_schema=False)
app.add_api_route("/api/predict/passive-aggression", predict_passive_aggression, methods=["POST"], include_in_schema=False)
app.add_api_route("/api/predict/passive-aggression/experimental", predict_passive_aggression_experimental, methods=["POST"], include_in_schema=False)

