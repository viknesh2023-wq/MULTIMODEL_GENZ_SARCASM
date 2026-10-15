"""
FastAPI application entry point.

Loads both text sarcasm and multimodal models at startup via lifespan handler.
Models are NEVER reloaded per request.
"""
import sys
import logging
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Ensure backend directory is in python path
backend_dir = str(Path(__file__).resolve().parent.parent)
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from app.routes import predict, model, data
from app.services import model_service
from app.services.multimodal_service import get_multimodal_manager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models once when the server starts."""
    logger.info("=" * 60)
    logger.info("  Multimodal Sarcasm Intelligence — Backend Starting")
    logger.info("=" * 60)
    try:
        model_service.load_model()
        logger.info("✓ XLM-R text sarcasm model ready.")
    except Exception as exc:
        logger.error(f"✗ XLM-R text model load failed: {exc}")
        logger.warning("Server will start but /api/predict/text may return 503.")

    try:
        mm_manager = get_multimodal_manager()
        mm_manager.load_all_models()
        logger.info("✓ Multimodal pipeline ready (5 branches + Fusion MLP).")
    except Exception as exc:
        logger.error(f"✗ Multimodal pipeline load failed: {exc}")
        logger.warning("Server will start but /api/predict/multimodal may return 503.")

    yield
    logger.info("Backend shutting down.")


app = FastAPI(
    title="Multimodal Sarcasm Intelligence API",
    description=(
        "Production backend for the Gen-Z Multimodal Sarcasm Detection Research Platform. "
        "Provides real request-time inference for both text-only XLM-R sarcasm detection "
        "and 5-branch multimodal fusion classification (2,847-D)."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# Allow the Vite dev server and localhost origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:5174",
        "http://localhost:3000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:5174",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routes
app.include_router(predict.router)
app.include_router(model.router)
app.include_router(data.router)


@app.get("/")
async def root():
    return {
        "name": "Multimodal Sarcasm Intelligence API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/api/health",
        "multimodal_predict": "/api/predict/multimodal",
        "text_predict": "/api/predict/text",
    }
