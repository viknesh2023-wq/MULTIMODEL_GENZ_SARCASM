"""
Multimodal Inference Pipeline Service (Milestone 5)
Project: Gen-Z Multimodal Sarcasm Detection System
Hardware: NVIDIA GeForce RTX 4050 Laptop GPU / CUDA Acceleration

Manages the 5-branch feature extraction and 2,847-D fusion classification pipeline:
1. XLM-R Text Contextual Representation (768-D, masked mean-pooled)
2. OpenAI CLIP ViT-B/32 Visual Representation (512-D)
3. InternVL3-2B Dynamic Tiled Visual Representation (1,536-D)
4. XLM-R Sentiment Softmax Probabilities (3-D: [Negative, Neutral, Positive])
5. XLM-R Emotion Continuous Sigmoid Probabilities (28-D GoEmotions)
6. Trained MultimodalFusionMLP (2,847-D input with trained StandardScaler)
"""

import os
import sys
import time
import logging
import threading
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
import clip

PROJECT_ROOT = Path(r"D:\NLP_PROJECT")
if str(PROJECT_ROOT / "fusion") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "fusion"))

from fusion_inference import MultimodalFusionInference, EXPECTED_DIMENSION

logger = logging.getLogger(__name__)

# Model directories
TEXT_MODEL_DIR = PROJECT_ROOT / "models" / "xlmr_text_sarcasm_full"
SENT_MODEL_DIR = PROJECT_ROOT / "models" / "xlmr_sentiment_model"
EMO_MODEL_DIR = PROJECT_ROOT / "models" / "xlmr_emotion_model"
FUSION_DIR = PROJECT_ROOT / "models" / "multimodal_fusion"
VLM_MODEL_NAME = "OpenGVLab/InternVL3-2B"
CLIP_MODEL_NAME = "ViT-B/32"

MAX_TEXT_LENGTH = 128
VLM_IMAGE_SIZE = 448
VLM_MAX_TILES = 12

EMOTION_LABELS = [
    "admiration", "amusement", "anger", "annoyance", "approval",
    "caring", "confusion", "curiosity", "desire", "disappointment",
    "disapproval", "disgust", "embarrassment", "excitement", "fear",
    "gratitude", "grief", "joy", "love", "nervousness",
    "optimism", "pride", "realization", "relief", "remorse",
    "sadness", "surprise", "neutral"
]

SENTIMENT_LABELS = ["negative", "neutral", "positive"]

# Dynamic tiling helper for InternVL3-2B
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
vlm_transform = transforms.Compose([
    transforms.Lambda(lambda img: img.convert("RGB")),
    transforms.Resize((VLM_IMAGE_SIZE, VLM_IMAGE_SIZE), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


def dynamic_preprocess(image: Image.Image, image_size=448, max_num=12, use_thumbnail=True) -> torch.Tensor:
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    target_ratios = set(
        (i, j) for n in range(1, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if 1 <= i * j <= max_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    best_diff = float("inf")
    best_ratio = (1, 1)
    area = orig_width * orig_height

    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_diff:
            best_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_diff:
            target_area = image_size * image_size * ratio[0] * ratio[1]
            if 0.5 * target_area < area < 2 * target_area:
                best_ratio = ratio

    target_width = image_size * best_ratio[0]
    target_height = image_size * best_ratio[1]
    resized_img = image.resize((target_width, target_height))

    processed_images = []
    total_tiles = best_ratio[0] * best_ratio[1]
    for i in range(total_tiles):
        box = (
            (i % best_ratio[0]) * image_size,
            (i // best_ratio[0]) * image_size,
            ((i % best_ratio[0]) + 1) * image_size,
            ((i // best_ratio[0]) + 1) * image_size,
        )
        processed_images.append(resized_img.crop(box))

    if use_thumbnail and len(processed_images) != 1:
        processed_images.append(image.resize((image_size, image_size)))

    return torch.stack([vlm_transform(img) for img in processed_images])


class MultimodalInferenceManager:
    """
    Singleton Manager with resource-aware concurrency control.
    Avoids duplicating models in memory and prevents GPU VRAM exhaustion.
    """
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.inference_lock = threading.Lock()
        
        # Models
        self.text_tokenizer = None
        self.text_encoder = None
        self.sent_tokenizer = None
        self.sent_model = None
        self.emo_tokenizer = None
        self.emo_model = None
        self.clip_model = None
        self.clip_preprocess = None
        self.vlm_model = None
        self.fusion_engine = None
        
        self.loaded = False
        self.load_error: Optional[str] = None
        self.load_times: Dict[str, float] = {}
        self._initialized = True

    def load_all_models(self) -> None:
        """
        Loads all required models following the verified VRAM budget.
        - Text encoder: Reused from existing text model or loaded onto CUDA.
        - Sentiment: CPU (0 VRAM).
        - Emotion: CPU (0 VRAM).
        - CLIP: CUDA (360 MB VRAM).
        - InternVL3-2B: CUDA bfloat16 (3.9 GB VRAM).
        - Fusion MLP: CUDA (14 MB VRAM).
        """
        if self.loaded:
            return

        with self.inference_lock:
            try:
                logger.info("Initializing Multimodal Inference Pipeline models...")

                # 1. XLM-R Sarcasm Text Model & Tokenizer
                t0 = time.time()
                from app.services import model_service
                if model_service.is_loaded() and model_service._model is not None:
                    logger.info("Reusing resident XLM-R model from model_service.")
                    self.text_tokenizer = model_service._tokenizer
                    self.text_encoder = model_service._model.roberta
                else:
                    logger.info(f"Loading XLM-R from {TEXT_MODEL_DIR}...")
                    self.text_tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_DIR)
                    cls_model = AutoModelForSequenceClassification.from_pretrained(TEXT_MODEL_DIR)
                    self.text_encoder = cls_model.roberta.to(self.device).eval()
                self.load_times["text_encoder"] = round(time.time() - t0, 3)

                # 2. Sentiment Model (CPU)
                t0 = time.time()
                logger.info(f"Loading Sentiment model from {SENT_MODEL_DIR} on CPU...")
                self.sent_tokenizer = AutoTokenizer.from_pretrained(SENT_MODEL_DIR)
                self.sent_model = AutoModelForSequenceClassification.from_pretrained(SENT_MODEL_DIR).eval()
                self.load_times["sentiment"] = round(time.time() - t0, 3)

                # 3. Emotion Model (CPU)
                t0 = time.time()
                logger.info(f"Loading Emotion model from {EMO_MODEL_DIR} on CPU...")
                self.emo_tokenizer = AutoTokenizer.from_pretrained(EMO_MODEL_DIR)
                self.emo_model = AutoModelForSequenceClassification.from_pretrained(EMO_MODEL_DIR).eval()
                self.load_times["emotion"] = round(time.time() - t0, 3)

                # 4. CLIP ViT-B/32 (CUDA)
                t0 = time.time()
                logger.info("Loading OpenAI CLIP ViT-B/32 on CUDA...")
                self.clip_model, self.clip_preprocess = clip.load(CLIP_MODEL_NAME, device=self.device)
                self.clip_model.eval()
                self.load_times["clip"] = round(time.time() - t0, 3)

                # 5. InternVL3-2B (CUDA bfloat16)
                t0 = time.time()
                logger.info(f"Loading {VLM_MODEL_NAME} (bfloat16) on CUDA...")
                self.vlm_model = AutoModel.from_pretrained(
                    VLM_MODEL_NAME,
                    torch_dtype=torch.bfloat16,
                    low_cpu_mem_usage=True,
                    trust_remote_code=True
                ).to(self.device).eval()
                self.load_times["vlm"] = round(time.time() - t0, 3)

                # 6. Fusion Classifier & Scaler
                t0 = time.time()
                logger.info(f"Loading MultimodalFusionMLP from {FUSION_DIR}...")
                self.fusion_engine = MultimodalFusionInference(model_dir=FUSION_DIR, device=str(self.device))
                self.load_times["fusion"] = round(time.time() - t0, 3)

                self.loaded = True
                self.load_error = None
                logger.info("All 5 multimodal feature extractors + Fusion MLP loaded successfully.")

            except Exception as exc:
                self.loaded = False
                self.load_error = str(exc)
                logger.error(f"Failed to initialize multimodal pipeline: {exc}", exc_info=True)
                raise

    def is_ready(self) -> bool:
        return self.loaded and (self.load_error is None)

    def get_status(self) -> Dict[str, Any]:
        vram_allocated_mb = 0.0
        vram_reserved_mb = 0.0
        if torch.cuda.is_available():
            vram_allocated_mb = round(torch.cuda.memory_allocated() / (1024 ** 2), 2)
            vram_reserved_mb = round(torch.cuda.memory_reserved() / (1024 ** 2), 2)

        return {
            "loaded": self.loaded,
            "error": self.load_error,
            "device": str(self.device),
            "vram_allocated_mb": vram_allocated_mb,
            "vram_reserved_mb": vram_reserved_mb,
            "load_times": self.load_times,
            "branches": {
                "text_768d": self.text_encoder is not None,
                "clip_512d": self.clip_model is not None,
                "vlm_1536d": self.vlm_model is not None,
                "sentiment_3d": self.sent_model is not None,
                "emotion_28d": self.emo_model is not None,
                "fusion_mlp": self.fusion_engine is not None,
            }
        }

    def predict_multimodal(self, text: str, image: Image.Image) -> Dict[str, Any]:
        """
        Executes request-time multimodal inference:
        Extracts 5 branches -> canonical concatenation (2847-D) -> fusion MLP.
        Thread-safe under GPU concurrency lock.
        """
        if not self.loaded:
            self.load_all_models()

        start_total = time.time()
        latencies: Dict[str, float] = {}

        with self.inference_lock:
            # 1. Text Feature (768-D)
            t0 = time.time()
            text_inputs = self.text_tokenizer(
                text,
                max_length=MAX_TEXT_LENGTH,
                padding=True,
                truncation=True,
                return_tensors="pt"
            )
            text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}
            with torch.inference_mode():
                text_outputs = self.text_encoder(**text_inputs)
                last_hidden = text_outputs.last_hidden_state
                mask = text_inputs["attention_mask"].unsqueeze(-1).expand(last_hidden.size()).float()
                sum_embeddings = torch.sum(last_hidden * mask, dim=1)
                sum_mask = mask.sum(dim=1).clamp(min=1e-9)
                feat_text = (sum_embeddings / sum_mask).float().cpu().numpy()[0].astype(np.float32)
            latencies["text_ms"] = round((time.time() - t0) * 1000, 2)

            # 2. CLIP Image Feature (512-D)
            t0 = time.time()
            clip_tensor = self.clip_preprocess(image).unsqueeze(0).to(self.device)
            with torch.inference_mode():
                feat_clip = self.clip_model.encode_image(clip_tensor).float().cpu().numpy().reshape(-1).astype(np.float32)
            latencies["clip_ms"] = round((time.time() - t0) * 1000, 2)

            # 3. InternVL3-2B Image Feature (1,536-D)
            t0 = time.time()
            pixel_values = dynamic_preprocess(
                image,
                image_size=VLM_IMAGE_SIZE,
                max_num=VLM_MAX_TILES,
                use_thumbnail=True
            ).to(self.device, dtype=torch.bfloat16)

            with torch.inference_mode():
                vit_embeds = self.vlm_model.extract_feature(pixel_values)
                # vit_embeds shape: (tiles, 256, 1536)
                # Pooling: average across tokens, then average across tiles
                feat_vlm = vit_embeds.mean(dim=1).mean(dim=0).float().cpu().numpy().astype(np.float32)
            latencies["vlm_ms"] = round((time.time() - t0) * 1000, 2)

            # 4. Sentiment Probabilities (3-D)
            t0 = time.time()
            sent_inputs = self.sent_tokenizer(
                text,
                max_length=MAX_TEXT_LENGTH,
                padding=True,
                truncation=True,
                return_tensors="pt"
            )
            with torch.inference_mode():
                sent_out = self.sent_model(**sent_inputs)
                sent_probs = torch.softmax(sent_out.logits, dim=-1).float().cpu().numpy()[0].astype(np.float32)
            latencies["sentiment_ms"] = round((time.time() - t0) * 1000, 2)

            # 5. Emotion Probabilities (28-D)
            t0 = time.time()
            emo_inputs = self.emo_tokenizer(
                text,
                max_length=MAX_TEXT_LENGTH,
                padding=True,
                truncation=True,
                return_tensors="pt"
            )
            with torch.inference_mode():
                emo_out = self.emo_model(**emo_inputs)
                emo_probs = torch.sigmoid(emo_out.logits).float().cpu().numpy()[0].astype(np.float32)
            latencies["emotion_ms"] = round((time.time() - t0) * 1000, 2)

            # Canonical Concatenation: [768 + 512 + 1536 + 3 + 28 = 2847]
            canonical_vector = np.concatenate([
                feat_text,
                feat_clip,
                feat_vlm,
                sent_probs,
                emo_probs
            ], axis=0).astype(np.float32)

            if canonical_vector.shape != (EXPECTED_DIMENSION,):
                raise ValueError(f"Concatenated vector shape mismatch: expected ({EXPECTED_DIMENSION},), got {canonical_vector.shape}")
            if not np.all(np.isfinite(canonical_vector)):
                raise ValueError("Concatenated vector contains NaN or infinite values")

            # 6. Fusion Classifier Prediction
            t0 = time.time()
            fusion_result = self.fusion_engine.predict_vector(canonical_vector)
            latencies["fusion_ms"] = round((time.time() - t0) * 1000, 2)

            total_latency_ms = round((time.time() - start_total) * 1000, 2)
            latencies["total_ms"] = total_latency_ms

        # Build response payload
        sentiment_dict = {
            "negative": round(float(sent_probs[0]), 6),
            "neutral": round(float(sent_probs[1]), 6),
            "positive": round(float(sent_probs[2]), 6),
        }

        emotion_dict = {
            label: round(float(emo_probs[i]), 6)
            for i, label in enumerate(EMOTION_LABELS)
        }

        return {
            "status": "success",
            "prediction": {
                "label": fusion_result["label_name"],
                "class_id": fusion_result["predicted_label"],
                "probabilities": {
                    "non_sarcastic": fusion_result["prob_non_sarcastic"],
                    "sarcastic": fusion_result["prob_sarcastic"],
                },
                "confidence": fusion_result["confidence"],
            },
            "sentiment": sentiment_dict,
            "emotions": {
                "label_probabilities": emotion_dict,
                "top_emotions": sorted(
                    [{"emotion": k, "probability": v} for k, v in emotion_dict.items()],
                    key=lambda x: x["probability"],
                    reverse=True
                )[:5]
            },
            "model": {
                "fusion_model": "MultimodalFusionMLP",
                "feature_dimensions": EXPECTED_DIMENSION,
                "branches": {
                    "text_xlmr_dim": 768,
                    "clip_dim": 512,
                    "internvl_dim": 1536,
                    "sentiment_dim": 3,
                    "emotion_dim": 28,
                }
            },
            "performance": {
                "total_latency_ms": total_latency_ms,
                "latency_breakdown_ms": latencies,
            }
        }


# Singleton accessor
_manager = MultimodalInferenceManager()

def get_multimodal_manager() -> MultimodalInferenceManager:
    return _manager
