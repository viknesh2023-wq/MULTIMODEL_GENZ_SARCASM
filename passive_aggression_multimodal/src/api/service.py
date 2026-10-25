"""
src/api/service.py

Inference service layer managing checkpoint lifecycles, in-memory safe feature extraction,
fallback routing, and predictions.
"""

from __future__ import annotations

import io
import json
import logging
from pathlib import Path
import pickle
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image

from src.models.gated_multimodal_fusion import MultimodalFusionClassifier
from src.models.text_baseline import TextBaselineModel, pragmatic_text_cleaner
from src.models.text_encoder import PretrainedTextEncoder

logger = logging.getLogger("Phase9-InferenceService")

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = ROOT_DIR / "configs/phase9/api_config.json"

CLASS_NAMES = ["non_sarcastic", "sarcastic"]
PRIMARY_DISCLAIMER = (
    "Auxiliary Sarcasm Detection Only: Sarcasm predictions do not constitute passive-aggression ground truth. "
    "Operational definition of passive aggression requires covert interpersonal hostility or blame with plausible deniability."
)


class ModelInferenceService:
    """Manages loaded model lifecycles and executes stateless predictions."""

    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = config_path or CONFIG_PATH
        self.config = self._load_config()

        self.device = self._select_device()
        self.text_baseline: Optional[TextBaselineModel] = None
        self.text_encoder: Optional[PretrainedTextEncoder] = None
        self.multimodal_model: Optional[MultimodalFusionClassifier] = None
        self.pa_silver_model: Optional[Any] = None

        self.model_status: Dict[str, str] = {
            "text_tfidf_baseline": "unloaded",
            "gated_multimodal_fusion": "unloaded",
            "pa_silver_tfidf_v1": "unloaded",
        }
        self.loaded_models: List[str] = []

    def _load_config(self) -> Dict[str, Any]:
        if self.config_path.exists():
            with open(self.config_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {
            "device": "auto",
            "max_text_length": 5000,
            "max_image_size_bytes": 10485760,
            "allowed_image_mime_types": ["image/jpeg", "image/png", "image/webp"],
            "checkpoints": {
                "text_tfidf_baseline": "artifacts/phase8a/text_baseline/best_text_baseline.pkl",
                "gated_multimodal_fusion": "artifacts/phase8a/multimodal_fusion/best_multimodal_fusion.pt",
            },
        }

    def _select_device(self) -> torch.device:
        cfg_dev = self.config.get("device", "auto").lower()
        if cfg_dev == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        elif cfg_dev == "cpu":
            return torch.device("cpu")
        else:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def initialize_models(self) -> None:
        """Loads and verifies checkpoints once during startup."""
        logger.info("Initializing models on device: %s...", self.device)

        # 1. Load Text Baseline (TF-IDF + Logistic Regression)
        tb_rel = self.config.get("checkpoints", {}).get("text_tfidf_baseline")
        if tb_rel:
            tb_path = ROOT_DIR / tb_rel
            if tb_path.exists():
                try:
                    self.text_baseline = TextBaselineModel.load(tb_path)
                    # Verify smoke prediction
                    test_pred = self.text_baseline.predict(["Smoke test sentence."])
                    test_probs = self.text_baseline.predict_proba(["Smoke test sentence."])
                    if len(test_pred) == 1 and test_probs.shape == (1, 2):
                        self.model_status["text_tfidf_baseline"] = "ready"
                        self.loaded_models.append("text_tfidf_baseline")
                        logger.info("Loaded Text Baseline: %s", tb_rel)
                    else:
                        self.model_status["text_tfidf_baseline"] = "failed_smoke_test"
                except Exception as e:
                    logger.error("Failed to load Text Baseline: %s", e)
                    self.model_status["text_tfidf_baseline"] = f"error: {str(e)}"
            else:
                self.model_status["text_tfidf_baseline"] = f"checkpoint_missing: {tb_rel}"

        # 2. Load Multimodal Gated Fusion Model & Encoders
        mf_rel = self.config.get("checkpoints", {}).get("gated_multimodal_fusion")
        if mf_rel:
            mf_path = ROOT_DIR / mf_rel
            if mf_path.exists():
                try:
                    # Load text encoder
                    self.text_encoder = PretrainedTextEncoder(
                        "FacebookAI/xlm-roberta-base",
                        device=str(self.device),
                        local_files_only=True,
                    )
                    self.text_encoder.eval()

                    # Load fusion classifier
                    mf_ckpt = torch.load(mf_path, map_location=self.device)
                    self.multimodal_model = MultimodalFusionClassifier(
                        image_dim=768,
                        text_dim=768,
                        hidden_dim=512,
                        num_classes_primary=2,
                        num_classes_auxiliary=None,
                        dropout=0.1,
                    ).to(self.device)
                    self.multimodal_model.load_state_dict(mf_ckpt["model_state_dict"])
                    self.multimodal_model.eval()

                    # Smoke test multimodal forward pass
                    dummy_t = torch.zeros((1, 768), device=self.device)
                    dummy_v = torch.zeros((1, 768), device=self.device)
                    dummy_m = torch.zeros(1, dtype=torch.bool, device=self.device)
                    with torch.no_grad():
                        _ = self.multimodal_model(
                            comment_features=dummy_t,
                            image_features=dummy_v,
                            image_masks=dummy_m,
                        )

                    self.model_status["gated_multimodal_fusion"] = "ready"
                    self.loaded_models.append("gated_multimodal_fusion")
                    logger.info("Loaded Gated Multimodal Fusion Model: %s", mf_rel)
                except Exception as e:
                    logger.error("Failed to load Multimodal Model: %s", e)
                    self.model_status["gated_multimodal_fusion"] = f"error: {str(e)}"
            else:
                self.model_status["gated_multimodal_fusion"] = f"checkpoint_missing: {mf_rel}"

        # 3. Load Exploratory Silver Passive-Aggression Model (Phase 10)
        pa_path = ROOT_DIR / "artifacts/phase10/pa_silver_tfidf_v1.pkl"
        if pa_path.exists():
            try:
                with open(pa_path, "rb") as f:
                    self.pa_silver_model = pickle.load(f)
                self.model_status["pa_silver_tfidf_v1"] = "ready"
                self.loaded_models.append("pa_silver_tfidf_v1")
                logger.info("Loaded Exploratory Silver PA Baseline: %s", pa_path)
            except Exception as e:
                logger.warning("Could not load Silver PA Baseline: %s", e)
                self.model_status["pa_silver_tfidf_v1"] = f"error: {str(e)}"
        else:
            self.model_status["pa_silver_tfidf_v1"] = "checkpoint_missing"

    def predict_passive_aggression_experimental(self, text: str) -> Dict[str, Any]:
        """
        Executes exploratory passive aggression prediction using the silver-trained baseline.
        Explicitly documents silver label provenance and lack of independent human ground truth validation.
        """
        if self.pa_silver_model is None or self.model_status.get("pa_silver_tfidf_v1") != "ready":
            # Attempt lazy load
            pa_path = ROOT_DIR / "artifacts/phase10/pa_silver_tfidf_v1.pkl"
            if pa_path.exists():
                with open(pa_path, "rb") as f:
                    self.pa_silver_model = pickle.load(f)
                self.model_status["pa_silver_tfidf_v1"] = "ready"
            else:
                raise RuntimeError("Exploratory silver passive aggression model is not loaded or checkpoint missing.")

        t0 = time.perf_counter()
        probs = self.pa_silver_model.predict_proba([text])[0]
        pred_idx = int(np.argmax(probs))
        latency = (time.perf_counter() - t0) * 1000.0

        label_map = {0: "non_passive_aggressive", 1: "passive_aggressive"}

        return {
            "task": "passive_aggression_detection",
            "model_id": "pa_silver_tfidf_v1",
            "predicted_label": label_map.get(pred_idx, "ambiguous"),
            "class_scores": {
                "non_passive_aggressive": round(float(probs[0]), 4),
                "passive_aggressive": round(float(probs[1]), 4),
            },
            "label_provenance": "LLM_SILVER",
            "evaluation_status": "LLM_SILVER_DIAGNOSTIC_EVALUATION_AVAILABLE",
            "latency_ms": round(latency, 2),
            "warnings": [
                "EXPERIMENTAL MODEL: Trained exclusively on automated LLM silver labels.",
                "This model has NOT been validated against independent human ground truth.",
                "The primary human benchmark remains BLOCKED_PENDING_HUMAN_ANNOTATION.",
            ],
        }

    def predict_sarcasm_text(self, text: str) -> Dict[str, Any]:
        """Runs fast, robust text-only sarcasm prediction."""
        if self.text_baseline is None or self.model_status.get("text_tfidf_baseline") != "ready":
            raise RuntimeError("Text baseline model is not ready.")

        t0 = time.perf_counter()
        clean = pragmatic_text_cleaner(text)
        probs = self.text_baseline.predict_proba([clean])[0]
        pred_idx = int(np.argmax(probs))
        latency = (time.perf_counter() - t0) * 1000.0

        return {
            "task": "sarcasm_detection",
            "model_id": "text_tfidf_baseline",
            "predicted_label": CLASS_NAMES[pred_idx],
            "class_scores": {
                "non_sarcastic": round(float(probs[0]), 4),
                "sarcastic": round(float(probs[1]), 4),
            },
            "modality_used": ["text"],
            "fallback_applied": False,
            "model_version": "phase8a_tfidf_v1",
            "latency_ms": round(latency, 2),
            "warnings": [PRIMARY_DISCLAIMER],
        }

    def decode_and_extract_image_features(self, image_bytes: bytes) -> torch.Tensor:
        """
        Safely decodes an image in-memory and extracts 768-dim spatial grid features.
        Raises ValueError on corrupted or invalid image content.
        """
        # Enforce size ceiling
        max_bytes = self.config.get("max_image_size_bytes", 10485760)
        if len(image_bytes) > max_bytes:
            raise ValueError(f"Uploaded image exceeds size ceiling of {max_bytes} bytes.")

        try:
            # Prevent decompression bomb DOS attacks
            Image.MAX_IMAGE_PIXELS = 10_000_000
            with Image.open(io.BytesIO(image_bytes)) as img:
                # Verify image can actually be decoded
                img.verify()

            # Reopen for pixel extraction
            with Image.open(io.BytesIO(image_bytes)) as img:
                img_rgb = img.convert("RGB")
                img_resized = img_rgb.resize((224, 224))
                arr = np.asarray(img_resized, dtype=np.float32) / 255.0  # [224, 224, 3]

                # 16x16 grid cell statistics: 256 cells x 3 channels = 768 dimensions
                grid_cells = arr.reshape(16, 14, 16, 14, 3).mean(axis=(1, 3))
                v_emb = torch.from_numpy(grid_cells.flatten()).float()
                # L2 normalize
                norm = torch.norm(v_emb, p=2) + 1e-8
                v_emb = v_emb / norm

                if not torch.isfinite(v_emb).all():
                    raise ValueError("Non-finite values encountered in image feature extraction.")
                return v_emb.unsqueeze(0)  # [1, 768]
        except Exception as e:
            raise ValueError(f"Invalid, unsupported, or corrupted image payload: {e}")

    def predict_sarcasm_multimodal(
        self,
        text: str,
        image_bytes: Optional[bytes] = None,
        fallback_on_invalid_image: bool = False,
    ) -> Dict[str, Any]:
        """
        Executes multimodal sarcasm prediction with explicit missing-modality fallback.
        """
        t0 = time.perf_counter()
        warnings: List[str] = [PRIMARY_DISCLAIMER]

        # Case 1: Image is absent -> Explicit text fallback
        if not image_bytes or len(image_bytes) == 0:
            res = self.predict_sarcasm_text(text)
            res["fallback_applied"] = True
            res["warnings"].append("No image provided; executed text-only baseline fallback.")
            return res

        # Case 2: Attempt image feature extraction
        v_feat = None
        try:
            v_feat = self.decode_and_extract_image_features(image_bytes)
        except ValueError as val_err:
            if fallback_on_invalid_image:
                logger.warning("Image decoding failed (%s). Falling back to text baseline.", val_err)
                res = self.predict_sarcasm_text(text)
                res["fallback_applied"] = True
                res["warnings"].append(
                    f"Image invalid or unreadable ({str(val_err)}); routed through text-only fallback."
                )
                return res
            else:
                # Structured error requested
                raise

        # Case 3: Multimodal model check
        if (
            self.multimodal_model is None
            or self.text_encoder is None
            or self.model_status.get("gated_multimodal_fusion") != "ready"
        ):
            # Fall back to text baseline if multimodal pipeline is unavailable
            res = self.predict_sarcasm_text(text)
            res["fallback_applied"] = True
            res["warnings"].append(
                "Multimodal model checkpoint unavailable; executed text-only fallback."
            )
            return res

        # Case 4: Full Multimodal Prediction
        clean_text = pragmatic_text_cleaner(text)
        with torch.no_grad():
            t_feat = self.text_encoder([clean_text]).to(self.device)
            v_feat = v_feat.to(self.device)
            v_mask = torch.tensor([True], dtype=torch.bool, device=self.device)

            out = self.multimodal_model(
                comment_features=t_feat,
                image_features=v_feat,
                image_masks=v_mask,
            )
            logits = out["primary_logits"]
            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
            pred_idx = int(np.argmax(probs))

        latency = (time.perf_counter() - t0) * 1000.0

        return {
            "task": "sarcasm_detection",
            "model_id": "gated_multimodal_fusion",
            "predicted_label": CLASS_NAMES[pred_idx],
            "class_scores": {
                "non_sarcastic": round(float(probs[0]), 4),
                "sarcastic": round(float(probs[1]), 4),
            },
            "modality_used": ["text", "image"],
            "fallback_applied": False,
            "model_version": "phase8a_gated_fusion_v1",
            "latency_ms": round(latency, 2),
            "warnings": warnings,
        }
