"""Pretrained feature extraction interfaces for vision and language encoders.

Supports:
- Vision Encoder: CLIP / SigLIP architecture interface (output_dim=768)
- Text Encoder: DeBERTa / RoBERTa / XLM-R architecture interface (output_dim=768)
- Missing modality masking (zero vector + boolean mask flag)
- Record ID alignment and batch processing
- Safe offline validation fixture mode without unauthorized large model downloads
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from src.preprocessing.image_preprocessor import ImagePreprocessor


class VisionFeatureExtractor:
    """Pretrained vision feature extraction interface for CLIP/SigLIP."""

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        output_dim: int = 768,
        device: str = "cpu",
        offline_test_mode: bool = False,
    ):
        self.model_name = model_name
        self.output_dim = output_dim
        self.device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
        self.offline_test_mode = offline_test_mode
        self.preprocessor = ImagePreprocessor(target_size=(224, 224))
        self.is_real_model_loaded = False
        self.model = None

        if not self.offline_test_mode:
            try:
                from transformers import CLIPVisionModelWithProjection
                # Attempt to load from local cache only without downloading
                self.model = CLIPVisionModelWithProjection.from_pretrained(
                    self.model_name, local_files_only=True
                ).to(self.device)
                self.model.eval()
                self.is_real_model_loaded = True
            except Exception:
                # Local weights not cached; operate in structured interface mode
                self.is_real_model_loaded = False

    def extract_batch(self, image_paths: List[Optional[str]]) -> Dict[str, Any]:
        """
        Extracts vision features for a batch of image paths.
        Handles missing images by populating zero vectors with modality_mask=False.
        """
        batch_size = len(image_paths)
        embeddings = torch.zeros((batch_size, self.output_dim), dtype=torch.float32)
        modality_mask = torch.zeros(batch_size, dtype=torch.bool)
        valid_indices = []
        valid_tensors = []

        for idx, p in enumerate(image_paths):
            if p and os.path.isfile(p):
                try:
                    tensor = self.preprocessor.preprocess_to_tensor(p)
                    valid_tensors.append(tensor)
                    valid_indices.append(idx)
                    modality_mask[idx] = True
                except Exception:
                    modality_mask[idx] = False

        if valid_tensors:
            batch_tensor = torch.stack(valid_tensors).to(self.device)
            if self.is_real_model_loaded and self.model is not None:
                with torch.no_grad():
                    outputs = self.model(pixel_values=batch_tensor)
                    feats = outputs.image_embeds if hasattr(outputs, "image_embeds") else outputs.pooler_output
                    for i, orig_idx in enumerate(valid_indices):
                        embeddings[orig_idx] = feats[i].cpu()
            else:
                # Deterministic normalized visual feature projection from pixel statistics for interface verification
                with torch.no_grad():
                    # Compute spatial pooling over channels [C, H, W] -> [C * 4] -> projected to output_dim
                    pooled = torch.mean(batch_tensor, dim=(-2, -1))  # [B, 3]
                    std_pooled = torch.std(batch_tensor, dim=(-2, -1))  # [B, 3]
                    base_stats = torch.cat([pooled, std_pooled], dim=-1)  # [B, 6]
                    # Deterministic linear projection matrix (fixed seed)
                    gen = torch.Generator().manual_seed(42)
                    proj = torch.randn(6, self.output_dim, generator=gen, device=batch_tensor.device)
                    projected = torch.matmul(base_stats, proj)
                    # L2 normalize
                    projected = torch.nn.functional.normalize(projected, p=2, dim=-1)
                    for i, orig_idx in enumerate(valid_indices):
                        embeddings[orig_idx] = projected[i].cpu()

        return {
            "embeddings": embeddings,
            "modality_mask": modality_mask,
            "output_dim": self.output_dim,
            "model_name": self.model_name,
            "execution_status": "REAL_PRETRAINED" if self.is_real_model_loaded else "INTERFACE_STUB_PENDING_WEIGHTS",
        }


class TextFeatureExtractor:
    """Pretrained text feature extraction interface for DeBERTa / RoBERTa."""

    def __init__(
        self,
        model_name: str = "microsoft/deberta-v3-base",
        output_dim: int = 768,
        device: str = "cpu",
        offline_test_mode: bool = False,
    ):
        self.model_name = model_name
        self.output_dim = output_dim
        self.device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
        self.offline_test_mode = offline_test_mode
        self.is_real_model_loaded = False
        self.tokenizer = None
        self.model = None

        if not self.offline_test_mode:
            try:
                from transformers import AutoModel, AutoTokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, local_files_only=True)
                self.model = AutoModel.from_pretrained(self.model_name, local_files_only=True).to(self.device)
                self.model.eval()
                self.is_real_model_loaded = True
            except Exception:
                self.is_real_model_loaded = False

    def extract_batch(self, texts: List[str]) -> Dict[str, Any]:
        """Extracts text embeddings for a batch of strings."""
        batch_size = len(texts)
        embeddings = torch.zeros((batch_size, self.output_dim), dtype=torch.float32)

        if self.is_real_model_loaded and self.model is not None and self.tokenizer is not None:
            with torch.no_grad():
                encoded = self.tokenizer(
                    texts, padding=True, truncation=True, max_length=128, return_tensors="pt"
                ).to(self.device)
                outputs = self.model(**encoded)
                # Mean pooling
                mask = encoded["attention_mask"].unsqueeze(-1)
                pooled = torch.sum(outputs.last_hidden_state * mask, dim=1) / torch.clamp(mask.sum(dim=1), min=1e-9)
                embeddings = pooled.cpu()
        else:
            # Deterministic character n-gram hashing projection for interface test validation
            for idx, txt in enumerate(texts):
                words = txt.split()
                vec = torch.zeros(self.output_dim)
                for w in words:
                    h = int(torch.tensor(list(w.encode("utf-8")), dtype=torch.uint8).sum()) % self.output_dim
                    vec[h] += 1.0
                norm_vec = torch.nn.functional.normalize(vec, p=2, dim=0)
                embeddings[idx] = norm_vec

        return {
            "embeddings": embeddings,
            "output_dim": self.output_dim,
            "model_name": self.model_name,
            "execution_status": "REAL_PRETRAINED" if self.is_real_model_loaded else "INTERFACE_STUB_PENDING_WEIGHTS",
        }
