"""Vision Feature Extraction Interface for Phase 3.

Provides unified feature extraction from images, interfacing with real pretrained CLIP
checkpoints or deterministic interface projections when checkpoints are unavailable locally,
with explicit missing-modality masks and flags.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
from PIL import Image


class PretrainedVisionEncoder(nn.Module):
    """
    Vision feature extractor producing fixed 768-dimensional visual scene representations.
    Supports real pretrained CLIP when cached, or deterministic patch projections with
    explicit missing-modality flags when offline.
    """

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        output_dim: int = 768,
        device: str = "cpu",
        local_files_only: bool = True,
    ):
        super().__init__()
        self.model_name = model_name
        self.output_dim = output_dim
        self.device = torch.device(device)
        self.is_real_pretrained = False
        self.clip_model = None
        self.clip_processor = None

        # Attempt to load real pretrained CLIP
        try:
            from transformers import CLIPModel, CLIPProcessor
            self.clip_processor = CLIPProcessor.from_pretrained(
                model_name, local_files_only=local_files_only
            )
            self.clip_model = CLIPModel.from_pretrained(
                model_name, local_files_only=local_files_only
            )
            self.clip_model.to(self.device)
            self.clip_model.eval()
            for p in self.clip_model.parameters():
                p.requires_grad = False
            self.is_real_pretrained = True
        except Exception:
            # Deterministic projection fallback for offline execution
            self.is_real_pretrained = False

    def forward(
        self, image_paths: List[Optional[Union[str, Path]]]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extracts visual embeddings and missing-modality masks.
        Args:
            image_paths: List of absolute file paths to images (or None if missing).
        Returns:
            embeddings: Tensor of shape [batch_size, output_dim]
            masks: BoolTensor of shape [batch_size] (True if image exists and was extracted)
        """
        batch_size = len(image_paths)
        embeddings = torch.zeros((batch_size, self.output_dim), dtype=torch.float32)
        masks = torch.zeros(batch_size, dtype=torch.bool)

        if batch_size == 0:
            return embeddings, masks

        for i, path in enumerate(image_paths):
            if path is None or not os.path.exists(str(path)):
                continue

            try:
                with Image.open(str(path)) as img:
                    img = img.convert("RGB")

                    if self.is_real_pretrained and self.clip_model is not None and self.clip_processor is not None:
                        inputs = self.clip_processor(images=img, return_tensors="pt")
                        inputs = {k: v.to(self.device) for k, v in inputs.items()}
                        with torch.no_grad():
                            img_features = self.clip_model.get_image_features(**inputs)
                            # Project or normalize to output_dim
                            if img_features.shape[-1] != self.output_dim:
                                img_features = nn.functional.adaptive_avg_pool1d(
                                    img_features.unsqueeze(1), self.output_dim
                                ).squeeze(1)
                            emb = img_features.squeeze(0).cpu()
                    else:
                        # Deterministic image representation from visual content (offline mode)
                        img_resized = img.resize((224, 224))
                        arr = np.asarray(img_resized, dtype=np.float32) / 255.0  # [224, 224, 3]
                        # Compute spatial grid statistics (16x16 grid = 256 cells x 3 channels = 768 dims)
                        grid_cells = arr.reshape(16, 14, 16, 14, 3).mean(axis=(1, 3))  # [16, 16, 3]
                        emb = torch.from_numpy(grid_cells.flatten()).float()  # 768 dims
                        # L2 normalize
                        emb = emb / (torch.norm(emb, p=2) + 1e-8)

                    if torch.isfinite(emb).all():
                        embeddings[i] = emb
                        masks[i] = True
            except Exception:
                # Corrupt or unreadable image: remains masked zero-vector
                continue

        return embeddings, masks
