"""Feature Store for Caching Multimodal Pretrained Embeddings.

Provides indexed storage, retrieval, and serialization for:
- Text embeddings (comment, caption, OCR, dialogue context)
- Vision embeddings (image scene)
- Modality availability boolean masks
- Strict record-to-embedding ID alignment
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import torch


@dataclass
class MultimodalBatchFeatures:
    """Container for batch multimodal representations and masks."""
    example_ids: List[str]
    comment_features: torch.Tensor       # [B, D_text]
    image_features: torch.Tensor         # [B, D_image]
    context_features: torch.Tensor       # [B, D_text]
    ocr_features: torch.Tensor           # [B, D_text]
    image_masks: torch.Tensor            # [B] bool
    context_masks: torch.Tensor          # [B] bool
    ocr_masks: torch.Tensor              # [B] bool
    labels: Optional[torch.Tensor] = None # [B] int (optional targets)

    def to(self, device: Union[str, torch.device]) -> MultimodalBatchFeatures:
        """Move all tensors to device."""
        device = torch.device(device)
        return MultimodalBatchFeatures(
            example_ids=self.example_ids,
            comment_features=self.comment_features.to(device),
            image_features=self.image_features.to(device),
            context_features=self.context_features.to(device),
            ocr_features=self.ocr_features.to(device),
            image_masks=self.image_masks.to(device),
            context_masks=self.context_masks.to(device),
            ocr_masks=self.ocr_masks.to(device),
            labels=self.labels.to(device) if self.labels is not None else None,
        )


class FeatureStore:
    """
    On-disk and in-memory cache for extracted multimodal features.
    Guarantees strict 1-to-1 ordering and alignment with record IDs.
    """

    def __init__(self, storage_dir: Union[str, Path] = "data/features/phase3"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.registry: Dict[str, Dict[str, Any]] = {}

    def save_features(
        self,
        name: str,
        example_ids: List[str],
        comment_features: torch.Tensor,
        image_features: torch.Tensor,
        context_features: torch.Tensor,
        ocr_features: torch.Tensor,
        image_masks: torch.Tensor,
        context_masks: torch.Tensor,
        ocr_masks: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """Serializes feature tensors to a single checkpoint."""
        # Shape and alignment validations
        n = len(example_ids)
        assert comment_features.shape[0] == n, f"Comment features count {comment_features.shape[0]} != {n}"
        assert image_features.shape[0] == n, f"Image features count {image_features.shape[0]} != {n}"
        assert image_masks.shape[0] == n, f"Image masks count {image_masks.shape[0]} != {n}"

        out_path = self.storage_dir / f"{name}.pt"
        payload = {
            "example_ids": example_ids,
            "comment_features": comment_features.cpu(),
            "image_features": image_features.cpu(),
            "context_features": context_features.cpu(),
            "ocr_features": ocr_features.cpu(),
            "image_masks": image_masks.cpu(),
            "context_masks": context_masks.cpu(),
            "ocr_masks": ocr_masks.cpu(),
            "labels": labels.cpu() if labels is not None else None,
            "metadata": metadata or {},
        }
        torch.save(payload, out_path)
        return out_path

    def load_features(self, name: str) -> MultimodalBatchFeatures:
        """Loads cached features by name."""
        file_path = self.storage_dir / f"{name}.pt"
        if not file_path.exists():
            raise FileNotFoundError(f"Feature file not found: {file_path}")

        payload = torch.load(file_path, map_location="cpu", weights_only=False)
        return MultimodalBatchFeatures(
            example_ids=payload["example_ids"],
            comment_features=payload["comment_features"],
            image_features=payload["image_features"],
            context_features=payload["context_features"],
            ocr_features=payload["ocr_features"],
            image_masks=payload["image_masks"],
            context_masks=payload["context_masks"],
            ocr_masks=payload["ocr_masks"],
            labels=payload.get("labels"),
        )


class MultimodalDataset(torch.utils.data.Dataset):
    """PyTorch Dataset wrapper around indexed MultimodalBatchFeatures."""

    def __init__(self, features: MultimodalBatchFeatures, indices: Optional[List[int]] = None):
        self.features = features
        self.indices = indices if indices is not None else list(range(len(features.example_ids)))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int) -> Dict[str, Any]:
        idx = self.indices[i]
        return {
            "example_id": self.features.example_ids[idx],
            "comment_features": self.features.comment_features[idx],
            "image_features": self.features.image_features[idx],
            "context_features": self.features.context_features[idx],
            "ocr_features": self.features.ocr_features[idx],
            "image_masks": self.features.image_masks[idx],
            "context_masks": self.features.context_masks[idx],
            "ocr_masks": self.features.ocr_masks[idx],
            "labels": self.features.labels[idx] if self.features.labels is not None else torch.tensor(-1),
        }

    @staticmethod
    def collate_fn(batch: List[Dict[str, Any]]) -> MultimodalBatchFeatures:
        """Collates individual item dicts into a MultimodalBatchFeatures container."""
        return MultimodalBatchFeatures(
            example_ids=[item["example_id"] for item in batch],
            comment_features=torch.stack([item["comment_features"] for item in batch]),
            image_features=torch.stack([item["image_features"] for item in batch]),
            context_features=torch.stack([item["context_features"] for item in batch]),
            ocr_features=torch.stack([item["ocr_features"] for item in batch]),
            image_masks=torch.stack([item["image_masks"] for item in batch]),
            context_masks=torch.stack([item["context_masks"] for item in batch]),
            ocr_masks=torch.stack([item["ocr_masks"] for item in batch]),
            labels=torch.stack([item["labels"] for item in batch]),
        )
