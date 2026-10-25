"""Trainable Classification Head for Multimodal Passive-Aggression Detection.

Consumes fused multimodal representations and computes calibrated class logits,
probabilities, predictions, and loss with abstention capabilities.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

PRIMARY_LABEL_TO_IDX: Dict[str, int] = {
    "passive_aggressive": 0,
    "non_passive_aggressive": 1,
    "ambiguous": 2,
}

PRIMARY_IDX_TO_LABEL: Dict[int, str] = {
    0: "passive_aggressive",
    1: "non_passive_aggressive",
    2: "ambiguous",
}

SARCASM_LABEL_TO_IDX: Dict[str, int] = {
    "sarcastic": 0,
    "non_sarcastic": 1,
    "ambiguous": 2,
}

SARCASM_IDX_TO_LABEL: Dict[int, str] = {
    0: "sarcastic",
    1: "non_sarcastic",
    2: "ambiguous",
}


class ClassificationHead(nn.Module):
    """
    Trainable feed-forward classification head:
    fused_features -> Linear -> LayerNorm -> Activation (GELU/ReLU) -> Dropout -> Linear(num_classes)
    """

    def __init__(
        self,
        in_dim: int = 512,
        hidden_dim: int = 256,
        num_classes: int = 3,
        dropout: float = 0.1,
        activation: str = "gelu",
        class_mapping: Optional[Dict[str, int]] = None,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.dropout_rate = dropout
        self.activation_name = activation.lower()
        self.class_mapping = class_mapping or PRIMARY_LABEL_TO_IDX
        self.idx_to_class = {v: k for k, v in self.class_mapping.items()}

        if self.activation_name == "gelu":
            act_fn = nn.GELU()
        elif self.activation_name == "relu":
            act_fn = nn.ReLU()
        else:
            raise ValueError(f"Unsupported activation '{activation}'. Choose 'gelu' or 'relu'.")

        self.projection = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            act_fn,
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, fused_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            fused_features: Tensor of shape [batch_size, in_dim]
        Returns:
            logits: Tensor of shape [batch_size, num_classes]
        """
        if not torch.is_tensor(fused_features):
            raise TypeError(f"Expected torch.Tensor, got {type(fused_features)}")

        if not torch.isfinite(fused_features).all():
            raise ValueError("Input features contain NaN or Inf values.")

        if fused_features.dim() != 2:
            raise ValueError(
                f"Expected 2D tensor [batch_size, in_dim], got shape {fused_features.shape}"
            )

        if fused_features.shape[-1] != self.in_dim:
            raise ValueError(
                f"Feature dimension mismatch: expected {self.in_dim}, got {fused_features.shape[-1]}"
            )

        logits = self.projection(fused_features)
        return logits

    def compute_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        class_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Computes Cross-Entropy loss with input validation.
        """
        if logits.dim() != 2 or logits.shape[-1] != self.num_classes:
            raise ValueError(f"Logits shape mismatch: expected [B, {self.num_classes}], got {logits.shape}")

        if targets.dim() != 1 or targets.shape[0] != logits.shape[0]:
            raise ValueError(f"Targets shape mismatch: expected [B={logits.shape[0]}], got {targets.shape}")

        if (targets < 0).any() or (targets >= self.num_classes).any():
            raise ValueError(f"Target labels out of bounds [0, {self.num_classes - 1}]. Got {targets}")

        return F.cross_entropy(logits, targets, weight=class_weights)

    def predict(
        self,
        logits: torch.Tensor,
        confidence_threshold: Optional[float] = None,
        abstention_class_idx: int = 2,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes predictions from logits using argmax or optional confidence thresholding.
        Returns:
            predictions: [batch_size] predicted class indices
            probabilities: [batch_size, num_classes] softmax probabilities
        """
        probs = F.softmax(logits, dim=-1)
        max_probs, preds = torch.max(probs, dim=-1)

        if confidence_threshold is not None:
            # If max predicted probability is below threshold, abstain to ambiguous
            low_conf_mask = max_probs < confidence_threshold
            preds[low_conf_mask] = abstention_class_idx

        return preds, probs

    def save_checkpoint(self, path: str, extra_metadata: Optional[Dict[str, Any]] = None) -> None:
        """Saves weights and architecture metadata."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        checkpoint = {
            "state_dict": self.state_dict(),
            "in_dim": self.in_dim,
            "hidden_dim": self.hidden_dim,
            "num_classes": self.num_classes,
            "dropout": self.dropout_rate,
            "activation": self.activation_name,
            "class_mapping": self.class_mapping,
            "extra_metadata": extra_metadata or {},
        }
        torch.save(checkpoint, path)

    @classmethod
    def load_checkpoint(cls, path: str, map_location: str = "cpu") -> ClassificationHead:
        """Loads weights and instantiates configured ClassificationHead."""
        checkpoint = torch.load(path, map_location=map_location, weights_only=False)
        head = cls(
            in_dim=checkpoint["in_dim"],
            hidden_dim=checkpoint["hidden_dim"],
            num_classes=checkpoint["num_classes"],
            dropout=checkpoint["dropout"],
            activation=checkpoint["activation"],
            class_mapping=checkpoint["class_mapping"],
        )
        head.load_state_dict(checkpoint["state_dict"])
        return head
