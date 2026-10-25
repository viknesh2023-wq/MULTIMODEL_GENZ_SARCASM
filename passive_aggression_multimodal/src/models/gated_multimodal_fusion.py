"""Gated Multimodal Fusion Model with Trainable Classification Heads (Phase 3 Task G).

Integrates visual, textual, dialogue context, and OCR modalities via learned
adaptive gating and attaches primary and auxiliary classification heads.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.classification_head import (
    ClassificationHead,
    PRIMARY_LABEL_TO_IDX,
    SARCASM_LABEL_TO_IDX,
)
from src.preprocessing.fusion_interface import GatedMultimodalFusion


class MultimodalFusionClassifier(nn.Module):
    """
    End-to-End Multimodal Fusion Model:
    (Comment, Image, Context, OCR) -> GatedMultimodalFusion (512-dim) -> ClassificationHead (3 classes)
    """

    def __init__(
        self,
        image_dim: int = 768,
        text_dim: int = 768,
        hidden_dim: int = 512,
        num_classes_primary: int = 3,
        num_classes_auxiliary: Optional[int] = 3,
        dropout: float = 0.1,
        class_mapping_primary: Optional[Dict[str, int]] = None,
        class_mapping_auxiliary: Optional[Dict[str, int]] = None,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.has_auxiliary = num_classes_auxiliary is not None

        # Gated fusion core
        self.fusion = GatedMultimodalFusion(
            image_dim=image_dim,
            text_dim=text_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

        # Primary Passive-Aggression classification head
        self.primary_head = ClassificationHead(
            in_dim=hidden_dim,
            hidden_dim=256,
            num_classes=num_classes_primary,
            dropout=dropout,
            class_mapping=class_mapping_primary or PRIMARY_LABEL_TO_IDX,
        )

        # Optional Auxiliary Sarcasm classification head
        if self.has_auxiliary:
            self.auxiliary_head = ClassificationHead(
                in_dim=hidden_dim,
                hidden_dim=256,
                num_classes=num_classes_auxiliary,
                dropout=dropout,
                class_mapping=class_mapping_auxiliary or SARCASM_LABEL_TO_IDX,
            )
        else:
            self.auxiliary_head = None

    def forward(
        self,
        comment_features: torch.Tensor,
        image_features: Optional[torch.Tensor] = None,
        context_features: Optional[torch.Tensor] = None,
        ocr_features: Optional[torch.Tensor] = None,
        image_masks: Optional[torch.Tensor] = None,
        context_masks: Optional[torch.Tensor] = None,
        ocr_masks: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through gating and task heads.
        Returns:
            dict containing:
            - 'primary_logits': [B, 3]
            - 'auxiliary_logits': [B, 3] (if auxiliary enabled)
            - 'gate_weights': [B, 4]
            - 'fused_rep': [B, hidden_dim]
        """
        fused_rep, gate_weights = self.fusion(
            comment_embeds=comment_features,
            image_embeds=image_features,
            context_embeds=context_features,
            ocr_embeds=ocr_features,
            image_mask=image_masks,
            context_mask=context_masks,
            ocr_mask=ocr_masks,
        )

        primary_logits = self.primary_head(fused_rep)
        out = {
            "primary_logits": primary_logits,
            "gate_weights": gate_weights,
            "fused_rep": fused_rep,
        }

        if self.has_auxiliary and self.auxiliary_head is not None:
            out["auxiliary_logits"] = self.auxiliary_head(fused_rep)

        return out

    def compute_loss(
        self,
        outputs: Union[Dict[str, torch.Tensor], torch.Tensor],
        targets: torch.Tensor,
        class_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Computes loss on primary logits."""
        logits = outputs["primary_logits"] if isinstance(outputs, dict) else outputs
        return self.primary_head.compute_loss(logits, targets, class_weights=class_weights)

    def predict(
        self,
        comment_features: torch.Tensor,
        image_features: Optional[torch.Tensor] = None,
        context_features: Optional[torch.Tensor] = None,
        ocr_features: Optional[torch.Tensor] = None,
        image_masks: Optional[torch.Tensor] = None,
        context_masks: Optional[torch.Tensor] = None,
        ocr_masks: Optional[torch.Tensor] = None,
        confidence_threshold: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns (predictions, probabilities, gate_weights).
        """
        out = self.forward(
            comment_features=comment_features,
            image_features=image_features,
            context_features=context_features,
            ocr_features=ocr_features,
            image_masks=image_masks,
            context_masks=context_masks,
            ocr_masks=ocr_masks,
        )
        preds, probs = self.primary_head.predict(
            out["primary_logits"], confidence_threshold=confidence_threshold
        )
        return preds, probs, out["gate_weights"]
