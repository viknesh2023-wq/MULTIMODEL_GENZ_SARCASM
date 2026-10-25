"""Context-Aware Multimodal Fusion Model (Phase 3 Task H).

Explicitly models conversational interaction dynamics:
1. Target comment features
2. Dialogue history / Post caption context features
3. Visual scene features
4. OCR textual evidence

Applies context-comment cross-attention gating before multimodal fusion, preserving
the distinct identity and temporal precedence of the target comment.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.classification_head import ClassificationHead, PRIMARY_LABEL_TO_IDX
from src.preprocessing.fusion_interface import ModalityProjection


class ContextAwareMultimodalFusion(nn.Module):
    """
    Context-Aware Multimodal Architecture:
    - Context-Comment Cross-Attention Gating: Learns conversational incongruity
    - Image-Comment Fusion: Projects visual scene and comment to shared space
    - OCR Integration: Injects text-in-image tokens
    - ClassificationHead: 3-class primary output logits
    """

    def __init__(
        self,
        image_dim: int = 768,
        text_dim: int = 768,
        hidden_dim: int = 512,
        num_classes: int = 3,
        num_attention_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Projections to shared hidden space
        self.comment_proj = ModalityProjection(text_dim, hidden_dim, dropout)
        self.context_proj = ModalityProjection(text_dim, hidden_dim, dropout)
        self.image_proj = ModalityProjection(image_dim, hidden_dim, dropout)
        self.ocr_proj = ModalityProjection(text_dim, hidden_dim, dropout)

        # Multihead cross-attention: Target Comment (Query) attending to Context History (Key, Value)
        self.comment_context_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_attention_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Context gating layer: decides how strongly conversational context modulates the comment
        self.context_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid(),
        )

        # Multimodal fusion network: combines context-modulated comment + image + OCR
        self.multimodal_gate = nn.Sequential(
            nn.Linear(hidden_dim * 3, 3),
            nn.Softmax(dim=-1),
        )

        self.fusion_linear = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Trainable 3-class classification head
        self.classification_head = ClassificationHead(
            in_dim=hidden_dim,
            hidden_dim=256,
            num_classes=num_classes,
            dropout=dropout,
            class_mapping=PRIMARY_LABEL_TO_IDX,
        )

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
        batch_size = comment_features.shape[0]
        device = comment_features.device

        # 1. Project target comment
        h_comment = self.comment_proj(comment_features)  # [B, H]

        # 2. Project context
        if context_features is None:
            context_features = torch.zeros((batch_size, 768), device=device)
            context_masks = torch.zeros(batch_size, dtype=torch.bool, device=device)
        elif context_masks is None:
            context_masks = torch.ones(batch_size, dtype=torch.bool, device=device)
        h_context = self.context_proj(context_features, context_masks)  # [B, H]

        # Cross-attention: Comment as query, context as key/value
        q = h_comment.unsqueeze(1)  # [B, 1, H]
        k = h_context.unsqueeze(1)  # [B, 1, H]
        v = h_context.unsqueeze(1)  # [B, 1, H]
        attn_out, attn_weights = self.comment_context_attention(q, k, v)
        attn_out = attn_out.squeeze(1)  # [B, H]

        # Context gate: modulate comment representation based on conversational incongruity
        gate = self.context_gate(torch.cat([h_comment, attn_out], dim=-1))
        # Mask out context influence if context is unavailable
        gate = gate * context_masks.unsqueeze(-1).float()
        h_modulated_comment = h_comment + gate * attn_out

        # 3. Vision modality
        if image_features is None:
            image_features = torch.zeros((batch_size, 768), device=device)
            image_masks = torch.zeros(batch_size, dtype=torch.bool, device=device)
        elif image_masks is None:
            image_masks = torch.ones(batch_size, dtype=torch.bool, device=device)
        h_image = self.image_proj(image_features, image_masks)

        # 4. OCR text modality
        if ocr_features is None:
            ocr_features = torch.zeros((batch_size, 768), device=device)
            ocr_masks = torch.zeros(batch_size, dtype=torch.bool, device=device)
        elif ocr_masks is None:
            ocr_masks = torch.ones(batch_size, dtype=torch.bool, device=device)
        h_ocr = self.ocr_proj(ocr_features, ocr_masks)

        # 5. Multimodal Gating across (Modulated Comment, Image, OCR)
        fused_stack = torch.cat([h_modulated_comment, h_image, h_ocr], dim=-1)  # [B, H*3]
        weights = self.multimodal_gate(fused_stack)  # [B, 3]

        w_c = weights[:, 0].unsqueeze(-1)
        w_i = weights[:, 1].unsqueeze(-1)
        w_o = weights[:, 2].unsqueeze(-1)
        weighted_stack = torch.cat([h_modulated_comment * w_c, h_image * w_i, h_ocr * w_o], dim=-1)

        fused_rep = self.fusion_linear(weighted_stack)  # [B, H]
        primary_logits = self.classification_head(fused_rep)  # [B, 3]

        return {
            "primary_logits": primary_logits,
            "fused_rep": fused_rep,
            "multimodal_weights": weights,
            "context_gate": gate,
        }

    def compute_loss(
        self,
        outputs: Union[Dict[str, torch.Tensor], torch.Tensor],
        targets: torch.Tensor,
        class_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        logits = outputs["primary_logits"] if isinstance(outputs, dict) else outputs
        return self.classification_head.compute_loss(logits, targets, class_weights=class_weights)
