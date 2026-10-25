"""Multimodal fusion architecture interface for passive-aggression detection.

Connects:
- Visual features (CLIP/SigLIP: 768-dim)
- Comment / caption text features (DeBERTa/RoBERTa: 768-dim)
- Conversation context features (DeBERTa/RoBERTa: 768-dim)
- Extracted image OCR text features (DeBERTa/RoBERTa: 768-dim)

Implements:
1. Gated multimodal projection to shared representation space.
2. Modality availability masking (handling missing images, OCR, or threads).
3. Early fusion (concatenation baseline) & Cross-attention fusion interfaces.
4. Multitask classification heads:
   - Primary: Passive-Aggression (3 classes: passive_aggressive, non_passive_aggressive, ambiguous)
   - Auxiliary: Sarcasm (3 classes: sarcastic, non_sarcastic, ambiguous)
   - Subtype: Pragmatic Subtypes (7 classes)
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


PRIMARY_LABELS = ["passive_aggressive", "non_passive_aggressive", "ambiguous"]
SARCASM_LABELS = ["sarcastic", "non_sarcastic", "ambiguous"]
SUBTYPE_LABELS = [
    "fake_politeness",
    "indirect_criticism",
    "backhanded_compliment",
    "implied_blame",
    "sarcastic_agreement",
    "dismissiveness",
    "other_or_unclear",
]


class ModalityProjection(nn.Module):
    """Projects modality embeddings into a shared dimension with normalization."""

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x: [B, in_dim]
        mask: [B] (bool tensor: True if modality is present)
        """
        out = self.proj(x)
        if mask is not None:
            # Zero out absent modalities
            mask_expanded = mask.unsqueeze(-1).to(out.dtype)
            out = out * mask_expanded
        return out


class GatedMultimodalFusion(nn.Module):
    """
    Multimodal Fusion module that dynamically combines:
    - Target comment
    - Visual scene
    - Conversation context
    - Embedded image OCR text
    """

    def __init__(
        self,
        image_dim: int = 768,
        text_dim: int = 768,
        hidden_dim: int = 512,
        fusion_type: str = "concat_gated",
        dropout: float = 0.1,
    ):
        super().__init__()
        self.fusion_type = fusion_type
        self.hidden_dim = hidden_dim

        # Projections to shared hidden_dim
        self.comment_proj = ModalityProjection(text_dim, hidden_dim, dropout)
        self.image_proj = ModalityProjection(image_dim, hidden_dim, dropout)
        self.context_proj = ModalityProjection(text_dim, hidden_dim, dropout)
        self.ocr_proj = ModalityProjection(text_dim, hidden_dim, dropout)

        # Modality gating weights: learned gate deciding contribution of each modality
        self.gate_network = nn.Sequential(
            nn.Linear(hidden_dim * 4, 4),
            nn.Softmax(dim=-1),
        )

        # Fused projection
        self.fusion_linear = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        comment_embeds: torch.Tensor,
        image_embeds: Optional[torch.Tensor] = None,
        context_embeds: Optional[torch.Tensor] = None,
        ocr_embeds: Optional[torch.Tensor] = None,
        image_mask: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        ocr_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            fused_rep: [B, hidden_dim]
            gate_weights: [B, 4]
        """
        batch_size = comment_embeds.shape[0]
        device = comment_embeds.device

        # Comment is the primary target modality and always present
        h_comment = self.comment_proj(comment_embeds)

        # Vision modality
        if image_embeds is None:
            image_embeds = torch.zeros((batch_size, 768), device=device)
            image_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)
        elif image_mask is None:
            image_mask = torch.ones(batch_size, dtype=torch.bool, device=device)
        h_image = self.image_proj(image_embeds, image_mask)

        # Conversation Context modality
        if context_embeds is None:
            context_embeds = torch.zeros((batch_size, 768), device=device)
            context_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)
        elif context_mask is None:
            context_mask = torch.ones(batch_size, dtype=torch.bool, device=device)
        h_context = self.context_proj(context_embeds, context_mask)

        # OCR text modality
        if ocr_embeds is None:
            ocr_embeds = torch.zeros((batch_size, 768), device=device)
            ocr_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)
        elif ocr_mask is None:
            ocr_mask = torch.ones(batch_size, dtype=torch.bool, device=device)
        h_ocr = self.ocr_proj(ocr_embeds, ocr_mask)

        # Concatenate 4 representations
        stacked = torch.cat([h_comment, h_image, h_context, h_ocr], dim=-1)  # [B, hidden_dim * 4]

        # Calculate dynamic modality attention gates
        raw_gates = self.gate_network(stacked)  # [B, 4]

        # Zero-out gate attention for absent modalities and renormalize
        availability = torch.stack(
            [
                torch.ones(batch_size, dtype=torch.float32, device=device),
                image_mask.float(),
                context_mask.float(),
                ocr_mask.float(),
            ],
            dim=-1,
        )
        masked_gates = raw_gates * availability
        gate_weights = masked_gates / torch.clamp(masked_gates.sum(dim=-1, keepdim=True), min=1e-8)

        # Modality re-weighting
        gated_comment = h_comment * gate_weights[:, 0:1]
        gated_image = h_image * gate_weights[:, 1:2]
        gated_context = h_context * gate_weights[:, 2:3]
        gated_ocr = h_ocr * gate_weights[:, 3:4]

        fused_combined = torch.cat([gated_comment, gated_image, gated_context, gated_ocr], dim=-1)
        fused_rep = self.fusion_linear(fused_combined)

        return fused_rep, gate_weights


class PassiveAggressionMultimodalModel(nn.Module):
    """
    Multitask Multimodal Architecture for Passive-Aggression Detection.
    Connects multimodal fusion with primary classification and auxiliary sarcasm head.
    """

    def __init__(
        self,
        image_dim: int = 768,
        text_dim: int = 768,
        hidden_dim: int = 512,
        num_pa_classes: int = len(PRIMARY_LABELS),
        num_sarcasm_classes: int = len(SARCASM_LABELS),
        num_subtype_classes: int = len(SUBTYPE_LABELS),
        dropout: float = 0.2,
    ):
        super().__init__()
        self.fusion = GatedMultimodalFusion(
            image_dim=image_dim,
            text_dim=text_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

        # Primary Head: Passive Aggression (3-way)
        self.pa_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_pa_classes),
        )

        # Auxiliary Head: Sarcasm (3-way)
        self.sarcasm_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_sarcasm_classes),
        )

        # Subtype Head: Pragmatic Subtypes (7-way)
        self.subtype_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_subtype_classes),
        )

    def forward(
        self,
        comment_embeds: torch.Tensor,
        image_embeds: Optional[torch.Tensor] = None,
        context_embeds: Optional[torch.Tensor] = None,
        ocr_embeds: Optional[torch.Tensor] = None,
        image_mask: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        ocr_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        fused_rep, gate_weights = self.fusion(
            comment_embeds=comment_embeds,
            image_embeds=image_embeds,
            context_embeds=context_embeds,
            ocr_embeds=ocr_embeds,
            image_mask=image_mask,
            context_mask=context_mask,
            ocr_mask=ocr_mask,
        )

        pa_logits = self.pa_classifier(fused_rep)
        sarcasm_logits = self.sarcasm_classifier(fused_rep)
        subtype_logits = self.subtype_classifier(fused_rep)

        return {
            "pa_logits": pa_logits,
            "sarcasm_logits": sarcasm_logits,
            "subtype_logits": subtype_logits,
            "fused_representation": fused_rep,
            "gate_weights": gate_weights,
        }


def validate_fusion_shapes(batch_size: int = 4) -> Dict[str, Any]:
    """Validates tensor shapes and forward pass under diverse modality configurations."""
    model = PassiveAggressionMultimodalModel()
    model.eval()

    comment = torch.randn(batch_size, 768)
    image = torch.randn(batch_size, 768)
    context = torch.randn(batch_size, 768)
    ocr = torch.randn(batch_size, 768)

    # Varied masks
    image_mask = torch.tensor([True, True, False, False])
    context_mask = torch.tensor([True, False, True, False])
    ocr_mask = torch.tensor([False, True, True, False])

    with torch.no_grad():
        out = model(
            comment_embeds=comment,
            image_embeds=image,
            context_embeds=context,
            ocr_embeds=ocr,
            image_mask=image_mask,
            context_mask=context_mask,
            ocr_mask=ocr_mask,
        )

    assert out["pa_logits"].shape == (batch_size, 3), f"Invalid pa_logits shape: {out['pa_logits'].shape}"
    assert out["sarcasm_logits"].shape == (batch_size, 3), f"Invalid sarcasm_logits shape: {out['sarcasm_logits'].shape}"
    assert out["subtype_logits"].shape == (batch_size, 7), f"Invalid subtype_logits shape: {out['subtype_logits'].shape}"
    assert out["fused_representation"].shape == (batch_size, 512), f"Invalid fused shape: {out['fused_representation'].shape}"
    assert out["gate_weights"].shape == (batch_size, 4), f"Invalid gate_weights shape: {out['gate_weights'].shape}"

    return {
        "status": "VALIDATED",
        "batch_size": batch_size,
        "pa_logits_shape": list(out["pa_logits"].shape),
        "sarcasm_logits_shape": list(out["sarcasm_logits"].shape),
        "subtype_logits_shape": list(out["subtype_logits"].shape),
        "fused_dim": list(out["fused_representation"].shape),
        "gate_weights_shape": list(out["gate_weights"].shape),
    }


if __name__ == "__main__":
    res = validate_fusion_shapes(4)
    print("Fusion validation results:", res)
