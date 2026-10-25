"""Error Analysis and Interpretability Framework (Phase 3 Task L).

Categorizes misclassifications into operational failure archetypes and evaluates
prediction sensitivity to modality removal at inference time.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn

ERROR_TAXONOMY = {
    "SARCASM_CONFUSED_WITH_PA": "Sarcasm without interpersonal hostility misclassified as passive aggression.",
    "SUBTLE_WORDING_MISSED": "Passive aggression overlooked due to understated, polite, or veiled wording.",
    "DIRECT_OVERT_AGGRESSION_CONFUSED": "Overt hostility or direct toxicity confused with covert passive aggression.",
    "GENUINE_PRAISE_MISCLASSIFIED": "Sincere appreciation or compliments incorrectly flagged as backhanded.",
    "IMAGE_COMMENT_MISMATCH": "Visual scene incongruity incorrectly biasing text interpretation.",
    "OCR_FAILURE_OR_ABSENT": "Inability to read text embedded within images obscuring meaning.",
    "MISSING_CONTEXT": "Absence of dialogue history leading to ambiguous interpretation.",
    "AMBIGUOUS_UNCERTAINTY": "Model prediction exhibits near-uniform uncertainty across classes.",
}


def diagnose_error(
    ground_truth: str,
    predicted: str,
    probabilities: Dict[str, float],
    has_image: bool = False,
    has_context: bool = False,
    has_ocr: bool = False,
    is_sarcastic_source: bool = False,
) -> str:
    """Diagnoses root cause of prediction discrepancy based on metadata and probabilities."""
    if ground_truth == predicted:
        return "CORRECT"

    max_prob = max(probabilities.values()) if probabilities else 1.0
    if max_prob < 0.55:
        return "AMBIGUOUS_UNCERTAINTY"

    if predicted == "passive_aggressive" and is_sarcastic_source:
        return "SARCASM_CONFUSED_WITH_PA"

    if ground_truth == "passive_aggressive" and predicted == "non_passive_aggressive":
        if not has_context:
            return "MISSING_CONTEXT"
        return "SUBTLE_WORDING_MISSED"

    if ground_truth == "non_passive_aggressive" and predicted == "passive_aggressive":
        if has_image and not has_ocr:
            return "IMAGE_COMMENT_MISMATCH"
        return "GENUINE_PRAISE_MISCLASSIFIED"

    return "AMBIGUOUS_UNCERTAINTY"


def evaluate_modality_sensitivity(
    model: nn.Module,
    comment_features: torch.Tensor,
    image_features: torch.Tensor,
    context_features: torch.Tensor,
    ocr_features: torch.Tensor,
    image_masks: torch.Tensor,
    context_masks: torch.Tensor,
    ocr_masks: torch.Tensor,
) -> Dict[str, Any]:
    """
    Evaluates prediction shifts when individual modalities are zeroed out at inference time.
    """
    model.eval()
    with torch.no_grad():
        # Baseline: Full input
        base_out = model(
            comment_features=comment_features,
            image_features=image_features,
            context_features=context_features,
            ocr_features=ocr_features,
            image_masks=image_masks,
            context_masks=context_masks,
            ocr_masks=ocr_masks,
        )
        base_logits = base_out["primary_logits"] if isinstance(base_out, dict) else base_out
        base_preds = torch.argmax(base_logits, dim=-1)

        # 1. Image zeroed
        no_img_out = model(
            comment_features=comment_features,
            image_features=torch.zeros_like(image_features),
            context_features=context_features,
            ocr_features=ocr_features,
            image_masks=torch.zeros_like(image_masks),
            context_masks=context_masks,
            ocr_masks=ocr_masks,
        )
        no_img_logits = no_img_out["primary_logits"] if isinstance(no_img_out, dict) else no_img_out
        no_img_preds = torch.argmax(no_img_logits, dim=-1)

        # 2. Context zeroed
        no_ctx_out = model(
            comment_features=comment_features,
            image_features=image_features,
            context_features=torch.zeros_like(context_features),
            ocr_features=ocr_features,
            image_masks=image_masks,
            context_masks=torch.zeros_like(context_masks),
            ocr_masks=ocr_masks,
        )
        no_ctx_logits = no_ctx_out["primary_logits"] if isinstance(no_ctx_out, dict) else no_ctx_out
        no_ctx_preds = torch.argmax(no_ctx_logits, dim=-1)

    # Shift proportions
    n = len(base_preds)
    img_shift_rate = float((base_preds != no_img_preds).float().mean().item()) if n > 0 else 0.0
    ctx_shift_rate = float((base_preds != no_ctx_preds).float().mean().item()) if n > 0 else 0.0

    return {
        "total_instances_evaluated": n,
        "image_ablation_shift_rate": img_shift_rate,
        "context_ablation_shift_rate": ctx_shift_rate,
    }
