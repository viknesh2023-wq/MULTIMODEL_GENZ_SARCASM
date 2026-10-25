"""Prediction Error and Modality Sensitivity Analysis Script (Phase 3 Task L).

Analyzes classification errors against the failure taxonomy and evaluates
modality sensitivity shifts when images or dialogue contexts are ablated at inference time.
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.error_analysis import ERROR_TAXONOMY, diagnose_error, evaluate_modality_sensitivity
from src.features.feature_store import FeatureStore
from src.models.context_aware_fusion import ContextAwareMultimodalFusion
from src.models.gated_multimodal_fusion import MultimodalFusionClassifier


def analyze_errors_and_interpretability(
    predictions_file: str = "artifacts/phase3/evaluation/final_model_predictions.jsonl",
    output_dir: str = "artifacts/phase3/evaluation",
) -> Dict[str, Any]:
    print("\n" + "=" * 70)
    print("PHASE 3 TASK L: Error Analysis and Modality Sensitivity")
    print("=" * 70)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Parse prediction records
    preds_by_model: Dict[str, List[Dict[str, Any]]] = {}
    with open(predictions_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            m = rec["model"]
            if m not in preds_by_model:
                preds_by_model[m] = []
            preds_by_model[m].append(rec)

    error_summary = {}
    for m, recs in preds_by_model.items():
        err_counts = Counter()
        mismatches = []
        for r in recs:
            diag = diagnose_error(
                ground_truth=r["ground_truth"],
                predicted=r["predicted"],
                probabilities=r["probabilities"],
                has_image="image" in r["modalities"],
                has_context="context" in r["modalities"],
                has_ocr="ocr" in r["modalities"],
            )
            err_counts[diag] += 1
            if diag != "CORRECT":
                mismatches.append({
                    "example_id": r["example_id"],
                    "ground_truth": r["ground_truth"],
                    "predicted": r["predicted"],
                    "diagnosis": diag,
                    "probabilities": r["probabilities"],
                })

        error_summary[m] = {
            "total_evaluated": len(recs),
            "correct_count": err_counts["CORRECT"],
            "error_distribution": dict(err_counts),
            "error_rate": (len(recs) - err_counts["CORRECT"]) / max(len(recs), 1),
            "mismatch_samples": mismatches[:5],
        }

    # 2. Modality Sensitivity Analysis on Multimodal Models
    fs = FeatureStore(storage_dir="data/features/phase3")
    feat = fs.load_features("synthetic_pilot_350")
    test_idx = list(range(len(feat.example_ids)))[-54:]  # held-out slice

    # Load Gated Multimodal Fusion
    fusion_model = MultimodalFusionClassifier(image_dim=768, text_dim=768, hidden_dim=512, num_classes_primary=3, num_classes_auxiliary=None)
    ckpt_f = torch.load("artifacts/phase3/multimodal_fusion/best_checkpoint.pt", map_location="cpu", weights_only=False)
    fusion_model.load_state_dict(ckpt_f["model_state_dict"])

    sens_fusion = evaluate_modality_sensitivity(
        model=fusion_model,
        comment_features=feat.comment_features[test_idx],
        image_features=feat.image_features[test_idx],
        context_features=feat.context_features[test_idx],
        ocr_features=feat.ocr_features[test_idx],
        image_masks=feat.image_masks[test_idx],
        context_masks=feat.context_masks[test_idx],
        ocr_masks=feat.ocr_masks[test_idx],
    )

    # Load Context-Aware Fusion
    ctx_model = ContextAwareMultimodalFusion(image_dim=768, text_dim=768, hidden_dim=512, num_classes=3, num_attention_heads=4)
    ckpt_c = torch.load("artifacts/phase3/context_aware_fusion/best_checkpoint.pt", map_location="cpu", weights_only=False)
    ctx_model.load_state_dict(ckpt_c["model_state_dict"])

    sens_ctx = evaluate_modality_sensitivity(
        model=ctx_model,
        comment_features=feat.comment_features[test_idx],
        image_features=feat.image_features[test_idx],
        context_features=feat.context_features[test_idx],
        ocr_features=feat.ocr_features[test_idx],
        image_masks=feat.image_masks[test_idx],
        context_masks=feat.context_masks[test_idx],
        ocr_masks=feat.ocr_masks[test_idx],
    )

    full_analysis = {
        "error_taxonomy": ERROR_TAXONOMY,
        "model_error_profiles": error_summary,
        "modality_sensitivity": {
            "gated_multimodal_fusion": sens_fusion,
            "context_aware_fusion": sens_ctx,
        },
    }

    out_file = out_dir / "error_analysis_report.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(full_analysis, f, indent=2)

    print(f"[+] Error analysis completed and saved to {out_file}")
    for m, prof in error_summary.items():
        print(f"  Model '{m}': Error Rate = {prof['error_rate'] * 100:.1f}% ({prof['correct_count']}/{prof['total_evaluated']} correct)")
    print(f"  Gated Fusion Inference Shift Rates: Image Zeroing = {sens_fusion['image_ablation_shift_rate']*100:.1f}%, Context Zeroing = {sens_fusion['context_ablation_shift_rate']*100:.1f}%")
    print(f"  Context-Aware Fusion Shift Rates: Image Zeroing = {sens_ctx['image_ablation_shift_rate']*100:.1f}%, Context Zeroing = {sens_ctx['context_ablation_shift_rate']*100:.1f}%")

    return full_analysis


if __name__ == "__main__":
    analyze_errors_and_interpretability()
