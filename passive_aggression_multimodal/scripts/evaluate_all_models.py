"""Master Evaluation Script Across All Phase 3 Models (Task K).

Loads trained checkpoints for all model families (Traditional TF-IDF, Transformer Text,
Gated Multimodal Fusion, and Context-Aware Fusion) and generates comprehensive
held-out test metrics and serialized prediction records.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List
import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.metrics import compute_classification_metrics
from src.features.feature_store import FeatureStore, MultimodalDataset
from src.models.classification_head import PRIMARY_IDX_TO_LABEL, PRIMARY_LABEL_TO_IDX
from src.models.context_aware_fusion import ContextAwareMultimodalFusion
from src.models.gated_multimodal_fusion import MultimodalFusionClassifier
from src.models.text_baseline import TextBaselineModel
from src.models.text_transformer import TransformerTextClassifier


def run_full_evaluation(
    feature_file: str = "synthetic_pilot_350",
    output_dir: str = "artifacts/phase3/evaluation",
    seed: int = 42,
) -> Dict[str, Any]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("PHASE 3 TASK K: Comprehensive Evaluation Across All Models")
    print("=" * 70)

    # Load test split
    fs = FeatureStore(storage_dir="data/features/phase3")
    feat = fs.load_features(feature_file)
    n = len(feat.example_ids)

    np.random.seed(seed)
    indices = np.random.permutation(n).tolist()
    n_train = int(0.70 * n)
    n_val = int(0.15 * n)
    test_idx = indices[n_train + n_val :]

    test_ds = MultimodalDataset(feat, test_idx)
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False, collate_fn=MultimodalDataset.collate_fn)
    class_names = [PRIMARY_IDX_TO_LABEL[i] for i in range(3)]

    # Raw texts for traditional baseline
    raw_texts = []
    with open("data/synthetic/synthetic_pilot_350.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            raw_texts.append(r.get("target_comment") or "")
    test_raw_texts = [raw_texts[i] for i in test_idx]
    test_labels = feat.labels[test_idx].cpu().numpy().tolist()
    test_ids = [feat.example_ids[i] for i in test_idx]

    results = {}
    prediction_records = []

    # 1. TF-IDF + Logistic Regression
    print("[*] Evaluating Traditional TF-IDF Baseline...")
    tfidf_path = "artifacts/phase3/text_baseline/best_text_baseline_primary.pkl"
    if os.path.exists(tfidf_path):
        tfidf_model = TextBaselineModel.load(tfidf_path)
        preds_tfidf = tfidf_model.predict(test_raw_texts)
        probs_tfidf = tfidf_model.predict_proba(test_raw_texts)
        metrics_tfidf = compute_classification_metrics(test_labels, preds_tfidf, class_names=class_names, prefix="test_")
        results["TFIDF_LogisticRegression"] = metrics_tfidf

        for eid, y_true, y_pred, prob in zip(test_ids, test_labels, preds_tfidf, probs_tfidf):
            prediction_records.append({
                "example_id": eid,
                "model": "TFIDF_LogisticRegression",
                "ground_truth": class_names[y_true],
                "predicted": class_names[y_pred],
                "probabilities": {class_names[c]: float(prob[c]) for c in range(3)},
                "modalities": ["comment"],
            })

    # 2. Transformer Text Baseline
    print("[*] Evaluating Pretrained Transformer Text Baseline...")
    trans_path = "artifacts/phase3/text_transformer/best_checkpoint.pt"
    if os.path.exists(trans_path):
        trans_model = TransformerTextClassifier(load_encoder_weights=False)
        ckpt = torch.load(trans_path, map_location="cpu", weights_only=False)
        trans_model.load_state_dict(ckpt["model_state_dict"])
        trans_model.eval()

        with torch.no_grad():
            x_test = feat.comment_features[test_idx]
            logits = trans_model(x_test)
            probs_trans = torch.softmax(logits, dim=-1).numpy()
            preds_trans = np.argmax(probs_trans, axis=-1)

        metrics_trans = compute_classification_metrics(test_labels, preds_trans, class_names=class_names, prefix="test_")
        results["Transformer_Text"] = metrics_trans

        for eid, y_true, y_pred, prob in zip(test_ids, test_labels, preds_trans, probs_trans):
            prediction_records.append({
                "example_id": eid,
                "model": "Transformer_Text",
                "ground_truth": class_names[y_true],
                "predicted": class_names[y_pred],
                "probabilities": {class_names[c]: float(prob[c]) for c in range(3)},
                "modalities": ["comment"],
            })

    # 3. Gated Multimodal Fusion
    print("[*] Evaluating Gated Multimodal Fusion Model...")
    fusion_path = "artifacts/phase3/multimodal_fusion/best_checkpoint.pt"
    if os.path.exists(fusion_path):
        fusion_model = MultimodalFusionClassifier(
            image_dim=768, text_dim=768, hidden_dim=512, num_classes_primary=3, num_classes_auxiliary=None
        )
        ckpt = torch.load(fusion_path, map_location="cpu", weights_only=False)
        fusion_model.load_state_dict(ckpt["model_state_dict"])
        fusion_model.eval()

        all_preds = []
        all_probs = []
        with torch.no_grad():
            for batch in test_loader:
                out = fusion_model(
                    comment_features=batch.comment_features,
                    image_features=batch.image_features,
                    context_features=batch.context_features,
                    ocr_features=batch.ocr_features,
                    image_masks=batch.image_masks,
                    context_masks=batch.context_masks,
                    ocr_masks=batch.ocr_masks,
                )
                p = torch.softmax(out["primary_logits"], dim=-1).numpy()
                all_probs.append(p)
                all_preds.append(np.argmax(p, axis=-1))

        probs_fusion = np.concatenate(all_probs, axis=0)
        preds_fusion = np.concatenate(all_preds, axis=0)
        metrics_fusion = compute_classification_metrics(test_labels, preds_fusion, class_names=class_names, prefix="test_")
        results["Gated_Multimodal_Fusion"] = metrics_fusion

        for eid, y_true, y_pred, prob in zip(test_ids, test_labels, preds_fusion, probs_fusion):
            prediction_records.append({
                "example_id": eid,
                "model": "Gated_Multimodal_Fusion",
                "ground_truth": class_names[y_true],
                "predicted": class_names[y_pred],
                "probabilities": {class_names[c]: float(prob[c]) for c in range(3)},
                "modalities": ["comment", "image", "context", "ocr"],
            })

    # 4. Context-Aware Multimodal Fusion
    print("[*] Evaluating Context-Aware Multimodal Fusion Model...")
    ctx_path = "artifacts/phase3/context_aware_fusion/best_checkpoint.pt"
    if os.path.exists(ctx_path):
        ctx_model = ContextAwareMultimodalFusion(
            image_dim=768, text_dim=768, hidden_dim=512, num_classes=3, num_attention_heads=4
        )
        ckpt = torch.load(ctx_path, map_location="cpu", weights_only=False)
        ctx_model.load_state_dict(ckpt["model_state_dict"])
        ctx_model.eval()

        all_preds = []
        all_probs = []
        with torch.no_grad():
            for batch in test_loader:
                out = ctx_model(
                    comment_features=batch.comment_features,
                    image_features=batch.image_features,
                    context_features=batch.context_features,
                    ocr_features=batch.ocr_features,
                    image_masks=batch.image_masks,
                    context_masks=batch.context_masks,
                    ocr_masks=batch.ocr_masks,
                )
                p = torch.softmax(out["primary_logits"], dim=-1).numpy()
                all_probs.append(p)
                all_preds.append(np.argmax(p, axis=-1))

        probs_ctx = np.concatenate(all_probs, axis=0)
        preds_ctx = np.concatenate(all_preds, axis=0)
        metrics_ctx = compute_classification_metrics(test_labels, preds_ctx, class_names=class_names, prefix="test_")
        results["Context_Aware_Multimodal_Fusion"] = metrics_ctx

        for eid, y_true, y_pred, prob in zip(test_ids, test_labels, preds_ctx, probs_ctx):
            prediction_records.append({
                "example_id": eid,
                "model": "Context_Aware_Multimodal_Fusion",
                "ground_truth": class_names[y_true],
                "predicted": class_names[y_pred],
                "probabilities": {class_names[c]: float(prob[c]) for c in range(3)},
                "modalities": ["comment", "image", "context", "ocr"],
            })

    # Save outputs
    preds_file = out_dir / "final_model_predictions.jsonl"
    with open(preds_file, "w", encoding="utf-8") as f:
        for rec in prediction_records:
            f.write(json.dumps(rec) + "\n")

    summary_file = out_dir / "final_evaluation_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\n[+] Final predictions saved to {preds_file} ({len(prediction_records)} records)")
    print(f"[+] Evaluation summary saved to {summary_file}")
    return results


if __name__ == "__main__":
    run_full_evaluation()
