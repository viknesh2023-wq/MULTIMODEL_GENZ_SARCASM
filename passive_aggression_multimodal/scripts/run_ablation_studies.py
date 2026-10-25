"""Controlled Baseline and Ablation Studies (Phase 3 Task J).

Executes systematic ablation across 8 distinct configurations on the exact same
held-out evaluation subset:
- Experiment A: Comment only: TF-IDF + Logistic Regression
- Experiment B: Comment only: TF-IDF + Linear SVM
- Experiment C: Comment + Context: Pretrained Transformer
- Experiment D: Image + Comment Fusion (Context & OCR masked)
- Experiment E: Image + Comment + Context (OCR masked)
- Experiment F: Image + Comment + Context + OCR (Full multimodal)
- Experiment G: Comment + Context without Image (Image ablation)
- Experiment H: Image + Comment without Context (Context ablation)
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.metrics import compute_classification_metrics
from src.features.feature_store import FeatureStore, MultimodalBatchFeatures, MultimodalDataset
from src.models.classification_head import PRIMARY_IDX_TO_LABEL, PRIMARY_LABEL_TO_IDX
from src.models.context_aware_fusion import ContextAwareMultimodalFusion
from src.models.gated_multimodal_fusion import MultimodalFusionClassifier
from src.models.text_baseline import TextBaselineModel
from src.models.text_transformer import TransformerTextClassifier
from src.training.trainer import StandardTrainer, TrainingConfig


def create_ablated_features(
    raw_feat: MultimodalBatchFeatures,
    keep_image: bool = True,
    keep_context: bool = True,
    keep_ocr: bool = True,
) -> MultimodalBatchFeatures:
    """Clones features and zeros out representations & masks for ablated modalities."""
    feat = copy.deepcopy(raw_feat)
    batch_size = len(feat.example_ids)

    if not keep_image:
        feat.image_features = torch.zeros_like(feat.image_features)
        feat.image_masks = torch.zeros(batch_size, dtype=torch.bool)

    if not keep_context:
        feat.context_features = torch.zeros_like(feat.context_features)
        feat.context_masks = torch.zeros(batch_size, dtype=torch.bool)

    if not keep_ocr:
        feat.ocr_features = torch.zeros_like(feat.ocr_features)
        feat.ocr_masks = torch.zeros(batch_size, dtype=torch.bool)

    return feat


def run_all_ablations(
    feature_file: str = "synthetic_pilot_350",
    output_dir: str = "artifacts/phase3/evaluation",
    seed: int = 42,
) -> Dict[str, Any]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("PHASE 3 TASK J: Controlled Baseline & Ablation Experiments")
    print("=" * 70)

    # 1. Load base features and create exact split indices
    fs = FeatureStore(storage_dir="data/features/phase3")
    base_feat = fs.load_features(feature_file)
    n = len(base_feat.example_ids)

    np.random.seed(seed)
    indices = np.random.permutation(n).tolist()
    n_train = int(0.70 * n)
    n_val = int(0.15 * n)
    train_idx = indices[:n_train]
    val_idx = indices[n_train : n_train + n_val]
    test_idx = indices[n_train + n_val :]

    print(f"Dataset: {n} instances | Train: {len(train_idx)} | Val: {len(val_idx)} | Test: {len(test_idx)}")
    class_names = [PRIMARY_IDX_TO_LABEL[i] for i in range(3)]

    # Load raw text comments for TF-IDF baselines
    raw_texts = []
    raw_labels = []
    with open("data/synthetic/synthetic_pilot_350.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            raw_texts.append(r.get("target_comment") or "")
            raw_labels.append(PRIMARY_LABEL_TO_IDX.get(r.get("passive_aggression_label"), -1))

    train_raw_texts = [raw_texts[i] for i in train_idx]
    train_raw_labels = [raw_labels[i] for i in train_idx]
    val_raw_texts = [raw_texts[i] for i in val_idx]
    val_raw_labels = [raw_labels[i] for i in val_idx]
    test_raw_texts = [raw_texts[i] for i in test_idx]
    test_raw_labels = [raw_labels[i] for i in test_idx]

    ablation_results = {}

    # -------------------------------------------------------------
    # Experiment A: Comment only: TF-IDF + Logistic Regression
    # -------------------------------------------------------------
    print("\n[*] Running Experiment A: Comment only (TF-IDF + Logistic Regression)...")
    t0 = time.time()
    clf_a = TextBaselineModel("logistic_regression", c_param=0.5, random_state=seed)
    clf_a.fit(train_raw_texts, train_raw_labels)
    preds_a = clf_a.predict(test_raw_texts)
    metrics_a = compute_classification_metrics(test_raw_labels, preds_a, class_names=class_names, prefix="test_")
    ablation_results["Exp_A_TFIDF_LogReg"] = {
        "name": "Exp A: Comment only (TF-IDF + LogReg)",
        "inputs": "Comment only",
        "macro_f1": metrics_a["test_macro_f1"],
        "accuracy": metrics_a["test_accuracy"],
        "balanced_accuracy": metrics_a["test_balanced_accuracy"],
        "runtime_sec": time.time() - t0,
    }

    # -------------------------------------------------------------
    # Experiment B: Comment only: TF-IDF + Linear SVM
    # -------------------------------------------------------------
    print("[*] Running Experiment B: Comment only (TF-IDF + Linear SVM)...")
    t0 = time.time()
    clf_b = TextBaselineModel("linear_svm", c_param=0.5, random_state=seed)
    clf_b.fit(train_raw_texts, train_raw_labels)
    preds_b = clf_b.predict(test_raw_texts)
    metrics_b = compute_classification_metrics(test_raw_labels, preds_b, class_names=class_names, prefix="test_")
    ablation_results["Exp_B_TFIDF_LinearSVM"] = {
        "name": "Exp B: Comment only (TF-IDF + Linear SVM)",
        "inputs": "Comment only",
        "macro_f1": metrics_b["test_macro_f1"],
        "accuracy": metrics_b["test_accuracy"],
        "balanced_accuracy": metrics_b["test_balanced_accuracy"],
        "runtime_sec": time.time() - t0,
    }

    # -------------------------------------------------------------
    # Helper for neural ablations
    # -------------------------------------------------------------
    def train_and_eval_neural_ablation(exp_id: str, exp_name: str, input_desc: str, feat_obj, model_cls, model_kwargs):
        print(f"[*] Running {exp_name}...")
        t0 = time.time()
        train_ds = MultimodalDataset(feat_obj, train_idx)
        val_ds = MultimodalDataset(feat_obj, val_idx)
        test_ds = MultimodalDataset(feat_obj, test_idx)

        train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, collate_fn=MultimodalDataset.collate_fn)
        val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, collate_fn=MultimodalDataset.collate_fn)
        test_loader = DataLoader(test_ds, batch_size=16, shuffle=False, collate_fn=MultimodalDataset.collate_fn)

        model = model_cls(**model_kwargs)
        cfg = TrainingConfig(epochs=10, learning_rate=1e-3, batch_size=16, early_stopping_patience=3, seed=seed, class_names=class_names)
        trainer = StandardTrainer(model=model, config=cfg, checkpoint_dir=out_dir / exp_id)
        trainer.fit(train_loader, val_loader)

        test_loss, test_metrics, _, _ = trainer.evaluate(test_loader, prefix="test_")
        ablation_results[exp_id] = {
            "name": exp_name,
            "inputs": input_desc,
            "macro_f1": test_metrics["test_macro_f1"],
            "accuracy": test_metrics["test_accuracy"],
            "balanced_accuracy": test_metrics["test_balanced_accuracy"],
            "test_loss": test_loss,
            "runtime_sec": time.time() - t0,
        }

    # Exp C: Comment + context: transformer
    train_and_eval_neural_ablation(
        "Exp_C_Transformer_Comment_Context",
        "Exp C: Comment + Context (Transformer)",
        "Comment + Context",
        create_ablated_features(base_feat, keep_image=False, keep_context=True, keep_ocr=False),
        MultimodalFusionClassifier,
        {"image_dim": 768, "text_dim": 768, "hidden_dim": 512, "num_classes_primary": 3, "num_classes_auxiliary": None},
    )

    # Exp D: Image + comment (Context & OCR masked)
    train_and_eval_neural_ablation(
        "Exp_D_Image_Comment",
        "Exp D: Image + Comment (Image-Text Fusion)",
        "Image + Comment",
        create_ablated_features(base_feat, keep_image=True, keep_context=False, keep_ocr=False),
        MultimodalFusionClassifier,
        {"image_dim": 768, "text_dim": 768, "hidden_dim": 512, "num_classes_primary": 3, "num_classes_auxiliary": None},
    )

    # Exp E: Image + comment + context (OCR masked)
    train_and_eval_neural_ablation(
        "Exp_E_Image_Comment_Context",
        "Exp E: Image + Comment + Context",
        "Image + Comment + Context",
        create_ablated_features(base_feat, keep_image=True, keep_context=True, keep_ocr=False),
        ContextAwareMultimodalFusion,
        {"image_dim": 768, "text_dim": 768, "hidden_dim": 512, "num_classes": 3},
    )

    # Exp F: Image + comment + context + OCR (Full Multimodal)
    train_and_eval_neural_ablation(
        "Exp_F_Full_Multimodal_OCR",
        "Exp F: Full Multimodal (Image + Comment + Context + OCR)",
        "Image + Comment + Context + OCR",
        create_ablated_features(base_feat, keep_image=True, keep_context=True, keep_ocr=True),
        ContextAwareMultimodalFusion,
        {"image_dim": 768, "text_dim": 768, "hidden_dim": 512, "num_classes": 3},
    )

    # Exp G: Comment + context, without image (Image ablation)
    train_and_eval_neural_ablation(
        "Exp_G_Image_Ablation",
        "Exp G: Image Ablation (Comment + Context)",
        "Comment + Context (No Image)",
        create_ablated_features(base_feat, keep_image=False, keep_context=True, keep_ocr=True),
        ContextAwareMultimodalFusion,
        {"image_dim": 768, "text_dim": 768, "hidden_dim": 512, "num_classes": 3},
    )

    # Exp H: Image + comment, without context (Context ablation)
    train_and_eval_neural_ablation(
        "Exp_H_Context_Ablation",
        "Exp H: Context Ablation (Image + Comment + OCR)",
        "Image + Comment + OCR (No Context)",
        create_ablated_features(base_feat, keep_image=True, keep_context=False, keep_ocr=True),
        ContextAwareMultimodalFusion,
        {"image_dim": 768, "text_dim": 768, "hidden_dim": 512, "num_classes": 3},
    )

    # Print summary table
    print("\n" + "=" * 80)
    print(f"{'Experiment':<35} | {'Inputs':<30} | {'Macro-F1':<10} | {'Accuracy':<10}")
    print("-" * 80)
    for exp_id, res in ablation_results.items():
        print(f"{res['name']:<35} | {res['inputs']:<30} | {res['macro_f1']:<10.4f} | {res['accuracy']:<10.4f}")
    print("=" * 80)

    # Save to disk
    out_file = out_dir / "ablation_study_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(ablation_results, f, indent=2)
    print(f"[+] Ablation study results saved to {out_file}")

    return ablation_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Controlled Ablation Studies")
    parser.add_argument("--feature_file", type=str, default="synthetic_pilot_350")
    parser.add_argument("--output_dir", type=str, default="artifacts/phase3/evaluation")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_all_ablations(
        feature_file=args.feature_file,
        output_dir=args.output_dir,
        seed=args.seed,
    )
