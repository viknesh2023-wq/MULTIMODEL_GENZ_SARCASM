"""Train Context-Aware Multimodal Fusion Model (Phase 3 Task H).

Trains ContextAwareMultimodalFusion + ClassificationHead incorporating
dialogue context cross-attention gating, visual scenes, and OCR text.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict
import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.features.feature_store import FeatureStore, MultimodalDataset
from src.models.classification_head import PRIMARY_IDX_TO_LABEL
from src.models.context_aware_fusion import ContextAwareMultimodalFusion
from src.training.trainer import StandardTrainer, TrainingConfig


def train_context_aware_fusion(
    feature_file: str = "synthetic_pilot_350",
    artifact_dir: str = "artifacts/phase3/context_aware_fusion",
    epochs: int = 15,
    lr: float = 1e-3,
    batch_size: int = 16,
    seed: int = 42,
) -> Dict[str, Any]:
    print("\n" + "=" * 60)
    print("TASK H: Training Context-Aware Multimodal Fusion Model")
    print("=" * 60)

    out_dir = Path(artifact_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fs = FeatureStore(storage_dir="data/features/phase3")
    feat = fs.load_features(feature_file)
    n = len(feat.example_ids)
    print(f"[*] Loaded {n} multimodal feature instances from {feature_file}.")

    # Deterministic split 70/15/15
    np.random.seed(seed)
    indices = np.random.permutation(n).tolist()
    n_train = int(0.70 * n)
    n_val = int(0.15 * n)

    train_idx = indices[:n_train]
    val_idx = indices[n_train : n_train + n_val]
    test_idx = indices[n_train + n_val :]

    train_ds = MultimodalDataset(feat, train_idx)
    val_ds = MultimodalDataset(feat, val_idx)
    test_ds = MultimodalDataset(feat, test_idx)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, collate_fn=MultimodalDataset.collate_fn
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, collate_fn=MultimodalDataset.collate_fn
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, collate_fn=MultimodalDataset.collate_fn
    )

    class_names = [PRIMARY_IDX_TO_LABEL[i] for i in range(3)]
    config = TrainingConfig(
        learning_rate=lr,
        epochs=epochs,
        batch_size=batch_size,
        early_stopping_patience=4,
        seed=seed,
        device="cpu",
        class_names=class_names,
    )

    model = ContextAwareMultimodalFusion(
        image_dim=feat.image_features.shape[-1],
        text_dim=feat.comment_features.shape[-1],
        hidden_dim=512,
        num_classes=3,
        num_attention_heads=4,
        dropout=0.1,
    )

    trainer = StandardTrainer(
        model=model,
        config=config,
        checkpoint_dir=out_dir,
    )

    print(f"[*] Training on {len(train_idx)} instances, validating on {len(val_idx)} instances...")
    fit_summary = trainer.fit(train_loader, val_loader)

    print("\n[*] Evaluating Best Context-Aware Checkpoint on Held-Out Test Set...")
    test_loss, test_metrics, test_preds, test_probs = trainer.evaluate(test_loader, prefix="test_")

    print(f"[+] Held-Out Test Evaluation:")
    print(f"    Test Loss: {test_loss:.4f}")
    print(f"    Test Macro-F1: {test_metrics['test_macro_f1']:.4f}")
    print(f"    Test Accuracy: {test_metrics['test_accuracy']:.4f}")
    print(f"    Test Balanced Accuracy: {test_metrics['test_balanced_accuracy']:.4f}")
    for cname, p_metrics in test_metrics["test_per_class"].items():
        print(f"    Class '{cname}': Precision={p_metrics['precision']:.3f}, Recall={p_metrics['recall']:.3f}, F1={p_metrics['f1']:.3f}")

    results = {
        "model_architecture": "ContextAwareMultimodalFusion (Cross-Attention Gating) + ClassificationHead",
        "training_config": {
            "epochs": epochs,
            "learning_rate": lr,
            "batch_size": batch_size,
            "seed": seed,
        },
        "best_val_macro_f1": fit_summary["best_val_macro_f1"],
        "best_checkpoint": fit_summary["best_checkpoint"],
        "test_metrics": test_metrics,
        "training_history": fit_summary["history"],
    }

    out_file = out_dir / "context_aware_fusion_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"[+] Results saved to {out_file}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Context-Aware Multimodal Fusion")
    parser.add_argument("--feature_file", type=str, default="synthetic_pilot_350")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train_context_aware_fusion(
        feature_file=args.feature_file,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        seed=args.seed,
    )
