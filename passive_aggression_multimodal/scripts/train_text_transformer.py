"""Training Script for Pretrained Transformer Text Baseline (Phase 3 Task D).

Trains a 3-class ClassificationHead on 768-dim XLM-RoBERTa representations.
Tracks training/validation curves, applies early stopping on val macro-F1,
and evaluates on held-out test data.
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
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.features.feature_store import FeatureStore
from src.models.classification_head import PRIMARY_IDX_TO_LABEL
from src.models.text_transformer import TransformerTextClassifier
from src.training.trainer import StandardTrainer, TrainingConfig


def train_transformer_baseline(
    feature_file: str = "synthetic_pilot_350",
    artifact_dir: str = "artifacts/phase3/text_transformer",
    epochs: int = 15,
    lr: float = 1e-3,
    batch_size: int = 16,
    seed: int = 42,
) -> Dict[str, Any]:
    print("\n" + "=" * 60)
    print("TASK D: Training Pretrained Transformer Text Baseline")
    print("=" * 60)

    out_dir = Path(artifact_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load pre-extracted real features
    fs = FeatureStore(storage_dir="data/features/phase3")
    feat = fs.load_features(feature_file)
    n = len(feat.example_ids)
    print(f"[*] Loaded {n} feature instances from {feature_file}. Comment feature dim: {feat.comment_features.shape[-1]}")

    # Deterministic split 70/15/15
    np.random.seed(seed)
    indices = np.random.permutation(n)
    n_train = int(0.70 * n)
    n_val = int(0.15 * n)

    train_idx = indices[:n_train]
    val_idx = indices[n_train : n_train + n_val]
    test_idx = indices[n_train + n_val :]

    x_train, y_train = feat.comment_features[train_idx], feat.labels[train_idx]
    x_val, y_val = feat.comment_features[val_idx], feat.labels[val_idx]
    x_test, y_test = feat.comment_features[test_idx], feat.labels[test_idx]

    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=batch_size, shuffle=False)

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

    model = TransformerTextClassifier(
        model_name_or_path="FacebookAI/xlm-roberta-base",
        num_classes=3,
        dropout=0.1,
        device="cpu",
        load_encoder_weights=False,  # using cached 768-dim features
    )

    trainer = StandardTrainer(
        model=model,
        config=config,
        checkpoint_dir=out_dir,
    )

    print(f"[*] Training on {len(train_idx)} instances, validating on {len(val_idx)} instances...")
    fit_summary = trainer.fit(train_loader, val_loader)

    # Evaluate best checkpoint on held-out test data
    print("\n[*] Evaluating Best Checkpoint on Held-Out Test Set...")
    test_loss, test_metrics, test_preds, test_probs = trainer.evaluate(test_loader, prefix="test_")

    print(f"[+] Held-Out Test Evaluation:")
    print(f"    Test Loss: {test_loss:.4f}")
    print(f"    Test Macro-F1: {test_metrics['test_macro_f1']:.4f}")
    print(f"    Test Accuracy: {test_metrics['test_accuracy']:.4f}")
    print(f"    Test Balanced Accuracy: {test_metrics['test_balanced_accuracy']:.4f}")
    for cname, p_metrics in test_metrics["test_per_class"].items():
        print(f"    Class '{cname}': Precision={p_metrics['precision']:.3f}, Recall={p_metrics['recall']:.3f}, F1={p_metrics['f1']:.3f}")

    results = {
        "model_architecture": "XLM-RoBERTa-base + ClassificationHead",
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

    out_file = out_dir / "text_transformer_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"[+] Results saved to {out_file}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Transformer Text Baseline")
    parser.add_argument("--feature_file", type=str, default="synthetic_pilot_350")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train_transformer_baseline(
        feature_file=args.feature_file,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        seed=args.seed,
    )
