"""Standardized PyTorch Trainer for Phase 3 Model Training.

Implements reproducible training loops, early stopping, gradient clipping,
checkpoint management, and validation tracking with zero test-set leakage.
"""
from __future__ import annotations

import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.evaluation.metrics import compute_classification_metrics
from src.training.checkpoint_manager import CheckpointManager


@dataclass
class TrainingConfig:
    """Hyperparameters and configuration for PyTorch training."""
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 15
    batch_size: int = 16
    early_stopping_patience: int = 4
    grad_clip_norm: float = 1.0
    seed: int = 42
    device: str = "cpu"
    class_names: Optional[List[str]] = None


def set_seed(seed: int = 42) -> None:
    """Sets deterministic random seeds across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class StandardTrainer:
    """
    Modular trainer executing train/validation loops, early stopping,
    and automatic checkpoint persistence.
    """

    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        checkpoint_dir: Union[str, Path],
        class_weights: Optional[torch.Tensor] = None,
    ):
        self.config = config
        self.device = torch.device(config.device)
        self.model = model.to(self.device)
        self.class_weights = class_weights.to(self.device) if class_weights is not None else None

        set_seed(self.config.seed)

        self.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir=checkpoint_dir,
            metric_name="val_macro_f1",
            mode="max",
        )

        self.history: List[Dict[str, Any]] = []

    def train_epoch(
        self,
        train_loader: DataLoader,
        loss_fn: Optional[Callable] = None,
    ) -> float:
        """Trains model for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch in train_loader:
            self.optimizer.zero_grad()

            # Handle different batch structures
            if isinstance(batch, (tuple, list)):
                x, y = batch[0].to(self.device), batch[1].to(self.device)
                logits = self.model(x)
                if loss_fn:
                    loss = loss_fn(logits, y)
                elif hasattr(self.model, "compute_loss"):
                    loss = self.model.compute_loss(logits, y, class_weights=self.class_weights)
                else:
                    loss = nn.functional.cross_entropy(logits, y, weight=self.class_weights)
            elif hasattr(batch, "comment_features"):  # MultimodalBatchFeatures
                batch = batch.to(self.device)
                outputs = self.model(
                    comment_features=batch.comment_features,
                    image_features=batch.image_features,
                    context_features=batch.context_features,
                    ocr_features=batch.ocr_features,
                    image_masks=batch.image_masks,
                    context_masks=batch.context_masks,
                    ocr_masks=batch.ocr_masks,
                )
                logits = outputs["primary_logits"] if isinstance(outputs, dict) else outputs
                loss = nn.functional.cross_entropy(logits, batch.labels, weight=self.class_weights)
            else:
                raise ValueError("Unrecognized batch format.")

            loss.backward()
            if self.config.grad_clip_norm > 0:
                nn.utils.clip_grad_norm_(
                    filter(lambda p: p.requires_grad, self.model.parameters()),
                    self.config.grad_clip_norm,
                )
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / max(num_batches, 1)

    def evaluate(
        self,
        val_loader: DataLoader,
        prefix: str = "val_",
    ) -> Tuple[float, Dict[str, Any], np.ndarray, np.ndarray]:
        """Evaluates model over validation or test loader."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        all_preds = []
        all_targets = []
        all_probs = []

        with torch.no_grad():
            for batch in val_loader:
                if isinstance(batch, (tuple, list)):
                    x, y = batch[0].to(self.device), batch[1].to(self.device)
                    logits = self.model(x)
                    loss = nn.functional.cross_entropy(logits, y, weight=self.class_weights)
                    targets = y
                elif hasattr(batch, "comment_features"):
                    batch = batch.to(self.device)
                    outputs = self.model(
                        comment_features=batch.comment_features,
                        image_features=batch.image_features,
                        context_features=batch.context_features,
                        ocr_features=batch.ocr_features,
                        image_masks=batch.image_masks,
                        context_masks=batch.context_masks,
                        ocr_masks=batch.ocr_masks,
                    )
                    logits = outputs["primary_logits"] if isinstance(outputs, dict) else outputs
                    loss = nn.functional.cross_entropy(logits, batch.labels, weight=self.class_weights)
                    targets = batch.labels
                else:
                    raise ValueError("Unrecognized batch format.")

                probs = nn.functional.softmax(logits, dim=-1)
                preds = torch.argmax(probs, dim=-1)

                total_loss += loss.item()
                num_batches += 1
                all_preds.extend(preds.cpu().numpy().tolist())
                all_targets.extend(targets.cpu().numpy().tolist())
                all_probs.extend(probs.cpu().numpy().tolist())

        avg_loss = total_loss / max(num_batches, 1)
        metrics = compute_classification_metrics(
            all_targets, all_preds, class_names=self.config.class_names, prefix=prefix
        )
        metrics[f"{prefix}loss"] = avg_loss
        return avg_loss, metrics, np.array(all_preds), np.array(all_probs)

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ) -> Dict[str, Any]:
        """Runs the complete training and validation cycle with early stopping."""
        patience_counter = 0
        best_val_f1 = -float("inf")

        for epoch in range(1, self.config.epochs + 1):
            epoch_start = time.time()
            train_loss = self.train_epoch(train_loader)
            val_loss, val_metrics, _, _ = self.evaluate(val_loader, prefix="val_")
            val_f1 = val_metrics["val_macro_f1"]
            epoch_duration = time.time() - epoch_start

            is_best = val_f1 > best_val_f1
            if is_best:
                best_val_f1 = val_f1
                patience_counter = 0
            else:
                patience_counter += 1

            self.checkpoint_manager.save_checkpoint(
                epoch=epoch,
                model=self.model,
                optimizer=self.optimizer,
                metrics=val_metrics,
                config=asdict(self.config),
                is_best=is_best,
            )

            epoch_record = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_macro_f1": val_f1,
                "val_accuracy": val_metrics["val_accuracy"],
                "duration_seconds": epoch_duration,
                "is_best": is_best,
            }
            self.history.append(epoch_record)
            print(
                f"  Epoch [{epoch:02d}/{self.config.epochs:02d}] "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val Macro-F1: {val_f1:.4f} | "
                f"Val Acc: {val_metrics['val_accuracy']:.4f} "
                f"({'*BEST*' if is_best else f'patience: {patience_counter}'}) [{epoch_duration:.1f}s]"
            )

            if patience_counter >= self.config.early_stopping_patience:
                print(f"[*] Early stopping triggered at epoch {epoch} (patience={self.config.early_stopping_patience}).")
                break

        # Save training history
        history_path = self.checkpoint_manager.checkpoint_dir / "training_history.json"
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump({
                "config": asdict(self.config),
                "history": self.history,
                "best_val_f1": best_val_f1,
            }, f, indent=2)

        # Load best weights before returning
        self.checkpoint_manager.load_best_checkpoint(self.model, device=self.config.device)
        return {
            "best_val_macro_f1": best_val_f1,
            "best_checkpoint": str(self.checkpoint_manager.best_checkpoint_path),
            "history": self.history,
        }
