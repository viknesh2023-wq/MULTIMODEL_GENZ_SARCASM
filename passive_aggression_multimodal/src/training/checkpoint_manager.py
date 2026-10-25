"""Checkpoint Manager for Saving, Tracking, and Reloading PyTorch Models.

Maintains top-k checkpoints based on validation metrics, records training history,
and computes cryptographic checksums for reproducibility audits.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import torch


class CheckpointManager:
    """Manages model checkpoints and training state tracking."""

    def __init__(
        self,
        checkpoint_dir: Union[str, Path],
        metric_name: str = "val_macro_f1",
        mode: str = "max",
        max_to_keep: int = 3,
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.metric_name = metric_name
        self.mode = mode.lower()
        self.max_to_keep = max_to_keep

        if self.mode not in ("min", "max"):
            raise ValueError(f"Mode must be 'min' or 'max', got {mode}")

        self.best_metric = -float("inf") if self.mode == "max" else float("inf")
        self.best_epoch = -1
        self.best_checkpoint_path: Optional[Path] = None
        self.saved_checkpoints: List[Dict[str, Any]] = []

    def is_better(self, metric_val: float) -> bool:
        """Determines if new metric value improves upon current best."""
        if self.mode == "max":
            return metric_val > self.best_metric
        else:
            return metric_val < self.best_metric

    def save_checkpoint(
        self,
        epoch: int,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        metrics: Optional[Dict[str, Any]] = None,
        config: Optional[Dict[str, Any]] = None,
        is_best: bool = False,
    ) -> Path:
        """Saves a model checkpoint with full training state."""
        metrics = metrics or {}
        filename = f"checkpoint_epoch_{epoch:03d}.pt"
        if is_best:
            filename = "best_checkpoint.pt"

        save_path = self.checkpoint_dir / filename

        # Compute SHA-256 for data integrity
        payload = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "metrics": metrics,
            "config": config or {},
        }
        torch.save(payload, save_path)

        # Compute file checksum
        sha256 = hashlib.sha256()
        with open(save_path, "rb") as f:
            while chunk := f.read(65536):
                sha256.update(chunk)
        checksum = sha256.hexdigest()

        metric_val = metrics.get(self.metric_name, 0.0)
        ckpt_meta = {
            "epoch": epoch,
            "path": str(save_path),
            "metric_name": self.metric_name,
            "metric_value": metric_val,
            "sha256": checksum,
            "is_best": is_best,
        }
        self.saved_checkpoints.append(ckpt_meta)

        if is_best or self.is_better(metric_val):
            self.best_metric = metric_val
            self.best_epoch = epoch
            self.best_checkpoint_path = save_path
            # Also write a dedicated best_checkpoint.pt copy if this wasn't already best_checkpoint.pt
            if filename != "best_checkpoint.pt":
                best_path = self.checkpoint_dir / "best_checkpoint.pt"
                torch.save(payload, best_path)
                self.best_checkpoint_path = best_path

        # Save registry json
        with open(self.checkpoint_dir / "checkpoint_registry.json", "w", encoding="utf-8") as f:
            json.dump({
                "best_epoch": self.best_epoch,
                "best_metric": self.best_metric,
                "best_checkpoint": str(self.best_checkpoint_path),
                "checkpoints": self.saved_checkpoints,
            }, f, indent=2)

        return save_path

    def load_best_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        device: str = "cpu",
    ) -> Dict[str, Any]:
        """Loads state_dict of best checkpoint into model."""
        best_path = self.best_checkpoint_path or (self.checkpoint_dir / "best_checkpoint.pt")
        if not best_path.exists():
            raise FileNotFoundError(f"Best checkpoint not found at {best_path}")

        payload = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(payload["model_state_dict"])
        if optimizer and payload.get("optimizer_state_dict"):
            optimizer.load_state_dict(payload["optimizer_state_dict"])

        return payload
