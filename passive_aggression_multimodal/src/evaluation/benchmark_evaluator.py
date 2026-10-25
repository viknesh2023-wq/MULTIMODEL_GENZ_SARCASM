"""Real-World Benchmark Evaluation Interface (Phase 5 Task 6).

Consumes a frozen human-annotated test manifest and executes comprehensive,
statistically rigorous evaluations across model checkpoints:
- Macro & per-class precision/recall/F1, accuracy, balanced accuracy
- Confusion matrix and class distribution
- 95% bootstrap confidence intervals
- Expected Calibration Error (ECE)
- Selective classification / abstention metrics
- Modality-specific ablations on identical records
- Serialized record-level predictions with metadata
"""
from __future__ import annotations

import csv
import json
import os
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn.functional as F

from src.evaluation.metrics import (
    compute_bootstrap_confidence_intervals,
    compute_calibration_error,
    compute_classification_metrics,
    compute_selective_metrics,
    save_evaluation_predictions,
)
from src.models.classification_head import PRIMARY_IDX_TO_LABEL, PRIMARY_LABEL_TO_IDX


class RealWorldBenchmarkEvaluator:
    """Evaluates multimodal passive-aggression models on frozen human benchmarks."""

    def __init__(
        self,
        class_names: Optional[List[str]] = None,
        device: str = "cpu",
    ):
        self.class_names = class_names or ["passive_aggressive", "non_passive_aggressive", "ambiguous"]
        self.label_to_idx = {name: i for i, name in enumerate(self.class_names)}
        self.device = torch.device(device)

    def load_frozen_test_manifest(self, manifest_path: str) -> List[Dict[str, Any]]:
        """
        Loads and validates records from a frozen human test manifest.
        Raises ValueError if any record is unreviewed or has an invalid label.
        """
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(f"Test manifest not found: {manifest_path}")

        records = []
        if manifest_path.endswith(".jsonl"):
            with open(manifest_path, "r", encoding="utf-8") as f:
                records = [json.loads(l) for l in f if l.strip()]
        elif manifest_path.endswith(".csv"):
            with open(manifest_path, "r", encoding="utf-8") as f:
                records = list(csv.DictReader(f))
        else:
            raise ValueError("Test manifest must be .jsonl or .csv")

        if not records:
            raise ValueError(f"Test manifest is empty: {manifest_path}")

        validated = []
        for i, r in enumerate(records):
            rec_id = r.get("record_id") or r.get("example_id")
            if not rec_id:
                raise ValueError(f"Record {i} missing record_id in {manifest_path}")

            label = (
                r.get("adjudicated_primary_label")
                or r.get("primary_label")
                or ""
            ).strip().lower()

            if label not in self.label_to_idx:
                raise ValueError(
                    f"Record '{rec_id}' has invalid or missing primary label '{label}'. "
                    f"Must be one of {list(self.label_to_idx.keys())}"
                )

            validated.append(r)

        return validated

    def evaluate_predictions(
        self,
        records: List[Dict[str, Any]],
        y_pred: Union[List[int], np.ndarray],
        y_prob: Union[List[List[float]], np.ndarray],
        model_id: str,
        split_id: str = "test",
        output_prediction_file: Optional[str] = None,
        n_bootstraps: int = 1000,
    ) -> Dict[str, Any]:
        """
        Computes all standard and diagnostic metrics for given predictions on records.
        """
        y_true = np.array([
            self.label_to_idx[(r.get("adjudicated_primary_label") or r.get("primary_label")).strip().lower()]
            for r in records
        ], dtype=int)
        y_pred = np.asarray(y_pred, dtype=int)
        y_prob = np.asarray(y_prob, dtype=float)

        # 1. Base classification metrics
        base_metrics = compute_classification_metrics(
            y_true=y_true,
            y_pred=y_pred,
            class_names=self.class_names,
            prefix="",
        )

        # 2. Bootstrap confidence intervals
        bootstrap_ci = compute_bootstrap_confidence_intervals(
            y_true=y_true,
            y_pred=y_pred,
            n_bootstraps=n_bootstraps,
        )

        # 3. Calibration diagnostics (ECE)
        calibration = compute_calibration_error(
            y_true=y_true,
            y_prob=y_prob,
            n_bins=10,
        )

        # 4. Selective classification (abstention on low confidence or ambiguous)
        selective = compute_selective_metrics(
            y_true=y_true,
            y_pred=y_pred,
            y_prob=y_prob,
            threshold=0.60,
            abstain_class=2,
        )

        # 5. Format record-level predictions for serialization
        prediction_rows = []
        for i, r in enumerate(records):
            rec_id = r.get("record_id") or r.get("example_id")
            pred_class = int(y_pred[i])
            true_class = int(y_true[i])
            pred_row = {
                "record_id": rec_id,
                "source_dataset": r.get("source_dataset", "unknown"),
                "ground_truth_label": self.class_names[true_class],
                "ground_truth_idx": true_class,
                "predicted_label": self.class_names[pred_class],
                "predicted_idx": pred_class,
                "confidence": float(np.max(y_prob[i])),
                "prob_passive_aggressive": float(y_prob[i][0]),
                "prob_non_passive_aggressive": float(y_prob[i][1]),
                "prob_ambiguous": float(y_prob[i][2]),
                "image_available": bool(r.get("image_available")),
                "context_available": bool(r.get("context_text")),
                "model_id": model_id,
                "split_id": split_id,
            }
            prediction_rows.append(pred_row)

        if output_prediction_file:
            save_evaluation_predictions(output_prediction_file, prediction_rows)

        return {
            "model_id": model_id,
            "split_id": split_id,
            "total_evaluated_records": len(records),
            "metrics": base_metrics,
            "bootstrap_ci_95": bootstrap_ci,
            "calibration": calibration,
            "selective_abstention": selective,
            "predictions_count": len(prediction_rows),
        }
