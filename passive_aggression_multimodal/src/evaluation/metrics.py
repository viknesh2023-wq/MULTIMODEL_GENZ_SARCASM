"""Comprehensive Evaluation Metrics for Passive-Aggression and Sarcasm Tasks.

Calculates Macro-F1, Macro-Precision, Macro-Recall, per-class metrics, confusion matrix,
accuracy, balanced accuracy, bootstrap confidence intervals, calibration diagnostics (ECE),
abstention/selective classification, and structured prediction export.
"""
from __future__ import annotations

import csv
import json
import os
from typing import Any, Dict, List, Optional, Union
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def compute_classification_metrics(
    y_true: Union[List[int], np.ndarray],
    y_pred: Union[List[int], np.ndarray],
    class_names: Optional[List[str]] = None,
    prefix: str = "",
) -> Dict[str, Any]:
    """
    Computes standard multi-class classification metrics.
    Args:
        y_true: Ground truth class integer indices.
        y_pred: Predicted class integer indices.
        class_names: Optional list of class names ordered by index.
        prefix: Prefix for metric dictionary keys (e.g. 'val_' or 'test_').
    Returns:
        Dictionary of computed metrics.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    if len(y_true) == 0:
        return {}

    num_classes = len(class_names) if class_names else max(int(np.max(y_true)), int(np.max(y_pred))) + 1
    labels = list(range(num_classes))

    acc = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    macro_prec = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    macro_rec = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    prec_per_class = precision_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    rec_per_class = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    f1_per_class = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels).tolist()

    per_class_dict = {}
    class_distribution = {}
    for idx in range(num_classes):
        cname = class_names[idx] if (class_names and idx < len(class_names)) else f"class_{idx}"
        supp = int(np.sum(y_true == idx))
        class_distribution[cname] = supp
        per_class_dict[cname] = {
            "precision": float(prec_per_class[idx]),
            "recall": float(rec_per_class[idx]),
            "f1": float(f1_per_class[idx]),
            "support": supp,
        }

    metrics = {
        f"{prefix}accuracy": acc,
        f"{prefix}balanced_accuracy": bal_acc,
        f"{prefix}macro_f1": macro_f1,
        f"{prefix}macro_precision": macro_prec,
        f"{prefix}macro_recall": macro_rec,
        f"{prefix}weighted_f1": weighted_f1,
        f"{prefix}per_class": per_class_dict,
        f"{prefix}class_distribution": class_distribution,
        f"{prefix}confusion_matrix": cm,
        f"{prefix}total_samples": len(y_true),
    }
    return metrics


def compute_bootstrap_confidence_intervals(
    y_true: Union[List[int], np.ndarray],
    y_pred: Union[List[int], np.ndarray],
    n_bootstraps: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> Dict[str, Dict[str, float]]:
    """
    Computes non-parametric bootstrap percentile confidence intervals for classification metrics.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    n = len(y_true)
    if n == 0:
        return {}

    rng = np.random.default_rng(seed)
    f1_scores = []
    acc_scores = []
    bal_acc_scores = []

    for _ in range(n_bootstraps):
        indices = rng.integers(0, n, size=n)
        yt_sample = y_true[indices]
        yp_sample = y_pred[indices]

        acc_scores.append(accuracy_score(yt_sample, yp_sample))
        bal_acc_scores.append(balanced_accuracy_score(yt_sample, yp_sample))
        f1_scores.append(f1_score(yt_sample, yp_sample, average="macro", zero_division=0))

    alpha = (1.0 - confidence_level) / 2.0
    lower_pct = alpha * 100
    upper_pct = (1.0 - alpha) * 100

    return {
        "macro_f1": {
            "lower": float(np.percentile(f1_scores, lower_pct)),
            "upper": float(np.percentile(f1_scores, upper_pct)),
            "mean": float(np.mean(f1_scores)),
        },
        "accuracy": {
            "lower": float(np.percentile(acc_scores, lower_pct)),
            "upper": float(np.percentile(acc_scores, upper_pct)),
            "mean": float(np.mean(acc_scores)),
        },
        "balanced_accuracy": {
            "lower": float(np.percentile(bal_acc_scores, lower_pct)),
            "upper": float(np.percentile(bal_acc_scores, upper_pct)),
            "mean": float(np.mean(bal_acc_scores)),
        },
    }


def compute_calibration_error(
    y_true: Union[List[int], np.ndarray],
    y_prob: Union[List[List[float]], np.ndarray],
    n_bins: int = 10,
) -> Dict[str, Any]:
    """
    Computes Expected Calibration Error (ECE) and Maximum Calibration Error (MCE).
    """
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    if len(y_true) == 0:
        return {"ece": 0.0, "mce": 0.0, "bins": []}

    confidences = np.max(y_prob, axis=1)
    predictions = np.argmax(y_prob, axis=1)
    accuracies = predictions == y_true

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    mce = 0.0
    bins_data = []

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper) if i > 0 else (confidences >= bin_lower) & (confidences <= bin_upper)
        prop_in_bin = float(np.mean(in_bin))

        if prop_in_bin > 0:
            acc_in_bin = float(np.mean(accuracies[in_bin]))
            conf_in_bin = float(np.mean(confidences[in_bin]))
            abs_diff = abs(acc_in_bin - conf_in_bin)
            ece += abs_diff * prop_in_bin
            mce = max(mce, abs_diff)
            bins_data.append({
                "bin_lower": float(bin_lower),
                "bin_upper": float(bin_upper),
                "accuracy": acc_in_bin,
                "confidence": conf_in_bin,
                "count": int(np.sum(in_bin)),
            })
        else:
            bins_data.append({
                "bin_lower": float(bin_lower),
                "bin_upper": float(bin_upper),
                "accuracy": 0.0,
                "confidence": 0.0,
                "count": 0,
            })

    return {
        "ece": float(ece),
        "mce": float(mce),
        "bins": bins_data,
    }


def compute_selective_metrics(
    y_true: Union[List[int], np.ndarray],
    y_pred: Union[List[int], np.ndarray],
    y_prob: Union[List[List[float]], np.ndarray],
    threshold: float = 0.5,
    abstain_class: Optional[int] = 2,
) -> Dict[str, Any]:
    """
    Computes selective classification metrics under confidence thresholding or explicit abstention.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    n = len(y_true)
    if n == 0:
        return {}

    confidences = np.max(y_prob, axis=1)
    # Mask of examples the model commits to
    covered_mask = confidences >= threshold
    if abstain_class is not None:
        # Also treat predicted ambiguous class as non-committed if evaluating binary decisions
        covered_mask = covered_mask & (y_pred != abstain_class)

    coverage = float(np.mean(covered_mask))
    if np.sum(covered_mask) == 0:
        return {
            "coverage": 0.0,
            "selective_macro_f1": 0.0,
            "selective_accuracy": 0.0,
            "covered_samples": 0,
            "total_samples": n,
        }

    sel_true = y_true[covered_mask]
    sel_pred = y_pred[covered_mask]

    return {
        "coverage": coverage,
        "selective_macro_f1": float(f1_score(sel_true, sel_pred, average="macro", zero_division=0)),
        "selective_accuracy": float(accuracy_score(sel_true, sel_pred)),
        "covered_samples": int(np.sum(covered_mask)),
        "total_samples": n,
    }


def save_evaluation_predictions(
    filepath: str,
    records: List[Dict[str, Any]],
) -> None:
    """
    Saves predictions as a versioned JSONL or CSV file with metadata.
    """
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    if filepath.endswith(".jsonl"):
        with open(filepath, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    elif filepath.endswith(".csv"):
        if not records:
            return
        fieldnames = list(records[0].keys())
        with open(filepath, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(records)
    else:
        raise ValueError(f"Unsupported format: {filepath}. Use .csv or .jsonl")
