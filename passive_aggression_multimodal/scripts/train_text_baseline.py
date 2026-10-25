"""Train and Evaluate Traditional Text-Only Baselines (Phase 3 Task C).

Fits TF-IDF + Logistic Regression and TF-IDF + Linear SVM on training data only.
Selects best model via validation macro-F1 and evaluates on held-out test data.
Strictly separates primary passive-aggression benchmark from auxiliary sarcasm benchmark.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.metrics import compute_classification_metrics
from src.models.classification_head import (
    PRIMARY_LABEL_TO_IDX,
    PRIMARY_IDX_TO_LABEL,
    SARCASM_LABEL_TO_IDX,
    SARCASM_IDX_TO_LABEL,
)
from src.models.text_baseline import TextBaselineModel


def load_synthetic_primary_data(
    file_path: str = "data/synthetic/synthetic_pilot_350.jsonl",
    seed: int = 42,
) -> Tuple[List[str], List[int], List[str], List[int], List[str], List[int]]:
    """Loads and splits synthetic pilot data into 70/15/15 train/val/test."""
    texts = []
    labels = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            # Combine comment with context if available
            comment = r.get("target_comment") or ""
            context = r.get("conversation_context") or r.get("caption") or ""
            full_text = f"{context} {comment}".strip() if context else comment
            lbl_str = r.get("passive_aggression_label")
            if lbl_str in PRIMARY_LABEL_TO_IDX:
                texts.append(full_text)
                labels.append(PRIMARY_LABEL_TO_IDX[lbl_str])

    # Deterministic split with stratification
    np.random.seed(seed)
    n = len(texts)
    indices = np.random.permutation(n)
    n_train = int(0.70 * n)
    n_val = int(0.15 * n)

    train_idx = indices[:n_train]
    val_idx = indices[n_train : n_train + n_val]
    test_idx = indices[n_train + n_val :]

    train_texts = [texts[i] for i in train_idx]
    train_labels = [labels[i] for i in train_idx]
    val_texts = [texts[i] for i in val_idx]
    val_labels = [labels[i] for i in val_idx]
    test_texts = [texts[i] for i in test_idx]
    test_labels = [labels[i] for i in test_idx]

    return train_texts, train_labels, val_texts, val_labels, test_texts, test_labels


def run_primary_text_baseline(
    artifact_dir: Path,
    seed: int = 42,
) -> Dict[str, Any]:
    """Runs model selection and evaluation for primary passive-aggression benchmark."""
    print("\n" + "=" * 60)
    print("TASK C: Primary Task Baseline (Synthetic Benchmark 350)")
    print("=" * 60)

    train_texts, train_labels, val_texts, val_labels, test_texts, test_labels = (
        load_synthetic_primary_data(seed=seed)
    )
    print(f"Split sizes: Train={len(train_texts)}, Val={len(val_texts)}, Test={len(test_texts)}")

    class_names = [PRIMARY_IDX_TO_LABEL[i] for i in range(3)]
    candidate_configs = [
        ("logistic_regression", 0.5),
        ("logistic_regression", 1.0),
        ("logistic_regression", 5.0),
        ("linear_svm", 0.5),
        ("linear_svm", 1.0),
        ("linear_svm", 5.0),
    ]

    best_val_f1 = -1.0
    best_model = None
    best_config = None
    model_selection_results = []

    for model_type, c_param in candidate_configs:
        model = TextBaselineModel(
            model_type=model_type,
            c_param=c_param,
            class_weight="balanced",
            random_state=seed,
            class_mapping=PRIMARY_LABEL_TO_IDX,
        )
        model.fit(train_texts, train_labels)

        val_preds = model.predict(val_texts)
        val_metrics = compute_classification_metrics(val_labels, val_preds, class_names=class_names, prefix="val_")
        val_f1 = val_metrics["val_macro_f1"]

        model_selection_results.append({
            "model_type": model_type,
            "c_param": c_param,
            "val_macro_f1": val_f1,
            "val_accuracy": val_metrics["val_accuracy"],
        })
        print(f"  Config [{model_type:20s} C={c_param:4.1f}] -> Val Macro-F1: {val_f1:.4f}, Val Acc: {val_metrics['val_accuracy']:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_model = model
            best_config = (model_type, c_param)

    print(f"\n[*] Best Config Selected via Val F1: {best_config} with Val Macro-F1: {best_val_f1:.4f}")

    # Evaluate on held-out test set
    test_preds = best_model.predict(test_texts)
    test_probs = best_model.predict_proba(test_texts)
    test_metrics = compute_classification_metrics(test_labels, test_preds, class_names=class_names, prefix="test_")
    print(f"[+] Held-Out Test Evaluation:")
    print(f"    Test Macro-F1: {test_metrics['test_macro_f1']:.4f}")
    print(f"    Test Accuracy: {test_metrics['test_accuracy']:.4f}")
    print(f"    Test Balanced Accuracy: {test_metrics['test_balanced_accuracy']:.4f}")
    for cname, p_metrics in test_metrics["test_per_class"].items():
        print(f"    Class '{cname}': Precision={p_metrics['precision']:.3f}, Recall={p_metrics['recall']:.3f}, F1={p_metrics['f1']:.3f} (support={p_metrics['support']})")

    # Save artifacts
    save_path = artifact_dir / "best_text_baseline_primary.pkl"
    best_model.save(save_path, metadata={"config": best_config, "val_macro_f1": best_val_f1})

    return {
        "best_config": best_config,
        "best_val_macro_f1": best_val_f1,
        "model_selection": model_selection_results,
        "test_metrics": test_metrics,
        "model_path": str(save_path),
    }


def load_sarcasm_split(
    file_path: str,
    max_samples: Optional[int] = 5000,
) -> Tuple[List[str], List[int]]:
    """Loads sarcasm benchmark split with valid binary labels."""
    texts = []
    labels = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            lbl = r.get("original_label")
            if lbl == 1 or lbl == "1":
                idx = SARCASM_LABEL_TO_IDX["sarcastic"]
            elif lbl == 0 or lbl == "0":
                idx = SARCASM_LABEL_TO_IDX["non_sarcastic"]
            elif isinstance(lbl, dict) and "multi_label" in lbl:
                idx = SARCASM_LABEL_TO_IDX["sarcastic"] if str(lbl["multi_label"]) == "1" else SARCASM_LABEL_TO_IDX["non_sarcastic"]
            else:
                continue

            comment = r.get("target_comment") or ""
            context = r.get("caption") or ""
            full_text = f"{context} {comment}".strip() if context else comment
            texts.append(full_text)
            labels.append(idx)
            if max_samples and len(texts) >= max_samples:
                break
    return texts, labels


def run_sarcasm_text_baseline(
    artifact_dir: Path,
    seed: int = 42,
) -> Dict[str, Any]:
    """Runs text baseline on auxiliary sarcasm task (grouped splits)."""
    print("\n" + "=" * 60)
    print("TASK C: Auxiliary Sarcasm Task Baseline (Grouped Splits)")
    print("=" * 60)

    train_texts, train_labels = load_sarcasm_split("data/splits/grouped_train.jsonl", max_samples=5000)
    val_texts, val_labels = load_sarcasm_split("data/splits/grouped_validation.jsonl", max_samples=1000)
    test_texts, test_labels = load_sarcasm_split("data/splits/grouped_test.jsonl", max_samples=1000)
    print(f"Auxiliary Split sizes: Train={len(train_texts)}, Val={len(val_texts)}, Test={len(test_texts)}")

    class_names = ["sarcastic", "non_sarcastic"]
    model = TextBaselineModel(
        model_type="logistic_regression",
        c_param=1.0,
        class_weight="balanced",
        random_state=seed,
        class_mapping=SARCASM_LABEL_TO_IDX,
    )
    model.fit(train_texts, train_labels)

    val_preds = model.predict(val_texts)
    val_metrics = compute_classification_metrics(val_labels, val_preds, class_names=class_names, prefix="val_")
    print(f"  Auxiliary Val Macro-F1: {val_metrics['val_macro_f1']:.4f}, Val Acc: {val_metrics['val_accuracy']:.4f}")

    test_preds = model.predict(test_texts)
    test_metrics = compute_classification_metrics(test_labels, test_preds, class_names=class_names, prefix="test_")
    print(f"[+] Auxiliary Held-Out Test Evaluation:")
    print(f"    Test Macro-F1: {test_metrics['test_macro_f1']:.4f}, Test Acc: {test_metrics['test_accuracy']:.4f}")

    save_path = artifact_dir / "best_text_baseline_sarcasm.pkl"
    model.save(save_path, metadata={"task": "sarcasm", "test_metrics": test_metrics})

    return {
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "model_path": str(save_path),
    }


def main():
    artifact_dir = Path("artifacts/phase3/text_baseline")
    artifact_dir.mkdir(parents=True, exist_ok=True)

    primary_results = run_primary_text_baseline(artifact_dir)
    sarcasm_results = run_sarcasm_text_baseline(artifact_dir)

    all_results = {
        "primary_passive_aggression_benchmark": primary_results,
        "auxiliary_sarcasm_benchmark": sarcasm_results,
    }

    out_file = artifact_dir / "text_baseline_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[+] Baseline results saved to {out_file}")


if __name__ == "__main__":
    main()
