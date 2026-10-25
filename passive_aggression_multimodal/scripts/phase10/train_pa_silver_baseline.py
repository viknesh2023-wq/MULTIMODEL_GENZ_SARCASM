"""
scripts/phase10/train_pa_silver_baseline.py

Phase 10: Exploratory Passive-Aggression Silver Baseline Model.
Trains a TF-IDF (word + char n-grams) with regularized Logistic Regression classifier
exclusively on accepted LLM-annotated silver labels.

Governance and Integrity:
- Model Identifier: pa_silver_tfidf_v1 (explicit silver attribution)
- Trained on: data/splits/phase10/silver_train.csv
- Tuned on: data/splits/phase10/silver_validation.csv
- Evaluated on: data/splits/phase10/silver_test.csv (diagnostic test set)
- Provenance tag: LLM_SILVER
- Primary Benchmark Gate: BLOCKED_PENDING_HUMAN_ANNOTATION (preserved)
- Silver Evaluation Gate: LLM_SILVER_DIAGNOSTIC_EVALUATION_AVAILABLE
- Generates:
  - artifacts/phase10/pa_silver_tfidf_v1.pkl
  - artifacts/phase10/pa_silver_metadata.json
  - reports/phase10/silver_model_evaluation_report.md
  - reports/phase10/tables/evaluation_metrics.json
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import pickle
import time
from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import FeatureUnion, Pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Phase10-TrainSilverBaseline")

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
SPLITS_DIR = ROOT_DIR / "data/splits/phase10"
ARTIFACTS_DIR = ROOT_DIR / "artifacts/phase10"
REPORTS_DIR = ROOT_DIR / "reports/phase10"
TABLES_DIR = REPORTS_DIR / "tables"

MODEL_ID = "pa_silver_tfidf_v1"
CLASS_MAPPING = {"non_passive_aggressive": 0, "passive_aggressive": 1}
REVERSE_CLASS_MAPPING = {0: "non_passive_aggressive", 1: "passive_aggressive"}


def load_split_csv(path: Path) -> Tuple[List[str], List[int], List[Dict[str, Any]]]:
    texts = []
    labels = []
    metadata = []

    if not path.exists():
        raise FileNotFoundError(f"Split file missing: {path}")

    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            lbl_str = row.get("silver_label", "").strip()
            if lbl_str not in CLASS_MAPPING:
                continue
            text = row.get("text", "").strip()
            ctx = row.get("dialogue_context", "").strip()
            # If dialogue context is available, append context to comment for enrichment
            full_text = f"Context: {ctx} Comment: {text}" if ctx else text

            texts.append(full_text)
            labels.append(CLASS_MAPPING[lbl_str])
            metadata.append(row)

    return texts, labels, metadata


def build_pipeline(c_val: float = 1.0) -> Pipeline:
    word_vec = TfidfVectorizer(
        ngram_range=(1, 3),
        max_features=5000,
        sublinear_tf=True,
        strip_accents="unicode",
    )
    char_vec = TfidfVectorizer(
        ngram_range=(2, 5),
        analyzer="char_wb",
        max_features=5000,
        sublinear_tf=True,
    )
    union = FeatureUnion([
        ("word_tfidf", word_vec),
        ("char_tfidf", char_vec),
    ])
    clf = LogisticRegression(
        C=c_val,
        class_weight="balanced",
        max_iter=1000,
        random_state=42,
    )
    return Pipeline([("features", union), ("classifier", clf)])


def run_training_and_evaluation() -> Dict[str, Any]:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    train_path = SPLITS_DIR / "silver_train.csv"
    val_path = SPLITS_DIR / "silver_validation.csv"
    test_path = SPLITS_DIR / "silver_test.csv"

    logger.info("Loading splits...")
    X_train, y_train, meta_train = load_split_csv(train_path)
    X_val, y_val, meta_val = load_split_csv(val_path)
    X_test, y_test, meta_test = load_split_csv(test_path)

    logger.info("Loaded Train: %d, Val: %d, Test: %d", len(X_train), len(X_val), len(X_test))

    # Check minimum class representation
    for name, y in [("Train", y_train), ("Val", y_val), ("Test", y_test)]:
        c0 = y.count(0)
        c1 = y.count(1)
        logger.info("%s set class breakdown: Non-PA=%d, PA=%d", name, c0, c1)

    t0 = time.time()

    # Hyperparameter tuning on validation split
    best_c = 1.0
    best_val_f1 = -1.0
    best_model = None

    for c in [0.1, 0.5, 1.0, 2.0, 5.0]:
        pipe = build_pipeline(c_val=c)
        pipe.fit(X_train, y_train)
        if len(X_val) > 0 and len(set(y_val)) > 1:
            val_preds = pipe.predict(X_val)
            f1 = f1_score(y_val, val_preds, average="macro", zero_division=0)
            if f1 > best_val_f1:
                best_val_f1 = f1
                best_c = c
                best_model = pipe
        else:
            best_model = pipe
            best_c = 1.0
            break

    train_duration = round(time.time() - t0, 3)
    logger.info("Best C parameter: %s (Val Macro-F1: %.4f)", best_c, best_val_f1)

    # Final Diagnostic Test Evaluation
    y_test_pred = best_model.predict(X_test)
    y_test_proba = best_model.predict_proba(X_test)[:, 1] if hasattr(best_model, "predict_proba") else []

    acc = accuracy_score(y_test, y_test_pred)
    bal_acc = balanced_accuracy_score(y_test, y_test_pred)
    macro_f1 = f1_score(y_test, y_test_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_test, y_test_pred, average="weighted", zero_division=0)

    p_pa = precision_score(y_test, y_test_pred, pos_label=1, zero_division=0)
    r_pa = recall_score(y_test, y_test_pred, pos_label=1, zero_division=0)
    f1_pa = f1_score(y_test, y_test_pred, pos_label=1, zero_division=0)

    p_non = precision_score(y_test, y_test_pred, pos_label=0, zero_division=0)
    r_non = recall_score(y_test, y_test_pred, pos_label=0, zero_division=0)
    f1_non = f1_score(y_test, y_test_pred, pos_label=0, zero_division=0)

    cm = confusion_matrix(y_test, y_test_pred, labels=[0, 1]).tolist()

    # Save Checkpoint
    checkpoint_file = ARTIFACTS_DIR / f"{MODEL_ID}.pkl"
    with open(checkpoint_file, "wb") as f:
        pickle.dump(best_model, f)

    # Save Model Metadata
    metadata = {
        "model_id": MODEL_ID,
        "algorithm": "TF-IDF (Word 1-3, Char 2-5) + Logistic Regression",
        "training_timestamp": datetime.now(timezone.utc).isoformat(),
        "train_duration_sec": train_duration,
        "best_hyperparameters": {"C": best_c, "class_weight": "balanced", "random_state": 42},
        "class_mapping": CLASS_MAPPING,
        "dataset_provenance": "LLM_SILVER",
        "sample_sizes": {
            "train": len(X_train),
            "validation": len(X_val),
            "diagnostic_test": len(X_test),
        },
        "diagnostic_test_metrics": {
            "accuracy": round(acc, 4),
            "balanced_accuracy": round(bal_acc, 4),
            "macro_f1": round(macro_f1, 4),
            "weighted_f1": round(weighted_f1, 4),
            "passive_aggressive": {"precision": round(p_pa, 4), "recall": round(r_pa, 4), "f1": round(f1_pa, 4), "support": y_test.count(1)},
            "non_passive_aggressive": {"precision": round(p_non, 4), "recall": round(r_non, 4), "f1": round(f1_non, 4), "support": y_test.count(0)},
            "confusion_matrix": cm,
        },
        "primary_benchmark_gate": "BLOCKED_PENDING_HUMAN_ANNOTATION",
        "silver_evaluation_gate": "LLM_SILVER_DIAGNOSTIC_EVALUATION_AVAILABLE",
        "disclaimer": "Metrics represent concordance with automated LLM silver labels, NOT human ground truth.",
    }

    meta_file = ARTIFACTS_DIR / "pa_silver_metadata.json"
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    # Save metrics JSON table
    metrics_json = TABLES_DIR / "evaluation_metrics.json"
    with open(metrics_json, "w", encoding="utf-8") as f:
        json.dump(metadata["diagnostic_test_metrics"], f, indent=2)

    # Generate Markdown Evaluation Report
    eval_report = REPORTS_DIR / "silver_model_evaluation_report.md"
    md_content = f"""# Phase 10: Exploratory Silver-Label Model Training and Diagnostic Evaluation Report

**Model ID:** `{MODEL_ID}`  
**Architecture:** TF-IDF (Word n-grams 1-3, Character n-grams 2-5) + Logistic Regression (`C={best_c}`)  
**Training Date:** {logging.Formatter().formatTime(logging.LogRecord('', 0, '', 0, '', (), None))}  
**Label Provenance:** `LLM_SILVER` (Exploratory Silver Dataset)  
**Human Evaluation Gate:** `BLOCKED_PENDING_HUMAN_ANNOTATION`  
**Silver Evaluation Gate:** `LLM_SILVER_DIAGNOSTIC_EVALUATION_AVAILABLE`  

---

## 1. Executive Summary

This report documents the training and diagnostic evaluation of an exploratory baseline classifier (`{MODEL_ID}`) trained exclusively on accepted LLM-annotated silver labels.

> [!WARNING]
> **CRITICAL SCIENTIFIC GOVERNANCE NOTICE:**  
> The metrics reported below measure empirical concordance with automated LLM silver consensus labels. They DO NOT represent accuracy against human-verified ground truth. The primary benchmark status remains strictly locked at `BLOCKED_PENDING_HUMAN_ANNOTATION`.

---

## 2. Diagnostic Test Set Performance

Evaluated on the leakage-resistant diagnostic test split (`data/splits/phase10/silver_test.csv`):

| Metric | Score | Note |
| :--- | :--- | :--- |
| **Accuracy** | **{round(acc * 100, 2)}%** | Raw diagnostic test accuracy |
| **Balanced Accuracy** | **{round(bal_acc * 100, 2)}%** | Mean of recall across classes |
| **Macro-F1** | **{round(macro_f1, 4)}** | Unweighted class average |
| **Weighted-F1** | **{round(weighted_f1, 4)}** | Frequency-weighted class average |
| **Training Duration** | **{train_duration}s** | Fast CPU/GPU vectorized training |

### Per-Class Performance Breakdown

| Class | Precision | Recall | F1-Score | Support ($N$) |
| :--- | :--- | :--- | :--- | :--- |
| **`non_passive_aggressive`** | {round(p_non, 4)} | {round(r_non, 4)} | {round(f1_non, 4)} | {y_test.count(0)} |
| **`passive_aggressive`** | {round(p_pa, 4)} | {round(r_pa, 4)} | {round(f1_pa, 4)} | {y_test.count(1)} |

### Confusion Matrix

| Ground Truth (Silver) \\ Predicted | Predicted `non_passive_aggressive` | Predicted `passive_aggressive` |
| :--- | :--- | :--- |
| **Actual `non_passive_aggressive`** | **{cm[0][0]}** (TN) | {cm[0][1]} (FP) |
| **Actual `passive_aggressive`** | {cm[1][0]} (FN) | **{cm[1][1]}** (TP) |

---

## 3. Provenance and Artifact Registration

- **Trained Model Checkpoint:** `{str(checkpoint_file.relative_to(ROOT_DIR))}`
- **Metadata Registration:** `{str(meta_file.relative_to(ROOT_DIR))}`
- **Metrics Table:** `{str(metrics_json.relative_to(ROOT_DIR))}`
- **Human Response Gate:** Untouched (`annotator_1_responses.csv`, `annotator_2_responses.csv` remain zero completed).
"""
    with open(eval_report, "w", encoding="utf-8") as f:
        f.write(md_content)

    logger.info("Exploratory silver baseline trained and evaluated successfully.")
    return metadata


if __name__ == "__main__":
    run_training_and_evaluation()
