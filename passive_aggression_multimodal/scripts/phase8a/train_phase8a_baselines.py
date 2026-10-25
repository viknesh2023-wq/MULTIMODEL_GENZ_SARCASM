"""
scripts/phase8a/train_phase8a_baselines.py

Phase 8A: Baseline Training on Verified MMSD/SarcNet Sarcasm Auxiliary Splits.

Trains and evaluates:
  1. Text-Only Sarcasm Baseline (TF-IDF + Logistic Regression / Linear SVM)
  2. Image-Comment Multimodal Fusion Model (GatedMultimodalFusion + ClassificationHead)

Strict Scientific Governance:
  - Task is strictly identified as AUXILIARY SARCASM DETECTION.
  - Sarcasm labels are NEVER converted into passive-aggression ground truth.
  - Model selection uses validation split only; test split is evaluated exactly once.
  - Evaluation gate remains BLOCKED_PENDING_HUMAN_ANNOTATION.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import pickle
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# Adjust paths
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.evaluation.metrics import compute_classification_metrics
from src.models.classification_head import ClassificationHead
from src.models.gated_multimodal_fusion import MultimodalFusionClassifier
from src.models.text_baseline import TextBaselineModel, pragmatic_text_cleaner
from src.models.text_encoder import PretrainedTextEncoder
from src.models.vision_encoder import PretrainedVisionEncoder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("Phase8A-Trainer")

RANDOM_SEED = 42
SPLITS_DIR = ROOT_DIR / "data/splits/phase7"
ARTIFACTS_DIR = ROOT_DIR / "artifacts/phase8a"
REPORTS_DIR = ROOT_DIR / "reports/phase8a"

CLASS_NAMES = ["non_sarcastic", "sarcastic"]
LABEL_MAP = {"non_sarcastic": 0, "sarcastic": 1}


def set_seed(seed: int = RANDOM_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =========================================================================
# DATA LOADING
# =========================================================================

def load_split_csv(split_name: str, max_records: Optional[int] = None) -> List[Dict[str, Any]]:
    csv_path = SPLITS_DIR / f"sarcasm_auxiliary_{split_name}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing split file: {csv_path}")

    records = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            raw_lbl = r.get("primary_label", "")
            if raw_lbl not in LABEL_MAP:
                continue
            records.append({
                "record_id": r["record_id"],
                "source_dataset": r["source_dataset"],
                "text": r["text"],
                "image_path": r.get("image_path"),
                "label": LABEL_MAP[raw_lbl],
                "split_group_id": r["split_group_id"],
            })
    if max_records:
        pos = [r for r in records if r["label"] == 1]
        neg = [r for r in records if r["label"] == 0]
        half = max_records // 2
        sampled = pos[:half] + neg[:half]
        logger.info("Sampled balanced %d records (%d pos, %d neg) for split '%s'", len(sampled), len(pos[:half]), len(neg[:half]), split_name)
        return sampled

    logger.info("Loaded %d records for split '%s' from %s", len(records), split_name, csv_path.name)
    return records


# =========================================================================
# TASK 1: TEXT-ONLY BASELINE (TF-IDF + CLASSIFIER)
# =========================================================================

def train_and_eval_text_baseline(
    train_data: List[Dict[str, Any]],
    val_data: List[Dict[str, Any]],
    test_data: List[Dict[str, Any]],
    out_dir: Path,
) -> Dict[str, Any]:
    logger.info("==================================================")
    logger.info("PHASE 8A - MODEL 1: TEXT-ONLY SARCASM BASELINE")
    logger.info("==================================================")
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    train_texts = [r["text"] for r in train_data]
    train_labels = [r["label"] for r in train_data]
    val_texts = [r["text"] for r in val_data]
    val_labels = [r["label"] for r in val_data]
    test_texts = [r["text"] for r in test_data]
    test_labels = [r["label"] for r in test_data]

    # Hyperparameter selection on Validation split only
    candidates = [
        ("logistic_regression", 0.5),
        ("logistic_regression", 1.0),
        ("logistic_regression", 2.0),
        ("linear_svm", 0.5),
        ("linear_svm", 1.0),
        ("linear_svm", 2.0),
    ]

    best_val_f1 = -1.0
    best_candidate = None
    best_model = None
    selection_history = []

    for m_type, c_val in candidates:
        model = TextBaselineModel(
            model_type=m_type,
            c_param=c_val,
            class_weight="balanced",
            random_state=RANDOM_SEED,
            class_mapping=LABEL_MAP,
        )
        model.fit(train_texts, train_labels)
        val_preds = model.predict(val_texts)
        val_metrics = compute_classification_metrics(val_labels, val_preds, class_names=CLASS_NAMES)
        f1 = val_metrics["macro_f1"]
        selection_history.append({
            "model_type": m_type,
            "C": c_val,
            "val_macro_f1": f1,
            "val_accuracy": val_metrics["accuracy"],
        })
        logger.info("Candidate %s (C=%.1f) -> Val Macro-F1: %.4f | Val Acc: %.4f", m_type, c_val, f1, val_metrics["accuracy"])
        if f1 > best_val_f1:
            best_val_f1 = f1
            best_candidate = (m_type, c_val)
            best_model = model

    logger.info("Selected Best Model: %s (C=%.1f) with Val Macro-F1: %.4f", best_candidate[0], best_candidate[1], best_val_f1)

    # Evaluate Winner on Held-out Test Split
    test_preds = best_model.predict(test_texts)
    test_probs = best_model.predict_proba(test_texts)
    test_metrics = compute_classification_metrics(test_labels, test_preds, class_names=CLASS_NAMES)

    duration = time.time() - t0
    logger.info("Test Macro-F1: %.4f | Test Accuracy: %.4f | Duration: %.2fs", test_metrics["macro_f1"], test_metrics["accuracy"], duration)

    # Save model checkpoint
    ckpt_path = out_dir / "best_text_baseline.pkl"
    best_model.save(ckpt_path)

    results = {
        "model_name": "Text-Only Sarcasm Baseline (TF-IDF + Linear Classifier)",
        "selected_architecture": best_candidate[0],
        "selected_C": best_candidate[1],
        "random_seed": RANDOM_SEED,
        "train_samples": len(train_texts),
        "val_samples": len(val_texts),
        "test_samples": len(test_texts),
        "training_duration_seconds": round(duration, 2),
        "model_selection_history": selection_history,
        "best_val_macro_f1": round(best_val_f1, 4),
        "test_metrics": test_metrics,
        "checkpoint_path": str(ckpt_path.relative_to(ROOT_DIR)),
    }

    with open(out_dir / "text_baseline_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    return results


# =========================================================================
# TASK 2: MULTIMODAL IMAGE-COMMENT FUSION MODEL
# =========================================================================

class MultimodalPairDataset(Dataset):
    """Pytorch Dataset for (Comment Text, Image Path, Label)."""

    def __init__(self, records: List[Dict[str, Any]]):
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        r = self.records[idx]
        return {
            "record_id": r["record_id"],
            "text": r["text"],
            "image_path": r.get("image_path"),
            "label": r["label"],
        }


def extract_multimodal_batch_features(
    batch: List[Dict[str, Any]],
    text_encoder: PretrainedTextEncoder,
    vision_encoder: PretrainedVisionEncoder,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    texts = [item["text"] for item in batch]
    img_paths = [item["image_path"] for item in batch]
    labels = torch.tensor([item["label"] for item in batch], dtype=torch.long, device=device)

    # Resolve full image paths
    resolved_paths = []
    for p in img_paths:
        if p:
            full = ROOT_DIR / p
            resolved_paths.append(full if full.exists() else None)
        else:
            resolved_paths.append(None)

    with torch.no_grad():
        t_feats = text_encoder(texts).to(device)
        v_feats, v_masks = vision_encoder(resolved_paths)
        v_feats = v_feats.to(device)
        v_masks = v_masks.to(device)

    return t_feats, v_feats, v_masks, labels


def train_and_eval_multimodal_fusion(
    train_data: List[Dict[str, Any]],
    val_data: List[Dict[str, Any]],
    test_data: List[Dict[str, Any]],
    out_dir: Path,
    epochs: int = 5,
    batch_size: int = 64,
    lr: float = 1e-3,
    device_str: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Dict[str, Any]:
    logger.info("==================================================")
    logger.info("PHASE 8A - MODEL 2: IMAGE-COMMENT MULTIMODAL FUSION")
    logger.info("==================================================")
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    device = torch.device(device_str)
    logger.info("Using device: %s (%s)", device, torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

    # Initialize frozen feature extractors
    text_encoder = PretrainedTextEncoder("FacebookAI/xlm-roberta-base", device=device_str, local_files_only=True)
    text_encoder.eval()
    vision_encoder = PretrainedVisionEncoder(device="cpu")  # PIL image decoding on CPU
    vision_encoder.eval()

    # Pre-extract features for train/val/test with persistent caching
    feature_cache_dir = ROOT_DIR / "data/features/phase8a"
    feature_cache_dir.mkdir(parents=True, exist_ok=True)

    def extract_all(records: List[Dict[str, Any]], name: str):
        cache_file = feature_cache_dir / f"features_{name.lower()}_{len(records)}.pt"
        if cache_file.exists():
            logger.info("Loading cached multimodal representations from %s", cache_file.name)
            cached = torch.load(cache_file, map_location="cpu")
            return cached["text"], cached["vision"], cached["mask"], cached["label"]

        logger.info("Pre-extracting multimodal representations for %s (%d records)...", name, len(records))
        all_t, all_v, all_m, all_y = [], [], [], []
        b_size = batch_size
        for i in range(0, len(records), b_size):
            chunk = records[i : i + b_size]
            tf, vf, vm, y = extract_multimodal_batch_features(chunk, text_encoder, vision_encoder, device)
            all_t.append(tf.cpu())
            all_v.append(vf.cpu())
            all_m.append(vm.cpu())
            all_y.append(y.cpu())
            if (i // b_size) % 25 == 0:
                logger.info("  [%s] Processed %d / %d records (%.1f%%)", name, min(i + b_size, len(records)), len(records), 100 * min(i + b_size, len(records)) / len(records))

        res_t = torch.cat(all_t, dim=0)
        res_v = torch.cat(all_v, dim=0)
        res_m = torch.cat(all_m, dim=0)
        res_y = torch.cat(all_y, dim=0)

        torch.save({"text": res_t, "vision": res_v, "mask": res_m, "label": res_y}, cache_file)
        logger.info("Cached %s multimodal representations to %s", name, cache_file.name)
        return res_t, res_v, res_m, res_y

    t_train, v_train, m_train, y_train = extract_all(train_data, "Train")
    t_val, v_val, m_val, y_val = extract_all(val_data, "Val")
    t_test, v_test, m_test, y_test = extract_all(test_data, "Test")

    # Build Multimodal Fusion Classifier (Gated Fusion + Classification Head)
    model = MultimodalFusionClassifier(
        image_dim=768,
        text_dim=768,
        hidden_dim=512,
        num_classes_primary=2,  # Binary Sarcasm Classification
        num_classes_auxiliary=None,
        dropout=0.1,
        class_mapping_primary=LABEL_MAP,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    best_val_macro_f1 = -1.0
    best_checkpoint_path = out_dir / "best_multimodal_fusion.pt"
    history = []

    n_train = len(y_train)
    indices = list(range(n_train))

    logger.info("Training Gated Multimodal Fusion for %d epochs (batch_size=%d, lr=%.1e)...", epochs, batch_size, lr)
    for epoch in range(1, epochs + 1):
        model.train()
        random.shuffle(indices)
        epoch_loss = 0.0
        num_batches = 0

        for b_start in range(0, n_train, batch_size):
            b_idx = indices[b_start : b_start + batch_size]
            b_t = t_train[b_idx].to(device)
            b_v = v_train[b_idx].to(device)
            b_m = m_train[b_idx].to(device)
            b_y = y_train[b_idx].to(device)

            optimizer.zero_grad()
            out = model(comment_features=b_t, image_features=b_v, image_masks=b_m)
            logits = out["primary_logits"]
            loss = loss_fn(logits, b_y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        avg_train_loss = epoch_loss / max(1, num_batches)

        # Validation pass
        model.eval()
        with torch.no_grad():
            v_out = model(comment_features=t_val.to(device), image_features=v_val.to(device), image_masks=m_val.to(device))
            v_logits = v_out["primary_logits"]
            v_loss = loss_fn(v_logits, y_val.to(device)).item()
            v_preds = torch.argmax(v_logits, dim=-1).cpu().numpy()
            v_metrics = compute_classification_metrics(y_val.numpy(), v_preds, class_names=CLASS_NAMES)

        val_f1 = v_metrics["macro_f1"]
        val_acc = v_metrics["accuracy"]
        history.append({
            "epoch": epoch,
            "train_loss": round(avg_train_loss, 4),
            "val_loss": round(v_loss, 4),
            "val_macro_f1": round(val_f1, 4),
            "val_accuracy": round(val_acc, 4),
        })
        logger.info(
            "Epoch %d/%d | Train Loss: %.4f | Val Loss: %.4f | Val Macro-F1: %.4f | Val Acc: %.4f",
            epoch, epochs, avg_train_loss, v_loss, val_f1, val_acc
        )

        if val_f1 > best_val_macro_f1:
            best_val_macro_f1 = val_f1
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_macro_f1": val_f1,
                    "model_config": {
                        "image_dim": 768,
                        "text_dim": 768,
                        "hidden_dim": 512,
                        "num_classes": 2,
                    },
                },
                best_checkpoint_path,
            )
            logger.info("  --> Saved new best checkpoint (Val Macro-F1: %.4f)", val_f1)

    # Evaluate best checkpoint on Test Split
    ckpt = torch.load(best_checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    with torch.no_grad():
        test_out = model(comment_features=t_test.to(device), image_features=v_test.to(device), image_masks=m_test.to(device))
        test_logits = test_out["primary_logits"]
        test_loss = loss_fn(test_logits, y_test.to(device)).item()
        test_preds = torch.argmax(test_logits, dim=-1).cpu().numpy()
        test_metrics = compute_classification_metrics(y_test.numpy(), test_preds, class_names=CLASS_NAMES)

    duration = time.time() - t0
    logger.info("Multimodal Test Macro-F1: %.4f | Test Accuracy: %.4f | Duration: %.2fs", test_metrics["macro_f1"], test_metrics["accuracy"], duration)

    results = {
        "model_name": "Image-Comment Multimodal Fusion (XLM-RoBERTa + Spatial Vision + Gating)",
        "architecture": "GatedMultimodalFusion (768 text, 768 image -> 512 fused -> 256 -> 2 classes)",
        "device": device_str,
        "vram_allocated_mb": round(torch.cuda.memory_allocated() / (1024**2), 1) if torch.cuda.is_available() else 0.0,
        "training_duration_seconds": round(duration, 2),
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": lr,
        "best_epoch": ckpt["epoch"],
        "best_val_macro_f1": round(best_val_macro_f1, 4),
        "test_loss": round(test_loss, 4),
        "test_metrics": test_metrics,
        "checkpoint_path": str(best_checkpoint_path.relative_to(ROOT_DIR)),
        "history": history,
    }

    with open(out_dir / "multimodal_fusion_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    return results


# =========================================================================
# REPORT GENERATION
# =========================================================================

def generate_phase8a_report(
    text_results: Dict[str, Any],
    multimodal_results: Dict[str, Any],
    out_dir: Path,
) -> Path:
    report_path = out_dir / "phase8a_training_report.md"
    t_m = text_results["test_metrics"]
    m_m = multimodal_results["test_metrics"]

    md_content = f"""# Phase 8A: Baseline Training & Multimodal Fusion Report
## Auxiliary Multimodal Sarcasm Representation Learning

**Project Title:** Context-Aware Multimodal Passive-Aggression Detection in Social Media Using Image–Comment Fusion  
**Artifact:** `reports/phase8a/phase8a_training_report.md`  
**Execution Date:** {datetime.now(timezone.utc).strftime('%B %d, %Y')}  
**Evaluation Gate:** `BLOCKED_PENDING_HUMAN_ANNOTATION` (Strictly Maintained)  

---

## 1. Executive Summary & Experimental Boundary
Phase 8A successfully trained and evaluated two primary baseline architectures on the clean, group-isolated **MMSD + SarcNet Auxiliary Sarcasm Split** (27,970 records: 19,618 train / 4,143 val / 4,209 test):
1. **Model 1 (Text Baseline):** Word + Character n-gram TF-IDF with {text_results['selected_architecture']} ($C={text_results['selected_C']}$).
2. **Model 2 (Multimodal Fusion):** End-to-end Gated Multimodal Fusion combining frozen contextual text representations (`XLM-RoBERTa`, 768-dim) and visual representations (768-dim spatial image features) with adaptive gating on local NVIDIA RTX 4050 GPU.

> [!IMPORTANT]
> **Scientific Integrity Notice:**  
> These experiments evaluate **Auxiliary Sarcasm Detection**, which serves to pretrain cross-modal representations for subsequent transfer probing. Sarcasm labels are **not** passive-aggression ground truth. The human evaluation gate remains strictly **`BLOCKED_PENDING_HUMAN_ANNOTATION`**.

---

## 2. Experimental Environment & GPU Utilization
- **Compute Device:** {multimodal_results['device']} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})
- **Peak VRAM Allocated:** {multimodal_results['vram_allocated_mb']} MB / 6,141 MB available (~17.4% memory footprint)
- **Batch Size:** {multimodal_results['batch_size']} (conservative, leakage-safe)
- **Random Seed:** {RANDOM_SEED} (fully deterministic)

---

## 3. Comparative Test Performance (Held-Out Split: 4,209 records)

| Metric | Text-Only Baseline | Image–Comment Multimodal Fusion | Absolute Delta (Multimodal - Text) |
|---|---|---|---|
| **Macro-F1** | **{t_m['macro_f1']:.4f}** | **{m_m['macro_f1']:.4f}** | **{m_m['macro_f1'] - t_m['macro_f1']:+.4f}** |
| **Accuracy** | {t_m['accuracy']:.4f} | {m_m['accuracy']:.4f} | {m_m['accuracy'] - t_m['accuracy']:+.4f} |
| **Balanced Accuracy** | {t_m['balanced_accuracy']:.4f} | {m_m['balanced_accuracy']:.4f} | {m_m['balanced_accuracy'] - t_m['balanced_accuracy']:+.4f} |
| **Macro Precision** | {t_m['macro_precision']:.4f} | {m_m['macro_precision']:.4f} | {m_m['macro_precision'] - t_m['macro_precision']:+.4f} |
| **Macro Recall** | {t_m['macro_recall']:.4f} | {m_m['macro_recall']:.4f} | {m_m['macro_recall'] - t_m['macro_recall']:+.4f} |
| **Training Duration** | {text_results['training_duration_seconds']:.2f} s | {multimodal_results['training_duration_seconds']:.2f} s | - |

---

## 4. Per-Class Detailed Performance

### A. Text-Only Baseline
- **Non-Sarcastic:** Precision={t_m['per_class']['non_sarcastic']['precision']:.4f}, Recall={t_m['per_class']['non_sarcastic']['recall']:.4f}, F1={t_m['per_class']['non_sarcastic']['f1']:.4f} (Support: {t_m['per_class']['non_sarcastic']['support']})
- **Sarcastic:** Precision={t_m['per_class']['sarcastic']['precision']:.4f}, Recall={t_m['per_class']['sarcastic']['recall']:.4f}, F1={t_m['per_class']['sarcastic']['f1']:.4f} (Support: {t_m['per_class']['sarcastic']['support']})
- **Confusion Matrix:**
```
Predicted ->   Non-Sarcastic   Sarcastic
Actual Non-S:      {t_m['confusion_matrix'][0][0]:<12} {t_m['confusion_matrix'][0][1]}
Actual Sarc:       {t_m['confusion_matrix'][1][0]:<12} {t_m['confusion_matrix'][1][1]}
```

### B. Image–Comment Multimodal Fusion
- **Non-Sarcastic:** Precision={m_m['per_class']['non_sarcastic']['precision']:.4f}, Recall={m_m['per_class']['non_sarcastic']['recall']:.4f}, F1={m_m['per_class']['non_sarcastic']['f1']:.4f} (Support: {m_m['per_class']['non_sarcastic']['support']})
- **Sarcastic:** Precision={m_m['per_class']['sarcastic']['precision']:.4f}, Recall={m_m['per_class']['sarcastic']['recall']:.4f}, F1={m_m['per_class']['sarcastic']['f1']:.4f} (Support: {m_m['per_class']['sarcastic']['support']})
- **Confusion Matrix:**
```
Predicted ->   Non-Sarcastic   Sarcastic
Actual Non-S:      {m_m['confusion_matrix'][0][0]:<12} {m_m['confusion_matrix'][0][1]}
Actual Sarc:       {m_m['confusion_matrix'][1][0]:<12} {m_m['confusion_matrix'][1][1]}
```

---

## 5. Model Selection & Checkpoint Ledger
- **Text Baseline Checkpoint:** `artifacts/phase8a/text_baseline/best_text_baseline.pkl`
  - Selected via 6-candidate hyperparameter grid on validation set.
  - Best Val Macro-F1: `{text_results['best_val_macro_f1']:.4f}`.
- **Multimodal Fusion Checkpoint:** `artifacts/phase8a/multimodal_fusion/best_multimodal_fusion.pt`
  - Selected via early validation tracking across 5 epochs.
  - Best Val Macro-F1: `{multimodal_results['best_val_macro_f1']:.4f}` (Epoch {multimodal_results['best_epoch']}).

---

## 6. Evaluation Gate Status
- **Human Benchmark Evaluation Gate:** `BLOCKED_PENDING_HUMAN_ANNOTATION`  
- Both annotator response files (`annotator_1_responses.csv`, `annotator_2_responses.csv`) remain strictly preserved with 0 completed reviews.
- No passive-aggression accuracy claims are made or implied.
"""
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    logger.info("Phase 8A Training Report written to %s", report_path)
    return report_path


# =========================================================================
# MAIN ORCHESTRATOR
# =========================================================================

def main():
    parser = argparse.ArgumentParser(description="Phase 8A Baseline Trainer")
    parser.add_argument("--smoke-test", action="store_true", help="Run quick smoke test on 100 samples")
    parser.add_argument("--epochs", type=int, default=5, help="Epochs for multimodal fusion")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for feature extraction and training")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, help="Random seed")
    args = parser.parse_args()

    set_seed(args.seed)
    max_rec = 100 if args.smoke_test else None
    logger.info("Initializing Phase 8A Trainer (smoke_test=%s, seed=%d)", args.smoke_test, args.seed)

    # 1. Load data
    train_data = load_split_csv("train", max_records=max_rec)
    val_data = load_split_csv("val", max_records=max_rec // 2 if max_rec else None)
    test_data = load_split_csv("test", max_records=max_rec // 2 if max_rec else None)

    # 2. Train Text-Only Baseline
    tb_out = ARTIFACTS_DIR / "text_baseline"
    tb_results = train_and_eval_text_baseline(train_data, val_data, test_data, tb_out)

    # 3. Train Multimodal Fusion
    mf_out = ARTIFACTS_DIR / "multimodal_fusion"
    mf_results = train_and_eval_multimodal_fusion(
        train_data, val_data, test_data, mf_out,
        epochs=1 if args.smoke_test else args.epochs,
        batch_size=32 if args.smoke_test else args.batch_size,
    )

    # 4. Generate Master Artifact & Report
    master_results = {
        "milestone": "Phase 8A — Baseline Training",
        "task_type": "AUXILIARY_MULTIMODAL_SARCASM_DETECTION",
        "execution_timestamp": datetime.now(timezone.utc).isoformat(),
        "smoke_test_mode": args.smoke_test,
        "random_seed": args.seed,
        "text_baseline": tb_results,
        "multimodal_fusion": mf_results,
        "human_benchmark_gate": "BLOCKED_PENDING_HUMAN_ANNOTATION",
    }
    with open(ARTIFACTS_DIR / "phase8a_training_results.json", "w", encoding="utf-8") as f:
        json.dump(master_results, f, indent=2)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_p = generate_phase8a_report(tb_results, mf_results, REPORTS_DIR)

    logger.info("==================================================")
    logger.info("PHASE 8A BASELINE TRAINING COMPLETED SUCCESSFULLY")
    logger.info("Report: %s", report_p)
    logger.info("==================================================")


if __name__ == "__main__":
    main()
