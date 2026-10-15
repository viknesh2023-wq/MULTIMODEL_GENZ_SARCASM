"""
Train Multimodal Fusion Classifier & Run 8-Way Ablation Benchmark (Milestone 4)
Project: Gen-Z Multimodal Sarcasm Detection System
Hardware: NVIDIA GeForce RTX 4050 Laptop GPU / CUDA Acceleration

Trains and evaluates:
1. Full 2,847-D Multimodal Fusion Classifier
2. Complete 8-Way Modality Ablations:
   - Config 1: Text only (768-D)
   - Config 2: Text + Sentiment (771-D)
   - Config 3: Text + Emotion (796-D)
   - Config 4: Text + Sentiment + Emotion (799-D)
   - Config 5: Text + CLIP (1,280-D)
   - Config 6: Text + InternVL (2,304-D)
   - Config 7: Text + CLIP + InternVL (2,816-D)
   - Config 8: Full Five-Source Fusion (2,847-D)

Saves best model, scalers, predictions, and ablation metrics to models/multimodal_fusion/
"""

import sys
import os
import time
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    f1_score,
    confusion_matrix,
)

# ==============================================================================
# CONFIGURATION
# ==============================================================================

PROJECT_ROOT = Path(r"D:\NLP_PROJECT")
DATA_DIR = PROJECT_ROOT / "FINAL_DATASETS" / "MULTIMODAL_SARCASM" / "FUSION_DATA"
OUTPUT_DIR = PROJECT_ROOT / "models" / "multimodal_fusion"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = OUTPUT_DIR / "training.log"

class FlushingFileHandler(logging.FileHandler):
    def emit(self, record):
        super().emit(record)
        self.flush()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        FlushingFileHandler(LOG_FILE, mode="w", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("Train_Fusion")

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ==============================================================================
# MODEL ARCHITECTURE
# ==============================================================================

class MultimodalFusionMLP(nn.Module):
    def __init__(self, in_features: int, num_classes: int = 2, dropout_p: float = 0.3):
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        
        self.net = nn.Sequential(
            nn.Linear(in_features, 1024),
            nn.Mish(),
            nn.Dropout(dropout_p),
            nn.Linear(1024, 512),
            nn.Mish(),
            nn.Dropout(dropout_p * 0.67),
            nn.Linear(512, 128),
            nn.Mish(),
            nn.Dropout(dropout_p * 0.33),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

# ==============================================================================
# ABLATION CONFIGURATIONS DEFINITION
# ==============================================================================

# Column layout in 2847-D vector:
# [0:768]     -> Text (768)
# [768:1280]  -> CLIP (512)
# [1280:2816] -> InternVL (1536)
# [2816:2819] -> Sentiment (3)
# [2819:2847] -> Emotion (28)

ABLATIONS = [
    {
        "name": "text_only",
        "description": "XLM-R Contextual Text Only (768-D)",
        "dim": 768,
        "cols": list(range(0, 768)),
    },
    {
        "name": "text_sentiment",
        "description": "Text + Sentiment Valence (771-D)",
        "dim": 771,
        "cols": list(range(0, 768)) + list(range(2816, 2819)),
    },
    {
        "name": "text_emotion",
        "description": "Text + Affective Emotion (796-D)",
        "dim": 796,
        "cols": list(range(0, 768)) + list(range(2819, 2847)),
    },
    {
        "name": "text_sentiment_emotion",
        "description": "Text + Sentiment + Emotion (799-D)",
        "dim": 799,
        "cols": list(range(0, 768)) + list(range(2816, 2847)),
    },
    {
        "name": "text_clip",
        "description": "Text + CLIP ViT-B/32 (1,280-D)",
        "dim": 1280,
        "cols": list(range(0, 1280)),
    },
    {
        "name": "text_vlm",
        "description": "Text + InternVL3-2B Semantic (2,304-D)",
        "dim": 2304,
        "cols": list(range(0, 768)) + list(range(1280, 2816)),
    },
    {
        "name": "text_clip_vlm",
        "description": "Text + CLIP + InternVL (2,816-D)",
        "dim": 2816,
        "cols": list(range(0, 2816)),
    },
    {
        "name": "full_fusion",
        "description": "Full Five-Source Multimodal Fusion (2,847-D)",
        "dim": 2847,
        "cols": list(range(0, 2847)),
    },
]

# ==============================================================================
# TRAINING & EVALUATION FUNCTIONS
# ==============================================================================

def train_and_eval_configuration(
    cfg: dict,
    X_train_full: np.ndarray,
    y_train: np.ndarray,
    X_val_full: np.ndarray,
    y_val: np.ndarray,
    X_test_full: np.ndarray,
    y_test: np.ndarray,
    device: torch.device,
    batch_size: int = 128,
    max_epochs: int = 25,
    patience: int = 7,
    save_artifacts: bool = False,
):
    cols = cfg["cols"]
    cfg_name = cfg["name"]
    in_dim = cfg["dim"]
    logger.info(f"--- Running Experiment: {cfg['description']} ({in_dim}-D) ---")

    t_exp_start = time.time()

    # 1. Feature Preprocessing without leakage: Fit scaler strictly on X_train
    scaler = StandardScaler(copy=True)
    X_tr = scaler.fit_transform(X_train_full[:, cols])
    X_v = scaler.transform(X_val_full[:, cols])
    X_te = scaler.transform(X_test_full[:, cols])

    # 2. DataLoaders
    train_ds = TensorDataset(torch.from_numpy(X_tr).float(), torch.from_numpy(y_train).long())
    val_ds = TensorDataset(torch.from_numpy(X_v).float(), torch.from_numpy(y_val).long())
    test_ds = TensorDataset(torch.from_numpy(X_te).float(), torch.from_numpy(y_test).long())

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size * 2, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size * 2, shuffle=False)

    # 3. Model, Loss, Optimizer
    model = MultimodalFusionMLP(in_features=in_dim, num_classes=2, dropout_p=0.3).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)

    best_val_macro_f1 = -1.0
    best_epoch = -1
    best_state_dict = None
    no_improve_epochs = 0
    history = []

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)

    for epoch in range(1, max_epochs + 1):
        model.train()
        train_loss = 0.0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            logits = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(by)
        train_loss /= len(y_train)
        scheduler.step()

        # Validation
        model.eval()
        val_preds, val_targets = [], []
        with torch.inference_mode():
            for bx, by in val_loader:
                bx = bx.to(device)
                logits = model(bx)
                preds = torch.argmax(logits, dim=-1).cpu().numpy()
                val_preds.extend(preds)
                val_targets.extend(by.numpy())

        val_preds = np.array(val_preds)
        val_targets = np.array(val_targets)
        val_acc = accuracy_score(val_targets, val_preds)
        val_macro_f1 = f1_score(val_targets, val_preds, average="macro")

        history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "val_accuracy": round(val_acc, 6),
            "val_macro_f1": round(val_macro_f1, 6),
        })

        if val_macro_f1 > best_val_macro_f1:
            best_val_macro_f1 = val_macro_f1
            best_epoch = epoch
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1

        if no_improve_epochs >= patience:
            logger.info(f"Early stopping at epoch {epoch} (Best epoch: {best_epoch}, Val Macro F1: {best_val_macro_f1:.6f})")
            break

    train_duration = time.time() - t_exp_start

    # 4. Final Evaluation on Untouched Test Split
    model.load_state_dict(best_state_dict)
    model.to(device)
    model.eval()

    test_preds, test_probs, test_targets = [], [], []
    t_inf_start = time.time()
    with torch.inference_mode():
        for bx, by in test_loader:
            bx = bx.to(device)
            logits = model(bx)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            preds = np.argmax(probs, axis=-1)
            test_preds.extend(preds)
            test_probs.extend(probs)
            test_targets.extend(by.numpy())
    test_inf_time = time.time() - t_inf_start

    test_preds = np.array(test_preds)
    test_probs = np.array(test_probs)
    test_targets = np.array(test_targets)

    test_acc = accuracy_score(test_targets, test_preds)
    test_macro_f1 = f1_score(test_targets, test_preds, average="macro")
    test_weighted_f1 = f1_score(test_targets, test_preds, average="weighted")
    p_macro, r_macro, _, _ = precision_recall_fscore_support(test_targets, test_preds, average="macro", zero_division=0)
    p_per, r_per, f1_per, sup_per = precision_recall_fscore_support(test_targets, test_preds, average=None, zero_division=0)
    cm = confusion_matrix(test_targets, test_preds).tolist()

    latency_per_sample_ms = (test_inf_time / len(test_targets)) * 1000
    peak_vram_mb = torch.cuda.max_memory_allocated(device) / (1024**2) if torch.cuda.is_available() else 0.0

    logger.info(f"[{cfg_name.upper()}] Val Macro F1: {best_val_macro_f1:.4f} | Test Acc: {test_acc:.4f} | Test Macro F1: {test_macro_f1:.4f} | Peak VRAM: {peak_vram_mb:.1f} MB")

    result = {
        "configuration": cfg_name,
        "description": cfg["description"],
        "input_dimension": in_dim,
        "best_epoch": best_epoch,
        "val_macro_f1": float(best_val_macro_f1),
        "test_accuracy": float(test_acc),
        "test_macro_f1": float(test_macro_f1),
        "test_macro_precision": float(p_macro),
        "test_macro_recall": float(r_macro),
        "test_weighted_f1": float(test_weighted_f1),
        "per_class_metrics": {
            "0_non_sarcastic": {
                "precision": float(p_per[0]),
                "recall": float(r_per[0]),
                "f1": float(f1_per[0]),
                "support": int(sup_per[0]),
            },
            "1_sarcastic": {
                "precision": float(p_per[1]),
                "recall": float(r_per[1]),
                "f1": float(f1_per[1]),
                "support": int(sup_per[1]),
            },
        },
        "confusion_matrix": cm,
        "training_time_seconds": round(train_duration, 2),
        "test_inference_time_seconds": round(test_inf_time, 4),
        "latency_per_sample_ms": round(latency_per_sample_ms, 3),
        "peak_gpu_vram_mb": round(peak_vram_mb, 2),
    }

    # If save_artifacts is True (e.g. for full_fusion), save weights, scaler, predictions
    if save_artifacts:
        torch.save(best_state_dict, OUTPUT_DIR / "fusion_model.pt")
        joblib.dump(scaler, OUTPUT_DIR / "fusion_scaler.joblib")

        with open(OUTPUT_DIR / "training_history.json", "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

        # Save test predictions CSV
        test_df = pd.read_csv(PROJECT_ROOT / "FINAL_DATASETS" / "MULTIMODAL_SARCASM" / "SPLIT" / "test.csv")
        preds_df = pd.DataFrame({
            "id": test_df["id"],
            "ground_truth": test_targets,
            "predicted_label": test_preds,
            "prob_non_sarcastic": test_probs[:, 0],
            "prob_sarcastic": test_probs[:, 1],
        })
        preds_df.to_csv(OUTPUT_DIR / "test_predictions.csv", index=False)

        # Model configuration
        model_config = {
            "model_architecture": "MultimodalFusionMLP",
            "in_features": in_dim,
            "hidden_layers": [1024, 512, 128],
            "num_classes": 2,
            "activation": "Mish",
            "dropout_rates": [0.3, 0.2, 0.1],
            "normalization": "StandardScaler",
            "feature_slices": {
                "text": {"start": 0, "end": 768, "dim": 768},
                "clip": {"start": 768, "end": 1280, "dim": 512},
                "vlm": {"start": 1280, "end": 2816, "dim": 1536},
                "sentiment": {"start": 2816, "end": 2819, "dim": 3},
                "emotion": {"start": 2819, "end": 2847, "dim": 28},
            },
            "canonical_label_map": {0: "non-sarcastic", 1: "sarcastic"},
        }
        with open(OUTPUT_DIR / "model_config.json", "w", encoding="utf-8") as f:
            json.dump(model_config, f, indent=2)

    return result

def main():
    logger.info("=" * 75)
    logger.info("   MILESTONE 4: MULTIMODAL FUSION TRAINING & 8-WAY ABLATIONS")
    logger.info("=" * 75)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Execution Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    # Load Assembled Matrices
    logger.info(f"Loading assembled matrices from: {DATA_DIR}...")
    X_train = np.load(DATA_DIR / "X_train.npy")
    y_train = np.load(DATA_DIR / "y_train.npy")
    X_val = np.load(DATA_DIR / "X_val.npy")
    y_val = np.load(DATA_DIR / "y_val.npy")
    X_test = np.load(DATA_DIR / "X_test.npy")
    y_test = np.load(DATA_DIR / "y_test.npy")

    logger.info(f"Loaded X_train: {X_train.shape}, X_val: {X_val.shape}, X_test: {X_test.shape}")

    # Run All 8 Ablation Experiments
    all_ablation_results = []
    for cfg in ABLATIONS:
        is_full = (cfg["name"] == "full_fusion")
        res = train_and_eval_configuration(
            cfg=cfg,
            X_train_full=X_train,
            y_train=y_train,
            X_val_full=X_val,
            y_val=y_val,
            X_test_full=X_test,
            y_test=y_test,
            device=device,
            batch_size=128,
            max_epochs=25,
            patience=7,
            save_artifacts=is_full,
        )
        all_ablation_results.append(res)

    # Save Ablation Results
    ablation_summary_file = OUTPUT_DIR / "ablation_benchmark_results.json"
    with open(ablation_summary_file, "w", encoding="utf-8") as f:
        json.dump(all_ablation_results, f, indent=2)

    # Convert to CSV table for clear audit and reporting
    df_ablation = pd.DataFrame([{
        "Configuration": r["configuration"],
        "Description": r["description"],
        "Dimensions": r["input_dimension"],
        "Best Epoch": r["best_epoch"],
        "Val Macro F1": round(r["val_macro_f1"], 4),
        "Test Accuracy": round(r["test_accuracy"], 4),
        "Test Macro F1": round(r["test_macro_f1"], 4),
        "Test Macro P": round(r["test_macro_precision"], 4),
        "Test Macro R": round(r["test_macro_recall"], 4),
        "Test Weighted F1": round(r["test_weighted_f1"], 4),
        "Class 0 F1": round(r["per_class_metrics"]["0_non_sarcastic"]["f1"], 4),
        "Class 1 F1": round(r["per_class_metrics"]["1_sarcastic"]["f1"], 4),
        "Train Time (s)": r["training_time_seconds"],
        "Latency (ms)": r["latency_per_sample_ms"],
        "Peak VRAM (MB)": r["peak_gpu_vram_mb"],
    } for r in all_ablation_results])

    df_ablation.to_csv(OUTPUT_DIR / "ablation_benchmark_summary.csv", index=False)
    logger.info(f"Saved ablation summary to {OUTPUT_DIR / 'ablation_benchmark_summary.csv'}")

    # Save Full Fusion Test Results as Primary File
    full_fusion_res = [r for r in all_ablation_results if r["configuration"] == "full_fusion"][0]
    with open(OUTPUT_DIR / "test_results.json", "w", encoding="utf-8") as f:
        json.dump(full_fusion_res, f, indent=2)

    # Write completion marker
    with open(OUTPUT_DIR / "TRAINING_COMPLETE.txt", "w", encoding="utf-8") as f:
        f.write("MULTIMODAL FUSION CLASSIFIER & 8-WAY ABLATION TRAINING COMPLETE\n")
        f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Full Fusion Test Accuracy: {full_fusion_res['test_accuracy']:.6f}\n")
        f.write(f"Full Fusion Test Macro F1: {full_fusion_res['test_macro_f1']:.6f}\n")
        f.write(f"Full Fusion Test Weighted F1: {full_fusion_res['test_weighted_f1']:.6f}\n")
        f.write(f"Class 0 (Non-sarcastic) F1: {full_fusion_res['per_class_metrics']['0_non_sarcastic']['f1']:.6f}\n")
        f.write(f"Class 1 (Sarcastic) F1: {full_fusion_res['per_class_metrics']['1_sarcastic']['f1']:.6f}\n")

    logger.info("=" * 75)
    logger.info("   ABLATION BENCHMARK SUMMARY:")
    logger.info("=" * 75)
    for _, row in df_ablation.iterrows():
        logger.info(f"   {row['Description']:<45} | Val F1: {row['Val Macro F1']:.4f} | Test Acc: {row['Test Accuracy']:.4f} | Test F1: {row['Test Macro F1']:.4f}")
    logger.info("=" * 75)
    logger.info("Milestone 4 Fusion Training & Ablations completed successfully!")

if __name__ == "__main__":
    main()
