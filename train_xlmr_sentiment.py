"""
XLM-R Sentiment Model Training Script (Milestone 2)
Project: Gen-Z Multimodal Sarcasm Detection System
Hardware: NVIDIA GeForce RTX 4050 Laptop GPU / CUDA Acceleration
Author: Senior NLP / ML Engineer

Fine-tunes FacebookAI/xlm-roberta-base for 3-class sentiment classification:
- 0: Negative
- 1: Neutral
- 2: Positive
Trained on 107,678 training samples, evaluated on 7,916 validation samples,
and tested on 16,730 test samples. Saves the trained model and final metrics.
"""

import os
import sys
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
)
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)

# ==============================================================================
# 1. CONFIGURATION & PATHS
# ==============================================================================

PROJECT_ROOT = Path(r"D:\NLP_PROJECT")
DATA_DIR = PROJECT_ROOT / "FINAL_DATASETS" / "SENTIMENT" / "SPLIT"
OUTPUT_DIR = PROJECT_ROOT / "models" / "xlmr_sentiment_model"

TRAIN_FILE = DATA_DIR / "train.csv"
VAL_FILE = DATA_DIR / "validation.csv"
TEST_FILE = DATA_DIR / "test.csv"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Training Hyperparameters
MODEL_NAME = "FacebookAI/xlm-roberta-base"
NUM_LABELS = 3
MAX_LENGTH = 128
NUM_EPOCHS = 3
LEARNING_RATE = 2e-5
PER_DEVICE_TRAIN_BATCH_SIZE = 16
GRADIENT_ACCUMULATION_STEPS = 2
PER_DEVICE_EVAL_BATCH_SIZE = 32
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
SEED = 42

ID2LABEL = {0: "Negative", 1: "Neutral", 2: "Positive"}
LABEL2ID = {"Negative": 0, "Neutral": 1, "Positive": 2}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(OUTPUT_DIR / "training.log", mode="w", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("Train_Sentiment")

# ==============================================================================
# 2. METRICS COMPUTATION
# ==============================================================================

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    if isinstance(logits, tuple):
        logits = logits[0]
    preds = np.argmax(logits, axis=-1)

    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="macro", zero_division=0
    )
    return {
        "accuracy": float(acc),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
    }

# ==============================================================================
# 3. MAIN TRAINING ROUTINE
# ==============================================================================

def main():
    logger.info("=" * 75)
    logger.info("   MILESTONE 2: XLM-R 3-CLASS SENTIMENT MODEL TRAINING")
    logger.info("=" * 75)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for training.")

    gpu_name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    logger.info(f"GPU: {gpu_name} ({vram_gb:.2f} GB VRAM)")

    # 1. Load Data
    logger.info("Loading dataset splits...")
    df_train = pd.read_csv(TRAIN_FILE, usecols=["text", "label"])
    df_val = pd.read_csv(VAL_FILE, usecols=["text", "label"])
    df_test = pd.read_csv(TEST_FILE, usecols=["text", "label"])

    # Clean text nulls if any
    df_train["text"] = df_train["text"].fillna("").astype(str)
    df_val["text"] = df_val["text"].fillna("").astype(str)
    df_test["text"] = df_test["text"].fillna("").astype(str)

    logger.info(f"Train samples:      {len(df_train):,}")
    logger.info(f"Validation samples: {len(df_val):,}")
    logger.info(f"Test samples:       {len(df_test):,}")

    train_dataset = Dataset.from_pandas(df_train[["text", "label"]])
    val_dataset = Dataset.from_pandas(df_val[["text", "label"]])
    test_dataset = Dataset.from_pandas(df_test[["text", "label"]])

    # 2. Tokenization
    logger.info(f"Loading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize_fn(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=MAX_LENGTH,
            padding=False,
        )

    logger.info("Tokenizing datasets...")
    train_dataset = train_dataset.map(tokenize_fn, batched=True, batch_size=1000)
    val_dataset = val_dataset.map(tokenize_fn, batched=True, batch_size=1000)
    test_dataset = test_dataset.map(tokenize_fn, batched=True, batch_size=1000)

    # Prepare columns
    train_dataset = train_dataset.rename_column("label", "labels")
    val_dataset = val_dataset.rename_column("label", "labels")
    test_dataset = test_dataset.rename_column("label", "labels")

    train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    val_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    test_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True)

    # 3. Model Initialization
    logger.info(f"Initializing XLM-R 3-class classifier: {MODEL_NAME}")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    # 4. Training Arguments
    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        warmup_ratio=WARMUP_RATIO,
        weight_decay=WEIGHT_DECAY,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        logging_steps=100,
        fp16=True,
        gradient_checkpointing=True,
        dataloader_num_workers=0,
        seed=SEED,
        report_to="none",
    )

    # 5. Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    logger.info("Starting training loop...")
    train_result = trainer.train()
    logger.info(f"Training completed. Global steps: {train_result.global_step}")

    # 6. Evaluation on Unseen Test Set
    logger.info("Evaluating best checkpoint on unseen test split (16,730 samples)...")
    test_predictions = trainer.predict(test_dataset)
    test_logits = test_predictions.predictions
    test_labels = test_predictions.label_ids
    test_preds = np.argmax(test_logits, axis=-1)

    test_acc = accuracy_score(test_labels, test_preds)
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        test_labels, test_preds, average="macro", zero_division=0
    )
    conf_matrix = confusion_matrix(test_labels, test_preds).tolist()
    class_rep = classification_report(test_labels, test_preds, target_names=["Negative", "Neutral", "Positive"], output_dict=True)

    test_results = {
        "model": MODEL_NAME,
        "task": "3-class Sentiment Classification",
        "train_samples": len(df_train),
        "validation_samples": len(df_val),
        "test_samples": len(df_test),
        "epochs": NUM_EPOCHS,
        "effective_batch_size": PER_DEVICE_TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS,
        "learning_rate": LEARNING_RATE,
        "test_accuracy": float(test_acc),
        "test_macro_precision": float(p_macro),
        "test_macro_recall": float(r_macro),
        "test_macro_f1": float(f1_macro),
        "confusion_matrix": conf_matrix,
        "classification_report": class_rep,
        "labels": ID2LABEL,
    }

    results_file = OUTPUT_DIR / "final_test_results.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(test_results, f, indent=2)

    # 7. Save Final Model Artifacts
    logger.info(f"Saving final model and tokenizer to: {OUTPUT_DIR}")
    trainer.save_model(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))

    with open(OUTPUT_DIR / "TRAINING_COMPLETE.txt", "w", encoding="utf-8") as f:
        f.write(f"XLM-R Sentiment Model Training Complete.\n")
        f.write(f"Test Accuracy: {test_acc:.6f}\n")
        f.write(f"Test Macro-F1: {f1_macro:.6f}\n")

    logger.info("=" * 75)
    logger.info(f"   TEST ACCURACY: {test_acc:.6f}")
    logger.info(f"   TEST MACRO F1: {f1_macro:.6f}")
    logger.info(f"   CONFUSION MATRIX (Negative, Neutral, Positive):")
    for row in conf_matrix:
        logger.info(f"      {row}")
    logger.info("=" * 75)
    logger.info("XLM-R Sentiment Model training finished successfully.")


if __name__ == "__main__":
    main()
