"""
XLM-R 28-Class Multi-Label Emotion Model Training Script (Milestone 3)
Project: Gen-Z Multimodal Sarcasm Detection System
Hardware: NVIDIA GeForce RTX 4050 Laptop GPU / CUDA Acceleration
Author: Senior NLP / PyTorch Pipeline Engineer

Fine-tunes FacebookAI/xlm-roberta-base for 28-class multi-label emotion classification
using the GoEmotions taxonomy (BCEWithLogitsLoss).
Trained on 43,410 training samples, evaluated on 5,426 validation samples,
and tested on 5,427 test samples.

Reports:
- Micro F1, Macro F1
- Micro Precision, Micro Recall
- Macro Precision, Macro Recall
- Hamming Loss
- Exact-match accuracy (strict subset accuracy)
- Per-emotion precision, recall, F1, and support
Saves outputs to: models/xlmr_emotion_model/
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
    f1_score,
    precision_recall_fscore_support,
    hamming_loss,
    accuracy_score,
)

# ==============================================================================
# CONFIGURATION
# ==============================================================================

PROJECT_ROOT = Path(r"D:\NLP_PROJECT")
DATA_DIR = PROJECT_ROOT / "FINAL_DATASETS" / "EMOTION" / "SPLIT"
LABEL_MAP_FILE = PROJECT_ROOT / "FINAL_DATASETS" / "EMOTION" / "emotion_label_map.csv"
OUTPUT_DIR = PROJECT_ROOT / "models" / "xlmr_emotion_model"

TRAIN_FILE = DATA_DIR / "train.csv"
VAL_FILE = DATA_DIR / "validation.csv"
TEST_FILE = DATA_DIR / "test.csv"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "FacebookAI/xlm-roberta-base"
NUM_LABELS = 28
MAX_LENGTH = 128
NUM_EPOCHS = 3
LEARNING_RATE = 2e-5
PER_DEVICE_TRAIN_BATCH_SIZE = 16
GRADIENT_ACCUMULATION_STEPS = 2
PER_DEVICE_EVAL_BATCH_SIZE = 32
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
SEED = 42

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(OUTPUT_DIR / "training.log", mode="w", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("Train_Emotion")

def load_label_map():
    df = pd.read_csv(LABEL_MAP_FILE)
    id2label = dict(zip(df["emotion_id"].astype(int), df["emotion"]))
    label2id = dict(zip(df["emotion"], df["emotion_id"].astype(int)))
    return id2label, label2id

ID2LABEL, LABEL2ID = load_label_map()

def encode_multi_hot(id_str: str) -> list:
    vec = [0.0] * NUM_LABELS
    for part in str(id_str).split(","):
        part = part.strip()
        if part:
            try:
                vec[int(part)] = 1.0
            except ValueError:
                pass
    return vec

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    if isinstance(logits, tuple):
        logits = logits[0]
    probs = 1.0 / (1.0 + np.exp(-logits))
    preds = (probs >= 0.5).astype(np.float32)

    macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
    micro_f1 = f1_score(labels, preds, average="micro", zero_division=0)
    p_macro, r_macro, _, _ = precision_recall_fscore_support(labels, preds, average="macro", zero_division=0)

    return {
        "macro_f1": float(macro_f1),
        "micro_f1": float(micro_f1),
        "macro_precision": float(p_macro),
        "macro_recall": float(r_macro),
    }

def main():
    logger.info("=" * 75)
    logger.info("   MILESTONE 3: XLM-R 28-CLASS MULTI-LABEL EMOTION TRAINING")
    logger.info("=" * 75)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required.")

    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    df_train = pd.read_csv(TRAIN_FILE, usecols=["text", "emotion_ids"])
    df_val = pd.read_csv(VAL_FILE, usecols=["text", "emotion_ids"])
    df_test = pd.read_csv(TEST_FILE, usecols=["text", "emotion_ids"])

    df_train["labels"] = df_train["emotion_ids"].apply(encode_multi_hot)
    df_val["labels"] = df_val["emotion_ids"].apply(encode_multi_hot)
    df_test["labels"] = df_test["emotion_ids"].apply(encode_multi_hot)

    df_train["text"] = df_train["text"].fillna("").astype(str)
    df_val["text"] = df_val["text"].fillna("").astype(str)
    df_test["text"] = df_test["text"].fillna("").astype(str)

    logger.info(f"Train samples:      {len(df_train):,}")
    logger.info(f"Validation samples: {len(df_val):,}")
    logger.info(f"Test samples:       {len(df_test):,}")

    train_dataset = Dataset.from_pandas(df_train[["text", "labels"]])
    val_dataset = Dataset.from_pandas(df_val[["text", "labels"]])
    test_dataset = Dataset.from_pandas(df_test[["text", "labels"]])

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize_fn(batch):
        return tokenizer(batch["text"], truncation=True, max_length=MAX_LENGTH, padding=False)

    logger.info("Tokenizing datasets...")
    train_dataset = train_dataset.map(tokenize_fn, batched=True, batch_size=1000)
    val_dataset = val_dataset.map(tokenize_fn, batched=True, batch_size=1000)
    test_dataset = test_dataset.map(tokenize_fn, batched=True, batch_size=1000)

    train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    val_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    test_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True)

    logger.info(f"Initializing XLM-R multi-label classifier: {MODEL_NAME}")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        problem_type="multi_label_classification",
    )

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

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    logger.info("Starting emotion training loop...")
    trainer.train()

    # Evaluation on Unseen Test Split
    logger.info("Evaluating best model on test split (5,427 samples)...")
    test_predictions = trainer.predict(test_dataset)
    test_logits = test_predictions.predictions
    test_labels = test_predictions.label_ids
    test_probs = 1.0 / (1.0 + np.exp(-test_logits))
    test_preds = (test_probs >= 0.5).astype(np.float32)

    macro_f1 = f1_score(test_labels, test_preds, average="macro", zero_division=0)
    micro_f1 = f1_score(test_labels, test_preds, average="micro", zero_division=0)
    macro_p, macro_r, _, _ = precision_recall_fscore_support(test_labels, test_preds, average="macro", zero_division=0)
    micro_p, micro_r, _, _ = precision_recall_fscore_support(test_labels, test_preds, average="micro", zero_division=0)

    h_loss = hamming_loss(test_labels, test_preds)
    exact_match = accuracy_score(test_labels, test_preds)

    # Per-Emotion Metrics
    p_per, r_per, f1_per, sup_per = precision_recall_fscore_support(test_labels, test_preds, average=None, zero_division=0)
    per_emotion_metrics = {}
    for i in range(NUM_LABELS):
        emo_name = ID2LABEL[i]
        per_emotion_metrics[emo_name] = {
            "emotion_id": i,
            "precision": float(p_per[i]),
            "recall": float(r_per[i]),
            "f1": float(f1_per[i]),
            "support": int(sup_per[i]),
        }

    results = {
        "model": MODEL_NAME,
        "task": "28-class Multi-Label Emotion Classification (GoEmotions)",
        "train_samples": len(df_train),
        "validation_samples": len(df_val),
        "test_samples": len(df_test),
        "classification_threshold": 0.5,
        "test_macro_f1": float(macro_f1),
        "test_micro_f1": float(micro_f1),
        "test_macro_precision": float(macro_p),
        "test_macro_recall": float(macro_r),
        "test_micro_precision": float(micro_p),
        "test_micro_recall": float(micro_r),
        "hamming_loss": float(h_loss),
        "exact_match_accuracy": float(exact_match),
        "per_emotion_metrics": per_emotion_metrics,
        "labels": ID2LABEL,
    }

    # Save results to both requested filenames
    results_file = OUTPUT_DIR / "emotion_test_results.json"
    final_file = OUTPUT_DIR / "final_test_results.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    with open(final_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    trainer.save_model(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))

    with open(OUTPUT_DIR / "TRAINING_COMPLETE.txt", "w", encoding="utf-8") as f:
        f.write(f"XLM-R Emotion Model Training Complete.\n")
        f.write(f"Test Micro-F1: {micro_f1:.6f}\n")
        f.write(f"Test Macro-F1: {macro_f1:.6f}\n")
        f.write(f"Hamming Loss: {h_loss:.6f}\n")
        f.write(f"Exact-Match Accuracy: {exact_match:.6f}\n")

    logger.info("=" * 75)
    logger.info(f"   TEST MICRO F1:          {micro_f1:.6f}")
    logger.info(f"   TEST MACRO F1:          {macro_f1:.6f}")
    logger.info(f"   TEST HAMMING LOSS:      {h_loss:.6f}")
    logger.info(f"   EXACT-MATCH ACCURACY:   {exact_match:.6f}")
    logger.info("=" * 75)
    logger.info("Emotion Model training finished successfully.")

if __name__ == "__main__":
    main()
