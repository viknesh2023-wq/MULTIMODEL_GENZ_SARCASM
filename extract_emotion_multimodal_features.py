"""
Extract 28-D Emotion Probability Features for Multimodal Dataset
Project: Gen-Z Multimodal Sarcasm Detection System
Hardware: NVIDIA RTX 4050 Laptop GPU / CUDA Acceleration

Loads the fine-tuned XLM-R 28-class Emotion Model and extracts 28-dimensional
sigmoid probability vectors for all 33,077 samples in the multimodal dataset.
"""

import os
import sys
import time
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

PROJECT_ROOT = Path(r"D:\NLP_PROJECT")
MODEL_DIR = PROJECT_ROOT / "models" / "xlmr_emotion_model"
DATASET_CSV = PROJECT_ROOT / "FINAL_DATASETS" / "MULTIMODAL_SARCASM" / "multimodal_sarcasm_combined.csv"
SPLIT_DIR = PROJECT_ROOT / "FINAL_DATASETS" / "MULTIMODAL_SARCASM" / "SPLIT"
OUTPUT_DIR = PROJECT_ROOT / "FINAL_DATASETS" / "MULTIMODAL_SARCASM" / "EMOTION_FEATURES"

FEATURE_DIR = OUTPUT_DIR / "features"
CHECKPOINT_FILE = OUTPUT_DIR / "checkpoint.csv"
FINAL_DATASET_FILE = OUTPUT_DIR / "emotion_feature_dataset.csv"
SUMMARY_FILE = OUTPUT_DIR / "extraction_summary.json"
LOG_FILE = OUTPUT_DIR / "extraction.log"

MAX_LENGTH = 128
FEATURE_DIM = 28
BATCH_SIZE = 128

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FEATURE_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("Emotion_Feature_Extractor")

def get_safe_stem(image_path_value: str) -> str:
    stem = Path(str(image_path_value)).stem
    return "".join(c if (c.isalnum() or c in "_-") else "_" for c in stem)

def load_split_mappings():
    split_map = {}
    for split_name in ["train", "validation", "test"]:
        p = SPLIT_DIR / f"{split_name}.csv"
        if p.exists():
            df_sp = pd.read_csv(p, usecols=["id"])
            for sid in df_sp["id"]:
                split_map[sid] = split_name
    return split_map

def validate_feature_file(file_path: Path) -> bool:
    if not file_path.exists():
        return False
    try:
        arr = np.load(file_path)
        return arr.shape == (FEATURE_DIM,) and arr.dtype == np.float32 and np.all(np.isfinite(arr))
    except Exception:
        return False

def main():
    start_total_time = time.time()
    logger.info("=" * 75)
    logger.info("   EXTRACTING 28-D EMOTION FEATURES FOR MULTIMODAL DATASET")
    logger.info("=" * 75)

    if not MODEL_DIR.exists():
        raise RuntimeError(f"Trained emotion model not found at: {MODEL_DIR}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Execution Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.to(device)
    model.eval()

    df = pd.read_csv(DATASET_CSV)
    total_samples = len(df)
    split_lookup = load_split_mappings()

    processed_records = {}
    if CHECKPOINT_FILE.exists():
        try:
            chk_df = pd.read_csv(CHECKPOINT_FILE)
            for _, r in chk_df.iterrows():
                feat_p = FEATURE_DIR / str(r["feature_file"])
                if validate_feature_file(feat_p):
                    processed_records[int(r["dataset_index"])] = r.to_dict()
        except Exception:
            pass

    records_to_process = []
    for idx, row in df.iterrows():
        dataset_idx = int(idx)
        safe_stem = get_safe_stem(row["image_path"])
        f_name = f"{dataset_idx:06d}_{safe_stem}.npy"
        f_path = FEATURE_DIR / f_name

        if dataset_idx in processed_records:
            continue
        if validate_feature_file(f_path):
            sample_id = str(row["id"])
            processed_records[dataset_idx] = {
                "dataset_index": dataset_idx,
                "id": sample_id,
                "feature_file": f_name,
                "feature_dimension": FEATURE_DIM,
                "label": int(row["label"]),
                "split": split_lookup.get(sample_id, "train"),
                "status": "success",
            }
            continue

        records_to_process.append((dataset_idx, row, f_name, f_path))

    logger.info(f"Remaining to extract: {len(records_to_process):,} / {total_samples:,}")

    if records_to_process:
        num_batches = (len(records_to_process) + BATCH_SIZE - 1) // BATCH_SIZE
        for b_idx in range(num_batches):
            batch_slice = records_to_process[b_idx * BATCH_SIZE : (b_idx + 1) * BATCH_SIZE]
            batch_indices = [item[0] for item in batch_slice]
            batch_rows = [item[1] for item in batch_slice]
            batch_filenames = [item[2] for item in batch_slice]
            batch_paths = [item[3] for item in batch_slice]

            texts = [str(r["text"]) if pd.notna(r["text"]) else "" for r in batch_rows]
            inputs = tokenizer(texts, padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.inference_mode():
                outputs = model(**inputs)
                probs = torch.sigmoid(outputs.logits).float().cpu().numpy()

            for i in range(len(batch_slice)):
                d_idx = batch_indices[i]
                row_data = batch_rows[i]
                f_name = batch_filenames[i]
                f_path = batch_paths[i]
                emb = probs[i].astype(np.float32)

                tmp_path = f_path.with_suffix(".tmp.npy")
                np.save(tmp_path, emb)
                os.replace(tmp_path, f_path)

                sample_id = str(row_data["id"])
                processed_records[d_idx] = {
                    "dataset_index": d_idx,
                    "id": sample_id,
                    "feature_file": f_name,
                    "feature_dimension": FEATURE_DIM,
                    "label": int(row_data["label"]),
                    "split": split_lookup.get(sample_id, "train"),
                    "status": "success",
                }

            if (b_idx + 1) % 10 == 0 or (b_idx + 1) == num_batches:
                chk_df = pd.DataFrame(list(processed_records.values())).sort_values("dataset_index")
                chk_df.to_csv(CHECKPOINT_FILE, index=False)

    final_df = pd.DataFrame(list(processed_records.values())).sort_values("dataset_index")
    final_df.to_csv(FINAL_DATASET_FILE, index=False)
    final_df.to_csv(CHECKPOINT_FILE, index=False)

    total_time = time.time() - start_total_time
    summary = {
        "milestone": "Emotion Feature Extraction (28-D)",
        "source_samples": total_samples,
        "valid_embeddings": len(final_df),
        "dimension": FEATURE_DIM,
        "execution_time_seconds": round(total_time, 2),
    }
    with open(SUMMARY_FILE, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Emotion extraction complete in {total_time:.2f} seconds.")

if __name__ == "__main__":
    main()
