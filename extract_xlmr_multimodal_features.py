"""
XLM-R Multimodal Text Feature Extraction Script (Milestone 1)
Project: Gen-Z Multimodal Sarcasm Detection System
Hardware: NVIDIA GeForce RTX 4050 Laptop GPU / CUDA Acceleration
Author: Senior NLP / PyTorch Pipeline Engineer

Extracts 768-dimensional contextual text representations from the fine-tuned
XLM-RoBERTa sarcasm model for all 33,077 samples in the multimodal dataset.
Applies masked mean pooling over the transformer encoder's last hidden state.
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

# ==============================================================================
# 1. CONFIGURATION & PATHS
# ==============================================================================

PROJECT_ROOT = Path(r"D:\NLP_PROJECT")
MODEL_DIR = PROJECT_ROOT / "models" / "xlmr_text_sarcasm_full"

DATASET_CSV = (
    PROJECT_ROOT
    / "FINAL_DATASETS"
    / "MULTIMODAL_SARCASM"
    / "multimodal_sarcasm_combined.csv"
)

SPLIT_DIR = PROJECT_ROOT / "FINAL_DATASETS" / "MULTIMODAL_SARCASM" / "SPLIT"

OUTPUT_DIR = (
    PROJECT_ROOT
    / "FINAL_DATASETS"
    / "MULTIMODAL_SARCASM"
    / "XLMR_FEATURES"
)

FEATURE_DIR = OUTPUT_DIR / "features"
CHECKPOINT_FILE = OUTPUT_DIR / "checkpoint.csv"
FINAL_DATASET_FILE = OUTPUT_DIR / "xlmr_feature_dataset.csv"
SUMMARY_FILE = OUTPUT_DIR / "extraction_summary.json"
LOG_FILE = OUTPUT_DIR / "extraction.log"
FAILURES_FILE = OUTPUT_DIR / "failures.csv"

# Hyperparameters aligned with training setup
MAX_LENGTH = 128
FEATURE_DIM = 768
DEFAULT_BATCH_SIZE = 64
CHECKPOINT_INTERVAL = 500  # Save checkpoint every 500 samples

# ==============================================================================
# 2. LOGGING SETUP
# ==============================================================================

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
logger = logging.getLogger("XLMR_Feature_Extractor")


# ==============================================================================
# 3. HELPER FUNCTIONS
# ==============================================================================

def get_safe_stem(image_path_value: str) -> str:
    """Extract a safe filesystem stem matching CLIP & VLM naming conventions."""
    stem = Path(str(image_path_value)).stem
    return "".join(c if (c.isalnum() or c in "_-") else "_" for c in stem)


def load_split_mappings():
    """Load verified sample ID to split mappings from SPLIT directory."""
    split_map = {}
    for split_name in ["train", "validation", "test"]:
        split_path = SPLIT_DIR / f"{split_name}.csv"
        if split_path.exists():
            df_sp = pd.read_csv(split_path, usecols=["id"])
            for sid in df_sp["id"]:
                split_map[sid] = split_name
    return split_map


def validate_feature_file(file_path: Path) -> bool:
    """Verify that a feature file exists, is valid float32 numpy array, shape (768,), no NaNs."""
    if not file_path.exists():
        return False
    try:
        arr = np.load(file_path)
        if arr.shape != (FEATURE_DIM,):
            return False
        if arr.dtype != np.float32:
            return False
        if not np.all(np.isfinite(arr)):
            return False
        return True
    except Exception:
        return False


# ==============================================================================
# 4. MAIN EXTRACTION PIPELINE
# ==============================================================================

def main():
    start_total_time = time.time()
    logger.info("=" * 75)
    logger.info("   MILESTONE 1: XLM-R MULTIMODAL TEXT FEATURE EXTRACTION")
    logger.info("=" * 75)

    # Device selection
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        device_str = f"cuda ({gpu_name}, {vram_gb:.2f} GB VRAM)"
        logger.info(f"Execution Device: {device_str}")
    else:
        device = torch.device("cpu")
        device_str = "cpu"
        logger.warning("CUDA not available! Falling back to CPU inference.")

    # Load Model & Tokenizer
    logger.info(f"Loading fine-tuned XLM-R model from: {MODEL_DIR}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
        cls_model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
        encoder = cls_model.roberta
        encoder.to(device)
        encoder.eval()
        logger.info("✓ Fine-tuned XLM-R RoBERTa encoder loaded successfully into memory.")
    except Exception as exc:
        logger.error(f"Failed to load XLM-R model checkpoint: {exc}", exc_info=True)
        sys.exit(1)

    # Load Dataset
    logger.info(f"Loading multimodal dataset from: {DATASET_CSV}")
    df = pd.read_csv(DATASET_CSV)
    total_samples = len(df)
    logger.info(f"Total dataset records: {total_samples:,}")

    # Build split lookup
    split_lookup = load_split_mappings()
    logger.info(f"Verified split memberships mapped: {len(split_lookup):,} samples.")

    # Check existing progress from checkpoint
    processed_records = {}
    if CHECKPOINT_FILE.exists():
        try:
            chk_df = pd.read_csv(CHECKPOINT_FILE)
            for _, r in chk_df.iterrows():
                feat_name = str(r["feature_file"])
                feat_p = FEATURE_DIR / feat_name
                if validate_feature_file(feat_p):
                    processed_records[int(r["dataset_index"])] = r.to_dict()
            logger.info(f"Resuming: Found {len(processed_records):,} valid cached features in checkpoint.")
        except Exception as e:
            logger.warning(f"Error reading checkpoint file, validating disk directly: {e}")

    # Identify remaining records
    records_to_process = []
    for idx, row in df.iterrows():
        dataset_idx = int(idx)
        safe_stem = get_safe_stem(row["image_path"])
        feature_filename = f"{dataset_idx:06d}_{safe_stem}.npy"
        feature_path = FEATURE_DIR / feature_filename

        # If already validated in checkpoint and on disk, skip
        if dataset_idx in processed_records:
            continue

        # If file exists on disk and is valid, register it
        if validate_feature_file(feature_path):
            sample_id = str(row["id"])
            split_name = split_lookup.get(sample_id, row.get("original_split", "train"))
            processed_records[dataset_idx] = {
                "dataset_index": dataset_idx,
                "id": sample_id,
                "feature_file": feature_filename,
                "feature_dimension": FEATURE_DIM,
                "label": int(row["label"]),
                "split": split_name,
                "source": str(row["source"]),
                "language": str(row["language"]),
                "status": "success",
            }
            continue

        records_to_process.append((dataset_idx, row, feature_filename, feature_path))

    logger.info(f"Already valid & cached: {len(processed_records):,} / {total_samples:,}")
    logger.info(f"Remaining to extract:   {len(records_to_process):,} / {total_samples:,}")

    failures = []
    batch_size = DEFAULT_BATCH_SIZE

    if records_to_process:
        num_batches = (len(records_to_process) + batch_size - 1) // batch_size
        logger.info(f"Starting batched feature extraction: {num_batches} batches (batch size: {batch_size})...")

        for b_idx in range(num_batches):
            batch_slice = records_to_process[b_idx * batch_size : (b_idx + 1) * batch_size]
            batch_indices = [item[0] for item in batch_slice]
            batch_rows = [item[1] for item in batch_slice]
            batch_filenames = [item[2] for item in batch_slice]
            batch_paths = [item[3] for item in batch_slice]

            # Prepare text strings (handle missing / non-string gracefully)
            texts = [str(r["text"]) if pd.notna(r["text"]) else "" for r in batch_rows]

            try:
                # Tokenization
                inputs = tokenizer(
                    texts,
                    padding=True,
                    truncation=True,
                    max_length=MAX_LENGTH,
                    return_tensors="pt",
                )
                inputs = {k: v.to(device) for k, v in inputs.items()}

                # Inference & Masked Mean Pooling
                with torch.inference_mode():
                    outputs = encoder(**inputs)
                    last_hidden = outputs.last_hidden_state  # (B, L, 768)
                    attention_mask = inputs["attention_mask"]  # (B, L)
                    input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
                    sum_embeddings = torch.sum(last_hidden * input_mask_expanded, dim=1)
                    sum_mask = input_mask_expanded.sum(dim=1).clamp(min=1e-9)
                    mean_pooled = (sum_embeddings / sum_mask).float().cpu().numpy()  # (B, 768)

                # Save each embedding atomically
                for i in range(len(batch_slice)):
                    d_idx = batch_indices[i]
                    row_data = batch_rows[i]
                    f_name = batch_filenames[i]
                    f_path = batch_paths[i]
                    emb = mean_pooled[i].astype(np.float32)

                    # Validation check
                    if emb.shape != (FEATURE_DIM,) or not np.all(np.isfinite(emb)):
                        raise ValueError(f"Invalid embedding vector generated for sample index {d_idx}")

                    # Atomic save
                    tmp_path = f_path.with_suffix(".tmp.npy")
                    np.save(tmp_path, emb)
                    os.replace(tmp_path, f_path)

                    sample_id = str(row_data["id"])
                    split_name = split_lookup.get(sample_id, row_data.get("original_split", "train"))

                    processed_records[d_idx] = {
                        "dataset_index": d_idx,
                        "id": sample_id,
                        "feature_file": f_name,
                        "feature_dimension": FEATURE_DIM,
                        "label": int(row_data["label"]),
                        "split": split_name,
                        "source": str(row_data["source"]),
                        "language": str(row_data["language"]),
                        "status": "success",
                    }

            except torch.cuda.OutOfMemoryError:
                logger.warning(f"CUDA Out-of-Memory at batch {b_idx + 1}. Retrying sample-by-sample...")
                torch.cuda.empty_cache()
                for i in range(len(batch_slice)):
                    d_idx = batch_indices[i]
                    row_data = batch_rows[i]
                    f_name = batch_filenames[i]
                    f_path = batch_paths[i]
                    single_text = str(row_data["text"]) if pd.notna(row_data["text"]) else ""

                    try:
                        inp = tokenizer(
                            single_text,
                            padding=True,
                            truncation=True,
                            max_length=MAX_LENGTH,
                            return_tensors="pt",
                        )
                        inp = {k: v.to(device) for k, v in inp.items()}
                        with torch.inference_mode():
                            out = encoder(**inp)
                            lh = out.last_hidden_state
                            m = inp["attention_mask"].unsqueeze(-1).float()
                            emb = ((lh * m).sum(dim=1) / m.sum(dim=1).clamp(min=1e-9)).squeeze(0).float().cpu().numpy()

                        tmp_path = f_path.with_suffix(".tmp.npy")
                        np.save(tmp_path, emb.astype(np.float32))
                        os.replace(tmp_path, f_path)

                        sample_id = str(row_data["id"])
                        split_name = split_lookup.get(sample_id, row_data.get("original_split", "train"))
                        processed_records[d_idx] = {
                            "dataset_index": d_idx,
                            "id": sample_id,
                            "feature_file": f_name,
                            "feature_dimension": FEATURE_DIM,
                            "label": int(row_data["label"]),
                            "split": split_name,
                            "source": str(row_data["source"]),
                            "language": str(row_data["language"]),
                            "status": "success",
                        }
                    except Exception as single_err:
                        logger.error(f"Failed sample index {d_idx} ({row_data.get('id')}): {single_err}")
                        failures.append({
                            "dataset_index": d_idx,
                            "id": str(row_data["id"]),
                            "error": str(single_err)
                        })

            except Exception as batch_err:
                logger.error(f"Batch {b_idx + 1} processing error: {batch_err}", exc_info=True)
                for i in range(len(batch_slice)):
                    failures.append({
                        "dataset_index": batch_indices[i],
                        "id": str(batch_rows[i]["id"]),
                        "error": str(batch_err)
                    })

            # Checkpoint saving
            if (b_idx + 1) % 10 == 0 or (b_idx + 1) == num_batches:
                chk_df = pd.DataFrame(list(processed_records.values())).sort_values("dataset_index")
                chk_df.to_csv(CHECKPOINT_FILE, index=False)
                elapsed = time.time() - start_total_time
                done_cnt = len(processed_records)
                logger.info(f"Progress: [{done_cnt:,}/{total_samples:,}] ({(done_cnt/total_samples)*100:.1f}%) | Elapsed: {elapsed:.1f}s")

    # Final Checkpoint & Metadata
    logger.info("Consolidating final metadata index...")
    final_df = pd.DataFrame(list(processed_records.values())).sort_values("dataset_index")
    final_df.to_csv(FINAL_DATASET_FILE, index=False)
    final_df.to_csv(CHECKPOINT_FILE, index=False)

    if failures:
        pd.DataFrame(failures).to_csv(FAILURES_FILE, index=False)
        logger.warning(f"Saved {len(failures)} failures to {FAILURES_FILE}")
    elif FAILURES_FILE.exists():
        FAILURES_FILE.unlink()

    total_duration = time.time() - start_total_time

    # Split coverage
    split_counts = final_df["split"].value_counts().to_dict()

    summary = {
        "milestone": "Milestone 1: XLM-R Multimodal Text Feature Extraction",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_path": str(MODEL_DIR),
        "source_dataset_rows": total_samples,
        "valid_indexed_embeddings": len(final_df),
        "missing_embeddings": total_samples - len(final_df),
        "failures_count": len(failures),
        "embedding_dimension": FEATURE_DIM,
        "pooling_method": "masked_mean_pooling",
        "max_sequence_length": MAX_LENGTH,
        "device": device_str,
        "execution_time_seconds": round(total_duration, 2),
        "splits_coverage": split_counts,
    }

    with open(SUMMARY_FILE, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info("=" * 75)
    logger.info("   EXTRACTION COMPLETE & METADATA WRITTEN")
    logger.info(f"   Indexed Samples:  {len(final_df):,} / {total_samples:,}")
    logger.info(f"   Failures:         {len(failures)}")
    logger.info(f"   Splits Coverage:  {split_counts}")
    logger.info(f"   Execution Time:   {total_duration:.2f} seconds ({total_duration/60:.2f} minutes)")
    logger.info("=" * 75)


if __name__ == "__main__":
    main()
