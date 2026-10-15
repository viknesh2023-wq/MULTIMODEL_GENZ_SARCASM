"""
Assemble 2,847-D Multimodal Feature Matrices (Milestone 4 - Phase A & B)
Project: Gen-Z Multimodal Sarcasm Detection System
Hardware: NVIDIA GeForce RTX 4050 Laptop GPU / CUDA Acceleration

Reconciles sample identifiers, splits, and labels across all five feature stores:
1. XLM-R Contextual Text (768-D)
2. OpenAI CLIP Visual (512-D)
3. InternVL3-2B Semantic (1,536-D)
4. XLM-R Sentiment Probabilities (3-D)
5. XLM-R Emotion Probabilities (28-D)

Total Concatenated Dimension: 2,847
"""

import sys
import os
import time
import json
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(r"D:\NLP_PROJECT")
DATASET_CSV = PROJECT_ROOT / "FINAL_DATASETS" / "MULTIMODAL_SARCASM" / "multimodal_sarcasm_combined.csv"
SPLIT_DIR = PROJECT_ROOT / "FINAL_DATASETS" / "MULTIMODAL_SARCASM" / "SPLIT"

STORES_DIR = PROJECT_ROOT / "FINAL_DATASETS" / "MULTIMODAL_SARCASM"
XLMR_DIR = STORES_DIR / "XLMR_FEATURES"
CLIP_DIR = STORES_DIR / "CLIP_FEATURES"
VLM_DIR = STORES_DIR / "VLM_FEATURES"
SENT_DIR = STORES_DIR / "SENTIMENT_FEATURES"
EMO_DIR = STORES_DIR / "EMOTION_FEATURES"

OUTPUT_DIR = PROJECT_ROOT / "FINAL_DATASETS" / "MULTIMODAL_SARCASM" / "FUSION_DATA"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = OUTPUT_DIR / "assembly.log"

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
logger = logging.getLogger("Assemble_Fusion")

FEATURE_SPEC = [
    ("text", 768, XLMR_DIR / "features", 0, 768),
    ("clip", 512, CLIP_DIR / "features", 768, 1280),
    ("vlm", 1536, VLM_DIR / "features", 1280, 2816),
    ("sentiment", 3, SENT_DIR / "features", 2816, 2819),
    ("emotion", 28, EMO_DIR / "features", 2819, 2847),
]
TOTAL_DIM = 2847

def load_canonical_splits():
    split_map = {}
    for split_name in ["train", "validation", "test"]:
        sp_file = SPLIT_DIR / f"{split_name}.csv"
        if not sp_file.exists():
            raise FileNotFoundError(f"Split file missing: {sp_file}")
        df_sp = pd.read_csv(sp_file, usecols=["id", "label"])
        for _, r in df_sp.iterrows():
            split_map[str(r["id"])] = {
                "split": split_name,
                "label": int(r["label"])
            }
    return split_map

def main():
    logger.info("=" * 75)
    logger.info("   PHASE A & B: RECONCILE AND ASSEMBLE 2,847-D MULTIMODAL MATRICES")
    logger.info("=" * 75)

    start_time = time.time()

    # 1. Load Master Dataset
    logger.info(f"Loading master multimodal dataset: {DATASET_CSV}")
    master_df = pd.read_csv(DATASET_CSV)
    total_samples = len(master_df)
    logger.info(f"Total master records: {total_samples:,}")

    # 2. Canonical Split Reconciliation
    logger.info(f"Loading authoritative split assignments from: {SPLIT_DIR}")
    canonical_splits = load_canonical_splits()
    logger.info(f"Total canonical split records: {len(canonical_splits):,}")

    if len(canonical_splits) != total_samples:
        raise ValueError(f"Mismatch between master records ({total_samples}) and split records ({len(canonical_splits)})")

    # 3. Load Feature Metadata Files
    logger.info("Verifying metadata files across all 5 stores...")
    xlmr_meta = pd.read_csv(XLMR_DIR / "xlmr_feature_dataset.csv")
    clip_meta = pd.read_csv(CLIP_DIR / "clip_feature_dataset.csv")
    vlm_meta = pd.read_csv(VLM_DIR / "vlm_feature_dataset.csv")
    sent_meta = pd.read_csv(SENT_DIR / "sentiment_feature_dataset.csv")
    emo_meta = pd.read_csv(EMO_DIR / "emotion_feature_dataset.csv")

    meta_dfs = {
        "text": xlmr_meta,
        "clip": clip_meta,
        "vlm": vlm_meta,
        "sentiment": sent_meta,
        "emotion": emo_meta
    }

    for name, df_m in meta_dfs.items():
        if len(df_m) != total_samples:
            raise ValueError(f"Feature store '{name}' row count mismatch: {len(df_m)} != {total_samples}")
        if df_m["dataset_index"].nunique() != total_samples:
            raise ValueError(f"Feature store '{name}' has duplicate dataset_index values!")

    # Verify ID and Label Reconciliation
    logger.info("Checking 1-to-1 sample identity alignment...")
    master_ids = master_df["id"].astype(str).tolist()
    master_labels = master_df["label"].astype(int).tolist()

    reconciliation_errors = []
    for idx in range(total_samples):
        cid = master_ids[idx]
        clabel = master_labels[idx]
        sp_info = canonical_splits.get(cid)
        if not sp_info:
            reconciliation_errors.append(f"Row {idx}: ID {cid} missing from canonical splits!")
        elif sp_info["label"] != clabel:
            reconciliation_errors.append(f"Row {idx}: Label mismatch for ID {cid} ({clabel} vs {sp_info['label']})")

    if reconciliation_errors:
        logger.error(f"Encountered {len(reconciliation_errors)} reconciliation errors!")
        for err in reconciliation_errors[:10]:
            logger.error(f"  - {err}")
        raise ValueError("ID and label reconciliation failed.")

    logger.info("Sample identity, label, and split reconciliation PASSED (100% agreement).")

    # 4. Preallocate Master Matrix
    logger.info(f"Allocating master matrix: ({total_samples}, {TOTAL_DIM}) float32 (~376.7 MB)...")
    X_full = np.empty((total_samples, TOTAL_DIM), dtype=np.float32)
    y_full = np.array(master_labels, dtype=np.int64)

    # 5. Multithreaded Modality-by-Modality Feature Loading
    feature_filenames = emo_meta["feature_file"].tolist()
    num_workers = min(16, os.cpu_count() or 8)
    logger.info(f"Loading modalities using ThreadPoolExecutor with {num_workers} parallel workers...")

    for mod_name, mod_dim, mod_dir, col_start, col_end in FEATURE_SPEC:
        t_mod_start = time.time()
        logger.info(f"Loading modality: '{mod_name}' (dim={mod_dim}, cols [{col_start}:{col_end}]) from {mod_dir.name}...")

        def load_feature(idx_fn):
            idx, fn = idx_fn
            vec = np.load(mod_dir / fn)
            if vec.shape != (mod_dim,):
                raise ValueError(f"Shape error in {fn}: expected ({mod_dim},), got {vec.shape}")
            if vec.dtype != np.float32:
                vec = vec.astype(np.float32)
            X_full[idx, col_start:col_end] = vec

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            list(executor.map(load_feature, enumerate(feature_filenames)))

        t_mod_elapsed = time.time() - t_mod_start
        logger.info(f"Modality '{mod_name}' loaded successfully in {t_mod_elapsed:.2f} seconds.")

    # Validate Entire Matrix
    logger.info("Validating full matrix numerical integrity (NaN, Inf, Finiteness)...")
    if not np.all(np.isfinite(X_full)):
        raise ValueError("Master feature matrix contains non-finite values (NaN or Inf)!")
    logger.info("Full matrix numerical integrity verified: 0 NaNs, 0 Infs.")

    # 6. Partition Into Official Splits
    logger.info("Partitioning into official Train, Validation, and Test splits...")
    train_indices = []
    val_indices = []
    test_indices = []

    for idx in range(total_samples):
        cid = master_ids[idx]
        sp_name = canonical_splits[cid]["split"]
        if sp_name == "train":
            train_indices.append(idx)
        elif sp_name == "validation":
            val_indices.append(idx)
        elif sp_name == "test":
            test_indices.append(idx)
        else:
            raise ValueError(f"Unknown split name: {sp_name}")

    train_indices = np.array(train_indices, dtype=np.int64)
    val_indices = np.array(val_indices, dtype=np.int64)
    test_indices = np.array(test_indices, dtype=np.int64)

    logger.info(f"Split sizes: Train={len(train_indices):,}, Val={len(val_indices):,}, Test={len(test_indices):,}")

    if len(train_indices) != 25384 or len(val_indices) != 3891 or len(test_indices) != 3802:
        raise ValueError("Partitioned split counts do not match expected sizes (25,384 / 3,891 / 3,802)!")

    X_train = X_full[train_indices]
    y_train = y_full[train_indices]

    X_val = X_full[val_indices]
    y_val = y_full[val_indices]

    X_test = X_full[test_indices]
    y_test = y_full[test_indices]

    logger.info(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    logger.info(f"X_val shape:   {X_val.shape}, y_val shape:   {y_val.shape}")
    logger.info(f"X_test shape:  {X_test.shape}, y_test shape:  {y_test.shape}")

    # Check label distributions
    train_labels_dist = {int(k): int(v) for k, v in pd.Series(y_train).value_counts().items()}
    val_labels_dist = {int(k): int(v) for k, v in pd.Series(y_val).value_counts().items()}
    test_labels_dist = {int(k): int(v) for k, v in pd.Series(y_test).value_counts().items()}
    total_labels_dist = {int(k): int(v) for k, v in pd.Series(y_full).value_counts().items()}

    logger.info(f"Train labels (0 / 1): {train_labels_dist}")
    logger.info(f"Val labels (0 / 1):   {val_labels_dist}")
    logger.info(f"Test labels (0 / 1):  {test_labels_dist}")
    logger.info(f"Total labels (0 / 1): {total_labels_dist}")

    # 7. Save Assembled Matrices
    logger.info(f"Saving assembled matrices to: {OUTPUT_DIR}")
    np.save(OUTPUT_DIR / "X_train.npy", X_train)
    np.save(OUTPUT_DIR / "y_train.npy", y_train)

    np.save(OUTPUT_DIR / "X_val.npy", X_val)
    np.save(OUTPUT_DIR / "y_val.npy", y_val)

    np.save(OUTPUT_DIR / "X_test.npy", X_test)
    np.save(OUTPUT_DIR / "y_test.npy", y_test)

    # Save indices and IDs for traceability
    np.save(OUTPUT_DIR / "train_indices.npy", train_indices)
    np.save(OUTPUT_DIR / "val_indices.npy", val_indices)
    np.save(OUTPUT_DIR / "test_indices.npy", test_indices)

    # Save Metadata CSV
    fusion_meta = pd.DataFrame({
        "dataset_index": list(range(total_samples)),
        "id": master_ids,
        "split": [canonical_splits[cid]["split"] for cid in master_ids],
        "label": master_labels,
        "feature_file": feature_filenames,
        "source": master_df["source"].tolist(),
        "image_path": master_df["image_path"].tolist(),
    })
    fusion_meta.to_csv(OUTPUT_DIR / "fusion_metadata.csv", index=False)
    logger.info(f"Saved fusion_metadata.csv ({len(fusion_meta):,} rows)")

    # 8. Save Assembly Summary
    elapsed_time = time.time() - start_time
    summary = {
        "task": "Milestone 4 - Assemble 2,847-D Multimodal Feature Matrices",
        "total_samples": total_samples,
        "feature_dimension": TOTAL_DIM,
        "modality_slices": {
            "text": {"start": 0, "end": 768, "dim": 768},
            "clip": {"start": 768, "end": 1280, "dim": 512},
            "vlm": {"start": 1280, "end": 2816, "dim": 1536},
            "sentiment": {"start": 2816, "end": 2819, "dim": 3},
            "emotion": {"start": 2819, "end": 2847, "dim": 28},
        },
        "splits": {
            "train": {
                "samples": int(len(train_indices)),
                "matrix_shape": list(X_train.shape),
                "labels": train_labels_dist,
            },
            "validation": {
                "samples": int(len(val_indices)),
                "matrix_shape": list(X_val.shape),
                "labels": val_labels_dist,
            },
            "test": {
                "samples": int(len(test_indices)),
                "matrix_shape": list(X_test.shape),
                "labels": test_labels_dist,
            },
        },
        "dtype": "float32",
        "total_labels": total_labels_dist,
        "assembly_time_seconds": round(elapsed_time, 2),
    }

    with open(OUTPUT_DIR / "assembly_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Assembly completed successfully in {elapsed_time:.2f} seconds!")
    logger.info("=" * 75)

if __name__ == "__main__":
    main()
