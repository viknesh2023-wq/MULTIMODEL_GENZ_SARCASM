import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(r"D:\NLP_PROJECT")

DATASET_CSV = (
    PROJECT_ROOT
    / "FINAL_DATASETS"
    / "MULTIMODAL_SARCASM"
    / "multimodal_sarcasm_combined.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "FINAL_DATASETS"
    / "MULTIMODAL_SARCASM"
    / "CLIP_FEATURES"
)

FEATURE_DIR = OUTPUT_DIR / "features"

CHECKPOINT_FILE = OUTPUT_DIR / "checkpoint.csv"

FINAL_DATASET_FILE = (
    OUTPUT_DIR / "clip_feature_dataset.csv"
)

FAILURE_FILE = (
    OUTPUT_DIR / "clip_feature_failures.csv"
)


# ============================================================
# SETTINGS
# ============================================================

MODEL_NAME = "ViT-B/32"

# None = complete 33,077 dataset
MAX_SAMPLES = None


# ============================================================
# CREATE DIRECTORIES
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

FEATURE_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# SYSTEM INFORMATION
# ============================================================

print()
print("=" * 75)
print("          CLIP ViT-B/32 COMPLETE FEATURE EXTRACTION")
print("=" * 75)

print(
    f"Python executable : {sys.executable}"
)

print(
    f"Python version    : {sys.version.split()[0]}"
)

print(
    f"PyTorch           : {torch.__version__}"
)

print(
    f"CUDA available    : {torch.cuda.is_available()}"
)

if torch.cuda.is_available():

    print(
        f"GPU               : "
        f"{torch.cuda.get_device_name(0)}"
    )

    total_vram = (
        torch.cuda.get_device_properties(0)
        .total_memory
        / (1024 ** 3)
    )

    print(
        f"Total VRAM        : "
        f"{total_vram:.2f} GB"
    )

print("=" * 75)


# ============================================================
# CHECK CUDA
# ============================================================

if not torch.cuda.is_available():

    raise RuntimeError(
        "CUDA GPU is not available."
    )


# ============================================================
# LOAD CLIP
# ============================================================

print()
print("=" * 75)
print("Loading OpenAI CLIP...")
print("=" * 75)

import clip

load_start = time.time()

device = "cuda"

model, preprocess = clip.load(
    MODEL_NAME,
    device=device
)

model.eval()

load_time = (
    time.time()
    - load_start
)

print(
    f"Model load time : "
    f"{load_time:.2f} seconds"
)

print(
    f"Model           : {MODEL_NAME}"
)

print(
    f"Device          : {device}"
)


# ============================================================
# CLIP EMBEDDING DIMENSION
# ============================================================

with torch.inference_mode():

    dummy = torch.zeros(
        1,
        3,
        224,
        224,
        device=device
    )

    embedding_dimension = (
        model.encode_image(dummy)
        .shape[-1]
    )

print(
    f"Embedding dimension : "
    f"{embedding_dimension}"
)

# Expected:
# 512


# ============================================================
# LOAD DATASET
# ============================================================

print()
print("=" * 75)
print("Loading multimodal dataset...")
print("=" * 75)

df = pd.read_csv(
    DATASET_CSV
)

print(
    f"Dataset rows : {len(df):,}"
)

print(
    f"Columns      : {list(df.columns)}"
)


# ============================================================
# COLUMN FINDER
# ============================================================

def find_column(dataframe, candidates):

    columns_lower = {
        str(col).lower().strip(): col
        for col in dataframe.columns
    }

    # Exact match
    for candidate in candidates:

        key = candidate.lower().strip()

        if key in columns_lower:

            return columns_lower[key]

    # Partial match
    for col in dataframe.columns:

        col_lower = (
            str(col).lower().strip()
        )

        for candidate in candidates:

            if candidate.lower() in col_lower:

                return col

    return None


IMAGE_COLUMN = find_column(
    df,
    [
        "image_path",
        "image",
        "img_path",
        "img",
        "image_file",
        "image_filename",
        "filename",
        "file_name",
        "image_name",
        "path",
    ]
)

TEXT_COLUMN = find_column(
    df,
    [
        "text",
        "tweet",
        "caption",
        "sentence",
        "comment",
        "ocr",
    ]
)

LABEL_COLUMN = find_column(
    df,
    [
        "label",
        "sarcasm",
        "sarcastic",
        "is_sarcastic",
        "target",
    ]
)

SPLIT_COLUMN = find_column(
    df,
    [
        "split",
        "dataset_split",
        "original_split",
        "set",
    ]
)

SOURCE_COLUMN = find_column(
    df,
    [
        "source",
        "dataset",
        "source_dataset",
    ]
)


print()
print("Detected columns")
print("-" * 50)

print(
    f"Image : {IMAGE_COLUMN}"
)

print(
    f"Text  : {TEXT_COLUMN}"
)

print(
    f"Label : {LABEL_COLUMN}"
)

print(
    f"Split : {SPLIT_COLUMN}"
)

print(
    f"Source: {SOURCE_COLUMN}"
)

print("-" * 50)


if IMAGE_COLUMN is None:

    raise RuntimeError(
        "Could not find image column."
    )


# ============================================================
# DATASET INDEX
# ============================================================

df = df.reset_index(
    drop=True
)

df["_dataset_index"] = df.index


# ============================================================
# SAMPLE SELECTION
# ============================================================

if MAX_SAMPLES is None:

    work_df = df.copy()

else:

    work_df = df.head(
        MAX_SAMPLES
    ).copy()


TOTAL = len(work_df)

print()
print(
    f"Samples selected : {TOTAL:,}"
)


# ============================================================
# IMAGE DIRECTORIES
# ============================================================

COMMON_IMAGE_DIRECTORIES = [

    PROJECT_ROOT,

    PROJECT_ROOT / "06_MMSD2.0",

    PROJECT_ROOT / "06_MMSD2.0" / "images",

    PROJECT_ROOT / "MEMOTION",

    PROJECT_ROOT / "MEMOTION" / "images",

    PROJECT_ROOT / "FINAL_DATASETS",

    PROJECT_ROOT
    / "FINAL_DATASETS"
    / "MULTIMODAL_SARCASM",

    PROJECT_ROOT
    / "FINAL_DATASETS"
    / "MULTIMODAL_SARCASM"
    / "images",
]


# ============================================================
# IMAGE PATH RESOLUTION
# ============================================================

def resolve_image_path(value):

    if pd.isna(value):

        return None

    raw = str(value).strip()

    if not raw:

        return None


    # --------------------------------------------------------
    # Absolute path
    # --------------------------------------------------------

    path = Path(raw)

    if path.is_absolute():

        if path.exists():

            return path


    # --------------------------------------------------------
    # Relative to dataset directory
    # --------------------------------------------------------

    candidate = (
        DATASET_CSV.parent
        / raw
    )

    if candidate.exists():

        return candidate


    # --------------------------------------------------------
    # Relative to project root
    # --------------------------------------------------------

    candidate = (
        PROJECT_ROOT
        / raw
    )

    if candidate.exists():

        return candidate


    # --------------------------------------------------------
    # Common directories
    # --------------------------------------------------------

    for directory in COMMON_IMAGE_DIRECTORIES:

        candidate = (
            directory
            / raw
        )

        if candidate.exists():

            return candidate


    # --------------------------------------------------------
    # Filename only
    # --------------------------------------------------------

    filename = Path(raw).name

    for directory in COMMON_IMAGE_DIRECTORIES:

        candidate = (
            directory
            / filename
        )

        if candidate.exists():

            return candidate


    # --------------------------------------------------------
    # Recursive search
    # --------------------------------------------------------

    try:

        matches = list(
            PROJECT_ROOT.rglob(filename)
        )

        if len(matches) > 0:

            return matches[0]

    except Exception:

        pass


    return None


# ============================================================
# CHECKPOINT LOADING
# ============================================================

checkpoint_records = []

processed_indices = set()


if CHECKPOINT_FILE.exists():

    try:

        checkpoint_df = pd.read_csv(
            CHECKPOINT_FILE
        )

        checkpoint_records = (
            checkpoint_df
            .to_dict("records")
        )


        if "dataset_index" in checkpoint_df.columns:

            processed_indices = set(

                checkpoint_df[
                    "dataset_index"
                ]
                .dropna()
                .astype(int)
                .tolist()

            )


        print()
        print(
            f"Existing checkpoint records : "
            f"{len(checkpoint_records):,}"
        )

    except Exception as error:

        print()
        print(
            "WARNING: Could not read checkpoint."
        )

        print(error)


# ============================================================
# FAILURE LOADING
# ============================================================

failure_records = []


if FAILURE_FILE.exists():

    try:

        failure_df = pd.read_csv(
            FAILURE_FILE
        )

        failure_records = (
            failure_df
            .to_dict("records")
        )

    except Exception:

        failure_records = []


# ============================================================
# SAVE FUNCTIONS
# ============================================================

def save_checkpoint():

    if len(checkpoint_records) == 0:

        return

    checkpoint_df = pd.DataFrame(
        checkpoint_records
    )

    checkpoint_df.to_csv(
        CHECKPOINT_FILE,
        index=False
    )


def save_failures():

    if len(failure_records) == 0:

        return

    failure_df = pd.DataFrame(
        failure_records
    )

    failure_df.to_csv(
        FAILURE_FILE,
        index=False
    )


# ============================================================
# FEATURE EXTRACTION
# ============================================================

@torch.inference_mode()
def extract_clip_feature(image_path):

    image = Image.open(
        image_path
    ).convert("RGB")

    image_tensor = preprocess(
        image
    ).unsqueeze(0).to(
        device
    )

    feature = model.encode_image(
        image_tensor
    )

    feature = (
        feature
        .float()
        .cpu()
        .numpy()
        .reshape(-1)
    )

    if feature.shape != (
        embedding_dimension,
    ):

        raise RuntimeError(
            "Unexpected CLIP feature "
            f"shape: {feature.shape}"
        )


    if not np.all(
        np.isfinite(feature)
    ):

        raise RuntimeError(
            "CLIP feature contains "
            "NaN or Inf."
        )


    return feature


# ============================================================
# START EXTRACTION
# ============================================================

print()
print("=" * 75)
print("STARTING CLIP FEATURE EXTRACTION")
print("=" * 75)

print(
    f"Total images : {TOTAL:,}"
)

print(
    f"Already done : "
    f"{len(processed_indices):,}"
)

print(
    f"Remaining    : "
    f"{max(0, TOTAL - len(processed_indices)):,}"
)

print("=" * 75)


overall_start = time.time()

success_count = 0

skip_count = 0

failure_count = 0


# ============================================================
# MAIN LOOP
# ============================================================

for position, row in work_df.iterrows():

    dataset_index = int(
        row["_dataset_index"]
    )


    # --------------------------------------------------------
    # IMAGE PATH
    # --------------------------------------------------------

    image_value = row[
        IMAGE_COLUMN
    ]

    image_path = (
        resolve_image_path(
            image_value
        )
    )


    # --------------------------------------------------------
    # FEATURE NAME
    # --------------------------------------------------------

    if image_path is not None:

        image_stem = (
            image_path.stem
        )

    else:

        image_stem = (
            Path(
                str(image_value)
            ).stem
        )


    safe_stem = "".join(

        character

        if (
            character.isalnum()
            or character in "_-"
        )

        else "_"

        for character
        in image_stem

    )


    feature_filename = (
        f"{dataset_index:06d}_"
        f"{safe_stem}.npy"
    )


    feature_path = (
        FEATURE_DIR
        / feature_filename
    )


    # --------------------------------------------------------
    # RESUME CHECK
    # --------------------------------------------------------

    if (
        dataset_index in processed_indices
        and feature_path.exists()
    ):

        skip_count += 1

        continue


    # --------------------------------------------------------
    # IMAGE NOT FOUND
    # --------------------------------------------------------

    if image_path is None:

        failure_count += 1

        failure_records.append({

            "dataset_index":
                dataset_index,

            "image_value":
                str(image_value),

            "error":
                "IMAGE_NOT_FOUND",

        })


        print()

        print(
            f"[{position + 1:,}/{TOTAL:,}] "
            f"FAILED - IMAGE NOT FOUND"
        )

        print(
            f"Image value: "
            f"{image_value}"
        )


        if (
            failure_count % 10
            == 0
        ):

            save_failures()


        continue


    # --------------------------------------------------------
    # EXTRACT
    # --------------------------------------------------------

    start_time = time.time()


    try:

        feature = extract_clip_feature(
            image_path
        )


        # ----------------------------------------------------
        # SAVE FEATURE
        # ----------------------------------------------------

        np.save(
            feature_path,
            feature
        )


        elapsed = (
            time.time()
            - start_time
        )


        # ----------------------------------------------------
        # METADATA
        # ----------------------------------------------------

        record = {

            "dataset_index":
                dataset_index,

            "image_reference":
                str(image_value),

            "image_path":
                str(image_path),

            "feature_file":
                feature_filename,

            "feature_dimension":
                embedding_dimension,

            "status":
                "success",

            "extraction_time_seconds":
                round(
                    elapsed,
                    4
                ),

        }


        if TEXT_COLUMN is not None:

            record["text"] = row[
                TEXT_COLUMN
            ]


        if LABEL_COLUMN is not None:

            record["label"] = row[
                LABEL_COLUMN
            ]


        if SPLIT_COLUMN is not None:

            record["split"] = row[
                SPLIT_COLUMN
            ]


        if SOURCE_COLUMN is not None:

            record["source"] = row[
                SOURCE_COLUMN
            ]


        checkpoint_records.append(
            record
        )

        processed_indices.add(
            dataset_index
        )

        success_count += 1


        # ----------------------------------------------------
        # ETA
        # ----------------------------------------------------

        total_elapsed = (
            time.time()
            - overall_start
        )

        completed_total = (
            success_count
            + skip_count
        )


        if completed_total > 0:

            average_time = (
                total_elapsed
                / completed_total
            )

        else:

            average_time = elapsed


        remaining = (
            TOTAL
            - completed_total
        )


        eta_seconds = (
            remaining
            * average_time
        )


        # ----------------------------------------------------
        # PROGRESS
        # ----------------------------------------------------

        print(

            f"[{position + 1:,}/{TOTAL:,}] "
            f"SUCCESS | "
            f"{elapsed:.2f}s | "
            f"{image_path.name} | "
            f"ETA: "
            f"{eta_seconds / 3600:.2f}h"

        )


        # ----------------------------------------------------
        # CHECKPOINT EVERY 25
        # ----------------------------------------------------

        if (
            success_count % 25
            == 0
        ):

            save_checkpoint()

            save_failures()


    except Exception as error:

        failure_count += 1

        failure_records.append({

            "dataset_index":
                dataset_index,

            "image_value":
                str(image_value),

            "resolved_image_path":
                str(image_path),

            "error":
                repr(error),

        })


        print()

        print(

            f"[{position + 1:,}/{TOTAL:,}] "
            f"FAILED"

        )

        print(
            f"Image: "
            f"{image_path.name}"
        )

        print(
            f"Error: "
            f"{repr(error)}"
        )


        save_failures()


    # --------------------------------------------------------
    # GPU CLEANUP
    # --------------------------------------------------------

    if (
        position % 100
        == 0
    ):

        torch.cuda.empty_cache()


# ============================================================
# FINAL SAVE
# ============================================================

save_checkpoint()

save_failures()


# ============================================================
# FINAL DATASET
# ============================================================

if len(checkpoint_records) > 0:

    final_df = pd.DataFrame(
        checkpoint_records
    )

    final_df = final_df.sort_values(
        "dataset_index"
    )

    final_df.to_csv(
        FINAL_DATASET_FILE,
        index=False
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

total_runtime = (
    time.time()
    - overall_start
)


print()
print()
print("=" * 75)
print("             CLIP EXTRACTION FINISHED")
print("=" * 75)

print(
    f"Dataset rows      : "
    f"{TOTAL:,}"
)

print(
    f"New successful    : "
    f"{success_count:,}"
)

print(
    f"Already existed   : "
    f"{skip_count:,}"
)

print(
    f"Failed            : "
    f"{failure_count:,}"
)

print(
    f"Total runtime     : "
    f"{total_runtime / 3600:.2f} hours"
)

print()
print("OUTPUTS")
print("-" * 75)

print(
    "CLIP .npy features:"
)

print(
    FEATURE_DIR
)

print()

print(
    "Checkpoint:"
)

print(
    CHECKPOINT_FILE
)

print()

print(
    "Final metadata:"
)

print(
    FINAL_DATASET_FILE
)

print()

print(
    "Failed images:"
)

print(
    FAILURE_FILE
)

print("=" * 75)
print("DONE")
print("=" * 75)