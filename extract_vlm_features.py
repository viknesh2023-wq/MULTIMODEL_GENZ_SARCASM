import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from transformers import AutoModel, AutoTokenizer
from torchvision import transforms

warnings.filterwarnings("ignore")


# ============================================================
# 1. CONFIGURATION
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
    / "VLM_FEATURES"
)

FEATURE_DIR = OUTPUT_DIR / "features"

CHECKPOINT_FILE = OUTPUT_DIR / "checkpoint.csv"

FINAL_DATASET_FILE = OUTPUT_DIR / "vlm_feature_dataset.csv"

FAILURE_FILE = OUTPUT_DIR / "vlm_feature_failures.csv"


# ============================================================
# COMPLETE DATASET
# ============================================================

MAX_SAMPLES = None


# ============================================================
# INTERNVL
# ============================================================

MODEL_NAME = "OpenGVLab/InternVL3-2B"

IMAGE_SIZE = 448

MAX_NUM = 12


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
print("        INTERNVL3-2B COMPLETE FEATURE EXTRACTION")
print("=" * 75)

print(f"Python executable : {sys.executable}")
print(f"Python version    : {sys.version.split()[0]}")
print(f"PyTorch           : {torch.__version__}")

import transformers

print(f"Transformers      : {transformers.__version__}")

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
        "\nCUDA GPU is not available.\n"
        "This script requires your NVIDIA GPU."
    )


# ============================================================
# CHECK DATASET
# ============================================================

if not DATASET_CSV.exists():

    raise FileNotFoundError(
        f"\nDataset not found:\n{DATASET_CSV}"
    )


print()
print("Loading multimodal dataset...")

df = pd.read_csv(
    DATASET_CSV
)

print(
    f"Total dataset rows : {len(df):,}"
)

print(
    f"Columns            : "
    f"{list(df.columns)}"
)


# ============================================================
# FIND COLUMNS
# ============================================================

def find_column(dataframe, candidates):

    columns_lower = {
        str(col).lower().strip(): col
        for col in dataframe.columns
    }

    # Exact match
    for candidate in candidates:

        candidate_lower = (
            candidate.lower().strip()
        )

        if candidate_lower in columns_lower:

            return columns_lower[
                candidate_lower
            ]

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
    f"Image column : {IMAGE_COLUMN}"
)

print(
    f"Text column  : {TEXT_COLUMN}"
)

print(
    f"Label column : {LABEL_COLUMN}"
)

print(
    f"Split column : {SPLIT_COLUMN}"
)

print(
    f"Source       : {SOURCE_COLUMN}"
)

print("-" * 50)


if IMAGE_COLUMN is None:

    raise ValueError(
        "\nCould not find the image column.\n"
        "Check the column names printed above."
    )


# ============================================================
# USE COMPLETE DATASET
# ============================================================

if MAX_SAMPLES is None:

    work_df = df.copy()

else:

    work_df = df.head(
        MAX_SAMPLES
    ).copy()


work_df = work_df.reset_index(
    drop=False
)

work_df.rename(
    columns={
        "index": "_original_index"
    },
    inplace=True
)


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

    PROJECT_ROOT / "MEMOTION",

    PROJECT_ROOT / "MEMOTION" / "images",

    PROJECT_ROOT / "06_MMSD2.0" / "images",

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
# IMAGE TRANSFORM
# ============================================================

IMAGENET_MEAN = (
    0.485,
    0.456,
    0.406
)

IMAGENET_STD = (
    0.229,
    0.224,
    0.225
)


transform = transforms.Compose([

    transforms.Lambda(
        lambda image:
        image.convert("RGB")
    ),

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE),
        interpolation=
        transforms.InterpolationMode.BICUBIC
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD
    ),

])


# ============================================================
# INTERNVL DYNAMIC TILING
# ============================================================

def dynamic_preprocess(
    image,
    image_size=448,
    max_num=12,
    use_thumbnail=True
):

    orig_width, orig_height = (
        image.size
    )

    aspect_ratio = (
        orig_width
        / orig_height
    )


    target_ratios = set(

        (i, j)

        for n in range(
            1,
            max_num + 1
        )

        for i in range(
            1,
            n + 1
        )

        for j in range(
            1,
            n + 1
        )

        if (
            1
            <= i * j
            <= max_num
        )

    )


    target_ratios = sorted(
        target_ratios,
        key=lambda x:
        x[0] * x[1]
    )


    def find_best_ratio():

        best_ratio_diff = (
            float("inf")
        )

        best_ratio = (
            1,
            1
        )

        area = (
            orig_width
            * orig_height
        )


        for ratio in target_ratios:

            target_aspect_ratio = (
                ratio[0]
                / ratio[1]
            )

            ratio_diff = abs(
                aspect_ratio
                - target_aspect_ratio
            )


            if ratio_diff < best_ratio_diff:

                best_ratio_diff = (
                    ratio_diff
                )

                best_ratio = ratio


            elif (
                ratio_diff
                == best_ratio_diff
            ):

                target_area = (
                    image_size
                    * image_size
                    * ratio[0]
                    * ratio[1]
                )

                if (
                    area
                    > 0.5 * target_area
                    and
                    area
                    < 2 * target_area
                ):

                    best_ratio = ratio


        return best_ratio


    target_ratio = (
        find_best_ratio()
    )


    target_width = (
        image_size
        * target_ratio[0]
    )

    target_height = (
        image_size
        * target_ratio[1]
    )


    resized_img = image.resize(
        (
            target_width,
            target_height
        )
    )


    processed_images = []


    total_tiles = (
        target_ratio[0]
        * target_ratio[1]
    )


    for i in range(
        total_tiles
    ):

        box = (

            (i % target_ratio[0])
            * image_size,

            (i // target_ratio[0])
            * image_size,

            ((i % target_ratio[0]) + 1)
            * image_size,

            ((i // target_ratio[0]) + 1)
            * image_size,

        )


        split_img = (
            resized_img.crop(box)
        )


        processed_images.append(
            split_img
        )


    # Add thumbnail
    if (
        use_thumbnail
        and
        len(processed_images) != 1
    ):

        thumbnail = image.resize(
            (
                image_size,
                image_size
            )
        )

        processed_images.append(
            thumbnail
        )


    pixel_values = torch.stack(

        [
            transform(img)
            for img in processed_images
        ]

    )


    return pixel_values


# ============================================================
# LOAD INTERNVL3-2B
# ============================================================

print()
print("=" * 75)
print("Loading InternVL3-2B...")
print("=" * 75)

load_start = time.time()


model = AutoModel.from_pretrained(

    MODEL_NAME,

    torch_dtype=torch.bfloat16,

    low_cpu_mem_usage=True,

    trust_remote_code=True,

).eval().cuda()


tokenizer = AutoTokenizer.from_pretrained(

    MODEL_NAME,

    trust_remote_code=True,

    use_fast=False

)


load_time = (
    time.time()
    - load_start
)


print()
print(
    f"Model load time : "
    f"{load_time:.2f} seconds"
)

print(
    f"Model dtype     : "
    f"{next(model.parameters()).dtype}"
)

print(
    f"Model device    : "
    f"{next(model.parameters()).device}"
)


if torch.cuda.is_available():

    allocated = (
        torch.cuda.memory_allocated()
        / (1024 ** 3)
    )

    reserved = (
        torch.cuda.memory_reserved()
        / (1024 ** 3)
    )

    print(
        f"GPU allocated   : "
        f"{allocated:.2f} GB"
    )

    print(
        f"GPU reserved    : "
        f"{reserved:.2f} GB"
    )


print("=" * 75)


# ============================================================
# LOAD EXISTING CHECKPOINT
# ============================================================

checkpoint_records = []

processed_feature_files = set()


if CHECKPOINT_FILE.exists():

    try:

        checkpoint_df = pd.read_csv(
            CHECKPOINT_FILE
        )

        checkpoint_records = (
            checkpoint_df
            .to_dict("records")
        )


        if (
            "feature_file"
            in checkpoint_df.columns
        ):

            processed_feature_files = set(

                checkpoint_df[
                    "feature_file"
                ]
                .dropna()
                .astype(str)

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
# LOAD EXISTING FAILURES
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
# INTERNVL FEATURE EXTRACTION
# ============================================================

@torch.inference_mode()
def extract_vlm_feature(
    image_path
):

    image = Image.open(
        image_path
    ).convert("RGB")


    pixel_values = dynamic_preprocess(

        image,

        image_size=IMAGE_SIZE,

        max_num=MAX_NUM,

        use_thumbnail=True

    )


    pixel_values = pixel_values.to(

        device="cuda",

        dtype=torch.bfloat16

    )


    # --------------------------------------------------------
    # INTERNVL FEATURE EXTRACTION
    # --------------------------------------------------------

    vit_embeds = (
        model.extract_feature(
            pixel_values
        )
    )


    # Expected shape:
    #
    # number_of_tiles × 256 × 1536
    #
    # Example:
    #
    # 3 × 256 × 1536
    # --------------------------------------------------------

    if vit_embeds.ndim != 3:

        raise RuntimeError(

            "Unexpected InternVL "
            f"feature shape: "
            f"{tuple(vit_embeds.shape)}"

        )


    # --------------------------------------------------------
    # Average visual tokens
    # --------------------------------------------------------

    tile_features = (
        vit_embeds.mean(
            dim=1
        )
    )


    # --------------------------------------------------------
    # Average tiles
    # --------------------------------------------------------

    image_feature = (
        tile_features.mean(
            dim=0
        )
    )


    # --------------------------------------------------------
    # CPU + float32 + NumPy
    # --------------------------------------------------------

    image_feature = (

        image_feature

        .float()

        .cpu()

        .numpy()

    )


    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    if image_feature.shape != (
        1536,
    ):

        raise RuntimeError(

            "Unexpected final "
            f"feature shape: "
            f"{image_feature.shape}"

        )


    if not np.all(
        np.isfinite(
            image_feature
        )
    ):

        raise RuntimeError(
            "Feature contains NaN "
            "or Inf values."
        )


    return (
        image_feature,
        tuple(vit_embeds.shape)
    )


# ============================================================
# START EXTRACTION
# ============================================================

print()
print("=" * 75)
print("STARTING COMPLETE DATASET EXTRACTION")
print("=" * 75)

print(
    f"Total images : {TOTAL:,}"
)

print(
    f"Already done : "
    f"{len(processed_feature_files):,}"
)

print(
    f"Remaining    : "
    f"{max(0, TOTAL - len(processed_feature_files)):,}"
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

    original_index = int(
        row["_original_index"]
    )


    image_value = row[
        IMAGE_COLUMN
    ]


    # --------------------------------------------------------
    # Resolve image
    # --------------------------------------------------------

    image_path = (
        resolve_image_path(
            image_value
        )
    )


    # --------------------------------------------------------
    # Feature filename
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

        f"{original_index:06d}"
        f"_{safe_stem}.npy"

    )


    feature_path = (
        FEATURE_DIR
        / feature_filename
    )


    # --------------------------------------------------------
    # RESUME CHECK
    # --------------------------------------------------------

    if (

        feature_filename
        in processed_feature_files

        and

        feature_path.exists()

    ):

        skip_count += 1

        continue


    # --------------------------------------------------------
    # IMAGE NOT FOUND
    # --------------------------------------------------------

    if image_path is None:

        failure_count += 1


        failure_records.append({

            "row_index":
                original_index,

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
    # EXTRACT FEATURE
    # --------------------------------------------------------

    start_time = time.time()


    try:

        feature, raw_shape = (
            extract_vlm_feature(
                image_path
            )
        )


        # ----------------------------------------------------
        # SAVE NUMPY FEATURE
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
        # CREATE METADATA RECORD
        # ----------------------------------------------------

        record = {

            "row_index":
                original_index,

            "image_path":
                str(image_path),

            "feature_file":
                feature_filename,

            "feature_dimension":
                1536,

            "raw_vlm_feature_shape":
                str(raw_shape),

            "extraction_time_seconds":
                round(
                    elapsed,
                    4
                ),

        }


        # ----------------------------------------------------
        # TEXT
        # ----------------------------------------------------

        if TEXT_COLUMN is not None:

            record["text"] = str(
                row[TEXT_COLUMN]
            )


        # ----------------------------------------------------
        # LABEL
        # ----------------------------------------------------

        if LABEL_COLUMN is not None:

            record["label"] = row[
                LABEL_COLUMN
            ]


        # ----------------------------------------------------
        # SPLIT
        # ----------------------------------------------------

        if SPLIT_COLUMN is not None:

            record["split"] = row[
                SPLIT_COLUMN
            ]


        # ----------------------------------------------------
        # SOURCE
        # ----------------------------------------------------

        if SOURCE_COLUMN is not None:

            record["source"] = row[
                SOURCE_COLUMN
            ]


        checkpoint_records.append(
            record
        )


        processed_feature_files.add(
            feature_filename
        )


        success_count += 1


        # ----------------------------------------------------
        # TIME / ETA
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
        # CHECKPOINT EVERY 10
        # ----------------------------------------------------

        if (
            success_count % 10
            == 0
        ):

            save_checkpoint()

            save_failures()


    except Exception as error:

        failure_count += 1


        failure_records.append({

            "row_index":
                original_index,

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
        position % 50
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
print("                 EXTRACTION FINISHED")
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
    "VLM .npy features:"
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
    "Final metadata dataset:"
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