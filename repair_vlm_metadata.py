import re
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(r"D:\NLP_PROJECT")

MASTER_CSV = (
    PROJECT_ROOT
    / "FINAL_DATASETS"
    / "MULTIMODAL_SARCASM"
    / "multimodal_sarcasm_combined.csv"
)

VLM_DIR = (
    PROJECT_ROOT
    / "FINAL_DATASETS"
    / "MULTIMODAL_SARCASM"
    / "VLM_FEATURES"
)

FEATURE_DIR = VLM_DIR / "features"

OUTPUT_METADATA = VLM_DIR / "vlm_feature_dataset.csv"
OUTPUT_CHECKPOINT = VLM_DIR / "checkpoint.csv"


# ============================================================
# LOAD MASTER DATASET
# ============================================================

print()
print("=" * 75)
print("        REPAIRING INTERNVL METADATA")
print("=" * 75)

print()
print("Loading master dataset...")

df = pd.read_csv(MASTER_CSV)

print(f"Master dataset rows : {len(df):,}")


# ============================================================
# PREPARE MASTER DATASET
# ============================================================

df = df.reset_index(drop=True)

df["_dataset_index"] = df.index


# ============================================================
# FIND IMAGE COLUMN
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

        col_lower = str(col).lower().strip()

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
    ],
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
    ],
)

LABEL_COLUMN = find_column(
    df,
    [
        "label",
        "sarcasm",
        "sarcastic",
        "is_sarcastic",
        "target",
    ],
)

SPLIT_COLUMN = find_column(
    df,
    [
        "split",
        "dataset_split",
        "set",
    ],
)

SOURCE_COLUMN = find_column(
    df,
    [
        "source",
        "dataset",
        "source_dataset",
    ],
)


print()
print("Detected columns")
print("-" * 50)
print("Image :", IMAGE_COLUMN)
print("Text  :", TEXT_COLUMN)
print("Label :", LABEL_COLUMN)
print("Split :", SPLIT_COLUMN)
print("Source:", SOURCE_COLUMN)
print("-" * 50)


if IMAGE_COLUMN is None:

    raise RuntimeError(
        "Could not find image column."
    )


# ============================================================
# READ ALL VLM FEATURES
# ============================================================

feature_files = sorted(
    FEATURE_DIR.glob("*.npy")
)

print()
print(
    f"VLM .npy files found : {len(feature_files):,}"
)


# ============================================================
# BUILD INDEX
# ============================================================

feature_map = {}

pattern = re.compile(
    r"^(\d{6})_(.+)\.npy$"
)


invalid_names = []

for feature_file in feature_files:

    match = pattern.match(
        feature_file.name
    )

    if match is None:

        invalid_names.append(
            feature_file.name
        )

        continue

    dataset_index = int(
        match.group(1)
    )

    image_stem = match.group(2)

    feature_map[dataset_index] = {
        "feature_file": feature_file.name,
        "image_stem": image_stem,
        "feature_path": feature_file,
    }


print(
    f"Valid indexed features : "
    f"{len(feature_map):,}"
)

print(
    f"Invalid filenames       : "
    f"{len(invalid_names):,}"
)


# ============================================================
# CHECK DUPLICATE INDICES
# ============================================================

if len(feature_map) != len(
    [f for f in feature_files if pattern.match(f.name)]
):

    print()
    print(
        "WARNING: duplicate dataset indices detected."
    )


# ============================================================
# BUILD CLEAN METADATA
# ============================================================

records = []

missing_features = []

invalid_features = []

extra_features = []

print()
print("Validating features...")
print()


for dataset_index, row in df.iterrows():

    if dataset_index not in feature_map:

        missing_features.append(
            dataset_index
        )

        continue


    info = feature_map[
        dataset_index
    ]

    feature_path = info[
        "feature_path"
    ]


    # --------------------------------------------------------
    # Verify feature
    # --------------------------------------------------------

    try:

        feature = np.load(
            feature_path
        )

        if feature.shape != (1536,):

            invalid_features.append({
                "dataset_index":
                    dataset_index,
                "feature_file":
                    info["feature_file"],
                "shape":
                    str(feature.shape),
            })

            continue


        if not np.all(
            np.isfinite(feature)
        ):

            invalid_features.append({
                "dataset_index":
                    dataset_index,
                "feature_file":
                    info["feature_file"],
                "shape":
                    str(feature.shape),
                "error":
                    "NaN_or_Inf",
            })

            continue


    except Exception as error:

        invalid_features.append({
            "dataset_index":
                dataset_index,
            "feature_file":
                info["feature_file"],
            "error":
                repr(error),
        })

        continue


    # --------------------------------------------------------
    # Create metadata
    # --------------------------------------------------------

    record = {

        "dataset_index":
            int(dataset_index),

        "image_reference":
            str(row[IMAGE_COLUMN]),

        "image_path":
            str(row[IMAGE_COLUMN]),

        "feature_file":
            info["feature_file"],

        "feature_dimension":
            1536,

        "num_tiles":
            None,

        "status":
            "success",

        "raw_vlm_feature_shape":
            "(tiles, 256, 1536)",

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


    records.append(record)


# ============================================================
# FIND EXTRA FEATURES
# ============================================================

master_indices = set(
    range(len(df))
)

feature_indices = set(
    feature_map.keys()
)

extra_indices = (
    feature_indices
    - master_indices
)

for index in sorted(extra_indices):

    extra_features.append(
        feature_map[index][
            "feature_file"
        ]
    )


# ============================================================
# CREATE DATAFRAME
# ============================================================

metadata_df = pd.DataFrame(
    records
)


# ============================================================
# SAVE CLEAN METADATA
# ============================================================

metadata_df.to_csv(
    OUTPUT_METADATA,
    index=False
)

metadata_df.to_csv(
    OUTPUT_CHECKPOINT,
    index=False
)


# ============================================================
# FINAL REPORT
# ============================================================

print()
print("=" * 75)
print("                 REPAIR COMPLETE")
print("=" * 75)

print()
print(
    f"Master dataset rows       : "
    f"{len(df):,}"
)

print(
    f"VLM .npy files             : "
    f"{len(feature_files):,}"
)

print(
    f"Valid indexed features     : "
    f"{len(feature_map):,}"
)

print(
    f"Metadata records created   : "
    f"{len(metadata_df):,}"
)

print(
    f"Missing features           : "
    f"{len(missing_features):,}"
)

print(
    f"Invalid features           : "
    f"{len(invalid_features):,}"
)

print(
    f"Extra feature files        : "
    f"{len(extra_features):,}"
)

print()

print("Expected:")
print("  Master rows     = 33,077")
print("  Metadata rows   = 33,077")
print("  Missing         = 0")
print("  Invalid         = 0")
print()

print("Metadata:")
print(OUTPUT_METADATA)

print()

print("Checkpoint:")
print(OUTPUT_CHECKPOINT)

print()
print("=" * 75)
print("DONE")
print("=" * 75)