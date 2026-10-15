import os
import json
import zipfile
import shutil
import pandas as pd

# ============================================================
# DATASET 2 CREATION
# Multimodal Sarcasm Dataset
#
# Sources:
#   1. MMSD2.0
#   2. MEMOTION 3.0
#
# Final schema:
# id,text,image_path,label,source,language,original_split,
# original_id,text_source,sarcastic_original,humour,
# offensive,motivational,overall
#
# IMPORTANT:
# - This script creates the MASTER Dataset 2 only.
# - It does NOT create train/validation/test splits.
# - Run the split script AFTER this script succeeds.
# ============================================================


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = r"D:\NLP_PROJECT"

# ---------- MMSD2.0 ----------
MMSD_ROOT = os.path.join(
    PROJECT_ROOT,
    "06_MMSD2.0",
    "DATA1"
)

MMSD_TRAIN_JSON = os.path.join(
    MMSD_ROOT,
    "train (1).json"
)

MMSD_VALID_JSON = os.path.join(
    MMSD_ROOT,
    "valid (1).json"
)

MMSD_TEST_JSON = os.path.join(
    MMSD_ROOT,
    "test (1).json"
)

# Possible MMSD image locations
MMSD_IMAGE_ROOTS = [
    os.path.join(MMSD_ROOT, "dataset_image"),
    os.path.join(MMSD_ROOT, "dataset_image_extracted"),
    os.path.join(MMSD_ROOT, "dataset_image", "dataset_image"),
]

# ---------- MEMOTION3.0 ----------
MEMOTION_ROOT = os.path.join(
    PROJECT_ROOT,
    "MEMOTION"
)

MEMOTION_TRAIN_CSV = os.path.join(
    MEMOTION_ROOT,
    "TRAIN_TEST.csv"
)

MEMOTION_VALID_CSV = os.path.join(
    MEMOTION_ROOT,
    "updated_val.csv"
)

MEMOTION_TRAIN_IMAGE_ROOT = os.path.join(
    MEMOTION_ROOT,
    "TRAIN_TEST_IMAGE"
)

MEMOTION_VALID_IMAGE_ROOT = os.path.join(
    MEMOTION_ROOT,
    "valImages"
)

# ---------- FINAL OUTPUT ----------
OUTPUT_ROOT = os.path.join(
    PROJECT_ROOT,
    "FINAL_DATASETS",
    "MULTIMODAL_SARCASM"
)

ANALYSIS_ROOT = os.path.join(
    OUTPUT_ROOT,
    "analysis"
)

MASTER_OUTPUT = os.path.join(
    OUTPUT_ROOT,
    "multimodal_sarcasm_combined.csv"
)

MISSING_IMAGES_OUTPUT = os.path.join(
    ANALYSIS_ROOT,
    "missing_images.csv"
)

EMPTY_TEXT_OUTPUT = os.path.join(
    ANALYSIS_ROOT,
    "empty_text.csv"
)

DUPLICATE_OUTPUT = os.path.join(
    ANALYSIS_ROOT,
    "duplicate_records.csv"
)

VALIDATION_REPORT = os.path.join(
    ANALYSIS_ROOT,
    "dataset2_validation_report.txt"
)


# ============================================================
# 2. SETTINGS
# ============================================================

IMAGE_EXTENSIONS = [
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".JPG",
    ".JPEG",
    ".PNG",
    ".WEBP",
]

FINAL_COLUMNS = [
    "id",
    "text",
    "image_path",
    "label",
    "source",
    "language",
    "original_split",
    "original_id",
    "text_source",
    "sarcastic_original",
    "humour",
    "offensive",
    "motivational",
    "overall",
]


# ============================================================
# 3. UTILITY FUNCTIONS
# ============================================================

def normalize_id(value):
    """
    Convert image IDs such as:
        862902619928506372
        862902619928506372.0
    into a clean string ID.
    """
    if pd.isna(value):
        return ""

    text = str(value).strip()

    if text.endswith(".0"):
        try:
            number = float(text)
            if number.is_integer():
                return str(int(number))
        except Exception:
            pass

    return text


def clean_text(value):
    if pd.isna(value):
        return ""

    return str(value).strip()


def safe_value(value):
    if pd.isna(value):
        return pd.NA

    text = str(value).strip()

    if text == "":
        return pd.NA

    return text


def normalize_sarcasm_label(value):
    """
    MMSD2.0:
        0 = non-sarcastic
        1 = sarcastic

    MEMOTION3.0:
        not_sarcastic -> 0
        general -> 1
        twisted_meaning -> 1
        very_twisted -> 1
    """

    if pd.isna(value):
        return None

    value = str(value).strip().lower()

    # MMSD labels
    if value in ["0", "0.0"]:
        return 0

    if value in ["1", "1.0"]:
        return 1

    # MEMOTION labels
    if value == "not_sarcastic":
        return 0

    if value in [
        "general",
        "twisted_meaning",
        "very_twisted",
    ]:
        return 1

    return None


def find_image(image_id, roots):
    """
    Search recursively for an image whose filename stem
    exactly matches image_id.

    Supports jpg/jpeg/png/webp and case differences.
    """

    image_id = normalize_id(image_id)

    if not image_id:
        return None, []

    candidates = []

    for root in roots:

        if not os.path.isdir(root):
            continue

        # First: direct filename checks
        for ext in IMAGE_EXTENSIONS:
            direct = os.path.join(
                root,
                image_id + ext
            )

            if os.path.isfile(direct):
                candidates.append(os.path.abspath(direct))

        # Second: recursive search
        if not candidates:
            for current_root, _, files in os.walk(root):

                for filename in files:

                    stem, ext = os.path.splitext(filename)

                    if (
                        stem == image_id
                        and ext.lower()
                        in [".jpg", ".jpeg", ".png", ".webp"]
                    ):
                        candidates.append(
                            os.path.abspath(
                                os.path.join(
                                    current_root,
                                    filename
                                )
                            )
                        )

    # Remove duplicates while preserving order
    candidates = list(dict.fromkeys(candidates))

    if not candidates:
        return None, []

    # Deterministic extension priority
    priority = {
        ".jpg": 0,
        ".jpeg": 1,
        ".png": 2,
        ".webp": 3,
    }

    candidates.sort(
        key=lambda x: (
            priority.get(
                os.path.splitext(x)[1].lower(),
                99
            ),
            x.lower()
        )
    )

    return candidates[0], candidates


def try_extract_mmsd_archive():
    """
    If MMSD images were not extracted, attempt to extract
    the main ZIP archive.

    This does NOT delete the original archive.
    """

    possible_zips = []

    for filename in os.listdir(MMSD_ROOT):
        if filename.lower().endswith(".zip"):
            possible_zips.append(
                os.path.join(MMSD_ROOT, filename)
            )

    if not possible_zips:
        return False, "No MMSD ZIP archive found."

    # Prefer an archive containing 'dataset_image'
    possible_zips.sort(
        key=lambda x: (
            "dataset_image" not in os.path.basename(x).lower(),
            x.lower()
        )
    )

    zip_path = possible_zips[0]

    extract_root = os.path.join(
        MMSD_ROOT,
        "dataset_image_extracted"
    )

    os.makedirs(extract_root, exist_ok=True)

    print("\nMMSD image files were not found in the normal folders.")
    print("Attempting to extract:")
    print(zip_path)

    try:
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(extract_root)

        print("MMSD archive extraction completed.")

        # Add extracted root to search list
        if extract_root not in MMSD_IMAGE_ROOTS:
            MMSD_IMAGE_ROOTS.append(extract_root)

        return True, f"Extracted to: {extract_root}"

    except Exception as e:
        return False, (
            "MMSD archive extraction failed: "
            + repr(e)
        )


def load_json(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"MMSD JSON file not found:\n{path}"
        )

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(
            f"Expected a JSON list in:\n{path}"
        )

    return pd.DataFrame(data)


def print_source_report(df, source_name):
    print("\n" + "=" * 70)
    print(source_name)
    print("=" * 70)

    print(f"Rows: {len(df):,}")

    if len(df) == 0:
        return

    print("\nLabels:")
    print(
        df["label"]
        .value_counts()
        .sort_index()
    )

    print("\nLanguage:")
    print(df["language"].value_counts())

    print("\nOriginal split:")
    print(df["original_split"].value_counts())


# ============================================================
# 4. CREATE OUTPUT DIRECTORIES
# ============================================================

os.makedirs(
    OUTPUT_ROOT,
    exist_ok=True
)

os.makedirs(
    ANALYSIS_ROOT,
    exist_ok=True
)


# ============================================================
# 5. CHECK INPUT FILES
# ============================================================

print("=" * 70)
print("DATASET 2 CREATION")
print("=" * 70)

print("\nChecking input files...")

required_files = [
    MMSD_TRAIN_JSON,
    MMSD_VALID_JSON,
    MMSD_TEST_JSON,
    MEMOTION_TRAIN_CSV,
    MEMOTION_VALID_CSV,
]

for path in required_files:

    if os.path.isfile(path):
        print("[OK]", path)

    else:
        print("[MISSING]", path)


missing_input_files = [
    path
    for path in required_files
    if not os.path.isfile(path)
]

if missing_input_files:
    raise FileNotFoundError(
        "\nOne or more required input files are missing:\n"
        + "\n".join(missing_input_files)
        + "\n\nCheck the filenames and extensions."
    )


# ============================================================
# 6. FIND MMSD IMAGES
# ============================================================

print("\n" + "=" * 70)
print("CHECKING MMSD2.0 IMAGE FOLDER")
print("=" * 70)

existing_mmsd_roots = [
    root
    for root in MMSD_IMAGE_ROOTS
    if os.path.isdir(root)
]

for root in existing_mmsd_roots:
    print("[FOUND FOLDER]", root)

if not existing_mmsd_roots:
    print(
        "\nNo normal MMSD image folder was found."
    )

# Load a small sample from train to determine whether images
# can be found before processing everything.
sample_mmsd = load_json(
    MMSD_TRAIN_JSON
).head(20)

sample_found = 0

for image_id in sample_mmsd["image_id"]:
    image_path, _ = find_image(
        image_id,
        MMSD_IMAGE_ROOTS
    )

    if image_path:
        sample_found += 1

print(
    f"\nMMSD sample images found: "
    f"{sample_found}/{len(sample_mmsd)}"
)

if sample_found == 0:

    extracted, message = try_extract_mmsd_archive()

    print(message)

    # Test again
    sample_found = 0

    for image_id in sample_mmsd["image_id"]:
        image_path, _ = find_image(
            image_id,
            MMSD_IMAGE_ROOTS
        )

        if image_path:
            sample_found += 1

    print(
        f"MMSD sample images found after extraction: "
        f"{sample_found}/{len(sample_mmsd)}"
    )


if sample_found == 0:

    raise FileNotFoundError(
        "\nSTOPPED: MMSD images could not be located.\n\n"
        "Your MMSD JSON files are valid, but the corresponding "
        "image files are not currently accessible.\n\n"
        "Check that the MMSD image archive has been extracted "
        "under:\n"
        + MMSD_ROOT
        + "\n\n"
        "The script will NOT create a Dataset 2 containing "
        "only MEMOTION."
    )


# ============================================================
# 7. PROCESS MMSD2.0
# ============================================================

print("\n" + "=" * 70)
print("PROCESSING MMSD2.0")
print("=" * 70)

mmsd_files = [
    (
        "train",
        MMSD_TRAIN_JSON
    ),
    (
        "validation",
        MMSD_VALID_JSON
    ),
    (
        "test",
        MMSD_TEST_JSON
    ),
]

mmsd_records = []
mmsd_missing_images = []
mmsd_empty_text = []
mmsd_ambiguous_images = []

for split_name, json_path in mmsd_files:

    data = load_json(json_path)

    print(
        f"\n{split_name}: "
        f"{len(data):,} rows"
    )

    required = [
        "image_id",
        "text",
        "label",
    ]

    missing_cols = [
        c
        for c in required
        if c not in data.columns
    ]

    if missing_cols:
        raise ValueError(
            f"MMSD {split_name} missing columns: "
            f"{missing_cols}"
        )

    for _, row in data.iterrows():

        original_id = normalize_id(
            row["image_id"]
        )

        text = clean_text(
            row["text"]
        )

        label = normalize_sarcasm_label(
            row["label"]
        )

        image_path, candidates = find_image(
            original_id,
            MMSD_IMAGE_ROOTS
        )

        record_number = (
            len(mmsd_records) + 1
        )

        if text == "":
            mmsd_empty_text.append({
                "source": "MMSD2.0",
                "original_split": split_name,
                "original_id": original_id,
                "reason": "empty_text",
            })

        if image_path is None:
            mmsd_missing_images.append({
                "source": "MMSD2.0",
                "original_split": split_name,
                "original_id": original_id,
                "text": text,
                "reason": "image_not_found",
            })
            continue

        if len(candidates) > 1:
            mmsd_ambiguous_images.append({
                "source": "MMSD2.0",
                "original_split": split_name,
                "original_id": original_id,
                "selected_image": image_path,
                "number_of_candidates": len(candidates),
                "candidates": " | ".join(candidates),
            })

        if label is None:
            print(
                "WARNING: invalid MMSD label:",
                row["label"],
                "ID:",
                original_id
            )
            continue

        if text == "":
            # Text + image architecture requires text.
            continue

        mmsd_records.append({
            "id": f"MM_{record_number:08d}",
            "text": text,
            "image_path": image_path,
            "label": label,
            "source": "MMSD2.0",
            "language": "English",
            "original_split": split_name,
            "original_id": original_id,
            "text_source": "comment",
            "sarcastic_original": str(
                row["label"]
            ).strip(),
            "humour": pd.NA,
            "offensive": pd.NA,
            "motivational": pd.NA,
            "overall": pd.NA,
        })


# ============================================================
# 8. PROCESS MEMOTION3.0
# ============================================================

print("\n" + "=" * 70)
print("PROCESSING MEMOTION3.0")
print("=" * 70)

memotion_files = [
    (
        "train",
        MEMOTION_TRAIN_CSV,
        MEMOTION_TRAIN_IMAGE_ROOT
    ),
    (
        "validation",
        MEMOTION_VALID_CSV,
        MEMOTION_VALID_IMAGE_ROOT
    ),
]

memotion_records = []
memotion_missing_images = []
memotion_empty_text = []
memotion_ambiguous_images = []

for split_name, csv_path, image_root in memotion_files:

    data = pd.read_csv(
        csv_path,
        low_memory=False
    )

    print(
        f"\n{split_name}: "
        f"{len(data):,} rows"
    )

    required = [
        "image_id",
        "humour",
        "sarcastic",
        "offensive",
        "motivational",
        "overall",
        "ocr",
    ]

    missing_cols = [
        c
        for c in required
        if c not in data.columns
    ]

    if missing_cols:
        raise ValueError(
            f"MEMOTION {split_name} missing columns: "
            f"{missing_cols}"
        )

    for _, row in data.iterrows():

        original_id = normalize_id(
            row["image_id"]
        )

        text = clean_text(
            row["ocr"]
        )

        label = normalize_sarcasm_label(
            row["sarcastic"]
        )

        image_path, candidates = find_image(
            original_id,
            [image_root]
        )

        if text == "":
            memotion_empty_text.append({
                "source": "MEMOTION3.0",
                "original_split": split_name,
                "original_id": original_id,
                "reason": "empty_ocr",
            })

        if image_path is None:
            memotion_missing_images.append({
                "source": "MEMOTION3.0",
                "original_split": split_name,
                "original_id": original_id,
                "text": text,
                "reason": "image_not_found",
            })
            continue

        if len(candidates) > 1:
            memotion_ambiguous_images.append({
                "source": "MEMOTION3.0",
                "original_split": split_name,
                "original_id": original_id,
                "selected_image": image_path,
                "number_of_candidates": len(candidates),
                "candidates": " | ".join(candidates),
            })

        if label is None:
            print(
                "WARNING: invalid MEMOTION sarcasm label:",
                row["sarcastic"],
                "ID:",
                original_id
            )
            continue

        if text == "":
            # Text + image architecture requires text.
            continue

        record_number = (
            len(memotion_records) + 1
        )

        memotion_records.append({
            "id": f"ME_{record_number:08d}",
            "text": text,
            "image_path": image_path,
            "label": label,
            "source": "MEMOTION3.0",
            "language": "Hinglish",
            "original_split": split_name,
            "original_id": original_id,
            "text_source": "OCR",
            "sarcastic_original": str(
                row["sarcastic"]
            ).strip(),
            "humour": safe_value(
                row["humour"]
            ),
            "offensive": safe_value(
                row["offensive"]
            ),
            "motivational": safe_value(
                row["motivational"]
            ),
            "overall": safe_value(
                row["overall"]
            ),
        })


# ============================================================
# 9. CREATE DATAFRAMES
# ============================================================

mmsd_df = pd.DataFrame(
    mmsd_records,
    columns=FINAL_COLUMNS
)

memotion_df = pd.DataFrame(
    memotion_records,
    columns=FINAL_COLUMNS
)

combined = pd.concat(
    [mmsd_df, memotion_df],
    ignore_index=True
)


# ============================================================
# 10. CHECK DUPLICATES
# ============================================================

duplicate_mask = combined.duplicated(
    subset=[
        "source",
        "original_split",
        "original_id"
    ],
    keep=False
)

duplicates = combined[
    duplicate_mask
].copy()

if len(duplicates) > 0:
    duplicates.to_csv(
        DUPLICATE_OUTPUT,
        index=False,
        encoding="utf-8-sig"
    )


# ============================================================
# 11. SAVE VALIDATION REPORTS
# ============================================================

missing_images = pd.DataFrame(
    mmsd_missing_images +
    memotion_missing_images
)

empty_text = pd.DataFrame(
    mmsd_empty_text +
    memotion_empty_text
)

ambiguous_images = pd.DataFrame(
    mmsd_ambiguous_images +
    memotion_ambiguous_images
)

if len(missing_images) > 0:
    missing_images.to_csv(
        MISSING_IMAGES_OUTPUT,
        index=False,
        encoding="utf-8-sig"
    )

if len(empty_text) > 0:
    empty_text.to_csv(
        EMPTY_TEXT_OUTPUT,
        index=False,
        encoding="utf-8-sig"
    )


# ============================================================
# 12. FINAL DATA VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("FINAL DATASET 2 VALIDATION")
print("=" * 70)

print(
    f"\nMMSD2.0 valid rows: "
    f"{len(mmsd_df):,}"
)

print(
    f"MEMOTION3.0 valid rows: "
    f"{len(memotion_df):,}"
)

print(
    f"Dataset 2 total: "
    f"{len(combined):,}"
)

print(
    f"\nMissing images: "
    f"{len(missing_images):,}"
)

print(
    f"Empty text/OCR: "
    f"{len(empty_text):,}"
)

print(
    f"Duplicate records: "
    f"{len(duplicates):,}"
)

print(
    f"Ambiguous image matches: "
    f"{len(ambiguous_images):,}"
)


# Check all final required columns
missing_final_columns = [
    c
    for c in FINAL_COLUMNS
    if c not in combined.columns
]

if missing_final_columns:
    raise ValueError(
        "Final dataset missing columns: "
        + str(missing_final_columns)
    )


# Check labels
invalid_labels = combined[
    ~combined["label"].isin([0, 1])
]

if len(invalid_labels) > 0:
    raise ValueError(
        "Invalid labels remain in final Dataset 2."
    )


# Check empty text
invalid_text = combined[
    combined["text"].fillna("").astype(str).str.strip() == ""
]

if len(invalid_text) > 0:
    raise ValueError(
        "Empty text remains in final Dataset 2."
    )


# Check missing image paths
invalid_paths = combined[
    ~combined["image_path"].apply(
        lambda p: isinstance(p, str)
        and os.path.isfile(p)
    )
]

if len(invalid_paths) > 0:
    raise ValueError(
        f"{len(invalid_paths):,} final rows have invalid image paths."
    )


# Check unique IDs
if combined["id"].duplicated().any():
    raise ValueError(
        "Duplicate final IDs detected."
    )


# ============================================================
# 13. PRINT DISTRIBUTIONS
# ============================================================

print("\n" + "-" * 70)
print("SOURCE DISTRIBUTION")
print("-" * 70)

print(
    combined["source"]
    .value_counts()
)

print("\nLABEL DISTRIBUTION")
print("-" * 70)

print(
    combined["label"]
    .value_counts()
    .sort_index()
)

print("\nSOURCE x LABEL")
print("-" * 70)

print(
    pd.crosstab(
        combined["source"],
        combined["label"]
    )
)

print("\nLANGUAGE DISTRIBUTION")
print("-" * 70)

print(
    combined["language"]
    .value_counts()
)

print("\nORIGINAL SPLIT DISTRIBUTION")
print("-" * 70)

print(
    combined["original_split"]
    .value_counts()
)

print("\nTEXT SOURCE DISTRIBUTION")
print("-" * 70)

print(
    combined["text_source"]
    .value_counts()
)


# ============================================================
# 14. SAVE MASTER DATASET
# ============================================================

combined = combined[
    FINAL_COLUMNS
].copy()

combined.to_csv(
    MASTER_OUTPUT,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 15. WRITE COMPLETE REPORT
# ============================================================

with open(
    VALIDATION_REPORT,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "DATASET 2 - MULTIMODAL SARCASM "
        "VALIDATION REPORT\n"
    )

    f.write("=" * 70 + "\n\n")

    f.write(
        "DATASET SOURCES\n"
    )

    f.write(
        "MMSD2.0 + MEMOTION3.0\n\n"
    )

    f.write(
        f"MMSD2.0 valid rows: "
        f"{len(mmsd_df):,}\n"
    )

    f.write(
        f"MEMOTION3.0 valid rows: "
        f"{len(memotion_df):,}\n"
    )

    f.write(
        f"TOTAL valid rows: "
        f"{len(combined):,}\n\n"
    )

    f.write(
        f"Missing images: "
        f"{len(missing_images):,}\n"
    )

    f.write(
        f"Empty text/OCR: "
        f"{len(empty_text):,}\n"
    )

    f.write(
        f"Duplicate records: "
        f"{len(duplicates):,}\n"
    )

    f.write(
        f"Ambiguous image matches: "
        f"{len(ambiguous_images):,}\n\n"
    )

    f.write(
        "SOURCE DISTRIBUTION\n"
    )
    f.write(
        combined["source"]
        .value_counts()
        .to_string()
    )
    f.write("\n\n")

    f.write(
        "LABEL DISTRIBUTION\n"
    )
    f.write(
        combined["label"]
        .value_counts()
        .sort_index()
        .to_string()
    )
    f.write("\n\n")

    f.write(
        "SOURCE x LABEL\n"
    )
    f.write(
        pd.crosstab(
            combined["source"],
            combined["label"]
        ).to_string()
    )
    f.write("\n\n")

    f.write(
        "LANGUAGE DISTRIBUTION\n"
    )
    f.write(
        combined["language"]
        .value_counts()
        .to_string()
    )
    f.write("\n\n")

    f.write(
        "ORIGINAL SPLIT DISTRIBUTION\n"
    )
    f.write(
        combined["original_split"]
        .value_counts()
        .to_string()
    )
    f.write("\n\n")

    f.write(
        "TEXT SOURCE DISTRIBUTION\n"
    )
    f.write(
        combined["text_source"]
        .value_counts()
        .to_string()
    )
    f.write("\n\n")

    f.write(
        "FINAL COLUMNS\n"
    )
    f.write(
        "\n".join(
            FINAL_COLUMNS
        )
    )
    f.write("\n\n")

    f.write(
        "MMSD IMAGE ROOTS SEARCHED\n"
    )

    for root in MMSD_IMAGE_ROOTS:
        f.write(
            root + "\n"
        )


# ============================================================
# 16. FINAL MESSAGE
# ============================================================

print("\n" + "=" * 70)
print("DATASET 2 CREATED SUCCESSFULLY")
print("=" * 70)

print(
    f"\nMASTER DATASET:\n{MASTER_OUTPUT}"
)

print(
    f"\nVALIDATION REPORT:\n{VALIDATION_REPORT}"
)

if os.path.isfile(MISSING_IMAGES_OUTPUT):
    print(
        f"\nMissing-image report:\n"
        f"{MISSING_IMAGES_OUTPUT}"
    )

if os.path.isfile(EMPTY_TEXT_OUTPUT):
    print(
        f"\nEmpty-text report:\n"
        f"{EMPTY_TEXT_OUTPUT}"
    )

print(
    "\nIMPORTANT:"
)
print(
    "Dataset 2 has NOT been split yet."
)
print(
    "Next step: run the Dataset 2 train/validation/test split script."
)
