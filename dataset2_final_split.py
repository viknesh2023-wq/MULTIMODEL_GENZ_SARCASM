import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

# ============================================================
# DATASET 2 - FINAL TRAIN / VALIDATION / TEST SPLIT
# ============================================================

print("=" * 70)
print("DATASET 2 FINAL SPLIT")
print("=" * 70)

# ------------------------------------------------------------
# 1. PATHS
# ------------------------------------------------------------

INPUT_FILE = Path(
    r"D:\NLP_PROJECT\FINAL_DATASETS\MULTIMODAL_SARCASM"
    r"\multimodal_sarcasm_combined.csv"
)

OUTPUT_DIR = Path(
    r"D:\NLP_PROJECT\FINAL_DATASETS\MULTIMODAL_SARCASM\SPLIT"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# 2. LOAD MASTER DATASET
# ------------------------------------------------------------

print("\nLoading Dataset 2 master...")

df = pd.read_csv(INPUT_FILE)

print(f"[OK] Master dataset loaded")
print(f"Total rows before cleaning: {len(df):,}")


# ------------------------------------------------------------
# 3. REMOVE EMPTY TEXT / OCR
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("REMOVING EMPTY TEXT / OCR")
print("=" * 70)

# Convert missing text to empty string
df["text"] = df["text"].fillna("").astype(str)

# Count empty text
empty_mask = df["text"].str.strip() == ""

empty_count = empty_mask.sum()

print(f"Empty text/OCR records found: {empty_count}")

# Remove only those records
df = df.loc[~empty_mask].copy()

print(f"Rows after removing empty text/OCR: {len(df):,}")

# Expected:
# 33,077 rows


# ------------------------------------------------------------
# 4. BASIC VALIDATION
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("BASIC VALIDATION")
print("=" * 70)

# Convert label to integer
df["label"] = df["label"].astype(int)

# Check source counts
print("\nSOURCE DISTRIBUTION")
print("-" * 70)
print(df["source"].value_counts().to_string())

# Check missing image paths
missing_images = (
    df["image_path"].isna()
    | (df["image_path"].astype(str).str.strip() == "")
).sum()

print(f"\nMissing image paths: {missing_images}")

# Check duplicate records
duplicate_records = df.duplicated().sum()

print(f"Duplicate complete records: {duplicate_records}")


# ------------------------------------------------------------
# 5. SEPARATE MMSD AND MEMOTION
# ------------------------------------------------------------

mmsd = df[df["source"] == "MMSD2.0"].copy()
memotion = df[df["source"] == "MEMOTION3.0"].copy()

print("\n" + "=" * 70)
print("SOURCE COUNTS")
print("=" * 70)

print(f"MMSD2.0     : {len(mmsd):,}")
print(f"MEMOTION3.0 : {len(memotion):,}")
print(f"TOTAL       : {len(df):,}")


# ============================================================
# 6. MMSD2.0
#    KEEP OFFICIAL DATASET SPLITS
# ============================================================

print("\n" + "=" * 70)
print("PROCESSING MMSD2.0")
print("=" * 70)

mmsd_train = mmsd[
    mmsd["original_split"].astype(str).str.lower() == "train"
].copy()

mmsd_val = mmsd[
    mmsd["original_split"].astype(str).str.lower() == "validation"
].copy()

mmsd_test = mmsd[
    mmsd["original_split"].astype(str).str.lower() == "test"
].copy()

print(f"MMSD Train       : {len(mmsd_train):,}")
print(f"MMSD Validation  : {len(mmsd_val):,}")
print(f"MMSD Test        : {len(mmsd_test):,}")


# ============================================================
# 7. MEMOTION
# ============================================================

print("\n" + "=" * 70)
print("PROCESSING MEMOTION3.0")
print("=" * 70)

memotion_train_source = memotion[
    memotion["original_split"].astype(str).str.lower() == "train"
].copy()

memotion_val = memotion[
    memotion["original_split"].astype(str).str.lower() == "validation"
].copy()

print(f"MEMOTION usable TRAIN/TEST source : {len(memotion_train_source):,}")
print(f"MEMOTION usable VALIDATION        : {len(memotion_val):,}")


# ------------------------------------------------------------
# 8. SPLIT MEMOTION TRAIN/TEST
# ------------------------------------------------------------

print("\nSplitting MEMOTION TRAIN/TEST 80/20...")

memotion_train, memotion_test = train_test_split(
    memotion_train_source,
    test_size=0.20,
    random_state=42,
    stratify=memotion_train_source["label"]
)

print(f"MEMOTION Train : {len(memotion_train):,}")
print(f"MEMOTION Test  : {len(memotion_test):,}")


# ============================================================
# 9. COMBINE MMSD + MEMOTION
# ============================================================

print("\n" + "=" * 70)
print("COMBINING FINAL SPLITS")
print("=" * 70)

# TRAIN
train = pd.concat(
    [
        mmsd_train,
        memotion_train
    ],
    ignore_index=True
)

# VALIDATION
validation = pd.concat(
    [
        mmsd_val,
        memotion_val
    ],
    ignore_index=True
)

# TEST
test = pd.concat(
    [
        mmsd_test,
        memotion_test
    ],
    ignore_index=True
    )


# ------------------------------------------------------------
# 10. SHUFFLE EACH SPLIT
# ------------------------------------------------------------

train = train.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)

validation = validation.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)

test = test.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)


# ============================================================
# 11. FINAL COUNTS
# ============================================================

print("\n" + "=" * 70)
print("FINAL SPLIT COUNTS")
print("=" * 70)

print(f"Train       : {len(train):,}")
print(f"Validation  : {len(validation):,}")
print(f"Test        : {len(test):,}")
print(f"TOTAL       : {len(train) + len(validation) + len(test):,}")


# ============================================================
# 12. ACTUAL IMAGE-PATH LEAKAGE CHECK
# ============================================================

print("\n" + "=" * 70)
print("IMAGE-LEVEL LEAKAGE CHECK")
print("=" * 70)


def get_image_keys(data):
    """
    Use the actual stored image path.
    Normalize case and whitespace.
    """
    return set(
        data["image_path"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
    )


train_images = get_image_keys(train)
validation_images = get_image_keys(validation)
test_images = get_image_keys(test)


train_val_overlap = train_images & validation_images
train_test_overlap = train_images & test_images
validation_test_overlap = validation_images & test_images


print(
    f"Train <-> Validation image overlap : "
    f"{len(train_val_overlap)}"
)

print(
    f"Train <-> Test image overlap       : "
    f"{len(train_test_overlap)}"
)

print(
    f"Validation <-> Test image overlap  : "
    f"{len(validation_test_overlap)}"
)


# ============================================================
# 13. DUPLICATE IMAGE CHECK INSIDE EACH SPLIT
# ============================================================

print("\n" + "=" * 70)
print("DUPLICATE IMAGE CHECK")
print("=" * 70)

train_duplicate_images = train["image_path"].astype(str).str.lower().duplicated().sum()
validation_duplicate_images = validation["image_path"].astype(str).str.lower().duplicated().sum()
test_duplicate_images = test["image_path"].astype(str).str.lower().duplicated().sum()

print(f"Train duplicate images      : {train_duplicate_images}")
print(f"Validation duplicate images : {validation_duplicate_images}")
print(f"Test duplicate images       : {test_duplicate_images}")


# ============================================================
# 14. LABEL DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("LABEL DISTRIBUTION")
print("=" * 70)

print("\nTRAIN")
print(train["label"].value_counts().sort_index().to_string())

print("\nVALIDATION")
print(validation["label"].value_counts().sort_index().to_string())

print("\nTEST")
print(test["label"].value_counts().sort_index().to_string())


# ============================================================
# 15. SOURCE DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("SOURCE DISTRIBUTION")
print("=" * 70)

print("\nTRAIN")
print(train["source"].value_counts().to_string())

print("\nVALIDATION")
print(validation["source"].value_counts().to_string())

print("\nTEST")
print(test["source"].value_counts().to_string())


# ============================================================
# 16. LANGUAGE DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("LANGUAGE DISTRIBUTION")
print("=" * 70)

print("\nTRAIN")
print(train["language"].value_counts().to_string())

print("\nVALIDATION")
print(validation["language"].value_counts().to_string())

print("\nTEST")
print(test["language"].value_counts().to_string())


# ============================================================
# 17. SAVE FINAL CSV FILES
# ============================================================

print("\n" + "=" * 70)
print("SAVING FINAL DATASET 2 SPLITS")
print("=" * 70)

train_file = OUTPUT_DIR / "train.csv"
validation_file = OUTPUT_DIR / "validation.csv"
test_file = OUTPUT_DIR / "test.csv"

train.to_csv(
    train_file,
    index=False,
    encoding="utf-8-sig"
)

validation.to_csv(
    validation_file,
    index=False,
    encoding="utf-8-sig"
)

test.to_csv(
    test_file,
    index=False,
    encoding="utf-8-sig"
)

print(f"[SAVED] {train_file}")
print(f"[SAVED] {validation_file}")
print(f"[SAVED] {test_file}")


# ============================================================
# 18. CREATE REPORT
# ============================================================

report_file = OUTPUT_DIR / "dataset2_final_split_report.txt"

report = f"""
======================================================================
DATASET 2 FINAL SPLIT REPORT
======================================================================

INPUT
------
{INPUT_FILE}

FINAL USABLE DATASET
--------------------
Total usable records: {len(df):,}

MMSD2.0:     {len(mmsd):,}
MEMOTION3.0: {len(memotion):,}

Empty text/OCR removed: {empty_count}

FINAL SPLIT
-----------
Train:       {len(train):,}
Validation:  {len(validation):,}
Test:        {len(test):,}
TOTAL:       {len(train) + len(validation) + len(test):,}

MMSD2.0 SPLIT
-------------
Train:       {len(mmsd_train):,}
Validation:  {len(mmsd_val):,}
Test:        {len(mmsd_test):,}

MEMOTION3.0 SPLIT
----------------
Usable TRAIN/TEST source: {len(memotion_train_source):,}
Train:                    {len(memotion_train):,}
Test:                     {len(memotion_test):,}
Validation:               {len(memotion_val):,}

IMAGE LEAKAGE
-------------
Train <-> Validation: {len(train_val_overlap)}
Train <-> Test:       {len(train_test_overlap)}
Validation <-> Test:  {len(validation_test_overlap)}

DUPLICATE IMAGES
----------------
Train:       {train_duplicate_images}
Validation:  {validation_duplicate_images}
Test:        {test_duplicate_images}

DUPLICATE COMPLETE RECORDS IN MASTER
-------------------------------------
{duplicate_records}

MISSING IMAGE PATHS
-------------------
{missing_images}

RANDOM STATE
------------
42

SPLIT METHOD
------------
MMSD2.0 official train/validation/test split preserved.

MEMOTION usable TRAIN/TEST records were split:
80% Train
20% Test

MEMOTION usable validation records were preserved
as Validation.

======================================================================
"""

report_file.write_text(
    report,
    encoding="utf-8"
)

print(f"[SAVED] {report_file}")


# ============================================================
# 19. FINAL STATUS
# ============================================================

print("\n" + "=" * 70)

if (
    len(train_val_overlap) == 0
    and len(train_test_overlap) == 0
    and len(validation_test_overlap) == 0
    and train_duplicate_images == 0
    and validation_duplicate_images == 0
    and test_duplicate_images == 0
):
    print("DATASET 2 FINAL SPLIT VALIDATION: PASSED")
else:
    print("WARNING: DATASET 2 VALIDATION FOUND AN ISSUE")

print("=" * 70)

print("\nDataset 2 processing complete.")