"""
Dataset 4 — GoEmotions Final Dataset Creation
Project: D:/NLP_PROJECT

Purpose:
- Inspect and standardize the official GoEmotions train/dev/test TSV files.
- Preserve the official train/dev/test split.
- Preserve the original 28 emotion labels.
- Keep multi-label examples as multi-label.
- Do NOT remove the small number of cross-split text duplicates.
- Produce a combined master CSV, split CSVs, and a validation report.

Expected input:
D:/NLP_PROJECT/GoEmotions/
    train.tsv
    dev.tsv
    test.tsv
    emotions

Expected TSV structure:
text<TAB>label_ids<TAB>example_id

Expected emotions file:
one emotion name per line, in label-ID order.
"""

from pathlib import Path
import pandas as pd
import re
from collections import Counter

# ============================================================
# PATHS
# ============================================================

ROOT = Path(r"D:\NLP_PROJECT")
INPUT_DIR = ROOT / "GoEmotions"

OUTPUT_DIR = ROOT / "FINAL_DATASETS" / "EMOTION"
SPLIT_DIR = OUTPUT_DIR / "SPLIT"
ANALYSIS_DIR = OUTPUT_DIR / "analysis"

for d in [OUTPUT_DIR, SPLIT_DIR, ANALYSIS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

FILES = {
    "train": INPUT_DIR / "train.tsv",
    "validation": INPUT_DIR / "dev.tsv",
    "test": INPUT_DIR / "test.tsv",
}
EMOTIONS_FILE = INPUT_DIR / "emotions.txt"

# ============================================================
# HELPERS
# ============================================================

def read_emotions(path):
    labels = []
    with open(path, "r", encoding="utf-8-sig") as f:
        for line in f:
            x = line.strip()
            if x:
                labels.append(x)

    return labels


def read_goemotions_tsv(path, split_name):
    rows = []

    with open(path, "r", encoding="utf-8-sig") as f:
        for line_no, raw in enumerate(f, start=1):
            line = raw.rstrip("\n\r")

            if not line.strip():
                continue

            parts = line.split("\t")

            if len(parts) != 3:
                raise ValueError(
                    f"{path.name}: line {line_no} has {len(parts)} columns; "
                    f"expected 3."
                )

            text, label_ids, example_id = parts

            rows.append({
                "id": example_id.strip(),
                "text": text.strip(),
                "emotion_ids": label_ids.strip(),
                "original_split": split_name,
            })

    return pd.DataFrame(rows)


def parse_label_ids(value):
    if pd.isna(value):
        return []

    value = str(value).strip()

    if not value:
        return []

    # GoEmotions normally uses comma-separated numeric label IDs.
    pieces = [x.strip() for x in value.split(",") if x.strip()]

    ids = []
    for p in pieces:
        if not re.fullmatch(r"\d+", p):
            raise ValueError(f"Invalid emotion ID: {p}")
        ids.append(int(p))

    return ids


# ============================================================
# 1. CHECK INPUTS
# ============================================================

print("=" * 80)
print("DATASET 4 — GOEMOTIONS FINAL CREATION")
print("=" * 80)

print("\n[1/8] Checking input files...")

for name, path in FILES.items():
    if not path.exists():
        raise FileNotFoundError(f"Missing {name} file: {path}")
    print(f"  OK: {path}")

if not EMOTIONS_FILE.exists():
    raise FileNotFoundError(f"Missing emotions file: {EMOTIONS_FILE}")

print(f"  OK: {EMOTIONS_FILE}")

# ============================================================
# 2. LOAD LABEL MAP
# ============================================================

print("\n[2/8] Loading emotion label map...")

emotion_names = read_emotions(EMOTIONS_FILE)

print(f"  Number of emotion labels: {len(emotion_names)}")

if len(emotion_names) != 28:
    raise ValueError(
        f"Expected 28 GoEmotions labels, found {len(emotion_names)}."
    )

id_to_emotion = {
    i: name for i, name in enumerate(emotion_names)
}

for i, name in id_to_emotion.items():
    print(f"  {i:2d} -> {name}")

# ============================================================
# 3. LOAD SPLITS
# ============================================================

print("\n[3/8] Loading train / validation / test...")

frames = []

for split_name, path in FILES.items():
    df = read_goemotions_tsv(path, split_name)
    print(f"  {split_name:10s}: {len(df):,} rows")
    frames.append(df)

master = pd.concat(frames, ignore_index=True)

# ============================================================
# 4. VALIDATION / CLEANING
# ============================================================

print("\n[4/8] Validating and standardizing...")

initial_total = len(master)

# Empty text
empty_text_mask = master["text"].isna() | (
    master["text"].astype(str).str.strip() == ""
)
empty_text_count = int(empty_text_mask.sum())

# Empty IDs
empty_id_mask = master["id"].isna() | (
    master["id"].astype(str).str.strip() == ""
)
empty_id_count = int(empty_id_mask.sum())

if empty_text_count > 0:
    print(f"  Removing empty text rows: {empty_text_count}")
    master = master.loc[~empty_text_mask].copy()

if empty_id_count > 0:
    print(f"  Removing empty ID rows: {empty_id_count}")
    master = master.loc[~empty_id_mask].copy()

# Parse labels
master["emotion_id_list"] = master["emotion_ids"].apply(parse_label_ids)

# Invalid / empty label checks
invalid_label_rows = []
empty_label_rows = []

for idx, ids in master["emotion_id_list"].items():
    if len(ids) == 0:
        empty_label_rows.append(idx)
    elif any(x < 0 or x >= len(emotion_names) for x in ids):
        invalid_label_rows.append(idx)

if empty_label_rows:
    raise ValueError(
        f"Found {len(empty_label_rows)} rows with no emotion label."
    )

if invalid_label_rows:
    raise ValueError(
        f"Found {len(invalid_label_rows)} rows with invalid emotion IDs."
    )

# Convert IDs to names
master["emotion_labels"] = master["emotion_id_list"].apply(
    lambda ids: "|".join(id_to_emotion[i] for i in ids)
)

# Number of labels per example
master["num_emotions"] = master["emotion_id_list"].apply(len)

# This dataset is English
master["language"] = "English"
master["source"] = "GoEmotions"

# Keep original split
master["split"] = master["original_split"]

# Check duplicate IDs
duplicate_id_count = int(
    master["id"].duplicated(keep=False).sum()
)

if duplicate_id_count > 0:
    raise ValueError(
        f"Duplicate example IDs found: {duplicate_id_count} rows."
    )

# Check exact duplicate records within each split
duplicate_record_counts = {}
for split in ["train", "validation", "test"]:
    subset = master[master["split"] == split]
    duplicate_record_counts[split] = int(
        subset.duplicated(subset=["text", "emotion_labels"]).sum()
    )

# Cross-split text overlap
split_text_sets = {
    split: set(master.loc[master["split"] == split, "text"])
    for split in ["train", "validation", "test"]
}

train_val_overlap = split_text_sets["train"] & split_text_sets["validation"]
train_test_overlap = split_text_sets["train"] & split_text_sets["test"]
val_test_overlap = split_text_sets["validation"] & split_text_sets["test"]

unique_cross_split_texts = (
    train_val_overlap
    | train_test_overlap
    | val_test_overlap
)

# Do NOT remove these; preserve official splits.
# They are documented in the report.

# ============================================================
# 5. FINAL COLUMN ORDER
# ============================================================

print("\n[5/8] Preparing final master dataset...")

final_columns = [
    "id",
    "text",
    "emotion_labels",
    "emotion_ids",
    "num_emotions",
    "language",
    "source",
    "original_split",
]

final_master = master[final_columns].copy()

# ============================================================
# 6. SAVE MASTER + SPLITS
# ============================================================

print("\n[6/8] Saving files...")

master_path = OUTPUT_DIR / "emotion_combined.csv"
final_master.to_csv(master_path, index=False, encoding="utf-8-sig")

for split_name, filename in [
    ("train", "train.csv"),
    ("validation", "validation.csv"),
    ("test", "test.csv"),
]:
    split_df = final_master[
        final_master["original_split"] == split_name
    ].copy()

    split_path = SPLIT_DIR / filename
    split_df.to_csv(split_path, index=False, encoding="utf-8-sig")

    print(f"  {split_name:10s}: {len(split_df):,} rows -> {split_path}")

# Save label map
label_map_df = pd.DataFrame({
    "emotion_id": list(id_to_emotion.keys()),
    "emotion": list(id_to_emotion.values())
})
label_map_path = OUTPUT_DIR / "emotion_label_map.csv"
label_map_df.to_csv(
    label_map_path,
    index=False,
    encoding="utf-8-sig"
)

# ============================================================
# 7. ANALYSIS
# ============================================================

print("\n[7/8] Creating analysis report...")

split_counts = final_master["original_split"].value_counts().to_dict()
emotion_counter = Counter()

for ids in final_master["emotion_ids"]:
    for x in parse_label_ids(ids):
        emotion_counter[id_to_emotion[x]] += 1

multi_label_counts = (
    final_master["num_emotions"]
    .value_counts()
    .sort_index()
)

report_lines = []

report_lines.append("=" * 80)
report_lines.append("DATASET 4 — GOEMOTIONS VALIDATION REPORT")
report_lines.append("=" * 80)
report_lines.append("")

report_lines.append("INPUT")
report_lines.append(f"Input directory: {INPUT_DIR}")
report_lines.append("")

report_lines.append("FINAL DATASET")
report_lines.append(f"Initial rows: {initial_total:,}")
report_lines.append(f"Final rows: {len(final_master):,}")
report_lines.append(f"Rows removed for empty text: {empty_text_count:,}")
report_lines.append(f"Rows removed for empty ID: {empty_id_count:,}")
report_lines.append(f"Invalid label rows: {len(invalid_label_rows):,}")
report_lines.append(f"Empty label rows: {len(empty_label_rows):,}")
report_lines.append("")

report_lines.append("SPLIT COUNTS")
for split in ["train", "validation", "test"]:
    report_lines.append(
        f"{split.capitalize():12s}: {split_counts.get(split, 0):,}"
    )
report_lines.append("")

report_lines.append("LABEL INFORMATION")
report_lines.append(f"Number of emotion classes: {len(emotion_names)}")
report_lines.append("Classification type: MULTI-LABEL")
report_lines.append("Language: English")
report_lines.append("Source: GoEmotions")
report_lines.append("")

report_lines.append("EMOTION LABELS")
for i, name in id_to_emotion.items():
    report_lines.append(f"{i:2d}: {name}")
report_lines.append("")

report_lines.append("EMOTION FREQUENCY")
for name in emotion_names:
    report_lines.append(
        f"{name:16s}: {emotion_counter.get(name, 0):,}"
    )
report_lines.append("")

report_lines.append("NUMBER OF EMOTIONS PER EXAMPLE")
for n, count in multi_label_counts.items():
    report_lines.append(f"{int(n)} emotion(s): {int(count):,}")
report_lines.append("")

report_lines.append("DUPLICATE CHECKS")
report_lines.append(
    f"Duplicate IDs across full dataset: {duplicate_id_count:,}"
)
for split in ["train", "validation", "test"]:
    report_lines.append(
        f"Duplicate complete records within {split}: "
        f"{duplicate_record_counts[split]:,}"
    )
report_lines.append("")

report_lines.append("CROSS-SPLIT EXACT TEXT OVERLAP")
report_lines.append(
    "These overlaps are documented but NOT removed, "
    "because the official GoEmotions train/dev/test split is preserved."
)
report_lines.append(
    f"Train <-> Validation: {len(train_val_overlap):,}"
)
report_lines.append(
    f"Train <-> Test:       {len(train_test_overlap):,}"
)
report_lines.append(
    f"Validation <-> Test:  {len(val_test_overlap):,}"
)
report_lines.append(
    f"Unique cross-split text strings: {len(unique_cross_split_texts):,}"
)
report_lines.append("")

report_lines.append("OUTPUT FILES")
report_lines.append(f"Master: {master_path}")
report_lines.append(f"Train: {SPLIT_DIR / 'train.csv'}")
report_lines.append(f"Validation: {SPLIT_DIR / 'validation.csv'}")
report_lines.append(f"Test: {SPLIT_DIR / 'test.csv'}")
report_lines.append(f"Label map: {label_map_path}")
report_lines.append("")

report_lines.append("FINAL STATUS")
report_lines.append("PASSED")
report_lines.append("")
report_lines.append(
    "Dataset 4 preserves GoEmotions as a 28-class multi-label "
    "emotion dataset with the official train/dev/test split."
)
report_lines.append(
    "Cross-split exact-text overlaps were checked and documented; "
    "they were intentionally not removed."
)

report_path = ANALYSIS_DIR / "dataset4_validation_report.txt"

with open(report_path, "w", encoding="utf-8") as f:
    f.write("\n".join(report_lines))

# ============================================================
# 8. FINAL SUMMARY
# ============================================================

print("\n[8/8] COMPLETE")
print("=" * 80)
print(f"Final total       : {len(final_master):,}")
print(f"Train             : {split_counts.get('train', 0):,}")
print(f"Validation        : {split_counts.get('validation', 0):,}")
print(f"Test              : {split_counts.get('test', 0):,}")
print(f"Emotion classes   : {len(emotion_names)}")
print(f"Multi-label       : YES")
print(f"Cross-split text  : {len(unique_cross_split_texts):,} unique")
print("")
print(f"Master             : {master_path}")
print(f"Train              : {SPLIT_DIR / 'train.csv'}")
print(f"Validation         : {SPLIT_DIR / 'validation.csv'}")
print(f"Test               : {SPLIT_DIR / 'test.csv'}")
print(f"Label map          : {label_map_path}")
print(f"Report             : {report_path}")
print("=" * 80)
