import pandas as pd
from sklearn.model_selection import train_test_split
import os

# ============================================================
# 1. PATHS
# ============================================================

INPUT_FILE = r"D:\NLP_PROJECT\FINAL_DATASETS\TEXT_SARCASM\text_sarcasm_combined.csv"

OUTPUT_DIR = r"D:\NLP_PROJECT\FINAL_DATASETS\TEXT_SARCASM\SPLIT"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 2. LOAD DATASET
# ============================================================

print("Loading dataset...")

df = pd.read_csv(INPUT_FILE)

print("Total rows:", len(df))
print("Columns:")
print(df.columns.tolist())


# ============================================================
# 3. REMOVE EMPTY TEXT
# ============================================================

df["text"] = df["text"].fillna("").astype(str)

df = df[df["text"].str.strip() != ""]

print("\nRows after removing empty text:", len(df))


# ============================================================
# 4. REMOVE DUPLICATE TEXT
# ============================================================

df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)

print("Rows after removing duplicates:", len(df))


# ============================================================
# 5. SEPARATE REAL AND SYNTHETIC DATA
# ============================================================

synthetic_sources = [
    "GenZ-SarC",
    "Multilingual-Synthetic"
]

real_sources = [
    "SARC",
    "SarcasmCorpusV2"
]

real_df = df[df["source"].isin(real_sources)].copy()
synthetic_df = df[df["source"].isin(synthetic_sources)].copy()

print("\nREAL DATA:", len(real_df))
print("SYNTHETIC DATA:", len(synthetic_df))


# ============================================================
# 6. SPLIT REAL DATA
#
# 80% REAL TRAIN
# 10% VALIDATION
# 10% TEST
# ============================================================

real_train, temp = train_test_split(
    real_df,
    test_size=0.20,
    random_state=42,
    stratify=real_df["label"]
)

real_validation, real_test = train_test_split(
    temp,
    test_size=0.50,
    random_state=42,
    stratify=temp["label"]
)


# ============================================================
# 7. ADD SYNTHETIC DATA ONLY TO TRAIN
# ============================================================

train_df = pd.concat(
    [real_train, synthetic_df],
    ignore_index=True
)

validation_df = real_validation.copy()
test_df = real_test.copy()


# ============================================================
# 8. SHUFFLE TRAINING DATA
# ============================================================

train_df = train_df.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)


# ============================================================
# 9. SHUFFLE VALIDATION AND TEST
# ============================================================

validation_df = validation_df.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)

test_df = test_df.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)


# ============================================================
# 10. SAVE FILES
# ============================================================

train_file = os.path.join(
    OUTPUT_DIR,
    "train.csv"
)

validation_file = os.path.join(
    OUTPUT_DIR,
    "validation.csv"
)

test_file = os.path.join(
    OUTPUT_DIR,
    "test.csv"
)

train_df.to_csv(train_file, index=False)
validation_df.to_csv(validation_file, index=False)
test_df.to_csv(test_file, index=False)


# ============================================================
# 11. DISPLAY RESULTS
# ============================================================

print("\n========================================")
print("FINAL DATASET SPLIT")
print("========================================")

print("\nTRAIN:", len(train_df))
print("VALIDATION:", len(validation_df))
print("TEST:", len(test_df))

print("\n----------------------------------------")
print("TRAIN LABEL DISTRIBUTION")
print("----------------------------------------")
print(train_df["label"].value_counts())

print("\n----------------------------------------")
print("VALIDATION LABEL DISTRIBUTION")
print("----------------------------------------")
print(validation_df["label"].value_counts())

print("\n----------------------------------------")
print("TEST LABEL DISTRIBUTION")
print("----------------------------------------")
print(test_df["label"].value_counts())


# ============================================================
# 12. SOURCE DISTRIBUTION
# ============================================================

print("\n========================================")
print("SOURCE DISTRIBUTION")
print("========================================")

print("\nTRAIN SOURCES:")
print(train_df["source"].value_counts())

print("\nVALIDATION SOURCES:")
print(validation_df["source"].value_counts())

print("\nTEST SOURCES:")
print(test_df["source"].value_counts())


# ============================================================
# 13. LANGUAGE DISTRIBUTION
# ============================================================

print("\n========================================")
print("LANGUAGE DISTRIBUTION")
print("========================================")

print("\nTRAIN:")
print(train_df["language"].value_counts())

print("\nVALIDATION:")
print(validation_df["language"].value_counts())

print("\nTEST:")
print(test_df["language"].value_counts())


# ============================================================
# 14. FINAL FILE LOCATIONS
# ============================================================

print("\n========================================")
print("FILES CREATED")
print("========================================")

print(train_file)
print(validation_file)
print(test_file)

print("\nDONE!")