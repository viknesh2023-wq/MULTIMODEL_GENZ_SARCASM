import os
import pandas as pd
from pathlib import Path

ROOT = Path(r"D:\NLP_PROJECT")

SENTIMIX_DIR = ROOT / "sentimix"
DRAVIDIAN_DIR = ROOT / "DravidianCodeMix"

print("=" * 80)
print("DATASET 3 — CORRECTED INSPECTION")
print("=" * 80)

# ============================================================
# 1. SENTIMIX
# ============================================================

print("\n" + "=" * 80)
print("1. SENTIMIX FILES")
print("=" * 80)

sentimix_files = list(SENTIMIX_DIR.rglob("*"))

for f in sentimix_files:
    if f.is_file():
        print(f)

print("\n--- Reading SentiMix files ---")

for f in sentimix_files:

    if not f.is_file():
        continue

    if f.suffix.lower() not in [".tsv", ".txt"]:
        continue

    print("\n" + "-" * 70)
    print("FILE:", f.name)
    print("PATH:", f)

    try:
        df = pd.read_csv(
            f,
            sep="\t",
            header=None,
            dtype=str,
            on_bad_lines="skip"
        )

        print("Rows:", len(df))
        print("Columns:", len(df.columns))

        print("\nFirst 5 rows:")
        print(df.head().to_string(index=False))

        print("\nLast column distribution:")

        if len(df.columns) > 0:
            print(
                df.iloc[:, -1]
                .value_counts(dropna=False)
                .to_string()
            )

    except Exception as e:
        print("ERROR:", e)


# ============================================================
# 2. DRAVIDIAN CODE MIX
# ============================================================

print("\n" + "=" * 80)
print("2. DRAVIDIAN CODEMIX FILES")
print("=" * 80)

dravidian_files = []

for f in DRAVIDIAN_DIR.rglob("*"):
    if not f.is_file():
        continue

    name = f.name.lower()

    if "sentiment" in name:
        dravidian_files.append(f)

print("\nSentiment-related files found:")

for f in dravidian_files:
    print(" -", f.name)


# ============================================================
# 3. READ DRAVIDIAN FILES
# ============================================================

print("\n" + "=" * 80)
print("3. READING DRAVIDIAN SENTIMENT FILES")
print("=" * 80)

for f in dravidian_files:

    print("\n" + "-" * 70)
    print("FILE:", f.name)
    print("PATH:", f)

    try:

        # Try semicolon first
        df = pd.read_csv(
            f,
            sep=";",
            header=None,
            dtype=str,
            on_bad_lines="skip"
        )

        # If semicolon did not split correctly,
        # try comma.
        if len(df.columns) <= 1:

            df = pd.read_csv(
                f,
                sep=",",
                header=None,
                dtype=str,
                on_bad_lines="skip"
            )

        print("Rows:", len(df))
        print("Columns:", len(df.columns))

        print("\nFirst 5 rows:")
        print(df.head().to_string(index=False))

        print("\nUnique values by column:")

        for col in df.columns:

            unique_count = df[col].nunique(dropna=True)

            if unique_count <= 20:

                print(
                    f"\nColumn {col} "
                    f"({unique_count} unique values):"
                )

                print(
                    df[col]
                    .value_counts(dropna=False)
                    .to_string()
                )

    except Exception as e:

        print("ERROR:", e)


# ============================================================
# 4. FINISHED
# ============================================================

print("\n" + "=" * 80)
print("INSPECTION COMPLETE")
print("=" * 80)

print("""
IMPORTANT:
Do NOT create Dataset 3 yet.

Paste the complete output from this script into ChatGPT.
""")