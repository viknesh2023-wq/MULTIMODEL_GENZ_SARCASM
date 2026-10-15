import pandas as pd
from pathlib import Path
import re


# ============================================================
# 1. PROJECT CONFIGURATION
# ============================================================

BASE_DIR = Path(r"D:\NLP_PROJECT")

# These are the folders we know from your project
SARC_DIR = BASE_DIR / "sarc"
SCV_DIR = BASE_DIR / "isarcasm_replacement"
GENZ_DIR = BASE_DIR / "GenZ-Sarc"

# Output folder
OUTPUT_DIR = BASE_DIR / "FINAL_DATASETS" / "TEXT_SARCASM"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. FINAL DATASET COLUMNS
# ============================================================

FINAL_COLUMNS = [
    "id",
    "text",
    "label",
    "language",
    "source",
    "style",
    "topic",
    "slang_used",
    "abbreviations_used",
    "emoji_used"
]


# ============================================================
# 3. HELPER FUNCTIONS
# ============================================================

def print_section(title):
    print("\n")
    print("=" * 80)
    print(title)
    print("=" * 80)


def clean_text(text):
    """
    Basic cleaning only.

    We intentionally KEEP:
    - slang
    - abbreviations
    - emojis
    - punctuation
    """

    if pd.isna(text):
        return ""

    text = str(text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def duplicate_key(text):
    """
    Temporary normalized version used only
    for detecting duplicate text.
    """

    if pd.isna(text):
        return ""

    text = str(text).lower()

    text = re.sub(r"\s+", " ", text)

    return text.strip()


def normalize_label(value):
    """
    Convert all dataset labels to:

    0 = Non-Sarcastic
    1 = Sarcastic
    """

    if pd.isna(value):
        return None

    value = str(value).strip().lower()

    label_map = {

        "0": 0,
        "1": 1,

        "sarc": 1,
        "sarcastic": 1,
        "yes": 1,
        "true": 1,

        "notsarc": 0,
        "not_sarc": 0,
        "non-sarcastic": 0,
        "non_sarcastic": 0,
        "non sarcastic": 0,
        "not sarcastic": 0,
        "no": 0,
        "false": 0
    }

    return label_map.get(value, None)


def find_column(df, possible_names):

    lookup = {
        str(column).strip().lower(): column
        for column in df.columns
    }

    for name in possible_names:

        name = name.lower()

        if name in lookup:
            return lookup[name]

    return None


# ============================================================
# 4. CHECK KNOWN DATASET FOLDERS
# ============================================================

print_section("CHECKING PROJECT FOLDERS")


print("Project folder:")
print(BASE_DIR)


if not BASE_DIR.exists():

    raise FileNotFoundError(
        f"\nProject folder does not exist:\n{BASE_DIR}"
    )


known_folders = {
    "SARC": SARC_DIR,
    "Sarcasm Corpus V2": SCV_DIR,
    "GenZ-SarC": GENZ_DIR
}


for name, folder in known_folders.items():

    print(f"\n{name}:")
    print(folder)

    if folder.exists():

        print("STATUS: FOUND")

    else:

        print("STATUS: NOT FOUND")

        raise FileNotFoundError(
            f"\nRequired folder not found:\n{folder}"
        )


# ============================================================
# 5. LOAD SARC
# ============================================================

print_section("1. LOADING SARC")


sarc_files = []


for file in SARC_DIR.rglob("*"):

    if not file.is_file():
        continue

    filename = file.name.lower()

    if "train-balanced" in filename:

        sarc_files.append(file)


if len(sarc_files) == 0:

    raise FileNotFoundError(
        f"""
Could not find the SARC train-balanced file.

Expected it somewhere inside:
{SARC_DIR}
"""
    )


sarc_file = sarc_files[0]


print("Using SARC file:")
print(sarc_file)


# SARC file structure
sarc_columns = [
    "label",
    "text",
    "author",
    "subreddit",
    "score",
    "ups",
    "downs",
    "date",
    "created_utc",
    "parent_comment"
]


sarc = pd.read_csv(
    sarc_file,
    sep="\t",
    header=None,
    names=sarc_columns,
    on_bad_lines="skip"
)


print("\nSARC rows loaded:")
print(len(sarc))


# ------------------------------------------------------------
# Standardize SARC
# ------------------------------------------------------------

sarc_final = pd.DataFrame()


sarc_final["id"] = [
    f"SARC_{i:06d}"
    for i in range(1, len(sarc) + 1)
]


sarc_final["text"] = (
    sarc["text"]
    .apply(clean_text)
)


sarc_final["label"] = (
    sarc["label"]
    .apply(normalize_label)
)


sarc_final["language"] = "English"


sarc_final["source"] = "SARC"


# SARC does not provide sarcasm style
sarc_final["style"] = pd.NA


# Preserve subreddit as topic
sarc_final["topic"] = (
    sarc["subreddit"]
    .replace("", pd.NA)
)


# These are not explicitly available
sarc_final["slang_used"] = pd.NA

sarc_final["abbreviations_used"] = pd.NA

sarc_final["emoji_used"] = pd.NA


sarc_final = sarc_final[FINAL_COLUMNS]


print("\nSARC standardized successfully.")


# ============================================================
# 6. LOAD SARCASM CORPUS V2
# ============================================================

print_section("2. LOADING SARCASM CORPUS V2")


scv_file_names = [
    "GEN-sarc-notsarc.csv",
    "HYP-sarc-notsarc.csv",
    "RQ-sarc-notsarc.csv"
]


scv_frames = []


for file_name in scv_file_names:

    file_path = SCV_DIR / file_name


    if not file_path.exists():

        raise FileNotFoundError(
            f"\nSarcasm Corpus V2 file not found:\n{file_path}"
        )


    print("\nLoading:")
    print(file_name)


    df = pd.read_csv(
        file_path
    )


    print("Rows:", len(df))

    print("Columns:", list(df.columns))


    # --------------------------------------------------------
    # Determine style
    # --------------------------------------------------------

    if file_name.startswith("GEN"):

        style = "GEN"

    elif file_name.startswith("HYP"):

        style = "HYP"

    elif file_name.startswith("RQ"):

        style = "RQ"

    else:

        style = pd.NA


    # --------------------------------------------------------
    # Find text column
    # --------------------------------------------------------

    text_column = find_column(
        df,
        ["text"]
    )


    # Find label column
    label_column = find_column(
        df,
        ["class", "label"]
    )


    if text_column is None:

        raise ValueError(
            f"\nTEXT column not found in:\n{file_name}"
        )


    if label_column is None:

        raise ValueError(
            f"\nLABEL column not found in:\n{file_name}"
        )


    # --------------------------------------------------------
    # Standardize
    # --------------------------------------------------------

    temp = pd.DataFrame()


    temp["id"] = [
        f"SCV_{style}_{i:06d}"
        for i in range(1, len(df) + 1)
    ]


    temp["text"] = (
        df[text_column]
        .apply(clean_text)
    )


    temp["label"] = (
        df[label_column]
        .apply(normalize_label)
    )


    temp["language"] = "English"


    temp["source"] = "SarcasmCorpusV2"


    temp["style"] = style


    # Topic not available
    temp["topic"] = pd.NA


    # These fields are not explicitly provided
    temp["slang_used"] = pd.NA

    temp["abbreviations_used"] = pd.NA

    temp["emoji_used"] = pd.NA


    temp = temp[FINAL_COLUMNS]


    scv_frames.append(temp)


# Combine GEN + HYP + RQ

scv_final = pd.concat(
    scv_frames,
    ignore_index=True
)


print("\nSarcasm Corpus V2 combined successfully.")

print(
    "Total rows:",
    len(scv_final)
)


# ============================================================
# 7. LOAD ALL GENZ-SARC FILES
# ============================================================

print_section("3. LOADING GENZ-SARC")


all_genz_files = list(
    GENZ_DIR.rglob("*")
)


genz_files = []


for file in all_genz_files:

    if not file.is_file():
        continue

    if file.suffix.lower() != ".csv":
        continue

    if "genz_sarc_batch" not in file.name.lower():
        continue

    genz_files.append(file)


if len(genz_files) == 0:

    raise FileNotFoundError(
        f"""
No GenZ-SarC batch files found.

Expected files such as:
genz_sarc_batch1.csv
genz_sarc_batch2.csv
...
genz_sarc_batch20.csv

inside:
{GENZ_DIR}
"""
    )


# Sort by batch number

def get_batch_number(file):

    match = re.search(
        r"batch(\d+)",
        file.name.lower()
    )

    if match:

        return int(match.group(1))

    return 999


genz_files = sorted(
    genz_files,
    key=get_batch_number
)


print(
    "GenZ-SarC files found:",
    len(genz_files)
)


genz_frames = []


for file_path in genz_files:

    print(
        "\nLoading:",
        file_path.name
    )


    df = pd.read_csv(
        file_path
    )


    print(
        "Rows:",
        len(df)
    )


    text_column = find_column(
        df,
        ["text"]
    )


    label_column = find_column(
        df,
        ["label", "class"]
    )


    if text_column is None:

        raise ValueError(
            f"\nTEXT column not found in:\n{file_path}"
        )


    if label_column is None:

        raise ValueError(
            f"\nLABEL column not found in:\n{file_path}"
        )


    temp = pd.DataFrame()


    temp["id"] = [
        f"GENZ_{file_path.stem}_{i:06d}"
        for i in range(1, len(df) + 1)
    ]


    temp["text"] = (
        df[text_column]
        .apply(clean_text)
    )


    temp["label"] = (
        df[label_column]
        .apply(normalize_label)
    )


    temp["language"] = "English"


    temp["source"] = "GenZ-SarC"


    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    if "style" in df.columns:

        temp["style"] = df["style"]

    else:

        temp["style"] = pd.NA


    if "topic" in df.columns:

        temp["topic"] = df["topic"]

    else:

        temp["topic"] = pd.NA


    if "slang_used" in df.columns:

        temp["slang_used"] = df["slang_used"]

    else:

        temp["slang_used"] = pd.NA


    if "abbreviations_used" in df.columns:

        temp["abbreviations_used"] = (
            df["abbreviations_used"]
        )

    else:

        temp["abbreviations_used"] = pd.NA


    if "emoji_used" in df.columns:

        temp["emoji_used"] = df["emoji_used"]

    else:

        temp["emoji_used"] = pd.NA


    temp = temp[FINAL_COLUMNS]


    genz_frames.append(temp)


# Combine all GenZ batches

genz_final = pd.concat(
    genz_frames,
    ignore_index=True
)


print("\nGenZ-SarC combined successfully.")

print(
    "Total rows:",
    len(genz_final)
)


# ============================================================
# 8. FIND MULTILINGUAL SYNTHETIC AUTOMATICALLY
# ============================================================

print_section("4. FINDING MULTILINGUAL SYNTHETIC DATASET")


print(
    "Searching inside:"
)

print(
    BASE_DIR
)


multi_candidates = []


# Search EVERY folder under NLP_PROJECT

for file in BASE_DIR.rglob("*"):

    if not file.is_file():
        continue


    # Only CSV and Excel
    if file.suffix.lower() not in [
        ".csv",
        ".xlsx",
        ".xls"
    ]:

        continue


    filename = file.name.lower()


    # Ignore files that are clearly unrelated
    if (
        "multilingual" in filename
        or "synthetic" in filename
        or "synthes" in filename
    ):

        multi_candidates.append(file)


if len(multi_candidates) == 0:

    raise FileNotFoundError(
        f"""
Could not find the Multilingual Synthetic dataset.

The code searched inside:
{BASE_DIR}

Please make sure your multilingual synthetic CSV/Excel
file is somewhere inside D:\\NLP_PROJECT.
"""
    )


print(
    "\nPossible multilingual/synthetic files found:"
)


for file in multi_candidates:

    print(
        " -",
        file
    )


# ------------------------------------------------------------
# Try to select the correct file
# ------------------------------------------------------------

selected_file = None


# Prefer filename containing multilingual + synthetic

for file in multi_candidates:

    filename = file.name.lower()

    if (
        "multilingual" in filename
        and "synthetic" in filename
    ):

        selected_file = file

        break


# If not found, use multilingual filename

if selected_file is None:

    for file in multi_candidates:

        if "multilingual" in file.name.lower():

            selected_file = file

            break


# If still not found, use first candidate

if selected_file is None:

    selected_file = multi_candidates[0]


MULTI_FILE = selected_file


print(
    "\nSelected multilingual dataset:"
)

print(
    MULTI_FILE
)


# ============================================================
# 9. LOAD MULTILINGUAL FILE
# ============================================================

print_section("5. LOADING MULTILINGUAL SYNTHETIC")


if MULTI_FILE.suffix.lower() == ".csv":

    multi = pd.read_csv(
        MULTI_FILE
    )

elif MULTI_FILE.suffix.lower() in [
    ".xlsx",
    ".xls"
]:

    multi = pd.read_excel(
        MULTI_FILE
    )

else:

    raise ValueError(
        "Unsupported multilingual dataset format."
    )


print(
    "Rows loaded:",
    len(multi)
)


print(
    "\nColumns:"
)

print(
    list(multi.columns)
)


# ------------------------------------------------------------
# Find columns
# ------------------------------------------------------------

text_column = find_column(
    multi,
    ["text"]
)


label_column = find_column(
    multi,
    ["label", "class"]
)


language_column = find_column(
    multi,
    ["language", "lang"]
)


if text_column is None:

    raise ValueError(
        "\nTEXT column not found in multilingual dataset."
    )


if label_column is None:

    raise ValueError(
        "\nLABEL column not found in multilingual dataset."
    )


if language_column is None:

    raise ValueError(
        "\nLANGUAGE column not found in multilingual dataset."
    )


# ============================================================
# 10. STANDARDIZE MULTILINGUAL DATASET
# ============================================================

multi_final = pd.DataFrame()


multi_final["id"] = [
    f"MS_{i:06d}"
    for i in range(1, len(multi) + 1)
]


multi_final["text"] = (
    multi[text_column]
    .apply(clean_text)
)


multi_final["label"] = (
    multi[label_column]
    .apply(normalize_label)
)


multi_final["language"] = (
    multi[language_column]
)


multi_final["source"] = (
    "Multilingual-Synthetic"
)


if "style" in multi.columns:

    multi_final["style"] = (
        multi["style"]
    )

else:

    multi_final["style"] = pd.NA


if "topic" in multi.columns:

    multi_final["topic"] = (
        multi["topic"]
    )

else:

    multi_final["topic"] = pd.NA


if "slang_used" in multi.columns:

    multi_final["slang_used"] = (
        multi["slang_used"]
    )

else:

    multi_final["slang_used"] = pd.NA


if "abbreviations_used" in multi.columns:

    multi_final["abbreviations_used"] = (
        multi["abbreviations_used"]
    )

else:

    multi_final["abbreviations_used"] = pd.NA


if "emoji_used" in multi.columns:

    multi_final["emoji_used"] = (
        multi["emoji_used"]
    )

else:

    multi_final["emoji_used"] = pd.NA


multi_final = multi_final[
    FINAL_COLUMNS
]


print(
    "\nMultilingual Synthetic standardized successfully."
)


# ============================================================
# 11. COMBINE ALL FOUR DATASETS
# ============================================================

print_section("6. COMBINING ALL FOUR DATASETS")


print(
    "SARC:",
    len(sarc_final)
)


print(
    "Sarcasm Corpus V2:",
    len(scv_final)
)


print(
    "GenZ-SarC:",
    len(genz_final)
)


print(
    "Multilingual Synthetic:",
    len(multi_final)
)


final_df = pd.concat(
    [
        sarc_final,
        scv_final,
        genz_final,
        multi_final
    ],
    ignore_index=True
)


print(
    "\nTotal records before cleaning:",
    len(final_df)
)


# ============================================================
# 12. REMOVE EMPTY TEXT
# ============================================================

print_section("7. REMOVING EMPTY TEXT")


final_df["text"] = (
    final_df["text"]
    .fillna("")
    .astype(str)
    .apply(clean_text)
)


before = len(final_df)


final_df = final_df[
    final_df["text"].str.len() > 0
].copy()


after = len(final_df)


print(
    "Empty text rows removed:",
    before - after
)


# ============================================================
# 13. REMOVE INVALID LABELS
# ============================================================

print_section("8. VALIDATING LABELS")


before = len(final_df)


final_df = final_df[
    final_df["label"].isin([0, 1])
].copy()


after = len(final_df)


print(
    "Invalid label rows removed:",
    before - after
)


final_df["label"] = (
    final_df["label"]
    .astype(int)
)


# ============================================================
# 14. REMOVE DUPLICATE TEXT
# ============================================================

print_section("9. REMOVING DUPLICATES")


final_df["_duplicate_key"] = (
    final_df["text"]
    .apply(duplicate_key)
)


before = len(final_df)


final_df = final_df.drop_duplicates(
    subset=["_duplicate_key"],
    keep="first"
).copy()


after = len(final_df)


print(
    "Duplicate rows removed:",
    before - after
)


final_df = final_df.drop(
    columns=["_duplicate_key"]
)


# ============================================================
# 15. RESET INDEX
# ============================================================

final_df = final_df.reset_index(
    drop=True
)


# ============================================================
# 16. CREATE FINAL UNIQUE IDS
# ============================================================

print_section("10. CREATING FINAL UNIQUE IDS")


prefix_map = {

    "SARC": "SARC",

    "SarcasmCorpusV2": "SCV",

    "GenZ-SarC": "GENZ",

    "Multilingual-Synthetic": "MS"
}


counters = {}

new_ids = []


for source in final_df["source"]:

    if source not in counters:

        counters[source] = 0


    counters[source] += 1


    prefix = prefix_map.get(
        source,
        "DATA"
    )


    new_ids.append(
        f"{prefix}_{counters[source]:06d}"
    )


final_df["id"] = new_ids


# ============================================================
# 17. FINAL COLUMN ORDER
# ============================================================

final_df = final_df[
    FINAL_COLUMNS
]


# ============================================================
# 18. SAVE FINAL MASTER DATASET
# ============================================================

print_section("11. SAVING FINAL DATASET")


output_file = (
    OUTPUT_DIR /
    "text_sarcasm_combined.csv"
)


final_df.to_csv(
    output_file,
    index=False,
    encoding="utf-8-sig"
)


print(
    "\nFINAL DATASET SAVED:"
)

print(
    output_file
)


# ============================================================
# 19. FINAL DATASET REPORT
# ============================================================

print_section("FINAL DATASET REPORT")


print(
    "TOTAL RECORDS:",
    len(final_df)
)


# ------------------------------------------------------------
# Columns
# ------------------------------------------------------------

print(
    "\nFINAL COLUMNS:"
)


for number, column in enumerate(
    final_df.columns,
    start=1
):

    print(
        f"{number}. {column}"
    )


# ------------------------------------------------------------
# Class distribution
# ------------------------------------------------------------

print(
    "\n"
    + "-" * 60
)

print(
    "CLASS DISTRIBUTION"
)


class_distribution = (
    final_df["label"]
    .value_counts()
    .sort_index()
)


print(
    class_distribution
)


print(
    "\n0 = Non-Sarcastic"
)

print(
    "1 = Sarcastic"
)


# ------------------------------------------------------------
# Class percentage
# ------------------------------------------------------------

print(
    "\nCLASS PERCENTAGE"
)


class_percentage = (
    final_df["label"]
    .value_counts(
        normalize=True
    )
    .sort_index()
    * 100
)


print(
    class_percentage.round(2)
)


# ------------------------------------------------------------
# Source distribution
# ------------------------------------------------------------

print(
    "\n"
    + "-" * 60
)

print(
    "SOURCE DISTRIBUTION"
)


print(
    final_df["source"]
    .value_counts()
)


# ------------------------------------------------------------
# Language distribution
# ------------------------------------------------------------

print(
    "\n"
    + "-" * 60
)

print(
    "LANGUAGE DISTRIBUTION"
)


print(
    final_df["language"]
    .value_counts(
        dropna=False
    )
)


# ------------------------------------------------------------
# Style distribution
# ------------------------------------------------------------

print(
    "\n"
    + "-" * 60
)

print(
    "STYLE DISTRIBUTION"
)


print(
    final_df["style"]
    .value_counts(
        dropna=False
    )
)


# ------------------------------------------------------------
# Missing values
# ------------------------------------------------------------

print(
    "\n"
    + "-" * 60
)

print(
    "MISSING VALUES"
)


print(
    final_df.isna().sum()
)


# ------------------------------------------------------------
# Duplicate check
# ------------------------------------------------------------

print(
    "\n"
    + "-" * 60
)

print(
    "FINAL DUPLICATE CHECK"
)


remaining_duplicates = (
    final_df["text"]
    .apply(duplicate_key)
    .duplicated()
    .sum()
)


print(
    "Duplicate texts remaining:",
    remaining_duplicates
)


# ------------------------------------------------------------
# Label validation
# ------------------------------------------------------------

print(
    "\n"
    + "-" * 60
)

print(
    "FINAL LABEL VALIDATION"
)


invalid_labels = final_df[
    ~final_df["label"].isin([0, 1])
]


print(
    "Invalid labels:",
    len(invalid_labels)
)


# ------------------------------------------------------------
# Required columns
# ------------------------------------------------------------

print(
    "\n"
    + "-" * 60
)

print(
    "FINAL COLUMN VALIDATION"
)


missing_columns = [
    column
    for column in FINAL_COLUMNS
    if column not in final_df.columns
]


if len(missing_columns) == 0:

    print(
        "All 10 required columns are present."
    )

else:

    print(
        "Missing columns:",
        missing_columns
    )


# ============================================================
# 20. SHOW SAMPLE RECORDS
# ============================================================

print_section("FIRST 10 RECORDS")


pd.set_option(
    "display.max_columns",
    None
)

pd.set_option(
    "display.width",
    250
)

pd.set_option(
    "display.max_colwidth",
    100
)


print(
    final_df.head(10).to_string(
        index=False
    )
)


# ============================================================
# 21. FINAL SUCCESS MESSAGE
# ============================================================

print("\n")


print("=" * 80)

print(
    "       DATASET 1 CREATION COMPLETED SUCCESSFULLY"
)

print("=" * 80)


print(
    "\nFinal file:"
)

print(
    output_file
)


print(
    "\nTotal records:"
)

print(
    len(final_df)
)


print(
    "\nModel input:"
)

print(
    "text"
)


print(
    "\nModel target:"
)

print(
    "label"
)


print(
    "\nLabel mapping:"
)

print(
    "0 = Non-Sarcastic"
)

print(
    "1 = Sarcastic"
)


print(
    "\nDatasets combined:"
)

print(
    "1. SARC"
)

print(
    "2. Sarcasm Corpus V2"

)

print(
    "3. GenZ-SarC"
)

print(
    "4. Multilingual-Synthetic"
)


print("\n" + "=" * 80)