import os
import csv
import re
import pandas as pd


# ============================================================
# DATASET 3 - FINAL SENTIMENT DATASET CREATION
# ============================================================
#
# DATASETS USED
#
# 1. TweetEval Sentiment
#    Language: English
#    Labels:
#       0 = Negative
#       1 = Neutral
#       2 = Positive
#
# 2. SentiMix
#    Language: Hinglish
#    Labels:
#       0 = Negative
#       1 = Neutral
#       2 = Positive
#
# 3. DravidianCodeMix
#    Languages:
#       Tamil-English
#       Malayalam-English
#       Kannada-English
#
#    Only:
#       Positive
#       Negative
#
#    Excluded:
#       Mixed_feelings
#       unknown_state
#       not-language
#
# 4. Synthetic Neutral DravidianCodeMix
#    12,000 rows
#       Tamil-English      = 4,000
#       Malayalam-English  = 4,000
#       Kannada-English    = 4,000
#
#    Label:
#       1 = Neutral
#
#    IMPORTANT:
#       Synthetic data is TRAIN ONLY.
#
# ============================================================


# ============================================================
# 1. PROJECT PATH
# ============================================================

ROOT = r"D:\NLP_PROJECT"


# ============================================================
# 2. INPUT DATASET PATHS
# ============================================================

# ------------------------------------------------------------
# TweetEval
# IMPORTANT:
# TweetEval files are directly inside:
#
# D:\NLP_PROJECT\TweetEval\
# ------------------------------------------------------------

TWEETEVAL = os.path.join(
    ROOT,
    "TweetEval"
)


# ------------------------------------------------------------
# SentiMix
# ------------------------------------------------------------

SENTIMIX = os.path.join(
    ROOT,
    "sentimix"
)


# ------------------------------------------------------------
# DravidianCodeMix
# ------------------------------------------------------------

DRAVIDIAN = os.path.join(
    ROOT,
    "DravidianCodeMix"
)


# ------------------------------------------------------------
# Synthetic Neutral Dataset
# ------------------------------------------------------------

SYNTHETIC = os.path.join(
    ROOT,
    "neutral_darav",
    "synthetic_neutral_codemixed_full.csv"
)


# ============================================================
# 3. OUTPUT PATHS
# ============================================================

OUTPUT = os.path.join(
    ROOT,
    "FINAL_DATASETS",
    "SENTIMENT"
)


SPLIT_OUTPUT = os.path.join(
    OUTPUT,
    "SPLIT"
)


ANALYSIS_OUTPUT = os.path.join(
    OUTPUT,
    "analysis"
)


# Create output folders

os.makedirs(
    OUTPUT,
    exist_ok=True
)

os.makedirs(
    SPLIT_OUTPUT,
    exist_ok=True
)

os.makedirs(
    ANALYSIS_OUTPUT,
    exist_ok=True
)


# ============================================================
# 4. OUTPUT FILES
# ============================================================

MASTER_FILE = os.path.join(
    OUTPUT,
    "sentiment_combined.csv"
)


TRAIN_FILE = os.path.join(
    SPLIT_OUTPUT,
    "train.csv"
)


VALIDATION_FILE = os.path.join(
    SPLIT_OUTPUT,
    "validation.csv"
)


TEST_FILE = os.path.join(
    SPLIT_OUTPUT,
    "test.csv"
)


UNLABELED_TEST_FILE = os.path.join(
    OUTPUT,
    "sentimix_unlabeled_test.csv"
)


REPORT_FILE = os.path.join(
    ANALYSIS_OUTPUT,
    "dataset3_validation_report.txt"
)


# ============================================================
# 5. HELPER FUNCTIONS
# ============================================================

def clean_text(text):

    if pd.isna(text):
        return ""

    text = str(text)

    # Remove BOM
    text = text.replace(
        "\ufeff",
        ""
    )

    # Replace non-breaking spaces
    text = text.replace(
        "\xa0",
        " "
    )

    # Normalize whitespace
    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


def normalize_text(text):

    text = clean_text(
        text
    )

    return text.lower()


def create_standard_dataframe(
    df,
    source,
    language,
    split,
    synthetic=False
):

    df = df.copy()

    # Clean text
    df["text"] = (
        df["text"]
        .apply(clean_text)
    )

    # Remove empty text
    df = df[
        df["text"] != ""
    ].copy()

    # Add metadata
    df["source"] = source

    df["language"] = language

    df["original_split"] = split

    df["synthetic"] = synthetic

    return df[
        [
            "id",
            "text",
            "label",
            "sentiment",
            "language",
            "source",
            "original_split",
            "synthetic"
        ]
    ]


# ============================================================
# 6. TWEETEVAL LABEL MAPPING
# ============================================================

def tweet_eval_label(value):

    value = clean_text(
        value
    )

    mapping = {

        "0": (
            0,
            "Negative"
        ),

        "1": (
            1,
            "Neutral"
        ),

        "2": (
            2,
            "Positive"
        )

    }

    return mapping.get(
        value,
        (None, None)
    )


# ============================================================
# 7. SENTIMIX LABEL MAPPING
# ============================================================

def sentimix_label(value):

    value = clean_text(
        value
    )

    mapping = {

        "0": (
            0,
            "Negative"
        ),

        "1": (
            1,
            "Neutral"
        ),

        "2": (
            2,
            "Positive"
        )

    }

    return mapping.get(
        value,
        (None, None)
    )


# ============================================================
# 8. DRAVIDIAN LABEL MAPPING
# ============================================================

def dravidian_label(value):

    value = clean_text(
        value
    )

    # Positive
    if value == "Positive":

        return (
            2,
            "Positive"
        )

    # Negative
    if value == "Negative":

        return (
            0,
            "Negative"
        )

    # Everything else is excluded.
    #
    # This includes:
    #
    # Mixed_feelings
    # unknown_state
    # not-Tamil
    # not-Malayalam
    # not-Kannada
    #
    return (
        None,
        None
    )


# ============================================================
# 9. DRAVIDIAN FILE READER
# ============================================================

def read_dravidian_file(
    filepath,
    delimiter
):

    rows = []

    with open(
        filepath,
        "r",
        encoding="utf-8-sig",
        errors="replace"
    ) as file:

        reader = csv.reader(
            file,
            delimiter=delimiter
        )

        for row in reader:

            if len(row) < 2:
                continue

            text = row[0]

            label = row[1]

            rows.append(
                [
                    text,
                    label
                ]
            )

    return pd.DataFrame(
        rows,
        columns=[
            "text",
            "label"
        ]
    )


# ============================================================
# 10. START
# ============================================================

print()
print("=" * 70)
print(
    "DATASET 3 - SENTIMENT DATASET CREATION"
)
print("=" * 70)
print()

print(
    "Project root:",
    ROOT
)

print()


# Storage for all datasets

all_data = []


# Storage for report

report = []


# ============================================================
# 11. CHECK INPUT FOLDERS
# ============================================================

print(
    "Checking input folders..."
)

print()

required_directories = {

    "TweetEval":
        TWEETEVAL,

    "SentiMix":
        SENTIMIX,

    "DravidianCodeMix":
        DRAVIDIAN

}


for name, path in (
    required_directories.items()
):

    if not os.path.isdir(path):

        raise FileNotFoundError(
            f"{name} folder not found:\n{path}"
        )

    print(
        f"  OK: {name}"
    )


if not os.path.isfile(
    SYNTHETIC
):

    raise FileNotFoundError(
        "Synthetic Neutral dataset not found:\n"
        + SYNTHETIC
    )


print(
    "  OK: Synthetic Neutral dataset"
)

print()


# ============================================================
# 12. TWEETEVAL
# ============================================================

print(
    "[1/4] Processing TweetEval..."
)

print()


tweet_splits = {

    "train": (
        "train_text.txt",
        "train_labels.txt"
    ),

    "validation": (
        "val_text.txt",
        "val_labels.txt"
    ),

    "test": (
        "test_text.txt",
        "test_labels.txt"
    )

}


tweet_counts = {}


for split, files in (
    tweet_splits.items()
):

    text_file = os.path.join(
        TWEETEVAL,
        files[0]
    )

    label_file = os.path.join(
        TWEETEVAL,
        files[1]
    )


    # Check files

    if not os.path.isfile(
        text_file
    ):

        raise FileNotFoundError(
            f"TweetEval text file not found:\n"
            f"{text_file}"
        )


    if not os.path.isfile(
        label_file
    ):

        raise FileNotFoundError(
            f"TweetEval label file not found:\n"
            f"{label_file}"
        )


    # Read text

    with open(
        text_file,
        "r",
        encoding="utf-8",
        errors="replace"
    ) as file:

        texts = [
            line.rstrip("\r\n")
            for line in file
        ]


    # Read labels

    with open(
        label_file,
        "r",
        encoding="utf-8",
        errors="replace"
    ) as file:

        labels = [
            line.rstrip("\r\n")
            for line in file
        ]


    # Check count

    if len(texts) != len(labels):

        raise ValueError(

            f"TweetEval {split} count mismatch.\n"
            f"Text rows   : {len(texts)}\n"
            f"Label rows  : {len(labels)}"

        )


    rows = []

    invalid_labels = 0


    for index, (
        text,
        raw_label
    ) in enumerate(
        zip(texts, labels),
        start=1
    ):

        label, sentiment = (
            tweet_eval_label(
                raw_label
            )
        )


        if label is None:

            invalid_labels += 1

            continue


        rows.append({

            "id":
                f"TE_{split}_{index:06d}",

            "text":
                text,

            "label":
                label,

            "sentiment":
                sentiment

        })


    df = pd.DataFrame(
        rows
    )


    df = create_standard_dataframe(

        df,

        source="TweetEval",

        language="English",

        split=split,

        synthetic=False

    )


    all_data.append(
        df
    )


    tweet_counts[
        split
    ] = len(df)


    report.append(

        f"TweetEval {split}: "
        f"{len(df):,} rows; "
        f"{invalid_labels:,} invalid labels removed"

    )


    print(

        f"  {split:<12}: "
        f"{len(df):,} rows"

    )


print()


# ============================================================
# 13. SENTIMIX
# ============================================================

print(
    "[2/4] Processing SentiMix..."
)

print()


senti_train = os.path.join(
    SENTIMIX,
    "FinalTrainingOnly (1).tsv"
)


senti_validation = os.path.join(
    SENTIMIX,
    "ValidationOnly (1).tsv"
)


senti_test = os.path.join(
    SENTIMIX,
    "FinalTest (1).tsv"
)


# Check files

for filepath in [

    senti_train,

    senti_validation,

    senti_test

]:

    if not os.path.isfile(
        filepath
    ):

        raise FileNotFoundError(
            f"SentiMix file not found:\n"
            f"{filepath}"
        )


# ------------------------------------------------------------
# SentiMix labeled files
# ------------------------------------------------------------

for split, filepath in [

    (
        "train",
        senti_train
    ),

    (
        "validation",
        senti_validation
    )

]:

    df_raw = pd.read_csv(

        filepath,

        sep="\t",

        header=None,

        names=[
            "id",
            "text",
            "label"
        ],

        dtype=str,

        keep_default_na=False

    )


    rows = []

    invalid_labels = 0


    for index, row in (
        df_raw.iterrows()
    ):

        label, sentiment = (
            sentimix_label(
                row["label"]
            )
        )


        if label is None:

            invalid_labels += 1

            continue


        rows.append({

            "id":
                f"SM_{split}_{index + 1:06d}",

            "text":
                row["text"],

            "label":
                label,

            "sentiment":
                sentiment

        })


    df = pd.DataFrame(
        rows
    )


    df = create_standard_dataframe(

        df,

        source="SentiMix",

        language="Hinglish",

        split=split,

        synthetic=False

    )


    all_data.append(
        df
    )


    report.append(

        f"SentiMix {split}: "
        f"{len(df):,} rows; "
        f"{invalid_labels:,} invalid labels removed"

    )


    print(

        f"  {split:<12}: "
        f"{len(df):,} rows"

    )


# ------------------------------------------------------------
# SentiMix unlabeled test
# ------------------------------------------------------------

df_unlabeled = pd.read_csv(

    senti_test,

    sep="\t",

    header=None,

    names=[
        "id",
        "text"
    ],

    dtype=str,

    keep_default_na=False

)


df_unlabeled["text"] = (
    df_unlabeled["text"]
    .apply(clean_text)
)


df_unlabeled = df_unlabeled[
    df_unlabeled["text"] != ""
].copy()


df_unlabeled["id"] = [

    f"SM_UNLABELED_TEST_{i + 1:06d}"

    for i in range(
        len(df_unlabeled)
    )

]


df_unlabeled[
    "language"
] = "Hinglish"


df_unlabeled[
    "source"
] = "SentiMix"


df_unlabeled[
    "original_split"
] = "test_unlabeled"


df_unlabeled = df_unlabeled[

    [
        "id",
        "text",
        "language",
        "source",
        "original_split"
    ]

]


df_unlabeled.to_csv(

    UNLABELED_TEST_FILE,

    index=False,

    encoding="utf-8-sig"

)


print(

    f"  Unlabeled test: "
    f"{len(df_unlabeled):,} rows"

)

print()


# ============================================================
# 14. DRAVIDIAN CODEMIX
# ============================================================

print(
    "[3/4] Processing DravidianCodeMix..."
)

print()


dravidian_datasets = [

    (
        "Kannada-English",

        "kannada_sentiment_train.csv",

        "kannada_sentiment_dev.csv",

        "kannada_sentiment_test.csv"

    ),

    (
        "Malayalam-English",

        "mal_full_sentiment_train.csv",

        "mal_full_sentiment_dev.csv",

        "mal_full_sentiment_test.csv"

    ),

    (
        "Tamil-English",

        "tamil_sentiment_full_train.csv",

        "tamil_sentiment_full_dev.csv",

        "tamil_sentiment_full_test.csv"

    )

]


for (
    language,
    train_file,
    validation_file,
    test_file
) in dravidian_datasets:


    split_files = [

        (
            "train",
            train_file
        ),

        (
            "validation",
            validation_file
        ),

        (
            "test",
            test_file
        )

    ]


    for split, filename in split_files:


        filepath = os.path.join(

            DRAVIDIAN,

            filename

        )


        if not os.path.isfile(
            filepath
        ):

            raise FileNotFoundError(

                "DravidianCodeMix file not found:\n"
                + filepath

            )


        # ----------------------------------------------------
        # Known delimiter structure
        #
        # Tamil train = TAB
        # Everything else = semicolon
        # ----------------------------------------------------

        if (
            language == "Tamil-English"
            and split == "train"
        ):

            delimiter = "\t"

        else:

            delimiter = ";"


        df_raw = read_dravidian_file(

            filepath,

            delimiter

        )


        rows = []

        excluded = 0


        for index, row in (
            df_raw.iterrows()
        ):


            label, sentiment = (
                dravidian_label(
                    row["label"]
                )
            )


            if label is None:

                excluded += 1

                continue


            # Unique IDs
            #
            # TA = Tamil
            # MA = Malayalam
            # KA = Kannada

            language_code = {

                "Tamil-English": "TA",

                "Malayalam-English": "ML",

                "Kannada-English": "KN"

            }[language]


            rows.append({

                "id":
                    (
                        f"DCM_"
                        f"{language_code}_"
                        f"{split}_"
                        f"{index + 1:06d}"
                    ),

                "text":
                    row["text"],

                "label":
                    label,

                "sentiment":
                    sentiment

            })


        df = pd.DataFrame(
            rows
        )


        df = create_standard_dataframe(

            df,

            source="DravidianCodeMix",

            language=language,

            split=split,

            synthetic=False

        )


        all_data.append(
            df
        )


        report.append(

            f"DravidianCodeMix "
            f"{language} "
            f"{split}: "
            f"{len(df):,} kept; "
            f"{excluded:,} excluded"

        )


        print(

            f"  {language:<20} "
            f"{split:<12}: "
            f"{len(df):,} rows "
            f"(excluded {excluded:,})"

        )


print()


# ============================================================
# 15. SYNTHETIC NEUTRAL DATA
# ============================================================

print(
    "[4/4] Processing synthetic Neutral dataset..."
)

print()


synthetic = pd.read_csv(

    SYNTHETIC,

    dtype=str,

    keep_default_na=False

)


# Required columns

required_columns = {

    "id",
    "text",
    "label",
    "sentiment",
    "language",
    "source"

}


missing_columns = (

    required_columns
    -
    set(synthetic.columns)

)


if missing_columns:

    raise ValueError(

        "Synthetic dataset is missing columns:\n"
        + str(
            sorted(
                missing_columns
            )
        )

    )


# Clean

synthetic["text"] = (
    synthetic["text"]
    .apply(clean_text)
)


synthetic["label"] = (
    synthetic["label"]
    .astype(str)
    .str.strip()
)


synthetic["sentiment"] = (
    synthetic["sentiment"]
    .astype(str)
    .str.strip()
)


synthetic["language"] = (
    synthetic["language"]
    .astype(str)
    .str.strip()
)


synthetic["source"] = (
    synthetic["source"]
    .astype(str)
    .str.strip()
)


# ------------------------------------------------------------
# Strict validation
# ------------------------------------------------------------

valid_languages = {

    "Tamil-English",

    "Malayalam-English",

    "Kannada-English"

}


invalid_synthetic = synthetic[

    (synthetic["text"] == "")

    |

    (synthetic["label"] != "1")

    |

    (
        synthetic["sentiment"].str.lower()
        != "neutral"
    )

    |

    (
        ~synthetic["language"].isin(
            valid_languages
        )
    )

]


if len(invalid_synthetic) > 0:

    raise ValueError(

        "Synthetic Neutral validation FAILED.\n"
        f"Invalid rows: "
        f"{len(invalid_synthetic):,}"

    )


# Force numeric label

synthetic["label"] = 1


# Synthetic data must be TRAIN

synthetic[
    "original_split"
] = "train"


synthetic[
    "synthetic"
] = True


synthetic = synthetic[

    [
        "id",
        "text",
        "label",
        "sentiment",
        "language",
        "source",
        "original_split",
        "synthetic"
    ]

]


all_data.append(
    synthetic
)


report.append(

    f"Synthetic Neutral: "
    f"{len(synthetic):,} rows; "
    f"TRAIN ONLY"

)


print(

    f"  Synthetic Neutral: "
    f"{len(synthetic):,} rows"

)

print()


# ============================================================
# 16. COMBINE ALL DATASETS
# ============================================================

print(
    "Combining all datasets..."
)

print()


master = pd.concat(

    all_data,

    ignore_index=True

)


print(

    f"Combined rows before cleaning: "
    f"{len(master):,}"

)


# ============================================================
# 17. CLEAN MASTER
# ============================================================

master["text"] = (
    master["text"]
    .apply(clean_text)
)


master["sentiment"] = (
    master["sentiment"]
    .apply(clean_text)
)


master["language"] = (
    master["language"]
    .apply(clean_text)
)


master["source"] = (
    master["source"]
    .apply(clean_text)
)


master["label"] = pd.to_numeric(

    master["label"],

    errors="coerce"

)


master["id"] = (
    master["id"]
    .astype(str)
)


# ============================================================
# 18. REMOVE INVALID ROWS
# ============================================================

before_cleaning = len(
    master
)


master = master[

    master["text"].ne("")

    & master["label"].isin(
        [0, 1, 2]
    )

    & master["sentiment"].isin(
        [
            "Negative",
            "Neutral",
            "Positive"
        ]
    )

].copy()


removed_invalid = (

    before_cleaning
    -
    len(master)

)


# ============================================================
# 19. DUPLICATE ID CHECK
# ============================================================

duplicate_ids = int(

    master["id"]
    .duplicated()
    .sum()

)


if duplicate_ids > 0:

    print(
        "WARNING:"
    )

    print(
        f"Duplicate IDs found: "
        f"{duplicate_ids:,}"
    )

    # Make IDs unique rather than silently deleting data.
    #
    # This should normally be zero because IDs were generated
    # with source/language/split information.

    duplicated_mask = (
        master["id"]
        .duplicated(
            keep=False
        )
    )

    duplicate_rows = (
        master[duplicated_mask]
        .copy()
    )

    for idx in duplicate_rows.index:

        original_id = master.loc[
            idx,
            "id"
        ]

        master.loc[
            idx,
            "id"
        ] = (
            f"{original_id}_DUP_{idx}"
        )


# ============================================================
# 20. NORMALIZED TEXT
# ============================================================

master[
    "_normalized_text"
] = (

    master["text"]
    .apply(normalize_text)

)


# ============================================================
# 21. CREATE TRAIN / VALIDATION / TEST
# ============================================================

train_df = master[

    master["original_split"]
    == "train"

].copy()


validation_df = master[

    master["original_split"]
    == "validation"

].copy()


test_df = master[

    master["original_split"]
    == "test"

].copy()


print(
    "Initial split sizes:"
)

print(
    f"  Train      : "
    f"{len(train_df):,}"
)

print(
    f"  Validation : "
    f"{len(validation_df):,}"
)

print(
    f"  Test       : "
    f"{len(test_df):,}"
)

print()


# ============================================================
# 22. REMOVE DUPLICATE TEXT INSIDE TRAIN
# ============================================================

before_train = len(
    train_df
)


train_df = (

    train_df
    .drop_duplicates(
        subset="_normalized_text",
        keep="first"
    )

)


train_internal_duplicates = (

    before_train
    -
    len(train_df)

)


# ============================================================
# 23. REMOVE VALIDATION TEXT LEAKING FROM TRAIN
# ============================================================

train_texts = set(

    train_df[
        "_normalized_text"
    ]

)


before_validation = len(
    validation_df
)


validation_df = validation_df[

    ~validation_df[
        "_normalized_text"
    ].isin(
        train_texts
    )

].copy()


validation_train_overlap_removed = (

    before_validation
    -
    len(validation_df)

)


# Remove validation internal duplicates

before_validation_internal = len(
    validation_df
)


validation_df = (

    validation_df
    .drop_duplicates(
        subset="_normalized_text",
        keep="first"
    )

)


validation_internal_duplicates = (

    before_validation_internal
    -
    len(validation_df)

)


# ============================================================
# 24. REMOVE TEST TEXT LEAKING FROM TRAIN/VALIDATION
# ============================================================

used_texts = (

    train_texts
    |
    set(
        validation_df[
            "_normalized_text"
        ]
    )

)


before_test = len(
    test_df
)


test_df = test_df[

    ~test_df[
        "_normalized_text"
    ].isin(
        used_texts
    )

].copy()


test_overlap_removed = (

    before_test
    -
    len(test_df)

)


# Remove test internal duplicates

before_test_internal = len(
    test_df
)


test_df = (

    test_df
    .drop_duplicates(
        subset="_normalized_text",
        keep="first"
    )

)


test_internal_duplicates = (

    before_test_internal
    -
    len(test_df)

)


# ============================================================
# 25. FINAL LEAKAGE CHECK
# ============================================================

train_texts_final = set(

    train_df[
        "_normalized_text"
    ]

)


validation_texts_final = set(

    validation_df[
        "_normalized_text"
    ]

)


test_texts_final = set(

    test_df[
        "_normalized_text"
    ]

)


train_validation_overlap = len(

    train_texts_final
    &
    validation_texts_final

)


train_test_overlap = len(

    train_texts_final
    &
    test_texts_final

)


validation_test_overlap = len(

    validation_texts_final
    &
    test_texts_final

)


# ============================================================
# 26. REMOVE HELPER COLUMN
# ============================================================

for df in [

    train_df,

    validation_df,

    test_df

]:

    df.drop(

        columns=[
            "_normalized_text"
        ],

        inplace=True

    )


# ============================================================
# 27. FINAL MASTER
# ============================================================

final_master = pd.concat(

    [
        train_df,
        validation_df,
        test_df
    ],

    ignore_index=True

)


final_columns = [

    "id",

    "text",

    "label",

    "sentiment",

    "language",

    "source",

    "original_split",

    "synthetic"

]


final_master = final_master[
    final_columns
]


# ============================================================
# 28. FINAL STATISTICS
# ============================================================

print()
print("=" * 70)
print(
    "FINAL DATASET STATISTICS"
)
print("=" * 70)

print()

print(
    f"Total rows : "
    f"{len(final_master):,}"
)

print(
    f"Train      : "
    f"{len(train_df):,}"
)

print(
    f"Validation : "
    f"{len(validation_df):,}"
)

print(
    f"Test       : "
    f"{len(test_df):,}"
)

print()


# ============================================================
# SENTIMENT
# ============================================================

print(
    "Sentiment distribution:"
)

print(

    final_master[
        "sentiment"
    ].value_counts()

)

print()


# ============================================================
# LANGUAGE
# ============================================================

print(
    "Language distribution:"
)

print(

    final_master[
        "language"
    ].value_counts()

)

print()


# ============================================================
# SOURCE
# ============================================================

print(
    "Source distribution:"
)

print(

    final_master[
        "source"
    ].value_counts()

)

print()


# ============================================================
# 29. SAVE DATASETS
# ============================================================

print(
    "Saving files..."
)

print()


final_master.to_csv(

    MASTER_FILE,

    index=False,

    encoding="utf-8-sig"

)


train_df.to_csv(

    TRAIN_FILE,

    index=False,

    encoding="utf-8-sig"

)


validation_df.to_csv(

    VALIDATION_FILE,

    index=False,

    encoding="utf-8-sig"

)


test_df.to_csv(

    TEST_FILE,

    index=False,

    encoding="utf-8-sig"

)


# ============================================================
# 30. FINAL QUALITY CHECKS
# ============================================================

empty_text = int(

    final_master[
        "text"
    ]
    .apply(clean_text)
    .eq("")
    .sum()

)


invalid_labels = int(

    (
        ~final_master[
            "label"
        ].isin(
            [0, 1, 2]
        )
    )
    .sum()

)


duplicate_final_ids = int(

    final_master[
        "id"
    ]
    .duplicated()
    .sum()

)


duplicate_final_text = int(

    final_master[
        "text"
    ]
    .apply(normalize_text)
    .duplicated()
    .sum()

)


# Synthetic must remain train-only

synthetic_outside_train = int(

    (

        final_master[
            "synthetic"
        ].astype(bool)

        &

        (
            final_master[
                "original_split"
            ]
            != "train"
        )

    )
    .sum()

)


# Check labels in each split

train_labels = set(
    train_df["label"]
)

validation_labels = set(
    validation_df["label"]
)

test_labels = set(
    test_df["label"]
)


# ============================================================
# 31. REPORT
# ============================================================

report_lines = []


report_lines.append(
    "=" * 70
)

report_lines.append(
    "DATASET 3 - SENTIMENT VALIDATION REPORT"
)

report_lines.append(
    "=" * 70
)

report_lines.append("")


# ------------------------------------------------------------
# Input processing
# ------------------------------------------------------------

report_lines.append(
    "INPUT PROCESSING"
)

report_lines.append(
    "-" * 70
)


for line in report:

    report_lines.append(
        line
    )


report_lines.append("")


# ------------------------------------------------------------
# Final size
# ------------------------------------------------------------

report_lines.append(
    "FINAL DATASET SIZE"
)

report_lines.append(
    "-" * 70
)

report_lines.append(
    f"Total: {len(final_master):,}"
)

report_lines.append(
    f"Train: {len(train_df):,}"
)

report_lines.append(
    f"Validation: {len(validation_df):,}"
)

report_lines.append(
    f"Test: {len(test_df):,}"
)

report_lines.append("")


# ------------------------------------------------------------
# Sentiment
# ------------------------------------------------------------

report_lines.append(
    "SENTIMENT DISTRIBUTION"
)

report_lines.append(
    "-" * 70
)


for sentiment in [

    "Negative",

    "Neutral",

    "Positive"

]:

    count = int(

        (
            final_master[
                "sentiment"
            ]
            == sentiment
        )
        .sum()

    )

    report_lines.append(

        f"{sentiment}: "
        f"{count:,}"

    )


report_lines.append("")


# ------------------------------------------------------------
# Language
# ------------------------------------------------------------

report_lines.append(
    "LANGUAGE DISTRIBUTION"
)

report_lines.append(
    "-" * 70
)


for language, count in (

    final_master[
        "language"
    ]
    .value_counts()
    .sort_index()
    .items()

):

    report_lines.append(

        f"{language}: "
        f"{count:,}"

    )


report_lines.append("")


# ------------------------------------------------------------
# Source
# ------------------------------------------------------------

report_lines.append(
    "SOURCE DISTRIBUTION"
)

report_lines.append(
    "-" * 70
)


for source, count in (

    final_master[
        "source"
    ]
    .value_counts()
    .sort_index()
    .items()

):

    report_lines.append(

        f"{source}: "
        f"{count:,}"

    )


report_lines.append("")


# ------------------------------------------------------------
# Split labels
# ------------------------------------------------------------

report_lines.append(
    "SPLIT LABEL DISTRIBUTION"
)

report_lines.append(
    "-" * 70
)


for name, df in [

    (
        "TRAIN",
        train_df
    ),

    (
        "VALIDATION",
        validation_df
    ),

    (
        "TEST",
        test_df
    )

]:

    report_lines.append("")

    report_lines.append(
        name
    )


    for label, sentiment in [

        (
            0,
            "Negative"
        ),

        (
            1,
            "Neutral"
        ),

        (
            2,
            "Positive"
        )

    ]:

        count = int(

            (
                df[
                    "label"
                ]
                == label
            )
            .sum()

        )

        report_lines.append(

            f"  {label} "
            f"({sentiment}): "
            f"{count:,}"

        )


report_lines.append("")


# ------------------------------------------------------------
# Data quality
# ------------------------------------------------------------

report_lines.append(
    "DATA QUALITY"
)

report_lines.append(
    "-" * 70
)

report_lines.append(
    f"Empty text: "
    f"{empty_text}"
)

report_lines.append(
    f"Invalid labels: "
    f"{invalid_labels}"
)

report_lines.append(
    f"Duplicate IDs: "
    f"{duplicate_final_ids}"
)

report_lines.append(
    f"Duplicate normalized texts: "
    f"{duplicate_final_text}"
)

report_lines.append("")


# ------------------------------------------------------------
# Leakage
# ------------------------------------------------------------

report_lines.append(
    "LEAKAGE CHECK"
)

report_lines.append(
    "-" * 70
)

report_lines.append(
    f"Train <-> Validation: "
    f"{train_validation_overlap}"
)

report_lines.append(
    f"Train <-> Test: "
    f"{train_test_overlap}"
)

report_lines.append(
    f"Validation <-> Test: "
    f"{validation_test_overlap}"
)

report_lines.append("")


# ------------------------------------------------------------
# Removed rows
# ------------------------------------------------------------

report_lines.append(
    "ROWS REMOVED"
)

report_lines.append(
    "-" * 70
)

report_lines.append(
    f"Invalid/empty rows: "
    f"{removed_invalid:,}"
)

report_lines.append(
    f"Train internal duplicates: "
    f"{train_internal_duplicates:,}"
)

report_lines.append(
    f"Validation rows overlapping train: "
    f"{validation_train_overlap_removed:,}"
)

report_lines.append(
    f"Validation internal duplicates: "
    f"{validation_internal_duplicates:,}"
)

report_lines.append(
    f"Test rows overlapping train/validation: "
    f"{test_overlap_removed:,}"
)

report_lines.append(
    f"Test internal duplicates: "
    f"{test_internal_duplicates:,}"
)

report_lines.append("")


# ------------------------------------------------------------
# Synthetic data
# ------------------------------------------------------------

synthetic_count = int(

    final_master[
        "synthetic"
    ]
    .astype(bool)
    .sum()

)


report_lines.append(
    "SYNTHETIC DATA CHECK"
)

report_lines.append(
    "-" * 70
)

report_lines.append(
    f"Synthetic rows: "
    f"{synthetic_count:,}"
)

report_lines.append(
    f"Synthetic rows outside train: "
    f"{synthetic_outside_train}"
)

report_lines.append("")


# ============================================================
# 32. FINAL PASS / FAIL
# ============================================================

checks = {

    "No empty text":
        empty_text == 0,

    "All labels valid":
        invalid_labels == 0,

    "IDs unique":
        duplicate_final_ids == 0,

    "Final text unique":
        duplicate_final_text == 0,

    "Train-validation leakage absent":
        train_validation_overlap == 0,

    "Train-test leakage absent":
        train_test_overlap == 0,

    "Validation-test leakage absent":
        validation_test_overlap == 0,

    "Synthetic data train-only":
        synthetic_outside_train == 0,

    "Train contains Negative, Neutral, Positive":
        train_labels == {
            0,
            1,
            2
        },

    "Validation contains all 3 labels":
        validation_labels == {
            0,
            1,
            2
        },

    "Test contains all 3 labels":
        test_labels == {
            0,
            1,
            2
        }

}


report_lines.append(
    "FINAL CHECKS"
)

report_lines.append(
    "-" * 70
)


all_pass = True


for check, passed in (
    checks.items()
):

    if passed:

        report_lines.append(
            f"PASS: {check}"
        )

    else:

        report_lines.append(
            f"FAIL: {check}"
        )

        all_pass = False


report_lines.append("")


# ============================================================
# FINAL STATUS
# ============================================================

if all_pass:

    report_lines.append(
        "OVERALL STATUS: PASSED"
    )

else:

    report_lines.append(
        "OVERALL STATUS: REVIEW REQUIRED"
    )


report_lines.append("")


# ============================================================
# OUTPUT FILES
# ============================================================

report_lines.append(
    "OUTPUT FILES"
)

report_lines.append(
    "-" * 70
)

report_lines.append(
    MASTER_FILE
)

report_lines.append(
    TRAIN_FILE
)

report_lines.append(
    VALIDATION_FILE
)

report_lines.append(
    TEST_FILE
)

report_lines.append(
    UNLABELED_TEST_FILE
)

report_lines.append(
    REPORT_FILE
)


# ============================================================
# SAVE REPORT
# ============================================================

with open(

    REPORT_FILE,

    "w",

    encoding="utf-8"

) as file:

    file.write(
        "\n".join(
            report_lines
        )
    )


# ============================================================
# 33. FINAL CONSOLE OUTPUT
# ============================================================

print()
print("=" * 70)
print(
    "DATASET 3 COMPLETE"
)
print("=" * 70)

print()

print(
    f"Master     : "
    f"{MASTER_FILE}"
)

print(
    f"Train      : "
    f"{TRAIN_FILE}"
)

print(
    f"Validation : "
    f"{VALIDATION_FILE}"
)

print(
    f"Test       : "
    f"{TEST_FILE}"
)

print(
    f"Report     : "
    f"{REPORT_FILE}"
)

print()

print(
    f"Total rows : "
    f"{len(final_master):,}"
)

print(
    f"Train      : "
    f"{len(train_df):,}"
)

print(
    f"Validation : "
    f"{len(validation_df):,}"
)

print(
    f"Test       : "
    f"{len(test_df):,}"
)

print()


if all_pass:

    print(
        "OVERALL STATUS: PASSED"
    )

else:

    print(
        "OVERALL STATUS: REVIEW REQUIRED"
    )


print(
    "=" * 70
)