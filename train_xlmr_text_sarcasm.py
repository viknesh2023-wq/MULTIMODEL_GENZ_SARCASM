import os
import json
import numpy as np
import torch

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
)

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = "FacebookAI/xlm-roberta-base"

DATA_DIR = r"D:\NLP_PROJECT\FINAL_DATASETS\TEXT_SARCASM\SPLIT"

# IMPORTANT:
# Completely NEW folder.
# The old pilot folder is NOT used.
OUTPUT_DIR = r"D:\NLP_PROJECT\models\xlmr_text_sarcasm_full"

TRAIN_FILE = os.path.join(DATA_DIR, "train.csv")
VAL_FILE = os.path.join(DATA_DIR, "validation.csv")
TEST_FILE = os.path.join(DATA_DIR, "test.csv")


# ============================================================
# TRAINING SETTINGS
# ============================================================

MAX_LENGTH = 128

NUM_LABELS = 2

PER_DEVICE_TRAIN_BATCH_SIZE = 8
PER_DEVICE_EVAL_BATCH_SIZE = 8

GRADIENT_ACCUMULATION_STEPS = 2

NUM_EPOCHS = 3

LEARNING_RATE = 2e-5

WEIGHT_DECAY = 0.01

WARMUP_RATIO = 0.1

SEED = 42


# ============================================================
# METRICS
# ============================================================

def compute_metrics(eval_pred):

    logits, labels = eval_pred

    if isinstance(logits, tuple):
        logits = logits[0]

    predictions = np.argmax(
        logits,
        axis=-1
    )

    accuracy = accuracy_score(
        labels,
        predictions
    )

    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            labels,
            predictions,
            average="macro",
            zero_division=0,
        )
    )

    return {
        "accuracy": float(accuracy),
        "precision_macro": float(precision),
        "recall_macro": float(recall),
        "f1_macro": float(f1),
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("XLM-R TEXT SARCASM — FULL TRAINING")
    print("=" * 70)

    # --------------------------------------------------------
    # GPU CHECK
    # --------------------------------------------------------

    print("\nPyTorch:", torch.__version__)

    print(
        "CUDA available:",
        torch.cuda.is_available()
    )

    if not torch.cuda.is_available():

        raise RuntimeError(
            "CUDA is not available."
        )

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )

    total_vram = (
        torch.cuda.get_device_properties(0)
        .total_memory
    )

    print(
        "Total VRAM:",
        round(
            total_vram / (1024 ** 3),
            2
        ),
        "GB"
    )

    # --------------------------------------------------------
    # DATASET CHECK
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("CHECKING DATASET FILES")
    print("=" * 70)

    for path in [
        TRAIN_FILE,
        VAL_FILE,
        TEST_FILE,
    ]:

        if not os.path.exists(path):

            raise FileNotFoundError(
                f"Dataset file not found:\n{path}"
            )

        print("FOUND:", path)

    # --------------------------------------------------------
    # LOAD DATASETS
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("LOADING DATASETS")
    print("=" * 70)

    dataset = load_dataset(
        "csv",
        data_files={
            "train": TRAIN_FILE,
            "validation": VAL_FILE,
            "test": TEST_FILE,
        },
    )

    print("\nDataset sizes:")

    print(
        "Train      :",
        len(dataset["train"])
    )

    print(
        "Validation :",
        len(dataset["validation"])
    )

    print(
        "Test       :",
        len(dataset["test"])
    )

    print("\nColumns:")

    print(
        dataset["train"].column_names
    )

    # --------------------------------------------------------
    # LABEL DISTRIBUTION
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("LABEL DISTRIBUTION")
    print("=" * 70)

    for split in [
        "train",
        "validation",
        "test",
    ]:

        labels = np.array(
            dataset[split]["label"]
        )

        unique, counts = np.unique(
            labels,
            return_counts=True
        )

        print(f"\n{split.upper()}")

        for label, count in zip(
            unique,
            counts
        ):

            print(
                f"Label {label}: {count}"
            )

    # --------------------------------------------------------
    # TOKENIZER
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("LOADING TOKENIZER")
    print("=" * 70)

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    print("Tokenizer loaded.")

    # --------------------------------------------------------
    # TOKENIZATION
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("TOKENIZING DATASETS")
    print("=" * 70)

    def tokenize_function(examples):

        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=MAX_LENGTH,
        )

    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        batch_size=1000,
        num_proc=1,
        desc="Tokenizing",
    )

    columns_to_remove = [
        column
        for column in tokenized_dataset["train"].column_names
        if column not in [
            "input_ids",
            "attention_mask",
            "label",
        ]
    ]

    tokenized_dataset = (
        tokenized_dataset.remove_columns(
            columns_to_remove
        )
    )

    tokenized_dataset = (
        tokenized_dataset.rename_column(
            "label",
            "labels"
        )
    )

    print("\nTokenization complete.")

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("LOADING XLM-R MODEL")
    print("=" * 70)

    model = (
        AutoModelForSequenceClassification
        .from_pretrained(
            MODEL_NAME,
            num_labels=NUM_LABELS,
        )
    )

    print("Model loaded.")

    # --------------------------------------------------------
    # DATA COLLATOR
    # --------------------------------------------------------

    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        padding=True,
    )

    # --------------------------------------------------------
    # CREATE NEW OUTPUT DIRECTORY
    # --------------------------------------------------------

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    print("\n" + "=" * 70)
    print("OUTPUT DIRECTORY")
    print("=" * 70)

    print(OUTPUT_DIR)

    # --------------------------------------------------------
    # TRAINING ARGUMENTS
    # --------------------------------------------------------

    training_args = TrainingArguments(

        output_dir=OUTPUT_DIR,

        overwrite_output_dir=False,

        num_train_epochs=NUM_EPOCHS,

        learning_rate=LEARNING_RATE,

        weight_decay=WEIGHT_DECAY,

        warmup_ratio=WARMUP_RATIO,

        per_device_train_batch_size=(
            PER_DEVICE_TRAIN_BATCH_SIZE
        ),

        per_device_eval_batch_size=(
            PER_DEVICE_EVAL_BATCH_SIZE
        ),

        gradient_accumulation_steps=(
            GRADIENT_ACCUMULATION_STEPS
        ),

        fp16=True,

        bf16=False,

        gradient_checkpointing=True,

        # Windows-safe
        dataloader_num_workers=0,

        # Evaluate once per epoch
        eval_strategy="epoch",

        # Save once per epoch
        save_strategy="epoch",

        save_total_limit=2,

        # Select best checkpoint using Macro-F1
        load_best_model_at_end=True,

        metric_for_best_model="f1_macro",

        greater_is_better=True,

        logging_strategy="steps",

        logging_steps=100,

        logging_first_step=True,

        optim="adamw_torch",

        seed=SEED,

        data_seed=SEED,

        report_to="none",

        remove_unused_columns=True,
    )

    # --------------------------------------------------------
    # TRAINER
    # --------------------------------------------------------

    trainer = Trainer(

        model=model,

        args=training_args,

        train_dataset=(
            tokenized_dataset["train"]
        ),

        eval_dataset=(
            tokenized_dataset["validation"]
        ),

        processing_class=tokenizer,

        data_collator=data_collator,

        compute_metrics=compute_metrics,
    )

    # --------------------------------------------------------
    # TRAINING INFORMATION
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("TRAINING CONFIGURATION")
    print("=" * 70)

    print(
        "Model:",
        MODEL_NAME
    )

    print(
        "Training samples:",
        len(tokenized_dataset["train"])
    )

    print(
        "Validation samples:",
        len(tokenized_dataset["validation"])
    )

    print(
        "Test samples:",
        len(tokenized_dataset["test"])
    )

    print(
        "Epochs:",
        NUM_EPOCHS
    )

    print(
        "Batch size:",
        PER_DEVICE_TRAIN_BATCH_SIZE
    )

    print(
        "Gradient accumulation:",
        GRADIENT_ACCUMULATION_STEPS
    )

    print(
        "Effective batch size:",
        PER_DEVICE_TRAIN_BATCH_SIZE
        * GRADIENT_ACCUMULATION_STEPS
    )

    print(
        "Learning rate:",
        LEARNING_RATE
    )

    print(
        "Max sequence length:",
        MAX_LENGTH
    )

    print(
        "FP16:",
        True
    )

    print(
        "Gradient checkpointing:",
        True
    )

    print(
        "Evaluation:",
        "Every epoch"
    )

    print(
        "Best model metric:",
        "Macro-F1"
    )

    print("\nTEST SET WILL NOT BE USED DURING TRAINING.")

    # --------------------------------------------------------
    # IMPORTANT:
    # NO OLD CHECKPOINT RESUME
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("CLEAN START")
    print("=" * 70)

    print(
        "\nThe old pilot checkpoint will NOT be loaded."
    )

    print(
        "\nOld pilot directory:"
    )

    print(
        r"D:\NLP_PROJECT\models\xlmr_text_sarcasm"
    )

    print(
        "\nNew full-training directory:"
    )

    print(
        OUTPUT_DIR
    )

    print(
        "\nStarting from pretrained XLM-R weights."
    )

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("STARTING FULL TRAINING")
    print("=" * 70)

    train_result = trainer.train()

    # --------------------------------------------------------
    # SAVE FINAL MODEL
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("SAVING FINAL MODEL")
    print("=" * 70)

    trainer.save_model(
        OUTPUT_DIR
    )

    tokenizer.save_pretrained(
        OUTPUT_DIR
    )

    print(
        "\nFinal model saved to:"
    )

    print(
        OUTPUT_DIR
    )

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("FINAL VALIDATION EVALUATION")
    print("=" * 70)

    validation_results = trainer.evaluate(
        eval_dataset=(
            tokenized_dataset["validation"]
        )
    )

    for key, value in validation_results.items():

        if isinstance(value, float):

            print(
                f"{key}: {value:.6f}"
            )

        else:

            print(
                f"{key}: {value}"
            )

    # --------------------------------------------------------
    # TEST
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("FINAL TEST EVALUATION")
    print("=" * 70)

    print(
        "\nThe test set is being used now."
    )

    print(
        "It was NOT used during training."
    )

    test_output = trainer.predict(
        tokenized_dataset["test"]
    )

    test_logits = test_output.predictions

    if isinstance(
        test_logits,
        tuple
    ):

        test_logits = test_logits[0]

    test_predictions = np.argmax(
        test_logits,
        axis=-1
    )

    test_labels = np.array(
        tokenized_dataset["test"]["labels"]
    )

    # --------------------------------------------------------
    # TEST METRICS
    # --------------------------------------------------------

    test_accuracy = accuracy_score(
        test_labels,
        test_predictions
    )

    (
        test_precision,
        test_recall,
        test_f1,
        _
    ) = precision_recall_fscore_support(
        test_labels,
        test_predictions,
        average="macro",
        zero_division=0,
    )

    print("\n" + "=" * 70)
    print("FINAL TEST METRICS")
    print("=" * 70)

    print(
        f"\nAccuracy        : {test_accuracy:.6f}"
    )

    print(
        f"Macro Precision : {test_precision:.6f}"
    )

    print(
        f"Macro Recall    : {test_recall:.6f}"
    )

    print(
        f"Macro F1        : {test_f1:.6f}"
    )

    # --------------------------------------------------------
    # CONFUSION MATRIX
    # --------------------------------------------------------

    cm = confusion_matrix(
        test_labels,
        test_predictions
    )

    print("\n" + "=" * 70)
    print("CONFUSION MATRIX")
    print("=" * 70)

    print(
        "\n                  Predicted"
    )

    print(
        "                  0          1"
    )

    print(
        f"Actual 0     {cm[0, 0]:10d} {cm[0, 1]:10d}"
    )

    print(
        f"Actual 1     {cm[1, 0]:10d} {cm[1, 1]:10d}"
    )

    # --------------------------------------------------------
    # SAVE RESULTS
    # --------------------------------------------------------

    final_results = {

        "model": MODEL_NAME,

        "dataset": "TEXT_SARCASM",

        "train_samples": int(
            len(
                tokenized_dataset["train"]
            )
        ),

        "validation_samples": int(
            len(
                tokenized_dataset["validation"]
            )
        ),

        "test_samples": int(
            len(
                tokenized_dataset["test"]
            )
        ),

        "max_length": MAX_LENGTH,

        "epochs": NUM_EPOCHS,

        "learning_rate": LEARNING_RATE,

        "train_batch_size":
            PER_DEVICE_TRAIN_BATCH_SIZE,

        "eval_batch_size":
            PER_DEVICE_EVAL_BATCH_SIZE,

        "gradient_accumulation_steps":
            GRADIENT_ACCUMULATION_STEPS,

        "effective_batch_size":
            (
                PER_DEVICE_TRAIN_BATCH_SIZE
                * GRADIENT_ACCUMULATION_STEPS
            ),

        "test_accuracy":
            float(test_accuracy),

        "test_macro_precision":
            float(test_precision),

        "test_macro_recall":
            float(test_recall),

        "test_macro_f1":
            float(test_f1),

        "confusion_matrix":
            cm.tolist(),

        "labels": {
            "0": "non-sarcastic",
            "1": "sarcastic",
        },
    }

    results_path = os.path.join(
        OUTPUT_DIR,
        "final_test_results.json"
    )

    with open(
        results_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            final_results,
            file,
            indent=4
        )

    print(
        "\nResults saved to:"
    )

    print(
        results_path
    )

    # --------------------------------------------------------
    # SAVE TRAINING CONFIG
    # --------------------------------------------------------

    config = {

        "model_name": MODEL_NAME,

        "dataset_directory": DATA_DIR,

        "output_directory": OUTPUT_DIR,

        "max_length": MAX_LENGTH,

        "epochs": NUM_EPOCHS,

        "learning_rate": LEARNING_RATE,

        "weight_decay": WEIGHT_DECAY,

        "warmup_ratio": WARMUP_RATIO,

        "train_batch_size":
            PER_DEVICE_TRAIN_BATCH_SIZE,

        "eval_batch_size":
            PER_DEVICE_EVAL_BATCH_SIZE,

        "gradient_accumulation_steps":
            GRADIENT_ACCUMULATION_STEPS,

        "effective_batch_size":
            (
                PER_DEVICE_TRAIN_BATCH_SIZE
                * GRADIENT_ACCUMULATION_STEPS
            ),

        "fp16": True,

        "bf16": False,

        "gradient_checkpointing": True,

        "seed": SEED,
    }

    config_path = os.path.join(
        OUTPUT_DIR,
        "training_config.json"
    )

    with open(
        config_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            config,
            file,
            indent=4
        )

    # --------------------------------------------------------
    # COMPLETION FILE
    # --------------------------------------------------------

    completion_path = os.path.join(
        OUTPUT_DIR,
        "TRAINING_COMPLETE.txt"
    )

    with open(
        completion_path,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            "XLM-R full text sarcasm training completed.\n"
        )

        file.write(
            f"Test Accuracy: {test_accuracy:.6f}\n"
        )

        file.write(
            f"Test Macro-F1: {test_f1:.6f}\n"
        )

    # --------------------------------------------------------
    # FINAL OUTPUT
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("XLM-R FULL TRAINING COMPLETE")
    print("=" * 70)

    print(
        "\nFinal model:"
    )

    print(
        OUTPUT_DIR
    )

    print(
        "\nTest Accuracy:"
    )

    print(
        f"{test_accuracy:.6f}"
    )

    print(
        "\nTest Macro-F1:"
    )

    print(
        f"{test_f1:.6f}"
    )

    print(
        "\nTest Macro Precision:"
    )

    print(
        f"{test_precision:.6f}"
    )

    print(
        "\nTest Macro Recall:"
    )

    print(
        f"{test_recall:.6f}"
    )

    print("\nDONE.")


# ============================================================
# WINDOWS ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()