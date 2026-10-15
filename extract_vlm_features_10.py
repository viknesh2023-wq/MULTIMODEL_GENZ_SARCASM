# ================================================================
# INTERNVL3-2B — MULTIMODAL VLM FEATURE DATASET CREATOR
#
# Purpose:
#   Pass every dataset image through the FROZEN InternVL3-2B
#   vision encoder and save visual features for the downstream
#   multimodal sarcasm fusion model.
#
# IMPORTANT:
#   InternVL is NOT trained.
#   InternVL parameters remain frozen.
#
# Hardware:
#   RTX 4050 Laptop GPU — 6 GB
#
# Environment:
#   Python 3.12.10
#   PyTorch 2.11.0+cu128
#   Transformers 4.52.4
# ================================================================

import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T

from PIL import Image
from transformers import AutoModel


# ================================================================
# CONFIGURATION
# ================================================================

MODEL_NAME = "OpenGVLab/InternVL3-2B"

PROJECT_ROOT = r"D:\NLP_PROJECT"

DATASET_CSV = (
    r"D:\NLP_PROJECT\FINAL_DATASETS"
    r"\MULTIMODAL_SARCASM"
    r"\multimodal_sarcasm_combined.csv"
)

OUTPUT_DIR = (
    r"D:\NLP_PROJECT\FINAL_DATASETS"
    r"\MULTIMODAL_SARCASM"
    r"\VLM_FEATURES"
)

FEATURE_DIR = os.path.join(
    OUTPUT_DIR,
    "features"
)

CHECKPOINT_FILE = os.path.join(
    OUTPUT_DIR,
    "checkpoint.csv"
)

FAILED_FILE = os.path.join(
    OUTPUT_DIR,
    "failed.csv"
)

FINAL_METADATA_FILE = os.path.join(
    OUTPUT_DIR,
    "vlm_feature_dataset.csv"
)


# ----------------------------------------------------------------
# IMPORTANT
#
# FIRST TEST:
#
# MAX_SAMPLES = 10
#
# AFTER THE 10 IMAGE TEST SUCCEEDS:
#
# MAX_SAMPLES = None
#
# None = process entire dataset.
# ----------------------------------------------------------------

MAX_SAMPLES = 10


# ----------------------------------------------------------------
# InternVL settings
# ----------------------------------------------------------------

IMAGE_SIZE = 448

MIN_NUM = 1
MAX_NUM = 6

DEVICE = "cuda"

DTYPE = torch.bfloat16


# ================================================================
# IMAGE NORMALIZATION
# ================================================================

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


transform = T.Compose([
    T.Lambda(
        lambda img: img.convert("RGB")
    ),

    T.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    T.ToTensor(),

    T.Normalize(
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD
    )
])


# ================================================================
# DYNAMIC IMAGE TILING
# ================================================================

def find_closest_aspect_ratio(
    aspect_ratio,
    target_ratios,
    width,
    height,
    image_size
):

    best_ratio_diff = float("inf")

    best_ratio = (1, 1)

    area = width * height

    for ratio in target_ratios:

        target_aspect_ratio = (
            ratio[0] / ratio[1]
        )

        ratio_diff = abs(
            aspect_ratio -
            target_aspect_ratio
        )

        if ratio_diff < best_ratio_diff:

            best_ratio_diff = ratio_diff

            best_ratio = ratio

        elif ratio_diff == best_ratio_diff:

            if (
                area >
                0.5
                * image_size
                * image_size
                * ratio[0]
                * ratio[1]
            ):

                best_ratio = ratio

    return best_ratio


def dynamic_preprocess(
    image,
    min_num=MIN_NUM,
    max_num=MAX_NUM,
    image_size=IMAGE_SIZE
):

    orig_width, orig_height = image.size

    aspect_ratio = (
        orig_width / orig_height
    )

    target_ratios = []

    for n in range(
        min_num,
        max_num + 1
    ):

        for i in range(1, n + 1):

            for j in range(1, n + 1):

                if (
                    i * j <= max_num
                    and i * j >= min_num
                ):

                    target_ratios.append(
                        (i, j)
                    )

    target_ratios = list(
        set(target_ratios)
    )

    target_ratios.sort(
        key=lambda x: x[0] * x[1]
    )

    target_aspect_ratio = (
        find_closest_aspect_ratio(
            aspect_ratio,
            target_ratios,
            orig_width,
            orig_height,
            image_size
        )
    )

    target_width = (
        image_size *
        target_aspect_ratio[0]
    )

    target_height = (
        image_size *
        target_aspect_ratio[1]
    )

    blocks = (
        target_aspect_ratio[0] *
        target_aspect_ratio[1]
    )

    resized_img = image.resize(
        (
            target_width,
            target_height
        )
    )

    processed_images = []

    for i in range(blocks):

        box = (
            (
                i %
                target_aspect_ratio[0]
            ) * image_size,

            (
                i //
                target_aspect_ratio[0]
            ) * image_size,

            (
                (
                    i %
                    target_aspect_ratio[0]
                ) + 1
            ) * image_size,

            (
                (
                    i //
                    target_aspect_ratio[0]
                ) + 1
            ) * image_size
        )

        split_img = resized_img.crop(box)

        processed_images.append(
            split_img
        )

    # Add thumbnail when multiple tiles exist.
    if len(processed_images) != 1:

        thumbnail = image.resize(
            (
                image_size,
                image_size
            )
        )

        processed_images.append(
            thumbnail
        )

    return processed_images


# ================================================================
# FIND IMAGE COLUMN
# ================================================================

def find_image_column(df):

    candidates = [

        "image_path",
        "image",
        "image_file",
        "image_filename",
        "filename",
        "file_name",
        "img_path",
        "img",
        "image_id"
    ]

    for column in candidates:

        if column in df.columns:

            return column

    return None


# ================================================================
# IMAGE PATH RESOLUTION
# ================================================================

def resolve_image_path(
    value,
    csv_directory
):

    if pd.isna(value):

        return None

    value = str(value).strip()

    if not value:

        return None

    # ------------------------------------------------------------
    # 1. Absolute path
    # ------------------------------------------------------------

    if os.path.isabs(value):

        if os.path.isfile(value):

            return value

    # ------------------------------------------------------------
    # 2. Relative to CSV directory
    # ------------------------------------------------------------

    candidate = os.path.join(
        csv_directory,
        value
    )

    if os.path.isfile(candidate):

        return candidate

    # ------------------------------------------------------------
    # 3. Relative to project root
    # ------------------------------------------------------------

    candidate = os.path.join(
        PROJECT_ROOT,
        value
    )

    if os.path.isfile(candidate):

        return candidate

    # ------------------------------------------------------------
    # 4. Common multimodal dataset folders
    # ------------------------------------------------------------

    search_directories = [

        os.path.join(
            PROJECT_ROOT,
            "06_MMSD2.0"
        ),

        os.path.join(
            PROJECT_ROOT,
            "MEMOTION"
        ),

        os.path.join(
            PROJECT_ROOT,
            "FINAL_DATASETS",
            "MULTIMODAL_SARCASM"
        ),

        os.path.join(
            PROJECT_ROOT,
            "FINAL_DATASETS",
            "MULTIMODAL_SARCASM",
            "images"
        )
    ]

    filename = os.path.basename(value)

    for directory in search_directories:

        candidate = os.path.join(
            directory,
            filename
        )

        if os.path.isfile(candidate):

            return candidate

    # ------------------------------------------------------------
    # 5. Recursive search inside project
    #
    # This is slower, so only use it as a last resort.
    # ------------------------------------------------------------

    for root, dirs, files in os.walk(
        PROJECT_ROOT
    ):

        if filename in files:

            return os.path.join(
                root,
                filename
            )

    return None


# ================================================================
# FEATURE AGGREGATION
# ================================================================

def aggregate_vlm_features(
    vit_embeds
):

    """
    InternVL produces:

        [num_tiles, 256, 1536]

    We need one fixed-size vector per image.

    First:
        mean over the 256 visual tokens

    Result:
        [num_tiles, 1536]

    Then:
        mean over image tiles

    Final:
        [1536]

    This produces a fixed-size representation that can later
    be concatenated with CLIP/XLM-R features.
    """

    # [tiles, 256, 1536]
    tile_features = vit_embeds.mean(
        dim=1
    )

    # [tiles, 1536]
    image_feature = tile_features.mean(
        dim=0
    )

    # [1536]
    return image_feature


# ================================================================
# LOAD MODEL
# ================================================================

def load_model():

    print("\n" + "=" * 70)

    print(
        "LOADING INTERNVL3-2B"
    )

    print("=" * 70)

    print(
        f"Model : {MODEL_NAME}"
    )

    print(
        f"Dtype : {DTYPE}"
    )

    print(
        f"Device: {DEVICE}"
    )

    start = time.time()

    model = AutoModel.from_pretrained(
        MODEL_NAME,

        torch_dtype=DTYPE,

        low_cpu_mem_usage=True,

        trust_remote_code=True
    ).eval()

    model = model.to(
        DEVICE
    )

    # Explicitly freeze model.
    for parameter in model.parameters():

        parameter.requires_grad = False

    elapsed = (
        time.time() - start
    )

    print(
        f"\nModel load time : "
        f"{elapsed:.2f} seconds"
    )

    print(
        "Model status    : FROZEN"
    )

    if torch.cuda.is_available():

        allocated = (
            torch.cuda.memory_allocated()
            / 1024**3
        )

        reserved = (
            torch.cuda.memory_reserved()
            / 1024**3
        )

        print(
            f"GPU allocated   : "
            f"{allocated:.2f} GB"
        )

        print(
            f"GPU reserved    : "
            f"{reserved:.2f} GB"
        )

    return model


# ================================================================
# PROCESS ONE IMAGE
# ================================================================

def process_image(
    model,
    image_path
):

    image = Image.open(
        image_path
    ).convert("RGB")

    tiles = dynamic_preprocess(
        image
    )

    pixel_values = torch.stack(
        [
            transform(tile)
            for tile in tiles
        ]
    )

    pixel_values = pixel_values.to(
        device=DEVICE,
        dtype=DTYPE
    )

    with torch.inference_mode():

        vit_embeds = model.extract_feature(
            pixel_values
        )

    # Validate features.
    if not torch.isfinite(
        vit_embeds
    ).all():

        raise ValueError(
            "InternVL generated NaN/Inf features."
        )

    # Convert:
    #
    # [tiles, 256, 1536]
    #
    # →
    #
    # [1536]

    image_feature = aggregate_vlm_features(
        vit_embeds
    )

    image_feature = (
        image_feature
        .detach()
        .cpu()
        .float()
    )

    return (
        image_feature,
        len(tiles)
    )


# ================================================================
# MAIN
# ================================================================

def main():

    print("=" * 70)

    print(
        " INTERNVL3-2B VLM FEATURE DATASET CREATOR "
    )

    print("=" * 70)

    print(
        "\nIMPORTANT:"
    )

    print(
        "InternVL is FROZEN."
    )

    print(
        "We are extracting features, NOT training InternVL."
    )

    print()

    # ------------------------------------------------------------
    # Environment
    # ------------------------------------------------------------

    print(
        f"Python       : "
        f"{sys.version.split()[0]}"
    )

    print(
        f"PyTorch      : "
        f"{torch.__version__}"
    )

    import transformers

    print(
        f"Transformers : "
        f"{transformers.__version__}"
    )

    print(
        f"CUDA         : "
        f"{torch.cuda.is_available()}"
    )

    if not torch.cuda.is_available():

        raise RuntimeError(
            "CUDA is not available."
        )

    print(
        f"GPU          : "
        f"{torch.cuda.get_device_name(0)}"
    )

    print(
        f"VRAM         : "
        f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB"
    )

    # ------------------------------------------------------------
    # Create output directories
    # ------------------------------------------------------------

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    os.makedirs(
        FEATURE_DIR,
        exist_ok=True
    )

    # ------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------

    print("\n" + "=" * 70)

    print(
        "LOADING DATASET"
    )

    print("=" * 70)

    if not os.path.isfile(
        DATASET_CSV
    ):

        raise FileNotFoundError(
            f"Dataset not found:\n{DATASET_CSV}"
        )

    df = pd.read_csv(
        DATASET_CSV
    )

    print(
        f"Total rows     : "
        f"{len(df):,}"
    )

    print(
        "\nColumns:"
    )

    print(
        list(df.columns)
    )

    image_column = find_image_column(
        df
    )

    if image_column is None:

        raise ValueError(
            "\nCould not identify image column.\n"
            "Please check your CSV columns."
        )

    print(
        f"\nImage column   : "
        f"{image_column}"
    )

    # ------------------------------------------------------------
    # Find text and label columns
    # ------------------------------------------------------------

    text_candidates = [
        "text",
        "tweet",
        "caption",
        "sentence",
        "content",
        "ocr"
    ]

    label_candidates = [
        "label",
        "sarcasm",
        "sarcastic",
        "sarcasm_label"
    ]

    text_column = None

    for column in text_candidates:

        if column in df.columns:

            text_column = column

            break

    label_column = None

    for column in label_candidates:

        if column in df.columns:

            label_column = column

            break

    print(
        f"Text column    : "
        f"{text_column}"
    )

    print(
        f"Label column   : "
        f"{label_column}"
    )

    # ------------------------------------------------------------
    # Select number of samples
    # ------------------------------------------------------------

    if MAX_SAMPLES is None:

        working_df = df.copy()

        print(
            "\nMODE            : FULL DATASET"
        )

    else:

        working_df = df.head(
            MAX_SAMPLES
        ).copy()

        print(
            f"\nMODE            : PILOT ({MAX_SAMPLES} images)"
        )

    print(
        f"Images to process: "
        f"{len(working_df):,}"
    )

    # ------------------------------------------------------------
    # Load checkpoint
    # ------------------------------------------------------------

    completed_indices = set()

    if os.path.isfile(
        CHECKPOINT_FILE
    ):

        checkpoint_df = pd.read_csv(
            CHECKPOINT_FILE
        )

        if "dataset_index" in checkpoint_df.columns:

            completed_indices = set(
                checkpoint_df[
                    "dataset_index"
                ].astype(int)
                .tolist()
            )

        print(
            f"\nExisting completed: "
            f"{len(completed_indices):,}"
        )

    # ------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------

    model = load_model()

    # ------------------------------------------------------------
    # Existing results
    # ------------------------------------------------------------

    if os.path.isfile(
        CHECKPOINT_FILE
    ):

        results_df = pd.read_csv(
            CHECKPOINT_FILE
        )

    else:

        results_df = pd.DataFrame()

    if os.path.isfile(
        FAILED_FILE
    ):

        failed_df = pd.read_csv(
            FAILED_FILE
        )

    else:

        failed_df = pd.DataFrame()

    # ------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------

    successful = 0

    failed = 0

    skipped = 0

    total_start = time.time()

    # ------------------------------------------------------------
    # PROCESS
    # ------------------------------------------------------------

    for position, (
        dataset_index,
        row
    ) in enumerate(
        working_df.iterrows(),
        start=1
    ):

        print("\n" + "-" * 70)

        print(
            f"[{position}/{len(working_df)}]"
        )

        print(
            f"Dataset index: "
            f"{dataset_index}"
        )

        # --------------------------------------------------------
        # Skip completed
        # --------------------------------------------------------

        if dataset_index in completed_indices:

            print(
                "STATUS: ALREADY PROCESSED"
            )

            skipped += 1

            continue

        # --------------------------------------------------------
        # Resolve image
        # --------------------------------------------------------

        image_reference = row[
            image_column
        ]

        image_path = resolve_image_path(
            image_reference,
            os.path.dirname(
                DATASET_CSV
            )
        )

        print(
            f"Image reference: "
            f"{image_reference}"
        )

        if image_path is None:

            print(
                "STATUS: IMAGE NOT FOUND"
            )

            failed += 1

            failed_row = {
                "dataset_index":
                    dataset_index,

                "image_reference":
                    str(image_reference),

                "error":
                    "Image could not be resolved."
            }

            failed_df = pd.concat(
                [
                    failed_df,
                    pd.DataFrame(
                        [failed_row]
                    )
                ],
                ignore_index=True
            )

            failed_df.to_csv(
                FAILED_FILE,
                index=False
            )

            continue

        print(
            f"Resolved path: "
            f"{image_path}"
        )

        # --------------------------------------------------------
        # Process image
        # --------------------------------------------------------

        try:

            start = time.time()

            image_feature, num_tiles = (
                process_image(
                    model,
                    image_path
                )
            )

            elapsed = (
                time.time() - start
            )

            # ----------------------------------------------------
            # Feature filename
            # ----------------------------------------------------

            feature_filename = (
                f"feature_{dataset_index:06d}.npy"
            )

            feature_path = os.path.join(
                FEATURE_DIR,
                feature_filename
            )

            # ----------------------------------------------------
            # Save feature
            # ----------------------------------------------------

            np.save(
                feature_path,
                image_feature.numpy()
            )

            # ----------------------------------------------------
            # Metadata
            # ----------------------------------------------------

            result = {

                "dataset_index":
                    dataset_index,

                "image_reference":
                    str(image_reference),

                "image_path":
                    image_path,

                "feature_file":
                    feature_filename,

                "feature_dimension":
                    int(image_feature.shape[0]),

                "num_tiles":
                    num_tiles,

                "processing_seconds":
                    round(
                        elapsed,
                        3
                    ),

                "status":
                    "success"
            }

            # Preserve text.
            if text_column is not None:

                result[
                    "text"
                ] = row[
                    text_column
                ]

            # Preserve label.
            if label_column is not None:

                result[
                    "label"
                ] = row[
                    label_column
                ]

            # Preserve split if available.
            if "split" in df.columns:

                result[
                    "split"
                ] = row[
                    "split"
                ]

            # ----------------------------------------------------
            # Append checkpoint
            # ----------------------------------------------------

            results_df = pd.concat(
                [
                    results_df,
                    pd.DataFrame(
                        [result]
                    )
                ],
                ignore_index=True
            )

            results_df.to_csv(
                CHECKPOINT_FILE,
                index=False
            )

            successful += 1

            print(
                f"STATUS          : SUCCESS"
            )

            print(
                f"Feature shape   : "
                f"{tuple(image_feature.shape)}"
            )

            print(
                f"Tiles           : "
                f"{num_tiles}"
            )

            print(
                f"Time            : "
                f"{elapsed:.2f} seconds"
            )

            print(
                f"Saved           : "
                f"{feature_filename}"
            )

            # ----------------------------------------------------
            # Free GPU memory
            # ----------------------------------------------------

            torch.cuda.empty_cache()

        except Exception as e:

            failed += 1

            print(
                f"STATUS: ERROR"
            )

            print(
                f"ERROR: {e}"
            )

            failed_row = {

                "dataset_index":
                    dataset_index,

                "image_reference":
                    str(image_reference),

                "image_path":
                    str(image_path),

                "error":
                    str(e)
            }

            failed_df = pd.concat(
                [
                    failed_df,
                    pd.DataFrame(
                        [failed_row]
                    )
                ],
                ignore_index=True
            )

            failed_df.to_csv(
                FAILED_FILE,
                index=False
            )

            torch.cuda.empty_cache()

    # ------------------------------------------------------------
    # Final metadata
    # ------------------------------------------------------------

    print("\n" + "=" * 70)

    print(
        "CREATING FINAL FEATURE DATASET"
    )

    print("=" * 70)

    if os.path.isfile(
        CHECKPOINT_FILE
    ):

        final_df = pd.read_csv(
            CHECKPOINT_FILE
        )

        final_df.to_csv(
            FINAL_METADATA_FILE,
            index=False
        )

    else:

        final_df = pd.DataFrame()

    # ------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------

    total_time = (
        time.time()
        - total_start
    )

    print("\n" + "=" * 70)

    print(
        "FEATURE EXTRACTION COMPLETE"
    )

    print("=" * 70)

    print(
        f"Requested       : "
        f"{len(working_df):,}"
    )

    print(
        f"Successful      : "
        f"{successful:,}"
    )

    print(
        f"Failed          : "
        f"{failed:,}"
    )

    print(
        f"Skipped         : "
        f"{skipped:,}"
    )

    print(
        f"Total time      : "
        f"{total_time / 60:.2f} minutes"
    )

    if successful > 0:

        print(
            f"Average/image   : "
            f"{total_time / successful:.2f} seconds"
        )

    print(
        "\nOutput directory:"
    )

    print(
        OUTPUT_DIR
    )

    print(
        "\nFeature directory:"
    )

    print(
        FEATURE_DIR
    )

    print(
        "\nMetadata:"
    )

    print(
        FINAL_METADATA_FILE
    )

    print(
        "\nCheckpoint:"
    )

    print(
        CHECKPOINT_FILE
    )

    print(
        "\nFailed images:"
    )

    print(
        FAILED_FILE
    )

    if torch.cuda.is_available():

        print(
            "\nPeak GPU memory:"
        )

        print(
            f"{torch.cuda.max_memory_allocated() / 1024**3:.2f} GB"
        )

    print("\n" + "=" * 70)

    print(
        "DONE"
    )

    print("=" * 70)


# ================================================================
# RUN
# ================================================================

if __name__ == "__main__":

    main()