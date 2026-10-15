# ================================================================
#  INTERNVL3-2B  IMAGE UNDERSTANDING TEST
#  Target: Windows + RTX 4050 Laptop (6 GB VRAM)
#  Verified against: transformers 4.52.4, torch 2.11.0+cu128
#
#  Key points this script gets right:
#    1. Passes the dtype using the correct keyword for the INSTALLED
#       transformers version (torch_dtype < 4.56, dtype >= 4.56).
#       Passing `dtype=` on 4.52.x leaks into InternVLChatModel.__init__
#       and raises: unexpected keyword argument 'dtype'
#    2. Uses bfloat16 (InternVL3 checkpoints are stored in bf16; fp16
#       can overflow in the InternViT tower and produce "!!!!!!" output).
#    3. use_flash_attn=False  -> eager attention, no flash-attn needed.
#    4. pixel_values dtype/device always matched to the model.
#    5. Sanity-checks the vision features for NaN/Inf BEFORE generating,
#       so a broken install is detected instead of silently emitting "!".
# ================================================================

import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------
# CONFIGURATION
# ----------------------------------------------------------------

MODEL_NAME = "OpenGVLab/InternVL3-2B"

IMAGE_PATH = (
    r"C:\Users\Viknesh G\OneDrive\Pictures\Camera Roll"
    r"\WIN_20260729_12_57_53_Pro.jpg"
)

IMAGE_SIZE = 448

# 1280x720 -> resolves to a 2x1 tiling + 1 thumbnail = 3 tiles.
# Lower MAX_NUM to 4 or 2 if you ever hit OOM on larger images.
MIN_NUM = 1
MAX_NUM = 6

MAX_NEW_TOKENS = 256

QUESTION = (
    "<image>\n"
    "Describe this image in detail. Explain what is visible, including "
    "the people, objects, setting, activities, emotions, and overall context."
)


# ----------------------------------------------------------------
# IMPORTS THAT NEED THE ENVIRONMENT
# ----------------------------------------------------------------

import torch
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from PIL import Image

import transformers
from transformers import AutoTokenizer, AutoModel


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# ================================================================
#  ENVIRONMENT
# ================================================================

def parse_version(v):
    """Return (major, minor) from a version string like '4.52.4'."""
    parts = []
    for chunk in v.split(".")[:2]:
        digits = "".join(c for c in chunk if c.isdigit())
        parts.append(int(digits) if digits else 0)
    while len(parts) < 2:
        parts.append(0)
    return tuple(parts)


def dtype_kwarg(dtype):
    """
    transformers >= 4.56 renamed `torch_dtype` to `dtype`.
    transformers  < 4.56 does NOT recognise `dtype`; it falls through
    config parsing into the remote-code constructor and raises TypeError.
    """
    if parse_version(transformers.__version__) >= (4, 56):
        return {"dtype": dtype}
    return {"torch_dtype": dtype}


def report_environment():
    print("=" * 70)
    print("             INTERNVL3-2B IMAGE TEST")
    print("=" * 70)

    print(f"Python executable : {sys.executable}")
    print(f"Python version    : {sys.version.split()[0]}")
    print(f"PyTorch           : {torch.__version__}")
    print(f"Transformers      : {transformers.__version__}")
    print(f"Transformers path : {os.path.dirname(transformers.__file__)}")
    print(f"CUDA available    : {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"GPU               : {torch.cuda.get_device_name(0)}")
        total = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
        print(f"Total VRAM        : {total:.2f} GB")
        print(f"bfloat16 support  : {torch.cuda.is_bf16_supported()}")

    major, minor = parse_version(transformers.__version__)

    if (major, minor) >= (5, 0):
        print()
        print("!" * 70)
        print("WARNING: transformers 5.x is installed.")
        print("The InternVL3 remote code (modeling_internvl_chat.py) targets")
        print("the 4.3x-4.5x API. On 5.x it needs monkeypatches to load and")
        print("typically generates only '!' characters afterwards.")
        print("Fix, using the venv interpreter explicitly:")
        print(r'  D:\NLP_PROJECT\internvl_env\Scripts\python.exe -m pip '
              r'install "transformers==4.52.4"')
        print("!" * 70)

    if (major, minor) < (4, 37):
        print()
        print("WARNING: InternVL3 requires transformers >= 4.37.2")


def pick_dtype():
    """
    InternVL3 weights are stored in bfloat16. The RTX 4050 is Ada
    (compute capability 8.9) and has native bf16 support, so bf16 is
    the correct choice: same memory as fp16, but the wide exponent
    range avoids overflow-to-inf inside the InternViT vision tower,
    which is the usual cause of all-'!' generations.
    """
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


# ================================================================
#  DYNAMIC IMAGE PREPROCESSING (official InternVL tiling)
# ================================================================

def build_transform(input_size):
    return T.Compose([
        T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        T.Resize((input_size, input_size),
                 interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def find_closest_aspect_ratio(aspect_ratio, target_ratios,
                              width, height, image_size):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height

    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)

        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio

    return best_ratio


def dynamic_preprocess(image, min_num=MIN_NUM, max_num=MAX_NUM,
                       image_size=IMAGE_SIZE, use_thumbnail=True):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    target_ratios = set(
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if min_num <= i * j <= max_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )

    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    resized_img = image.resize((target_width, target_height))

    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        processed_images.append(resized_img.crop(box))

    assert len(processed_images) == blocks

    if use_thumbnail and len(processed_images) != 1:
        processed_images.append(image.resize((image_size, image_size)))

    return processed_images


def load_image_tensor(image_path, dtype, device,
                      input_size=IMAGE_SIZE, max_num=MAX_NUM):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"\nImage not found:\n{image_path}")

    print("\n" + "=" * 70)
    print("IMAGE PREPROCESSING")
    print("=" * 70)

    start = time.time()

    image = Image.open(image_path).convert("RGB")
    print(f"Image path        : {image_path}")
    print(f"Original size     : {image.size}")

    tiles = dynamic_preprocess(
        image, min_num=MIN_NUM, max_num=max_num,
        image_size=input_size, use_thumbnail=True
    )

    transform = build_transform(input_size)
    pixel_values = torch.stack([transform(tile) for tile in tiles])

    # Dtype and device MUST match the model, or extract_feature() will
    # silently produce garbage (or raise a dtype mismatch).
    pixel_values = pixel_values.to(device=device, dtype=dtype)

    print(f"Tiles produced    : {len(tiles)}")
    print(f"Pixel values shape: {tuple(pixel_values.shape)}")
    print(f"Device / dtype    : {pixel_values.device} / {pixel_values.dtype}")
    print(f"Preprocess time   : {time.time() - start:.2f} s")

    return pixel_values


# ================================================================
#  MODEL LOADING
# ================================================================

def load_model(dtype, device):
    print("\n" + "=" * 70)
    print("LOADING MODEL")
    print("=" * 70)
    print(f"Model  : {MODEL_NAME}")
    print(f"Dtype  : {dtype}")
    print(f"Device : {device}")

    start = time.time()

    # These kwargs are all accepted by InternVLChatModel.__init__
    # (config, vision_model, language_model, use_flash_attn).
    common_kwargs = dict(
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        use_flash_attn=False,   # eager attention; flash-attn not required
    )

    try:
        model = AutoModel.from_pretrained(
            MODEL_NAME,
            **dtype_kwarg(dtype),
            **common_kwargs,
        )
    except TypeError as err:
        if "dtype" not in str(err):
            raise
        # Last-resort fallback: load in the checkpoint's own dtype,
        # then cast. Costs a little extra host RAM during load.
        print(f"\nNOTE: dtype kwarg rejected ({err}). Retrying without it.")
        model = AutoModel.from_pretrained(MODEL_NAME, **common_kwargs)
        model = model.to(dtype)

    model = model.eval().to(device)

    load_time = time.time() - start
    print(f"\nModel load time   : {load_time:.2f} s")

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        print(f"GPU allocated     : "
              f"{torch.cuda.memory_allocated() / 1024 ** 3:.2f} GB")
        print(f"GPU reserved      : "
              f"{torch.cuda.memory_reserved() / 1024 ** 3:.2f} GB")

    # Confirm what the weights actually ended up as.
    first_param = next(model.parameters())
    print(f"Param dtype       : {first_param.dtype}")
    print(f"Param device      : {first_param.device}")

    return model


def load_tokenizer():
    print("\n" + "=" * 70)
    print("LOADING TOKENIZER")
    print("=" * 70)

    start = time.time()
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        use_fast=False,
    )
    print(f"Tokenizer load    : {time.time() - start:.2f} s")
    return tokenizer


# ================================================================
#  VISION SANITY CHECK
# ================================================================

def check_vision_features(model, pixel_values):
    """
    Runs only the vision tower + projector. If these come back NaN/Inf,
    generation will collapse to token id 0, which decodes to '!' in the
    Qwen2 vocabulary. This is the fast way to distinguish a numerics
    problem from a prompt/token-plumbing problem.
    """
    print("\n" + "=" * 70)
    print("VISION FEATURE SANITY CHECK")
    print("=" * 70)

    with torch.inference_mode():
        vit_embeds = model.extract_feature(pixel_values)

    finite = torch.isfinite(vit_embeds).all().item()
    abs_max = vit_embeds.abs().max().float().item()

    print(f"vit_embeds shape  : {tuple(vit_embeds.shape)}")
    print(f"vit_embeds dtype  : {vit_embeds.dtype}")
    print(f"All finite        : {finite}")
    print(f"Max |value|       : {abs_max:.4f}")

    if not finite:
        print("\nVision features contain NaN/Inf. Generation will produce"
              " '!' characters.")
        print("Switch to bfloat16, or reduce MAX_NUM, before retrying.")

    del vit_embeds
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return finite


# ================================================================
#  MAIN
# ================================================================

def main():
    report_environment()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = pick_dtype()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    model = load_model(dtype, device)
    tokenizer = load_tokenizer()

    pixel_values = load_image_tensor(IMAGE_PATH, dtype, device)
    num_patches = pixel_values.shape[0]

    vision_ok = check_vision_features(model, pixel_values)

    # ------------------------------------------------------------
    # GENERATION
    # ------------------------------------------------------------

    generation_config = dict(
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
    )

    print("\n" + "=" * 70)
    print("RUNNING IMAGE INFERENCE")
    print("=" * 70)
    print(f"Tiles             : {num_patches}")
    print(f"Image tokens      : ~{num_patches * 256}")
    print(f"Question          : {QUESTION.replace(chr(10), ' ')}")

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    start = time.time()

    with torch.inference_mode():
        # Single image: do NOT pass num_patches_list. chat() derives it
        # from pixel_values.shape[0] and builds the <img>...</img> block
        # and the IMG_CONTEXT tokens itself.
        response = model.chat(
            tokenizer,
            pixel_values,
            QUESTION,
            generation_config,
        )

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    inference_time = time.time() - start

    # ------------------------------------------------------------
    # RESULT
    # ------------------------------------------------------------

    print("\n" + "=" * 70)
    print("MODEL RESPONSE")
    print("=" * 70)
    print(response)

    stripped = response.replace("!", "").strip()
    if len(stripped) == 0:
        print("\n" + "!" * 70)
        print("DEGENERATE OUTPUT DETECTED (all '!').")
        print("Token id 0 in the Qwen2 vocabulary is '!', so this means the")
        print("logits were NaN or uniform. Checklist:")
        print("  1. transformers version -> pin 4.52.4 (or 4.49.0 / 4.37.2)")
        print("  2. dtype -> bfloat16, not float16")
        print("  3. no all_tied_weights_keys monkeypatch left in the script")
        print(f"  4. vision features finite this run: {vision_ok}")
        print("!" * 70)

    print("\n" + "=" * 70)
    print("PERFORMANCE")
    print("=" * 70)
    print(f"Inference time    : {inference_time:.2f} s")

    if torch.cuda.is_available():
        print(f"GPU allocated     : "
              f"{torch.cuda.memory_allocated() / 1024 ** 3:.2f} GB")
        print(f"GPU peak (infer)  : "
              f"{torch.cuda.max_memory_allocated() / 1024 ** 3:.2f} GB")
        print(f"GPU reserved      : "
              f"{torch.cuda.memory_reserved() / 1024 ** 3:.2f} GB")

    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()