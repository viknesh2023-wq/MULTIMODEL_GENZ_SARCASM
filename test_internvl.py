import time
import torch
from transformers import AutoModel, AutoTokenizer
import transformers.modeling_utils as modeling_utils


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = "OpenGVLab/InternVL3-2B"


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("        InternVL3-2B - GPU LOADING TEST")
print("=" * 70)

print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"PyTorch: {torch.__version__}")

if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    total_vram = props.total_memory / (1024 ** 3)

    print(f"GPU VRAM: {total_vram:.2f} GB")


# ============================================================
# CLEAR GPU MEMORY
# ============================================================

if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


# ============================================================
# START TIMER
# ============================================================

print("\n" + "=" * 70)
print("LOADING INTERNVL3-2B")
print("=" * 70)

print("\nModel:")
print(MODEL_NAME)

print("\nThe model is loaded directly onto the GPU.")
print("No CPU/GPU offloading is being used.")

start_time = time.perf_counter()


# ============================================================
# TRANSFORMERS 5.x COMPATIBILITY FIX
# ============================================================

# InternVL uses custom model code.
# Some InternVL versions expect this attribute while
# Transformers 5.x uses the newer naming internally.

if not hasattr(
    modeling_utils.PreTrainedModel,
    "all_tied_weights_keys"
):
    modeling_utils.PreTrainedModel.all_tied_weights_keys = {}


# ============================================================
# LOAD MODEL
# ============================================================

model = AutoModel.from_pretrained(
    MODEL_NAME,
    dtype=torch.float16,
    low_cpu_mem_usage=True,
    trust_remote_code=True
).eval()


# ============================================================
# SECOND COMPATIBILITY FIX
# ============================================================

if not hasattr(model, "all_tied_weights_keys"):
    model.all_tied_weights_keys = {}


# ============================================================
# MOVE MODEL TO GPU
# ============================================================

print("\nMoving model to GPU...")

model = model.cuda()

load_time = time.perf_counter() - start_time


# ============================================================
# LOAD TOKENIZER
# ============================================================

print("\nLoading tokenizer...")

tokenizer_start = time.perf_counter()

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    use_fast=False
)

tokenizer_time = time.perf_counter() - tokenizer_start


# ============================================================
# GPU MEMORY INFORMATION
# ============================================================

print("\n" + "=" * 70)
print("GPU MEMORY")
print("=" * 70)

if torch.cuda.is_available():

    allocated = (
        torch.cuda.memory_allocated()
        / (1024 ** 3)
    )

    reserved = (
        torch.cuda.memory_reserved()
        / (1024 ** 3)
    )

    peak = (
        torch.cuda.max_memory_allocated()
        / (1024 ** 3)
    )

    print(f"Allocated: {allocated:.2f} GB")
    print(f"Reserved:  {reserved:.2f} GB")
    print(f"Peak:      {peak:.2f} GB")


# ============================================================
# RESULTS
# ============================================================

print("\n" + "=" * 70)
print("RESULTS")
print("=" * 70)

print(f"\nModel loading time: {load_time:.2f} seconds")
print(f"Tokenizer loading:  {tokenizer_time:.2f} seconds")

total_time = load_time + tokenizer_time

print(f"Total startup time: {total_time:.2f} seconds")


# ============================================================
# TARGET CHECK
# ============================================================

print("\n" + "=" * 70)
print("30-60 SECOND TARGET")
print("=" * 70)

if total_time <= 30:

    print("STATUS: EXCELLENT")
    print("Model startup is under 30 seconds.")

elif total_time <= 60:

    print("STATUS: PASSED")
    print("Model startup is within the 30-60 second target.")

else:

    print("STATUS: OVER TARGET")
    print("Model startup is above 60 seconds.")


# ============================================================
# MODEL INFORMATION
# ============================================================

print("\n" + "=" * 70)
print("MODEL INFORMATION")
print("=" * 70)

print(f"\nModel class: {type(model).__name__}")
print(f"Tokenizer:   {type(tokenizer).__name__}")
print(f"Device:      {next(model.parameters()).device}")
print(f"Dtype:       {next(model.parameters()).dtype}")


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 70)
print("INTERNVL3-2B LOADING TEST COMPLETE")
print("=" * 70)