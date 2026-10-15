import time
import torch
from transformers import pipeline


MODEL_NAME = "OpenGVLab/InternVL3-2B"

IMAGE_PATH = r"C:/Users/Viknesh G/OneDrive/Pictures/Camera Roll/WIN_20260729_12_57_53_Pro.jpg"


print("=" * 70)
print("       INTERNVL3-2B OFFICIAL PIPELINE TEST")
print("=" * 70)

print("GPU:", torch.cuda.get_device_name(0))
print("CUDA:", torch.cuda.is_available())
print("PyTorch:", torch.__version__)

print("\n" + "=" * 70)
print("LOADING PIPELINE")
print("=" * 70)

start = time.perf_counter()

pipe = pipeline(
    "image-text-to-text",
    model=MODEL_NAME,
    trust_remote_code=True,
    device=0,
    dtype=torch.float16,
)

load_time = time.perf_counter() - start

print(f"\nPipeline loading time: {load_time:.2f} sec")


print("\n" + "=" * 70)
print("RUNNING IMAGE INFERENCE")
print("=" * 70)

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "url": IMAGE_PATH,
            },
            {
                "type": "text",
                "text": (
                    "What is in this image? "
                    "Describe the main subject and visible text "
                    "in one short sentence."
                ),
            },
        ],
    },
]


start = time.perf_counter()

output = pipe(
    text=messages,
    max_new_tokens=40,
    do_sample=False,
)

inference_time = time.perf_counter() - start


print("\n" + "=" * 70)
print("MODEL OUTPUT")
print("=" * 70)

print(output)


print("\n" + "=" * 70)
print("PERFORMANCE")
print("=" * 70)

print(f"Model loading: {load_time:.2f} sec")
print(f"Inference:     {inference_time:.2f} sec")

allocated = torch.cuda.memory_allocated() / (1024 ** 3)
peak = torch.cuda.max_memory_allocated() / (1024 ** 3)

print(f"GPU allocated: {allocated:.2f} GB")
print(f"GPU peak:      {peak:.2f} GB")

print("\n" + "=" * 70)
print("TEST COMPLETE")
print("=" * 70)