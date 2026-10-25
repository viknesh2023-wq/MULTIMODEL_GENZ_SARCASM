"""Image preprocessing module for multimodal passive-aggression classification.

Supports:
- Robust image loading, decoding, and mode normalization (RGBA/P/CMYK -> RGB).
- Resizing with aspect-ratio preserving padding or standard bicubic interpolation.
- Normalization according to vision transformer standards (CLIP / SigLIP ImageNet stats).
- PyTorch tensor conversion.
"""
from __future__ import annotations

import os
from typing import Optional, Tuple, Union
import numpy as np
from PIL import Image
import torch


class ImagePreprocessor:
    """Preprocesses social media images for vision transformer encoders."""

    # Standard vision transformer normalization parameters (CLIP / ImageNet)
    NORM_MEAN = (0.48145466, 0.4578275, 0.40821073)
    NORM_STD = (0.26862954, 0.26130258, 0.27577711)

    def __init__(self, target_size: Tuple[int, int] = (224, 224), keep_aspect_ratio: bool = True):
        self.target_size = target_size
        self.keep_aspect_ratio = keep_aspect_ratio

    def load_and_normalize_image(self, image_path: str) -> Image.Image:
        """Loads image from path and converts to standard 8-bit RGB color mode."""
        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"Image not found at {image_path}")

        with Image.open(image_path) as img:
            # Force decoding
            img.load()
            # Handle RGBA with transparency or palette images
            if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
                # Create a solid white background
                bg = Image.new("RGB", img.size, (255, 255, 255))
                alpha = img.convert("RGBA").split()[-1]
                bg.paste(img.convert("RGBA"), mask=alpha)
                return bg
            elif img.mode != "RGB":
                return img.convert("RGB")
            else:
                return img.copy()

    def resize_image(self, img: Image.Image) -> Image.Image:
        """Resizes image to target_size, optionally applying aspect-ratio letterbox padding."""
        if not self.keep_aspect_ratio:
            return img.resize(self.target_size, Image.Resampling.BICUBIC)

        target_w, target_h = self.target_size
        orig_w, orig_h = img.size

        # Compute scaling ratio
        scale = min(target_w / orig_w, target_h / orig_h)
        new_w = max(1, int(orig_w * scale))
        new_h = max(1, int(orig_h * scale))

        resized_img = img.resize((new_w, new_h), Image.Resampling.BICUBIC)

        # Center on white canvas
        canvas = Image.new("RGB", self.target_size, (255, 255, 255))
        paste_x = (target_w - new_w) // 2
        paste_y = (target_h - new_h) // 2
        canvas.paste(resized_img, (paste_x, paste_y))
        return canvas

    def preprocess_to_tensor(self, image_input: Union[str, Image.Image]) -> torch.Tensor:
        """
        Full transformation pipeline returning normalized PyTorch tensor [3, H, W]
        with standard zero-mean unit-variance scaling.
        """
        if isinstance(image_input, str):
            img = self.load_and_normalize_image(image_input)
        else:
            img = image_input.convert("RGB")

        resized = self.resize_image(img)

        # Convert to numpy float32 [H, W, C] in range [0, 1]
        np_img = np.array(resized, dtype=np.float32) / 255.0

        # Transpose to [C, H, W]
        np_img = np_img.transpose((2, 0, 1))

        # Normalize with mean and std
        mean = np.array(self.NORM_MEAN, dtype=np.float32).reshape((3, 1, 1))
        std = np.array(self.NORM_STD, dtype=np.float32).reshape((3, 1, 1))
        norm_img = (np_img - mean) / std

        return torch.from_numpy(norm_img)
