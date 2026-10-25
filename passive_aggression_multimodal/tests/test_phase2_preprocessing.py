"""Tests for Phase 2 text, context, image, and OCR preprocessing pipelines."""
from __future__ import annotations

import os
import tempfile
import pytest
import torch
from PIL import Image

from src.preprocessing.text_preprocessor import TextPreprocessor
from src.preprocessing.context_preprocessor import ContextPreprocessor
from src.preprocessing.image_preprocessor import ImagePreprocessor
from src.preprocessing.ocr_processor import OCRProcessor


def test_text_preprocessor_preserves_pragmatics():
    tp = TextPreprocessor(normalize_urls=True, normalize_mentions=True)
    raw = "WHY did you do that?! Never again... 🔥🔥 smh @user https://t.co/abc1234"
    cleaned = tp.preprocess(raw)

    # 1. Preserves capitalization if significant
    assert "WHY" in cleaned, "Capitalization should be preserved"
    # 2. Preserves negation
    assert "never" in cleaned.lower(), "Negation should be preserved"
    # 3. Preserves emojis
    assert "🔥" in cleaned, "Emojis should be preserved"
    # 4. Preserves slang
    assert "smh" in cleaned, "Slang acronyms should be preserved"
    # 5. Preserves repeated punctuation
    assert "?!" in cleaned or "..." in cleaned, "Repeated punctuation should be preserved"
    # 6. Normalizes URLs and user handles
    assert "<url>" in cleaned.lower(), "URLs should be normalized"
    assert "<user>" in cleaned.lower(), "Mentions should be normalized"


def test_context_preprocessor_message_ordering():
    cp = ContextPreprocessor()
    caption = "Original announcement"
    thread = ["Step 1: First event", "Step 2: Followup discussion", "Step 3: Concluding remark"]

    # 1. Test chronological turn formatting
    turns_formatted = cp.format_dialogue_turns(thread)
    assert "[Turn 1]" in turns_formatted
    assert "[Turn 2]" in turns_formatted
    assert "[Turn 3]" in turns_formatted

    # Preserves chronological ordering
    idx_1 = turns_formatted.index("Step 1")
    idx_2 = turns_formatted.index("Step 2")
    idx_3 = turns_formatted.index("Step 3")
    assert idx_1 < idx_2 < idx_3, "Thread messages must maintain strict sequential order"

    # 2. Test context-aware input packaging
    prefix, comment = cp.build_context_aware_input(
        target_comment="Target comment text",
        caption=caption,
        conversation_context=thread,
    )
    assert "[Post Caption]: Original announcement" in prefix
    assert "[Thread]:" in prefix
    assert comment == "Target comment text"


def test_image_preprocessor_resizing_and_tensors():
    ip = ImagePreprocessor(target_size=(224, 224))

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        # Create synthetic test image (RGBA mode to test conversion)
        img = Image.new("RGBA", (400, 300), color=(255, 0, 0, 128))
        img.save(tmp_path)

        norm_img = ip.load_and_normalize_image(tmp_path)
        assert norm_img.mode == "RGB"

        resized = ip.resize_image(norm_img)
        assert resized.size == (224, 224), f"Expected size (224, 224), got {resized.size}"

        tensor = ip.preprocess_to_tensor(tmp_path)
        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (3, 224, 224), f"Unexpected tensor shape: {tensor.shape}"

    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def test_ocr_processor_and_fallback():
    ocr = OCRProcessor()

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        # Create an image with high contrast shapes
        img = Image.new("RGB", (200, 100), color=(255, 255, 255))
        img.save(tmp_path)

        res = ocr.process_image(tmp_path)
        assert hasattr(res, "has_text")
        assert hasattr(res, "extracted_text")
        assert hasattr(res, "engine_name")
        assert res.status in ["SUCCESS_TEXT_DETECTED", "SUCCESS_NO_TEXT_FOUND", "FALLBACK_DETECTOR_TEXT", "FALLBACK_DETECTOR_NO_TEXT"]

        # Ensure no crash on invalid file
        invalid_res = ocr.process_image("non_existent_file.png")
        assert invalid_res.status == "ERROR"
        assert invalid_res.extracted_text == ""
        assert invalid_res.has_text is False

    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
