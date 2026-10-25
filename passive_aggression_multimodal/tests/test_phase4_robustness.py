"""Comprehensive Robustness and Modality-Sensitivity Tests (Phase 4 Task H).

Validates pipeline behavior against edge cases, corrupted inputs, missing modalities,
varying batch sizes, out-of-distribution tokens, and executes 6-way modality sensitivity checks.
"""
import io
import os
import pytest
import torch
import numpy as np
from PIL import Image

from src.preprocessing.text_preprocessor import PragmaticTextPreprocessor
from src.preprocessing.image_preprocessor import ImagePreprocessor
from src.models.gated_multimodal_fusion import MultimodalFusionClassifier
from src.models.classification_head import ClassificationHead
from src.evaluation.metrics import compute_selective_metrics


@pytest.fixture
def text_preprocessor():
    return PragmaticTextPreprocessor()


@pytest.fixture
def image_preprocessor():
    return ImagePreprocessor()


@pytest.fixture
def fusion_classifier():
    model = MultimodalFusionClassifier(
        image_dim=768,
        text_dim=768,
        hidden_dim=512,
        num_classes_primary=3,
        dropout=0.1,
    )
    model.eval()
    return model


def test_empty_and_whitespace_text_handling(text_preprocessor):
    """Ensure empty or whitespace-only strings are handled safely without crashing."""
    assert text_preprocessor.clean_encoding("") == ""
    assert text_preprocessor.clean_encoding("   \n\t  ") == "   \n\t  "
    feats = text_preprocessor.extract_pragmatic_features("")
    assert feats["char_length"] == 0
    assert feats["has_all_caps"] is False
    assert feats["has_emojis"] is False


def test_pragmatic_token_preservation(text_preprocessor):
    """Ensure emojis, punctuation repeats, negation, and ALL CAPS are preserved."""
    sample = "I NEVER said that... did you honestly believe me??? 😂😒 #NOTMYJOB"
    clean = text_preprocessor.clean_encoding(sample)
    assert "😂" in clean
    assert "😒" in clean
    assert "..." in clean
    assert "???" in clean
    assert "NEVER" in clean

    feats = text_preprocessor.extract_pragmatic_features(clean)
    assert feats["has_all_caps"] is True
    assert feats["has_repeated_punct"] is True
    assert feats["has_emojis"] is True


def test_extremely_long_comment_handling(text_preprocessor):
    """Ensure comments with > 2000 characters process safely without memory explosion."""
    long_comment = "So deeply impressed by your exceptional competence. " * 100  # ~5200 chars
    clean = text_preprocessor.clean_encoding(long_comment)
    assert len(clean) > 4000
    feats = text_preprocessor.extract_pragmatic_features(clean)
    assert feats["char_length"] > 4000
    assert feats["word_count"] > 500


def test_corrupt_and_missing_image_handling(image_preprocessor, tmp_path):
    """Ensure missing paths raise FileNotFoundError and corrupt images raise exceptions cleanly."""
    with pytest.raises(FileNotFoundError):
        image_preprocessor.load_and_normalize_image(str(tmp_path / "non_existent.jpg"))

    corrupt_file = tmp_path / "corrupt.jpg"
    with open(corrupt_file, "wb") as f:
        f.write(b"NOT_A_REAL_IMAGE_DATA_CORRUPT_BYTES")

    with pytest.raises(Exception):
        image_preprocessor.load_and_normalize_image(str(corrupt_file))


def test_batch_size_one_inference(fusion_classifier):
    """Ensure model executes forward pass with batch size = 1 without dimensionality collapse."""
    comment = torch.randn(1, 768)
    image = torch.randn(1, 768)
    image_mask = torch.tensor([1.0])

    with torch.no_grad():
        out = fusion_classifier(
            comment_features=comment,
            image_features=image,
            image_masks=image_mask,
        )

    assert out["primary_logits"].shape == (1, 3)
    assert out["fused_rep"].shape == (1, 512)
    assert out["gate_weights"].shape == (1, 4)
    assert not torch.isnan(out["primary_logits"]).any()


def test_invalid_feature_dimensions(fusion_classifier):
    """Ensure dimension mismatches raise an error."""
    wrong_dim_comment = torch.randn(2, 512)  # Expected 768
    with pytest.raises(Exception):
        fusion_classifier(comment_features=wrong_dim_comment)


def test_uncertain_prediction_selective_abstention():
    """Ensure uncertain predictions are routed to abstention under confidence thresholding."""
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 1])
    # Case with 2 highly confident and 2 uncertain predictions
    y_prob = np.array([
        [0.90, 0.05, 0.05],
        [0.10, 0.85, 0.05],
        [0.35, 0.35, 0.30],  # Max conf = 0.35 < 0.60
        [0.40, 0.40, 0.20],  # Max conf = 0.40 < 0.60
    ])

    sel = compute_selective_metrics(y_true, y_pred, y_prob, threshold=0.60, abstain_class=None)
    assert sel["coverage"] == 0.50
    assert sel["covered_samples"] == 2
    assert sel["selective_accuracy"] == 1.0


def test_six_way_modality_ablation_sensitivity(fusion_classifier):
    """
    Evaluates 6 modality combinations on identical inputs to verify gated sensitivity:
    1. Comment only
    2. Image + Comment
    3. Comment + Context
    4. Image + Comment + Context
    5. Image + Comment + OCR
    6. Full multimodal (Comment + Image + Context + OCR)
    """
    B = 4
    torch.manual_seed(42)
    comment = torch.randn(B, 768)
    image = torch.randn(B, 768)
    context = torch.randn(B, 768)
    ocr = torch.randn(B, 768)

    # 1. Comment only
    with torch.no_grad():
        out1 = fusion_classifier(
            comment_features=comment,
            image_features=None,
            context_features=None,
            ocr_features=None,
        )
    assert out1["primary_logits"].shape == (B, 3)

    # 2. Image + Comment
    with torch.no_grad():
        out2 = fusion_classifier(
            comment_features=comment,
            image_features=image,
            image_masks=torch.ones(B),
        )
    assert out2["primary_logits"].shape == (B, 3)

    # 3. Comment + Context
    with torch.no_grad():
        out3 = fusion_classifier(
            comment_features=comment,
            context_features=context,
            context_masks=torch.ones(B),
        )
    assert out3["primary_logits"].shape == (B, 3)

    # 4. Image + Comment + Context
    with torch.no_grad():
        out4 = fusion_classifier(
            comment_features=comment,
            image_features=image,
            context_features=context,
            image_masks=torch.ones(B),
            context_masks=torch.ones(B),
        )
    assert out4["primary_logits"].shape == (B, 3)

    # 5. Image + Comment + OCR
    with torch.no_grad():
        out5 = fusion_classifier(
            comment_features=comment,
            image_features=image,
            ocr_features=ocr,
            image_masks=torch.ones(B),
            ocr_masks=torch.ones(B),
        )
    assert out5["primary_logits"].shape == (B, 3)

    # 6. Full multimodal
    with torch.no_grad():
        out6 = fusion_classifier(
            comment_features=comment,
            image_features=image,
            context_features=context,
            ocr_features=ocr,
            image_masks=torch.ones(B),
            context_masks=torch.ones(B),
            ocr_masks=torch.ones(B),
        )
    assert out6["primary_logits"].shape == (B, 3)

    # Verify that different modality combinations yield distinct fused representations
    rep_comment = out1["fused_rep"]
    rep_img_comment = out2["fused_rep"]
    rep_full = out6["fused_rep"]

    assert not torch.allclose(rep_comment, rep_img_comment, atol=1e-4)
    assert not torch.allclose(rep_img_comment, rep_full, atol=1e-4)
