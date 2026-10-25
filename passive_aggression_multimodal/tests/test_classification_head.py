"""Unit tests for Trainable Classification Head in Phase 3."""
import os
import sys
import tempfile
from pathlib import Path
import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.classification_head import (
    ClassificationHead,
    PRIMARY_LABEL_TO_IDX,
    PRIMARY_IDX_TO_LABEL,
)


def test_classification_head_shapes():
    """Verify logits shape across varying batch sizes including B=1."""
    head = ClassificationHead(in_dim=512, hidden_dim=256, num_classes=3)

    # Batch size 4
    x = torch.randn(4, 512)
    logits = head(x)
    assert logits.shape == (4, 3)

    # Batch size 1
    x_single = torch.randn(1, 512)
    logits_single = head(x_single)
    assert logits_single.shape == (1, 3)


def test_classification_head_forward_backward():
    """Verify forward and backward gradients flow properly through the head."""
    head = ClassificationHead(in_dim=128, hidden_dim=64, num_classes=3)
    x = torch.randn(8, 128, requires_grad=True)
    targets = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1], dtype=torch.long)

    logits = head(x)
    loss = head.compute_loss(logits, targets)
    assert torch.isfinite(loss)
    assert loss.item() > 0.0

    loss.backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    for param in head.parameters():
        assert param.grad is not None


def test_classification_head_input_validation():
    """Verify rejection of NaN/Inf, dimension mismatch, and invalid target labels."""
    head = ClassificationHead(in_dim=64, hidden_dim=32, num_classes=3)

    # Non-finite input
    nan_tensor = torch.tensor([[float("nan")] * 64])
    with pytest.raises(ValueError, match="NaN or Inf"):
        head(nan_tensor)

    # Dimension mismatch
    wrong_dim = torch.randn(2, 32)
    with pytest.raises(ValueError, match="dimension mismatch"):
        head(wrong_dim)

    # Out of bounds targets
    valid_logits = head(torch.randn(2, 64))
    invalid_targets = torch.tensor([0, 5], dtype=torch.long)
    with pytest.raises(ValueError, match="out of bounds"):
        head.compute_loss(valid_logits, invalid_targets)


def test_classification_head_prediction_and_abstention():
    """Verify softmax probability computation and threshold-based abstention."""
    head = ClassificationHead(in_dim=64, hidden_dim=32, num_classes=3)
    logits = torch.tensor([
        [10.0, 1.0, 0.0],  # high confidence class 0
        [1.0, 1.1, 1.05],  # low confidence / uniform
    ])
    preds, probs = head.predict(logits, confidence_threshold=0.8, abstention_class_idx=2)
    assert preds[0].item() == 0  # Confident -> class 0
    assert preds[1].item() == 2  # Low confidence -> abstains to 2 (ambiguous)
    assert probs.shape == (2, 3)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(2), atol=1e-5)


def test_classification_head_save_and_reload():
    """Verify checkpoint serialization and parameter fidelity."""
    head = ClassificationHead(in_dim=128, hidden_dim=64, num_classes=3)
    head.eval()
    x = torch.randn(2, 128)
    with torch.no_grad():
        initial_logits = head(x)

    with tempfile.TemporaryDirectory() as tmp_dir:
        ckpt_path = os.path.join(tmp_dir, "test_head.pt")
        head.save_checkpoint(ckpt_path, extra_metadata={"experiment": "unit_test"})

        loaded_head = ClassificationHead.load_checkpoint(ckpt_path)
        loaded_head.eval()
        with torch.no_grad():
            reloaded_logits = loaded_head(x)

        assert torch.allclose(initial_logits, reloaded_logits, atol=1e-5)
        assert loaded_head.in_dim == head.in_dim
        assert loaded_head.num_classes == head.num_classes
