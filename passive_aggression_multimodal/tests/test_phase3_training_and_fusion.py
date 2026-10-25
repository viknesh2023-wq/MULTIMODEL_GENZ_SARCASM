"""Comprehensive Unit and Integration Tests for Phase 3 Models and Training."""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.metrics import compute_classification_metrics
from src.features.feature_store import FeatureStore, MultimodalDataset
from src.models.classification_head import PRIMARY_LABEL_TO_IDX, SARCASM_LABEL_TO_IDX
from src.models.context_aware_fusion import ContextAwareMultimodalFusion
from src.models.gated_multimodal_fusion import MultimodalFusionClassifier
from src.models.text_baseline import TextBaselineModel
from src.models.text_encoder import PretrainedTextEncoder
from src.models.vision_encoder import PretrainedVisionEncoder
from src.training.trainer import StandardTrainer, TrainingConfig


def test_text_encoder_dimensions_and_finite():
    """Verify pretrained text encoder outputs 768-dim finite embeddings."""
    encoder = PretrainedTextEncoder(
        model_name_or_path="FacebookAI/xlm-roberta-base",
        max_length=32,
        pooling="mean",
        device="cpu",
        local_files_only=True,
    )
    texts = ["This is a test comment.", "Another great response!"]
    emb = encoder(texts)
    assert emb.shape == (2, 768)
    assert torch.isfinite(emb).all()


def test_vision_encoder_missing_mask():
    """Verify vision encoder handles missing/None image paths with zero vector & False mask."""
    v_encoder = PretrainedVisionEncoder(output_dim=768, device="cpu", local_files_only=True)
    paths = [None, "non_existent_file_path_123.jpg"]
    emb, mask = v_encoder(paths)
    assert emb.shape == (2, 768)
    assert mask.shape == (2,)
    assert not mask.any()
    assert (emb == 0).all()


def test_gated_multimodal_fusion_forward_backward():
    """Verify GatedMultimodalFusion forward pass, backward gradient flow, and loss."""
    model = MultimodalFusionClassifier(
        image_dim=768, text_dim=768, hidden_dim=512, num_classes_primary=3, num_classes_auxiliary=3
    )
    b = 4
    c_emb = torch.randn(b, 768, requires_grad=True)
    i_emb = torch.randn(b, 768, requires_grad=True)
    ctx_emb = torch.randn(b, 768, requires_grad=True)
    ocr_emb = torch.randn(b, 768, requires_grad=True)
    img_mask = torch.tensor([True, False, True, False])
    ctx_mask = torch.tensor([True, True, False, False])
    ocr_mask = torch.tensor([False, True, False, True])
    targets = torch.tensor([0, 1, 2, 0], dtype=torch.long)

    out = model(
        comment_features=c_emb,
        image_features=i_emb,
        context_features=ctx_emb,
        ocr_features=ocr_emb,
        image_masks=img_mask,
        context_masks=ctx_mask,
        ocr_masks=ocr_mask,
    )
    assert out["primary_logits"].shape == (b, 3)
    assert out["auxiliary_logits"].shape == (b, 3)
    assert out["gate_weights"].shape == (b, 4)
    assert torch.isfinite(out["primary_logits"]).all()

    loss = model.compute_loss(out["primary_logits"], targets)
    assert torch.isfinite(loss)
    loss.backward()
    assert c_emb.grad is not None


def test_context_aware_fusion_forward_backward():
    """Verify ContextAwareMultimodalFusion multihead attention gating and loss flow."""
    model = ContextAwareMultimodalFusion(image_dim=768, text_dim=768, hidden_dim=512, num_classes=3)
    b = 2
    c_emb = torch.randn(b, 768, requires_grad=True)
    ctx_emb = torch.randn(b, 768, requires_grad=True)
    i_emb = torch.randn(b, 768, requires_grad=True)
    targets = torch.tensor([1, 2], dtype=torch.long)

    out = model(comment_features=c_emb, image_features=i_emb, context_features=ctx_emb)
    assert out["primary_logits"].shape == (b, 3)
    assert out["multimodal_weights"].shape == (b, 3)

    loss = model.compute_loss(out, targets)
    loss.backward()
    assert c_emb.grad is not None


def test_trainer_reproducibility():
    """Verify StandardTrainer trains without errors and saves valid checkpoints."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        from src.models.classification_head import ClassificationHead
        head = ClassificationHead(in_dim=16, hidden_dim=16, num_classes=3)
        cfg = TrainingConfig(epochs=2, batch_size=4, seed=42, class_names=["c0", "c1", "c2"])
        trainer = StandardTrainer(model=head, config=cfg, checkpoint_dir=tmp_dir)

        x_train = torch.randn(16, 16)
        y_train = torch.randint(0, 3, (16,))
        loader = DataLoader(TensorDataset(x_train, y_train), batch_size=4)

        summary = trainer.fit(loader, loader)
        assert summary["best_val_macro_f1"] >= 0.0
        assert os.path.exists(summary["best_checkpoint"])


def test_metric_computation():
    """Verify precision, recall, F1, and confusion matrix calculation."""
    y_true = [0, 1, 2, 0, 1, 2]
    y_pred = [0, 1, 1, 0, 1, 2]  # one mistake: true 2 predicted as 1
    names = ["pa", "non_pa", "amb"]
    metrics = compute_classification_metrics(y_true, y_pred, class_names=names, prefix="t_")

    assert metrics["t_accuracy"] == 5 / 6
    assert metrics["t_total_samples"] == 6
    assert "t_macro_f1" in metrics
    assert metrics["t_per_class"]["pa"]["precision"] == 1.0


def test_label_separation_integrity():
    """Verify primary and sarcasm label dictionaries are strictly separate."""
    # Ensure no label conflation
    assert PRIMARY_LABEL_TO_IDX["passive_aggressive"] == 0
    assert "sarcastic" not in PRIMARY_LABEL_TO_IDX
    assert SARCASM_LABEL_TO_IDX["sarcastic"] == 0
    assert "passive_aggressive" not in SARCASM_LABEL_TO_IDX
