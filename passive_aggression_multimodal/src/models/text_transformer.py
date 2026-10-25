"""Pretrained Transformer Text-Only Classifier (Phase 3 Task D).

Combines frozen real pretrained transformer encoder (FacebookAI/xlm-roberta-base)
with a trainable 3-class ClassificationHead.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn

from src.models.classification_head import ClassificationHead, PRIMARY_LABEL_TO_IDX
from src.models.text_encoder import PretrainedTextEncoder


class TransformerTextClassifier(nn.Module):
    """
    Text-only neural classifier:
    Tokenized input -> Frozen Transformer (768-dim) -> ClassificationHead (768 -> 256 -> num_classes)
    """

    def __init__(
        self,
        model_name_or_path: str = "FacebookAI/xlm-roberta-base",
        num_classes: int = 3,
        dropout: float = 0.1,
        device: str = "cpu",
        class_mapping: Optional[Dict[str, int]] = None,
        load_encoder_weights: bool = True,
    ):
        super().__init__()
        self.device = torch.device(device)
        self.encoder = None

        if load_encoder_weights:
            self.encoder = PretrainedTextEncoder(
                model_name_or_path=model_name_or_path,
                max_length=128,
                pooling="mean",
                device=device,
                local_files_only=True,
            )
            hidden_dim = self.encoder.hidden_dim
        else:
            hidden_dim = 768

        self.classification_head = ClassificationHead(
            in_dim=hidden_dim,
            hidden_dim=256,
            num_classes=num_classes,
            dropout=dropout,
            class_mapping=class_mapping or PRIMARY_LABEL_TO_IDX,
        )

    def forward(
        self,
        inputs: Optional[Union[torch.Tensor, List[str], str]] = None,
        comment_features: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        Forward pass.
        Accepts either 'inputs' or 'comment_features'.
        If inputs is a torch.Tensor, assumes it is already the pooled 768-dim feature representation.
        If inputs is text (or List[str]), runs encoder to extract features first.
        """
        if comment_features is not None:
            features = comment_features
        elif isinstance(inputs, torch.Tensor):
            features = inputs
        elif self.encoder is not None:
            features = self.encoder(inputs).to(self.device)
        else:
            raise ValueError("Encoder not initialized and raw text was provided.")

        return self.classification_head(features)

    def compute_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        class_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.classification_head.compute_loss(logits, targets, class_weights=class_weights)

    def predict(
        self,
        inputs: Union[torch.Tensor, List[str]],
        confidence_threshold: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.forward(inputs)
        return self.classification_head.predict(logits, confidence_threshold=confidence_threshold)
