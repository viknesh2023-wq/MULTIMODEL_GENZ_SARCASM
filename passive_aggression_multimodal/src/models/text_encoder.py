"""Pretrained Text Encoder Interface for Phase 3.

Loads real transformer checkpoints (e.g. FacebookAI/xlm-roberta-base) in evaluation mode
to extract fixed 768-dimensional contextual embeddings from social media comments and context.
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


class PretrainedTextEncoder(nn.Module):
    """
    Contextual text encoder using real pretrained Transformer models.
    Default checkpoint: FacebookAI/xlm-roberta-base (768 hidden dimensions).
    """

    def __init__(
        self,
        model_name_or_path: str = "FacebookAI/xlm-roberta-base",
        max_length: int = 128,
        pooling: str = "mean",
        device: str = "cpu",
        local_files_only: bool = True,
    ):
        super().__init__()
        self.model_name_or_path = model_name_or_path
        self.max_length = max_length
        self.pooling = pooling.lower()
        self.device = torch.device(device)

        if self.pooling not in ("mean", "cls"):
            raise ValueError(f"Pooling mode '{pooling}' not supported. Use 'mean' or 'cls'.")

        # Load real tokenizer and model
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name_or_path, local_files_only=local_files_only
            )
            self.model = AutoModel.from_pretrained(
                model_name_or_path, local_files_only=local_files_only
            )
        except Exception as e:
            # If local_files_only failed, try with fallback or raise
            if local_files_only:
                try:
                    self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, local_files_only=False)
                    self.model = AutoModel.from_pretrained(model_name_or_path, local_files_only=False)
                except Exception as ex2:
                    raise RuntimeError(
                        f"Failed to load text encoder '{model_name_or_path}': {ex2}"
                    ) from e
            else:
                raise RuntimeError(
                    f"Failed to load text encoder '{model_name_or_path}': {e}"
                ) from e

        self.hidden_dim = self.model.config.hidden_size
        self.model.to(self.device)
        self.model.eval()
        # Freeze encoder parameters for feature extraction
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, texts: Union[str, List[str]]) -> torch.Tensor:
        """
        Tokenizes and extracts pooled embeddings.
        Args:
            texts: Single text string or list of text strings.
        Returns:
            Tensor of shape [batch_size, hidden_dim]
        """
        if isinstance(texts, str):
            texts = [texts]

        if not texts:
            return torch.zeros((0, self.hidden_dim), dtype=torch.float32, device=self.device)

        # Handle empty strings gracefully
        cleaned_texts = [t if (t and t.strip()) else "<empty>" for t in texts]

        inputs = self.tokenizer(
            cleaned_texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            last_hidden_state = outputs.last_hidden_state  # [B, L, H]
            attention_mask = inputs["attention_mask"]      # [B, L]

            if self.pooling == "cls":
                embeddings = last_hidden_state[:, 0, :]
            else:  # mean pooling
                mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
                sum_embeddings = torch.sum(last_hidden_state * mask_expanded, 1)
                sum_mask = mask_expanded.sum(1)
                sum_mask = torch.clamp(sum_mask, min=1e-9)
                embeddings = sum_embeddings / sum_mask

        # Validate finite values
        if not torch.isfinite(embeddings).all():
            raise RuntimeError("Non-finite values detected in extracted text embeddings.")

        return embeddings.cpu()
