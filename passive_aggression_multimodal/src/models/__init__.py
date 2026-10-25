"""Phase 3 Neural Models Package."""
from src.models.classification_head import (
    ClassificationHead,
    PRIMARY_LABEL_TO_IDX,
    PRIMARY_IDX_TO_LABEL,
    SARCASM_LABEL_TO_IDX,
    SARCASM_IDX_TO_LABEL,
)

__all__ = [
    "ClassificationHead",
    "PRIMARY_LABEL_TO_IDX",
    "PRIMARY_IDX_TO_LABEL",
    "SARCASM_LABEL_TO_IDX",
    "SARCASM_IDX_TO_LABEL",
]
