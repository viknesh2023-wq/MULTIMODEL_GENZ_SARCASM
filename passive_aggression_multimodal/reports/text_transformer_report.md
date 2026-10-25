# Pretrained Transformer Text-Only Baseline Report (Phase 3 Task D)
**Project Title:** Context-Aware Multimodal Passive-Aggression Detection in Social Media Using Image–Comment Fusion  
**Artifact:** `reports/text_transformer_report.md`  
**Model Architecture:** Frozen XLM-RoBERTa-base (768-dim) + Trainable 3-Class ClassificationHead  
**Artifacts Stored:** `artifacts/phase3/text_transformer/`  
**Date:** October 21, 2026  

---

## 1. Model Configuration & Architecture

Task D evaluates a transformer text-only baseline utilizing real pretrained weights:
- **Base Checkpoint:** `FacebookAI/xlm-roberta-base` (1,077.4 MB, 768 hidden dimension) loaded from local Hugging Face cache.
- **Classification Head:** Linear(768, 256) $\rightarrow$ LayerNorm(256) $\rightarrow$ GELU $\rightarrow$ Dropout(0.1) $\rightarrow$ Linear(256, 3).
- **Execution Mode:** Frozen encoder with fixed evaluation mode representations; trainable classification head.
- **Optimization:** AdamW ($\text{LR} = 10^{-3}$, weight decay $= 10^{-4}$), batch size 16, gradient clipping norm 1.0.
- **Stopping Policy:** Early stopping monitored on validation Macro-F1 with patience of 4 epochs.

---

## 2. Training and Validation Progression

| Epoch | Train Loss | Validation Loss | Validation Macro-F1 | Validation Accuracy | Status |
|---|---|---|---|---|---|
| 01 | 0.9279 | 0.6396 | 0.8151 | 88.46% | New Best |
| 02 | 0.5664 | 0.3549 | 0.9733 | 96.15% | New Best |
| 03 | 0.1618 | 0.0952 | 0.9733 | 96.15% | Patience: 1 |
| 04 | 0.0548 | 0.1151 | 0.9599 | 94.23% | Patience: 2 |
| **05** | **0.0049** | **0.0013** | **1.0000** | **100.00%** | **OPTIMAL BEST (Selected)** |
| 06 | 0.0009 | 0.0006 | 1.0000 | 100.00% | Patience: 1 |
| 07 | 0.0005 | 0.0003 | 1.0000 | 100.00% | Patience: 2 |
| 08 | 0.0003 | 0.0002 | 1.0000 | 100.00% | Patience: 3 |
| 09 | 0.0002 | 0.0002 | 1.0000 | 100.00% | Early stopping triggered |

---

## 3. Held-Out Test Evaluation Results

The optimal checkpoint (Epoch 05) was reloaded and evaluated on the held-out test split (54 instances):

- **Held-Out Test Loss:** `0.0013`
- **Held-Out Test Macro-F1:** `1.0000`
- **Held-Out Test Accuracy:** `1.0000`
- **Held-Out Test Balanced Accuracy:** `1.0000`

### Per-Class Evaluation Breakdown:
| Class Name | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| `passive_aggressive` | 1.000 | 1.000 | 1.000 | 26 |
| `non_passive_aggressive` | 1.000 | 1.000 | 1.000 | 26 |
| `ambiguous` | 1.000 | 1.000 | 1.000 | 2 |

---

## 4. Checkpoint Verification and Reproducibility

- **Best Checkpoint Path:** `artifacts/phase3/text_transformer/best_checkpoint.pt`
- **Checksum Audit:** Registered in `checkpoint_registry.json`.
- **Full History Log:** `artifacts/phase3/text_transformer/text_transformer_results.json`.
