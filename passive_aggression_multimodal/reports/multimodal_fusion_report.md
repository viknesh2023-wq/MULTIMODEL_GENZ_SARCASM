# Gated Multimodal Fusion Model Report (Phase 3 Task G)
**Project Title:** Context-Aware Multimodal Passive-Aggression Detection in Social Media Using Image–Comment Fusion  
**Artifact:** `reports/multimodal_fusion_report.md`  
**Model Architecture:** `GatedMultimodalFusion` (512-dim) + `ClassificationHead` (3 classes)  
**Artifacts Stored:** `artifacts/phase3/multimodal_fusion/`  
**Date:** October 21, 2026  

---

## 1. Architectural Summary & Gating Mechanism

Task G implements the core **Gated Multimodal Fusion Architecture**:
1. **Modality Projections:** Projects visual scene representations (768-dim), target comment text (768-dim), conversation context (768-dim), and OCR text (768-dim) into a shared normalized subspace of $D_{\text{hidden}} = 512$ via `ModalityProjection`.
2. **Missing Modality Gating:** When an image, dialogue context, or OCR evidence is absent or masked, its representation is zero-masked prior to cross-modal concatenation.
3. **Adaptive Modality Gating:** A learned feedforward gating network assigns dynamic importance weights $\mathbf{w} \in \Delta^3$ across the available modalities:
   $$\mathbf{w} = \text{Softmax}(\mathbf{W}_g [\mathbf{h}_{\text{comment}}; \mathbf{h}_{\text{image}}; \mathbf{h}_{\text{context}}; \mathbf{h}_{\text{ocr}}] + \mathbf{b}_g)$$
4. **Primary Classification Head:** Trainable 3-class feedforward head predicting `passive_aggressive`, `non_passive_aggressive`, or `ambiguous`.

---

## 2. Training Progression & Early Stopping

The model was trained on the benchmark feature partition (244 train, 52 validation, 54 test):

| Epoch | Train Loss | Validation Loss | Validation Macro-F1 | Validation Accuracy | Status |
|---|---|---|---|---|---|
| 01 | 1.0071 | 0.8170 | 0.2165 | 48.08% | Initial |
| 02 | 0.7335 | 0.5564 | 0.9199 | 88.46% | New Best |
| 03 | 0.3360 | 0.1434 | 0.9733 | 96.15% | New Best |
| 04 | 0.1091 | 0.2472 | 0.9733 | 96.15% | Patience: 1 |
| **05** | **0.0037** | **0.0015** | **1.0000** | **100.00%** | **OPTIMAL BEST (Selected)** |
| 06 | 0.0013 | 0.0007 | 1.0000 | 100.00% | Patience: 1 |
| 07 | 0.0007 | 0.0005 | 1.0000 | 100.00% | Patience: 2 |
| 08 | 0.0006 | 0.0004 | 1.0000 | 100.00% | Patience: 3 |
| 09 | 0.0004 | 0.0003 | 1.0000 | 100.00% | Early stopping triggered |

---

## 3. Held-Out Test Evaluation Results

- **Held-Out Test Loss:** `0.0014`
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

## 4. Artifact & Checkpoint Inventory

- **Best Checkpoint:** `artifacts/phase3/multimodal_fusion/best_checkpoint.pt`
- **Audit Registry:** `artifacts/phase3/multimodal_fusion/checkpoint_registry.json`
- **Full History Log:** `artifacts/phase3/multimodal_fusion/multimodal_fusion_results.json`
