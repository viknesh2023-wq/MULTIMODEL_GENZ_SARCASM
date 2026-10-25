# Context-Aware Multimodal Fusion Model Report (Phase 3 Task H)
**Project Title:** Context-Aware Multimodal Passive-Aggression Detection in Social Media Using Image–Comment Fusion  
**Artifact:** `reports/context_aware_fusion_report.md`  
**Model Architecture:** `ContextAwareMultimodalFusion` (Comment-Context Multihead Cross-Attention + Multimodal Gating)  
**Artifacts Stored:** `artifacts/phase3/context_aware_fusion/`  
**Date:** October 21, 2026  

---

## 1. Architectural Formulation & Dialogue Modeling

Task H implements **Context-Aware Multimodal Fusion**, designed specifically to detect conversational passive aggression that emerges only when comparing the target comment against prior conversational context:

1. **Comment-as-Query Multihead Cross-Attention:**
   $$\mathbf{Q} = \mathbf{h}_{\text{comment}}, \quad \mathbf{K} = \mathbf{h}_{\text{context}}, \quad \mathbf{V} = \mathbf{h}_{\text{context}}$$
   $$\text{Attn}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{Softmax}\left(\frac{\mathbf{Q} \mathbf{K}^T}{\sqrt{d_k}}\right) \mathbf{V}$$
2. **Context Incongruity Gate:**
   $$\mathbf{g} = \mathbf{m}_{\text{context}} \odot \sigma\left(\mathbf{W}_c [\mathbf{h}_{\text{comment}}; \text{Attn}(\mathbf{Q}, \mathbf{K}, \mathbf{V})] + \mathbf{b}_c\right)$$
   $$\mathbf{h}_{\text{modulated}} = \mathbf{h}_{\text{comment}} + \mathbf{g} \odot \text{Attn}(\mathbf{Q}, \mathbf{K}, \mathbf{V})$$
   *Crucial Property:* When conversation context is missing ($\mathbf{m}_{\text{context}} = 0$), the gate evaluates strictly to zero, ensuring graceful degradation to comment-only representations without artifact leakage.
3. **Multimodal Gating Across Modulated Text, Vision & OCR:**
   Combines $\mathbf{h}_{\text{modulated}}$, $\mathbf{h}_{\text{image}}$, and $\mathbf{h}_{\text{ocr}}$ with learned modality softmax weights.
4. **Primary Classification Head:** Trainable 3-class linear projection head.

---

## 2. Training Progression & Early Stopping

The model was trained on the benchmark feature partition (244 train, 52 validation, 54 test):

| Epoch | Train Loss | Validation Loss | Validation Macro-F1 | Validation Accuracy | Status |
|---|---|---|---|---|---|
| 01 | 1.0921 | 0.8353 | 0.2165 | 48.08% | Initial |
| 02 | 0.9376 | 0.8288 | 0.6190 | 53.85% | New Best |
| 03 | 0.7086 | 0.6655 | 0.6277 | 55.77% | New Best |
| 04 | 0.5118 | 0.3822 | 0.8611 | 80.77% | New Best |
| 05 | 0.1072 | 0.1333 | 0.9599 | 94.23% | New Best |
| **06** | **0.0674** | **0.0021** | **1.0000** | **100.00%** | **OPTIMAL BEST (Selected)** |
| 07 | 0.0015 | 0.0010 | 1.0000 | 100.00% | Patience: 1 |
| 08 | 0.0009 | 0.0007 | 1.0000 | 100.00% | Patience: 2 |
| 09 | 0.0006 | 0.0005 | 1.0000 | 100.00% | Patience: 3 |
| 10 | 0.0005 | 0.0004 | 1.0000 | 100.00% | Early stopping triggered |

---

## 3. Held-Out Test Evaluation Results

- **Held-Out Test Loss:** `0.0024`
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

- **Best Checkpoint:** `artifacts/phase3/context_aware_fusion/best_checkpoint.pt`
- **Audit Registry:** `artifacts/phase3/context_aware_fusion/checkpoint_registry.json`
- **Full History Log:** `artifacts/phase3/context_aware_fusion/context_aware_fusion_results.json`
