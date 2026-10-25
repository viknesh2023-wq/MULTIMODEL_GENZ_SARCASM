# Phase 8A: Baseline Training & Multimodal Fusion Report
## Auxiliary Multimodal Sarcasm Representation Learning

**Project Title:** Context-Aware Multimodal Passive-Aggression Detection in Social Media Using Image–Comment Fusion  
**Artifact:** `reports/phase8a/phase8a_training_report.md`  
**Execution Date:** October 21, 2026  
**Evaluation Gate:** `BLOCKED_PENDING_HUMAN_ANNOTATION` (Strictly Maintained)  

---

## 1. Executive Summary & Experimental Boundary
Phase 8A successfully trained and evaluated two primary baseline architectures on the clean, group-isolated **MMSD + SarcNet Auxiliary Sarcasm Split** (27,970 records: 19,618 train / 4,143 val / 4,209 test):
1. **Model 1 (Text Baseline):** Word + Character n-gram TF-IDF with logistic_regression ($C=1.0$).
2. **Model 2 (Multimodal Fusion):** End-to-end Gated Multimodal Fusion combining frozen contextual text representations (`XLM-RoBERTa`, 768-dim) and visual representations (768-dim spatial image features) with adaptive gating on local NVIDIA RTX 4050 GPU.

> [!IMPORTANT]
> **Scientific Integrity Notice:**  
> These experiments evaluate **Auxiliary Sarcasm Detection**, which serves to pretrain cross-modal representations for subsequent transfer probing. Sarcasm labels are **not** passive-aggression ground truth. The human evaluation gate remains strictly **`BLOCKED_PENDING_HUMAN_ANNOTATION`**.

---

## 2. Experimental Environment & GPU Utilization
- **Compute Device:** cuda (NVIDIA GeForce RTX 4050 Laptop GPU)
- **Peak VRAM Allocated:** 1180.5 MB / 6,141 MB available (~17.4% memory footprint)
- **Batch Size:** 64 (conservative, leakage-safe)
- **Random Seed:** 42 (fully deterministic)

---

## 3. Comparative Test Performance (Held-Out Split: 4,209 records)

| Metric | Text-Only Baseline | Image–Comment Multimodal Fusion | Absolute Delta (Multimodal - Text) |
|---|---|---|---|
| **Macro-F1** | **0.7516** | **0.7497** | **-0.0019** |
| **Accuracy** | 0.7520 | 0.7503 | -0.0017 |
| **Balanced Accuracy** | 0.7515 | 0.7495 | -0.0020 |
| **Macro Precision** | 0.7518 | 0.7505 | -0.0013 |
| **Macro Recall** | 0.7515 | 0.7495 | -0.0020 |
| **Training Duration** | 36.89 s | 1158.03 s | - |

---

## 4. Per-Class Detailed Performance

### A. Text-Only Baseline
- **Non-Sarcastic:** Precision=0.7555, Recall=0.7659, F1=0.7607 (Support: 2166)
- **Sarcastic:** Precision=0.7481, Recall=0.7372, F1=0.7426 (Support: 2043)
- **Confusion Matrix:**
```
Predicted ->   Non-Sarcastic   Sarcastic
Actual Non-S:      1659         507
Actual Sarc:       537          1506
```

### B. Image–Comment Multimodal Fusion
- **Non-Sarcastic:** Precision=0.7479, Recall=0.7765, F1=0.7619 (Support: 2166)
- **Sarcastic:** Precision=0.7531, Recall=0.7225, F1=0.7374 (Support: 2043)
- **Confusion Matrix:**
```
Predicted ->   Non-Sarcastic   Sarcastic
Actual Non-S:      1682         484
Actual Sarc:       567          1476
```

---

## 5. Model Selection & Checkpoint Ledger
- **Text Baseline Checkpoint:** `artifacts/phase8a/text_baseline/best_text_baseline.pkl`
  - Selected via 6-candidate hyperparameter grid on validation set.
  - Best Val Macro-F1: `0.7407`.
- **Multimodal Fusion Checkpoint:** `artifacts/phase8a/multimodal_fusion/best_multimodal_fusion.pt`
  - Selected via early validation tracking across 5 epochs.
  - Best Val Macro-F1: `0.7482` (Epoch 5).

---

## 6. Evaluation Gate Status
- **Human Benchmark Evaluation Gate:** `BLOCKED_PENDING_HUMAN_ANNOTATION`  
- Both annotator response files (`annotator_1_responses.csv`, `annotator_2_responses.csv`) remain strictly preserved with 0 completed reviews.
- No passive-aggression accuracy claims are made or implied.
