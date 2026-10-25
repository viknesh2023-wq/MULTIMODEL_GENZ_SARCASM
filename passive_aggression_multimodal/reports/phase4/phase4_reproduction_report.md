# Phase 4 Experimental Reproduction and Audit Report (Task D)
**Project Title:** Context-Aware Multimodal Passive-Aggression Detection in Social Media Using Image–Comment Fusion  
**Auditor:** Senior ML Research Engineer & Experimental Auditor  
**Artifact:** `reports/phase4/phase4_reproduction_report.md`  
**Data Artifact:** `artifacts/phase4/reproduction_summary.json`  
**Date:** October 21, 2026  

---

## 1. Executive Summary & Reproduction Status

This audit executed an independent reproduction of Phase 3 experiments using identical configurations, seeds, and models. The audit had three primary objectives:
1. Verify whether Phase 3 reported results are numerically reproducible from saved checkpoints.
2. Investigate the anomalous **Macro-F1 0.2167** reported for Exp E (Context-Aware Fusion).
3. Evaluate models on the newly constructed **Clean Grouped Zero-Leakage Split** (`data/splits/clean_synthetic_splits_registry.json`) to determine whether performance holds when comment duplicates are eliminated.

### Overall Reproduction Status: `REPRODUCED_WITH_METHODOLOGICAL_QUALIFICATIONS`

---

## 2. Reproduction Results Matrix

| Experiment / Model | Phase 3 Reported | Phase 4 Reproduced | Clean Zero-Leakage Split | Reproduction Status | Key Finding / Mechanism |
|---|---|---|---|---|---|
| **TF-IDF + Logistic Regression** | Macro-F1: 1.0000 | Macro-F1: **1.0000** | Macro-F1: **1.0000** | `REPRODUCED` | Reaches 100% on original due to duplicate leakage; reaches 100% on clean split due to synthetic archetype markers. |
| **XLM-RoBERTa Text Baseline** | Macro-F1: 1.0000 | Macro-F1: **1.0000** | Macro-F1: **1.0000** | `REPRODUCED` | Reaches 100% on original and clean synthetic splits. |
| **Gated Multimodal Fusion** | Macro-F1: 1.0000 | Macro-F1: **1.0000** | Macro-F1: **1.0000** | `REPRODUCED` | Runs with zero-masked image and OCR vectors on synthetic data. |
| **Exp E: Context-Aware Fusion (Patience=3)** | Macro-F1: 0.2167 | Macro-F1: **0.2167** (orig ablation) | Macro-F1: **1.0000** (converged) | `EXPLAINED_AND_RESOLVED` | Premature early stopping at epoch 4 before multihead cross-attention converged. |
| **Auxiliary Sarcasm Baseline** | Macro-F1: 0.7260 | Macro-F1: **0.7260** | N/A (Grouped real splits) | `REPRODUCED` | 7,000 real multimodal records from MMSD/SarcNet. Real-world uncertainty verified. |

---

## 3. Investigation of Anomalous Results

### 3.1 The 100% Accuracy Ceiling on Synthetic Scenarios
The investigation confirms that the 100% accuracy reported across multiple models is caused by two compounding factors:
1. **Severe Duplicate Comment Leakage in Phase 3:** In the original split, 66.7% of the test set had identical twin comments in the training set.
2. **Synthetic Salience of Scenario Archetypes:** Even when exact comment duplicates are eliminated (Clean Grouped Split), synthetic scenarios use explicit archetype markers (e.g., *"Per my previous email"*, *"Bless your heart"*, *"With all due respect"*, *"Must be nice"*). Models easily achieve near-perfect classification on handcrafted synthetic sentences. This confirms that synthetic benchmarks are **insufficient to prove real-world generalization**.

### 3.2 The Exp E (Macro-F1: 0.2167) Anomaly
During Phase 3 ablations, Exp E (Image + Comment + Context without OCR) reported a catastrophic drop to Macro-F1 0.2167 (chance level on 3-class distribution).
- **Audit Diagnosis:** In `scripts/run_ablation_studies.py`, `early_stopping_patience` was hardcoded to `3` (with 10 maximum epochs). The multihead cross-attention layer required 4 warm-up epochs before finding the gradient descent trajectory.
- **Empirical Proof:** At epoch 4, because validation Macro-F1 had not improved past the initial epoch 1 level (0.2165), early stopping aborted the run, preserving untrained random weights.
- When `early_stopping_patience` was set to `5` in `scripts/reproduce_phase4_audit.py`, the exact same model reached convergence by epoch 5, achieving Macro-F1 1.0000.
- **Conclusion:** The low score of 0.2167 was an artifact of premature early stopping, not an inherent failure of the context-aware architecture.
