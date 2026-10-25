# Traditional Text-Only Baseline Report (Phase 3 Task C)
**Project Title:** Context-Aware Multimodal Passive-Aggression Detection in Social Media Using Image–Comment Fusion  
**Artifact:** `reports/text_baseline_report.md`  
**Models Evaluated:** TF-IDF + Logistic Regression, TF-IDF + Linear SVM  
**Artifacts Stored:** `artifacts/phase3/text_baseline/`  
**Date:** October 21, 2026  

---

## 1. Executive Summary

This report documents the execution of **Task C: Traditional Text-Only Baseline**. In strict accordance with Phase 3 guidelines:
1. Feature extraction used word n-grams (1, 2) and character n-grams (3, 5) fit exclusively on the training split to prevent vocabulary leakage.
2. Pragmatic cues (punctuation, emojis, casing emphasis, negation words) were preserved.
3. Model selection (Logistic Regression vs. Linear SVM across regularization parameters $C \in \{0.5, 1.0, 5.0\}$) was conducted strictly using validation Macro-F1.
4. Held-out test sets were evaluated only once on the selected best model.
5. Primary passive-aggression benchmarks and auxiliary sarcasm benchmarks were executed and evaluated independently.

---

## 2. Model Selection & Hyperparameter Search

### 2.1 Primary Task Benchmark (Synthetic Pilot 350)
- **Data Splits:** 244 Train (70%), 52 Validation (15%), 54 Test (15%).
- **Class Distribution:** `passive_aggressive` (51.4%), `non_passive_aggressive` (41.4%), `ambiguous` (7.1%).

| Candidate Architecture | Regularization $C$ | Validation Macro-F1 | Validation Accuracy | Selection Status |
|---|---|---|---|---|
| **TF-IDF + Logistic Regression** | **0.5** | **1.0000** | **1.0000** | **SELECTED (Optimal)** |
| TF-IDF + Logistic Regression | 1.0 | 1.0000 | 1.0000 | Candidate |
| TF-IDF + Logistic Regression | 5.0 | 1.0000 | 1.0000 | Candidate |
| TF-IDF + Linear SVM | 0.5 | 1.0000 | 1.0000 | Candidate |
| TF-IDF + Linear SVM | 1.0 | 1.0000 | 1.0000 | Candidate |
| TF-IDF + Linear SVM | 5.0 | 1.0000 | 1.0000 | Candidate |

### 2.2 Auxiliary Sarcasm Task Benchmark (Grouped Leakage-Free Splits)
- **Data Splits:** 5,000 Train, 1,000 Validation, 1,000 Held-Out Test.
- **Labels:** Verified source sarcasm (`0=non_sarcastic`, `1=sarcastic`).
- **Validation Macro-F1:** `0.7254` (Accuracy: `72.80%`).

---

## 3. Held-Out Test Evaluation Results

### 3.1 Primary Passive-Aggression Benchmark (Held-Out Test: 54 records)
- **Test Macro-F1:** `1.0000`
- **Test Accuracy:** `1.0000`
- **Test Balanced Accuracy:** `1.0000`

#### Per-Class Performance Breakdown:
| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| `passive_aggressive` | 1.000 | 1.000 | 1.000 | 26 |
| `non_passive_aggressive` | 1.000 | 1.000 | 1.000 | 26 |
| `ambiguous` | 1.000 | 1.000 | 1.000 | 2 |

*Note on Synthetic Benchmark:* The high synthetic performance reflects the distinct linguistic patterns embedded in synthetic scenario archetypes. Real social media data will exhibit lower certainty once human annotation is complete.

### 3.2 Auxiliary Sarcasm Benchmark (Held-Out Test: 1,000 records)
- **Test Macro-F1:** `0.7260`
- **Test Accuracy:** `0.7270`
- **Balanced Accuracy:** `0.7262`

---

## 4. Checkpoints and Artifact Inventory

All models, vectorizers, and evaluation metrics are saved to `artifacts/phase3/text_baseline/`:
- `best_text_baseline_primary.pkl`: Serialized pipeline for primary task.
- `best_text_baseline_sarcasm.pkl`: Serialized pipeline for auxiliary sarcasm task.
- `text_baseline_results.json`: Full metric logs, validation curves, and class mappings.
