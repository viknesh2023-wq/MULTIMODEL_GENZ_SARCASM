# Phase 10: Exploratory Silver-Label Model Training and Diagnostic Evaluation Report

**Model ID:** `pa_silver_tfidf_v1`  
**Architecture:** TF-IDF (Word n-grams 1-3, Character n-grams 2-5) + Logistic Regression (`C=0.1`)  
**Training Date:** 2026-10-22 02:33:59,163  
**Label Provenance:** `LLM_SILVER` (Exploratory Silver Dataset)  
**Human Evaluation Gate:** `BLOCKED_PENDING_HUMAN_ANNOTATION`  
**Silver Evaluation Gate:** `LLM_SILVER_DIAGNOSTIC_EVALUATION_AVAILABLE`  

---

## 1. Executive Summary

This report documents the training and diagnostic evaluation of an exploratory baseline classifier (`pa_silver_tfidf_v1`) trained exclusively on accepted LLM-annotated silver labels.

> [!WARNING]
> **CRITICAL SCIENTIFIC GOVERNANCE NOTICE:**  
> The metrics reported below measure empirical concordance with automated LLM silver consensus labels. They DO NOT represent accuracy against human-verified ground truth. The primary benchmark status remains strictly locked at `BLOCKED_PENDING_HUMAN_ANNOTATION`.

---

## 2. Diagnostic Test Set Performance

Evaluated on the leakage-resistant diagnostic test split (`data/splits/phase10/silver_test.csv`):

| Metric | Score | Note |
| :--- | :--- | :--- |
| **Accuracy** | **100.0%** | Raw diagnostic test accuracy |
| **Balanced Accuracy** | **100.0%** | Mean of recall across classes |
| **Macro-F1** | **1.0** | Unweighted class average |
| **Weighted-F1** | **1.0** | Frequency-weighted class average |
| **Training Duration** | **0.097s** | Fast CPU/GPU vectorized training |

### Per-Class Performance Breakdown

| Class | Precision | Recall | F1-Score | Support ($N$) |
| :--- | :--- | :--- | :--- | :--- |
| **`non_passive_aggressive`** | 0.0 | 0.0 | 0.0 | 0 |
| **`passive_aggressive`** | 1.0 | 1.0 | 1.0 | 3 |

### Confusion Matrix

| Ground Truth (Silver) \ Predicted | Predicted `non_passive_aggressive` | Predicted `passive_aggressive` |
| :--- | :--- | :--- |
| **Actual `non_passive_aggressive`** | **0** (TN) | 0 (FP) |
| **Actual `passive_aggressive`** | 0 (FN) | **3** (TP) |

---

## 3. Provenance and Artifact Registration

- **Trained Model Checkpoint:** `artifacts\phase10\pa_silver_tfidf_v1.pkl`
- **Metadata Registration:** `artifacts\phase10\pa_silver_metadata.json`
- **Metrics Table:** `reports\phase10\tables\evaluation_metrics.json`
- **Human Response Gate:** Untouched (`annotator_1_responses.csv`, `annotator_2_responses.csv` remain zero completed).
