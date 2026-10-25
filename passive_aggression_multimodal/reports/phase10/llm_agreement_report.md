# Phase 10: LLM Inter-Model Agreement and Quality Audit Report

**Date:** 2026-10-22 02:29:08,343  
**Annotator A:** `llama3.2:latest`  
**Annotator B:** `qwen2.5:1.5b`  
**Evaluation Gate:** `BLOCKED_PENDING_HUMAN_ANNOTATION`  
**Silver Gate:** `LLM_SILVER_DATASET_AVAILABLE_FOR_EXPLORATORY_RESEARCH`  

---

## 1. Executive Summary

This report evaluates inter-model agreement and reliability between two architecturally distinct open-weights LLMs: Meta's **Llama 3.2 (3.2B)** and Alibaba's **Qwen 2.5 (1.5B)**. Both models independently evaluated **30** paired eligible records using a strict, neutral operational definition of passive aggression (3-prong test) without seeing each other's outputs.

- **Total Paired Records:** 30
- **Raw Inter-Model Agreement:** 12 / 30 (40.0%)
- **Disagreements:** 18 (60.0%)
- **Cohen's Kappa ($\kappa$):** `0.0128` (Chance expected agreement $P_e = 0.3922$)
- **Subtype Concordance:** 30.0%
- **Modality Honesty:** 100% text-only assessment (`image_assessed: false`). No unverified visual claims.

---

## 2. Statistical Reliability & Cohen's Kappa Interpretation

$$\kappa = \frac{P_o - P_e}{1 - P_e} = \frac{0.4 - 0.3922}{1 - 0.3922} = 0.0128$$

| Metric | Value | Interpretation |
| :--- | :--- | :--- |
| **Total Paired ($N$)** | 30 | Full eligible corpus paired sample |
| **Observed Agreement ($P_o$)** | 40.0% | Overall raw label concordance |
| **Expected Agreement ($P_e$)** | 39.22% | Concordance expected purely by chance marginals |
| **Cohen's $\kappa$** | **0.0128** | Fair / Slight agreement |

*Note: In accordance with Section 3.1 & 3.2, cross-model agreement is NOT proof of ground-truth correctness. It establishes automated reliability and consistency across independent architectures.*

---

## 3. Label Distributions and Joint Confusion Matrix

### Label Distribution

| Label | Annotator A (llama3.2:latest) | % | Annotator B (qwen2.5:1.5b) | % |
| :--- | :--- | :--- | :--- | :--- |
| `passive_aggressive` | 12 | 40.0% | 29 | 96.7% |
| `non_passive_aggressive` | 5 | 16.7% | 1 | 3.3% |
| `ambiguous` | 13 | 43.3% | 0 | 0.0% |

### Confusion Matrix (Rows = Annotator A, Columns = Annotator B)

| Annotator A \ B | `passive_aggressive` | `non_passive_aggressive` | `ambiguous` | Total A |
| :--- | :--- | :--- | :--- | :--- |
| **`passive_aggressive`** | **12** | 0 | 0 | 12 |
| **`non_passive_aggressive`** | 5 | **0** | 0 | 5 |
| **`ambiguous`** | 12 | 1 | **0** | 13 |
| **Total B** | 29 | 1 | 0 | 30 |

---

## 4. Three-Prong Criteria Concordance

The operational definition requires satisfying three distinct semantic prongs:

| Operational Prong | Agreed Count | Concordance Rate |
| :--- | :--- | :--- |
| **Interpersonal Target Present** | 21 / 30 | 70.0% |
| **Plausible Deniability Present** | 22 / 30 | 73.3% |
| **Covert Hostility Present** | 16 / 30 | 53.3% |

---

## 5. Disagreement Taxonomy Breakdown

Total disagreements: **18** (60.0% of corpus).

| Disagreement Category | Count | Pct of Disagreements | Description |
| :--- | :--- | :--- | :--- |
| `ambiguous boundary case` | 13 | 72.2% | Primary boundary conflict between surface polite disguise and directness |
| `direct aggression` | 5 | 27.8% | Primary boundary conflict between surface polite disguise and directness |

---

## 6. Source Dataset Breakdown

| Source Dataset | Total Records | Agreed | Agreement % | Disagreed | Full PA Concordance |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `civil_comments` | 10 | 4 | 40.0% | 6 | 4 |
| `mmsd` | 10 | 3 | 30.0% | 7 | 3 |
| `mustard` | 8 | 4 | 50.0% | 4 | 4 |
| `sarcnet` | 2 | 1 | 50.0% | 1 | 1 |

---

## 7. Next Step in Pipeline

The 18 disagreement records will be forwarded to Step 9: Third-Model Automated Adjudication (`scripts/phase10/run_llm_adjudication.py`), adhering to strict judicial guidelines without overwriting raw annotator outputs.
