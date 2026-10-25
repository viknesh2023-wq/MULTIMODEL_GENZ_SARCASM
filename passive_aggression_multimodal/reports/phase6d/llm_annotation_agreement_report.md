# Phase 6D: LLM Dual-Annotation Inter-Model Agreement Report

**Project Title:** Context-Aware Multimodal Passive-Aggression Detection in Social Media Using Image–Comment Fusion  
**Date:** October 21, 2026  
**Status:** COMPLETE (Machine-Assisted Annotation and Adjudication)  
**Models Evaluated:**
- **Model 1:** `llama3.2:latest` (Prompt `v2.0.0`)
- **Model 2:** `qwen2.5:1.5b` (Prompt `v2.0.0`)
- **Adjudicator:** `llama3.2:latest` (Prompt `v2.0.0_adjudication`)

---

## 1. Executive Summary & Epistemic Warning

> [!WARNING]
> **MACHINE-GENERATED DATASET NOTICE:**  
> These metrics represent **inter-model agreement** between two local large language models (`llama3.2:latest` 3.2B and `qwen2.5:1.5b` 1.5B).  
> **Never represent these metrics as human inter-annotator agreement or ground truth.**  
> High agreement indicates shared model heuristics or convergent prompt compliance, **not empirical fact**. The human annotation benchmark gate remains strictly `BLOCKED_PENDING_HUMAN_ANNOTATION`.

---

## 2. Quantitative Inter-Model Agreement

| Metric Scope | Total Paired Records | Exact Matches | Disagreements | Raw Agreement (%) | Cohen's Kappa (κ) | Status / Interpretation |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **All Pilot Records** | 30 | 16 | 14 | **53.33%** | **0.1322** | Full pilot set |
| **Core English Benchmark** | 27 | 14 | 13 | **51.85%** | **0.1397** | Rigorous English subset |
| **Chinese Exploratory Subset** | 3 | 2 | 1 | **66.67%** | N/A (N=3) | Segregated exploratory analysis |

---

## 3. Confusion Matrix Analysis

### Overall Confusion Matrix (Rows: Model 1 `llama3.2:latest`, Columns: Model 2 `qwen2.5:1.5b`)

| Model 1 \ Model 2 | passive_aggressive | non_passive_aggressive | ambiguous | Total |
| :--- | :---: | :---: | :---: | :---: |
| **passive_aggressive** | 0 | 0 | 0 | 0 |
| **non_passive_aggressive** | 1 | 12 | 1 | 14 |
| **ambiguous** | 0 | 12 | 4 | 16 |
| **Total** | 1 | 24 | 5 | 30 |

### Core English Benchmark Confusion Matrix (27 Records)

| Model 1 \ Model 2 | passive_aggressive | non_passive_aggressive | ambiguous | Total |
| :--- | :---: | :---: | :---: | :---: |
| **passive_aggressive** | 0 | 0 | 0 | 0 |
| **non_passive_aggressive** | 1 | 11 | 0 | 12 |
| **ambiguous** | 0 | 12 | 3 | 15 |

---

## 4. Subtype Agreement Analysis

For records where both models independently agreed on `passive_aggressive`:
- **Both Models Agreed PA:** 0 records
- **Exact Subtype Concordance:** 0 records
- **Subtype Concordance Rate:** **0.0%**

---

## 5. Disagreement Audit & LLM Adjudication Log

Total disagreements requiring adjudication: **14**

| Record ID | Source | Scope | Model 1 (`llama3.2:latest`) | Model 2 (`qwen2.5:1.5b`) | Adjudicated Label | Root Cause Summary |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `mustard_2_490` | mustard | CORE_ENGLISH_BENCHMARK | `ambiguous` | `non_passive_aggressive` | **`ambiguous`** | Linguistic interpretation of tone and intent... |
| `mmsd_train_10780` | mmsd | CORE_ENGLISH_BENCHMARK | `ambiguous` | `non_passive_aggressive` | **`ambiguous`** | Linguistic interpretation of sarcasm/sarcasm detection... |
| `civil_comments_val_08920` | civil_comments | CORE_ENGLISH_BENCHMARK | `ambiguous` | `non_passive_aggressive` | **`non_passive_aggressive`** | Linguistic root cause: Insufficient context and tone analysi... |
| `mmsd_test_00104` | mmsd | CORE_ENGLISH_BENCHMARK | `ambiguous` | `non_passive_aggressive` | **`ambiguous`** | Lack of contextual information and unclear intent... |
| `civil_comments_val_04917` | civil_comments | CORE_ENGLISH_BENCHMARK | `non_passive_aggressive` | `passive_aggressive` | **`ambiguous`** | Linguistic root cause: Model 1 misinterprets the sarcastic t... |
| `civil_comments_val_02587` | civil_comments | CORE_ENGLISH_BENCHMARK | `ambiguous` | `non_passive_aggressive` | **`non_passive_aggressive`** | Linguistic root cause: Model 1 misinterprets the tone of the... |
| `mmsd_train_02546` | mmsd | CORE_ENGLISH_BENCHMARK | `ambiguous` | `non_passive_aggressive` | **`ambiguous`** | Linguistic ambiguity in interpretation of the phrase 'not al... |
| `sarcnet_test_00274` | sarcnet | CORE_ENGLISH_BENCHMARK | `ambiguous` | `non_passive_aggressive` | **`ambiguous`** | Linguistic ambiguity in pronoun interpretation... |
| `civil_comments_val_09401` | civil_comments | CORE_ENGLISH_BENCHMARK | `ambiguous` | `non_passive_aggressive` | **`non_passive_aggressive`** | Linguistic root cause: Model 1 misinterprets the comment's i... |
| `mmsd_train_06126` | mmsd | CORE_ENGLISH_BENCHMARK | `ambiguous` | `non_passive_aggressive` | **`ambiguous`** | Linguistic root cause: Insufficient contextual information a... |
| `mustard_2_111` | mustard | CORE_ENGLISH_BENCHMARK | `ambiguous` | `non_passive_aggressive` | **`ambiguous`** | Linguistic root cause: Insufficient contextual information t... |
| `mmsd_validation_02127` | mmsd | CORE_ENGLISH_BENCHMARK | `ambiguous` | `non_passive_aggressive` | **`ambiguous`** | Linguistic root cause: unclear target and lack of context... |
| `mmsd_validation_01859` | mmsd | CORE_ENGLISH_BENCHMARK | `ambiguous` | `non_passive_aggressive` | **`ambiguous`** | Linguistic ambiguity in tone and intent... |
| `sarcnet_train_00844` | sarcnet | NON_ENGLISH_SEPARATE_ANALYSIS | `non_passive_aggressive` | `ambiguous` | **`non_passive_aggressive`** | Linguistic root cause: Model 1 misinterprets the tone as pla... |

---

## 6. Modality & Verification Restrictions

1. **Modality Coverage:** `TEXT_ONLY` for all 30 records. Zero images were evaluated (`image_assessed = false`, `ocr_assessed = false`).
2. **Double-Blind Isolation:** Model 2 received zero predictions, labels, or rationales from Model 1.
3. **Preservation of Human Files:** `data/annotations/phase6a/annotator_1_responses.csv` and `annotator_2_responses.csv` remain untouched with completion count = 0.
