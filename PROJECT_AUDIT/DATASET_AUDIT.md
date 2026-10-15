# DATASET AUDIT REPORT: MULTIMODAL SARCASM INTELLIGENCE SYSTEM

**Audit Date:** October 10, 2026  
**Auditor:** Senior ML Engineer & NLP Research Scientist  
**Data Root:** `D:\NLP_PROJECT\FINAL_DATASETS`  

---

## 1. Executive Dataset Summary

All four datasets under `D:\NLP_PROJECT\FINAL_DATASETS` were rigorously inspected row-by-row using Pandas and NumPy verification scripts.

| Dataset Name | Master File Location | Actual Total Rows | Expected Rows | Match? | Total Nulls (Core Cols) | Text Duplicates | Train Rows | Val Rows | Test Rows | Integrity Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Text Sarcasm** | `TEXT_SARCASM\text_sarcasm_combined.csv` | **1,005,167** | 1,005,167 | **YES** | 0 | 0 | 812,130 | 96,518 | 96,519 | **100% VERIFIED** |
| **Multimodal Sarcasm** | `MULTIMODAL_SARCASM\multimodal_sarcasm_combined.csv`| **33,077** | 33,077 | **YES** | 0 | 0 | 25,384 | 3,891 | 3,802 | **100% VERIFIED** |
| **Sentiment** | `SENTIMENT\sentiment_combined.csv` | **132,324** | 132,324 | **YES** | 0 | 0 | 107,678 | 7,916 | 16,730 | **100% VERIFIED** |
| **Emotion** | `EMOTION\emotion_combined.csv` | **54,263** | 54,263 | **YES** | 0 | 270* | 43,410 | 5,426 | 5,427 | **100% VERIFIED** |

*\*Note on Emotion duplicates:* 270 identical text strings exist in GoEmotions because the original Reddit dataset had multiple annotators labeling the same utterance. These do not cause train/test split leakage.

---

## 2. Text Sarcasm Dataset Audit

**File Path:** `D:\NLP_PROJECT\FINAL_DATASETS\TEXT_SARCASM\text_sarcasm_combined.csv`  
**File Size:** 108.59 MB  
**Columns (10):** `id`, `text`, `label`, `language`, `source`, `style`, `topic`, `slang_used`, `abbreviations_used`, `emoji_used`  

### A. Label Distribution
- **Label 0 (Non-Sarcastic):** 499,348 (49.68%)
- **Label 1 (Sarcastic):** 505,819 (50.32%)
- **Class Balance:** Exceptionally well-balanced (near 50:50 ratio).

### B. Language Breakdown
- **English:** 985,168 (98.01%)
- **Tamil-English (Code-Mixed):** 6,680 (0.66%)
- **Kannada-English (Code-Mixed):** 6,660 (0.66%)
- **Malayalam-English (Code-Mixed):** 6,659 (0.66%)
- **Total Code-Mixed Dravidian Samples:** 19,999

### C. Source Breakdown
- **SARC (Self-Annotated Reddit Corpus):** 956,097 (95.12%)
- **Multilingual Synthetic Data:** 19,999 (1.99%)
- **GenZ-SarC (Gen-Z Social Corpus):** 19,984 (1.99%)
- **Sarcasm Corpus V2:** 9,087 (0.90%)

### D. Split Breakdown & Leakage Verification
Split directory: `D:\NLP_PROJECT\FINAL_DATASETS\TEXT_SARCASM\SPLIT\`

| Metric / Attribute | Train Split (`train.csv`) | Validation Split (`validation.csv`) | Test Split (`test.csv`) | Total |
| :--- | :---: | :---: | :---: | :---: |
| **Total Rows** | **812,130** | **96,518** | **96,519** | **1,005,167** |
| **Label 0 (Non-Sarcastic)** | 403,476 (49.68%) | 47,936 (49.66%) | 47,936 (49.66%) | 499,348 |
| **Label 1 (Sarcastic)** | 408,654 (50.32%) | 48,582 (50.34%) | 48,583 (50.34%) | 505,819 |
| **SARC Source** | 764,998 | 95,548 | 95,551 | 956,097 |
| **SarcasmCorpusV2** | 7,149 | 970 | 968 | 9,087 |
| **GenZ-SarC** | 19,984 | 0 | 0 | 19,984 |
| **Multilingual-Synthetic** | 19,999 | 0 | 0 | 19,999 |
| **Tamil-English** | 6,680 | 0 | 0 | 6,680 |
| **Kannada-English** | 6,660 | 0 | 0 | 6,660 |
| **Malayalam-English** | 6,659 | 0 | 0 | 6,659 |
| **Pure English** | 792,131 | 96,518 | 96,519 | 985,168 |

**Crucial Leakage Verification Findings:**
1. **Synthetic Sample Isolation:** All 19,999 synthetic samples and all 19,984 GenZ-SarC samples are 100% confined to `train.csv`. Exactly 0 appear in validation or test splits.
2. **Evaluation Split Purity:** Validation and test sets consist solely of natural, human-authored samples from SARC and SarcasmCorpusV2.
3. **Deduplication:** Text duplicate count across the dataset is 0.

---

## 3. Multimodal Sarcasm Dataset Audit

**File Path:** `D:\NLP_PROJECT\FINAL_DATASETS\MULTIMODAL_SARCASM\multimodal_sarcasm_combined.csv`  
**File Size:** 7.70 MB  
**Columns (14):** `id`, `text`, `image_path`, `label`, `source`, `language`, `original_split`, `original_id`, `text_source`, `sarcastic_original`, `humour`, `offensive`, `motivational`, `overall`  

### A. Label Distribution
- **Label 0 (Non-Sarcastic):** 14,568 (44.04%)
- **Label 1 (Sarcastic):** 18,509 (55.96%)
- **Total Rows:** 33,077

### B. Source Breakdown & Multilingual Context
- **MMSD2.0:** 24,635 (74.48%) — English Twitter multimodal sarcasm dataset with OCR/clean text.
- **MEMOTION 3.0:** 8,442 usable examples (25.52%) — Meme dataset containing English and Hinglish (Hindi-English code-mixed) multimodal samples.
- **Important Note:** This dataset is **not** exclusively English; MEMOTION 3.0 contributes rich South Asian code-mixed cultural context.

### C. Image Physical Existence Verification
- Every single image path in `image_path` was tested with `os.path.exists()`.
- **Existing Images on Disk:** **33,077 / 33,077 (100.0%)**.
- **Missing Images:** **0**.
- **Corrupt / Invalid Paths:** **0**.

### D. Split Breakdown
Split directory: `D:\NLP_PROJECT\FINAL_DATASETS\MULTIMODAL_SARCASM\SPLIT\`

| Split | Total Rows | Label 0 | Label 1 | MMSD2.0 Count | MEMOTION 3.0 Count | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Train (`train.csv`)** | 25,384 | 11,414 | 13,970 | 19,816 | 5,568 | Verified |
| **Validation (`validation.csv`)** | 3,891 | 1,488 | 2,403 | 2,410 | 1,481 | Verified |
| **Test (`test.csv`)** | 3,802 | 1,666 | 2,136 | 2,409 | 1,393 | Verified |
| **TOTAL** | **33,077** | **14,568** | **18,509** | **24,635** | **8,442** | **100% Match** |

---

## 4. Sentiment Dataset Audit

**File Path:** `D:\NLP_PROJECT\FINAL_DATASETS\SENTIMENT\sentiment_combined.csv`  
**File Size:** 22.33 MB  
**Columns (8):** `id`, `text`, `label`, `sentiment`, `language`, `source`, `original_split`, `synthetic`  

### A. Label Distribution
- **Label 2 (Positive):** 61,810 (46.71%)
- **Label 1 (Neutral):** 45,405 (34.31%)
- **Label 0 (Negative):** 25,109 (18.98%)
- **Total Rows:** 132,324

### B. Source Distribution
- **TweetEval Sentiment:** 59,865 (45.24%)
- **DravidianCodeMix:** 44,688 (33.77%)
- **SentiMix:** 15,771 (11.92%)
- **DravidianCodeMix-Synthetic (Neutral):** 12,000 (9.07%)

### C. Split Breakdown & Synthetic Isolation

| Split | Total Rows | Negative (0) | Neutral (1) | Positive (2) | Synthetic Count | TweetEval | Dravidian | SentiMix |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Train** | **107,678** | 18,555 | 37,994 | 51,129 | **12,000 (100%)** | 45,585 | 47,861 | 14,232 |
| **Validation** | **7,916** | 1,636 | 1,477 | 4,803 | **0 (0%)** | 2,000 | 4,377 | 1,539 |
| **Test** | **16,730** | 4,918 | 5,934 | 5,878 | **0 (0%)** | 12,280 | 4,450 | 0 |
| **TOTAL** | **132,324** | **25,109** | **45,405** | **61,810** | **12,000** | **59,865** | **56,688** | **15,771** |

**Crucial Findings:**
1. All 12,000 synthetic neutral examples are restricted entirely to `train.csv`.
2. No silent conversions of mixed-feelings or unknown classes into neutral were performed.
3. Dataset preparation is complete, but **no sentiment model has been trained yet**.

---

## 5. Emotion Dataset Audit

**File Path:** `D:\NLP_PROJECT\FINAL_DATASETS\EMOTION\emotion_combined.csv`  
**File Size:** 6.56 MB  
**Columns (8):** `id`, `text`, `emotion_labels`, `emotion_ids`, `num_emotions`, `language`, `source`, `original_split`  
**Label Mapping File:** `D:\NLP_PROJECT\FINAL_DATASETS\EMOTION\emotion_label_map.csv` (28 classes verified: 0: admiration to 27: neutral)  

### A. Emotion Multi-Label Distribution
- **Total Examples:** 54,263
- **Single Emotion ($N = 1$):** 45,446 (83.75%)
- **Two Emotions ($N = 2$):** 8,124 (14.97%)
- **Three Emotions ($N = 3$):** 655 (1.21%)
- **Four Emotions ($N = 4$):** 37 (0.07%)
- **Five Emotions ($N = 5$):** 1 (< 0.01%)
- **Total Examples with Multiple Emotions ($N > 1$):** **8,817 (16.25%)** — exactly matching expected count!

### B. Split Breakdown

| Split | Rows | Single Emotion | Multiple Emotions | Emotion Label Columns Present |
| :--- | :---: | :---: | :---: | :---: |
| **Train (`train.csv`)** | 43,410 | 36,368 | 7,042 | `emotion_labels`, `emotion_ids`, `num_emotions` |
| **Validation (`validation.csv`)** | 5,426 | 4,539 | 887 | `emotion_labels`, `emotion_ids`, `num_emotions` |
| **Test (`test.csv`)** | 5,427 | 4,539 | 888 | `emotion_labels`, `emotion_ids`, `num_emotions` |
| **TOTAL** | **54,263** | **45,446** | **8,817** | **100% Consistent** |

**Crucial Findings:**
1. All 54,263 examples have valid multi-label mappings.
2. Dataset preparation is complete, but **no emotion model has been trained yet**.
