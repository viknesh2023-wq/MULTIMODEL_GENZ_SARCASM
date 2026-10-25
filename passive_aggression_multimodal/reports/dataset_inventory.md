# Phase 1 Dataset Inventory Report

**Project Title:** Context-Aware Multimodal Passive-Aggression Detection in Social Media Using Image–Comment Fusion  
**Date of Audit:** October 21, 2026  
**Auditor / Engineer:** Senior AI Research Engineer & Data Auditor  
**Freeze Version:** `1.0.0-phase1-freeze`  
**Execution Environment:** Windows NT (x86_64), Python 3.12.10, PyTorch 2.14.0+cpu  

---

## 1. Executive Summary

During Phase 1, all priority multimodal and auxiliary research datasets were audited, retrieved via official and authorized distribution channels, cryptographically hashed, validated for structural integrity, and indexed into a unified schema representation.

| Dataset Name | Modality | Official Source URL | Verified Status | Usable Records | Images on Disk | File Size on Disk |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **MMSD / MMSD2.0** | Image + Tweet Text | [ZHCMOONWIND/MMSD3.0](https://github.com/ZHCMOONWIND/MMSD3.0) | `DOWNLOADED_AND_VERIFIED` | 24,635 | 24,635 | 2.47 GB (images) + 3.03 MB (JSONs) |
| **MMSD3.0 (Open Source)** | Multi-Image + Social Text | [ZHCMOONWIND/MMSD3.0](https://github.com/ZHCMOONWIND/MMSD3.0) | `DOWNLOADED_AND_VERIFIED` | 10,833 | N/A (Amazon subset links) | 5.59 MB (JSONs) |
| **SarcNet** | Image + Multilingual Text | [yuetanbupt/SarcNet](https://github.com/yuetanbupt/SarcNet) | `DOWNLOADED_AND_VERIFIED` | 3,335 | 3,335 | 360.10 MB (zip) + 577 KB (CSVs) |
| **GoEmotions** | Text-Only (Reddit Comments) | [google-research/goemotions](https://github.com/google-research/google-research/tree/master/goemotions) | `DOWNLOADED_AND_VERIFIED` | 54,263 | N/A (Text-Only) | 4.39 MB (TSVs) |
| **Civil Comments** | Text-Only (News Comment Threads) | [google/civil_comments](https://huggingface.co/datasets/google/civil_comments) | `DOWNLOADED_AND_VERIFIED` | 97,320 | N/A (Text-Only) | 19.98 MB (Parquet) |
| **MUStARD** | Video/Audio/Dialogue Context | [soujanyaporia/MUStARD](https://github.com/soujanyaporia/MUStARD) | `DOWNLOADED_AND_VERIFIED` | 690 | N/A (TV Sitcom Dialogue) | 374 KB (JSON) |
| **iSarcasm / iSarcasmEval** | Text-Only (Twitter IDs + Labels) | [headacheboy/data-iSarcasmEval](https://github.com/headacheboy/data-iSarcasmEval) | `PARTIALLY_AVAILABLE` | 4,486 (IDs) | N/A (Text Unhydrated) | 158 KB (CSVs) |
| **Synthetic Pilot V1** | Structured Scenarios | Phase 1 Controlled Generation | `DOWNLOADED_AND_VERIFIED` | 350 | Textual References | 320 KB (JSONL) |

---

## 2. Dataset-by-Dataset Audit Details

### A. MMSD / MMSD2.0 / MMSD3.0
- **Authors & Citations:** Cai et al., ACL 2019; Ying et al., 2023; ZHCMOONWIND et al., CVPR 2026.
- **Physical Location:** `data/raw/mmsd/` and `data/images/mmsd/`.
- **Files Verified:**
  - `mmsd_train.json` (19,816 records, 2,442,324 bytes, SHA-256 verified)
  - `mmsd_valid.json` (2,410 records, 298,901 bytes, SHA-256 verified)
  - `mmsd_test.json` (2,409 records, 297,800 bytes, SHA-256 verified)
  - `mmsd3_train_opensource.json` (7,583 records, 3,911,887 bytes)
  - `mmsd3_val_opensource.json` (1,626 records, 832,339 bytes)
  - `mmsd3_test_opensource.json` (1,624 records, 854,241 bytes)
- **Image Assets:** 24,635 images verified on disk via PIL decoder with 0 corrupted files.
- **Original Schema:** `{"image_id": int, "text": str, "label": int}` (0=non-sarcastic, 1=sarcastic).
- **PA Label Separation:** Preserved intact. Original sarcasm labels are NOT converted to passive-aggression labels.

### B. SarcNet
- **Authors & Citations:** Tan Yue, Xuzhao Shi, Rui Mao, Zonghai Hu, Erik Cambria (LREC-COLING 2024).
- **Physical Location:** `data/raw/sarcnet/` and `data/images/sarcnet/`.
- **Files Verified:**
  - `sarcnet_bundle.zip` (377,592,086 bytes, SHA-256 verified)
  - `SarcNetTrain.csv` (1,998 records, 344,548 bytes, GB18030 encoded)
  - `SarcNetVal.csv` (671 records, 112,657 bytes, GB18030 encoded)
  - `SarcNetTest.csv` (666 records, 120,048 bytes, GB18030 encoded)
- **Image Assets:** 3,335 image files extracted to `data/images/sarcnet/` and verified with PIL (0 corrupt).
- **Original Schema:** `Text, Imagepath, Text_label, Image_label, Multi_label`.
- **Significance for PA Research:** Provides isolated unimodal vs multimodal discrepancy labels, demonstrating cases where image and text diverge in intent.

### C. GoEmotions
- **Authors & Citations:** Demszky et al. (ACL 2020), Google Research.
- **Physical Location:** `data/raw/goemotions/`.
- **Files Verified:**
  - `train.tsv` (43,410 records, 3,519,053 bytes, SHA-256 matches upstream Google Research)
  - `dev.tsv` (5,426 records, 439,059 bytes)
  - `test.tsv` (5,427 records, 436,706 bytes)
  - `emotions.txt` (28 categories: 27 emotions + neutral)
- **Relevant Subtypes Profiled:**
  - Annoyance: 2,746 instances
  - Disapproval: 2,427 instances
  - Disappointment: 1,495 instances
  - Anger: 1,811 instances
- **Policy Note:** Emotion labels provide negative sentiment / annoyance signals for contrast testing, but are explicitly kept distinct from interpersonal passive aggression.

### D. Civil Comments
- **Authors & Citations:** Borkan et al. (WWW 2019), Jigsaw / Google.
- **Physical Location:** `data/raw/civil_comments/`.
- **Files Verified:** `civil_comments_val.parquet` (97,320 records, 20,955,569 bytes, CC0-1.0 Public Domain).
- **Feature Breakdown:** `text`, `toxicity`, `severe_toxicity`, `obscene`, `threat`, `insult`, `identity_attack`, `sexual_explicit`.
- **Toxicity Distribution (Threshold >= 0.5):**
  - Toxic: 5,617 (5.77%)
  - Non-Toxic: 91,703 (94.23%)
  - Insult: 3,923 (4.03%)
- **Policy Note:** Serves as a contrast pool for direct toxicity and non-hostile disagreement; toxicity is never equated to passive aggression.

### E. MUStARD
- **Authors & Citations:** Castro et al. (ACL 2019).
- **Physical Location:** `data/raw/mustard/`.
- **Files Verified:** `sarcasm_data.json` (690 dialogue instances, 374,629 bytes, MIT License).
- **Unique Feature:** Multi-turn conversational dialogue context with speaker identities across 4 shows (*The Big Bang Theory*, *Friends*, *The Golden Girls*, *Sarcasmaholics Anonymous*).

### F. iSarcasm / iSarcasmEval
- **Authors & Citations:** Abu Farha et al. (SemEval 2022 Task 6).
- **Physical Location:** `data/raw/isarcasm/`.
- **Files Present:** `isarcasm_train.csv` (3,578 tweet IDs/labels), `isarcasm_test.csv` (908 tweet IDs/labels).
- **Audit Finding:** The distribution contains tweet IDs and sarcasm labels without tweet text (due to historical Twitter redistribution policies). Marked as `PARTIALLY_AVAILABLE`.

---

## 3. Storage and Directory Footprint

- **Raw Data Directory (`data/raw/`):** ~405 MB
- **Image Directory (`data/images/`):** 24,635 images via NTFS junction (0 extra bytes) + 360 MB SarcNet images
- **Interim / Manifest Directory (`data/manifests/`, `data/interim/`):** ~35 MB
- **Synthetic Directory (`data/synthetic/`):** ~320 KB
- **Free Disk Space Remaining:** ~68.4 GB on drive `D:\`.
