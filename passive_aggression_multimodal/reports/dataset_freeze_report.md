# Phase 1 Dataset Freeze Report

**Project Title:** Context-Aware Multimodal Passive-Aggression Detection in Social Media Using Image–Comment Fusion  
**Freeze Manifest File:** `data/manifests/dataset_freeze_manifest.json`  
**Freeze Version:** `1.0.0-phase1-freeze`  
**Freeze Timestamp:** October 21, 2026 (UTC)  
**Auditor:** Senior AI Research Engineer & Data Auditor  

---

## 1. Freeze Purpose and Immutability Guarantee

This freeze establishes an immutable baseline for all datasets collected and verified in Phase 1. 
- All files in `data/raw/` are designated as **read-only source truth** and must not be overwritten or modified by future preprocessing, tokenization, or cleaning operations.
- Derived manifests (`data/manifests/`) and interim records (`data/interim/`) link directly to these frozen sources via cryptographic SHA-256 hashes.

---

## 2. Frozen Dataset Inventory

```
+---------------------------------------------------------------------------------------------------+
| Dataset          | Status                   | Records | Verified Images | Verified Size on Disk   |
+---------------------------------------------------------------------------------------------------+
| MMSD / MMSD2.0   | DOWNLOADED_AND_VERIFIED  | 24,635  | 24,635          | 2.47 GB img + 3.03 MB   |
| MMSD3.0 (OS)     | DOWNLOADED_AND_VERIFIED  | 10,833  | N/A (Amazon)    | 5.59 MB json            |
| SarcNet          | DOWNLOADED_AND_VERIFIED  |  3,335  |  3,335          | 377.6 MB zip + 577 KB   |
| GoEmotions       | DOWNLOADED_AND_VERIFIED  | 54,263  | N/A (Text-Only) | 4.39 MB tsv             |
| Civil Comments   | DOWNLOADED_AND_VERIFIED  | 97,320  | N/A (Text-Only) | 20.95 MB parquet        |
| MUStARD          | DOWNLOADED_AND_VERIFIED  |    690  | N/A (Dialogue)  | 374 KB json             |
| iSarcasm         | PARTIALLY_AVAILABLE      |  4,486  | N/A (Unhydrated)| 158 KB csv              |
| Synthetic Pilot  | DOWNLOADED_AND_VERIFIED  |    350  | N/A (Text-Ref)  | 320 KB jsonl            |
+---------------------------------------------------------------------------------------------------+
| TOTAL AUDITED    |                          | 195,912 | 27,970          | ~2.89 GB Total Media    |
+---------------------------------------------------------------------------------------------------+
```

---

## 3. Cryptographic Verification Table (Selected Core Files)

| File Path | Size (Bytes) | SHA-256 Checksum (First 16 chars) | Status |
| :--- | :--- | :--- | :--- |
| `data/raw/mmsd/mmsd_train.json` | 2,442,324 | `1217e4f3a7434526...` | Verified |
| `data/raw/mmsd/mmsd_valid.json` | 298,901 | `4f32a514d3a04297...` | Verified |
| `data/raw/mmsd/mmsd_test.json` | 297,800 | `ca91a789ef7ba60b...` | Verified |
| `data/raw/sarcnet/sarcnet_bundle.zip` | 377,592,086 | `869c9b139366ec96...` | Verified |
| `data/raw/sarcnet/SarcNetTrain.csv` | 344,548 | `47190013dbf4bfbe...` | Verified |
| `data/raw/goemotions/train.tsv` | 3,519,053 | `5ba52042a5f05eb5...` | Verified |
| `data/raw/civil_comments/civil_comments_val.parquet` | 20,955,569 | `f73dc570a2f58e1d...` | Verified |
| `data/raw/mustard/sarcasm_data.json` | 374,629 | `32e269436440adbb...` | Verified |
| `data/synthetic/synthetic_pilot_350.jsonl` | 320,412 | `c40788771aa7789d...` | Verified |

---

## 4. Known Blockers and Manual Actions Required

### A. iSarcasm Dataset Hydration (`PARTIALLY_AVAILABLE`)
- **Limitation:** The SemEval-2022 Task 6 dataset distribution provides tweet IDs and sarcasm labels. Tweet text is unhydrated per Twitter's distribution policy.
- **Manual Action Required from User:**
  1. If text-only iSarcasm data is desired for Phase 2 auxiliary sarcasm comparison, obtain an X/Twitter Developer API bearer token.
  2. Run the hydration script or utilize the pre-existing cached replacements under `D:\NLP_PROJECT\isarcasm_replacement` if authorized.
  3. If unhydrated, iSarcasm will remain an auxiliary reference and not impede multimodal training.

### B. MMSD3.0 X-Platform Subset (`AWAITING_MANUAL_ACTION`)
- **Limitation:** While MMSD 1.0, MMSD 2.0, and the Amazon Reviews 2023 portion of MMSD 3.0 are fully downloaded and verified with 24,635 images, the X-platform portion of MMSD 3.0 requires an access request form.
- **Manual Action Required from User:**
  - Submit the official access request form at [MMSD3 Access Form](https://forms.office.com/r/g3TZBSssqt) if multi-image Twitter posts are required for Phase 2 experiments.
