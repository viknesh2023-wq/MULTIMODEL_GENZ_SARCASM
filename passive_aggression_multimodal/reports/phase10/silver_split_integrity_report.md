# Phase 10: Silver Split Integrity and Leakage Prevention Report

**Date:** 2026-10-22 02:33:42,865  
**Splitting Strategy:** Group-Isolated Stratified Random Split  
**Random Seed:** `42`  
**Group Overlap Leakage:** `ZERO_DETECTED`  

---

## 1. Split Partition Summary

| Partition | Total Records | Unique Groups | `passive_aggressive` | `non_passive_aggressive` | File Path |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Train** | 10 | 10 | 8 | 2 | `data/splits/phase10/silver_train.csv` |
| **Validation** | 2 | 2 | 1 | 1 | `data/splits/phase10/silver_validation.csv` |
| **Diagnostic Test** | 3 | 3 | 3 | 0 | `data/splits/phase10/silver_test.csv` |
| **Total** | 15 | 15 | 12 | 3 | - |

---

## 2. Leakage Prevention Verification

- **Group Hash Isolation:** Verified that no duplicate group or shared conversation thread crosses split boundaries.
- **Diagnostic Test Set Quarantine:** The diagnostic test partition is reserved exclusively for final evaluation and was not accessed during acceptance threshold tuning.
- **Provenance Transparency:** Every record carries `provenance: LLM_SILVER`. Evaluation on these splits measures silver-label concordance, NOT ground truth.
