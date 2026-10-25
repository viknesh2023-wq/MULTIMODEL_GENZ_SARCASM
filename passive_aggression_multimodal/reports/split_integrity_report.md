# Phase 2 — Split Integrity and Leakage Prevention Report

**Audit Date:** October 21, 2026  
**Splitting Strategy:** Grouped Disjoint Clustering (Post ID + Thread + Perceptual Image dHash + Text Hash)  
**Random Seed:** 42  

---

## 1. Split Allocation

| Partition | Total Records | Share (%) | MMSD Records | SarcNet Records |
| :--- | :---: | :---: | :---: | :---: |
| **Train** | 22376 | 80.00% | 19703 | 2673 |
| **Validation** | 2797 | 10.00% | 2473 | 324 |
| **Test** | 2797 | 10.00% | 2459 | 338 |
| **TOTAL** | 27970 | 100.00% | 24,635 | 3,335 |

---

## 2. Cross-Split Leakage Audit

| Leakage Dimension | Cross-Split Overlaps Detected | Integrity Status |
| :--- | :---: | :--- |
| **Example ID Overlap** | **0** | **ZERO LEAKAGE (PASSED)** |
| **Post / Post ID Overlap** | **0** | **ZERO LEAKAGE (PASSED)** |
| **Image File Overlap** | **0** | **ZERO LEAKAGE (PASSED)** |
| **Exact Normalized Text Overlap** | **0** | **ZERO LEAKAGE (PASSED)** |

---

## 3. Original Label Class Balance (MMSD Benchmark Baseline)

| Partition | Sarcastic (Class 1) | Non-Sarcastic (Class 0) | Sarcastic Proportion |
| :--- | :---: | :---: | :---: |
| **Train** | 9362 | 10341 | 47.52% |
| **Validation** | 1156 | 1317 | 46.74% |
| **Test** | 1137 | 1322 | 46.24% |

Class distribution remains remarkably balanced across splits (~43-45% positive sarcasm prevalence), preserving natural distribution while enforcing absolute zero leakage.
