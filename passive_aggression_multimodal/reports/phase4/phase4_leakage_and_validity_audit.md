# Phase 4 Synthetic Benchmark Validity and Data Leakage Audit
**Project Title:** Context-Aware Multimodal Passive-Aggression Detection in Social Media Using Image–Comment Fusion  
**Auditor:** Senior ML Research Engineer & Experimental Auditor  
**Artifact:** `reports/phase4/phase4_leakage_and_validity_audit.md`  
**Date:** October 21, 2026  

---

## 1. Executive Summary: Root Cause of Phase 3 "100% Accuracy"

In Phase 3, four separate model architectures (TF-IDF + Logistic Regression, TF-IDF + Linear SVM, Pretrained Transformer, and Gated Multimodal Fusion) reported **100% test accuracy (Macro-F1: 1.0000)** on the 54-record held-out test split of the 350-scenario synthetic benchmark.

This independent audit conducted a forensic text and template analysis on `data/synthetic/synthetic_pilot_350.jsonl` and the Phase 3 splitting logic. 

### 🚨 Critical Audit Finding: Severe Train-Test Data Contamination
1. **Comment Redundancy in Source Data:** While `synthetic_pilot_350.jsonl` contains 350 records, it contains only **180 unique target comments**. During Phase 1 scenario generation, each scenario template was instantiated with 2 to 3 variants (e.g., text-only variant, multimodal variant, contrast variant) that reused identical or near-identical comment text.
2. **Naive Permutation Splitting:** In Phase 3, the dataset was split using an un-grouped random permutation:
   ```python
   np.random.seed(42)
   indices = np.random.permutation(350)
   # train: indices[:244], val: indices[244:296], test: indices[296:]
   ```
3. **Massive Cross-Split Leakage:**
   - **35 exact comments overlapped between the training set and the test set.**
   - **36 out of 54 test records (66.7%) had identical duplicate comments in the training set.**
   - **36 exact comments overlapped between train and validation.**
   - **12 scenario descriptions directly crossed the split boundary.**
4. **Impact on Evaluation Metrics:** The models did not learn generalized passive-aggression reasoning; they simply memorized exact lexical strings and distinctive template markers (*"Per my previous email"*, *"Bless your heart"*, *"With all due respect"*, *"Must be nice"*) that were present verbatim in both training and test sets.

---

## 2. Quantitative Evidence of Leakage

```text
Total Synthetic Records:                    350
Unique Target Comments:                     180
Comments with Duplicate Instances:          120 (reused 2–3 times)

Phase 3 Random Split (Seed 42):
  Training Instances:                       244 (70%)
  Validation Instances:                     52 (15%)
  Test Instances:                           54 (15%)

Cross-Split Contamination:
  Exact Comment Overlap (Train ∩ Test):     35 comments
  Test Records with Identical Train Twin:   36 / 54 (66.7% LEAKED)
  Exact Comment Overlap (Train ∩ Val):      36 comments
  Val Records with Identical Train Twin:    36 / 52 (69.2% LEAKED)
  Scenario Template Overlap (Train ∩ Test): 12 scenario archetypes
```

### Sample Duplicate Comments Crossing Train and Test:
- *"Per my previous email, following basic instructions usually yields an even crumb..."* (Appeared in Train and Test)
- *"Warmest regards to whoever felt inspired to leave all their pots for the cleaning staff..."* (Appeared in Train and Test)
- *"With all due respect, I am sure you tried your absolute hardest with those calculations..."* (Appeared in Train and Test)
- *"Bless your heart for having the bravery to display this in public view!!..."* (Appeared in Train and Test)
- *"So wonderful that some team members have the luxury of extended lunch hours..."* (Appeared in Train and Test)

---

## 3. Multimodal Modality Mask Audit in Synthetic Features

An inspection of `data/features/phase3/synthetic_pilot_350.pt` revealed an additional validity concern:
- **`image_masks.mean()` = 0.000 (0.0% images available):** In the synthetic dataset, images existed only as textual prompt descriptions (`image_description`); no physical image files were generated on disk. Consequently, the vision encoder assigned all synthetic records zero vectors $\mathbf{0} \in \mathbb{R}^{768}$ with `image_mask = False`.
- **`ocr_masks.mean()` = 0.000 (0.0% OCR available):** No optical character recognition text existed for synthetic records.
- **Scientific Implication:** The Phase 3 "multimodal" models were effectively evaluated on text-only inputs with zero-masked image and OCR vectors. Claims that multimodal fusion proved superior on the synthetic benchmark are empirically invalid.

---

## 4. Remediation: Clean Grouped Split (`clean_synthetic_splits_registry.json`)

To establish an honest, un-contaminated evaluation baseline, a **Grouped Deduplicated Clustered Split** was implemented:
- All records sharing the same normalized comment text were grouped together into atomic clusters.
- The 180 unique clusters were split 70/15/15 into train, validation, and test partitions.

### Verified Clean Split Properties:
```text
Clean Train Records:                        244 (69.7%)
Clean Validation Records:                   52 (14.9%)
Clean Test Records:                         54 (15.4%)
Train ∩ Test Exact Comment Overlap:         0 (0.0% - ZERO LEAKAGE)
Train ∩ Val Exact Comment Overlap:          0 (0.0% - ZERO LEAKAGE)
Val ∩ Test Exact Comment Overlap:           0 (0.0% - ZERO LEAKAGE)
Clean Test Class Distribution:              25 Non-PA, 24 PA, 5 Ambiguous
Registry Path:                              data/splits/clean_synthetic_splits_registry.json
```

This clean split ensures that models are evaluated on genuinely unseen comments, measuring true out-of-sample generalization.
