# Phase 2 — Multimodal Preprocessing, Leakage Prevention & Feature Extraction Plan

**Project Title:** Context-Aware Multimodal Passive-Aggression Detection in Social Media Using Image–Comment Fusion  
**Phase:** Phase 2 Technical Blueprint  
**Date:** October 21, 2026  
**Auditor / ML Engineer:** Senior Machine Learning Engineer & Multimodal NLP Researcher  

---

## 1. Architectural Objectives

Phase 2 bridges the gap between raw collected research datasets and future multimodal classifier training. The core objectives are:
1. **Zero Data Leakage:** Ensure no shared post IDs, conversation threads, duplicate texts, or identical/near-identical images cross train, validation, or test boundaries.
2. **Pragmatic Text Preservation:** Normalize social media text without destroying the subtle pragmatic markers (emojis, repeated punctuation, capitalization, negation, weaponized courtesy) essential for passive-aggression detection.
3. **Robust Image Preprocessing:** Standardize image format, resolution, and color spaces across 27,970 images while identifying duplicate perceptual hashes.
4. **Embedded Text Extraction (OCR):** Build an OCR pipeline to capture in-image textual cues from memes and screenshots.
5. **Standardized Encoder & Fusion Interfaces:** Implement provider-independent extraction interfaces for vision encoders (CLIP / SigLIP) and language encoders (DeBERTa / RoBERTa), along with fusion architecture specifications.
6. **Annotation Pilot Execution:** Sample and evaluate 150 diverse real-world social media examples across passive-aggressive subtypes and contrast classes.

---

## 2. Preprocessing & Leakage-Safe Splitting Blueprint

```
+-----------------------------------------------------------------------------+
|                          RAW DATASETS IN DATA/RAW                           |
|       (MMSD: 24,635 | SarcNet: 3,335 | GoEmotions: 54,263 | MUStARD: 690)   |
+-----------------------------------------------------------------------------+
                                      |
                                      v
+-----------------------------------------------------------------------------+
|                     GROUP IDENTIFICATION & DEDUPLICATION                    |
|   - Group Key: Post ID, Thread ID, Duplicate Image Hash, Exact Text Hash    |
|   - Perceptual Image Difference Hashing (dHash)                             |
+-----------------------------------------------------------------------------+
                                      |
                                      v
+-----------------------------------------------------------------------------+
|                     LEAKAGE-SAFE GROUPED DATA SPLITTING                     |
|   - Multimodal Core: 80% Train, 10% Validation, 10% Test (Seed: 42)         |
|   - Benchmark Preservation: Maintain official MMSD test set for parity      |
+-----------------------------------------------------------------------------+
                                      |
         +----------------------------+----------------------------+
         |                                                         |
         v                                                         v
+------------------------------------+    +------------------------------------+
|    TEXT & CONTEXT PREPROCESSING    |    |   IMAGE PREPROCESSING & OCR PILOT  |
| - Caption, Comment, Context, OCR   |    | - RGB conversion, 224/384px resize |
| - Preserve: emojis, negation, caps |    | - Local OCR extraction pilot       |
| - Derived fields in data/processed |    | - Audit output in data/processed   |
+------------------------------------+    +------------------------------------+
         |                                                         |
         +----------------------------+----------------------------+
                                      |
                                      v
+-----------------------------------------------------------------------------+
|                 ENCODER & FUSION ARCHITECTURE INTERFACES                    |
|   - Vision Encoder: CLIP / SigLIP (768-dim / 1152-dim)                      |
|   - Language Encoder: DeBERTa / RoBERTa (768-dim / 1024-dim)                |
|   - Multimodal Fusion Head: Concatenation + Dense Projection + Classifier   |
+-----------------------------------------------------------------------------+
                                      |
                                      v
+-----------------------------------------------------------------------------+
|               STRATIFIED PASSIVE-AGGRESSION ANNOTATION PILOT                |
|   - 150 real-world candidate records across all 6 PA subtypes + contrasts   |
|   - Multi-agent heuristic & multimodal evaluation with full provenance      |
+-----------------------------------------------------------------------------+
```

---

## 3. Implementation Plan by Component

### Component B: Leakage-Safe Grouped Splits
- **Configuration:** [`configs/split_config.yaml`](file:///d:/multimodel_passive_aggression/configs/split_config.yaml) setting fixed seed `42`, ratios `0.80 / 0.10 / 0.10`.
- **Script:** [`scripts/create_grouped_splits.py`](file:///d:/multimodel_passive_aggression/scripts/create_grouped_splits.py) grouping by `(post_id, conversation_id, image_hash, normalized_text)`.
- **Validation Script:** [`scripts/validate_splits.py`](file:///d:/multimodel_passive_aggression/scripts/validate_splits.py) asserting 0 intersection across splits.

### Component C: Text & Context Preprocessing
- **Modules:** [`src/preprocessing/text_preprocessor.py`](file:///d:/multimodel_passive_aggression/src/preprocessing/text_preprocessor.py) and [`src/preprocessing/context_preprocessor.py`](file:///d:/multimodel_passive_aggression/src/preprocessing/context_preprocessor.py).
- **Rules:** Preserve casing features (e.g. ALL CAPS sarcasm), repeated punctuation (`???`, `...`), emojis, and negation words (`never`, `not`, `hardly`).

### Component D: Image Auditing & Preprocessing
- **Module:** [`src/preprocessing/image_preprocessor.py`](file:///d:/multimodel_passive_aggression/src/preprocessing/image_preprocessor.py).
- **Outputs:** Verification of all 27,970 images, dHash perceptual hash calculation, standardized RGB tensor transformation pipelines.

### Component E: Local OCR Processor
- **Module:** [`src/preprocessing/ocr_processor.py`](file:///d:/multimodel_passive_aggression/src/preprocessing/ocr_processor.py) and [`scripts/run_ocr.py`](file:///d:/multimodel_passive_aggression/scripts/run_ocr.py).
- **Execution:** Run on a stratified pilot of 100 images (MMSD memes and SarcNet posts) to measure runtime, text detection frequency, and memory usage.

### Component F: Feature Extraction & Encoder Preparation
- **Module:** [`src/preprocessing/feature_extraction.py`](file:///d:/multimodel_passive_aggression/src/preprocessing/feature_extraction.py).
- **Specs:** Standardized batch feature extractor supporting CLIP ViT-B/32, SigLIP, DeBERTa-v3, and RoBERTa, with device fallback (CUDA/CPU) and missing-modality masking.

### Component G: Multimodal Fusion Interface
- **Specification:** [`reports/fusion_architecture_design.md`](file:///d:/multimodel_passive_aggression/reports/fusion_architecture_design.md) and [`src/preprocessing/fusion_interface.py`](file:///d:/multimodel_passive_aggression/src/preprocessing/fusion_interface.py) defining projection layers, cross-modal attention/concatenation, and dual classification heads.

### Component H: Annotation Pilot
- **Script:** [`scripts/run_annotation_pilot.py`](file:///d:/multimodel_passive_aggression/scripts/run_annotation_pilot.py).
- **Target:** 150 candidate examples with balanced subtype representations, provisional labels, and human review queues.
