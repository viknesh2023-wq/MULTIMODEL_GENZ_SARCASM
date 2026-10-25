# Phase 4 Repository and Artifact Audit Report
**Project Title:** Context-Aware Multimodal Passive-Aggression Detection in Social Media Using Image–Comment Fusion  
**Canonical Project Root:** `D:\NLP_PROJECT\passive_aggression_multimodal` (NTFS Junction: `D:\multimodel_passive_aggression`)  
**Auditor:** Senior ML Research Engineer & Experimental Auditor  
**Artifact:** `reports/phase4/phase4_repository_audit.md`  
**Date:** October 21, 2026  

---

## 1. Executive Summary

This independent audit verifies every reported claim, artifact path, row count, checksum, label distribution, and checkpoint across Phases 1, 2, and 3. The audit operates on physical evidence directly inspected on disk, establishing traceability for all experimental claims.

### Key Audit Findings:
1. **Directory Integrity:** Verified that `D:\multimodel_passive_aggression` and `D:\NLP_PROJECT\passive_aggression_multimodal` point to the exact same physical directory via NTFS Junction (`os.path.samefile == True`).
2. **Manifest Row Counts:** Verified that all 48,660 raw and derived records exist physically on disk with zero corruption.
3. **Multimodal Split Record Counts:** Verified that `grouped_train.jsonl` (22,376), `grouped_validation.jsonl` (2,797), and `grouped_test.jsonl` (2,797) sum exactly to 27,970 records.
4. **Image Decodability:** Verified that all 27,970 physical images (24,635 MMSD + 3,335 SarcNet) exist locally and decode cleanly.
5. **Real-World Passive-Aggression Label Availability:** **Confirmed 0 verified gold human-annotated passive-aggression labels** in the 27,970 real social media corpus. All 27,970 records have `passive_aggression_label: None`.
6. **Human Adjudication Status:** The 150-record annotation pilot and 150-record review queue have completed 0 human reviews (`human_verification_status: AWAITING_HUMAN_REVIEW`).
7. **Synthetic Pilot Integrity:** 350 synthetic records exist on disk, but 120 duplicate comments exist across scenario variants.
8. **Checkpoints:** All 5 Phase 3 checkpoints exist with verifiable SHA-256 hashes and state dictionaries.

---

## 2. Traceability Matrix: Reported vs. Physical Evidence

| Asset / Metric | Phase 3 Reported Value | Physically Verified Evidence | Audit Status | Evidence / Notes |
|---|---|---|---|---|
| **Canonical Junction** | Linked via NTFS Junction | `os.path.samefile(p1, p2) == True` | `VERIFIED` | Resolves to identical inode |
| **Unified Raw Manifest** | 48,660 records | 48,660 lines (50.38 MB) | `VERIFIED` | `data/interim/unified_raw_manifest.jsonl` |
| **Preprocessed Records** | 48,660 records | 48,660 lines (71.64 MB) | `VERIFIED` | `data/processed/preprocessed_multimodal_records.jsonl` |
| **MMSD Manifest** | 24,635 records | 24,635 lines (23.45 MB) | `VERIFIED` | `data/manifests/mmsd_manifest.jsonl` |
| **SarcNet Manifest** | 3,335 records | 3,335 lines (3.75 MB) | `VERIFIED` | `data/manifests/sarcnet_manifest.jsonl` |
| **MUStARD Manifest** | 690 records | 690 lines (0.83 MB) | `VERIFIED` | `data/manifests/mustard_manifest.jsonl` |
| **GoEmotions Manifest** | 10,000 records | 10,000 lines (9.33 MB) | `VERIFIED` | Intentionally sampled from 58k |
| **Civil Comments Manifest** | 10,000 records | 10,000 lines (13.03 MB) | `VERIFIED` | Intentionally sampled from 1.8M |
| **Grouped Train Split** | 22,376 records (80.0%) | 22,376 lines (22.45 MB) | `VERIFIED` | `data/splits/grouped_train.jsonl` |
| **Grouped Validation Split** | 2,797 records (10.0%) | 2,797 lines (2.83 MB) | `VERIFIED` | `data/splits/grouped_validation.jsonl` |
| **Grouped Test Split** | 2,797 records (10.0%) | 2,797 lines (2.81 MB) | `VERIFIED` | `data/splits/grouped_test.jsonl` |
| **Total Multimodal Records** | 27,970 records | 27,970 lines across splits | `VERIFIED` | Exactly matches MMSD (24,635) + SarcNet (3,335) |
| **MMSD Images on Disk** | 24,635 images | 24,635 image files in `data/images/mmsd` | `VERIFIED` | 100% physically present |
| **SarcNet Images on Disk** | 3,335 images | 3,335 image files in `data/images/sarcnet` | `VERIFIED` | 100% physically present |
| **Synthetic Pilot Records** | 350 scenarios | 350 lines (0.54 MB) | `VERIFIED` | `data/synthetic/synthetic_pilot_350.jsonl` |
| **Annotation Pilot Records** | 150 records | 150 lines (0.29 MB) | `VERIFIED` | `data/processed/annotation_pilot_150.jsonl` |
| **Human Review Queue** | 150 items (50 high priority) | 150 items queued in JSON | `VERIFIED` | `data/processed/human_review_queue_150.json` |
| **Real PA Ground Truth** | 0 human verified | `passive_aggression_label: None` (27,970) | `VERIFIED` | Real social media PA labels unannotated |
| **Automated Tests** | 36 passing | 36 / 36 tests passing in 22.98s | `VERIFIED` | Verified by automated pytest run |

---

## 3. Physical Checkpoint & Artifact Audit

All checkpoints generated during Phase 3 were audited for file presence, byte size, and SHA-256 integrity:

| Checkpoint Identifier | Physical File Path | Size (KB) | SHA-256 Hash Prefix | State Dict Contents |
|---|---|---:|---|---|
| **Primary TF-IDF Baseline** | `artifacts/phase3/text_baseline/best_text_baseline_primary.pkl` | 533.3 KB | `130fd42ac8113ba0` | Scikit-Learn Pipeline (FeatureUnion + LogReg $C=0.5$) |
| **Auxiliary Sarcasm Baseline** | `artifacts/phase3/text_baseline/best_text_baseline_sarcasm.pkl` | 992.9 KB | `d9c0cd493caffe45` | Scikit-Learn Pipeline (FeatureUnion + LogReg $C=1.0$) |
| **Transformer Text Baseline** | `artifacts/phase3/text_transformer/best_checkpoint.pt` | 2,331.0 KB | `5176852b0e68c31d` | PyTorch State Dict (Epoch 5, ClassificationHead) |
| **Gated Multimodal Fusion** | `artifacts/phase3/multimodal_fusion/best_checkpoint.pt` | 32,494.8 KB | `6f6c3dcd8b579d0f` | PyTorch State Dict (Epoch 5, GatedFusion + Head) |
| **Context-Aware Fusion** | `artifacts/phase3/context_aware_fusion/best_checkpoint.pt` | 47,849.7 KB | `56b35fc945f8e9b9` | PyTorch State Dict (Epoch 6, CrossAttn + Head) |
| **Final Predictions Log** | `artifacts/phase3/evaluation/final_model_predictions.jsonl` | 71.3 KB | `404d4ff7a8d64fe2` | 216 records (54 test items $\times$ 4 models) |
| **Ablation Results Summary** | `artifacts/phase3/evaluation/ablation_study_results.json` | 2.3 KB | `31b6dd322f30eecf` | Full 8-experiment metrics dictionary |

---

## 4. Pretrained Encoder Weights & Feature Store Provenance

| Encoder Identity | Source / Location | Local Checkpoint Size | Output Dimension | Physical Execution Status |
|---|---|---:|---:|---|
| `FacebookAI/xlm-roberta-base` | Offline Hugging Face Hub cache (`models--FacebookAI--xlm-roberta-base`) | 1,077.4 MB | 768 dims | `VERIFIED_REAL_PRETRAINED` (Loaded offline, evaluation freeze) |
| `openai/clip-vit-base-patch32` | Not in local HF cache | 0.0 MB | 768 dims | `OFFLINE_INTERFACE_FALLBACK` (Spatial grid patch projection) |
| `synthetic_pilot_350.pt` | `data/features/phase3/synthetic_pilot_350.pt` | 4.51 MB | 768 dims | `VERIFIED` (Comment: 768, Context: 768, Img: zero-vector, OCR: zero-vector) |

---

## 5. Audit Conclusion

The physical repository reflects high mechanical consistency:
- All file manifests, record counts, split counts, image assets, checkpoints, and unit test results match reported metrics.
- Crucially, the audit confirms the central limitation: **zero human-annotated passive-aggression ground truth exists in the real multimodal corpus**, meaning all primary model evaluations were performed exclusively on the 350 synthetic pilot scenarios.
