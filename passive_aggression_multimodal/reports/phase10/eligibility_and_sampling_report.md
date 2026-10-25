# Phase 10 Eligibility and Sampling Report
## Stratified Candidate Corpus Construction for Automated Annotation

**Project Title:** Context-Aware Multimodal Passive-Aggression Detection in Social Media Using Image–Comment Fusion  
**Artifact:** `reports/phase10/eligibility_and_sampling_report.md`  
**Date:** October 22, 2026  

---

## 1. Corpus Summary
- **Total Eligible Records Sampled:** 100
- **Deduplication Status:** 100% unique raw and normalized text SHA-256 hashes.
- **Excluded Duplicates Identified:** 12

## 2. Source Dataset Distribution
| Source Dataset | Count | Percentage |
|---|---|---|
| **mustard** | 43 | 43.0% |
| **civil_comments** | 28 | 28.0% |
| **mmsd** | 26 | 26.0% |
| **sarcnet** | 3 | 3.0% |

## 3. Modality Coverage
- **Text Only Records:** 71 (71.0%)
- **Image-Bearing Records:** 29 (29.0%)

## 4. Language Distribution
- **en**: 99 records
- **zh**: 1 records

## 5. Scientific Governance Boundary
- None of the source records inherit pre-existing sarcasm, toxicity, or emotion labels as passive-aggression truth.
- All records will be independently judged by dual LLMs according to the 3-prong operational definition.
