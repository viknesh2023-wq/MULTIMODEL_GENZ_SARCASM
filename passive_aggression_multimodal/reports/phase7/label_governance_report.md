# Phase 7 Label Governance & Eligibility Report

**Date:** October 21, 2026  
**Artifact:** `reports/phase7/label_governance_report.md`  

## 1. Governance Principles
1. **Semantic Separation:** Sarcasm $
eq$ Passive Aggression; Toxicity $
eq$ Passive Aggression; Emotion $
eq$ Passive Aggression.
2. **Immutable Provenance:** Source labels must never be rewritten to claim passive aggression without human annotation.
3. **No Metric Inflation:** LLM consensus cannot serve as validated ground truth.

## 2. Label Eligibility Registry
| Dataset | Governance Status | Permitted Objective | Supervised PA Eligible? | Auxiliary Eligible? |
| :--- | :--- | :--- | :---: | :---: |
| **mmsd** | `ELIGIBLE_FOR_AUXILIARY_TASK` | Multimodal Sarcasm Representation Learning | NO | YES |
| **sarcnet** | `ELIGIBLE_FOR_AUXILIARY_TASK` | Multilingual Cross-Modal Sarcasm Representation | NO | YES |
| **goemotions** | `ELIGIBLE_FOR_AUXILIARY_TASK` | Pragmatic Affect & Emotion Auxiliary Cues | NO | YES |
| **civil_comments** | `ELIGIBLE_FOR_AUXILIARY_TASK` | Overt Hostility Contrast Discrimination | NO | YES |
| **mustard** | `ELIGIBLE_FOR_AUXILIARY_TASK` | Dialogue Context Sarcasm Modeling | NO | YES |
| **synthetic_pilot** | `ELIGIBLE_FOR_EXPLORATORY_WEAK_SUPERVISION` | Pipeline Smoke Testing & Synthetic Pretraining | NO | YES |
| **silver_dataset_verified** | `ELIGIBLE_FOR_EXPLORATORY_WEAK_SUPERVISION` | Weak-Supervision Diagnostic Baselines | YES | YES |
| **phase6d_adjudicated** | `ELIGIBLE_FOR_EXPLORATORY_WEAK_SUPERVISION` | Error Diagnostics & Prompt Ablation Analysis | YES | NO |
| **phase6a_human_pilot** | `ELIGIBLE_FOR_HUMAN_ANNOTATION` | Real-World Ground Truth Benchmark Evaluation | NO | NO |
