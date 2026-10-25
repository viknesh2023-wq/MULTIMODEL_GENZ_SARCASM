# Phase 7 Dataset Inventory Report

**Date:** October 21, 2026  
**Artifact:** `reports/phase7/dataset_inventory_report.md`  

## 1. Executive Summary
- Total datasets cataloged: **9**
- Total records indexed: **49,089**
- Multimodal records with verified images: **27,970**
- Records with verified human passive-aggression ground truth: **0**
- Records with verified LLM silver passive-aggression labels: **20**

## 2. Dataset Inventory Table
| Dataset | Records | Unique Texts | Images on Disk | Language | Modality | Original Scheme | PA Labels? | Eligibility |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :---: | :--- |
| **MMSD / MMSD2.0 / MMSD3.0** | 24,635 | 24,009 | 24,635 | en | Image+Text | sarcasm (1=sarcastic, 0=non-sa... | NO | `ELIGIBLE_FOR_AUXILIARY_TASK` |
| **SarcNet Multilingual Sarcasm** | 3,335 | 3,305 | 3,335 | en, zh | Image+Text | multimodal sarcasm (text_label... | NO | `ELIGIBLE_FOR_AUXILIARY_TASK` |
| **GoEmotions (Manifest Subset)** | 10,000 | 9,980 | 0 | en | Text | 27 Reddit emotions + neutral... | NO | `ELIGIBLE_FOR_AUXILIARY_TASK` |
| **Civil Comments (Manifest Subset)** | 10,000 | 9,994 | 0 | en | Text | toxicity, severe_toxicity, ins... | NO | `ELIGIBLE_FOR_AUXILIARY_TASK` |
| **MUStARD Dialogue Sarcasm** | 690 | 681 | 0 | en | Text | sarcasm boolean, speaker, show... | NO | `ELIGIBLE_FOR_AUXILIARY_TASK` |
| **Synthetic PA Pilot 350** | 350 | 180 | 0 | en | Text | synthetic passive_aggression l... | NO | `ELIGIBLE_FOR_EXPLORATORY_WEAK_SUPERVISION` |
| **Phase 5D Verified Silver PA** | 19 | 19 | 0 | en | Text | passive_aggressive (silver ver... | NO | `ELIGIBLE_FOR_EXPLORATORY_WEAK_SUPERVISION` |
| **Phase 6D Adjudicated Pilot (30 Records)** | 30 | 30 | 14 | en (27), zh (3) | Image+Text | non_passive_aggressive (16), a... | NO | `ELIGIBLE_FOR_EXPLORATORY_WEAK_SUPERVISION` |
| **Phase 6A Human Pilot Template** | 30 | 30 | 14 | en (27), zh (3) | Image+Text | 0 completed reviews, 30 pendin... | NO | `ELIGIBLE_FOR_HUMAN_ANNOTATION` |
