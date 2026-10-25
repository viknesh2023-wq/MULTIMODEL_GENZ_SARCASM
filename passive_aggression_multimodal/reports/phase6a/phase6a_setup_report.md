# Phase 6A: Human Annotation Pilot Setup Report

**Project Title:** Context-Aware Multimodal Passive-Aggression Detection in Social Media Using Image–Comment Fusion  
**Canonical Project Root:** `D:\NLP_PROJECT\passive_aggression_multimodal`  
**Linked Workspace:** `D:\multimodel_passive_aggression`  
**Milestone:** Phase 6A — Human Annotation Pilot & Inter-Annotator Agreement Setup  
**Date:** October 21, 2026  
**Auditor:** Senior ML Research Auditor & Experimental Reliability Team  

---

## 1. Executive Summary

Phase 6A establishes the infrastructure and protocol for the first independent, double-blind human annotation pilot for passive-aggression detection in this research project.

Following the forensic audits of Phases 4 through 5E:
1. **Prior Multimodal Datasets Contained Zero PA Labels:** MMSD, SarcNet, and MUStARD provided sarcasm, dialogue irony, and sentiment labels, but zero verified human-annotated passive-aggression ground truth.
2. **LLM Consensus is Unreliable as Ground Truth:** Phase 5E demonstrated that when `llama3.2` and `qwen2.5` agree on passive aggression, **73.1% of those agreements are shared false positives** due to target blindness, politeness paranoia, and modality blindness.
3. **Double-Blind Pilot Infrastructure:** Phase 6A constructs a manageable, 30-record stratified pilot sampled across six linguistic archetypes, prepared in two strictly isolated annotator forms ([annotator_1_responses.csv](file:///d:/multimodel_passive_aggression/data/annotations/phase6a/annotator_1_responses.csv) and [annotator_2_responses.csv](file:///d:/multimodel_passive_aggression/data/annotations/phase6a/annotator_2_responses.csv)), with all model outputs strictly blinded.

---

## 2. Infrastructure Verification & Lineage Preservation

We verified that all foundational artifacts from Phases 1–5E remain completely intact:
- Guidelines and Schemas: [docs/phase4/annotation_guidelines.md](file:///d:/multimodel_passive_aggression/docs/phase4/annotation_guidelines.md), [docs/phase4/annotation_schema.md](file:///d:/multimodel_passive_aggression/docs/phase4/annotation_schema.md)
- Instructions: [docs/phase5/annotator_instructions.md](file:///d:/multimodel_passive_aggression/docs/phase5/annotator_instructions.md)
- Status Manifest: [data/annotations/phase5/human_review_status_manifest.json](file:///d:/multimodel_passive_aggression/data/annotations/phase5/human_review_status_manifest.json)
- Verified Silver Dataset: [data/annotations/phase5d/silver_dataset_verified.jsonl](file:///d:/multimodel_passive_aggression/data/annotations/phase5d/silver_dataset_verified.jsonl) (19 records)
- Forensic Agreement Tables: [reports/phase5e/tables/agreement_audit_68_complete.csv](file:///d:/multimodel_passive_aggression/reports/phase5e/tables/agreement_audit_68_complete.csv)

No pre-existing artifacts were modified or overwritten.

---

## 3. Pilot Sampling Architecture & Stratification

The 30 pilot records in [data/annotations/phase6a/pilot_sampling_manifest.csv](file:///d:/multimodel_passive_aggression/data/annotations/phase6a/pilot_sampling_manifest.csv) are stratified across six distinct linguistic archetypes:

| Archetype | Count | Source Coverage | Selection Objective | Key Example Records |
| :--- | :---: | :--- | :--- | :--- |
| **1. LLM Silver-Positive Candidates** | 5 | MUStARD (2), CivilComments (2), MMSD (1) | Verified silver positives from Phase 5D exhibiting classic covert hostility. | `mustard_2_490`, `civil_comments_val_06315`, `mustard_1_8407` |
| **2. Non-PA Contrast Candidates** | 5 | SarcNet (1), CivilComments (1), MUStARD (2), MMSD (1) | Sincere praise, dialogue confession, radio static fact, and friendly banter. | `sarcnet_test_00053`, `civil_comments_val_08920`, `mustard_2_372` |
| **3. Direct Aggression & Disagreement** | 5 | CivilComments (4), MMSD (1) | Overt toxicity, blunt debate, and direct argument lacking courtesy facade. | `civil_comments_val_04917`, `civil_comments_val_02587`, `civil_comments_val_00154` |
| **4. General Sarcasm & Institutional Critique** | 5 | MUStARD (1), SarcNet (1), CivilComments (2), MMSD (1) | Grievances aimed at abstract entities (Fox News, NFL, government) or existential venting. | `mustard_2_174`, `sarcnet_test_00274`, `civil_comments_val_09401` |
| **5. Ambiguous Context-Dependent** | 5 | CivilComments (1), MMSD (3), MUStARD (1) | Sentence fragments or ambiguous statements testing context sufficiency limits. | `civil_comments_val_00793`, `mmsd_train_06126`, `mustard_2_111` |
| **6. Multimodal Image-Dependent** | 5 | MMSD (3), SarcNet (2) | Posts where visual evidence (stunt photo, follower screenshot, portrait) is critical to intent. | `mmsd_validation_01859`, `mmsd_test_00916`, `sarcnet_train_00045` |
| **Total** | **30** | MUStARD (6), CivilComments (10), MMSD (10), SarcNet (4) | Balanced across sources, modalities, and agreement states (22 agreements, 8 disagreements). | Fully documented in manifest |

---

## 4. Language Scope and Modality Breakdown

### 4.1 Language Scope:
- **Core English Benchmark (`CORE_ENGLISH_BENCHMARK`):** Exactly **27 records (90.0%)** are in English across MUStARD, CivilComments, MMSD, and SarcNet. This forms the standardized English evaluation benchmark.
- **Non-English Separate Analysis (`NON_ENGLISH_SEPARATE_ANALYSIS`):** Exactly **3 records (10.0%)** are Chinese Weibo posts from SarcNet (`sarcnet_test_00053`, `sarcnet_train_00045`, `sarcnet_train_00844`).  
  *Governance Rule:* These non-English records are explicitly flagged in the sampling manifest and response files. They are segregated for exploratory cross-lingual analysis and are not silently mixed into the English-only benchmark.

### 4.2 Modality Coverage:
- **Text-Only Records:** Exactly **16 records (53.3%)** consist of text comments plus conversational turns.
- **Image-Bearing Records:** Exactly **14 records (46.7%)** include accompanying visual images in `data/images/`. All 14 image files have been verified as existing and decodable on disk.

---

## 5. Double-Blind Independence Protection

To eliminate bias and preserve experimental integrity:
1. **Complete Blindness:** Annotator 1 and Annotator 2 response templates contain **zero model predictions, zero model rationales, zero silver labels, and zero audit categories**. Annotators see only the raw social media comment, dialogue context, image path, and language metadata.
2. **File Isolation:**
   - Annotator 1 operates exclusively in [data/annotations/phase6a/annotator_1_responses.csv](file:///d:/multimodel_passive_aggression/data/annotations/phase6a/annotator_1_responses.csv).
   - Annotator 2 operates exclusively in [data/annotations/phase6a/annotator_2_responses.csv](file:///d:/multimodel_passive_aggression/data/annotations/phase6a/annotator_2_responses.csv).
3. **No Fabrication of Results:** The status manifest ([annotation_status_manifest.json](file:///d:/multimodel_passive_aggression/data/annotations/phase6a/annotation_status_manifest.json)) reports:
   - `annotator_01`: `completed_count: 0`, `status: "AWAITING_HUMAN_INPUT"`
   - `annotator_02`: `completed_count: 0`, `status: "AWAITING_HUMAN_INPUT"`
   - `inter_annotator_agreement`: `cohens_kappa: null`, `raw_agreement_percent: null`, `status: "AWAITING_COMPLETED_REVIEWS"`
4. **Adjudication Segregation:** Disagreements will be exported to `adjudication_queue.csv` only after both annotators have completed all 30 judgments.

---

## 6. Agreement Engine & Adjudication Pipeline

The agreement engine ([scripts/analyze_phase6a_agreement.py](file:///d:/multimodel_passive_aggression/scripts/analyze_phase6a_agreement.py)) supports:
- Automated completion tracking via `--check-status`.
- Calculation of observed raw agreement ($P_o$) and Cohen's Kappa ($\kappa$).
- Extraction of disagreements into the adjudication queue.
- Quality benchmark gate: $\kappa \ge 0.70$ required for gold dataset validation.

---

## 7. Current Governance State & Active Blockers

- **Evaluation Readiness Status:** **`BLOCKED_PENDING_HUMAN_ANNOTATION`**
- **Active Blockers:** Genuine human reviews must be entered into `annotator_1_responses.csv` and `annotator_2_responses.csv` before agreement calculation or model evaluation can proceed.
- **Phase 6A Setup Complete:** Awaiting human annotation completion and user authorization before proceeding to Phase 6B.
