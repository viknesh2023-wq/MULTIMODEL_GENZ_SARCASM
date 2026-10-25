# Phase 6A: Human Annotation Pilot Protocol

**Project Title:** Context-Aware Multimodal Passive-Aggression Detection in Social Media Using Image–Comment Fusion  
**Canonical Project Root:** `D:\NLP_PROJECT\passive_aggression_multimodal`  
**Linked Workspace:** `D:\multimodel_passive_aggression`  
**Milestone:** Phase 6A — Human Annotation Pilot Protocol  
**Date:** October 21, 2026  
**Version:** 6.0.0  
**Author:** Senior ML Research Auditor & Experimental Reliability Team  

---

## 1. Research Objective & Scientific Rationale

Phases 1 through 5E established that:
1. **Zero Verified Human Labels in Corpus:** Prior multimodal datasets (MMSD, SarcNet, MUStARD) contain sarcasm, humor, or dialogue irony labels, but **zero verified human-annotated passive-aggression labels**.
2. **LLM Consensus is Unreliable as Ground Truth:** Forensic auditing of 68 model-agreement records in Phase 5E demonstrated a **73.1% shared false-positive rate** when text LLMs evaluate passive aggression. LLM pseudo-labels cannot substitute for human judgment.
3. **Primary Evaluation Blocked:** Real-world benchmark performance evaluation remains strictly blocked (`BLOCKED_PENDING_HUMAN_ANNOTATION`) until independent human annotations are recorded and adjudicated.

### Objective of Phase 6A:
To execute a controlled, double-blind human annotation pilot on **30 stratified records**, measuring inter-annotator agreement ($\kappa$) and establishing the first scientifically defensible gold-standard benchmark labels for multimodal passive-aggression detection.

---

## 2. Pilot Sampling Architecture & Stratification

The 30-record pilot is drawn from the stratified 100-record manifest ([data/annotations/phase4/human_sampling_manifest_100.jsonl](file:///d:/multimodel_passive_aggression/data/annotations/phase4/human_sampling_manifest_100.jsonl)) across six distinct linguistic archetypes:

| Sampling Archetype | Record Count | Source Datasets | Selection Rationale | Key Examples |
| :--- | :---: | :--- | :--- | :--- |
| **1. LLM Silver-Positive Candidates** | 5 | MUStARD (2), CivilComments (2), MMSD (1) | Verified silver positives from Phase 5D exhibiting classic backhanded courtesy and covert blame. | `mustard_2_490`, `civil_comments_val_06315`, `mustard_1_8407` |
| **2. Non-PA Contrast Candidates** | 5 | SarcNet (1), CivilComments (1), MUStARD (2), MMSD (1) | Genuine non-hostile statements: sincere praise, character self-confession, factual radio static, and friendly banter. | `sarcnet_test_00053`, `civil_comments_val_08920`, `mustard_2_372` |
| **3. Direct Aggression & Disagreement** | 5 | CivilComments (4), MMSD (1) | Overt hostility, blunt political debate, and direct insults lacking polite camouflage or plausible deniability. | `civil_comments_val_04917`, `civil_comments_val_02587`, `civil_comments_val_00154` |
| **4. General Sarcasm & Institutional Critique** | 5 | MUStARD (1), SarcNet (1), CivilComments (2), MMSD (1) | Satire or grievances directed at abstract entities (Fox News, NFL, government) or existential venting. | `mustard_2_174`, `sarcnet_test_00274`, `civil_comments_val_09401` |
| **5. Ambiguous Context-Dependent** | 5 | CivilComments (1), MMSD (3), MUStARD (1) | Dangling fragments or isolated exclamations testing the limits of context sufficiency. | `civil_comments_val_00793`, `mmsd_train_06126`, `mustard_2_111` |
| **6. Multimodal Image-Dependent** | 5 | MMSD (3), SarcNet (2) | Posts where accompanying visual evidence (stunt photo, follower screenshot, portrait) is critical to determine target or intent. | `mmsd_validation_01859`, `mmsd_test_00916`, `sarcnet_train_00045` |
| **Total Pilot Size** | **30** | MUStARD (6), CivilComments (10), MMSD (10), SarcNet (4) | Fully balanced across archetypes, sources, and agreement states. | See `pilot_sampling_manifest.csv` |

---

## 3. Language Scope Policy & Modality Breakdown

### 3.1 Language Scope:
- **Core English Benchmark (`CORE_ENGLISH_BENCHMARK`):** Exactly **27 records (90.0%)** are in English. This forms the primary, standardized evaluation benchmark for the project.
- **Non-English Separate Analysis (`NON_ENGLISH_SEPARATE_ANALYSIS`):** Exactly **3 records (10.0%)** are Chinese Weibo posts from SarcNet (`sarcnet_test_00053`, `sarcnet_train_00045`, `sarcnet_train_00844`).  
  *Policy:* In accordance with Phase 6A rules, these non-English records are explicitly tagged and segregated for exploratory cross-lingual analysis. They are not silently commingled with the core English benchmark.

### 3.2 Modality Coverage:
- **Text-Only Records:** Exactly **16 records (53.3%)** consist of text utterance plus conversational dialogue turns.
- **Image-Bearing Records:** Exactly **14 records (46.7%)** include an accompanying image file in `data/images/`. All 14 image files have been verified as existing and decodable on disk.

---

## 4. Double-Blind Independence Protocol

To ensure unimpeachable scientific validity:
1. **Complete Isolation:** Annotator 1 ([annotator_1_responses.csv](file:///d:/multimodel_passive_aggression/data/annotations/phase6a/annotator_1_responses.csv)) and Annotator 2 ([annotator_2_responses.csv](file:///d:/multimodel_passive_aggression/data/annotations/phase6a/annotator_2_responses.csv)) annotate identical record lists independently.
2. **LLM Blindness:** All LLM predictions, model rationales, silver labels, consensus status, and Phase 5C/5D/5E audit categories are **strictly stripped and hidden** from the response files. Annotators observe only the raw comment text, dialogue context, image path, and language metadata.
3. **No Hidden Answer Key:** Annotators are instructed to record their own authentic pragmatic interpretation, without trying to guess what an AI model predicted.
4. **Adjudication Segregation:** The adjudication queue ([adjudication_queue.csv](file:///d:/multimodel_passive_aggression/data/annotations/phase6a/adjudication_queue.csv)) is populated only after both annotators have independently submitted completed responses.

---

## 5. Schema & Human Annotation Data Contract

For every record, annotators must record:
1. `primary_label`: Must be one of `passive_aggressive`, `non_passive_aggressive`, `ambiguous`.
2. `subtype`: Required if `passive_aggressive` (e.g. `backhanded_compliment`, `fake_politeness`, `indirect_criticism`, `implied_blame`, `sarcastic_agreement`, `dismissiveness`, `other_or_unclear`).
3. `target_evidence`: Explicit identification of the target (e.g. "Addressee in conversation", "Third-party entity", "Abstract institution", "None").
4. `covert_hostility_evidence`: Description of how hostility is veiled or encoded.
5. `plausible_deniability_evidence`: Description of the surface polite or literal phrasing that enables deniability.
6. `context_sufficiency`: Assessment of whether context is `SUFFICIENT`, `PARTIAL`, or `INSUFFICIENT`.
7. `image_reviewed`: Mandatory verification of whether the image was inspected (`YES`, `NO`, `NOT_APPLICABLE`).
8. `image_role`: Multimodal role assessment (`ESSENTIAL`, `HELPFUL`, `IRRELEVANT`, `ABSENT`).
9. `confidence`: `HIGH`, `MEDIUM`, or `LOW`.
10. `evidence_span`: Substring exhibiting passive-aggressive markers.

---

## 6. Agreement Analysis & Adjudication Pipeline

Inter-annotator reliability is computed via `scripts/analyze_phase6a_agreement.py`:
- **Raw Agreement Percentage:** $P_o = \frac{\text{matching pairs}}{N} \times 100\%$
- **Chance-Corrected Reliability:** Cohen's Kappa ($\kappa$):
  $$\kappa = \frac{P_o - P_e}{1 - P_e}$$
- **Quality Benchmark Gate:** A threshold of $\kappa \ge 0.70$ is required for reliable annotation quality before benchmark freezing.
- **Disagreement Adjudication:** Disagreements between Annotator 1 and Annotator 2 are routed to a senior adjudicator to produce the final consensus gold label.

---

## 7. Current Governance State

- **Annotation State:** `AWAITING_HUMAN_INPUT` (Annotator 1: 0 completed; Annotator 2: 0 completed).
- **Benchmark Evaluation Gate:** **STRICTLY BLOCKED** pending completion of independent human annotation and adjudication.
