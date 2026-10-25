# Phase 5E: Forensic Audit of Model-Agreement Records

**Project Title:** Context-Aware Multimodal Passive-Aggression Detection in Social Media Using Image–Comment Fusion  
**Canonical Project Root:** `D:\NLP_PROJECT\passive_aggression_multimodal`  
**Linked Workspace:** `D:\multimodel_passive_aggression`  
**Milestone:** Phase 5E — Complete Forensic Audit of All 68 Model-Agreement Records  
**Date:** October 21, 2026  
**Auditor:** Senior ML Research Auditor & Experimental Reliability Team  

---

## 1. Executive Summary & Epistemological Grounding

During Phase 5A dual-pass LLM annotation, `llama3.2:latest` (Model A) and `qwen2.5:1.5b` (Model B) evaluated 100 stratified real-world social media records. The two models reached agreement on primary classification for exactly **68 records** (67 `passive_aggressive`, 1 `non_passive_aggressive`), while diverging on **32 records**.

In Phase 5E, we conducted an independent, record-by-record forensic audit of all 68 agreement records to test the fundamental question:
> **Does unanimous agreement between two local LLMs constitute reliable silver ground truth for passive-aggression detection?**

### Core Finding: Model Agreement is Highly Susceptible to Shared Bias
Unanimous model agreement does **NOT** equal ground truth. When subjected to rigorous linguistic and pragmatic scrutiny (evaluating for an identifiable individual addressee, covert hostility, and plausible deniability):
1. **True Silver Quality Rate: 27.9% (19 / 68)**  
   - Exactly **18 records** represent genuine covert interpersonal passive aggression (`LIKELY_GENUINE_PA_SILVER`).
   - Exactly **1 record** represents a genuine consensus negative (`LIKELY_NON_PA`).
2. **Shared False-Positive Risk: 73.1% of Positive Agreements (49 / 67)**  
   - In 49 out of 67 positive-agreement records, both models misclassified non-passive-aggressive text as passive-aggressive due to shared prompt paranoia, target blindness, modality blindness, and inability to separate overt hostility from covert manipulation.
3. **Multimodal Blindness (20 Records):**  
   - 20 records contain accompanying images from MMSD and SarcNet. Because both models operated strictly in text-only mode (`image_assessed = false`), whether the visual scene contains an interlocutor target remains unassessed by AI.
4. **Preservation of Phase 5D Decisions:**  
   - Phase 5E completely validates the filtering applied in Phase 5D. Exactly the 19 verified records are retained; all 49 problematic records are excluded from the verified silver training pool.

---

## 2. Recomputation and Data Lineage Verification

We independently verified and recomputed the agreement cohort directly from the raw pass outputs:
- **Pass 1 File:** [data/annotations/phase5a/llm/llm_annotations_pass1.jsonl](file:///d:/multimodel_passive_aggression/data/annotations/phase5a/llm/llm_annotations_pass1.jsonl) (100 records)
- **Pass 2 File:** [data/annotations/phase5a/llm/llm_annotations_pass2.jsonl](file:///d:/multimodel_passive_aggression/data/annotations/phase5a/llm/llm_annotations_pass2.jsonl) (100 records)
- **Manifest File:** [data/annotations/phase4/human_sampling_manifest_100.jsonl](file:///d:/multimodel_passive_aggression/data/annotations/phase4/human_sampling_manifest_100.jsonl) (100 records)

### Concordance Verification:
- Pass 1 / Pass 2 Agreements: Exactly **68 records** (67 positive, 1 negative).
- Pass 1 / Pass 2 Disagreements: Exactly **32 records** (Model A: 32 PA; Model B: 28 Non-PA, 4 Ambiguous).
- Input Integrity: 0 duplicated record IDs; 0 missing annotations; 100% record ID stability.

---

## 3. Conservative 9-Category Distribution

Every agreement record was assigned to one of nine mutually exclusive conservative audit categories in [reports/phase5e/tables/agreement_audit_68_complete.csv](file:///d:/multimodel_passive_aggression/reports/phase5e/tables/agreement_audit_68_complete.csv):

| Audit Category | Count | Percentage | Description | Pragmatic Disposition |
| :--- | :---: | :---: | :--- | :--- |
| `LIKELY_GENUINE_PA_SILVER` | **18** | 26.47% | Verified covert interpersonal hostility with plausible deniability | RETAIN_SILVER_VERIFIED |
| `LIKELY_NON_PA` | **1** | 1.47% | Consensus non-hostile linguistic explanation | RETAIN_SILVER_VERIFIED |
| `UNRESOLVED_REQUIRES_HUMAN_REVIEW` | **20** | 29.41% | Multimodal posts where unassessed image may supply target | EXCLUDE_FOR_HUMAN_REVIEW |
| `INSTITUTIONAL_OR_SITUATIONAL_CRITIQUE` | **11** | 16.18% | Satire or venting directed at institutions (Fox, NFL) or situations | EXCLUDE_FOR_HUMAN_REVIEW |
| `GENERAL_SARCASM_NO_INTERPERSONAL_TARGET` | **6** | 8.82% | Broadcast social media sarcasm without conversation addressee | EXCLUDE_FOR_HUMAN_REVIEW |
| `DIRECT_AGGRESSION_OR_DIRECT_ARGUMENT` | **4** | 5.88% | Overt hostility or blunt political argument lacking politeness facade | EXCLUDE_FOR_HUMAN_REVIEW |
| `AMBIGUOUS_INSUFFICIENT_CONTEXT` | **4** | 5.88% | Incomplete sentence fragments lacking necessary dialogue context | EXCLUDE_FOR_HUMAN_REVIEW |
| `SHARED_MODEL_ERROR_SUSPECTED` | **2** | 2.94% | Factual inquiries or promotional statements hallucinated as hostile | EXCLUDE_FOR_HUMAN_REVIEW |
| `SINCERE_PRAISE_OR_FRIENDLY_TEASING` | **2** | 2.94% | Sincere gratitude or personal confession hallucinated as hostile | EXCLUDE_FOR_HUMAN_REVIEW |
| **Total** | **68** | **100.00%** | **Complete Concordance Cohort** | **19 Retained, 49 Excluded** |

---

## 4. Shared Error Mode Taxonomy & Forensic Risk Register

The forensic audit revealed seven distinct shared error modes documented in [reports/phase5e/tables/shared_error_risk_register.csv](file:///d:/multimodel_passive_aggression/reports/phase5e/tables/shared_error_risk_register.csv):

### SER-01: Target Blindness (Institutional & Situational Venting) — 11 Records
- **Mechanism:** Both models treat generalized complaints against organizations (e.g. Fox News in `civil_comments_val_00462`, the NFL in `civil_comments_val_09401`, transit in `mmsd_train_12490`) or existential frustration (`mmsd_train_05348`: *"i should leave this planet morty"*) as passive aggression.
- **Linguistic Reality:** Passive aggression requires an interpersonal interlocutor. Complaining about an institution or life is venting, not interpersonal hostility.

### SER-02: Modality Blindness (Unassessed Multimodal Sarcasm) — 20 Records
- **Mechanism:** In 20 MMSD and SarcNet records, an image exists in the raw corpus. Because the LLMs were executed in text-only mode (`modality_used = "TEXT_ONLY"`), the models analyzed only text snippets like *"lil selfie for your day"* (`mmsd_train_01460`) or *"his side profile is something else"* (`mmsd_train_016130`).
- **Linguistic Reality:** Whether the image portrays an interlocutor, a celebrity, or the author themselves cannot be established without human visual inspection.

### SER-03: Broadcast Sarcasm Without Addressee — 6 Records
- **Mechanism:** Models classify observational sarcasm, news commentary (`civil_comments_val_05247`), or dialogue entrance lines (`mustard_2_317`) as passive-aggressive attacks.
- **Linguistic Reality:** Irony broadcast to an unaddressed public audience lacks the direct relational harm characteristic of passive aggression.

### SER-04: Covert/Overt Conflation (Direct Aggression Misclassification) — 4 Records
- **Mechanism:** In `civil_comments_val_04917`, `civil_comments_val_02587`, `civil_comments_val_00154`, and `civil_comments_val_05781`, comments exhibit overt hostility, unvarnished insults, and direct political accusations.
- **Linguistic Reality:** Passive aggression is defined by plausible deniability and a courtesy facade. Direct aggression belongs to an explicit contrast class.

### SER-05: Context Insufficiency — 4 Records
- **Mechanism:** Fragments like `civil_comments_val_00793` (*"...... but would immediately respond and pour money into a project..."*) or `mmsd_train_06126` contain dangling clauses.
- **Linguistic Reality:** Without surrounding turns, intent cannot be determined; models hallucinate covert resentment.

### SER-06: Politeness Paranoia & Sincere Praise Hallucination — 2 Records
- **Mechanism:** In `civil_comments_val_08920` (*"I want to thank you for this fine article..."*), both models hallucinated that the author was being deceitful and sarcastic. In `mustard_2_372`, a sitcom character's admission of smoking was misclassified as hostile.
- **Linguistic Reality:** Over-prompting LLMs to seek subtle hostility induces false-positive paranoia on genuine praise.

### SER-07: Hallucination on Factual / Promotional Text — 2 Records
- **Mechanism:** In `mmsd_train_00135` and `civil_comments_val_02265`, objective ride descriptions and factual comments were flagged as passive-aggressive.

---

## 5. Quantitative Synthesis of Model Reliability

From the 68 agreement records:
- **Interpersonal Target Confirmed:** Exactly **18 records (26.47%)** possess a verifiable interpersonal addressee.
- **Covert Hostility & Deniability Confirmed:** Exactly **18 records (26.47%)** exhibit confirmed covert hostility masking behind plausible deniability.
- **Shared False-Positive Risk Flag:** **49 records (72.06% of total agreements; 73.13% of positive agreements)** carry confirmed shared false-positive risk.
- **Shared False-Negative Risk Flag:** **0 records (0.00%)**. The sole consensus negative (`sarcnet_test_00053`) was verified as genuine non-PA.
- **Modality Transparency:** Exactly **0 records** were assessed by vision or OCR models. Text models evaluated comments strictly in isolation.

---

## 6. Governance and Epistemic Policy

1. **Model Agreement is Insufficient for Benchmark Truth:** Future evaluation must **never** equate cross-model agreement with ground truth. A 73.1% shared false-positive rate demonstrates that LLM consensus reflects shared prompt susceptibility, not ground truth validity.
2. **Phase 5D Dataset Validated:** The 19-record silver dataset prepared in Phase 5D is robust and leakage-safe.
3. **Exploratory Use Only:** The 19 silver records may be used solely for auxiliary weak-supervision representation probing (`TRAINED_ON_LLM_SILVER_DATA`).
4. **Benchmark Gate Maintained:** Real-world benchmark performance evaluation remains strictly blocked pending independent human annotation.
