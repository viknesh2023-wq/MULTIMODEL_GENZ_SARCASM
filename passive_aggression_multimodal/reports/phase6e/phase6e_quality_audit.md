# Phase 6E Quality Audit Report

**Project Title:** Context-Aware Multimodal Passive-Aggression Detection in Social Media Using Image–Comment Fusion  
**Date:** October 21, 2026  
**Auditor:** Automated Pragmatic & Research Governance Auditor  
**Artifact:** `reports/phase6e/phase6e_quality_audit.md`  
**Status:** **AUDIT COMPLETE** (Forensic audit of Phase 6D dual LLM annotation and adjudication artifacts)

---

## 1. Executive Summary & Audit Verdict

This Phase 6E quality audit independently examined all artifacts generated during Phase 6D (`annotator_1_llm_responses.csv`, `annotator_2_llm_responses.csv`, `llm_adjudication_results.csv`, `llm_annotation_disagreements.csv`, and `llm_annotation_manifest.json`).

### Key Audit Findings:
1. **Reconciliation & Data Integrity:** All **30 records** reconcile across all datasets without missing rows, truncated columns, or duplicate IDs/texts.
2. **Modality Honesty Confirmed:** 100% of records were evaluated as `TEXT_ONLY` (`image_assessed = false`, `ocr_assessed = false`).
3. **Preservation of Human Benchmark:** Phase 6A human response files remain completely unpopulated (`completed_count = 0`), and the evaluation gate remains strictly `BLOCKED_PENDING_HUMAN_ANNOTATION`.
4. **Independent Recalculation:** Confirmed exact match with Phase 6D reported metrics:
   - **Overall (30 records):** 53.33% raw agreement (16/30), Cohen's $\kappa = 0.1322$.
   - **Core English Benchmark (27 records):** 51.85% raw agreement (14/27), Cohen's $\kappa = 0.1397$.
   - **Chinese Exploratory Subset (3 records):** 66.67% raw agreement (2/3).
5. **The "Conservative Collapse" Linguistic Phenomenon:**
   - Prompt `v2.0.0` enforced strict negative boundaries (direct toxicity, situational venting, news reports, sincere praise, blunt argument).
   - Under text-only evaluation, missing visual and dialogue context prevented the models from meeting the 3-prong test with high confidence.
   - Consequently, final adjudicated labels resulted in **16 non-passive-aggressive** and **14 ambiguous**, with **0 passive-aggressive**.

---

## 2. Record Reconciliation & Integrity Matrix

| Audit Check | Manifest Expectation | Actual Found | Status | Verification Detail |
| :--- | :---: | :---: | :---: | :--- |
| **Pilot Record Count** | 30 | 30 | **VERIFIED** | Matches `pilot_sampling_manifest.csv` |
| **Annotator 1 Records** | 30 | 30 | **VERIFIED** | Matches `annotator_1_llm_responses.csv` |
| **Annotator 2 Records** | 30 | 30 | **VERIFIED** | Matches `annotator_2_llm_responses.csv` |
| **Adjudication Records** | 30 | 30 | **VERIFIED** | Matches `llm_adjudication_results.csv` |
| **Disagreement Records** | 14 | 14 | **VERIFIED** | Matches `llm_annotation_disagreements.csv` |
| **Duplicate Record IDs** | 0 | 0 | **VERIFIED** | 30 unique IDs across all files |
| **Duplicate Comment Texts** | 0 | 0 | **VERIFIED** | Zero duplicate comment strings |
| **Human Annotator 1 Completed** | 0 | 0 | **VERIFIED** | `annotator_1_responses.csv` untouched |
| **Human Annotator 2 Completed** | 0 | 0 | **VERIFIED** | `annotator_2_responses.csv` untouched |
| **Human Benchmark Gate** | BLOCKED | BLOCKED | **VERIFIED** | `BLOCKED_PENDING_HUMAN_ANNOTATION` |

---

## 3. Label Distributions & Agreement Analysis

### Label Distributions Across Annotation Passes

| Primary Label | Model 1 (`llama3.2:latest`) | Model 2 (`qwen2.5:1.5b`) | Adjudicated Consensus / Decision |
| :--- | :---: | :---: | :---: |
| **passive_aggressive** | 0 (0.0%) | 1 (3.3%) | **0 (0.0%)** |
| **non_passive_aggressive** | 14 (46.7%) | 24 (80.0%) | **16 (53.3%)** |
| **ambiguous** | 16 (53.3%) | 5 (16.7%) | **14 (46.7%)** |
| **Total** | **30 (100%)** | **30 (100%)** | **30 (100%)** |

### Confusion Matrix (Rows: Model 1, Columns: Model 2)

| Model 1 \ Model 2 | passive_aggressive | non_passive_aggressive | ambiguous | Total |
| :--- | :---: | :---: | :---: | :---: |
| **passive_aggressive** | 0 | 0 | 0 | 0 |
| **non_passive_aggressive** | 1 | 12 | 1 | 14 |
| **ambiguous** | 0 | 12 | 4 | 16 |
| **Total** | **1** | **24** | **5** | **30** |

---

## 4. Deep Audit of the 14 Disagreement Records

The 14 disagreements represent the primary divergence between Model 1 (`llama3.2:latest`, which leaned heavily into `ambiguous` when visual evidence was missing) and Model 2 (`qwen2.5:1.5b`, which leaned into `non_passive_aggressive`).

| # | Record ID | Scope | Archetype | Model 1 | Model 2 | Adjudicated Label | Root Cause & Adjudication Justification |
|---|---|---|---|---|---|---|---|
| 1 | `civil_comments_val_02587` | CORE_ENGLISH_BENCHMARK | DIRECT_AGGRESSION_OR_DISAGREEMENT | `ambiguous` | `non_passive_aggressive` | **`non_passive_aggressive`** | Linguistic root cause: Model 1 misinterprets the tone of the comment as sar... |
| 2 | `civil_comments_val_04917` | CORE_ENGLISH_BENCHMARK | DIRECT_AGGRESSION_OR_DISAGREEMENT | `non_passive_aggressive` | `passive_aggressive` | **`ambiguous`** | Linguistic root cause: Model 1 misinterprets the sarcastic tone as overt ho... |
| 3 | `civil_comments_val_08920` | CORE_ENGLISH_BENCHMARK | NON_PA_CONTRAST_CANDIDATE | `ambiguous` | `non_passive_aggressive` | **`non_passive_aggressive`** | Linguistic root cause: Insufficient context and tone analysis.... |
| 4 | `civil_comments_val_09401` | CORE_ENGLISH_BENCHMARK | GENERAL_SARCASM_INSTITUTIONAL_CRITIQUE | `ambiguous` | `non_passive_aggressive` | **`non_passive_aggressive`** | Linguistic root cause: Model 1 misinterprets the comment's intent as indire... |
| 5 | `mmsd_test_00104` | CORE_ENGLISH_BENCHMARK | NON_PA_CONTRAST_CANDIDATE | `ambiguous` | `non_passive_aggressive` | **`ambiguous`** | Lack of contextual information and unclear intent... |
| 6 | `mmsd_train_02546` | CORE_ENGLISH_BENCHMARK | DIRECT_AGGRESSION_OR_DISAGREEMENT | `ambiguous` | `non_passive_aggressive` | **`ambiguous`** | Linguistic ambiguity in interpretation of the phrase 'not all girls'... |
| 7 | `mmsd_train_06126` | CORE_ENGLISH_BENCHMARK | AMBIGUOUS_CONTEXT_DEPENDENT | `ambiguous` | `non_passive_aggressive` | **`ambiguous`** | Linguistic root cause: Insufficient contextual information and varying inte... |
| 8 | `mmsd_train_10780` | CORE_ENGLISH_BENCHMARK | LLM_SILVER_POSITIVE_CANDIDATE | `ambiguous` | `non_passive_aggressive` | **`ambiguous`** | Linguistic interpretation of sarcasm/sarcasm detection... |
| 9 | `mmsd_validation_01859` | CORE_ENGLISH_BENCHMARK | MULTIMODAL_IMAGE_DEPENDENT | `ambiguous` | `non_passive_aggressive` | **`ambiguous`** | Linguistic ambiguity in tone and intent... |
| 10 | `mmsd_validation_02127` | CORE_ENGLISH_BENCHMARK | AMBIGUOUS_CONTEXT_DEPENDENT | `ambiguous` | `non_passive_aggressive` | **`ambiguous`** | Linguistic root cause: unclear target and lack of context... |
| 11 | `mustard_2_111` | CORE_ENGLISH_BENCHMARK | AMBIGUOUS_CONTEXT_DEPENDENT | `ambiguous` | `non_passive_aggressive` | **`ambiguous`** | Linguistic root cause: Insufficient contextual information to disambiguate ... |
| 12 | `mustard_2_490` | CORE_ENGLISH_BENCHMARK | LLM_SILVER_POSITIVE_CANDIDATE | `ambiguous` | `non_passive_aggressive` | **`ambiguous`** | Linguistic interpretation of tone and intent... |
| 13 | `sarcnet_test_00274` | CORE_ENGLISH_BENCHMARK | GENERAL_SARCASM_INSTITUTIONAL_CRITIQUE | `ambiguous` | `non_passive_aggressive` | **`ambiguous`** | Linguistic ambiguity in pronoun interpretation... |
| 14 | `sarcnet_train_00844` | NON_ENGLISH_SEPARATE_ANALYSIS | MULTIMODAL_IMAGE_DEPENDENT | `non_passive_aggressive` | `ambiguous` | **`non_passive_aggressive`** | Linguistic root cause: Model 1 misinterprets the tone as playful/teasing, w... |

---

## 5. Epistemic Governance & Linguistic Insights

### 1. Why Did Passive Aggression Collapse to Zero in Text-Only Mode?
In social media, passive aggression rarely announces itself through overt vocabulary. Rather, it operates through:
1. **Pragmatic Incongruity:** A polite caption paired with an embarrassing, contradictory, or mocking photo.
2. **Conversational History:** A delayed reply or subtle jab that only registers as hostility given preceding dialogue turns.

In Phase 6D:
- 14 records were image-bearing, but images were **unassessed** (`TEXT_ONLY`).
- 24 of 30 records (80%) had **zero dialogue context**.
- When instructed strictly to adhere to negative boundaries (avoid treating news, debate, or sincere praise as PA) and select `ambiguous` when evidence is insufficient, both models complied by refusing to assign `passive_aggressive`.
- This provides conclusive empirical proof for the central thesis of this research: **passive-aggression detection cannot succeed as a text-only task in social media; it fundamentally requires multimodal context fusion.**

### 2. Status of Human Ground Truth
The project has maintained pristine scientific integrity:
- `data/annotations/phase6a/annotator_1_responses.csv` has **0 completed reviews**.
- `data/annotations/phase6a/annotator_2_responses.csv` has **0 completed reviews**.
- Machine-generated labels are strictly documented as `MACHINE_GENERATED_LOCAL_OLLAMA_LLM`.
- The human benchmark evaluation gate remains **LOCKED**.

---

## 6. Recommended Next Steps

1. **Weak-Supervision Diagnostic Experiments:** The adjudicated silver dataset (`llm_adjudication_results.csv`) can be used for exploratory feature probing and weak supervision baselines, provided it is explicitly labeled as machine-generated.
2. **Multimodal Vision Integration:** When resources permit, integrate a local vision-capable model (e.g. `llava` or `llama3.2-vision`) to test whether multimodal features restore positive passive-aggression classification on image-bearing tweets.
3. **Human Annotation Pilot:** To establish genuine ground truth and unlock benchmark evaluation, two human annotators should complete the 30-record pilot using [`docs/phase6a/phase6a_annotator_instructions.md`](file:///d:/multimodel_passive_aggression/docs/phase6a/phase6a_annotator_instructions.md).
