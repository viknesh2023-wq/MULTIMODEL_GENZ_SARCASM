# Phase 5D: Silver Label Provenance & Data Lineage Report

**Project Title:** Context-Aware Multimodal Passive-Aggression Detection in Social Media Using Image–Comment Fusion  
**Canonical Project Root:** `D:\NLP_PROJECT\passive_aggression_multimodal`  
**Linked Workspace:** `D:\multimodel_passive_aggression`  
**Milestone:** Phase 5D — Silver Dataset Validation & Weak-Supervision Data Lineage  
**Date:** October 21, 2026  
**Auditor:** Senior ML Research Auditor & Data Governance Team  

---

## 1. Objective and Provenance Scope

This report provides the full forensic data lineage, provenance specification, and integrity tracking for all records evaluated and retained in Phase 5D. 

Under the experimental guidelines established in Phases 4, 5A, 5B, and 5C:
1. **Silver Labels are NOT Ground Truth:** LLM annotations are pseudo-labels produced by text-only models (`llama3.2:latest` and `qwen2.5:1.5b`) without access to visual or OCR context.
2. **Strict Provenance Tagging:** Any record ingested into downstream exploratory weak-supervision experiments must carry the immutable tag:
   ```json
   "provenance_tag": "TRAINED_ON_LLM_SILVER_DATA"
   ```
3. **Evaluation Quarantine:** Silver-labeled records must never enter or contaminate test, validation, or human-evaluation benchmark splits. Real-world benchmark performance evaluation remains strictly blocked pending human adjudication.

---

## 2. End-to-End Data Lineage Chain

The lineage of each silver-labeled record follows a deterministic 6-stage chain:

```
[Raw Multi-Source Corpus] (MUStARD, MMSD, SarcNet, CivilComments)
         │
         ▼
[Phase 4 Sampling Manifest] (100 stratified records; human_sampling_manifest_100.jsonl)
         │
         ▼
[Phase 5A Dual-LLM Annotation] (Pass 1: llama3.2:latest, Pass 2: qwen2.5:1.5b via prompt v1)
         │
         ▼
[Phase 5B Inter-Rater Audit] (68 exact agreements, 32 disagreements identified)
         │
         ▼
[Phase 5C Forensic Deep Audit] (Taxonomy breakdown; institutional venting, pop-culture venting flagged)
         │
         ▼
[Phase 5D Strict Verification] (Pragmatic hostility + addressee audit; zero near-duplicate leakage)
         │
         ├──► 19 Verified Silver Records (silver_dataset_verified.jsonl)
         ├──► 81 Excluded Records (excluded_records.csv)
         └──► 32 Audited Negative Candidates (negative_candidate_audit.csv)
```

---

## 3. Provenance Schema & Metadata Contract

Every record retained in [data/annotations/phase5d/silver_dataset_verified.jsonl](file:///d:/multimodel_passive_aggression/data/annotations/phase5d/silver_dataset_verified.jsonl) strictly adheres to the following JSON schema:

| Field Name | Type | Description | Mandatory Value / Constraint |
| :--- | :--- | :--- | :--- |
| `record_id` | String | Stable record identifier | Matches Phase 4 manifest `record_id` |
| `source_dataset` | String | Originating research corpus | One of `mustard`, `mmsd`, `sarcnet`, `civil_comments` |
| `source_record_id` | String | Original ID within the raw dataset | e.g. `2_490`, `val_06315`, `validation_01859` |
| `text` | String | Raw comment / utterance text | Verbatim original text |
| `context` | String/List | Dialogue turns / thread context | Preserved conversation context |
| `model_a_label` | String | Prediction from `llama3.2:latest` | `passive_aggressive` or `non_passive_aggressive` |
| `model_b_label` | String | Prediction from `qwen2.5:1.5b` | `passive_aggressive` or `non_passive_aggressive` |
| `consensus_exact_match` | Boolean | Agreement on primary label | `true` for all verified records |
| `silver_primary_label` | String | Final audited silver label | `passive_aggressive` (18) or `non_passive_aggressive` (1) |
| `silver_subtype` | String | Audited passive-aggression subtype | `backhanded_compliment`, `implied_blame`, `indirect_criticism`, `none` |
| `audit_decision` | String | Formal audit disposition | `VERIFIED_SILVER_POSITIVE` or `VERIFIED_SILVER_NEGATIVE` |
| `audit_reason` | String | Pragmatic justification | Explains confirmed addressee, covert hostility, and plausible deniability |
| `provenance_tag` | String | Downstream experiment tag | Strictly `"TRAINED_ON_LLM_SILVER_DATA"` |
| `annotation_models` | List[Str] | LLM models used in consensus | `["llama3.2:latest", "qwen2.5:1.5b"]` |
| `prompt_version` | String | Prompt specification version | `"v1"` (Phase 5A prompt) |
| `modality_used` | String | Modality assessed by LLM | Strictly `"TEXT_ONLY"` |
| `image_assessed` | Boolean | Vision model participation flag | Strictly `false` (no vision LLM executed) |
| `ocr_assessed` | Boolean | OCR text assessed by LLM | Strictly `false` |
| `leakage_check_passed` | Boolean | Deduplication & split integrity | Strictly `true` |

---

## 4. Itemized Provenance Ledger (All 19 Verified Records)

| Record ID | Source Dataset | Source Record ID | Verified Silver Label | Subtype | Models & Prompt | Audit Disposition |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `mustard_2_490` | `mustard` | `2_490` | `passive_aggressive` | `backhanded_compliment` | `llama3.2` + `qwen2.5` (v1) | `VERIFIED_SILVER_POSITIVE` |
| `civil_comments_val_06315` | `civil_comments` | `val_06315` | `passive_aggressive` | `backhanded_compliment` | `llama3.2` + `qwen2.5` (v1) | `VERIFIED_SILVER_POSITIVE` |
| `mustard_1_3660` | `mustard` | `1_3660` | `passive_aggressive` | `backhanded_compliment` | `llama3.2` + `qwen2.5` (v1) | `VERIFIED_SILVER_POSITIVE` |
| `mustard_1_8407` | `mustard` | `1_8407` | `passive_aggressive` | `implied_blame` | `llama3.2` + `qwen2.5` (v1) | `VERIFIED_SILVER_POSITIVE` |
| `mustard_2_439` | `mustard` | `2_439` | `passive_aggressive` | `implied_blame` | `llama3.2` + `qwen2.5` (v1) | `VERIFIED_SILVER_POSITIVE` |
| `mustard_2_77` | `mustard` | `2_77` | `passive_aggressive` | `implied_blame` | `llama3.2` + `qwen2.5` (v1) | `VERIFIED_SILVER_POSITIVE` |
| `civil_comments_val_07561` | `civil_comments` | `val_07561` | `passive_aggressive` | `backhanded_compliment` | `llama3.2` + `qwen2.5` (v1) | `VERIFIED_SILVER_POSITIVE` |
| `mmsd_validation_01859` | `mmsd` | `validation_01859` | `passive_aggressive` | `implied_blame` | `llama3.2` + `qwen2.5` (v1) | `VERIFIED_SILVER_POSITIVE` |
| `sarcnet_train_01587` | `sarcnet` | `train_01587` | `passive_aggressive` | `indirect_criticism` | `llama3.2` + `qwen2.5` (v1) | `VERIFIED_SILVER_POSITIVE` |
| `civil_comments_val_06377` | `civil_comments` | `val_06377` | `passive_aggressive` | `indirect_criticism` | `llama3.2` + `qwen2.5` (v1) | `VERIFIED_SILVER_POSITIVE` |
| `civil_comments_val_07107` | `civil_comments` | `val_07107` | `passive_aggressive` | `indirect_criticism` | `llama3.2` + `qwen2.5` (v1) | `VERIFIED_SILVER_POSITIVE` |
| `civil_comments_val_08072` | `civil_comments` | `val_08072` | `passive_aggressive` | `implied_blame` | `llama3.2` + `qwen2.5` (v1) | `VERIFIED_SILVER_POSITIVE` |
| `mmsd_train_10780` | `mmsd` | `train_10780` | `passive_aggressive` | `implied_blame` | `llama3.2` + `qwen2.5` (v1) | `VERIFIED_SILVER_POSITIVE` |
| `mustard_1_162` | `mustard` | `1_162` | `passive_aggressive` | `implied_blame` | `llama3.2` + `qwen2.5` (v1) | `VERIFIED_SILVER_POSITIVE` |
| `mustard_2_9` | `mustard` | `2_9` | `passive_aggressive` | `implied_blame` | `llama3.2` + `qwen2.5` (v1) | `VERIFIED_SILVER_POSITIVE` |
| `mustard_2_574` | `mustard` | `2_574` | `passive_aggressive` | `implied_blame` | `llama3.2` + `qwen2.5` (v1) | `VERIFIED_SILVER_POSITIVE` |
| `sarcnet_train_00844` | `sarcnet` | `train_00844` | `passive_aggressive` | `implied_blame` | `llama3.2` + `qwen2.5` (v1) | `VERIFIED_SILVER_POSITIVE` |
| `mmsd_test_02238` | `mmsd` | `test_02238` | `passive_aggressive` | `backhanded_compliment` | `llama3.2` + `qwen2.5` (v1) | `VERIFIED_SILVER_POSITIVE` |
| `sarcnet_test_00053` | `sarcnet` | `test_00053` | `non_passive_aggressive` | `none` | `llama3.2` + `qwen2.5` (v1) | `VERIFIED_SILVER_NEGATIVE` |

---

## 5. Source Dataset and Modality Coverage

### 5.1 Source Dataset Distribution:
- **MUStARD:** 8 records (42.1%) — Sitcom conversational dialogues with rich multi-turn context (The Big Bang Theory, Friends, The Golden Girls).
- **CivilComments:** 5 records (26.3%) — News thread reader-to-reader comments featuring polite rhetorical rebukes and handle-directed backhanded compliments.
- **MMSD:** 3 records (15.8%) — Social media tweets featuring backhanded evaluations and sarcastic compliance.
- **SarcNet:** 3 records (15.8%) — 2 Chinese social media Weibo posts demonstrating weaponized politeness and feigned ignorance; 1 neutral idiom explanation.

### 5.2 Modality Transparency:
- **Text Utterance:** 100% available and verified.
- **Dialogue Context:** Available for MUStARD (100%) and CivilComments (parent comments available).
- **Image Modality:** Image files exist for MMSD and SarcNet, but were **explicitly unassessed** during LLM annotation (`modality_used = "TEXT_ONLY"`, `image_assessed = false`).
- **OCR Modality:** OCR text exists in the raw multimodal corpus, but was **not ingested** into the text prompt during silver labeling (`ocr_assessed = false`).

---

## 6. Leakage & Near-Duplicate Analysis

To prevent data contamination across splits, every candidate underwent automated and manual integrity checks:
1. **Record ID Collision Test:**
   - Evaluated 19 verified records + 81 excluded records = 100 unique record IDs.
   - Result: 0 collisions.
2. **Exact Text Deduplication:**
   - Evaluated string equality across all 19 verified utterances.
   - Result: 19 unique strings (0 duplicates).
3. **Normalized Near-Duplicate Analysis:**
   - Texts were normalized via case-folding, regex stripping of all punctuation/whitespace, and character tokenization.
   - Result: 19 distinct normalized sequences (0 near-duplicate collisions).
4. **Cross-Split Quarantine:**
   - The silver dataset is stored exclusively in `data/annotations/phase5d/`.
   - The human annotation sampling pool remains in `data/annotations/phase4/human_sampling_manifest_100.jsonl` and `data/annotations/phase5/human_response_collection_template.csv`.
   - Human review status remains at `completed_count = 0` (`BLOCKED_PENDING_HUMAN_ANNOTATION`).

---

## 7. Audit Exclusions and Negative Candidate Analysis

### 7.1 Excluded Records Taxonomy (81 Records):
The 81 excluded records are documented in [data/annotations/phase5d/excluded_records.csv](file:///d:/multimodel_passive_aggression/data/annotations/phase5d/excluded_records.csv):
- **Cross-Model Disagreements (`EXCLUDED_DISAGREEMENT`, 32 records):**  
  Model A (`llama3.2`) and Model B (`qwen2.5`) disagreed on the primary label. Rather than resolving by arbitrary tie-break, all 32 records are preserved in the prioritized queue for human adjudication.
- **Overt Direct Aggression (`EXCLUDED_OVERT_AGGRESSION`, 8 records):**  
  Comments containing explicit vulgarity, unvarnished insults, or direct hostility without plausible deniability (e.g. *"You're a clown"*, *"shut up"*).
- **Institutional / Political Venting (`EXCLUDED_INSTITUTIONAL_VENTING`, 14 records):**  
  Sarcasm directed at abstract entities, sports franchises, news networks, or government bodies (e.g. NFL officiating, Fox News).
- **Existential / Pop-Culture Venting (`EXCLUDED_EXISTENTIAL_VENTING`, 6 records):**  
  General comedic sarcasm or cultural quotations without an interpersonal target (e.g. Rick and Morty references).
- **Direct Debate (`EXCLUDED_DIRECT_DEBATE`, 5 records):**  
  Blunt, non-covert political arguments or public fee inquiries lacking courtesy facades.
- **Insufficient Context (`EXCLUDED_INSUFFICIENT_CONTEXT`, 4 records):**  
  Ambiguous sentence fragments where target and intent cannot be resolved.
- **Unverified Target (`EXCLUDED_UNVERIFIED_TARGET`, 10 records):**  
  Sarcastic comments where the intended recipient cannot be distinguished between self-deprecation, generic humor, or third-party targets.
- **Sitcom Confession (`EXCLUDED_SELF_CONFESSION`, 1 record):**  
  Character self-disclosure lacking aggressive intent.
- **Shared False Praise Hallucination (`EXCLUDED_SHARED_FALSE_POSITIVE_PRAISE`, 1 record):**  
  `civil_comments_val_08920` (*"I want to thank you for this fine article..."*). Both models hallucinated passive aggression on sincere gratitude; eliminated during human auditor inspection.

### 7.2 Negative Candidates Audit (32 Records):
Documented in [data/annotations/phase5d/negative_candidate_audit.csv](file:///d:/multimodel_passive_aggression/data/annotations/phase5d/negative_candidate_audit.csv):
- 28 Model B negative disagreements.
- 3 Model A negative disagreements.
- 1 consensus negative (`sarcnet_test_00053`).
- **Audit Conclusion:** Neutral sentiment or zero toxicity is not evidence of non-passive-aggression. 31 candidates remain `UNVERIFIED_NEGATIVE_CANDIDATE` and are excluded from the verified subset.

---

## 8. Governance and Usage Policy for Downstream Phases

Any future exploratory experiment utilizing this silver dataset must adhere to the following rules:

1. **Tag Retention:** The field `"provenance_tag": "TRAINED_ON_LLM_SILVER_DATA"` must remain attached to all feature representations and training logs.
2. **No Claim of Real-World Generalization:** Any exploratory weak-supervision loss or alignment probe must explicitly report that training occurred on 19 silver-labeled examples and that generalization to real-world social media passive aggression is unproven.
3. **No Evaluation Benchmark Usage:** Under no circumstances may `silver_dataset_verified.jsonl` be used as a test or validation benchmark. Evaluation on real data remains blocked pending human annotation completion.
4. **Vision Modality Neutrality:** Because silver labels were generated via text-only LLMs, multimodal fusion heads trained with silver data must not claim validated image-text synergy until verified by multimodal human annotations.
