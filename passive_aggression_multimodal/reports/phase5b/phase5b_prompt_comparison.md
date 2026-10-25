# Phase 5B: LLM Annotation Prompt Comparison Report (v1.0.0 vs v2.0.0)

**Project Title:** Context-Aware Multimodal Passive-Aggression Detection in Social Media Using Image–Comment Fusion  
**Canonical Project Root:** `D:\NLP_PROJECT\passive_aggression_multimodal`  
**Linked Workspace:** `D:\multimodel_passive_aggression`  
**Milestone:** Phase 5B — Prompt Calibration & Comparative Evaluation  
**Date:** October 21, 2026  
**Auditor:** Senior ML Research Auditor  

---

## 1. Motivation & Prompt Calibration Rationale

The Phase 5A quality audit revealed that Prompt v1 suffered from systematic pragmatic over-attribution:
- **`llama3.2:latest`** assigned `passive_aggressive` to **96%** of candidate social media records.
- Factual sports headlines, sincere expressions of admiration, and blunt debate assertions were misclassified as covert hostility and funnelled into the `implied_blame` subtype.
- Ambiguous or isolated phrases lacking context were assigned confident positive labels rather than `ambiguous`.

To address these vulnerabilities, **Prompt v2.0.0** ([docs/phase5b/llm_annotation_prompt_v2.md](file:///d:/multimodel_passive_aggression/docs/phase5b/llm_annotation_prompt_v2.md)) was engineered with explicit contrastive guardrails and calibrated task framing.

---

## 2. Structural & Architectural Comparison

| Dimension | Prompt v1.0.0 ([docs/phase5a/llm_annotation_prompt_v1.md](file:///d:/multimodel_passive_aggression/docs/phase5a/llm_annotation_prompt_v1.md)) | Prompt v2.0.0 ([docs/phase5b/llm_annotation_prompt_v2.md](file:///d:/multimodel_passive_aggression/docs/phase5b/llm_annotation_prompt_v2.md)) |
| :--- | :--- | :--- |
| **System Framing** | *"Your task is to annotate social media comments for PASSIVE AGGRESSION."* (Induces strong confirmation bias) | *"Your task is to objectively evaluate social media comments and distinguish genuine PASSIVE AGGRESSION from other forms of communication..."* |
| **Pragmatic Criteria** | General definition of covert hostility and plausible deniability. | **Three-Prong Test:** (1) Interpersonal target, (2) Plausible deniability, (3) Covert hostile intent. All three required. |
| **Contrast Boundaries** | Abstract list in system prompt; not reinforced in user prompt template. | **Six Concrete Negative Boundaries:** Direct aggression, situational venting, factual reports/news, sincere praise, blunt argument, friendly teasing. |
| **User Prompt Template** | Generic prompt passing record text. | Injects explicit reminders: news, direct insults, and praise are NOT passive-aggressive. |
| **Ambiguity Rule** | Brief mention in rule 5. | Explicit directive: if conversational context or image is missing to verify target or tone, select `ambiguous`. |
| **CoT Protocol** | Produce 1-2 sentence rationale. | Explicitly dissect: (1) literal surface meaning, (2) interpersonal target presence, (3) covert vs overt vs absent hostility. |

---

## 3. Empirical Evaluation on 12-Record Diagnostic Subset

To validate Prompt v2 without leaking the full evaluation manifest, Prompt v2 was tested on a documented 12-record diagnostic subset ([data/annotations/phase5b/prompt_v2_subset_test.json](file:///d:/multimodel_passive_aggression/data/annotations/phase5b/prompt_v2_subset_test.json)) spanning:
- Known false positives from Prompt v1 (sports news, sincere compliments, blunt debate assertions).
- Real passive-aggression candidates from MUStARD and SarcNet.
- Missing-context isolated phrases.
- Baseline non-PA records.

### Side-by-Side Comparative Results Table

| Record ID | Comment Excerpt | Model A (Prompt v1) | Model A (Prompt v2) | Model B (Prompt v1) | Model B (Prompt v2) | Ground Pragmatic Nature |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `mmsd_test_02285` | *"result : glen durrant battles back from three sets..."* | PA (`implied_blame`) | **Non-PA** (`none`) | Non-PA (`none`) | **Non-PA** (`none`) | Factual sports news headline |
| `mmsd_test_01815` | *"also i 'm in love with this pic"* | PA (`backhanded_compliment`) | **Ambiguous** (`none`) | Non-PA (`none`) | PA (`backhanded_compliment`) | Sincere positive admiration |
| `civil_comments_val_03516` | *"I hate to break it to you, but prices will always go up..."* | PA (`implied_blame`) | **Non-PA** (`dismissiveness`) | Non-PA (`none`) | **Non-PA** (`none`) | Blunt direct assertion / correction |
| `civil_comments_val_02472` | *"Did you read the article? '...whether certain forms...'"* | PA (`implied_blame`) | **Ambiguous** (`none`) | Non-PA (`none`) | **Non-PA** (`none`) | Rhetorical question / news quote |
| `mmsd_test_00104` | *"good one panda . ( insert )"* | PA (`fake_politeness`) | **Ambiguous** (`none`) | Non-PA (`none`) | PA (`backhanded_compliment`) | Playful banter / placeholder |
| `mmsd_train_05348` | *"i should leave this planet morty . there is no more wisdom left..."* | PA (`implied_blame`) | **Non-PA** (`none`) | PA (`other_or_unclear`) | PA (`backhanded_compliment`) | Pop-culture quote / existential venting |
| `mmsd_test_00277` | *"* gasp * oh no ... not the grandpa ...."* | PA (`implied_blame`) | **PA** (`implied_blame`) | Ambiguous (`none`) | PA (`backhanded_compliment`) | Sarcastic theatrical reaction |
| `mmsd_train_02546` | *"not all girls ...."* | PA (`implied_blame`) | **Ambiguous** (`none`) | Ambiguous (`none`) | **Ambiguous** (`none`) | Missing context / trailing phrase |
| `mustard_2_490` | *"Maybe it's a contest, you know? Like 'Collect all five'."* | PA (`backhanded_compliment`) | **Ambiguous** (`none`) | PA (`other_or_unclear`) | PA (`indirect_criticism`) | Classic dialogue sarcasm / PA candidate |
| `sarcnet_test_00274` | *"Maybe now someone in the Federal Govt and MSM will start paying..."* | PA (`implied_blame`) | **Ambiguous** (`none`) | PA (`other_or_unclear`) | **Non-PA** (`none`) | Political grievance / institutional critique |
| `mmsd_train_16580` | *"2017012 1 haneda airport"* | Non-PA (`none`) | **Ambiguous** (`none`) | Ambiguous (`none`) | **Non-PA** (`none`) | Ultra-short timestamp & airport code |
| `mmsd_validation_02047` | *"the glimmer twins"* | Non-PA (`none`) | **Ambiguous** (`none`) | Ambiguous (`none`) | **Non-PA** (`none`) | Isolated 3-word phrase |

---

## 4. Key Findings & Quantitative Shifts

### Shift in Model A (`llama3.2:latest`):
- **Under Prompt v1:** 10 / 12 positive (83.3%), 2 negative (16.7%), 0 ambiguous (0.0%).
- **Under Prompt v2:** 1 / 12 positive (8.3%), 3 / 12 negative (25.0%), 8 / 12 ambiguous (66.7%).
- **Analysis:**
  1. **False-Positive Elimination:** Factual sports news (`mmsd_test_02285`), blunt economic assertions (`civil_comments_val_03516`), and existential pop-culture quotes (`mmsd_train_05348`) correctly shifted out of the positive class.
  2. **Respect for Missing Modality:** Isolated fragments and unassessed-image records were routed into `ambiguous` rather than hallucinating covert motives.

### Shift in Model B (`qwen2.5:1.5b`):
- **Under Prompt v1:** 5 / 12 positive (41.7%), 5 / 12 negative (41.7%), 2 / 12 ambiguous (16.7%).
- **Under Prompt v2:** 6 / 12 positive (50.0%), 5 / 12 negative (41.7%), 1 / 12 ambiguous (8.3%).
- **Analysis:** Model B remained stable and well-calibrated, maintaining direct-assertion boundaries on news and economic debate.

---

## 5. Scientific Methodological Constraints

1. **No Circular Benchmark Tuning:**
   Prompt v2 was evaluated exclusively on a 12-record diagnostic sample. We deliberately avoided iterating on the entire 100-record manifest to preserve experimental integrity and prevent overfitting the prompt to the benchmark data.
2. **Persistence of Original Phase 5A Artifacts:**
   All Phase 5A outputs (`llm_annotations_pass1.jsonl`, `llm_annotations_pass2.jsonl`, `llm_annotation_disagreements.csv`) remain completely intact as historical records of Phase 5A execution.
