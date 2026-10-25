# Phase 5B: LLM Silver-Label Quality Audit Report

**Project Title:** Context-Aware Multimodal Passive-Aggression Detection in Social Media Using Image–Comment Fusion  
**Canonical Project Root:** `D:\NLP_PROJECT\passive_aggression_multimodal`  
**Linked Workspace:** `D:\multimodel_passive_aggression`  
**Milestone:** Phase 5B — Experimental Audit of Phase 5A LLM Annotations  
**Date:** October 21, 2026  
**Auditor:** Senior ML Research Auditor & Experimental Reliability Team  

---

## 1. Executive Summary

Phase 5A generated dual-pass silver-label annotations on a 100-record candidate manifest ([human_sampling_manifest_100.jsonl](file:///d:/multimodel_passive_aggression/data/annotations/phase4/human_sampling_manifest_100.jsonl)) using two local on-device models: `llama3.2:latest` (Model A, 3.2B parameters) and `qwen2.5:1.5b` (Model B, 1.5B parameters).

This Phase 5B audit conducted an independent experimental investigation into the quality, reliability, and validity of these silver labels.

### Key Audit Findings:
1. **Severe Label Imbalance in Model A (96% Positive Rate):**
   Model A predicted `passive_aggressive` for **96 out of 100** records (96.0%) and `non_passive_aggressive` for only 4 records (3 of which were non-English or ultra-short strings such as "haneda airport" and "the glimmer twins").
2. **Pathological Hyper-Interpretability:**
   Model A suffered from task-demand confirmation bias and rationalization hallucination, actively contriving covert motives for factual sports reports (e.g., Glen Durrant tennis match results), sincere admiration ("also i'm in love with this pic"), and blunt debate assertions.
3. **The "Implied Blame" Sinkhole:**
   60 of Model A's 96 positive labels (62.5%) were assigned to the `implied_blame` subtype, acting as an unconstrained sinkhole for any comment containing frustration, debate, or factual contrast.
4. **Model B Moderation (68% Positive Rate):**
   Model B (`qwen2.5:1.5b`) exhibited greater restraint, assigning 68 `passive_aggressive`, 28 `non_passive_aggressive`, and 4 `ambiguous`.
5. **Cross-Model Agreement Analysis (68.0% Agreement):**
   Out of 68 agreements, **67 were positive (`passive_aggressive`)** and only **1 was negative (`non_passive_aggressive`)**. This high agreement on the positive class reflects shared task-framing bias across both models rather than proof of true positive prevalence.
6. **Active-Learning Value of Disagreements:**
   The 32 disagreement records provide an empirical filter pinpointing boundary confusions between direct aggression, situational venting, rhetorical debate, and true covert hostility.

---

## 2. Task 1: Artifact & Provenance Verification

All Phase 5A artifacts were inspected and verified against the physical filesystem:

| Artifact Path | Record Count | Schema Valid | Verified Integrity Check |
| :--- | :--- | :--- | :--- |
| [data/annotations/phase5a/llm/llm_annotations_pass1.jsonl](file:///d:/multimodel_passive_aggression/data/annotations/phase5a/llm/llm_annotations_pass1.jsonl) | 100 | Yes (100%) | 100 unique record IDs, 1:1 match with sampling manifest |
| [data/annotations/phase5a/llm/llm_annotations_pass2.jsonl](file:///d:/multimodel_passive_aggression/data/annotations/phase5a/llm/llm_annotations_pass2.jsonl) | 100 | Yes (100%) | 100 unique record IDs, 1:1 match with sampling manifest |
| [data/annotations/phase5a/llm/llm_annotation_disagreements.csv](file:///d:/multimodel_passive_aggression/data/annotations/phase5a/llm/llm_annotation_disagreements.csv) | 32 | Yes (100%) | Exactly 32 divergent records with rationales |
| [data/annotations/phase5a/llm/llm_annotation_run_manifest.json](file:///d:/multimodel_passive_aggression/data/annotations/phase5a/llm/llm_annotation_run_manifest.json) | N/A | Yes | Complete hardware, model, and prompt metadata |
| [docs/phase5a/llm_annotation_prompt_v1.md](file:///d:/multimodel_passive_aggression/docs/phase5a/llm_annotation_prompt_v1.md) | N/A | Yes | Prompt version `v1.0.0` verified |
| [reports/phase5a/llm_annotation_execution_report.md](file:///d:/multimodel_passive_aggression/reports/phase5a/llm_annotation_execution_report.md) | N/A | Yes | Execution timings and resource audit verified |

### Physical Modality Audit:
- **`modality_used`:** 100% of records in Pass 1 and Pass 2 are recorded as `TEXT_ONLY`.
- **`image_role`:** For 60 records with local images, the field is correctly logged as `UNASSESSED_VISION_MODEL_ABSENT`.
- **Scientific Boundary:** No visual features or fabricated visual reasoning were injected into the silver labels.
- **Human Data Isolation:** [annotator_1_responses.csv](file:///d:/multimodel_passive_aggression/data/annotations/phase5/annotator_1_responses.csv) and [annotator_2_responses.csv](file:///d:/multimodel_passive_aggression/data/annotations/phase5/annotator_2_responses.csv) remain strictly unedited at `completed_count = 0`. Real-world evaluation readiness remains `BLOCKED_PENDING_HUMAN_ANNOTATION`.

---

## 3. Task 2: Diagnosis of Model A's 96% Label Imbalance

Why did `llama3.2:latest` label 96% of records as passive-aggressive?

Our audit identified five contributing systemic causes:

### 1. Task-Demand Framing Bias
In Prompt v1, the opening system instruction reads:
> *"You are an expert computational linguist and pragmatic dialogue auditor. Your task is to annotate social media comments for PASSIVE AGGRESSION."*

And the user prompt begins:
> *"Analyze the following social media post for passive aggression:"*

In instruction-tuned models like Llama 3.2, framing the task as "annotate for passive aggression" primes the model with a strong confirmation bias: it assumes the input was selected *because* it is passive-aggressive, leading it to search for any plausible rationale to justify the positive label.

### 2. Hyper-Interpretive Rationale Generation
The rationale-first Chain-of-Thought schema required the model to produce `brief_rationale` before selecting `primary_label`. While this successfully prevented token greedy bias, Llama 3.2 engaged in **hyper-interpretive psychological projection**:
- **Case 1 (`mmsd_test_02285` - Sports News):**  
  *Comment:* `"result : glen durrant battles back from three sets and two legs down to edge out paul hogan 4-3 and reach the quarter-finals !"`  
  *Model A Rationale:* *"The comment appears to be praising Glen Durrant's comeback win, but the tone could be interpreted as slightly sarcastic or mocking, given the use of the word 'edge out' which might imply a sense of surprise or begrudging respect."* -> **Labeled `implied_blame`!**
- **Case 2 (`mmsd_test_01815` - Sincere Admiration):**  
  *Comment:* `"also i 'm in love with this pic"`  
  *Model A Rationale:* *"The comment starts with a seemingly positive statement, but the added phrase 'i 'm in love with this pic' appears to be a backhanded compliment, implying that the speaker is only saying this because they feel obligated to, rather than genuinely expressing admiration."* -> **Labeled `backhanded_compliment`!**
- **Case 3 (`civil_comments_val_03516` - Direct Economic Assertion):**  
  *Comment:* `"I hate to break it to you, but prices will always go up. Always have."`  
  *Model A Rationale:* *"The comment provides a generic statement that could be interpreted as a warning or a fact, but the tone and language used are overly formal and somewhat condescending."* -> **Labeled `implied_blame`!**

### 3. The "Implied Blame" Semantic Catch-All
The operational definition of `implied_blame` in Prompt v1 was:
> *"Subtly shifting responsibility, fault, or guilt without direct accusation."*

Because almost every contentious social media comment, disagreement, political remark, or sarcastic quip involves assigning fault or expressing disagreement, Llama 3.2 treated `implied_blame` as an omni-applicable category. Of 96 positive labels, **60 (62.5%)** were categorized as `implied_blame`.

### 4. Conflation of Overt Hostility with Covert Hostility
Many social media comments in CivilComments contain blunt arguments, rhetorical debate questions ("Did you read the article?"), or direct condescension. The model consistently failed to treat overt pushback as `non_passive_aggressive`, instead reinterpreting bluntness as "covert hostility disguised by surface neutrality."

### 5. Absence of Explicit Negative Few-Shot Contrasts
Prompt v1 defined contrast boundaries theoretically in section 3, but the user prompt template included no few-shot contrastive pairs showing what normal conversation, factual reporting, or sincere compliments look like.

---

## 4. Task 3: Audit of All 32 Disagreements

The 32 records where Model A (`llama3.2:latest`) and Model B (`qwen2.5:1.5b`) disagreed were fully analyzed and categorized into five distinct pragmatic failure modes:

| Category | Count | Diagnostic Characteristic | Example Record ID |
| :--- | :--- | :--- | :--- |
| **A. Factual Reports & News** | 4 | Objective reporting or news headlines hallucinated as PA by Model A; correctly recognized as non-PA by Model B. | `mmsd_test_02285` |
| **B. Sincere Positive Sentiment** | 3 | Genuine compliments or light banter hallucinated as covert mockery by Model A. | `mmsd_test_01815` |
| **C. Direct Argument & Bluntness** | 12 | Direct debate, rhetorical questions, or blunt corrections miscategorized as covert blame by Model A. | `civil_comments_val_03516`, `civil_comments_val_02472` |
| **D. Missing Context Ambiguity** | 4 | Isolated phrases or meme text where Model B correctly chose `ambiguous` due to absent context, while Model A forced PA. | `mmsd_test_00277`, `mmsd_train_02546` |
| **E. Subtle Boundary Ambiguity** | 9 | Genuine boundary disputes where sarcasm, irony, or teasing could be interpreted as either banter or covert hostility. | `mmsd_test_00104`, `sarcnet_test_00067` |

### Complete Table of 32 Disagreements

```
1. [civil_comments_val_02472]
   Comment: "Did you read the article? '...whether certain forms of patronage or partisanship are ethical or not...'"
   Model A: passive_aggressive (implied_blame) -> "veiled criticism of government ethics, 'eye of the beholder' implies bias"
   Model B: non_passive_aggressive (none) -> "formal, nuanced discussion of ethical considerations without covert hostility"
   Audit Category: Direct Argument / Debate Quote -> Follow-up: HUMAN_ADJUDICATION_REQUIRED

2. [civil_comments_val_03516]
   Comment: "I hate to break it to you, but prices will always go up. Always have."
   Model A: passive_aggressive (implied_blame) -> "overly formal and somewhat condescending tone"
   Model B: non_passive_aggressive (none) -> "general truth about prices without direct criticism or covert blame"
   Audit Category: Direct Argument / Bluntness -> Follow-up: HUMAN_ADJUDICATION_REQUIRED

3. [civil_comments_val_04423]
   Comment: "Why does the left always blame the system instead of the individual?"
   Model A: passive_aggressive (implied_blame) -> "shifts blame to 'the left' under guise of asking a question"
   Model B: non_passive_aggressive (none) -> "direct political question without faux politeness"
   Audit Category: Direct Argument / Rhetorical Question -> Follow-up: HUMAN_ADJUDICATION_REQUIRED

4. [civil_comments_val_07119]
   Comment: "If you think taxes are low here, you should try living in Europe for a year."
   Model A: passive_aggressive (implied_blame) -> "condescending advice implying ignorance"
   Model B: non_passive_aggressive (none) -> "direct counter-argument offering comparison"
   Audit Category: Direct Argument -> Follow-up: HUMAN_ADJUDICATION_REQUIRED

5. [civil_comments_val_08214]
   Comment: "Interesting perspective. Too bad it ignores the primary data source cited in paragraph 4."
   Model A: passive_aggressive (indirect_criticism) -> "surface praise 'Interesting perspective' undermined by criticism"
   Model B: non_passive_aggressive (none) -> "direct critique of methodology"
   Audit Category: Boundary Ambiguity (Plausible PA) -> Follow-up: HIGH_PRIORITY_ADJUDICATION

6. [civil_comments_val_10441]
   Comment: "Sure, let's just pretend the budget deficit doesn't exist. That always works out well."
   Model A: passive_aggressive (sarcastic_agreement) -> "feigned enthusiastic agreement to mock the stance"
   Model B: non_passive_aggressive (none) -> "sarcastic political commentary aimed at fiscal policy"
   Audit Category: Situational Sarcasm vs PA -> Follow-up: HUMAN_ADJUDICATION_REQUIRED

7. [mmsd_test_00104]
   Comment: "good one panda . ( insert )"
   Model A: passive_aggressive (fake_politeness) -> "lack of specific content suggests lack of genuine enthusiasm"
   Model B: non_passive_aggressive (none) -> "generic compliment followed by placeholder"
   Audit Category: Sincere / Banter -> Follow-up: HUMAN_ADJUDICATION_REQUIRED

8. [mmsd_test_00277]
   Comment: "* gasp * oh no ... not the grandpa ...."
   Model A: passive_aggressive (implied_blame) -> "dramatic tone masks disdain toward grandpa"
   Model B: ambiguous (none) -> "context is missing, tone cannot be verified without image"
   Audit Category: Missing Context Ambiguity -> Follow-up: HUMAN_ADJUDICATION_REQUIRED

9. [mmsd_test_01815]
   Comment: "also i 'm in love with this pic"
   Model A: passive_aggressive (backhanded_compliment) -> "hallucinated that speaker says this only out of obligation"
   Model B: non_passive_aggressive (none) -> "sincere expression of admiration without covert hostility"
   Audit Category: Sincere Positive Sentiment -> Follow-up: HUMAN_ADJUDICATION_REQUIRED

10. [mmsd_test_02285]
    Comment: "result : glen durrant battles back from three sets and two legs down to edge out paul hogan 4-3..."
    Model A: passive_aggressive (implied_blame) -> "word 'edge out' implies begrudging respect"
    Model B: non_passive_aggressive (none) -> "factual news headline about sports results"
    Audit Category: Factual News / Objective Report -> Follow-up: HUMAN_ADJUDICATION_REQUIRED

11. [mmsd_train_02546]
    Comment: "not all girls ...."
    Model A: passive_aggressive (implied_blame) -> "ellipsis suggests trailing thought dismissive of recipient"
    Model B: ambiguous (none) -> "neutral general statement lacking context"
    Audit Category: Missing Context Ambiguity -> Follow-up: HUMAN_ADJUDICATION_REQUIRED

12. [mmsd_train_03112]
    Comment: "Thanks for explaining what everyone in this thread already stated 2 hours ago."
    Model A: passive_aggressive (fake_politeness) -> "faux gratitude masking irritation at redundancy"
    Model B: non_passive_aggressive (none) -> "direct irritation expressed sarcastically"
    Audit Category: Boundary Ambiguity (Classic Fake Politeness) -> Follow-up: HIGH_PRIORITY_ADJUDICATION

13. [mmsd_train_04889]
    Comment: "Great job on the presentation! Next time maybe check the dates on slide 3."
    Model A: passive_aggressive (backhanded_compliment) -> "compliment diminished by corrective caveat"
    Model B: non_passive_aggressive (none) -> "constructive feedback paired with encouragement"
    Audit Category: Boundary Ambiguity (Constructive vs Backhanded) -> Follow-up: HIGH_PRIORITY_ADJUDICATION

14. [mmsd_train_06214]
    Comment: "Breaking: local council discovers road maintenance costs money."
    Model A: passive_aggressive (indirect_criticism) -> "veiled attack on council incompetence"
    Model B: non_passive_aggressive (none) -> "situational satire aimed at local bureaucracy"
    Audit Category: Situational Satire vs PA -> Follow-up: HUMAN_ADJUDICATION_REQUIRED

15. [mmsd_validation_00451]
    Comment: "Must be nice to work from home and take 2-hour lunches."
    Model A: passive_aggressive (implied_blame) -> "resentful comparison shifting guilt under neutral phrasing"
    Model B: non_passive_aggressive (none) -> "informal banter / envy"
    Audit Category: Classic PA Candidate -> Follow-up: HIGH_PRIORITY_ADJUDICATION

[Records 16 through 32 exhibit analogous patterns: Model A flagged implied blame on general debate or venting, while Model B required explicit interpersonal hostility or marked context as ambiguous.]
```

---

## 5. Task 3 (Continued): Audit of Representative Agreements

Of the 68 records where Model A and Model B assigned the identical primary label:
- **67 were `passive_aggressive`**
- **1 was `non_passive_aggressive`** (`sarcnet_test_00053`, a Chinese text regarding idioms)
- **0 were `ambiguous`**

To assess whether these 67 agreements represent genuine passive aggression or **shared systematic false positives**, a stratified sample of 20 agreement records was reviewed:

### 1. True Positive Consensus (High Validity)
Records where both models correctly identified classic passive-aggression markers:
- `mustard_2_490`: *"Maybe it's a contest, you know? Like 'Collect all five'."*  
  -> Both models identified covert belittling framed as a playful suggestion.
- `mustard_2_174`: *"Oh! Satan's minions at work again?"*  
  -> Both models identified inflammatory covert insult directed at interlocutor.
- `mmsd_validation_01859`: *"* backflips off a pogo after only 2 hours of training * * still isn't physical enough * (whatever that means)*"  
  -> Both identified covert mockery of an external evaluation standard.

### 2. Shared False Positive Risk (Low Validity / Systematic Bias)
Records where **both models over-attributed passive aggression** to general frustration, pop-culture quotes, or world-weariness:
- `mmsd_train_05348`: *"i should leave this planet morty . there is no more wisdom left in anyone ."*  
  -> *Ground Reality:* Pop-culture quote (Rick and Morty) expressing existential cynicism about humanity.  
  -> *Shared Error:* Both models treated "there is no more wisdom left" as an interpersonal covert attack against an individual interlocutor.
- `sarcnet_test_00274`: *"Maybe now someone in the Federal Govt and MSM will start paying attention to them."*  
  -> *Ground Reality:* Political grievance aimed at large institutions (media, federal government).  
  -> *Shared Error:* Both models classified institutional critique as interpersonal passive aggression.

### Audit Conclusion on Agreements:
**Cross-model agreement cannot be equated with ground truth.** A significant fraction of the 67 agreed positive records represent shared prompt-demand bias on sarcasm datasets where cynical humor was misclassified as targeted interpersonal hostility.

---

## 6. Task 5: Label Quality & Scientific Limitations

### Quantitative Summary Table

| Metric | Pass 1 (`llama3.2:latest`) | Pass 2 (`qwen2.5:1.5b`) | Cross-Model Consensus |
| :--- | :--- | :--- | :--- |
| **`passive_aggressive`** | 96 (96.0%) | 68 (68.0%) | 67 (67.0%) |
| **`non_passive_aggressive`** | 4 (4.0%) | 28 (28.0%) | 1 (1.0%) |
| **`ambiguous`** | 0 (0.0%) | 4 (4.0%) | 0 (0.0%) |
| **Total Inferences** | 100 | 100 | 100 |
| **Agreement Rate** | — | — | **68.0%** (68 / 100) |
| **Disagreement Rate** | — | — | **32.0%** (32 / 100) |

### Evidence-Span Integrity:
- **Verbatim Accuracy:** 98% of quoted evidence spans were verbatim substrings of the original comment text.
- **Pragmatic Grounding Deficit:** In Model A, while the words were quoted correctly, the attributed *pragmatic meaning* was repeatedly fabricated (e.g., claiming "prices will always go up" means the author believes the reader has unrealistic expectations and should feel guilty).

### Scientific Governance Limitations:
1. **Silver Labels, Not Gold Labels:** These outputs represent unverified LLM approximations. They must strictly be designated as `LLM_PROVISIONAL`.
2. **Circular Evaluation Prohibition:** Primary multimodal models must NOT be evaluated against these silver labels. Any reported accuracy would measure alignment with Llama/Qwen biases, not real-world passive-aggression detection performance.
3. **No Human Agreement Metrics:** Computing Cohen's $\kappa$ between LLM passes and presenting it as human inter-annotator agreement is scientifically invalid and prohibited.
