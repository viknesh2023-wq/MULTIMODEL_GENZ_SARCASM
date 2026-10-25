# Phase 6D: LLM Annotation Quality Audit & Governance Document

**Project:** Context-Aware Multimodal Passive-Aggression Detection in Social Media Using Image–Comment Fusion  
**Artifact:** `reports/phase6d/llm_annotation_quality_audit.md`  
**Date:** October 21, 2026  
**Auditor:** Automated Pragmatic & Experimental Governance Auditor  

---

## 1. Audit Scope & Objective

Phase 6D executed an end-to-end local LLM annotation workflow across the 30-record stratified Phase 6A pilot dataset.  
The audit verifies:
1. **Model Provenance & Independence:** Complete separation between Model 1 (`llama3.2:latest`) and Model 2 (`qwen2.5:1.5b`).
2. **Modality Truthfulness:** Honest reporting that visual elements were unassessed (`TEXT_ONLY`).
3. **Linguistic Boundary Adherence:** Application of the 6 negative boundaries (direct aggression, situational sarcasm, news reports, sincere praise, blunt dispute, friendly teasing).
4. **Preservation of Human Benchmark:** Verification that human annotation files remain completely unpopulated (count = 0).
5. **Adjudication Traceability:** Forensic justification for all resolved disagreements.

---

## 2. Modality & Vision Model Assessment

An inspection of the local Ollama environment confirmed:
- `llama3.2:latest` (3.2B parameters) - Installed
- `qwen2.5:1.5b` (1.5B parameters) - Installed
- `llama3.2:1b` (1.2B parameters) - Installed
- **Vision Models (`llava`, `llama3.2-vision`, etc.):** **NONE INSTALLED**

In accordance with strict research integrity policies:
- No large weights were downloaded or installed without user consent.
- All 30 records were processed using text and conversational context only.
- For 14 image-bearing records, metadata explicitly documents `modality_used = "TEXT_ONLY"`, `image_assessed = false`, and `ocr_assessed = false`.

---

## 3. Language Segregation Audit

The 30-record pilot contains 27 English records and 3 Chinese records from SarcNet (`sarcnet_test_00053`, `sarcnet_train_00095`, `sarcnet_train_00844`).
- **English Benchmark Performance:** Reported separately (27 records, raw agreement: 51.85%, κ: 0.1397).
- **Chinese Exploratory Performance:** Reported separately (3 records, raw agreement: 66.67%).
- **No Cross-Contamination:** Chinese records were never pooled into English benchmark metrics.

---

## 4. Disagreement Root-Cause Forensic Analysis

The adjudication pass resolved 14 model divergences:

### Record `mustard_2_490` (mustard)
- **Comment Text:** "Maybe it's a contest, you know? Like "Collect all five"."
- **Dialogue Context:** [Turn 1]: I know, I opened it up and it and there it was just floathing in there, like this tiny little hitchhiker.
- **Model 1 (`llama3.2:latest`):** `ambiguous` (none)  
  *Rationale:* The comment's tone and language are ambiguous, making it unclear whether the speaker is genuinely suggesting a contest or implying that the recipient's actions are inadequate. The phrase 'tiny little hitchhiker' could be interpreted as a mocking remark, but the speaker's intent is unclear.
- **Model 2 (`qwen2.5:1.5b`):** `non_passive_aggressive` (none)  
  *Rationale:* The comment does not target an individual or interlocutor, nor does it appear to blame or mock the target. The surface wording is polite and neutral, allowing the speaker to deny hostile intent.
- **Adjudicated Decision:** **`ambiguous`** (Subtype: `none`, Confidence: 1/5, Requires Human Review: False)
- **Root Cause & Adjudication Rationale:**  
  *Root Cause:* Linguistic interpretation of tone and intent  
  *Adjudicator Rationale:* The comment's tone and language are ambiguous, making it unclear whether the speaker is genuinely suggesting a contest or implying that the recipient's actions are inadequate. The phrase 'tiny little hitchhiker' could be interpreted as a mocking remark, but the speaker's intent is unclear. Both models' rationales highlight the uncertainty surrounding the speaker's intent, which is a key characteristic of passive aggression. However, without more context or information about the relationship between the speaker and the recipient, it is difficult to definitively label the comment as passive-aggressive or not.

### Record `mmsd_train_10780` (mmsd)
- **Comment Text:** "good to see the guys at <user> once again taking pride in doing a good job ."
- **Dialogue Context:** None
- **Model 1 (`llama3.2:latest`):** `ambiguous` (none)  
  *Rationale:* The comment appears to be praising the user's work, but the phrase 'taking pride in doing a good job' could be interpreted as slightly sarcastic or condescending, making the target unclear and the intent ambiguous.
- **Model 2 (`qwen2.5:1.5b`):** `non_passive_aggressive` (none)  
  *Rationale:* The comment is neutral and praising, with no indication of hostility or blame directed at an individual. The surface wording appears polite and does not allow the speaker to deny hostile intent.
- **Adjudicated Decision:** **`ambiguous`** (Subtype: `none`, Confidence: 1/5, Requires Human Review: False)
- **Root Cause & Adjudication Rationale:**  
  *Root Cause:* Linguistic interpretation of sarcasm/sarcasm detection  
  *Adjudicator Rationale:* The comment text appears to be praising the user's work, but the phrase 'taking pride in doing a good job' could be interpreted as slightly sarcastic or condescending. Model 1's classification as 'ambiguous' is supported by its nuanced rationale, while Model 2's classification as 'non_passive_aggressive' is not supported by its rationale, which fails to address the potential for sarcasm. The lack of clear indication of hostility or blame directed at an individual, combined with the plausible deniability, suggests that the comment may be open to multiple interpretations.

### Record `civil_comments_val_08920` (civil_comments)
- **Comment Text:** "Anthony, that is helpful information, thanks."
- **Dialogue Context:** None
- **Model 1 (`llama3.2:latest`):** `ambiguous` (none)  
  *Rationale:* Lacks context, tone, and target to determine hostility or intent. The phrase appears to be a genuine expression of gratitude, but its brevity and lack of surrounding information make it difficult to assess its true nature.
- **Model 2 (`qwen2.5:1.5b`):** `non_passive_aggressive` (none)  
  *Rationale:* The comment appears to be a sarcastic agreement, praising the information without any direct hostility or blame directed at an individual. The surface wording is polite and neutral, allowing the speaker to deny any hostile intent.
- **Adjudicated Decision:** **`non_passive_aggressive`** (Subtype: `none`, Confidence: 1/5, Requires Human Review: False)
- **Root Cause & Adjudication Rationale:**  
  *Root Cause:* Linguistic root cause: Insufficient context and tone analysis.  
  *Adjudicator Rationale:* Model 2's label is more conservative and aligns with the operational definition of passive aggression. The comment's polite wording and lack of direct hostility or blame directed at an individual suggest a sarcastic agreement rather than genuine gratitude. The brevity of the comment and the absence of context make it difficult to assess the true nature of the speaker's intent, but Model 2's rationale provides a plausible explanation for the comment's tone and intent.

### Record `mmsd_test_00104` (mmsd)
- **Comment Text:** "good one panda . ( insert )"
- **Dialogue Context:** None
- **Model 1 (`llama3.2:latest`):** `ambiguous` (none)  
  *Rationale:* Lack of context and unclear intent make it difficult to determine if the comment is passive-aggressive. The use of '(insert)' suggests a placeholder for an image, but the absence of an image makes it impossible to assess the target of the comment.
- **Model 2 (`qwen2.5:1.5b`):** `non_passive_aggressive` (none)  
  *Rationale:* The comment is not passive aggressive as it does not target an individual or convey covert hostility. The surface wording appears polite, praising, and neutral, allowing the speaker to deny hostile intent.
- **Adjudicated Decision:** **`ambiguous`** (Subtype: `none`, Confidence: 1/5, Requires Human Review: True)
- **Root Cause & Adjudication Rationale:**  
  *Root Cause:* Lack of contextual information and unclear intent  
  *Adjudicator Rationale:* The comment's intent is unclear due to the lack of context and the presence of '(insert)', which could be a placeholder for an image. Model 1's primary label of 'ambiguous' is supported by Model 2's primary label of 'non_passive_aggressive', as both models acknowledge the uncertainty in the comment's intent. However, without more context or information about the intended target, it is difficult to definitively label the comment as passive-aggressive or not.

### Record `civil_comments_val_04917` (civil_comments)
- **Comment Text:** "Social Security has been under attack since its creation by Republicans and DINOs.  The latest slap in the face of all Social Security recipients is the phony "Cost of Living" gift of the same traitors.  .3% increase for the coming year which will result in about $5 per month.  A legacy of both traitorous parties while they vote annual privatized military increases in a pay which none of the current Social Security recipients ever saw during their services.  But, they will soon be dead so they really don't matter and replaced by upcoming slashed benefits boomer children courtesy of the same traitorous One Party deadbeats.  Disgusting."
- **Dialogue Context:** None
- **Model 1 (`llama3.2:latest`):** `non_passive_aggressive` (none)  
  *Rationale:* The comment contains overt hostility and blame directed at Republicans and DINOs, with no plausible deniability. The language used is strongly critical and dismissive, with no apparent attempt to be polite or neutral.
- **Model 2 (`qwen2.5:1.5b`):** `passive_aggressive` (sarcastic_agreement)  
  *Rationale:* The comment contains sarcasm and mocking language directed at Social Security recipients, implying that the recipients are not affected by the changes, which is a covert hostility disguised as surface politeness.
- **Adjudicated Decision:** **`ambiguous`** (Subtype: `none`, Confidence: 3/5, Requires Human Review: True)
- **Root Cause & Adjudication Rationale:**  
  *Root Cause:* Linguistic root cause: Model 1 misinterprets the sarcastic tone as overt hostility, while Model 2 correctly identifies the sarcasm as a form of passive-aggressive language.  
  *Adjudicator Rationale:* The comment contains a mix of overt hostility and sarcastic language, making it challenging to determine the primary intent. While Model 1 correctly identifies the overt hostility, Model 2's identification of the sarcastic tone as passive-aggressive is plausible. However, the lack of an identifiable interpersonal target and the presence of plausible deniability make it difficult to conclusively label the comment as passive-aggressive.

### Record `civil_comments_val_02587` (civil_comments)
- **Comment Text:** "Your correct, Kon, I have no respect for your type of CONservative values that always target kicking the most vulnerable people while they're down.  Every time you rear your head or turn a corner, I'll be there to put you in your place.  LOL"
- **Dialogue Context:** None
- **Model 1 (`llama3.2:latest`):** `ambiguous` (none)  
  *Rationale:* Inference failure after retries: Ollama inference error for model 'llama3.2:latest': timed out
- **Model 2 (`qwen2.5:1.5b`):** `non_passive_aggressive` (none)  
  *Rationale:* The comment is not passive aggressive as it does not target an individual, does not appear to blame or mock the target, and does not undermine, demeans, mocks, or blame the target. The language is neutral and does not convey covert hostility.
- **Adjudicated Decision:** **`non_passive_aggressive`** (Subtype: `none`, Confidence: 1/5, Requires Human Review: False)
- **Root Cause & Adjudication Rationale:**  
  *Root Cause:* Linguistic root cause: Model 1 misinterprets the tone of the comment as sarcastic, while Model 2 correctly identifies the comment as sarcastic but labels it as non-passive-aggressive due to the lack of direct interpersonal target.  
  *Adjudicator Rationale:* Model 1's inference failure and Model 2's correct identification of sarcasm, but incorrect labeling due to lack of direct interpersonal target.

### Record `mmsd_train_02546` (mmsd)
- **Comment Text:** "not all girls ...."
- **Dialogue Context:** None
- **Model 1 (`llama3.2:latest`):** `ambiguous` (none)  
  *Rationale:* The comment lacks context, making it unclear whether the speaker is expressing a genuine statement or a veiled criticism. The phrase 'not all girls' is often used to imply that a particular statement or behavior applies to most people, but in this case, the target is unclear.
- **Model 2 (`qwen2.5:1.5b`):** `non_passive_aggressive` (none)  
  *Rationale:* The comment does not target an individual or convey hostility, and the surface wording appears polite, allowing the speaker to deny hostile intent.
- **Adjudicated Decision:** **`ambiguous`** (Subtype: `none`, Confidence: 1/5, Requires Human Review: False)
- **Root Cause & Adjudication Rationale:**  
  *Root Cause:* Linguistic ambiguity in interpretation of the phrase 'not all girls'  
  *Adjudicator Rationale:* The comment's ambiguity in meaning and lack of clear interpersonal target make it challenging to definitively label as passive-aggressive. Model 1's classification as 'ambiguous' is supported by the lack of context, while Model 2's classification as 'non_passive_aggressive' is based on the absence of hostility and polite surface wording. However, the phrase 'not all girls' can be interpreted in multiple ways, and without further context, it is unclear whether the speaker intends to make a genuine statement or convey a veiled criticism.

### Record `sarcnet_test_00274` (sarcnet)
- **Comment Text:** "Maybe now someone in the Federal Govt and MSM will start paying attention to them."
- **Dialogue Context:** None
- **Model 1 (`llama3.2:latest`):** `ambiguous` (none)  
  *Rationale:* The comment appears to be sarcastic, but the target is unclear. The phrase 'them' is ambiguous, and the tone could be interpreted as either mocking or genuine concern. Further context is needed to determine the intent.
- **Model 2 (`qwen2.5:1.5b`):** `non_passive_aggressive` (none)  
  *Rationale:* The comment does not target an individual or interlocutor, nor does it appear to blame or undermine the target. The surface wording is neutral and praising, allowing the speaker to deny hostile intent.
- **Adjudicated Decision:** **`ambiguous`** (Subtype: `none`, Confidence: 1/5, Requires Human Review: True)
- **Root Cause & Adjudication Rationale:**  
  *Root Cause:* Linguistic ambiguity in pronoun interpretation  
  *Adjudicator Rationale:* The comment's tone is ambiguous, and the pronoun 'them' could be interpreted in multiple ways. Model 1's primary label of 'ambiguous' is supported by the lack of clarity in the target and the speaker's intent. Model 2's primary label of 'non_passive_aggressive' is contradicted by the presence of ambiguity and the potential for covert hostility. Further context is needed to determine the speaker's intent and resolve the ambiguity.

### Record `civil_comments_val_09401` (civil_comments)
- **Comment Text:** "China bans the name Mohammad. Please check out link prior to deleting this post.

https://www.nytimes.com/2017/04/25/world/asia/china-xinjiang-ban-muslim-names-muhammad-jihad.html?_r=0"
- **Dialogue Context:** None
- **Model 1 (`llama3.2:latest`):** `ambiguous` (none)  
  *Rationale:* The comment appears to be indirectly criticizing China's actions, but the target (China) and intent (criticism) are unclear due to the lack of context and the presence of a neutral link. The comment's wording is polite and informative, making it difficult to determine the speaker's true intentions.
- **Model 2 (`qwen2.5:1.5b`):** `non_passive_aggressive` (none)  
  *Rationale:* The comment does not target an individual or interlocutor, nor does it appear to blame or mock the target. The surface wording is polite and neutral, allowing the speaker to deny hostile intent. The comment does not convey covert hostility or indirect aggression. The comment is a factual report and does not contain sarcasm or indirect communication.
- **Adjudicated Decision:** **`non_passive_aggressive`** (Subtype: `none`, Confidence: 1/5, Requires Human Review: False)
- **Root Cause & Adjudication Rationale:**  
  *Root Cause:* Linguistic root cause: Model 1 misinterprets the comment's intent as indirect criticism, while Model 2 correctly identifies it as a factual report.  
  *Adjudicator Rationale:* The comment's wording is polite and informative, and it does not appear to target an individual or interlocutor. The presence of a neutral link further supports the interpretation that the comment is a factual report rather than a veiled criticism. Model 2's label is more conservative and aligns with the operational definition of passive aggression.

### Record `mmsd_train_06126` (mmsd)
- **Comment Text:** "what is your favorite girl ? rt lauren like christina"
- **Dialogue Context:** None
- **Model 1 (`llama3.2:latest`):** `ambiguous` (none)  
  *Rationale:* The comment lacks context, making it unclear whether the speaker is genuinely asking about the target's favorite girl or expressing a subtle criticism. The phrase 'rt lauren like christina' could be interpreted as a veiled insult or a sarcastic comment.
- **Model 2 (`qwen2.5:1.5b`):** `non_passive_aggressive` (none)  
  *Rationale:* The comment does not target an individual or convey covert hostility. It is a neutral question about favorite girls, and the surface wording appears polite.
- **Adjudicated Decision:** **`ambiguous`** (Subtype: `none`, Confidence: 1/5, Requires Human Review: False)
- **Root Cause & Adjudication Rationale:**  
  *Root Cause:* Linguistic root cause: Insufficient contextual information and varying interpretations of the phrase 'rt lauren like christina'.  
  *Adjudicator Rationale:* The comment lacks clear context, making it difficult to determine the speaker's intent. Model 1's prediction of 'ambiguous' is supported by the lack of clear interpersonal target, covert hostility, and plausible deniability. Model 2's prediction of 'non_passive_aggressive' is contradicted by the potential for veiled insult or sarcasm in the phrase 'rt lauren like christina'. Further review is required to determine the speaker's intent.

### Record `mustard_2_111` (mustard)
- **Comment Text:** "What was that?!"
- **Dialogue Context:** [Turn 1]: Rachel is one of my closest friends. Although, being the only one who knows anything about this does makes me feel special. Okay! 
 [Turn 2]: Okay, stop-stop! Phoebe?! 
 [Turn 3]: Hey Rach!
- **Model 1 (`llama3.2:latest`):** `ambiguous` (none)  
  *Rationale:* The comment lacks clear context, making it difficult to determine if the speaker is genuinely surprised or expressing frustration. The dialogue context suggests a close relationship between the speaker and Rachel, but the tone of the comment is unclear.
- **Model 2 (`qwen2.5:1.5b`):** `non_passive_aggressive` (none)  
  *Rationale:* The comment is a simple inquiry without any hostile intent or covert hostility. The surface wording appears polite and does not allow the speaker to deny hostile intent.
- **Adjudicated Decision:** **`ambiguous`** (Subtype: `none`, Confidence: 1/5, Requires Human Review: False)
- **Root Cause & Adjudication Rationale:**  
  *Root Cause:* Linguistic root cause: Insufficient contextual information to disambiguate the speaker's intent.  
  *Adjudicator Rationale:* The dialogue context suggests a close relationship between the speaker and Rachel, but the tone of the comment is unclear. The comment lacks clear context, making it difficult to determine if the speaker is genuinely surprised or expressing frustration. Both models' rationales highlight the uncertainty surrounding the speaker's intent, leading to a conservative adjudication of 'ambiguous'.

### Record `mmsd_validation_02127` (mmsd)
- **Comment Text:** "pok éball vinyl ........ you 're all welcome ."
- **Dialogue Context:** None
- **Model 1 (`llama3.2:latest`):** `ambiguous` (none)  
  *Rationale:* The comment appears to be ambiguous due to the unclear target and lack of context. The phrase 'you're all welcome' could be interpreted as either a genuine invitation or a sarcastic remark.
- **Model 2 (`qwen2.5:1.5b`):** `non_passive_aggressive` (none)  
  *Rationale:* The comment appears polite and neutral, with no surface words that suggest hostility or blame directed at an individual. The comment does not provide any surface evidence of covert hostility or blame.
- **Adjudicated Decision:** **`ambiguous`** (Subtype: `none`, Confidence: 1/5, Requires Human Review: True)
- **Root Cause & Adjudication Rationale:**  
  *Root Cause:* Linguistic root cause: unclear target and lack of context  
  *Adjudicator Rationale:* The comment text does not provide sufficient context to determine the target of the comment. The phrase 'you're all welcome' could be interpreted as either a genuine invitation or a sarcastic remark, making it unclear whether the comment is passive-aggressive or not. Further analysis is required to determine the true intent behind the comment.

### Record `mmsd_validation_01859` (mmsd)
- **Comment Text:** "* backflips off a pogo after only 2 hours of training * * still isn 't physical enough * ( whatever that means )"
- **Dialogue Context:** None
- **Model 1 (`llama3.2:latest`):** `ambiguous` (none)  
  *Rationale:* The comment appears to be sarcastic, implying that the person's physical abilities are not improving despite 2 hours of training. However, the tone is unclear and the phrase 'whatever that means' suggests a lack of sincerity, making it difficult to determine the true intent of the comment.
- **Model 2 (`qwen2.5:1.5b`):** `non_passive_aggressive` (none)  
  *Rationale:* The comment does not target an individual or interlocutor, lacks surface politeness or deference, and does not convey covert hostility. The brevity and lack of context make it ambiguous, but it does not meet the criteria for passive aggression.
- **Adjudicated Decision:** **`ambiguous`** (Subtype: `none`, Confidence: 1/5, Requires Human Review: False)
- **Root Cause & Adjudication Rationale:**  
  *Root Cause:* Linguistic ambiguity in tone and intent  
  *Adjudicator Rationale:* The comment's tone is sarcastic, but the phrase 'whatever that means' suggests a lack of sincerity. Model 1's primary label of 'ambiguous' is supported by the lack of clear intent. Model 2's primary label of 'non_passive_aggressive' is not supported due to the presence of sarcasm and lack of surface politeness. The dialogue context and image availability do not provide sufficient information to determine the true intent of the comment.

### Record `sarcnet_train_00844` (sarcnet)
- **Comment Text:** "#鼠鼠我鸭#哇哦，原来鸭子是这样子的呀，学到了，课本上没有的知识耶"
- **Dialogue Context:** None
- **Model 1 (`llama3.2:latest`):** `non_passive_aggressive` (none)  
  *Rationale:* The comment uses a sarcastic tone, implying that the information is surprising or unexpected, but the language is polite and complimentary, suggesting a playful or teasing intent.
- **Model 2 (`qwen2.5:1.5b`):** `ambiguous` (none)  
  *Rationale:* Inference failure after retries: Ollama inference error for model 'qwen2.5:1.5b': timed out
- **Adjudicated Decision:** **`non_passive_aggressive`** (Subtype: `none`, Confidence: 1/5, Requires Human Review: False)
- **Root Cause & Adjudication Rationale:**  
  *Root Cause:* Linguistic root cause: Model 1 misinterprets the tone as playful/teasing, while Model 2 fails to provide a clear rationale due to inference error.  
  *Adjudicator Rationale:* The comment uses a sarcastic tone, implying that the information is surprising or unexpected, but the language is polite and complimentary, suggesting a playful or teasing intent. While Model 2 failed to provide a clear rationale, the tone and language used in the comment align with the definition of passive aggression, albeit in a playful manner.

---

## 5. Dataset Governance & Release Decision

> [!CAUTION]
> **GATE STATUS: BLOCKED_PENDING_HUMAN_ANNOTATION**  
> 1. The resulting dataset in `data/annotations/phase6d/llm_adjudication_results.csv` is suitable **strictly for weak-supervision ablation studies and diagnostic prompt engineering**.
> 2. It **MUST NOT** be used as an evaluation benchmark or reported as validated ground truth.
> 3. Downstream test splits remain uncompromised.
