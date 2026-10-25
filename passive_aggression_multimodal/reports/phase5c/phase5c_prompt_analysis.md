# Phase 5C: Linguistic and Pragmatic Prompt Analysis Report

**Project Title:** Context-Aware Multimodal Passive-Aggression Detection in Social Media Using Image–Comment Fusion  
**Canonical Project Root:** `D:\NLP_PROJECT\passive_aggression_multimodal`  
**Linked Workspace:** `D:\multimodel_passive_aggression`  
**Milestone:** Phase 5C — Linguistic Prompt Audit & Diagnostic Evaluation  
**Date:** October 21, 2026  
**Auditor:** Senior ML Research Auditor  

---

## 1. Executive Summary

This report provides a formal computational linguistics and pragmatic audit of the versioned annotation prompts used in the project:
1. **Prompt v1.0.0** ([docs/phase5a/llm_annotation_prompt_v1.md](file:///d:/multimodel_passive_aggression/docs/phase5a/llm_annotation_prompt_v1.md))
2. **Prompt v2.0.0** ([docs/phase5b/llm_annotation_prompt_v2.md](file:///d:/multimodel_passive_aggression/docs/phase5b/llm_annotation_prompt_v2.md))

We evaluate whether Prompt v2 successfully enforces theoretical distinctions, analyze the behavioral shifts observed in the 12-record diagnostic experiment, and discuss why label balance changes must not be conflated with ground-truth accuracy.

---

## 2. Six-Dimensional Theoretical Audit of Prompt v2.0.0

| Criterion | Formal Linguistic Requirement | Implementation in Prompt v2.0.0 | Audit Finding |
| :--- | :--- | :--- | :--- |
| **1. Interpersonal Target** | Hostility must be directed at an interlocutor, not an inanimate object or general state of the world. | Enforced in Section 1, Criterion 1: *"Hostility or blame is directed at an individual or conversation partner (NOT an external situation, weather, transit delay, or general state of the world)."* | **PASS.** Successfully eliminated false-positive projection on sports news and transit delays. |
| **2. Covert Hostility Grounded in Content** | Negative meaning must be pragmatically entailed by the text, not hallucinated via unstated assumptions. | Enforced in Section 6: Protocol requires explicitly stating literal meaning, target, and covert hostile intent. | **PASS.** Prevented hallucinating unstated obligations (e.g. on sincere admiration). |
| **3. Plausible Deniability** | Phrasing must retain a surface appearance of politeness, compliance, or neutrality. | Enforced in Section 1, Criterion 2: Literal surface words must appear polite/neutral, allowing speaker to deny malice. | **PASS.** Accurately distinguished faux politeness from direct insults. |
| **4. Direct vs Passive Boundary** | Overt aggression, toxicity, insults, and direct debate must NOT be categorized as passive-aggressive. | Enforced in Negative Boundaries 1 & 5: Direct insults and blunt arguments are explicitly labeled `non_passive_aggressive`. | **PASS.** Successfully re-routed blunt economic corrections out of PA. |
| **5. Context-Driven Ambiguity** | When conversational context, target identity, or vocal prosody is missing, model must abstain. | Enforced in Section 3 and Section 4: Isolated phrases or ambiguous meme text must be labeled `ambiguous`. | **PASS.** Routed isolated fragments (`not all girls...`) to `ambiguous`. |
| **6. Modality Honesty** | The prompt must never induce the model to claim visual reasoning when evaluating text only. | Enforced in User Prompt Template: Explicit note: *(Note: Image pixels unassessed; evaluate text and context)*. | **PASS.** No fabricated visual reasoning. |

---

## 3. Analysis of the 12-Record Diagnostic Experiment

The diagnostic comparison on the 12-record subset ([reports/phase5c/tables/prompt_v1_v2_comparison.csv](file:///d:/multimodel_passive_aggression/reports/phase5c/tables/prompt_v1_v2_comparison.csv)) revealed profound behavioral differences between model architectures:

```
Diagnostic Set (12 Records) Class Distribution Shift:

Model A (Llama 3.2 3B):
  Prompt v1:  10 PA (83.3%)  | 2 Non-PA (16.7%) | 0 Ambiguous (0.0%)
  Prompt v2:   1 PA (8.3%)   | 3 Non-PA (25.0%) | 8 Ambiguous (66.7%)

Model B (Qwen 2.5 1.5B):
  Prompt v1:   5 PA (41.7%)  | 5 Non-PA (41.7%) | 2 Ambiguous (16.7%)
  Prompt v2:   6 PA (50.0%)  | 5 Non-PA (41.7%) | 1 Ambiguous (8.3%)
```

### Key Analytical Insights:
1. **Llama 3.2's High Sensitivity to Context Guardrails:**
   Under Prompt v1, Llama 3.2 suffered from extreme task demand, forcing positive labels onto 83.3% of the diagnostic records. Under Prompt v2, when instructed that missing context warrants `ambiguous`, Llama 3.2 swung to high conservatism, routing **8 of the 12 records into `ambiguous`**.
2. **Qwen 2.5's Behavioral Stability:**
   Qwen 2.5 exhibited remarkable stability across prompt versions (5 PA vs 6 PA). It consistently recognized factual reports (`mmsd_test_02285`) and blunt assertions (`civil_comments_val_03516`) as `non_passive_aggressive` under both prompts.
3. **The Meaning of Label Shift:**
   A reduction in positive predictions is **NOT, by itself, proof of superior accuracy**. While Prompt v2 successfully cured obvious false positives (e.g. sports news), Llama 3.2's shift to 66.7% `ambiguous` on multimodal records indicates that a text-only model recognizes its own epistemic limitations when denied access to image and conversational thread context.

---

## 4. Scientific Prohibition: No Full-Set Prompt Iteration

A fundamental principle of scientific machine learning is the **strict separation of prompt development from benchmark evaluation**:
- If we iteratively tuned Prompt v2 against all 100 candidate records until the labels matched our expectations, the 100-record set would cease to be an evaluation set and become an overfitted training set.
- Such circular tuning constitutes **data dredging (p-hacking)** and destroys benchmark validity.
- Therefore, Prompt v1.0.0 and Prompt v2.0.0 remain permanently frozen in version control. Prompt v2 was tested strictly on the 12-record diagnostic subset.

---

## 5. Architectural Recommendations for Future Prompt Iterations

If a Prompt v3 is developed in future milestones (e.g., when a vision model is introduced or after human adjudication of the 32 disagreements):
1. **Multimodal CoT:** When a vision model (`llava` or `minicpm-v`) is available, the prompt should explicitly query **visual-textual incongruity** (e.g., *"Does the image portray a situation that contradicts the literal politeness of the comment?"*).
2. **Few-Shot Negative Contrasts:** Incorporating 3-4 validated contrastive few-shot examples directly into the prompt template will ground the model on the boundary between direct pushback, situational sarcasm, and true covert hostility.
