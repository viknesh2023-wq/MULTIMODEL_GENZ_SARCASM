# Phase 3 Error Analysis and Interpretability Report (Task L)
**Project Title:** Context-Aware Multimodal Passive-Aggression Detection in Social Media Using Image–Comment Fusion  
**Artifact:** `reports/phase3_error_analysis.md`  
**Analysis Script:** `scripts/analyze_prediction_errors.py`  
**Data Artifact:** `artifacts/phase3/evaluation/error_analysis_report.json`  
**Date:** October 21, 2026  

---

## 1. Operational Error Taxonomy

To systematically characterize prediction failures, eight operational failure archetypes are formalized:

| Archetype Code | Archetype Name | Operational Description | Typical Manifestation |
|---|---|---|---|
| **E1** | `SARCASM_CONFUSED_WITH_PA` | Sarcasm without interpersonal hostility misclassified as passive aggression | Ironic observations about weather or sports interpreted as hostile |
| **E2** | `SUBTLE_WORDING_MISSED` | Veiled interpersonal hostility overlooked by model | Overly polite or bureaucratic phrasing (*"Per my previous email..."*) |
| **E3** | `DIRECT_OVERT_AGGRESSION_CONFUSED` | Overt toxicity misclassified as covert passive aggression | Explicit profanity or harassment classified as subtle hostility |
| **E4** | `GENUINE_PRAISE_MISCLASSIFIED` | Sincere compliment flagged as a backhanded compliment | Enthusiastic praise misinterpreted due to sarcastic cues |
| **E5** | `IMAGE_COMMENT_MISMATCH` | Image incongruity misleading text interpretation | Whimsical meme image distorting an otherwise neutral comment |
| **E6** | `OCR_FAILURE_OR_ABSENT` | Unread embedded text in image obscuring contextual meaning | Social graphic whose punchline exists only within image text |
| **E7** | `MISSING_CONTEXT` | Absence of conversational history creating ambiguity | Terse response (*"Fine."*) ambiguous without prior messages |
| **E8** | `AMBIGUOUS_UNCERTAINTY` | Model prediction entropy near uniform | Near 0.33/0.33/0.33 probability distribution across classes |

---

## 2. Empirical Findings on Primary Benchmark vs. Real Auxiliary Data

### 2.1 Primary Benchmark Findings
- On the held-out test split of the 350-scenario benchmark ($N=54$), all models achieved 0.0% error rate.
- **Root Cause:** Synthetic scenarios were intentionally drafted during Phase 1 with salient archetype markers to serve as gold definitions. Consequently, neither the traditional baseline nor the neural models experienced misclassification on this subset.

### 2.2 Error Analysis on Real-World Auxiliary Benchmark
When evaluating models on the real multimodal benchmark (where test accuracy was $72.70\%$):
- **False Positive Dominance ($E1$ - Sarcasm Confusion):** The most frequent failure mode in social media text is the conflation of playful banter, irony, and non-hostile sarcasm with passive aggression. In social media comments, sarcastic inversion occurs frequently without any interpersonal resentment.
- **Context Starvation ($E7$ - Missing Context):** Comments such as *"Thanks for letting me know"* or *"That's totally fine"* are completely ambiguous when isolated from parent post captions or conversational turns.

---

## 3. Inference Modality Sensitivity Evaluation

Using `scripts/analyze_prediction_errors.py`, prediction stability was tested under inference-time modality ablation:
- **Image Zeroing:** Visual scene embeddings were zeroed out ($\mathbf{h}_{\text{image}} = \mathbf{0}, \mathbf{m}_{\text{image}} = 0$) at inference time.
  - *Result:* 0.0% prediction shift on synthetic scenarios, confirming that the gating layer successfully falls back to text modalities without crashing.
- **Context Zeroing:** Dialogue history embeddings were zeroed out ($\mathbf{h}_{\text{context}} = \mathbf{0}, \mathbf{m}_{\text{context}} = 0$) at inference time.
  - *Result:* The context gate $\mathbf{g} = \mathbf{m}_{\text{context}} \odot \sigma(\cdot)$ evaluated to zero, ensuring that the model gracefully reverted to comment-only inference.

---

## 4. Interpretability Caveats and Scientific Guardrails

In compliance with **Phase 3 Rule 13** (*"Do not describe feature perturbation or attention weights as definitive proof of model reasoning"*):
1. **Attention Weights are Not Explanations:** Multihead cross-attention weights between comments and context indicate correlation and representational alignment, not causal human intent or subjective linguistic reasoning.
2. **Post-Hoc Perturbation Limits:** Zeroing out a feature vector at inference time tests model robustness and sensitivity, not ground-truth psychological intent.
