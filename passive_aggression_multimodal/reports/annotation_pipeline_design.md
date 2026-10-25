# AI-Assisted Multimodal Annotation Pipeline Design

**Project Title:** Context-Aware Multimodal Passive-Aggression Detection in Social Media Using Image–Comment Fusion  
**Module:** Phase 1 Architecture Specification  
**Implementation Source:** `src/annotation/annotation_pipeline.py`  

---

## 1. Architectural Overview

To scale annotation across thousands of multimodal social media comments without sacrificing scientific rigor, the annotation pipeline employs a **multi-agent, multi-modal triadic adjudication architecture**:

```
                       [ Unified Record ]
                                |
             +------------------+------------------+
             |                                     |
             v                                     v
   [ Text-Only Annotator ]             [ Vision-Language Annotator ]
   (Lexical & Context Models)          (Multimodal Image-Text Fusion)
             |                                     |
             +------------------+------------------+
                                |
                                v
                   [ Adjudication Engine ]
                   - Reconciles predictions
                   - Flags agreements / splits
                   - Assigns provisional label
                   - Preserves full provenance
                                |
                                v
               [ Human-in-the-Loop Verification ]
               (Expert Review for Gold Standard Set)
```

---

## 2. Component Specifications

### A. Text-Only Annotator Module (`BaseAnnotator`)
- **Role:** Evaluates the target comment, parent comment, and dialogue history in isolation from visual media.
- **Phase 1 Implementation:** `RuleAndHeuristicTextAnnotator` analyzing lexical constructions, weaponized politeness patterns, implied blame syntax, and dismissive markers.
- **Production Integration Path:** Pluggable interface ready for local open-source transformer models (e.g. `DeBERTa-v3-base`, `RoBERTa-large`).

### B. Vision-Language Annotator Module (`VisionLanguageAnnotator`)
- **Role:** Evaluates the joint image–comment pair to detect cross-modal incongruity, visual irony, facial expression cues, and scene context.
- **Phase 1 Implementation:** Verifies image existence, file decodability, dimensions, and computes cross-modal baseline discrepancy metrics.
- **Production Integration Path:** Standardized interface for open local vision-language models:
  - `InternVL2-2B` / `InternVL2-4B` (Apache 2.0 / MIT)
  - `LLaVA-v1.6-Mistral-7B`
  - `Qwen2-VL-2B-Instruct` (Apache 2.0)
  - Can be executed via free local Hugging Face pipelines or Ollama endpoints without paid APIs.

### C. Adjudication Engine (`AdjudicationPipeline`)
- **Role:** Reconciles predictions from unimodal and multimodal annotators.
- **Agreement Classifications:**
  - `UNANIMOUS_AGREEMENT`: Both text and multimodal annotators arrive at identical primary labels and subtypes.
  - `MAJORITY_AGREEMENT`: Multi-model ensemble reaches a 2/3 consensus.
  - `DISAGREEMENT`: Text and visual annotators produce conflicting judgments (e.g., text is seemingly polite, but visual annotator flags visual disaster).
  - `UNIMODAL_ONLY`: Media asset is missing or not applicable; decision defaults to text analysis with appropriate confidence penalty.

---

## 3. Data Governance and Provenance Retention

Every adjudicated record emitted by the pipeline retains complete historical traceability:

```json
{
  "example_id": "mmsd_train_00042",
  "original_source_label": 1,
  "text_only_prediction": {
    "annotator_id": "heuristic_text_v1",
    "primary_label": "non_passive_aggressive",
    "confidence": 0.75,
    "explanation": "No overt linguistic markers identified in text."
  },
  "vlm_prediction": {
    "annotator_id": "vlm_interface_stub",
    "primary_label": "passive_aggressive",
    "confidence": 0.85,
    "explanation": "Incongruity detected between celebratory praise and collapsed infrastructure in image."
  },
  "agreement_status": "DISAGREEMENT",
  "provisional_label": "passive_aggressive",
  "adjudication_rationale": "Visual incongruity reverses literal textual polarity.",
  "synthetic_status": "REAL",
  "human_verification_status": "AWAITING_HUMAN_REVIEW",
  "human_verified_label": null
}
```

---

## 4. Quality Control and Inter-Annotator Agreement (IAA)

In subsequent Phase 2 operations, the pipeline will compute:
1. **Fleiss' Kappa ($\kappa$) and Krippendorff's Alpha ($\alpha$):** For measuring statistical agreement across text annotator, VLM, and human adjudicator.
2. **Confidence Thresholding:** Records with adjudication confidence $< 0.70$ or classified as `DISAGREEMENT` are automatically routed to human expert queues.
3. **Gold-Standard Partition Isolation:** Human-verified gold standard test sets are strictly quarantined from automated prompt tuning and model training folds.
