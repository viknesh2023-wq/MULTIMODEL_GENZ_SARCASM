# Phase 2 Passive-Aggression Annotation Pilot Report

## 1. Executive Summary & Verification Context

- **Target Sample Size:** 150 real candidate records.
- **Gold-Standard Status:** `PENDING_HUMAN_REVIEW` (strict compliance with Phase 2 rules: automated provisional labels are kept separate from human gold standards).
- **Random Seed:** 42 (fully reproducible).
- **Prioritized Review Queue:** 50 / 150 records flagged as `HIGH` priority due to ambiguity, cross-modal disagreement, or high uncertainty.

## 2. Dataset Sourcing & Stratification

| Source Dataset | Sampled Count | Sampling Rationale |
|---|---:|---|
| `mmsd` | 60 | Verified multimodal memes (sarcastic and non-sarcastic benchmark pairs) |
| `sarcnet` | 15 | Social media multimodal tweets with confirmed image assets |
| `mustard` | 15 | Multi-turn TV dialogue with sequential conversational context |
| `goemotions` | 35 | Fine-grained emotion controls (sadness, grief, neutral, gratitude, admiration) |
| `civil_comments` | 25 | Toxicity contrast pairs (constructive critique vs severe direct aggression) |
| **Total** | **150** | Stratified cross-phenomenon representation |

## 3. Label Distributions

### 3.1 Primary Passive-Aggression Provisional Distribution

| Provisional Label | Count | Percentage |
|---|---:|---:|
| `ambiguous` | 49 | 32.7% |
| `non_passive_aggressive` | 101 | 67.3% |

### 3.2 Subtype Provisional Distribution

| Subtype | Count | Percentage |
|---|---:|---:|
| `sarcastic_agreement` | 49 | 32.7% |
| `other_or_unclear` | 101 | 67.3% |

### 3.3 Independent Auxiliary Sarcasm Distribution

| Auxiliary Sarcasm Label | Count | Percentage |
|---|---:|---:|
| `sarcastic` | 50 | 33.3% |
| `non_sarcastic` | 85 | 56.7% |
| `ambiguous` | 15 | 10.0% |

### 3.4 Triadic Agreement Status

| Agreement Status | Count | Percentage |
|---|---:|---:|
| `DISAGREEMENT` | 50 | 33.3% |
| `UNIMODAL_ONLY` | 75 | 50.0% |
| `UNANIMOUS_AGREEMENT` | 25 | 16.7% |

## 4. Methodological Safeguards & Governance

1. **Separation of Original vs Provisional Labels:** Benchmark ground truths (`mmsd`, `goemotions`, etc.) are preserved verbatim in `original_label` and were never automatically overwritten.
2. **Independence of Sarcasm vs Passive-Aggression:** Sarcasm was tagged independently in `independent_sarcasm_label`. Sarcastic instances without hostility (friendly teasing, absurd humor) were quarantined from `passive_aggressive`.
3. **Honest Vision Status:** The vision module is explicitly marked as `INTERFACE_STUB_PENDING_VLM_WEIGHTS`. No fake vision-language predictions were fabricated.
4. **Human Review Queue:** Exported to `data/processed/human_review_queue_150.json` for human expert adjudication before any Phase 3 training.
