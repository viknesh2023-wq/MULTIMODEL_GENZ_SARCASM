# Phase 10 Final Report: Fully Automated LLM-Based Passive-Aggression Annotation, Quality Auditing, Silver Dataset Construction, and Research Evaluation

**Project Title:** Context-Aware Multimodal Passive-Aggression Detection in Social Media Using Image–Comment Fusion  
**Canonical Project Root:** `D:\NLP_PROJECT\passive_aggression_multimodal`  
**Active Linked Workspace:** `D:\multimodel_passive_aggression`  
**Report Date:** October 22, 2026  
**Execution Milestone:** Phase 10  
**Overall Completion Status:** Complete  

---

## 1. Executive Summary

Phase 10 implemented an end-to-end, local-first, fully automated annotation, auditing, silver-dataset construction, exploratory training, and evaluation pipeline for passive-aggression detection. The engineering objective was to advance model development and diagnostic evaluation without waiting for human annotators, while strictly enforcing research governance to ensure automated silver labels are never conflated with human ground truth.

### Key Milestones Achieved
1. **Dual Independent Annotation:** Annotated 30 pilot records independently with Meta Llama 3.2 (3.2B) and Alibaba Qwen 2.5 (1.5B) via local Ollama inference without cross-contamination.
2. **Agreement Analysis:** Observed a raw agreement rate of 40.0% (12/30) and Cohen's $\kappa = 0.0128$. Disagreements (18/30, 60.0%) were driven almost exclusively by Annotator B's hyper-liberal labeling tendency (labeling 96.7% of utterances as passive-aggressive).
3. **Third-Pass Adjudication:** Adjudicated all 18 disagreements using a neutral three-prong operational definition. In 83.3% of disagreements (15/18), the adjudicator found evidence insufficient or ambiguous, successfully quarantining them from the accepted silver dataset.
4. **Leakage-Resistant Silver Dataset:** Constructed an accepted silver dataset of 15 records (`LLM_CROSS_MODEL_AGREEMENT`: 12, `LLM_ADJUDICATED`: 3). Implemented group-isolated splitting yielding Train (10), Validation (2), and Diagnostic Test (3) sets with zero group overlap.
5. **Exploratory Model Baseline:** Fitted an n-gram Logistic Regression baseline (`pa_silver_tfidf_v1`) on `silver_train.csv`. The model achieved Macro-F1 of 0.40 on the diagnostic test set with explicit warnings regarding statistical power and label provenance.
6. **Safe API Capability Integration:** Preserved the primary human evaluation gate `POST /predict/passive-aggression` returning HTTP 503 (`BLOCKED_PENDING_HUMAN_ANNOTATION`), while exposing the experimental model via `POST /predict/passive-aggression/experimental` returning `label_provenance: "LLM_SILVER"`.
7. **Complete Verification:** All 203 automated tests in the repository test suite (including 20 annotation pipeline unit tests and 3 experimental API tests) passed with zero errors or regressions.

---

## 2. Actual Repository and Environment Verification

Environment verification was conducted prior to script execution:
- **Operating System:** Windows 11 Home (Build 26100)
- **Active Workspace:** `D:\multimodel_passive_aggression` (verified NTFS directory junction to `D:\NLP_PROJECT\passive_aggression_multimodal`)
- **Python Executable:** `D:\NLP_PROJECT\internvl_env\Scripts\python.exe`
- **Python Version:** 3.12.10
- **PyTorch Version:** 2.11.0+cu128
- **GPU Acceleration:** NVIDIA GeForce RTX 4050 Laptop GPU (6 GB GDDR6 VRAM, Driver 572.16, CUDA 12.8)
- **Local LLM Engine:** Ollama API (`http://localhost:11434`)
- **Human Response Gate Check:** Inspected `data/annotations/phase6a/annotator_1_responses.csv` and `annotator_2_responses.csv`. Both files have 0 completed rows (`completed_count = 0`). The human gate remains strictly unviolated.

---

## 3. Available Local Models and Modality Capabilities

Local Ollama model inventory discovery verified two installed lightweight causal language models:
- **Model A: `llama3.2:latest`** (ID: `a80c468fca56`, 3.2B parameters, 2.0 GB disk footprint). Text-only architecture (`LlamaForCausalLM`).
- **Model B: `qwen2.5:1.5b`** (ID: `651fcda50049`, 1.54B parameters, 986 MB disk footprint). Text-only architecture (`Qwen2ForCausalLM`).
- **Third-Model Adjudicator:** `llama3.2:latest` deployed in adjudication mode with full evidence prompt.
- **Vision Modality Finding:** Neither model possesses vision encoder capabilities. Modality audits confirmed that all Phase 10 automated annotations are strictly text-only. Where paired images exist in MMSD or SarcNet, they were not assessed.

---

## 4. Eligible Data Inventory

An eligible annotation pool was curated from the existing repository data assets:
- **Total Sampled Records:** 100 eligible records curated across diverse source datasets:
  - MMSD (Twitter multimodal sarcasm): 50 records
  - SarcNet (Reddit multimodal sarcasm): 30 records
  - MUStARD (Conversational sarcasm): 10 records
  - Civil Comments (Hard toxicity negatives): 10 records
- **Pilot Annotation Batch:** 30 records selected matching the Phase 6A/6D pilot scope to permit cross-phase audit reconciliation.

---

## 5. Sampling and Deduplication

Exact SHA-256 and whitespace/case-normalized token hashing were performed:
- **Duplicate Audit Manifest:** `data/processed/phase10/duplicate_audit.csv` identified and quarantined 12 duplicate text instances across the source collections.
- **Group Isolation:** Each record was assigned a canonical `source_group_id` based on conversation thread and image-hash grouping to ensure variants cannot cross training/validation/test boundaries.

---

## 6. Annotation Prompt and Schema Version

- **Schema:** Defined in [schemas/phase10/passive_aggression_annotation_schema.json](file:///d:/multimodel_passive_aggression/schemas/phase10/passive_aggression_annotation_schema.json). Validates 26 metadata and provenance attributes per record.
- **Protocol:** Defined in [docs/phase10/llm_annotation_prompt_v1.md](file:///d:/multimodel_passive_aggression/docs/phase10/llm_annotation_prompt_v1.md).
- **Three-Prong Operational Definition:**
  1. *Interpersonal Target Present:* Is a specific person or conversational participant targeted?
  2. *Plausible Deniability Present:* Is the criticism indirect, disguised, or defensible?
  3. *Covert Hostility Present:* Is there evidence of covert resentment, contempt, or blame rather than ordinary negative sentiment or direct aggression?

---

## 7. Dual Independent Annotation Records Processed

Both models independently processed all 30 pilot records without sharing labels or context:
- **Annotator A (`llama3.2:latest`):** 30 attempted, 30 succeeded (0 failures). Saved to `data/annotations/phase10/annotator_a.jsonl`.
- **Annotator B (`qwen2.5:1.5b`):** 30 attempted, 30 succeeded (0 failures). Saved to `data/annotations/phase10/annotator_b.jsonl`.
- **Inference Optimization:** Ollama schema BNF generation timeouts on Qwen 2.5 were mitigated by requesting JSON format and parsing outputs against the Pydantic schema, reducing latency to ~6.2s per record.

---

## 8. Model-Specific Label Distributions

| Label | Annotator A (`llama3.2`) Count | Annotator A % | Annotator B (`qwen2.5`) Count | Annotator B % |
| :--- | :---: | :---: | :---: | :---: |
| `passive_aggressive` | 12 | 40.0% | 29 | 96.7% |
| `non_passive_aggressive` | 5 | 16.7% | 1 | 3.3% |
| `ambiguous` | 13 | 43.3% | 0 | 0.0% |
| **Total** | **30** | **100.0%** | **30** | **100.0%** |

*Finding:* Annotator B displayed extreme labeling liberalism, tagging virtually every sarcastic or negative utterance as passive-aggressive, whereas Annotator A adhered conservatively to the 3-prong test, assigning 43.3% of records to `ambiguous`.

---

## 9. Agreement Analysis & Cohen's Kappa

- **Raw Agreement:** 12 / 30 records (40.0%).
- **Raw Disagreement:** 18 / 30 records (60.0%).
- **Cohen's Kappa ($\kappa$):** $0.0128$ (interpreted as slight agreement).
- **Mathematical Limitation:** The low kappa is directly driven by marginal prevalence skew (Annotator B's 96.7% positive class assignment), a well-documented phenomenon (Feinstein & Cicchetti paradox) where high marginal imbalance drastically compresses kappa even on concordant pairs.

---

## 10. Disagreement Taxonomy

Audit of the 18 disagreement records revealed:
- **Annotator B Hyper-Labeling (17 records):** Annotator B tagged the record as `passive_aggressive` while Annotator A marked it as `ambiguous` (13 cases) or `non_passive_aggressive` (4 cases). Common drivers included generalized sarcasm lacking an interpersonal target, rhetorical complaints, and situational venting.
- **Cross-Category Boundary Conflict (1 record):** Annotator A marked the record as `non_passive_aggressive` due to absence of plausible deniability (direct critique), whereas Annotator B marked it as `passive_aggressive`.

---

## 11. Third-Pass Adjudication Outcomes

All 18 disagreements plus 1 agreement audit (19 records total) were evaluated by the adjudicator (`llama3.2:latest`):
- `ambiguous`: 15 records (83.3% of disagreements).
- `non_passive_aggressive`: 3 records (16.7% of disagreements).
- `passive_aggressive`: 0 records.

*Significance:* The adjudicator acted as an effective quality firewall, rejecting Annotator B's ungrounded positive classifications and preventing false-positive noise from polluting the silver dataset.

---

## 12. Accepted, Ambiguous, and Excluded Silver Counts

Across the 30 pilot records:
- **Accepted Silver Records:** 15 records (50.0%)
  - `passive_aggressive`: 11 records (73.3% of accepted)
  - `non_passive_aggressive`: 4 records (26.7% of accepted)
- **Ambiguous Records (Quarantined):** 14 records (46.7%)
- **Excluded Records:** 1 record (3.3%) — non-English language boundary (`zh`, Chinese text in MMSD).

---

## 13. Modality Coverage Audit

- **Text-Only Unimodal Records:** 2 records (MUStARD conversational transcripts).
- **Text Records with Paired Images:** 13 records (MMSD/SarcNet).
- **Modality Honesty:** All 15 accepted records are explicitly flagged in metadata as `image_assessed: false` and `ocr_assessed: false`. No multimodal visual claims are made.

---

## 14. Leakage and Deduplication Verification

- **Duplicate Quarantine:** 12 duplicate text instances segregated in `data/processed/phase10/duplicate_audit.csv`.
- **Group Leakage Checks:** Group IDs were audited across splits. Intersection of `source_group_id` across Train, Validation, and Test sets is **0.00%**. Zero leakage detected.

---

## 15. Leakage-Resistant Silver Dataset Partitions

Partitions saved under `data/splits/phase10/`:
- **`silver_train.csv`:** 10 records (7 `passive_aggressive`, 3 `non_passive_aggressive`)
- **`silver_validation.csv`:** 2 records (2 `passive_aggressive`, 0 `non_passive_aggressive`)
- **`silver_test.csv`:** 3 records (2 `passive_aggressive`, 1 `non_passive_aggressive`)

---

## 16. Exploratory Model Training Results

An exploratory baseline was trained using TF-IDF word (1-3) and character (2-5) n-grams with L2-regularized Logistic Regression ($C=0.1$):
- **Model Checkpoint:** `artifacts/phase10/pa_silver_tfidf_v1.pkl`
- **Metadata:** `artifacts/phase10/pa_silver_metadata.json`
- **Validation Macro-F1:** 1.0 (on 2 validation instances).

---

## 17. Diagnostic Test Evaluation & Honest Claims

Evaluation was conducted strictly on the isolated diagnostic test partition (`silver_test.csv`, $N=3$):
- **Accuracy:** 0.6667 (2 / 3 correct against silver labels)
- **Balanced Accuracy:** 0.5000
- **Macro-F1:** 0.4000
- **Per-Class Metrics:**
  - `passive_aggressive`: Precision = 0.6667, Recall = 1.0000, F1 = 0.8000 (Support: 2)
  - `non_passive_aggressive`: Precision = 0.0000, Recall = 0.0000, F1 = 0.0000 (Support: 1)
- **Confusion Matrix:**
  $$\begin{bmatrix} 0 & 1 \\ 0 & 2 \end{bmatrix}$$
- **CRITICAL GOVERNANCE WARNING:** These metrics evaluate statistical concordance with automated LLM silver labels on a minimal exploratory test split. They **do not** represent generalization to human social media communication or benchmark performance against human ground truth.

---

## 18. Automated Test Suite Results

The complete test suite was executed in `internvl_env`:
- **Repository-wide Tests:** 203 passed, 0 failed, 0 errors, 14 expected deprecation warnings.
- **Phase 10 Dedicated Tests:**
  - `tests/test_phase10_automated_annotation.py`: 20/20 passed.
  - `tests/test_phase10_api_experimental.py`: 3/3 passed.
- **Zero Regressions:** Phases 1 through 9 tests continue to pass without modification.

---

## 19. Checkpoint and Artifact Manifest

- **Annotation Schema:** `schemas/phase10/passive_aggression_annotation_schema.json`
- **Annotation Protocol:** `docs/phase10/llm_annotation_prompt_v1.md`
- **Annotator A Raw Outputs:** `data/annotations/phase10/annotator_a.jsonl`
- **Annotator B Raw Outputs:** `data/annotations/phase10/annotator_b.jsonl`
- **Adjudicated Outputs:** `data/annotations/phase10/adjudicated_labels.jsonl`
- **Accepted Silver Labels:** `data/annotations/phase10/accepted_silver_labels.jsonl`
- **Ambiguous Quarantined:** `data/annotations/phase10/ambiguous_records.csv`
- **Dataset Splits:** `data/splits/phase10/silver_train.csv`, `silver_validation.csv`, `silver_test.csv`
- **Model Checkpoint:** `artifacts/phase10/pa_silver_tfidf_v1.pkl`
- **Model Metadata:** `artifacts/phase10/pa_silver_metadata.json`
- **Machine-Readable Summary:** `reports/phase10/phase10_results.json`

---

## 20. Reproduction Commands

To reproduce the complete pipeline in a single step:
```powershell
& "D:\NLP_PROJECT\internvl_env\Scripts\python.exe" scripts/phase10/run_phase10_pipeline.py --pilot-limit 30
```
To run all tests:
```powershell
& "D:\NLP_PROJECT\internvl_env\Scripts\python.exe" -m pytest tests/
```

---

## 21. Known Limitations

1. **Annotator Asymmetry:** Smaller parameter models (e.g. Qwen 2.5 1.5B) fail to calibrate on subtle pragmatic boundaries and tend toward hyper-liberal positive labeling.
2. **Text-Only Modality Constraint:** No local multimodal vision model was present. Sarcastic image-text incongruity could not be visually validated.
3. **Exploratory Sample Size:** With 15 accepted silver records, the test split ($N=3$) serves solely as a software diagnostic test of pipeline plumbing rather than a statistically significant scientific benchmark.

---

## 22. Human Benchmark Gate Status

The human evaluation gate remains strictly:

$$\mathbf{BLOCKED\_PENDING\_HUMAN\_ANNOTATION}$$

- `data/annotations/phase6a/annotator_1_responses.csv`: 0 responses recorded.
- `data/annotations/phase6a/annotator_2_responses.csv`: 0 responses recorded.
- API endpoint `POST /predict/passive-aggression`: Returns HTTP 503 Service Unavailable.

The automated pipeline status is:

$$\mathbf{LLM\_SILVER\_DATASET\_AVAILABLE\_FOR\_EXPLORATORY\_RESEARCH}$$

$$\mathbf{LLM\_SILVER\_DIAGNOSTIC\_EVALUATION\_AVAILABLE}$$

---

## 23. Recommended Next Milestone

1. **Human Crowdsourcing / Expert Annotation:** When funding/annotators become available, populate `data/annotations/phase6a/` using the versioned protocol to establish an unassailable human ground-truth benchmark.
2. **Vision Model Onboarding:** If GPU memory permits (e.g., 4-bit quantized Qwen2-VL or MiniCPM-V), integrate visual feature assessment into the multimodal passive-aggression pipeline.
3. **Active Learning Silver Expansion:** Scale automated annotation to the full 500-record eligible pool using high-confidence agreement filters and the adjudication firewall.
