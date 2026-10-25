# Phase 10 Preflight Verification Report
## Fully Automated LLM-Based Passive-Aggression Annotation, Quality Auditing, Silver Dataset Construction, and Research Evaluation

**Project Title:** Context-Aware Multimodal Passive-Aggression Detection in Social Media Using Image–Comment Fusion  
**Artifact:** `reports/phase10/phase10_preflight_report.md`  
**Execution Date:** October 22, 2026  
**Canonical Project Root:** `D:\NLP_PROJECT\passive_aggression_multimodal`  
**Active Linked Workspace:** `D:\multimodel_passive_aggression`  
**Evaluation Gate:** `BLOCKED_PENDING_HUMAN_ANNOTATION` (Strictly Maintained)  
**Silver Dataset Gate:** `LLM_SILVER_DATASET_AVAILABLE_FOR_EXPLORATORY_RESEARCH`  

---

## 1. Environment & Local Infrastructure Verification

The execution environment was audited and verified for Phase 10 automated annotation:
- **Operating System:** Windows 11 Home (Build 26100)
- **Active Python Environment:** `D:\NLP_PROJECT\internvl_env\Scripts\python.exe` (Python 3.12.10, PyTorch `2.11.0+cu128`, CUDA 12.8, RTX 4050 6GB Laptop GPU)
- **Local Ollama Daemon:** Verified online at `http://localhost:11434`
  - Model 1: `llama3.2:latest` (3.2B parameters, family: `llama`, size: 1,925.8 MB) — **Designated Annotator A**
  - Model 2: `qwen2.5:1.5b` (1.5B parameters, family: `qwen2`, size: 940.4 MB) — **Designated Annotator B**
  - Model 3: `llama3.2:1b` (1.2B parameters, family: `llama`, size: 1,259.9 MB) — **Designated Adjudicator Alternative / Arbiter**
- **Inference Verification:** Confirmed that all three local models generate valid JSON with structured schema enforcement via Ollama's native `/api/chat` interface.
- **Modality Scope:** All installed Ollama models are text-only. In accordance with Section 14, image-bearing records evaluated by these models will be explicitly reported as `image_assessed = false` and `modality_used = ["text"]`.

---

## 2. Inventory of Prior Annotation Artifacts & Provenance

| Artifact / Path | Format | Records | Status & Role |
|---|---|---|---|
| `data/annotations/phase6a/pilot_sampling_manifest.csv` | CSV | 30 | Pilot sampling manifest (6 MUStARD, 10 Civil Comments, 10 MMSD, 4 SarcNet) |
| `data/annotations/phase6a/annotator_1_responses.csv` | CSV | 30 | **Human Template (0 completed, strictly preserved)** |
| `data/annotations/phase6a/annotator_2_responses.csv` | CSV | 30 | **Human Template (0 completed, strictly preserved)** |
| `data/annotations/phase6d/annotator_1_llm_responses.csv` | CSV | 30 | Phase 6D LLM Annotator 1 outputs (`llama3.2:latest`) |
| `data/annotations/phase6d/annotator_2_llm_responses.csv` | CSV | 30 | Phase 6D LLM Annotator 2 outputs (`qwen2.5:1.5b`) |
| `data/annotations/phase6d/llm_adjudication_results.csv` | CSV | 30 | Phase 6D LLM Adjudication outputs |
| `data/annotations/phase5d/silver_dataset_verified.jsonl` | JSONL | 19 | Phase 5D verified silver subset (19 records) |
| `data/manifests/mmsd_manifest.jsonl` | JSONL | 24,635 | Multimodal sarcasm corpus with hydrated text & image references |
| `data/manifests/civil_comments_manifest.jsonl` | JSONL | 10,000 | Toxicity/social news comments with nuanced negative boundaries |
| `data/manifests/mustard_manifest.jsonl` | JSONL | 690 | Sitcom dialogue turns with rich conversational context |

---

## 3. Strict Scientific & Governance Boundaries

1. **Task Separation:** Sarcasm $\neq$ Passive Aggression, Toxicity $\neq$ Passive Aggression. Source dataset labels will never be converted into passive-aggression ground truth.
2. **Honest Label Provenance:** Labels produced by LLMs will bear explicit provenance identifiers (`LLM_SILVER`, `LLM_CROSS_MODEL_AGREEMENT`, `LLM_ADJUDICATED`). The tag `HUMAN_VERIFIED` is strictly prohibited.
3. **Dual Gate Architecture:**
   - Primary Human Evaluation Gate: `BLOCKED_PENDING_HUMAN_ANNOTATION` (Locked).
   - Exploratory Silver Gate: `LLM_SILVER_DATASET_AVAILABLE_FOR_EXPLORATORY_RESEARCH` (Active).
4. **Human Response File Protection:** Human template files (`annotator_1_responses.csv`, `annotator_2_responses.csv`) remain completely untouched.

---

## 4. Phase 10 Execution Plan

1. Define versioned annotation schema (`schemas/phase10/passive_aggression_annotation_schema.json`) and neutral operational prompt (`docs/phase10/llm_annotation_prompt_v1.md`).
2. Construct a stratified eligible corpus of 100 diverse candidate records across MUStARD, Civil Comments, MMSD, and SarcNet, incorporating hard negatives (direct aggression, benign sarcasm, sincere praise, institutional critique).
3. Apply cryptographic text deduplication and duplicate-group isolation.
4. Execute Pass 1 (`llama3.2:latest`) and Pass 2 (`qwen2.5:1.5b`) independent annotations with schema validation and resume caching.
5. Analyze inter-model agreement (raw %, Cohen's $\kappa$, subtype agreement).
6. Execute Pass 3 automated adjudication on disagreements using an impartial adjudication judge.
7. Apply conservative silver acceptance rules to create accepted silver, ambiguous, and excluded partitions.
8. Build group-isolated train/val/diagnostic test splits.
9. Train and evaluate an exploratory passive-aggression model (`pa_silver_tfidf_v1`) on the accepted silver dataset.
10. Implement automated test suite (`tests/test_phase10_automated_annotation.py`) and verify repository tests.
11. Safely integrate exploratory endpoint into FastAPI application.
