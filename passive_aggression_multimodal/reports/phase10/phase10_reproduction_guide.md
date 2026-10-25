# Phase 10 — Reproduction Guide: Automated LLM-Based Silver Pipeline

## 1. Project Context & Environment

- **Project Root (Canonical):** `D:\NLP_PROJECT\passive_aggression_multimodal`
- **Active Workspace:** `D:\multimodel_passive_aggression`
- **Python Environment:** `D:\NLP_PROJECT\internvl_env\Scripts\python.exe` (Python 3.12.10, PyTorch 2.11.0+cu128)
- **Local LLM Engine:** Ollama (`http://localhost:11434`)
  - Annotator A: `llama3.2:latest` (3.2B)
  - Annotator B: `qwen2.5:1.5b` (1.5B)
  - Adjudicator: `llama3.2:latest` (3.2B)

---

## 2. Prerequisites & Verification

### A. Verify Local Ollama Service
Before running any annotation script, confirm Ollama is responding and the required models are pulled:
```powershell
# In PowerShell:
curl http://localhost:11434/api/tags
ollama list
```
Expected output includes:
- `llama3.2:latest`
- `qwen2.5:1.5b`

### B. Activate Virtual Environment
```powershell
Set-Location -Path "D:\multimodel_passive_aggression"
$env:PYTHONPATH = "D:\multimodel_passive_aggression"
```

---

## 3. End-to-End Automated Pipeline Execution

To run the entire Phase 10 automated workflow in a single command, execute the master orchestrator:

```powershell
& "D:\NLP_PROJECT\internvl_env\Scripts\python.exe" scripts/phase10/run_phase10_pipeline.py --pilot-limit 30
```

The orchestrator executes the following sequence:
1. `prepare_eligible_corpus.py`: Stratifies eligible records and builds duplicate quarantine manifests.
2. `run_llm_annotation.py`: Runs Annotator A (`llama3.2:latest`) and Annotator B (`qwen2.5:1.5b`) independently with caching and resume support.
3. `analyze_llm_agreement.py`: Evaluates raw agreement, Cohen's $\kappa$, per-class agreement, and generates disagreement taxonomies.
4. `run_llm_adjudication.py`: Runs third-pass adjudication over disagreements with boundary condition auditing.
5. `build_silver_dataset.py`: Applies strict acceptance rules, separates ambiguous records, and creates group-isolated splits.
6. `train_pa_silver_baseline.py`: Fits an exploratory n-gram Logistic Regression model (`pa_silver_tfidf_v1`) on `silver_train.csv` and evaluates on `silver_test.csv`.

---

## 4. Step-by-Step Manual Execution

If you wish to execute each stage independently:

### Step 1: Prepare Eligible Corpus & Deduplication
```powershell
& "D:\NLP_PROJECT\internvl_env\Scripts\python.exe" scripts/phase10/prepare_eligible_corpus.py
```
**Outputs:**
- `data/annotations/phase10/eligible_records.csv`
- `data/annotations/phase10/sampling_manifest.csv`
- `data/processed/phase10/duplicate_audit.csv`
- `data/processed/phase10/duplicate_group_manifest.csv`

### Step 2: Run Dual LLM Annotation
```powershell
& "D:\NLP_PROJECT\internvl_env\Scripts\python.exe" scripts/phase10/run_llm_annotation.py --model-a llama3.2:latest --model-b qwen2.5:1.5b --limit 30
```
**Outputs:**
- `data/annotations/phase10/annotator_a.jsonl`
- `data/annotations/phase10/annotator_b.jsonl`
- `data/annotations/phase10/annotation_run_manifest.json`

### Step 3: Compute Agreement & Disagreement Taxonomy
```powershell
& "D:\NLP_PROJECT\internvl_env\Scripts\python.exe" scripts/phase10/analyze_llm_agreement.py
```
**Outputs:**
- `reports/phase10/llm_agreement_report.md`
- `reports/phase10/tables/label_distribution.csv`
- `reports/phase10/tables/disagreement_categories.csv`
- `reports/phase10/tables/source_breakdown.csv`

### Step 4: Run Adjudication on Disagreements
```powershell
& "D:\NLP_PROJECT\internvl_env\Scripts\python.exe" scripts/phase10/run_llm_adjudication.py --model llama3.2:latest
```
**Outputs:**
- `data/annotations/phase10/adjudicated_labels.jsonl`
- `data/annotations/phase10/adjudication_manifest.json`
- `reports/phase10/adjudication_audit_report.md`

### Step 5: Build Silver Dataset & Isolated Splits
```powershell
& "D:\NLP_PROJECT\internvl_env\Scripts\python.exe" scripts/phase10/build_silver_dataset.py
```
**Outputs:**
- `data/annotations/phase10/accepted_silver_labels.jsonl`
- `data/annotations/phase10/ambiguous_records.csv`
- `data/annotations/phase10/excluded_records.csv`
- `data/splits/phase10/silver_train.csv`
- `data/splits/phase10/silver_validation.csv`
- `data/splits/phase10/silver_test.csv`
- `data/splits/phase10/split_manifest.json`

### Step 6: Train Exploratory Silver Baseline Model
```powershell
& "D:\NLP_PROJECT\internvl_env\Scripts\python.exe" scripts/phase10/train_pa_silver_baseline.py
```
**Outputs:**
- `artifacts/phase10/pa_silver_tfidf_v1.pkl`
- `artifacts/phase10/pa_silver_metadata.json`
- `reports/phase10/silver_model_evaluation_report.md`

---

## 5. Automated Test Suite Execution

To run the complete repository test suite (203 tests across Phases 1 through 10):
```powershell
& "D:\NLP_PROJECT\internvl_env\Scripts\python.exe" -m pytest -v tests/
```

To run only the Phase 10 test modules:
```powershell
& "D:\NLP_PROJECT\internvl_env\Scripts\python.exe" -m pytest -v tests/test_phase10_automated_annotation.py tests/test_phase10_api_experimental.py
```

Expected output:
```
======================= 23 passed, 3 warnings in 16.92s =======================
```

---

## 6. Testing the Experimental API Endpoint

### A. Start the FastAPI Server
```powershell
& "D:\NLP_PROJECT\internvl_env\Scripts\python.exe" -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000
```

### B. Test the Experimental Passive-Aggression Endpoint
In a separate terminal:
```powershell
curl -X POST "http://127.0.0.1:8000/predict/passive-aggression/experimental" `
  -H "Content-Type: application/json" `
  -d '{"text": "Oh sure, thanks for letting me know after the meeting already ended!"}'
```

**Expected JSON Response:**
```json
{
  "task": "passive_aggression_detection",
  "prediction": "passive_aggressive",
  "confidence": 0.58,
  "probabilities": {
    "non_passive_aggressive": 0.42,
    "passive_aggressive": 0.58
  },
  "model_id": "pa_silver_tfidf_v1",
  "model_type": "tfidf_logistic_regression",
  "label_provenance": "LLM_SILVER",
  "evaluation_status": "LLM_SILVER_DIAGNOSTIC_EVALUATION_AVAILABLE",
  "warning": "EXPERIMENTAL MODEL: Trained and evaluated on automated LLM silver labels. Not validated against independent human ground truth."
}
```

### C. Verify the Primary Human Evaluation Gate Remains Blocked (HTTP 503)
```powershell
curl -i -X POST "http://127.0.0.1:8000/predict/passive-aggression" `
  -H "Content-Type: application/json" `
  -d '{"text": "Testing primary gate."}'
```
**Expected HTTP Response:**
`503 Service Unavailable` with message:
`"Passive aggression model is blocked pending independent human verification. See reports/phase10/."`
