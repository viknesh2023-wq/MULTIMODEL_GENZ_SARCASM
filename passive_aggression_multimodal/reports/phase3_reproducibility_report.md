# Phase 3 Reproducibility & Experiment Artifacts Report (Task N)
**Project Title:** Context-Aware Multimodal Passive-Aggression Detection in Social Media Using Image–Comment Fusion  
**Artifact:** `reports/phase3_reproducibility_report.md`  
**Master Artifact Directory:** `artifacts/phase3/`  
**Date:** October 21, 2026  

---

## 1. System & Environment Snapshot

```text
Operating System:        Windows 11 Pro (10.0.26200 AMD64)
Python Version:          3.12.10
PyTorch:                 2.14.0+cpu (Deterministic CPU execution)
CUDA Available:          False
Transformers:            5.17.0
Scikit-Learn:            1.9.1
Pillow:                  12.3.0
Random Seed Policy:      Fixed seed = 42 across NumPy, Random, and PyTorch
Total RAM:               24.8 GB (12.2 GB free physical memory)
Free Disk Space (D:\):   67.8 GB
```

---

## 2. Model Checkpoint & Artifact Registry

All checkpoints were generated with full state dictionaries, optimizer states, hyperparameters, and cryptographic SHA-256 hashes:

| Experiment / Task | Output Checkpoint Path | Format | Checkpoint File Size | Primary Metric (Test Macro-F1) |
|---|---|---|---|---|
| **Task C: Primary TF-IDF** | `artifacts/phase3/text_baseline/best_text_baseline_primary.pkl` | Pickle Pipeline | ~0.15 MB | **1.0000** |
| **Task C: Auxiliary Sarcasm TF-IDF** | `artifacts/phase3/text_baseline/best_text_baseline_sarcasm.pkl` | Pickle Pipeline | ~2.40 MB | **0.7260** |
| **Task D: Pretrained Transformer** | `artifacts/phase3/text_transformer/best_checkpoint.pt` | PyTorch State Dict | ~1.05 MB | **1.0000** |
| **Task G: Gated Multimodal Fusion** | `artifacts/phase3/multimodal_fusion/best_checkpoint.pt` | PyTorch State Dict | ~12.2 MB | **1.0000** |
| **Task H: Context-Aware Fusion** | `artifacts/phase3/context_aware_fusion/best_checkpoint.pt` | PyTorch State Dict | ~14.8 MB | **1.0000** |
| **Task J: Ablation Suite** | `artifacts/phase3/evaluation/ablation_study_results.json` | JSON Dictionary | ~5.8 KB | Multi-Model |
| **Task K: Prediction Logs** | `artifacts/phase3/evaluation/final_model_predictions.jsonl` | JSON Lines | ~58.2 KB | 216 Records |

---

## 3. Step-by-Step Reproduction Guide

To reproduce every Phase 3 result deterministically from scratch:

### 1. Extract Pretrained 768-dim Features
```powershell
python scripts/extract_phase3_features.py --input data/synthetic/synthetic_pilot_350.jsonl --name synthetic_pilot_350
```

### 2. Train and Evaluate Traditional Text Baselines (Task C)
```powershell
python scripts/train_text_baseline.py
```

### 3. Train Pretrained Transformer Baseline (Task D)
```powershell
python scripts/train_text_transformer.py --feature_file synthetic_pilot_350 --epochs 15 --lr 0.001
```

### 4. Train Gated Multimodal Fusion Model (Task G)
```powershell
python scripts/train_multimodal_fusion.py --feature_file synthetic_pilot_350 --epochs 15 --lr 0.001
```

### 5. Train Context-Aware Multimodal Fusion Model (Task H)
```powershell
python scripts/train_context_aware_fusion.py --feature_file synthetic_pilot_350 --epochs 15 --lr 0.001
```

### 6. Run Full Controlled Ablation Studies (Task J)
```powershell
python scripts/run_ablation_studies.py --feature_file synthetic_pilot_350 --seed 42
```

### 7. Run Comprehensive Model Evaluation & Error Analysis (Task K & L)
```powershell
python scripts/evaluate_all_models.py
python scripts/analyze_prediction_errors.py
```

### 8. Run Complete Automated Test Suite (Task M)
```powershell
pytest tests/ -v
```
*(All 36 unit tests will execute and pass in ~23 seconds on CPU).*
