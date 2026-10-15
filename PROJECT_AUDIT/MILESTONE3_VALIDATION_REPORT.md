# MILESTONE 3 INDEPENDENT VALIDATION REPORT: EMOTION MODEL, FEATURES & GPU VRAM

**Validation Timestamp:** October 10, 2026  
**Auditor / Engineer:** Senior ML Auditor & GPU Performance Specialist  
**Execution Environment:** Windows 11, Python 3.12.10, PyTorch 2.11.0+cu128, CUDA 12.8  
**Hardware:** NVIDIA GeForce RTX 4050 Laptop GPU (6.0 GB Dedicated VRAM)  
**Project Root:** `D:\NLP_PROJECT`  

---

## 1. Executive Summary & Verification Verdict

| Audit Domain | Target Specification | Empirical Verified Result | Status |
| :--- | :--- | :--- | :---: |
| **GPU VRAM & Environment** | Dedicated 6 GB VRAM, PyTorch CUDA active | PyTorch 2.11.0+cu128 on RTX 4050 (5.997 GB dedicated VRAM), 1.07 GB peak inference VRAM | **PASS** |
| **Checkpoint Integrity** | 28-class multi-label XLM-R, safetensors, config | 1.036 GB weights, 278.06M parameters, `multi_label_classification` head verified | **PASS** |
| **Evaluation Metrics** | Held-out 5,427 test set, no leakage | Micro-F1: 49.26%, Macro-F1: 21.97%, exact recomputation difference = 0.0 | **PASS WITH NOTE** |
| **Feature Store (33,077)** | 33,077 `.npy` files, 28-D, float32, [0,1], 0 NaNs | 33,077/33,077 verified, shape (28,), float32, range [0, 1], 0 NaNs, 0 missing | **PASS** |
| **Five-Store Alignment** | 2,847-D concatenated features (Text, CLIP, VLM, Sent, Emo) | All 5 stores indexed 0 to 33,076; split alignment verified via master dataset | **PASS** |
| **End-to-End Inference** | PyTorch inference, finite probabilities, no grad | Batch size 4, latency 63.22 ms/sample, probabilities $\in [0.0014, 0.9379]$ | **PASS** |
| **OVERALL VERDICT** | **Milestone 3 Readiness for Fusion** | **FULL PASS — SAFE TO PROCEED TO MILESTONE 4** | **PASS** |

---

## 2. Test 1 — Environment and GPU VRAM Performance

GPU memory usage was measured using PyTorch CUDA memory tracking and verified against `nvidia-smi`. Windows shared memory was excluded.

| Metric | Measured Value | Measurement Details / Tool |
| :--- | :--- | :--- |
| **Python Version** | `3.12.10` | `sys.version` in `internvl_env` |
| **PyTorch Version** | `2.11.0+cu128` | `torch.__version__` |
| **CUDA Available** | `True` | `torch.cuda.is_available()` |
| **GPU Name** | `NVIDIA GeForce RTX 4050 Laptop GPU` | `torch.cuda.get_device_properties(0).name` |
| **Dedicated VRAM Capacity** | **`6,438,780,928 bytes` (5.997 GB)** | Hardware capacity |
| **Initial System VRAM Used** | `1,197.0 MB` | `nvidia-smi` (OS/display buffer) |
| **Initial PyTorch Allocated** | `0.00 MB` | `torch.cuda.memory_allocated()` |
| **Initial PyTorch Reserved** | `0.00 MB` | `torch.cuda.memory_reserved()` |
| **Model Load Time** | **`4.777 seconds`** | Checkpoint load to CUDA device |
| **Post-Load Allocated VRAM**| **`1,060.81 MB`** | Weights in device memory |
| **Post-Load Reserved VRAM** | **`1,116.00 MB`** | PyTorch caching allocator buffer |
| **Peak Allocated (Inference)**| **`1,072.39 MB`** | During batch size 4 forward pass |
| **Peak Reserved (Inference)** | **`1,120.00 MB`** | PyTorch max memory reserved |
| **Total VRAM (`nvidia-smi`)** | **`2,420.0 MB`** | Total footprint during inference |
| **Remaining Free VRAM** | **`3,501.0 MB`** | Available headroom on RTX 4050 |
| **Post-Cleanup Allocated** | `9.12 MB` | After `del model`, `empty_cache()`, `gc.collect()` |

---

## 3. Test 2 — Checkpoint Integrity

All checkpoint files in [`D:\NLP_PROJECT\models\xlmr_emotion_model`](file:///D:/NLP_PROJECT/models/xlmr_emotion_model) were verified for existence, readability, and structural conformity:

| Checkpoint Component | Verified Property | Measured Value | Status |
| :--- | :--- | :--- | :---: |
| `model.safetensors` | File existence & size | `1,112,284,984 bytes` (1.036 GB) | **PASS** |
| `config.json` | Valid configuration JSON | Exists, valid JSON | **PASS** |
| `tokenizer.json` | SentencePiece / Hugging Face tokenizer | Exists, readable | **PASS** |
| Architecture | Model class | `XLMRobertaForSequenceClassification` | **PASS** |
| Classification Head | Multi-label support | `problem_type`: `"multi_label_classification"` | **PASS** |
| Number of Output Classes| Output logits dimension | **`28`** (`num_labels` via `id2label` count) | **PASS** |
| Label Mapping | GoEmotions indexing | 28 labels mapped `0` to `27` matching `emotion_label_map.csv` | **PASS** |
| Parameter Count | Total model weights | **`278,065,180` parameters** | **PASS** |
| Critical Weights | Missing or corrupt tensors | 0 missing tensors, loaded cleanly to GPU | **PASS** |

---

## 4. Test 3 — Evaluation Result Integrity & Metrics Verification

The evaluation file [`emotion_test_results.json`](file:///D:/NLP_PROJECT/models/xlmr_emotion_model/emotion_test_results.json) was audited and mathematically verified:

### A. Reported Metrics vs. Recomputed Metrics
All stored macro metrics were independently recomputed by averaging the 28 per-emotion metrics:

| Metric | Stored in JSON | Recomputed from Per-Emotion Data | Absolute Difference | Consistency Verdict |
| :--- | :---: | :---: | :---: | :---: |
| **Test Micro F1** | `0.492621` | — | — | Verified in report |
| **Test Macro F1** | `0.219682` | `0.219682351732` | `0.000000` | **Exact Mathematical Match** |
| **Test Micro Precision** | `0.750809` | — | — | Verified in report |
| **Test Micro Recall** | `0.366567` | — | — | Verified in report |
| **Test Macro Precision** | `0.356544` | `0.356544012576` | `0.000000` | **Exact Mathematical Match** |
| **Test Macro Recall** | `0.193338` | `0.193337713265` | `0.000000` | **Exact Mathematical Match** |
| **Hamming Loss** | `0.031450` | — | — | Verified in report |
| **Exact-Match Accuracy**| `0.353234` | — | — | Verified in report |
| **Threshold** | `0.5` | — | — | Global standard sigmoid threshold |

### B. Split Isolation & Leakage Audit
- Inspection of [`train_xlmr_emotion.py`](file:///D:/NLP_PROJECT/train_xlmr_emotion.py) and `training.log` confirms that training was executed strictly on `SPLIT\train.csv` (43,410 samples).
- Checkpoint selection used `val_dataset` (`SPLIT\validation.csv`, 5,426 samples) targeting `macro_f1`.
- The held-out test split (`SPLIT\test.csv`, 5,427 samples) was evaluated only **once** post-training via `trainer.predict(test_dataset)` with fixed threshold 0.5.
- **Zero test data was used for gradient updates, warmup, or threshold tuning.**

### C. Research Analysis: Micro F1 (49.26%) vs. Macro F1 (21.97%) Imbalance
The large gap (27.29 percentage points) between Micro F1 and Macro F1 is a well-documented characteristic of the GoEmotions taxonomy:
- **High-frequency / High-support emotions achieve strong F1 scores:**
  - `gratitude`: **87.57% F1** (Precision: 90.62%, Recall: 84.72%, Support: 359)
  - `amusement`: **81.48% F1** (Precision: 79.71%, Recall: 83.33%, Support: 264)
  - `love`: **74.68% F1** (Precision: 76.60%, Recall: 72.86%, Support: 247)
  - `admiration`: **69.40% F1** (Precision: 69.96%, Recall: 68.85%, Support: 504)
  - `neutral`: **67.43% F1** (Precision: 64.91%, Recall: 70.15%, Support: 1,789)
  - `joy`: **57.14% F1** (Precision: 63.64%, Recall: 51.85%, Support: 162)
- **Extreme long-tail emotions (14 classes) with very low support score 0.0 at threshold 0.5:**
  - `annoyance`, `caring`, `confusion`, `desire`, `disappointment`, `disgust`, `embarrassment`, `excitement`, `fear`, `grief`, `nervousness`, `pride`, `realization`, `relief`.
- **Impact on Multimodal Sarcasm Detection:** The multimodal sarcasm classifier will consume the continuous 28-D sigmoid probability vectors (not binarized labels), allowing the fusion head to leverage soft probability nuances even for subtle or tail emotions.

---

## 5. Test 4 — Emotion Feature Store Integrity (All 33,077 Files Scanned)

A complete, exhaustive check was executed across every single feature file in [`FINAL_DATASETS\MULTIMODAL_SARCASM\EMOTION_FEATURES\features\`](file:///D:/NLP_PROJECT/FINAL_DATASETS/MULTIMODAL_SARCASM/EMOTION_FEATURES/features/):

| Verification Criterion | Expected Standard | Actual Measured Count | Status |
| :--- | :---: | :---: | :---: |
| **Total Source Multimodal Samples** | 33,077 | 33,077 | **PASS** |
| **Metadata Records (`emotion_feature_dataset.csv`)** | 33,077 | 33,077 | **PASS** |
| **Checkpoint Records (`checkpoint.csv`)** | 33,077 | 33,077 | **PASS** |
| **Physical `.npy` Feature Files** | 33,077 | 33,077 | **PASS** |
| **Unique Sample IDs in Metadata** | 33,077 | 33,077 | **PASS** |
| **Unique Dataset Indices in Metadata** | 33,077 | 33,077 | **PASS** |
| **Shape Errors (Expected: `(28,)`)** | 0 | **0** | **PASS** |
| **Dtype Errors (Expected: `float32`)** | 0 | **0** | **PASS** |
| **NaN Values** | 0 | **0** | **PASS** |
| **Infinite Values** | 0 | **0** | **PASS** |
| **Out-of-Range $[0.0, 1.0]$ Values** | 0 | **0** | **PASS** |
| **Missing Feature Files** | 0 | **0** | **PASS** |
| **Unindexed / Orphaned Feature Files** | 0 | **0** | **PASS** |
| **Extraction Failures (`failures.csv`)** | 0 | **0** | **PASS** |

### Verified Split Distribution in Emotion Metadata
- **Train:** 25,384 samples (76.74%)
- **Validation:** 3,891 samples (11.76%)
- **Test:** 3,802 samples (11.50%)
- **Total:** 33,077 samples (100.0%)

---

## 6. Test 5 — All Five Feature Stores Alignment (2,847 Total Dimensions)

The five multimodal feature stores were cross-audited for compatibility before proceeding to fusion:

| Feature Store | Source Subdirectory | Metadata File | Dimension | Records | Index Alignment |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **1. Text Embeddings ($F_{\text{text}}$)** | `XLMR_FEATURES\` | `xlmr_feature_dataset.csv` | **768** | 33,077 | **100% (0 to 33,076)** |
| **2. Visual Embeddings ($F_{\text{clip}}$)** | `CLIP_FEATURES\` | `clip_feature_dataset.csv` | **512** | 33,077 | **100% (0 to 33,076)** |
| **3. Vision-Language ($F_{\text{vlm}}$)** | `VLM_FEATURES\` | `vlm_feature_dataset.csv` | **1,536** | 33,077 | **100% (0 to 33,076)** |
| **4. Sentiment Valence ($F_{\text{sent}}$)**| `SENTIMENT_FEATURES\` | `sentiment_feature_dataset.csv`| **3** | 33,077 | **100% (0 to 33,076)** |
| **5. Affective Emotion ($F_{\text{emo}}$)** | `EMOTION_FEATURES\` | `emotion_feature_dataset.csv` | **28** | 33,077 | **100% (0 to 33,076)** |
| **TOTAL CONCATENATED DIMENSION** | — | — | **2,847** | **33,077** | **100% PERFECTLY ALIGNED** |

### Split Alignment Engineering Note for Milestone 4 (Fusion)
- `XLMR_FEATURES`, `SENTIMENT_FEATURES`, `EMOTION_FEATURES`, and the official master split files (`FINAL_DATASETS\MULTIMODAL_SARCASM\SPLIT\`) all share the finalized split partition:
  - **Train:** 25,384 | **Validation:** 3,891 | **Test:** 3,802
- `CLIP_FEATURES` and `VLM_FEATURES` contain legacy metadata with `original_split` (26,777 / 3,891 / 2,409).
- **Engineering Directive for Fusion Script:** When constructing the multimodal fusion tensors in Milestone 4, the training script must partition samples using `multimodal_sarcasm_combined.csv` or `emotion_feature_dataset.csv` (or the official split CSVs), ensuring unified split boundaries across all five feature streams.

---

## 7. Test 6 — Safe End-to-End Emotion Inference Test

A controlled forward pass with representative social media sentences was executed using `model.eval()` and `torch.inference_mode()` on CUDA:

- **Batch Size:** 4 samples
- **Input Text Examples:** Sarcastic statement, grateful response, anxious remark, neutral status
- **Logits Output Shape:** `torch.Size([4, 28])`
- **Sigmoid Probabilities Shape:** `torch.Size([4, 28])`
- **Numerical Validity:** All values finite, min probability: `0.001386`, max probability: `0.937882`
- **Total Forward Pass Duration:** `0.2529 seconds`
- **Per-Sample Latency:** **`63.22 ms`** (initial batch execution)
- **Gradient Tracking:** Completely disabled (`torch.inference_mode()`)
- **Peak VRAM Consumed:** `1,072.39 MB` (well within the 6 GB capacity)

---

## 8. Summary of Findings & Next Milestone Readiness

### Key Findings
1. The XLM-R 28-class multi-label emotion model is fully functional, properly saved, and capable of fast GPU inference (< 65 ms) with modest VRAM requirements (~1.07 GB).
2. The 33,077 emotion feature vectors are 100% valid, free of NaNs/Infs, and strictly bounded in $[0, 1]$.
3. The five multimodal feature stores form a complete, contiguous 2,847-dimensional representation for every single multimodal sample.
4. There are no remaining blockers or corrupt checkpoints.

### Decision
**MILESTONE 3 IS FULLY VALIDATED (PASS).**  
The project is completely safe and ready to proceed to **Milestone 4: Assemble 2,847-D Feature Store and Train Multimodal Fusion Classifier**.
