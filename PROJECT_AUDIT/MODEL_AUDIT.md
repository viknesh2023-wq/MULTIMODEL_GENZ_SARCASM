# MODEL AND FEATURE AUDIT REPORT: MULTIMODAL SARCASM INTELLIGENCE SYSTEM

**Audit Date:** October 10, 2026 (Updated post-Milestone 4 execution)  
**Auditor:** Senior ML Engineer & NLP Research Scientist  
**Hardware Used:** NVIDIA GeForce RTX 4050 Laptop GPU (6 GB VRAM)  
**Execution Environment:** `D:\NLP_PROJECT\internvl_env`  

---

## 1. Overview of Models and Extracted Features

| Component | Target Artifact | Framework / Architecture | Dimension / Output | Verified Status |
| :--- | :--- | :--- | :---: | :--- |
| **XLM-R Text Sarcasm** | `models\xlmr_text_sarcasm_full\` | `FacebookAI/xlm-roberta-base` | 2 classes (Binary) | **TRAINED, TESTED, INTEGRATED** |
| **CLIP Visual Features** | `MULTIMODAL_SARCASM\CLIP_FEATURES\` | OpenAI CLIP ViT-B/32 | 512 dimensions | **EXTRACTED, INDEXED, VERIFIED** |
| **InternVL3-2B Features**| `MULTIMODAL_SARCASM\VLM_FEATURES\` | OpenGVLab InternVL3-2B | 1,536 dimensions | **EXTRACTED, INDEXED, VERIFIED** |
| **Multimodal Text Embeddings** | `MULTIMODAL_SARCASM\XLMR_FEATURES\`| XLM-R Fine-Tuned Pooled Encoder | 768 dimensions | **EXTRACTED, INDEXED, VERIFIED (M1)** |
| **Sentiment Model & 3-D Vectors** | `models\xlmr_sentiment_model\` | XLM-RoBERTa 3-Class Head | 3 classes (Pos/Neu/Neg) | **TRAINED, TESTED, EXTRACTED (M2)** |
| **Emotion Model & 28-D Vectors** | `models\xlmr_emotion_model\` | XLM-RoBERTa 28-Class BCE Head | 28 classes (GoEmotions) | **TRAINED, TESTED, EXTRACTED (M3)** |
| **2,847-D Assembled Matrices** | `MULTIMODAL_SARCASM\FUSION_DATA\` | Concatenated NumPy Matrices | 2,847 dimensions | **ASSEMBLED, RECONCILED (M4)** |
| **Multimodal Fusion Classifier** | `models\multimodal_fusion\` | 4-layer Mish MLP with Dropout | 2 classes (Binary) | **TRAINED, TESTED, BENCHMARKED (M4)**|
| **Explanation Generator** | `app\services\explanation_service.py` | Ollama `llama3.2:1b` (CPU) + Cognitive Reasoner | Structured JSON | **IMPLEMENTED, TESTED, VERIFIED (M6)** |

---

## 2. Multimodal Fusion Classifier Audit (Milestone 4 — Verified)

**Directory:** `D:\NLP_PROJECT\models\multimodal_fusion`  
**Architecture:** `MultimodalFusionMLP` (Linear 2847 $\to$ 1024 $\to$ Mish $\to$ Dropout 0.3 $\to$ Linear 512 $\to$ Mish $\to$ Dropout 0.2 $\to$ Linear 128 $\to$ Mish $\to$ Dropout 0.1 $\to$ Linear 2)  
**Input Dimension:** 2,847 ($768 + 512 + 1536 + 3 + 28 = 2,847$)  
**Training Samples:** 25,384 | **Validation Samples:** 3,891 | **Test Samples:** 3,802  
**Preprocessing:** `StandardScaler` fitted strictly on $X_{\text{train}}$ only (zero leakage)  
**Hardware & VRAM:** NVIDIA GeForce RTX 4050 Laptop GPU, Peak VRAM: **88.97 MB**  
**Training Duration:** 57.25 seconds (Early stopping at epoch 12, best epoch 5)  

### Empirical Evaluation Metrics (3,802 Untouched Test Samples)
Verified directly from `models\multimodal_fusion\test_results.json`:
- **Test Accuracy:** `0.796160` (~79.62%)
- **Test Macro F1:** `0.786393` (~78.64%)
- **Test Macro Precision:** `0.805620` (~80.56%)
- **Test Macro Recall:** `0.781671` (~78.17%)
- **Test Weighted F1:** `0.792039` (~79.20%)
- **Validation Macro F1:** `0.839880` (~83.99%)

### Confusion Matrix (Test Split)
```
                     Predicted Non-Sarcastic (0)    Predicted Sarcastic (1)
True Non-Sarcastic (0):            1,107                         559             (Total: 1,666)
True Sarcastic (1):                  216                       1,920             (Total: 2,136)
```

### Per-Class Performance
- **Non-Sarcastic (0):** Precision: 83.67%, Recall: 66.45%, **F1-Score: 74.07%** (Support: 1,666)
- **Sarcastic (1):** Precision: 77.45%, Recall: 89.89%, **F1-Score: 83.21%** (Support: 2,136)

---

## 3. 8-Way Modality Ablation Benchmark Audit (Milestone 4 — Verified)

Audited directly from `models\multimodal_fusion\ablation_benchmark_summary.csv`:

| Configuration | Description | Dim | Val F1 | Test Acc | Test Macro F1 | Weighted F1 | Class 0 F1 | Class 1 F1 | $\Delta$ F1 | Latency |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `text_only` | XLM-R Contextual Text | 768 | 0.6289 | 0.6281 | **0.6198** | 0.6268 | 0.5638 | 0.6758 | Baseline | 0.033 ms |
| `text_sentiment` | Text + Sentiment | 771 | 0.6754 | 0.6578 | **0.6445** | 0.6530 | 0.5758 | 0.7132 | **+2.47%** | 0.040 ms |
| `text_emotion` | Text + Emotion | 796 | 0.6614 | 0.6602 | **0.6469** | 0.6553 | 0.5783 | 0.7154 | **+2.71%** | 0.016 ms |
| `text_sentiment_emotion` | Text + Sent + Emo | 799 | 0.6753 | 0.6746 | **0.6582** | 0.6675 | 0.5834 | 0.7331 | **+3.84%** | 0.015 ms |
| `text_clip` | Text + CLIP ViT-B/32 | 1,280 | 0.8267 | 0.7898 | **0.7793** | 0.7852 | 0.7309 | 0.8276 | **+15.95%** | 0.017 ms |
| `text_vlm` | Text + InternVL3-2B | 2,304 | 0.8106 | 0.7609 | **0.7486** | 0.7555 | 0.6930 | 0.8042 | **+12.88%** | 0.034 ms |
| `text_clip_vlm` | Text + CLIP + InternVL | 2,816 | 0.8355 | 0.7922 | **0.7806** | 0.7869 | 0.7302 | 0.8311 | **+16.08%** | 0.105 ms |
| `full_fusion` | **Full Five-Source Fusion** | **2,847** | **0.8399** | **0.7962** | **0.7864** | **0.7920** | **0.7407** | **0.8321** | **+16.66%** | **0.104 ms** |

---

## 4. Reusable Inference Module Audit (`fusion_inference.py`)

- **Class:** `MultimodalFusionInference` in `D:\NLP_PROJECT\fusion\fusion_inference.py`
- **Model Checkpoint:** `fusion_model.pt`
- **Scaler Checkpoint:** `fusion_scaler.joblib`
- **Input Verification:** Rejects inputs not equal to (2847,), float32, or containing NaNs/Infs.
- **Latency:** **0.104 ms per sample** (~9,600 samples/sec on RTX 4050 GPU).
- **Execution:** Validated on dummy and real test vectors from $X_{\text{test}}$.

---

## 5. Local Explanation Reasoner Audit (Milestone 6 — Verified)

- **Module:** `backend\app\services\explanation_service.py`
- **Primary LLM:** Ollama `llama3.2:1b` (4-bit quantized, 1.3 GB) served locally via `http://127.0.0.1:11434`
- **Execution Hardware:** Strictly pinned to **Host CPU** (`options: {"num_gpu": 0}`) to prevent CUDA OOM on the 5.996 GB RTX 4050 Laptop GPU.
- **VRAM Leakage:** **0.00 MB measured delta** during explanation synthesis across all benchmark tests.
- **Visual Evidence Derivation:** Grounded in resident `InternVL3-2B` vision-language model (`<image>\nBriefly describe what is in this image in one factual sentence.`).
- **Fail-Safe Fallback:** Deterministic Cognitive Reasoner (<1 ms execution time) guaranteeing 100% uptime if Ollama is unreachable or generates malformed output.
- **Empirical Grounding Validation:** 4/4 benchmark test cases passed all 8 scientific and architectural checks (`backend/tests/evaluate_explanation_quality.py`).
- **Prediction Invariance:** Underlying classifier labels and probabilities are strictly invariant under explanation toggle ($\Delta p < 10^{-6}$).

