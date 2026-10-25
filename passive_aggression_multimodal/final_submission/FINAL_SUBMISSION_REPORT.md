# Final Submission Report: Context-Aware Multimodal Passive-Aggression Detection

**Project Title:** Context-Aware Multimodal Passive-Aggression Detection in Social Media Using Image–Comment Fusion  
**Canonical Project Root:** `D:\NLP_PROJECT\passive_aggression_multimodal`  
**Active Linked Workspace:** `D:\multimodel_passive_aggression`  
**Execution Environment:** Windows 11, Python 3.12.10, PyTorch 2.11.0+cu128, NVIDIA GeForce RTX 4050 Laptop GPU (6 GB VRAM)  
**Verification Baseline:** 203 / 203 Automated Unit & Integration Tests Passing (0 Failures)  
**Submission Date:** October 22, 2026  

---

## 1. Project Title & Problem Statement

### Title
Context-Aware Multimodal Passive-Aggression Detection in Social Media Using Image–Comment Fusion

### Problem Statement
Online communication frequently employs passive aggression—an indirect, veiled form of hostility where interpersonal resentment, blame, or criticism is masked behind superficial politeness, ambiguous sarcasm, or plausible deniability. While standard sentiment and direct toxicity models identify explicit abuse, they fail systematically on passive aggression because the hostility is not stated overtly. 

Furthermore, on modern visual platforms (e.g., Reddit, Twitter/X, Instagram), passive aggression is often **multimodal**: an innocuous or polite comment (e.g., *"What a beautiful view!"*) becomes hostile only when paired with incongruent visual context (e.g., a photo of a flooded highway or a broken appliance).

### Central Research Question
> **Can a multimodal deep learning system identify passive-aggressive communication patterns more reliably when combining post image, caption, and comment context than when analyzing text alone?**

---

## 2. Objectives and Scope

1. **Pragmatic Grounding:** Establish an operational, three-prong linguistic definition distinguishing genuine passive aggression from direct aggression, harmless sarcasm, and ordinary negative sentiment.
2. **Leakage-Resistant Dataset Architecture:** Build an audited multimodal data pipeline across 48,660 social media records and 27,970 verified images with zero cryptographic group leakage.
3. **Multimodal Model Engineering:** Train and evaluate text-only baselines, pretrained Transformer heads (XLM-RoBERTa), and Gated Multimodal Fusion architectures.
4. **Architectural & Empirical Auditing:** Investigate multimodal gating dynamics and modality contributions through controlled ablation studies.
5. **Research Governance & Safety Firewall:** Enforce strict epistemic separation between auxiliary sarcasm detection, automated LLM silver labels, and human ground truth, preserving a locked human evaluation gate (`BLOCKED_PENDING_HUMAN_ANNOTATION`).
6. **Production Serving:** Deploy a production-ready, stateless FastAPI inference engine with automatic image fallback and an interactive demonstration interface.

---

## 3. Dataset Sources, Counts, Preprocessing & Licensing Limitations

### Dataset Inventory

| Dataset Collection | Modality | Raw Records | Audited Images | Language | Licensing & Terms |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **MMSD 2.0** | Image + Tweet Text | 24,635 | 24,635 decodable | English (minor non-En) | Academic Research Only (Twitter TOS) |
| **SarcNet** | Image + Reddit Comment | 3,335 | 3,335 decodable | English | Academic Research Only (Reddit API) |
| **MUStARD** | Video/Audio/Dialogue Context | 690 | N/A (Dialogue text) | English | CC BY-NC 4.0 |
| **Civil Comments** | Hard Negative Toxicity Text | 20,000 | Text-only | English | CC0 Public Domain |
| **Total Curated** | **Multimodal / Text** | **48,660** | **27,970 decodable** | — | **Fully documented in [LICENSES.md](file:///d:/multimodel_passive_aggression/LICENSES.md)** |

### Preprocessing Pipeline
- **Text Preprocessing:** Case normalization, URL and hashtag normalization, emoji extraction/demojization, tokenization using XLM-RoBERTa sentencepiece tokenizer.
- **Image Preprocessing:** Strict validation via Pillow, resizing to $224 \times 224$, standard ImageNet normalization ($\mu = [0.485, 0.456, 0.406]$, $\sigma = [0.229, 0.224, 0.225]$).
- **Group-Aware Splitting:** 70/15/15 clustered split over MMSD + SarcNet (27,970 records) using connected-component union-find over conversation thread IDs and perceptual image hashes:
  - **Train:** 19,618 records
  - **Validation:** 4,143 records
  - **Held-out Test:** 4,209 records
  - **Cryptographic Inter-Split Leakage:** Integer **0** (0.00% overlap).

---

## 4. System Architecture & Data Flow

```
                      +---------------------------------------+
                      | Raw Social Media Ingestion (48,660)   |
                      | (MMSD, SarcNet, MUStARD, Civil Comm.) |
                      +---------------------------------------+
                                          |
                     +--------------------+--------------------+
                     |                                         |
                     v                                         v
        +-------------------------+               +-------------------------+
        | Text/Context Processing |               | Image Decoding & Resize |
        | (Normalization, Hashes) |               | (27,970 decodable JPEGs)|
        +-------------------------+               +-------------------------+
                     |                                         |
                     v                                         v
        +-------------------------+               +-------------------------+
        | XLM-RoBERTa 768-dim     |               | Spatial/Visual 768-dim  |
        | Text Feature Extractor  |               | Grid Feature Extractor  |
        +-------------------------+               +-------------------------+
                     |                                         |
                     +--------------------+--------------------+
                                          |
                                          v
                      +---------------------------------------+
                      | Gated Multimodal Fusion Architecture  |
                      | Gate: w_t, w_v in [0, 1]              |
                      | Fused = w_t * h_text + w_v * h_vis    |
                      +---------------------------------------+
                                          |
                                          v
                      +---------------------------------------+
                      | Classification Head (GELU, LayerNorm) |
                      +---------------------------------------+
                                          |
                     +--------------------+--------------------+
                     |                                         |
                     v                                         v
        +-------------------------+               +-------------------------+
        | Auxiliary Sarcasm Task  |               | Governance Safety Gate  |
        | Test Macro-F1: 0.7516   |               | HTTP 503 (Human Review) |
        +-------------------------+               +-------------------------+
```

---

## 5. Models Actually Implemented & Their Roles

1. **Text Baseline Model (`text_tfidf_baseline`):**
   - Word (1–3) and character (2–5) TF-IDF n-grams + L2-regularized Logistic Regression ($C=1.0$).
   - Serves as the primary high-speed baseline for social media sarcasm text.
2. **Pretrained Transformer Text Classifier (`text_roberta_neural`):**
   - Pretrained `FacebookAI/xlm-roberta-base` (768-dim pooled output) + 2-layer MLP classification head.
3. **Gated Multimodal Fusion Model (`gated_multimodal_fusion`):**
   - Modality dimension alignment (768 text, 768 visual) $\to$ Learned soft gating network $\to$ 512-dim fusion layer $\to$ 256-dim classification head.
4. **Context-Aware Cross-Attention Fusion (`context_aware_fusion`):**
   - Cross-attention mechanism computing multi-head attention between comment query tokens and context/image keys.
5. **Exploratory Silver Passive-Aggression Baseline (`pa_silver_tfidf_v1`):**
   - N-gram Logistic Regression trained exclusively on Phase 10 accepted LLM silver labels.

---

## 6. Training & Evaluation Methodology

- **Split Strategy:** Strict zero-leakage grouped split (Phase 7). All evaluation performed on 4,209 unseen held-out test samples.
- **Hardware Acceleration:** NVIDIA GeForce RTX 4050 Laptop GPU (PyTorch 2.11.0+cu128).
- **Optimization:** AdamW optimizer ($\text{lr} = 1 \times 10^{-3}$, weight decay $= 0.01$), cross-entropy loss, batch size 64, early stopping with validation Macro-F1 monitoring.
- **Metrics Tracked:** Macro-F1, Balanced Accuracy, Precision, Recall, Confusion Matrix, Expected Calibration Error (ECE), and 95% Bootstrap Confidence Intervals.

---

## 7. Existing Metrics and Model Comparisons

### Auxiliary Sarcasm Task (Evaluated on Phase 7 Held-out Test Split, $N=4,209$)

| Architecture | Modalities Used | Test Accuracy | Test Macro-F1 | Macro-Precision | Macro-Recall | Latency (ms) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **TF-IDF + Logistic Regression ($C=1.0$)** | Text Only | **0.7520** | **0.7516** | 0.7518 | 0.7515 | **~3 ms** |
| **Linear SVM ($C=0.5$)** | Text Only | 0.7364 | 0.7360 | 0.7361 | 0.7364 | ~4 ms |
| **XLM-RoBERTa Neural Head** | Text Only | 0.7463 | 0.7457 | 0.7465 | 0.7458 | ~28 ms |
| **Image-Only Neural Classifier** | Image Only | 0.5693 | 0.5478 | 0.5821 | 0.5693 | ~18 ms |
| **Concatenation Multimodal Fusion** | Text + Image | 0.7401 | 0.7397 | 0.7410 | 0.7402 | ~35 ms |
| **Residual Modality-Gated Fusion** | Text + Image | 0.7410 | 0.7404 | 0.7415 | 0.7411 | ~36 ms |
| **Gated Multimodal Fusion (Phase 8A/8)** | Text + Image | **0.7503** | **0.7497** | 0.7505 | 0.7495 | **~35 ms** |
| **Controlled Test-Time Image Ablation** | Text + Zero Image | 0.7503 | 0.7497 | 0.7505 | 0.7495 | ~35 ms |

### Critical Empirical Finding: Learned Gating Collapse
In the Gated Multimodal Fusion network, the learned gating weights converged to **$0.999997$ for text** and **$0.000003$ for image**. 
- Zeroing out the visual stream at test time produced a Delta Macro-F1 of **$+0.0000$**.
- **Scientific Reason:** The offline visual features extracted from social media images were low-level spatial color/intensity statistics rather than high-level semantic representations from a Vision-Language Model (VLM). The gating network successfully protected the model from degrading by ignoring visual noise.

---

## 8. API Endpoints and Live Demonstration Instructions

The backend is served via FastAPI on `http://127.0.0.1:8000`.

### Production Endpoints

| Endpoint | Method | Input | Output / Status | Role |
| :--- | :---: | :--- | :--- | :--- |
| `/` or `/demo` | `GET` | None | HTML5 Single-Page App | Interactive demonstration dashboard |
| `/health` | `GET` | None | `HealthResponse` (200) | Live telemetry, loaded models, GPU status |
| `/metadata` | `GET` | None | `MetadataResponse` (200) | Operational parameters and limitations |
| `/predict/sarcasm/text` | `POST` | `{"text": "..."}` | `SarcasmPredictionResponse` | High-speed text sarcasm inference |
| `/predict/sarcasm/multimodal`| `POST` | Multipart `text` + `image` | `SarcasmPredictionResponse` | Gated multimodal sarcasm with fallback |
| `/predict/passive-aggression`| `POST` | Any | **HTTP 503 Service Unavailable** | Strict human-ground-truth safety gate |
| `/predict/passive-aggression/experimental` | `POST` | `{"text": "..."}` | `PassiveAggressionResponse` | Exploratory silver baseline (`LLM_SILVER`) |
| `/docs` | `GET` | None | Swagger UI | Interactive OpenAPI documentation |

---

## 9. Test Results

The repository features 28 automated test modules verified using pytest:
- **Total Tests Collected:** 203
- **Passed:** **203 (100.0%)**
- **Failed:** **0**
- **Execution Duration:** 41.50 seconds
- **Deprecation Warnings:** 14 (harmless library deprecation warnings from starlette/sklearn)

---

## 10. Known Limitations & Future Work

1. **Absence of Semantic Vision-Language Alignment:** The offline visual stream relied on spatial feature representations. Future work should integrate 4-bit quantized Vision-Language Models (e.g., Qwen2-VL or MiniCPM-V) to capture sarcastic image-text incongruity directly.
2. **Exploratory Sample Size for Silver PA:** The Phase 10 silver dataset contains 15 accepted records, resulting in an exploratory test set of $N=3$. This evaluates software plumbing rather than scientific generalization.
3. **Pending Human Ground Truth:** The gold standard 30-record human annotation benchmark remains unpopulated. Completing double-blind human reviews is the primary prerequisite for unlocking production passive-aggression classification.

---

## 11. Clear Distinction: Sarcasm vs. Passive Aggression

- **Sarcasm:** Communication characterized by antiphrasis (saying the opposite of what is meant) or ironic incongruity between modalities. Sarcasm is frequently humorous, non-hostile, or aimed at situations (e.g., weather, traffic).
- **Passive Aggression:** Indirect communication providing evidence of **interpersonal hostility, resentment, blame, or covert criticism** with plausible deniability.
- **Project Rule:** Sarcasm models **cannot** be claimed or substituted as passive-aggression detectors. The project maintains explicit warning headers on all sarcasm outputs.

---

## 12. Clear Distinction: LLM Silver Labels vs. Human Ground Truth

- **LLM Silver Labels (Phase 10):** Machine-generated annotations produced via independent dual prompting (Llama 3.2 3B and Qwen 2.5 1.5B) and filtered through a conservative 3rd-pass adjudication firewall. They are explicitly marked with provenance `LLM_SILVER` and restricted to exploratory research.
- **Human Ground Truth (Phase 6A):** Strictly gated in `data/annotations/phase6a/annotator_1_responses.csv` and `annotator_2_responses.csv`. The project maintains an unviolated policy of **zero synthetic human data fabrication**.

---

## 13. Setup Instructions for Reproducing the Demo

### Step 1: Start the Backend Server
```powershell
& "D:\NLP_PROJECT\internvl_env\Scripts\python.exe" -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000
```

### Step 2: Access the Interactive Demo
Open your browser and navigate to:
```
http://127.0.0.1:8000/
```

### Step 3: Run the Complete Automated Test Suite
```powershell
& "D:\NLP_PROJECT\internvl_env\Scripts\python.exe" -m pytest tests/ --tb=short
```
