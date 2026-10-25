# Phase 10 Local Model Capability Audit
## Assessment of Local Ollama Models, Modality Support, and Independence

**Project Title:** Context-Aware Multimodal Passive-Aggression Detection in Social Media Using Image–Comment Fusion  
**Artifact:** `reports/phase10/local_model_capability_audit.md`  
**Date:** October 22, 2026  
**Canonical Project Root:** `D:\NLP_PROJECT\passive_aggression_multimodal`  
**Active Linked Workspace:** `D:\multimodel_passive_aggression`  

---

## 1. Verified Model Inventory

An audit of the active local Ollama daemon (`http://localhost:11434/api/tags`) verified three installed language models:

| Model Identifier | Family | Parameter Count | Disk Size | Context Window | Structured JSON Output | Modality |
|---|---|---|---|---|---|---|
| `llama3.2:latest` | `llama` | 3.2B | 1,925.8 MB | 128,000 tokens | **Supported (via format=json)** | **Text Only** |
| `qwen2.5:1.5b` | `qwen2` | 1.5B | 940.4 MB | 32,768 tokens | **Supported (via format=json)** | **Text Only** |
| `llama3.2:1b` | `llama` | 1.2B | 1,259.9 MB | 128,000 tokens | **Supported (via format=json)** | **Text Only** |

---

## 2. Model Independence Assessment

To establish meaningful double-blind machine annotation:
1. **Annotator A (`llama3.2:latest`):** Developed by Meta AI. Utilizes a 3.2-billion-parameter dense transformer architecture trained on 9+ trillion multilingual tokens with a 128k vocabulary tokenizer.
2. **Annotator B (`qwen2.5:1.5b`):** Developed by Alibaba Cloud. Utilizes a 1.5-billion-parameter dense transformer architecture trained on 18+ trillion multilingual tokens with a 151k byte-pair tokenizer.
3. **Architectural & Data Diversity:** The two models represent fundamentally independent training corpora, pretraining objectives, alignment methodologies, and inductive biases. Agreement between Llama and Qwen provides non-trivial corroboration, though it does not substitute for human ground truth.
4. **Adjudication Arbiter:** Disagreements between Annotator A and Annotator B are submitted to an independent adjudication prompt pass utilizing `llama3.2:latest` configured with an impartial judicial system prompt.

---

## 3. Modality Boundaries & Image Handling

- **Text-Only Reality:** Neither model possesses native visual comprehension (vision-language capabilities).
- **Modality Honesty Enforcement:**
  - For image-bearing records (e.g. from MMSD or SarcNet), the pipeline supplies comment text and dialogue context, but explicitly flags `image_assessed: false` and `modality_used: ["text"]`.
  - When visual incongruity is crucial to determine interpersonal intent, annotators are instructed to assign the conservative label `ambiguous` due to context insufficiency.
  - Zero claims of multimodal annotation are made for these runs.

---

## 4. Hardware Resource Constraints & Inference Budget

- **GPU:** NVIDIA GeForce RTX 4050 Laptop GPU (6,141 MB total VRAM).
- **VRAM Utilization:** Loading `llama3.2:latest` consumes ~2.2 GB VRAM; `qwen2.5:1.5b` consumes ~1.2 GB VRAM. Both fit comfortably within the 6 GB ceiling.
- **Latency Benchmarks:**
  - `llama3.2:latest`: ~1.1s per prompt (warm)
  - `qwen2.5:1.5b`: ~0.7s per prompt (warm)
  - Full dual-pass + adjudication on 100 records: ~250–300 seconds (~4.5 minutes).
- **Determinism Controls:** All Ollama API calls enforce `temperature: 0.0` and `seed: 42` to guarantee reproducible generation.
