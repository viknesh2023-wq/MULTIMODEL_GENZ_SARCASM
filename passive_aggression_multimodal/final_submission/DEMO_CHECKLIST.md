# Live Demonstration Checklist — College Project Review

**Project:** Context-Aware Multimodal Passive-Aggression Detection in Social Media Using Image–Comment Fusion  
**Target URL:** `http://127.0.0.1:8000/`  
**Swagger Docs:** `http://127.0.0.1:8000/docs`  
**Hardware:** NVIDIA RTX 4050 Laptop GPU (6 GB VRAM)  

---

## 🏁 Pre-Demo Startup (1 Minute)

1. **Verify Backend is Running:**
   Open PowerShell in `D:\multimodel_passive_aggression` and run:
   ```powershell
   & "D:\NLP_PROJECT\internvl_env\Scripts\python.exe" -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000
   ```
   *Expected output:* `Application startup complete. Uvicorn running on http://127.0.0.1:8000`

2. **Open the Demo in Browser:**
   Navigate to [http://127.0.0.1:8000/](http://127.0.0.1:8000/).
   *Verify:* Header displays `● Live API: Healthy`, `GPU: NVIDIA RTX 4050`, and `PA Gate: BLOCKED (0 human reviews)`.

---

## 🎬 Live Demonstration Flow (5–7 Minutes)

### Step 1: Health & Telemetry Verification (Tab 1 / Header)
- Point out the real-time telemetry badge: model initialized on CUDA, zero VRAM leaks, loaded models: `text_tfidf_baseline`, `gated_multimodal_fusion`, `pa_silver_tfidf_v1`.
- Explain the project scope: auxiliary task is multimodal sarcasm detection on 27,970 social media images; primary research goal is context-aware passive aggression.

### Step 2: Live Text Sarcasm Inference (Tab 1 — Left Card)
- Click the preset: **`Example 1 (Sarcasm)`** (*"Oh wonderful, another flat tire on Monday morning!"*).
- Click **"Run Text Sarcasm Model"**.
- Point out:
  - Prediction: **`sarcastic`** with high confidence (~91.5%).
  - Ultra-low latency: **~4–6 ms**.
  - Scientific warning: *"Auxiliary Sarcasm Detection Only: Sarcasm predictions do not constitute passive-aggression ground truth."*
- Click **`Example 3 (Non-Sarcastic)`** (*"Thank you so much for the detailed and constructive code review."*).
- Click **"Run Text Sarcasm Model"**.
- Point out: Prediction correctly flips to **`non sarcastic`** with 93%+ confidence.

### Step 3: Multimodal Sarcasm & Modality Fallback (Tab 1 — Right Card)
- Enter comment text: *"What a marvelous day to be stuck in a ditch."*
- Click the upload area and select a sample image from `data/images/sarcnet/1.jpg`.
- Click **"Run Multimodal Model"**.
- Point out:
  - Modalities Used: **`text + image`**.
  - Model: `gated_multimodal_fusion` executing on RTX 4050 GPU.
- **Demonstrate Modality Fallback:**
  - Click **"Clear Image"** to remove the image.
  - Click **"Run Multimodal Model"** with text alone.
  - Point out: The system **does not crash**; it applies graceful fallback (`fallback_applied: true`, `modality_used: ['text']`) and warns the reviewer that text-only fallback was executed.

### Step 4: The Research Governance & Passive-Aggression Safety Gate (Tab 2 — Left Card)
- Switch to **Tab 2: "2. Passive-Aggression Governance Gate"**.
- Explain the key research distinction: Sarcasm is antiphrasis/incongruity; passive aggression requires **interpersonal hostility with plausible deniability**.
- Click **"Test Production Safety Gate (Expect HTTP 503)"**.
- Point out:
  - The API returns **HTTP 503 Service Unavailable** (`BLOCKED_PENDING_HUMAN_ANNOTATION`).
  - Explain: *"Our system refuses to return fake or misleading passive-aggression scores using sarcasm proxies. It enforces an ethical research gate until human ground-truth reviews are complete."*

### Step 5: Exploratory Silver-Label Passive Aggression (Tab 2 — Right Card)
- Click **`Example X (Implied Blame)`** (*"Thanks for finally replying to my email after two whole weeks."*).
- Click **"Run Exploratory Silver Model"**.
- Point out:
  - Predicted label: **`passive aggressive`**.
  - Crucial badge: **`PROVENANCE: LLM_SILVER`**.
  - Explain: *"This model was trained exclusively on our Phase 10 consensus silver labels (Llama 3.2 + Qwen 2.5 + adjudication firewall). It is explicitly designated as exploratory."*

### Step 6: Empirical Evidence & Research Transparency (Tab 3)
- Switch to **Tab 3: "3. Model Evidence & Comparisons"**.
- Present the verified test comparison table:
  - Text Baseline: **0.7516 Macro-F1** (4,209 test samples).
  - Multimodal Gated Fusion: **0.7497 Macro-F1** (4,209 test samples).
  - Ablation Study: Zeroing out image features produced **Delta = 0.0000**, revealing that the gating network learned to collapse visual weights ($0.999997$ text vs $0.000003$ visual) to filter out spatial visual noise.
- Highlight software engineering rigor: **203 / 203 automated unit tests passing**.

---

## 💡 Key Answers for the Viva Panel

| Potential Examiner Question | Strong, Honest Answer |
| :--- | :--- |
| **"Why didn't image fusion beat the text-only baseline?"** | *"In offline feature extraction, spatial color/intensity grid features were extracted rather than high-level semantic Vision-Language features. Our gating network learned that text carried high signal (0.7516 F1) while spatial visual features were noisy. The gate collapsed visual weights to 0.000003 to protect model performance. This is an honest empirical finding rather than a failure."* |
| **"Why does your passive-aggression endpoint return 503?"** | *"Because no valid, human-annotated ground truth exists on real social media yet. Substituting sarcasm labels for passive aggression is scientifically invalid. We built the complete triadic annotation protocol and local LLM silver pipeline, but we strictly lock production PA inference until human reviews are complete."* |
| **"How did you prevent data leakage?"** | *"We implemented cryptographic SHA-256 deduplication and clustered grouped splits using connected-component union-find on conversation thread IDs and image perceptual hashes. Inter-split leakage across Train, Val, and Test is confirmed at 0.00%."* |
| **"What would you do next?"** | *"1. Onboard a 4-bit quantized Vision-Language Model (like Qwen2-VL or MiniCPM-V) to extract high-level semantic image embeddings. 2. Collect double-blind human annotations for the 30-record Phase 6A pilot to unlock the production benchmark."* |
