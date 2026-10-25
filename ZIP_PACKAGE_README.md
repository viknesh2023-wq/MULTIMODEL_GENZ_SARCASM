# Project Documentation & Setup Guide for Submission ZIP Archive

> **Project Title:** A Multimodal Vision-Language Framework for Gen-Z Sarcasm Detection Using XLM-R, CLIP and Vision-Language Models  
> **Official GitHub Repository:** [https://github.com/viknesh2023-wq/MULTIMODEL_GENZ_SARCASM](https://github.com/viknesh2023-wq/MULTIMODEL_GENZ_SARCASM)  
> **Submission Archive:** `MULTIMODEL_GENZ_SARCASM_SUBMISSION.zip` (0.86 GB / 922.60 MB)  
> **Tested Operating System:** Windows 10 / Windows 11 (64-bit)  
> **Target Runtime Ports:** Backend (`http://localhost:8000`), Frontend (`http://localhost:5173`)  

---

## 1. Executive Summary

This package contains the complete academic delivery for the **Gen-Z Multimodal Sarcasm Detection System**. The project addresses modern Internet sarcasm—where textual captions (e.g., *"Best day of my life!"*) clash with catastrophic, absurd, or ironic meme imagery. 

The core inference engine fuses **5 representation streams** into a deep 2,847-dimensional neural fusion classifier:
1. **Contextual Text Representation (768-D):** Fine-tuned multilingual XLM-RoBERTa Base encoder.
2. **Visual Scene Semantics (512-D):** OpenAI CLIP ViT-B/32 image encoder.
3. **High-Resolution Visual Semantics (1,536-D):** OpenGVLab InternVL3-2B vision-language model.
4. **Sentiment Valence Probabilities (3-D):** 3-class sentiment polarity (Negative, Neutral, Positive).
5. **Affective Emotion Spectrum (28-D):** 28-class GoEmotions continuous emotion intensity distribution.

The system features an interactive **React 18 + Vite** dashboard and a high-performance **FastAPI** backend with real-time prediction and Explainable AI (XAI) rationale generation.

---

## 2. What IS Included in this ZIP Archive

The archive contains **174,183 files** structured inside a single root directory (`MULTIMODEL_GENZ_SARCASM/`) for clean extraction:

### A. Complete Source Code
* **Frontend Web Dashboard (`frontend/`):**
  * Full React 18 + Vite + Tailwind CSS application.
  * All 10 interactive pages:
    1. `Overview.tsx`: System introduction and telemetry overview.
    2. `SarcasmDetection.tsx`: Text sarcasm detector with live Gen-Z presets.
    3. `MultimodalAnalysis.tsx`: Meme image upload + caption inference with 5-branch radar & gauges.
    4. `TextAnalysis.tsx`: Deep linguistic inspection and sub-word token analyzer.
    5. `ModelArchitecture.tsx`: Visual diagram of the 5-branch neural pipeline.
    6. `ModelPerformance.tsx`: Ablation benchmark comparison tables and confusion matrices.
    7. `FeatureExtraction.tsx`: Breakdown of the 2,847-D concatenated feature vector.
    8. `Dataset.tsx`: Interactive dataset viewer with statistical distributions.
    9. `ResearchNovelty.tsx`: Academic contributions and architectural advantages.
    10. `AboutProject.tsx`: Team 14 details, methodology, and university evaluation scope.
  * Reusable UI components: Confidence gauges, radar charts, header, sidebar, footer.
  * Axios API client configured for seamless local proxying to the backend.

* **Backend REST API (`backend/`):**
  * Asynchronous FastAPI server (`backend/app/main.py`).
  * Endpoints:
    * `POST /api/predict/text`: Text sarcasm prediction.
    * `POST /api/predict/multimodal`: Multimodal meme inference with 5 branches.
    * `POST /api/explain`: Local deterministic or LLM cognitive explanation.
    * `GET /api/health`: Health status & loaded model telemetry.
    * `GET /api/model/status`: Detailed model registry status.
    * `GET /api/data/benchmarks`: Pre-computed ablation performance metrics.
  * Services: `multimodal_service.py`, `model_service.py`, `explanation_service.py`.
  * Schemas: Pydantic v2 validation models (`schemas/schemas.py`).
  * Test suites: Automated end-to-end regression tests (`backend/tests/`).

### B. Pre-Trained Multimodal Neural Fusion Engine
* **`models/multimodal_fusion/fusion_model.pt` (14 MB):** The trained PyTorch neural fusion multilayer perceptron (MLP) with batch normalization and dropout.
* **`models/multimodal_fusion/fusion_scaler.joblib` (69 KB):** The fitted feature standardizer calibrated across the 2,847 feature dimensions.
* **`models/multimodal_fusion/model_config.json`:** Architecture specifications (input dimension: 2,847, hidden layers, activation functions).
* **`models/multimodal_fusion/ablation_benchmark_results.json`:** Comprehensive validation and test metrics.

### C. Tokenizers, Vocabularies & Model Configurations
* **`models/xlmr_text_sarcasm_full/`:** Tokenizer vocabulary (`sentencepiece.bpe.model`), `tokenizer.json`, `config.json`, `special_tokens_map.json`, and final test metrics (`final_test_results.json`).
* **`models/xlmr_sentiment_model/`:** Sentiment tokenizer, config, and test metrics.
* **`models/xlmr_emotion_model/`:** GoEmotions 28-class tokenizer, config, and test metrics.

### D. Portable 1-Click Startup Scripts
* **`start_backend.bat`:** Windows batch script that automatically detects the project root on **any drive** (`C:`, `D:`, etc.) and launches Uvicorn on port 8000.
* **`start_frontend.bat`:** Windows batch script that navigates into `frontend/` and starts the Vite development server on port 5173.

### E. Datasets & Test Media
* **`assets/sample_memes/`:** 5 representative test memes for instant demonstration in the UI without needing any downloads.
* **`MEMOTION/`:** Complete **8,500 meme images** with `TRAIN_TEST.csv` and `updated_val.csv`.
* **`FINAL_DATASETS/`:**
  * Multimodal sarcasm splits: `multimodal_train.csv`, `multimodal_valid.csv`, `multimodal_test.csv`.
  * Text sarcasm splits: `train.csv`, `validation.csv`, `test.csv`.
  * Compressed combined dataset: `text_sarcasm_combined.csv.gz` (37 MB).
  * Fusion feature matrices: `X_train.npy.gz` (52 MB), `X_val.npy` (44 MB), `X_test.npy` (43 MB), label arrays (`y_*.npy`), and `fusion_metadata.csv`.
* **`sarc/`:** Balanced SARC dataset split into parts 1, 2, and 3 (`train-balanced-sarc_part*.csv`) + `test-balanced.csv`.
* **Other NLP Datasets:** `DravidianCodeMix/`, `GenZ-Sarc/`, `GoEmotions/`, `TweetEval/`, `sentimix/`, `isarcasm/`, `Multiligual-Sysnthesised/`, `neutral_darav/`.

### F. Complete Documentation & University Deliverables
* **`GenZ_SarC_Complete_Architecture.png`:** High-resolution neural pipeline diagram.
* **`GenZ_SarC_Complete_Team_Documentation.docx`:** Team 14 comprehensive academic report.
* **`TEAM14_PROJECT_DEMO_SCRIPT.txt`:** Step-by-step viva demo script with sample prompts.
* **`PROJECT_DEMONSTRATION_SCRIPT.txt`:** Full oral presentation script.
* **`README.md`:** Main project installation guide.
* **`README_QUICKSTART.txt`:** Plain-text quick-start guide referencing the GitHub repo.
* **`PROJECT_AUDIT/`:** 16 detailed audit reports verifying model convergence and metrics.

---

## 3. What is NOT Included in this ZIP Archive (And Why)

To keep the delivery package clean, portable, and under 1 GB, the following non-portable or redundant items were excluded:

| Excluded Item | Size on Author's PC | Reason for Exclusion | How It Is Handled |
| :--- | :---: | :--- | :--- |
| **Python Virtual Environment (`internvl_env/`)** | ~4.82 GB | Virtual environments contain machine-specific C-binaries, DLLs, and absolute path symlinks that fail when copied to another PC. | Faculty creates a clean environment in 1 command using the provided instructions. |
| **Frontend `node_modules/`** | ~140 MB | JavaScript build dependencies should always be freshly installed on the host machine. | Run `npm install` inside `frontend/` (takes ~20 seconds). |
| **Intermediate Training Checkpoints (`optimizer.pt`, `checkpoint-*/`)** | ~29 GB | PyTorch Adam optimizer states and intermediate epochs are only used during model training, not during inference. | The final trained fusion model (`fusion_model.pt`) is already included in `models/multimodal_fusion/`. |
| **Large Checkpoint Weights (`model.safetensors`, 3 x 1.03 GB)** | ~3.1 GB | Raw 1.03 GB transformer weights exceed archive limits. | CLIP ViT-B/32 and InternVL3-2B auto-cache on first run. Pre-trained fusion weights (`fusion_model.pt`) and evaluation benchmarks are already packaged. |
| **Bulk MMSD 2.0 Images (24,635 images)** | ~2.57 GB | Excluded to keep the archive compact. | 8,500 `MEMOTION` images and sample test memes in `assets/sample_memes/` are fully included for live multimodal testing. |
| **Redundant Split Archives (`dataset_image.z0*`)** | ~2.5 GB | Redundant multi-part zip archives. | Excluded to prevent duplicate storage. |
| **Passive Aggression Module (`passive_aggression_multimodal/`)** | ~5.66 GB | Independent research project kept in a separate workspace. | Excluded per team specification. |
| **Git Commit History (`.git/`)** | ~2.15 GB | Internal version control objects. | Accessible online via the official GitHub repository link. |

---

## 4. Step-by-Step Setup Guide for Faculty

### System Requirements:
- **Operating System:** Windows 10 / 11 (64-bit)
- **Python:** Version `3.12.x` (64-bit) [Must check "Add python.exe to PATH" during installation]
- **Node.js:** Version `20.x` or `22.x` LTS (from [nodejs.org](https://nodejs.org))
- **RAM:** 16 GB minimum (24 GB recommended)
- **Hardware Acceleration:** NVIDIA GPU with CUDA 12.x (Recommended; CPU-only execution is also fully supported)

---

### Step 1: Extract the Archive
1. Right-click **`MULTIMODEL_GENZ_SARCASM_SUBMISSION.zip`**.
2. Select **Extract All...** and choose your destination directory (e.g., `C:\MULTIMODEL_GENZ_SARCASM` or `D:\MULTIMODEL_GENZ_SARCASM`).
3. Open a terminal (PowerShell or Command Prompt) and navigate into the extracted folder:
   ```powershell
   cd MULTIMODEL_GENZ_SARCASM
   ```

---

### Step 2: Set Up the Python Backend

In your terminal, create and activate a Python 3.12 virtual environment:

```powershell
# 1. Create virtual environment named internvl_env
py -3.12 -m venv internvl_env

# 2. Activate virtual environment (PowerShell)
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\internvl_env\Scripts\Activate.ps1
```
*(If using Command Prompt, activate with: `internvl_env\Scripts\activate.bat`)*

Upgrade pip and install the required backend dependencies:

```powershell
python -m pip install --upgrade pip setuptools wheel

# If you have an NVIDIA GPU (CUDA 12.8):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# OR if running on CPU without an NVIDIA GPU:
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install NLP, multimodal, and FastAPI packages:
pip install transformers==4.49.0 tokenizers sentencepiece safetensors
pip install scikit-learn==1.6.1 joblib==1.4.2 numpy==1.26.4 pandas==2.2.3
pip install pillow==11.1.0 ftfy regex tqdm
pip install fastapi==0.115.8 uvicorn[standard]==0.34.0 pydantic==2.10.6 python-multipart==0.0.20
pip install git+https://github.com/openai/CLIP.git
```

Create your `.env` configuration file:
```powershell
copy .env.example .env
```

---

### Step 3: Set Up the React Frontend

Open a **second, separate terminal window** and navigate into `frontend/`:

```powershell
cd MULTIMODEL_GENZ_SARCASM\frontend

# Install JavaScript dependencies:
npm install
```

---

### Step 4: Run the Application

You can start the entire system using the 1-click batch scripts or manual terminal commands:

#### Easy 1-Click Way:
1. Double-click **`start_backend.bat`** (Starts FastAPI server on `http://localhost:8000`).
2. Double-click **`start_frontend.bat`** (Starts Vite development server on `http://localhost:5173`).
3. Open your browser and navigate to:
   ```
   http://localhost:5173
   ```

#### Manual Terminal Way:
* **Terminal 1 (Backend):**
  ```powershell
  cd MULTIMODEL_GENZ_SARCASM\backend
  ..\internvl_env\Scripts\python.exe -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
  ```
* **Terminal 2 (Frontend):**
  ```powershell
  cd MULTIMODEL_GENZ_SARCASM\frontend
  npm run dev
  ```

---

## 5. How to Test Predictions During Evaluation

### Test 1: Text Sarcasm Detection
1. In the web dashboard (`http://localhost:5173`), click **Sarcasm Detection** in the left sidebar.
2. Select any Gen-Z preset prompt (e.g., *"Bro thought he cooked with that fit 💀"*) or enter:
   > *"Oh absolutely fantastic, another flight delay right before the weekend!"*
3. Click **Analyze Text**.
4. Observe the confidence gauge, probability bar, and **Sarcastic** classification verdict.

### Test 2: Multimodal Meme Sarcasm Detection
1. Click **Multimodal Analysis** in the left sidebar.
2. Under **Upload Image**, upload any sample meme from `assets/sample_memes/` (e.g., `682716753374351360.jpg`) or upload any image from your computer.
3. Enter an ironic caption (e.g., *"Best pizza I have ever made! 🍕✨"*).
4. Click **Run Multimodal Inference**.
5. Observe the complete 5-branch analysis:
   - Overall Sarcasm Verdict & Confidence Gauge
   - Sentiment Polarity (Negative / Neutral / Positive)
   - GoEmotions Spectrum (28-class radar chart)
   - 5-Branch Latency Breakdown
   - Natural Language Cognitive Rationale (Explainable AI)

### Test 3: Backend REST API Interactive Documentation
Visit `http://localhost:8000/docs` in your browser to inspect and execute OpenAPI Swagger endpoints directly.

---

## 6. Troubleshooting & FAQs

* **Error: `Running scripts is disabled on this system` (PowerShell):**  
  *Solution:* Run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned` in your PowerShell window before activating the environment.
* **Error: `Port 8000 is already in use`:**  
  *Solution:* Another Uvicorn process is running. Terminate it via PowerShell:
  `Get-Process -Id (Get-NetTCPConnection -LocalPort 8000).OwningProcess | Stop-Process -Force`
* **GPU vs. CPU Execution:**  
  * With an NVIDIA GPU (CUDA), multimodal meme inference takes **~3.5 seconds**.
  * On CPU-only hardware, PyTorch executes automatically on CPU. Multimodal meme inference takes **~35–50 seconds** due to CPU vision tiling. Text sarcasm remains instantaneous (**< 25 ms**).
* **Is Ollama Mandatory?**  
  * **No.** If Ollama is not installed, the built-in Deterministic Cognitive Reasoner generates explanations in **< 1 ms** automatically.

---

## 7. Academic Attribution & License

Developed for university research and final project evaluation (Team 14).  
Open-source foundation models utilized: **FacebookAI (XLM-RoBERTa)**, **OpenAI (CLIP)**, **OpenGVLab (InternVL)**, and **Meta (Llama 3.2)** under academic research licenses.
