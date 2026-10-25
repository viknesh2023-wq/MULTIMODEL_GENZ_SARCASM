# A Multimodal Vision-Language Framework for Gen-Z Sarcasm Detection Using XLM-R, CLIP and Vision-Language Models

> **GitHub Repository:** [https://github.com/viknesh2023-wq/MULTIMODEL_GENZ_SARCASM](https://github.com/viknesh2023-wq/MULTIMODEL_GENZ_SARCASM)  
> **Documentation Type:** Quick-Start Installation & Execution Guide for Faculty & Reviewers  
> **Tested Environment:** Windows 11 (64-bit), Python 3.12, Node.js 20+, NVIDIA RTX 4050 (CUDA 12.8) or CPU

---

## 📌 Introduction

This project is a research-grade **Multimodal AI Sarcasm Detection System** designed to identify sarcasm in modern social media discourse and internet memes. Sarcasm in Generation Z digital culture frequently arises from **cross-modal incongruity**—a clash between positive or sincere textual captions and contrasting, absurd, or catastrophic visual images. 

The application integrates five representation streams into a deep 2,847-dimensional neural fusion classifier:
1. **Contextual Multilingual Text (768-D):** Fine-tuned XLM-RoBERTa Base encoder (supporting English and Dravidian code-mixing).
2. **Visual Scene Semantics (512-D):** OpenAI CLIP ViT-B/32 image encoder.
3. **High-Resolution Visual Semantics (1,536-D):** OpenGVLab InternVL3-2B vision-language model.
4. **Sentiment Valence (3-D):** 3-class sentiment probabilities (Negative, Neutral, Positive).
5. **Affective Emotion Spectrum (28-D):** 28-class GoEmotions continuous intensity probabilities.

The complete system provides an interactive **React 18 + Vite** web dashboard and an asynchronous **FastAPI** backend with real-time inference and optional local natural-language explanations.

---

## ⚙️ Prerequisites

Ensure the following tools are installed before setting up the application:

| Software | Required Version | Purpose | Mandatory? |
| :--- | :--- | :--- | :---: |
| **Python** | `3.12.x` (64-bit) | Backend API, PyTorch inference, feature extractors | **Yes** |
| **Node.js & npm** | `v20.x` or `v22.x` (LTS) | Frontend web dashboard development server | **Yes** |
| **Git** | `2.40+` | Cloning repository and downloading dependencies | **Yes** |
| **NVIDIA Driver & CUDA** | CUDA `12.x` (Driver `>= 550.x`)| GPU hardware acceleration (NVIDIA RTX / GTX) | *Recommended* |
| **Ollama** | Latest Windows installer | Optional local LLM for natural language explanations | *Optional* |
| **Web Browser** | Chrome, Edge, Firefox | User interface dashboard (`http://localhost:5173`) | **Yes** |

### Hardware & Storage Requirements
- **Storage Needed:** Approximately **18–20 GB** free SSD disk space (virtual environment, dependencies, model checkpoints, and cached foundation weights).
- **RAM:** Minimum **16 GB** system memory (24 GB recommended).
- **GPU vs. CPU Execution:**
  - **With NVIDIA GPU (>= 6 GB VRAM):** Real-time multimodal meme inference takes **~3.5 seconds** per request.
  - **CPU-Only Mode (No GPU):** Fully supported. If no CUDA GPU is detected, PyTorch automatically runs on CPU. Multimodal meme inference takes **~35–50 seconds** due to CPU vision tiling. Text-only sarcasm classification remains instantaneous (**< 25 ms**).

---

## 📥 1. Download the Project

### Option A: Using Git (Recommended)
Open Command Prompt or PowerShell and clone the repository:
```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd NLP_PROJECT
```
*(If you extract the project manually, ensure your terminal is navigated to the project root, e.g., `cd /d D:\NLP_PROJECT`)*.

### Option B: Download as ZIP
1. On GitHub, click the green **Code** button and select **Download ZIP**.
2. Extract the ZIP archive to a folder of your choice (e.g., `D:\NLP_PROJECT`).
3. Open a terminal and navigate into the extracted folder.

---

## 📦 2. Download Required Model Artifacts

> **Important Note on Datasets:**  
> **You DO NOT need to download or retrain on raw datasets to run this application!**  
> All models are pre-trained. The finished application only requires the trained weight checkpoints listed below.

Because individual deep learning weights exceed GitHub's standard 100 MB upload limit, model checkpoints are distributed via external release links or packaged directly with the project delivery archive:

| Model / Artifact Name | Mandatory? | Already in Git? | File Size | Destination Path in Project |
| :--- | :---: | :---: | :---: | :--- |
| **Multimodal Fusion Classifier** (`fusion_model.pt`) | **Yes** | Included / Drive | `14 MB` | `models/multimodal_fusion/fusion_model.pt` |
| **Fusion Feature Scaler** (`fusion_scaler.joblib`) | **Yes** | Included / Drive | `69 KB` | `models/multimodal_fusion/fusion_scaler.joblib` |
| **XLM-R Text Sarcasm Model** (`model.safetensors`) | **Yes** | External / Drive | `1.03 GB` | `models/xlmr_text_sarcasm_full/model.safetensors` |
| **XLM-R Sentiment Model** (`model.safetensors`) | **Yes** | External / Drive | `1.03 GB` | `models/xlmr_sentiment_model/model.safetensors` |
| **XLM-R Emotion Model** (`model.safetensors`) | **Yes** | External / Drive | `1.03 GB` | `models/xlmr_emotion_model/model.safetensors` |
| **OpenGVLab/InternVL3-2B** | **Yes** | Auto-downloaded | `~3.8 GB` | Cached automatically in `~/.cache/huggingface/` |
| **OpenAI CLIP ViT-B/32** | **Yes** | Auto-downloaded | `~350 MB` | Cached automatically in `~/.cache/clip/` |
| **Ollama Llama 3.2 1B** | *Optional* | Pull via Ollama | `1.3 GB` | Managed locally by Ollama runtime |

### Verifying Model Files:
Before starting, ensure the `models/` directory structure looks like this:
```
NLP_PROJECT/
└── models/
    ├── multimodal_fusion/
    │   ├── fusion_model.pt
    │   ├── fusion_scaler.joblib
    │   └── model_config.json
    ├── xlmr_text_sarcasm_full/
    │   ├── model.safetensors
    │   ├── config.json
    │   └── tokenizer.json
    ├── xlmr_sentiment_model/
    │   ├── model.safetensors
    │   ├── config.json
    │   └── tokenizer.json
    └── xlmr_emotion_model/
        ├── model.safetensors
        ├── config.json
        └── tokenizer.json
```

---

## 🐍 3. Set Up the Python Backend

### Step 3.1: Create and Activate Virtual Environment
Open PowerShell in the project root directory (`D:\NLP_PROJECT`):
```powershell
# Navigate to project root
cd /d D:\NLP_PROJECT

# Create virtual environment named internvl_env using Python 3.12
py -3.12 -m venv internvl_env

# Activate the virtual environment (PowerShell)
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\internvl_env\Scripts\Activate.ps1
```
*(In Command Prompt, activate using: `internvl_env\Scripts\activate.bat`)*

### Step 3.2: Install Backend Dependencies
Upgrade pip and install the verified packages:
```powershell
python -m pip install --upgrade pip setuptools wheel

# 1. Install PyTorch with CUDA 12.8 support (for NVIDIA GPU):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# (OR if using CPU-Only hardware without NVIDIA GPU, run this instead):
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 2. Install NLP, Transformer, and API requirements:
pip install transformers==4.49.0 tokenizers sentencepiece safetensors
pip install scikit-learn==1.6.1 joblib==1.4.2 numpy==1.26.4 pandas==2.2.3
pip install pillow==11.1.0 ftfy regex tqdm
pip install fastapi==0.115.8 uvicorn[standard]==0.34.0 pydantic==2.10.6 python-multipart==0.0.20
pip install git+https://github.com/openai/CLIP.git
```

### Step 3.3: Configure Environment Variables
Copy the template `.env.example` file to `.env`:
```powershell
copy .env.example .env
```
Default `.env` configuration:
```env
BACKEND_HOST=0.0.0.0
BACKEND_PORT=8000
MODEL_PATH=D:\NLP_PROJECT\models\xlmr_text_sarcasm_full
VITE_API_BASE_URL=http://localhost:8000/api
```

---

## 💻 4. Set Up the React Frontend

Open a **separate, second terminal window** and navigate to `frontend/`:
```powershell
# Navigate to frontend directory
cd /d D:\NLP_PROJECT\frontend

# Install frontend npm dependencies
npm install

# Verify production build compilation (optional test)
npm run build
```

---

## 🦙 5. Ollama Setup (Optional Explanation Feature)

The system includes an Explainable AI (XAI) reasoner that translates multimodal predictions into natural-language explanations.

- **Is Ollama Mandatory?** **No.** If Ollama is not installed or offline, the application automatically uses an internal **Deterministic Cognitive Reasoner (< 1 ms)** with zero setup required.
- **To Enable Local LLM Explanations:**
  1. Download and install Ollama from [https://ollama.com](https://ollama.com).
  2. Verify in terminal:
     ```bash
     ollama --version
     ```
  3. Pull the verified 1.2-billion parameter model:
     ```bash
     ollama pull llama3.2:1b
     ```
  4. Confirm it is downloaded:
     ```bash
     ollama list
     ```
     *(The backend automatically connects to `http://127.0.0.1:11434` and pins inference strictly to the host CPU with `num_gpu: 0` to preserve GPU memory).*

---

## 🚀 6. How to Run the Complete Application

### Easy Way: Using Provided Batch Scripts

1. Double-click **`start_backend.bat`** (Starts FastAPI server on `http://localhost:8000`).
2. Double-click **`start_frontend.bat`** (Starts Vite React dashboard on `http://localhost:5173`).
3. Open your browser and navigate to:
   ```
   http://localhost:5173
   ```

---

### Manual Terminal Way (Step-by-Step)

#### Terminal 1 — Start Backend Server:
```powershell
cd /d D:\NLP_PROJECT\backend
..\internvl_env\Scripts\python.exe -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
*Expected Console Output:*
```
INFO: Multimodal Sarcasm Intelligence — Backend Starting
INFO: ✓ XLM-R text sarcasm model ready.
INFO: ✓ Multimodal pipeline ready (5 branches + Fusion MLP).
INFO: Uvicorn running on http://0.0.0.0:8000
```

#### Terminal 2 — Start Frontend Dashboard:
```powershell
cd /d D:\NLP_PROJECT\frontend
npm run dev
```
*Expected Console Output:*
```
VITE v5.2.12  ready in 450 ms
➜  Local:   http://localhost:5173/
```

---

## 🧪 7. Quick Verification Guide

### 1. Test Backend Liveness
Open your browser or run in PowerShell:
```powershell
Invoke-RestMethod -Uri "http://localhost:8000/api/health" -Method GET
```
*Expected Response:*
```json
{
  "status": "ok",
  "version": "1.0.0",
  "model_loaded": true,
  "multimodal_ready": true,
  "explanation_ready": true
}
```

### 2. Interactive Swagger API Docs
Visit: `http://localhost:8000/docs` to test endpoints directly.

### 3. Test Text Sarcasm in UI
1. In the browser (`http://localhost:5173`), click **Sarcasm Detection** in the left sidebar.
2. Select the example *"Bro thought he cooked with that fit 💀"* or enter:
   `"Oh absolutely fantastic, another flight delay right before the weekend!"`
3. Click **Analyze Text**. Verify that the confidence gauge and **Sarcastic** verdict appear.

### 4. Test Multimodal Sarcasm (Meme Upload)
1. Click **Multimodal Analysis** in the left sidebar.
2. Upload any test meme image (JPEG, PNG, or WEBP).
3. Enter a caption (e.g., *"Best pizza I have ever made! 🍕✨"*).
4. Click **Run Multimodal Inference**.
5. Observe the 5-branch latency breakdown, sentiment bars, emotion spectrum, and final verdict!

---

## 🛑 How to Stop the Application

To shut down the servers:
1. In the **Frontend terminal**, press `Ctrl + C`, then enter `Y`.
2. In the **Backend terminal**, press `Ctrl + C`.

---

## 🔧 Common Troubleshooting

| Symptom / Error | Likely Cause | Solution |
| :--- | :--- | :--- |
| **`python` is not recognized** | Python is not added to System PATH | Re-run Python installer, check **"Add python.exe to PATH"**, or specify full path to `python.exe`. |
| **`npm` is not recognized** | Node.js is not installed | Install Node.js LTS from [nodejs.org](https://nodejs.org). |
| **Running scripts is disabled** | PowerShell execution policy | Run: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned` |
| **Port 8000 already in use** | A previous Uvicorn process is still running | Kill the port in PowerShell: `Get-Process -Id (Get-NetTCPConnection -LocalPort 8000).OwningProcess \| Stop-Process -Force` |
| **Frontend shows "Backend Offline"** | Backend server is not running | Start backend via Terminal 1 or `start_backend.bat`. Confirm `http://localhost:8000/api/health` loads. |
| **CUDA out of memory (OOM)** | Background apps consuming VRAM | Close games or video editors. System requires ~5.4 GB free VRAM at peak. |
| **`fusion_scaler.joblib` not found** | Scaler file missing from models folder | Confirm `fusion_scaler.joblib` is placed inside `models/multimodal_fusion/`. |
| **Ollama shows offline / fallback** | Ollama is not installed or not running | Normal behavior! Built-in cognitive reasoner handles explanations in < 1 ms without Ollama. |

---

## 📁 Repository Directory Structure

```
NLP_PROJECT/
├── backend/
│   ├── app/
│   │   ├── main.py                    # FastAPI application entry point
│   │   ├── routes/
│   │   │   ├── predict.py             # Prediction endpoints (/text, /multimodal, /explain)
│   │   │   ├── model.py               # Model health & telemetry endpoints
│   │   │   └── data.py                # Dataset and feature status endpoints
│   │   ├── services/
│   │   │   ├── multimodal_service.py  # 5-branch feature extraction orchestrator
│   │   │   ├── model_service.py       # XLM-R text inference service
│   │   │   └── explanation_service.py # Decoupled local XAI reasoner
│   │   └── schemas/schemas.py         # Pydantic request & response schemas
│   └── tests/                         # Automated regression test suites
│
├── frontend/
│   ├── src/
│   │   ├── pages/                     # 10 interactive dashboard pages
│   │   ├── components/                # Header, Sidebar, Footer, Gauges
│   │   ├── services/api.ts            # Frontend Axios API client
│   │   └── App.tsx                    # Root React navigation router
│   ├── package.json                   # React 18, Vite, Tailwind dependencies
│   └── vite.config.ts                 # Dev server & backend proxy configuration
│
├── models/
│   ├── multimodal_fusion/             # MultimodalFusionMLP (fusion_model.pt + scaler)
│   ├── xlmr_text_sarcasm_full/        # Fine-tuned XLM-R text sarcasm classifier
│   ├── xlmr_sentiment_model/          # Fine-tuned XLM-R 3-class sentiment model
│   └── xlmr_emotion_model/            # Fine-tuned XLM-R 28-class GoEmotions model
│
├── FINAL_DATASETS/                    # Master CSV datasets (Text, Multimodal, Sentiment, Emotion)
├── start_backend.bat                  # One-click Windows startup script for backend
├── start_frontend.bat                 # One-click Windows startup script for frontend
├── .env.example                       # Environment variables template
├── README.txt                         # Comprehensive 26-section university handover manual
└── README.md                          # Clean GitHub quick-start installation guide (This file)
```

---

## 📄 License & Academic Attribution

This project is developed for university research and academic evaluation. The models build upon open-source foundation checkpoints from **FacebookAI (XLM-RoBERTa)**, **OpenAI (CLIP)**, **OpenGVLab (InternVL)**, and **Meta (Llama 3.2)**, licensed for academic research.
