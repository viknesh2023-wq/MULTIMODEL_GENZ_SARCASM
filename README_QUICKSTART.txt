================================================================================
A MULTIMODAL VISION-LANGUAGE FRAMEWORK FOR GEN-Z SARCASM DETECTION
USING XLM-R, CLIP AND VISION-LANGUAGE MODELS
================================================================================
Quick-Start Installation & Execution Guide for Faculty & Reviewers
GitHub Repository: [ADD YOUR GITHUB REPOSITORY URL HERE]
Document File    : README_QUICKSTART.txt (Plain Text Document)
Target Platform  : Windows 11 / 10 (64-bit) | Local Laptop Execution
Backend Port     : http://localhost:8000
Frontend Port    : http://localhost:5173
================================================================================


1. INTRODUCTION
--------------------------------------------------------------------------------
This project is an AI-powered Multimodal Sarcasm Detection System designed to
detect sarcasm in social media captions and image memes. In modern digital
culture, sarcasm often arises from cross-modal incongruity--where positive or
sincere text clashes with a contrasting, absurd, or catastrophic image.

The application combines five representation streams into a deep 2,847-D neural
fusion classifier:
  1. Contextual Multilingual Text (768-D) : Fine-tuned XLM-RoBERTa Base
  2. Visual Scene Semantics (512-D)       : OpenAI CLIP ViT-B/32
  3. High-Resolution Visuals (1,536-D)   : OpenGVLab InternVL3-2B
  4. Sentiment Valence (3-D)              : Negative, Neutral, Positive probs
  5. Affective Emotion Spectrum (28-D)    : 28 GoEmotions continuous probs

The system provides an interactive React 18 + Vite web dashboard and an
asynchronous FastAPI backend with real-time inference and optional local
explanations.


2. PREREQUISITES
--------------------------------------------------------------------------------
The following software is required:

  - Python 3.12 (64-bit)       : Backend API & PyTorch inference [Mandatory]
  - Node.js (v20+ or v22+ LTS) : Frontend web server & npm       [Mandatory]
  - Git for Windows            : Cloning repository              [Mandatory]
  - Modern Web Browser         : Chrome, Edge, or Firefox        [Mandatory]
  - NVIDIA GPU & CUDA 12.x     : Hardware acceleration           [Recommended]
  - Ollama Runtime             : Local LLM explanations          [Optional]

Hardware & Storage Requirements:
  - Free Disk Space : ~18 to 20 GB (virtual environment, dependencies, weights)
  - System RAM      : 16 GB minimum (24 GB recommended)
  - NVIDIA GPU      : Multimodal meme inference takes ~3.5 seconds on CUDA.
  - CPU-Only Mode   : Supported! If no NVIDIA GPU is detected, PyTorch runs on
                      CPU. Multimodal inference takes ~35-50s due to CPU vision
                      tiling. Text-only sarcasm classification remains <25 ms.


3. DOWNLOAD THE PROJECT
--------------------------------------------------------------------------------
Option A: Using Git (Recommended)
Open Command Prompt or PowerShell:
  git clone <YOUR_GITHUB_REPOSITORY_URL>
  cd NLP_PROJECT

Option B: Download as ZIP
1. On GitHub, click the green "Code" button and select "Download ZIP".
2. Extract the ZIP archive to a folder (e.g., D:\NLP_PROJECT).
3. Open a terminal and navigate into the folder:
  cd /d D:\NLP_PROJECT


4. REQUIRED MODEL ARTIFACTS
--------------------------------------------------------------------------------
NOTE: You DO NOT need to download or retrain on raw datasets!
All models are pre-trained. The application only requires the trained weight
checkpoints listed below:

+----------------------------+-----------+-----------+-------------------------+
| Model / Artifact Name      | Required? | Size      | Destination Path        |
+----------------------------+-----------+-----------+-------------------------+
| Multimodal Fusion Model    | Yes       | 14 MB     | models/multimodal_      |
| (fusion_model.pt)          |           |           | fusion/fusion_model.pt  |
|                            |           |           |                         |
| Fusion Feature Scaler      | Yes       | 69 KB     | models/multimodal_      |
| (fusion_scaler.joblib)     |           |           | fusion/fusion_scaler... |
|                            |           |           |                         |
| XLM-R Text Sarcasm Model   | Yes       | 1.03 GB   | models/xlmr_text_       |
| (model.safetensors)        |           |           | sarcasm_full/...        |
|                            |           |           |                         |
| XLM-R Sentiment Model      | Yes       | 1.03 GB   | models/xlmr_sentiment_  |
| (model.safetensors)        |           |           | model/...               |
|                            |           |           |                         |
| XLM-R Emotion Model        | Yes       | 1.03 GB   | models/xlmr_emotion_    |
| (model.safetensors)        |           |           | model/...               |
|                            |           |           |                         |
| OpenGVLab/InternVL3-2B     | Yes       | ~3.8 GB   | Auto-cached on 1st run  |
| OpenAI CLIP ViT-B/32       | Yes       | ~350 MB   | Auto-cached on 1st run  |
| Ollama Llama 3.2 1B        | Optional  | 1.3 GB    | Managed by Ollama       |
+----------------------------+-----------+-----------+-------------------------+

Expected Directory Structure:
  NLP_PROJECT/
  |-- models/
      |-- multimodal_fusion/
      |   |-- fusion_model.pt
      |   |-- fusion_scaler.joblib
      |   `-- model_config.json
      |-- xlmr_text_sarcasm_full/
      |   |-- model.safetensors
      |   |-- config.json
      |   `-- tokenizer.json
      |-- xlmr_sentiment_model/
      |   |-- model.safetensors
      |   |-- config.json
      |   `-- tokenizer.json
      `-- xlmr_emotion_model/
          |-- model.safetensors
          |-- config.json
          `-- tokenizer.json


5. SET UP THE PYTHON BACKEND
--------------------------------------------------------------------------------
Step 5.1: Create and Activate Virtual Environment
Open PowerShell in the project root:
  cd /d D:\NLP_PROJECT
  py -3.12 -m venv internvl_env
  Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
  .\internvl_env\Scripts\Activate.ps1

(In Command Prompt, activate with: internvl_env\Scripts\activate.bat)

Step 5.2: Install Backend Dependencies
Upgrade pip:
  python -m pip install --upgrade pip setuptools wheel

For NVIDIA GPU with CUDA 12.8 (Recommended):
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

(OR for CPU-only computers without an NVIDIA GPU, run):
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

Install Transformer, NLP, and API libraries:
  pip install transformers==4.49.0 tokenizers sentencepiece safetensors
  pip install scikit-learn==1.6.1 joblib==1.4.2 numpy==1.26.4 pandas==2.2.3
  pip install pillow==11.1.0 ftfy regex tqdm
  pip install fastapi==0.115.8 uvicorn[standard]==0.34.0 pydantic==2.10.6 python-multipart==0.0.20
  pip install git+https://github.com/openai/CLIP.git

Step 5.3: Configure Environment Variables
Copy the template .env.example to .env:
  copy .env.example .env


6. SET UP THE REACT FRONTEND
--------------------------------------------------------------------------------
Open a separate, second terminal window:
  cd /d D:\NLP_PROJECT\frontend
  npm install


7. OLLAMA SETUP (OPTIONAL EXPLANATION FEATURE)
--------------------------------------------------------------------------------
Is Ollama mandatory? NO. If Ollama is not installed or offline, the app
automatically uses its built-in Deterministic Cognitive Reasoner (<1 ms)
with zero configuration needed.

If local LLM explanations are desired:
  1. Download and install Ollama from: https://ollama.com
  2. Verify in terminal:
       ollama --version
  3. Pull the compact 1.2B model:
       ollama pull llama3.2:1b
  4. Verify:
       ollama list
  (The backend automatically pins Ollama to host CPU with "num_gpu": 0
  to preserve GPU memory for the vision model).


8. HOW TO RUN THE COMPLETE APPLICATION
--------------------------------------------------------------------------------
EASY METHOD (Using Provided Batch Files):
  1. Double-click start_backend.bat  (starts FastAPI server on port 8000)
  2. Double-click start_frontend.bat (starts React dashboard on port 5173)
  3. Open your browser and go to: http://localhost:5173

MANUAL METHOD (Two Terminal Windows):
Terminal 1 - Backend:
  cd /d D:\NLP_PROJECT\backend
  ..\internvl_env\Scripts\python.exe -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

Terminal 2 - Frontend:
  cd /d D:\NLP_PROJECT\frontend
  npm run dev

Open browser at: http://localhost:5173


9. QUICK VERIFICATION
--------------------------------------------------------------------------------
1. Check Backend Liveness:
   Open browser at: http://localhost:8000/api/health
   Should return: {"status": "ok", "model_loaded": true, "multimodal_ready": true}

2. Interactive API Documentation:
   Visit: http://localhost:8000/docs

3. Test Text Sarcasm in UI:
   - Click "Sarcasm Detection" in the left sidebar.
   - Enter: "Oh absolutely fantastic, another flight delay right before the weekend!"
   - Click "Analyze Text". The confidence gauge and "Sarcastic" badge will appear.

4. Test Multimodal Sarcasm (Meme Upload):
   - Click "Multimodal Analysis" in the left sidebar.
   - Upload any test meme image (JPEG, PNG, or WEBP).
   - Enter caption: "Best pizza I have ever made! 🍕✨"
   - Click "Run Multimodal Inference".
   - View sentiment bars, emotion spectrum, latency breakdown, and verdict!


10. HOW TO STOP THE APPLICATION
--------------------------------------------------------------------------------
To shut down the servers safely:
  1. In the Frontend terminal, press CTRL + C, then enter 'Y'.
  2. In the Backend terminal, press CTRL + C.


11. COMMON TROUBLESHOOTING
--------------------------------------------------------------------------------
- Python is not recognized:
  Re-run Python installer and check "Add python.exe to PATH".

- npm is not recognized:
  Install Node.js LTS from https://nodejs.org.

- Running scripts is disabled (PowerShell):
  Run: Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned

- Port 8000 already in use:
  In PowerShell, run:
  Get-Process -Id (Get-NetTCPConnection -LocalPort 8000).OwningProcess | Stop-Process -Force

- Frontend shows "Backend Offline":
  Ensure backend server is running in Terminal 1 or via start_backend.bat.
  Check http://localhost:8000/api/health in your browser.

- CUDA Out of Memory (OOM):
  Close other GPU applications. The system needs ~5.4 GB free VRAM at peak.

- fusion_scaler.joblib not found:
  Ensure fusion_scaler.joblib is in models/multimodal_fusion/.

- Ollama is offline or not installed:
  Normal behavior. The built-in cognitive reasoner handles explanations in <1 ms.


12. REPOSITORY DIRECTORY STRUCTURE
--------------------------------------------------------------------------------
NLP_PROJECT/
|-- backend/
|   |-- app/
|   |   |-- main.py                    # FastAPI entry point
|   |   |-- routes/                    # predict.py, model.py, data.py
|   |   |-- services/                  # multimodal_service.py, explanation_service.py
|   |   `-- schemas/schemas.py         # Request & response validation
|   `-- tests/                         # Automated test suites
|-- frontend/
|   |-- src/pages/                     # 10 dashboard pages
|   |-- package.json                   # React 18, Vite, Tailwind
|   `-- vite.config.ts                 # Proxy configuration (port 5173 -> 8000)
|-- models/
|   |-- multimodal_fusion/             # MultimodalFusionMLP + fusion_scaler.joblib
|   |-- xlmr_text_sarcasm_full/        # XLM-R text sarcasm classifier
|   |-- xlmr_sentiment_model/          # XLM-R 3-class sentiment model
|   `-- xlmr_emotion_model/            # XLM-R 28-class GoEmotions model
|-- FINAL_DATASETS/                    # Master CSV datasets (optional for inference)
|-- start_backend.bat                  # One-click backend startup script
|-- start_frontend.bat                 # One-click frontend startup script
|-- .env.example                       # Environment configuration template
|-- README.txt                         # Full 26-section university report manual
|-- README.md                          # GitHub Markdown quick-start guide
`-- README_QUICKSTART.txt              # Plain-text quick-start guide (This file)
================================================================================
