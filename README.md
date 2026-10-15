# Multimodal Sarcasm Intelligence

A research-grade multimodal AI dashboard and inference system for the project:
**"A Multimodal Vision-Language Framework for Gen-Z Sarcasm Detection using XLM-R, CLIP and Vision-Language Models"**

---

## 🔬 Project Status & Scientific Rigor

This application is built with complete scientific transparency:

- **Text Sarcasm Model (XLM-R):** `READY & OPERATIONAL`. Performs real-time inference on `D:\NLP_PROJECT\models\xlmr_text_sarcasm_full` fine-tuned on 1,005,167 samples.
- **Visual Features (CLIP ViT-B/32):** `PRE-EXTRACTED (33,077 × 512-D)`.
- **Vision-Language Features (InternVL3-2B):** `PRE-EXTRACTED (33,077 × 1536-D)`.
- **Sentiment & Emotion Models:** `COMING SOON`.
- **Multimodal Fusion Classifier:** `IN DEVELOPMENT (Planned 2847-D fusion)`.

*No fake or simulated predictions are fabricated for un-trained components.*

---

## 🛠️ Tech Stack

- **Backend:** Python 3.12, FastAPI, Uvicorn, PyTorch (CUDA), Hugging Face Transformers (`XLM-RoBERTa`)
- **Frontend:** React 18, Vite, TypeScript, Tailwind CSS, Recharts, Framer Motion, Lucide React
- **Architecture:** Modular Heterogeneous Multimodal Pipeline

---

## 🚀 Quick Start (Windows)

### Option 1: Using Batch Scripts

1. Double-click **`start_backend.bat`** to start the FastAPI server on `http://localhost:8000`.
2. Double-click **`start_frontend.bat`** to start the React UI on `http://localhost:5173`.
3. Open your browser at `http://localhost:5173`.

---

### Option 2: Manual Terminal Execution

#### 1. Backend Setup & Run
```powershell
# Navigate to project root
cd D:\NLP_PROJECT

# Run FastAPI backend using existing environment
D:\NLP_PROJECT\internvl_env\Scripts\python.exe -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
```

#### 2. Frontend Setup & Run
```powershell
# In a new PowerShell terminal:
$env:PATH = "C:\Program Files\nodejs;$env:PATH"
cd D:\NLP_PROJECT\frontend

# Install dependencies (first time only)
npm install

# Start development server
npm run dev
```

---

## 📡 REST API Endpoints

| Method | Endpoint | Description | Status |
|--------|----------|-------------|--------|
| `GET` | `/api/health` | Backend liveness status & model loaded flag | Operational |
| `GET` | `/api/model/status` | Current XLM-R load state and active compute device | Operational |
| `POST` | `/api/predict/text` | Real-time XLM-R text sarcasm classification with confidence | Operational |
| `GET` | `/api/model/metrics` | Test evaluation metrics (Accuracy: 77.19%, Macro F1: 77.19%) | Operational |
| `GET` | `/api/dataset/stats` | Sample splits and distributions for Text (1.005M) & Multimodal (33k) | Operational |
| `GET` | `/api/features/status` | Extraction status and dimension tracking for all modalities | Operational |
| `POST` | `/api/predict/multimodal` | Multimodal prediction stub returning descriptive `not_ready` payload | In Development |

---

## 📊 Pages & Dashboard Sections

1. **Overview:** Executive summary, component status cards, and dual-pipeline visualization.
2. **Sarcasm Detection:** Live XLM-R inference with radial confidence gauge, probability bars, and example selector.
3. **Text Analysis:** Rule-based linguistic statistics (slang, punctuation, casing, counters).
4. **Multimodal Analysis:** Vision-language pipeline staging with image preview and status badges.
5. **Model Architecture:** Interactive clickable diagram with 2847-D breakdown modal inspector.
6. **Model Performance:** Test set confusion matrix heatmap, metric comparison chart, and class breakdowns.
7. **Feature Extraction:** Pre-extracted feature monitoring and dimensional progression.
8. **Dataset:** Recharts visualizations of 1M text dataset and 33k multimodal splits.
9. **Research & Novelty:** Theoretical motivation, cross-modal incongruity, and multi-aspect contribution.
10. **About Project:** System environment specs (RTX 4050, PyTorch, Transformers) and artifact references.
