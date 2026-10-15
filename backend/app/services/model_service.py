"""
Model service — loads XLM-R once at startup and provides inference.
The model is NEVER reloaded per-request.
"""
import logging
import os
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

logger = logging.getLogger(__name__)

MODEL_PATH = r"D:\NLP_PROJECT\models\xlmr_text_sarcasm_full"
MAX_LENGTH = 128

# ── Singleton state ──────────────────────────────────────────────────────────
_tokenizer = None
_model = None
_device = None
_model_loaded = False
_load_error: str | None = None


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model() -> None:
    """Load XLM-R tokenizer + classification head. Called once at startup."""
    global _tokenizer, _model, _device, _model_loaded, _load_error

    try:
        logger.info(f"Loading XLM-R model from: {MODEL_PATH}")
        _device = get_device()
        logger.info(f"Using device: {_device}")

        _tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        _model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
        _model.to(_device)
        _model.eval()

        _model_loaded = True
        _load_error = None
        logger.info("XLM-R model loaded successfully.")
    except Exception as exc:
        _model_loaded = False
        _load_error = str(exc)
        logger.error(f"Failed to load model: {exc}", exc_info=True)
        raise


def is_loaded() -> bool:
    return _model_loaded


def get_load_error() -> str | None:
    return _load_error


def get_device_name() -> str:
    if _device is None:
        return "unknown"
    if _device.type == "cuda":
        return f"cuda ({torch.cuda.get_device_name(0)})"
    return "cpu"


def predict_text(text: str) -> dict:
    """
    Run inference on text. Returns dict with label, label_name,
    confidence, and raw probabilities.
    """
    if not _model_loaded:
        raise RuntimeError("Model is not loaded. Check startup logs.")

    inputs = _tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH,
        padding=True,
    )
    inputs = {k: v.to(_device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = _model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1).squeeze().cpu().tolist()

    # probs[0] = non-sarcastic, probs[1] = sarcastic
    non_sarc_prob = float(probs[0])
    sarc_prob = float(probs[1])
    label = int(sarc_prob > non_sarc_prob)
    confidence = max(non_sarc_prob, sarc_prob)
    label_name = "Sarcastic" if label == 1 else "Non-Sarcastic"

    return {
        "label": label,
        "label_name": label_name,
        "confidence": round(confidence, 4),
        "probabilities": {
            "non_sarcastic": round(non_sarc_prob, 4),
            "sarcastic": round(sarc_prob, 4),
        },
        "device": get_device_name(),
    }


def get_model_metrics() -> dict:
    """Read evaluation metrics from the saved JSON file."""
    metrics_path = Path(MODEL_PATH) / "final_test_results.json"
    if metrics_path.exists():
        with open(metrics_path, "r", encoding="utf-8") as f:
            return json.load(f)
    # Fallback hardcoded values
    return {
        "model": "FacebookAI/xlm-roberta-base",
        "train_samples": 812130,
        "validation_samples": 96518,
        "test_samples": 96519,
        "test_accuracy": 0.7719412758109803,
        "test_macro_precision": 0.771944280101952,
        "test_macro_recall": 0.7719563088832025,
        "test_macro_f1": 0.7719393151718419,
        "confusion_matrix": [[37112, 10824], [11188, 37395]],
        "epochs": 3,
        "learning_rate": 2e-05,
    }


def get_all_comprehensive_metrics() -> dict:
    """Read all model evaluation results and ablation benchmark summary from disk."""
    import csv

    project_root = Path(r"D:\NLP_PROJECT")

    # 1. Text Sarcasm
    text_path = project_root / "models" / "xlmr_text_sarcasm_full" / "final_test_results.json"
    text_metrics = {}
    if text_path.exists():
        try:
            with open(text_path, "r", encoding="utf-8") as f:
                text_metrics = json.load(f)
        except Exception as exc:
            logger.warning(f"Error loading text_metrics: {exc}")

    # 2. Sentiment
    sent_path = project_root / "models" / "xlmr_sentiment_model" / "final_test_results.json"
    sent_metrics = {}
    if sent_path.exists():
        try:
            with open(sent_path, "r", encoding="utf-8") as f:
                sent_metrics = json.load(f)
        except Exception as exc:
            logger.warning(f"Error loading sent_metrics: {exc}")

    # 3. Emotion
    emo_path = project_root / "models" / "xlmr_emotion_model" / "final_test_results.json"
    emo_metrics = {}
    if emo_path.exists():
        try:
            with open(emo_path, "r", encoding="utf-8") as f:
                emo_metrics = json.load(f)
        except Exception as exc:
            logger.warning(f"Error loading emo_metrics: {exc}")

    # 4. Multimodal Fusion
    fusion_path = project_root / "models" / "multimodal_fusion" / "test_results.json"
    fusion_metrics = {}
    if fusion_path.exists():
        try:
            with open(fusion_path, "r", encoding="utf-8") as f:
                fusion_metrics = json.load(f)
        except Exception as exc:
            logger.warning(f"Error loading fusion_metrics: {exc}")

    # 5. Ablations
    ablation_path = project_root / "models" / "multimodal_fusion" / "ablation_benchmark_summary.csv"
    ablations = []
    if ablation_path.exists():
        try:
            with open(ablation_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    ablations.append({
                        "configuration": row.get("Configuration", ""),
                        "description": row.get("Description", ""),
                        "dimensions": int(row.get("Dimensions", 0)),
                        "best_epoch": int(row.get("Best Epoch", 0)),
                        "val_macro_f1": float(row.get("Val Macro F1", 0.0)),
                        "test_accuracy": float(row.get("Test Accuracy", 0.0)),
                        "test_macro_f1": float(row.get("Test Macro F1", 0.0)),
                        "test_macro_precision": float(row.get("Test Macro P", 0.0)),
                        "test_macro_recall": float(row.get("Test Macro R", 0.0)),
                        "test_weighted_f1": float(row.get("Test Weighted F1", 0.0)),
                        "class_0_f1": float(row.get("Class 0 F1", 0.0)),
                        "class_1_f1": float(row.get("Class 1 F1", 0.0)),
                        "train_time_seconds": float(row.get("Train Time (s)", 0.0)),
                        "latency_ms": float(row.get("Latency (ms)", 0.0)),
                        "peak_vram_mb": float(row.get("Peak VRAM (MB)", 0.0)),
                    })
        except Exception as exc:
            logger.warning(f"Error loading ablation_summary: {exc}")

    return {
        "text_sarcasm": text_metrics,
        "sentiment": sent_metrics,
        "emotion": emo_metrics,
        "multimodal_fusion": fusion_metrics,
        "ablation_benchmark": ablations,
    }
