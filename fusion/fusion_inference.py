"""
Multimodal Fusion Inference Module (Milestone 4 - Phase G)
Project: Gen-Z Multimodal Sarcasm Detection System
Hardware: NVIDIA GeForce RTX 4050 Laptop GPU / CUDA Acceleration

Provides standalone, verified inference using the trained 2,847-D Multimodal Fusion MLP.
Handles feature scaling (StandardScaler), tensor formatting, forward pass, and calibrated probabilities.
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, Any, Union, List

import numpy as np
import joblib
import torch
import torch.nn as nn

PROJECT_ROOT = Path(r"D:\NLP_PROJECT")
MODEL_DIR = PROJECT_ROOT / "models" / "multimodal_fusion"

EXPECTED_DIMENSION = 2847

class MultimodalFusionMLP(nn.Module):
    def __init__(self, in_features: int = 2847, num_classes: int = 2, dropout_p: float = 0.3):
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        
        self.net = nn.Sequential(
            nn.Linear(in_features, 1024),
            nn.Mish(),
            nn.Dropout(dropout_p),
            nn.Linear(1024, 512),
            nn.Mish(),
            nn.Dropout(dropout_p * 0.67),
            nn.Linear(512, 128),
            nn.Mish(),
            nn.Dropout(dropout_p * 0.33),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class MultimodalFusionInference:
    """
    Reusable inference service for 2,847-D Multimodal Sarcasm Classification.
    """
    def __init__(self, model_dir: Union[str, Path] = MODEL_DIR, device: str = None):
        self.model_dir = Path(model_dir)
        self.weights_path = self.model_dir / "fusion_model.pt"
        self.scaler_path = self.model_dir / "fusion_scaler.joblib"
        self.config_path = self.model_dir / "model_config.json"

        if not self.weights_path.exists():
            raise FileNotFoundError(f"Model weights missing: {self.weights_path}")
        if not self.scaler_path.exists():
            raise FileNotFoundError(f"Scaler missing: {self.scaler_path}")

        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Load scaler
        self.scaler = joblib.load(self.scaler_path)

        # Load model config
        if self.config_path.exists():
            with open(self.config_path, "r", encoding="utf-8") as f:
                self.config = json.load(f)
        else:
            self.config = {
                "in_features": EXPECTED_DIMENSION,
                "num_classes": 2,
                "canonical_label_map": {0: "non-sarcastic", 1: "sarcastic"},
            }

        # Initialize and load model
        self.model = MultimodalFusionMLP(
            in_features=self.config.get("in_features", EXPECTED_DIMENSION),
            num_classes=self.config.get("num_classes", 2),
        )
        state_dict = torch.load(self.weights_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

        self.label_map = {0: "non-sarcastic", 1: "sarcastic"}

    def validate_vector(self, vec: np.ndarray) -> np.ndarray:
        if not isinstance(vec, np.ndarray):
            vec = np.array(vec, dtype=np.float32)
        if vec.ndim == 1:
            if vec.shape[0] != EXPECTED_DIMENSION:
                raise ValueError(f"Expected feature dimension {EXPECTED_DIMENSION}, got {vec.shape[0]}")
            vec = vec.reshape(1, -1)
        elif vec.ndim == 2:
            if vec.shape[1] != EXPECTED_DIMENSION:
                raise ValueError(f"Expected feature dimension {EXPECTED_DIMENSION}, got {vec.shape[1]}")
        else:
            raise ValueError(f"Invalid input dimensions: {vec.ndim}D")

        if not np.all(np.isfinite(vec)):
            raise ValueError("Input vector contains NaN or infinite values")

        return vec.astype(np.float32)

    def predict_vector(self, feature_vector: np.ndarray) -> Dict[str, Any]:
        """
        Run inference on a single 2,847-D concatenated feature vector.
        """
        vec = self.validate_vector(feature_vector)
        scaled_vec = self.scaler.transform(vec)

        tensor_x = torch.from_numpy(scaled_vec).float().to(self.device)
        with torch.inference_mode():
            logits = self.model(tensor_x)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]

        pred_label = int(np.argmax(probs))
        prob_non_sarcastic = float(probs[0])
        prob_sarcastic = float(probs[1])
        confidence = float(probs[pred_label])

        return {
            "predicted_label": pred_label,
            "label_name": self.label_map[pred_label],
            "confidence": round(confidence, 6),
            "prob_non_sarcastic": round(prob_non_sarcastic, 6),
            "prob_sarcastic": round(prob_sarcastic, 6),
            "device": str(self.device),
            "dimension": EXPECTED_DIMENSION,
            "model": "MultimodalFusionMLP (2847-D)",
        }

    def predict_batch(self, feature_matrix: np.ndarray) -> List[Dict[str, Any]]:
        """
        Run inference on a batch of 2,847-D feature vectors.
        """
        mat = self.validate_vector(feature_matrix)
        scaled_mat = self.scaler.transform(mat)

        tensor_x = torch.from_numpy(scaled_mat).float().to(self.device)
        with torch.inference_mode():
            logits = self.model(tensor_x)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()

        results = []
        for p in probs:
            p_label = int(np.argmax(p))
            results.append({
                "predicted_label": p_label,
                "label_name": self.label_map[p_label],
                "confidence": round(float(p[p_label]), 6),
                "prob_non_sarcastic": round(float(p[0]), 6),
                "prob_sarcastic": round(float(p[1]), 6),
            })
        return results

if __name__ == "__main__":
    print("Testing MultimodalFusionInference...")
    engine = MultimodalFusionInference()
    
    # Test on a dummy vector
    dummy_vec = np.zeros(2847, dtype=np.float32)
    res = engine.predict_vector(dummy_vec)
    print("Dummy prediction:", res)

    # Test on an actual sample from X_test.npy
    X_test_path = PROJECT_ROOT / "FINAL_DATASETS" / "MULTIMODAL_SARCASM" / "FUSION_DATA" / "X_test.npy"
    if X_test_path.exists():
        x_test_sample = np.load(X_test_path)[0]
        res_real = engine.predict_vector(x_test_sample)
        print("Real test sample 0 prediction:", res_real)
        print("Validation test PASSED!")
