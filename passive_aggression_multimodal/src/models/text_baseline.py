"""Traditional Text-Only Baseline Models (TF-IDF + Logistic Regression / Linear SVM).

Preserves pragmatic cues (punctuation, emojis, casing, negation) using combined word
and character n-gram representations with scikit-learn classifiers.
"""
from __future__ import annotations

import os
import pickle
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

from src.models.classification_head import PRIMARY_LABEL_TO_IDX, PRIMARY_IDX_TO_LABEL


def pragmatic_text_cleaner(text: str) -> str:
    """
    Light normalizer preserving emojis, capitalization emphasis, repeated punctuation, and negation.
    """
    if not text:
        return ""
    # Standardize whitespace while preserving case and punctuation
    text = re.sub(r"[\r\n\t]+", " ", text)
    text = re.sub(r" +", " ", text)
    return text.strip()


class TextBaselineModel:
    """
    Text-only baseline using combined Word + Char n-gram TF-IDF and Linear Classifiers.
    """

    def __init__(
        self,
        model_type: str = "logistic_regression",  # "logistic_regression" or "linear_svm"
        c_param: float = 1.0,
        class_weight: Optional[str] = "balanced",
        random_state: int = 42,
        class_mapping: Optional[Dict[str, int]] = None,
    ):
        self.model_type = model_type.lower()
        self.c_param = c_param
        self.class_weight = class_weight
        self.random_state = random_state
        self.class_mapping = class_mapping or PRIMARY_LABEL_TO_IDX
        self.idx_to_class = {v: k for k, v in self.class_mapping.items()}

        # Build feature union: word n-grams (1, 2) + character n-grams (3, 5)
        word_vectorizer = TfidfVectorizer(
            preprocessor=pragmatic_text_cleaner,
            ngram_range=(1, 2),
            min_df=2,
            max_features=10000,
            sublinear_tf=True,
        )
        char_vectorizer = TfidfVectorizer(
            preprocessor=pragmatic_text_cleaner,
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=3,
            max_features=15000,
            sublinear_tf=True,
        )
        self.feature_extractor = FeatureUnion([
            ("word_tfidf", word_vectorizer),
            ("char_tfidf", char_vectorizer),
        ])

        # Base classifier
        if self.model_type == "logistic_regression":
            self.classifier = LogisticRegression(
                C=self.c_param,
                class_weight=self.class_weight,
                max_iter=1000,
                random_state=self.random_state,
                solver="lbfgs",
            )
        elif self.model_type == "linear_svm":
            base_svm = LinearSVC(
                C=self.c_param,
                class_weight=self.class_weight,
                max_iter=2000,
                random_state=self.random_state,
            )
            # Calibrate to produce valid posterior probabilities
            self.classifier = CalibratedClassifierCV(base_svm, cv=3)
        else:
            raise ValueError(
                f"Unsupported model_type '{model_type}'. Choose 'logistic_regression' or 'linear_svm'."
            )

        self.pipeline = Pipeline([
            ("features", self.feature_extractor),
            ("clf", self.classifier),
        ])
        self.is_fitted = False

    def fit(self, texts: List[str], labels: List[int]) -> TextBaselineModel:
        """Fits vectorizers and classifier on training split."""
        clean_texts = [pragmatic_text_cleaner(t) for t in texts]
        self.pipeline.fit(clean_texts, labels)
        self.is_fitted = True
        return self

    def predict(self, texts: List[str]) -> np.ndarray:
        """Predicts class labels for text instances."""
        if not self.is_fitted:
            raise RuntimeError("Model has not been fitted.")
        clean_texts = [pragmatic_text_cleaner(t) for t in texts]
        return self.pipeline.predict(clean_texts)

    def predict_proba(self, texts: List[str]) -> np.ndarray:
        """Returns softmax-calibrated probability distributions over classes."""
        if not self.is_fitted:
            raise RuntimeError("Model has not been fitted.")
        clean_texts = [pragmatic_text_cleaner(t) for t in texts]
        return self.pipeline.predict_proba(clean_texts)

    def save(self, filepath: Union[str, Path], metadata: Optional[Dict[str, Any]] = None) -> None:
        """Serializes fitted pipeline to disk."""
        out_path = Path(filepath)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "pipeline": self.pipeline,
            "model_type": self.model_type,
            "c_param": self.c_param,
            "class_weight": self.class_weight,
            "random_state": self.random_state,
            "class_mapping": self.class_mapping,
            "idx_to_class": self.idx_to_class,
            "is_fitted": self.is_fitted,
            "metadata": metadata or {},
        }
        with open(out_path, "wb") as f:
            pickle.dump(payload, f)

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> TextBaselineModel:
        """Loads serialized model pipeline from disk."""
        with open(filepath, "rb") as f:
            payload = pickle.load(f)

        model = cls(
            model_type=payload["model_type"],
            c_param=payload["c_param"],
            class_weight=payload["class_weight"],
            random_state=payload["random_state"],
            class_mapping=payload["class_mapping"],
        )
        model.pipeline = payload["pipeline"]
        model.is_fitted = payload["is_fitted"]
        return model
