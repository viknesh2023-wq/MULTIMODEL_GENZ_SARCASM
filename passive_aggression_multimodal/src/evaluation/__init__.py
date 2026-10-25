"""Evaluation metrics, split generation, and benchmark evaluation for Phase 4 and Phase 5."""
from src.evaluation.metrics import (
    compute_classification_metrics,
    compute_bootstrap_confidence_intervals,
    compute_calibration_error,
    compute_selective_metrics,
    save_evaluation_predictions,
)
from src.evaluation.human_split_generator import HumanSplitGenerator
from src.evaluation.benchmark_evaluator import RealWorldBenchmarkEvaluator

__all__ = [
    "compute_classification_metrics",
    "compute_bootstrap_confidence_intervals",
    "compute_calibration_error",
    "compute_selective_metrics",
    "save_evaluation_predictions",
    "HumanSplitGenerator",
    "RealWorldBenchmarkEvaluator",
]
