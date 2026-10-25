"""Standalone Evaluator for Traditional Text Baselines.

Loads a saved text baseline checkpoint and generates full classification metrics,
confusion matrix, and predictions for a target jsonl dataset.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.metrics import compute_classification_metrics
from src.models.classification_head import PRIMARY_LABEL_TO_IDX, PRIMARY_IDX_TO_LABEL
from src.models.text_baseline import TextBaselineModel


def evaluate_text_baseline(
    model_path: str,
    test_jsonl: str,
    output_predictions: str = None,
) -> Dict[str, Any]:
    model = TextBaselineModel.load(model_path)
    texts = []
    labels = []
    record_ids = []

    with open(test_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            lbl = r.get("passive_aggression_label")
            if lbl in PRIMARY_LABEL_TO_IDX:
                comment = r.get("target_comment") or ""
                context = r.get("conversation_context") or r.get("caption") or ""
                full_text = f"{context} {comment}".strip() if context else comment
                texts.append(full_text)
                labels.append(PRIMARY_LABEL_TO_IDX[lbl])
                record_ids.append(r.get("example_id", ""))

    preds = model.predict(texts)
    probs = model.predict_proba(texts)
    class_names = [PRIMARY_IDX_TO_LABEL[i] for i in range(len(model.class_mapping))]
    metrics = compute_classification_metrics(labels, preds, class_names=class_names, prefix="test_")

    if output_predictions:
        preds_data = []
        for rid, true_l, pred_l, pr in zip(record_ids, labels, preds, probs):
            preds_data.append({
                "example_id": rid,
                "ground_truth": class_names[true_l],
                "predicted": class_names[pred_l],
                "probabilities": {class_names[i]: float(pr[i]) for i in range(len(class_names))},
            })
        out_p = Path(output_predictions)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(preds_data, f, indent=2)

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Text Baseline")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--test_data", type=str, required=True)
    parser.add_argument("--output_preds", type=str, default=None)
    args = parser.parse_args()

    results = evaluate_text_baseline(args.model, args.test_data, args.output_preds)
    print(json.dumps(results, indent=2))
