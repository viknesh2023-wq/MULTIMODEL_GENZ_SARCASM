"""Dataset inspection script analyzing text, labels, context, and schema across datasets."""
from __future__ import annotations

import csv
import json
import os
import sys
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utilities.logger import setup_console_logger

logger = setup_console_logger("dataset_inspector")


def inspect_mmsd():
    """Inspect MMSD / MMSD2.0 benchmark JSONs."""
    raw_dir = PROJECT_ROOT / "data" / "raw" / "mmsd"
    splits = ["mmsd_train.json", "mmsd_valid.json", "mmsd_test.json"]
    total_records = 0
    all_texts = []
    labels = Counter()
    image_ids = set()

    for s in splits:
        p = raw_dir / s
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
                total_records += len(data)
                for item in data:
                    txt = item.get("text", "")
                    lbl = item.get("label")
                    img_id = item.get("image_id")
                    if txt:
                        all_texts.append(txt)
                    labels[lbl] += 1
                    if img_id:
                        image_ids.add(str(img_id))

    lengths = [len(t.split()) for t in all_texts]
    return {
        "dataset": "MMSD / MMSD2.0",
        "total_records": total_records,
        "unique_texts": len(set(all_texts)),
        "unique_images": len(image_ids),
        "label_distribution": dict(labels),
        "text_length_words": {
            "min": int(np.min(lengths)) if lengths else 0,
            "max": int(np.max(lengths)) if lengths else 0,
            "mean": float(np.mean(lengths)) if lengths else 0.0,
            "median": float(np.median(lengths)) if lengths else 0.0,
        },
        "has_context": False,
        "has_images": True,
        "multimodal_type": "Image + Tweet Text",
    }


def inspect_sarcnet():
    """Inspect SarcNet CSVs."""
    raw_dir = PROJECT_ROOT / "data" / "raw" / "sarcnet"
    files = ["SarcNetTrain.csv", "SarcNetVal.csv", "SarcNetTest.csv"]
    total_records = 0
    all_texts = []
    multi_labels = Counter()
    text_labels = Counter()
    image_labels = Counter()
    image_paths = set()

    for fname in files:
        p = raw_dir / fname
        if p.exists():
            with open(p, "r", encoding="gb18030", errors="replace") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    total_records += 1
                    txt = row.get("Text", "").strip()
                    img = row.get("Imagepath", "").strip()
                    if txt:
                        all_texts.append(txt)
                    if img:
                        image_paths.add(img)
                    multi_labels[row.get("Multi_label", "").strip()] += 1
                    text_labels[row.get("Text_label", "").strip()] += 1
                    image_labels[row.get("Image_label", "").strip()] += 1

    lengths = [len(t) for t in all_texts]
    return {
        "dataset": "SarcNet",
        "total_records": total_records,
        "unique_texts": len(set(all_texts)),
        "unique_images": len(image_paths),
        "multi_label_distribution": dict(multi_labels),
        "text_label_distribution": dict(text_labels),
        "image_label_distribution": dict(image_labels),
        "text_length_chars": {
            "min": int(np.min(lengths)) if lengths else 0,
            "max": int(np.max(lengths)) if lengths else 0,
            "mean": float(np.mean(lengths)) if lengths else 0.0,
            "median": float(np.median(lengths)) if lengths else 0.0,
        },
        "has_context": False,
        "has_images": True,
        "multimodal_type": "Image + Chinese/English Social Post",
    }


def inspect_goemotions():
    """Inspect GoEmotions Reddit TSVs."""
    raw_dir = PROJECT_ROOT / "data" / "raw" / "goemotions"
    files = ["train.tsv", "dev.tsv", "test.tsv"]
    total_records = 0
    all_texts = []
    emotion_counter = Counter()

    # Load emotion mapping
    emotions_file = raw_dir / "emotions.txt"
    emotion_names = []
    if emotions_file.exists():
        with open(emotions_file, "r", encoding="utf-8") as f:
            emotion_names = [line.strip() for line in f if line.strip()]

    for fname in files:
        p = raw_dir / fname
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split("\t")
                    if len(parts) >= 2:
                        total_records += 1
                        txt = parts[0]
                        all_texts.append(txt)
                        # parts[1] is comma separated emotion indices
                        for idx_str in parts[1].split(","):
                            if idx_str.strip().isdigit():
                                idx = int(idx_str.strip())
                                name = emotion_names[idx] if idx < len(emotion_names) else f"emo_{idx}"
                                emotion_counter[name] += 1

    lengths = [len(t.split()) for t in all_texts]
    # Highlight PA-adjacent emotions: annoyance, disapproval, disappointment
    pa_adjacent = {
        "annoyance": emotion_counter.get("annoyance", 0),
        "disapproval": emotion_counter.get("disapproval", 0),
        "disappointment": emotion_counter.get("disappointment", 0),
        "anger": emotion_counter.get("anger", 0),
    }

    return {
        "dataset": "GoEmotions",
        "total_records": total_records,
        "unique_texts": len(set(all_texts)),
        "pa_adjacent_emotion_counts": pa_adjacent,
        "top_5_emotions": emotion_counter.most_common(5),
        "text_length_words": {
            "min": int(np.min(lengths)) if lengths else 0,
            "max": int(np.max(lengths)) if lengths else 0,
            "mean": float(np.mean(lengths)) if lengths else 0.0,
            "median": float(np.median(lengths)) if lengths else 0.0,
        },
        "has_context": False,
        "has_images": False,
        "multimodal_type": "Text-Only (Reddit Comments)",
    }


def inspect_civil_comments():
    """Inspect Civil Comments dataset."""
    val_path = PROJECT_ROOT / "data" / "raw" / "civil_comments" / "civil_comments_val.parquet"
    if not val_path.exists():
        return {"dataset": "Civil Comments", "error": "Not found"}

    df = pd.read_parquet(val_path)
    lengths = df["text"].dropna().str.split().str.len()

    # Toxicity breakdown
    toxic_count = int((df["toxicity"] >= 0.5).sum())
    non_toxic_count = int((df["toxicity"] < 0.5).sum())
    severe_toxic_count = int((df["severe_toxicity"] >= 0.5).sum())
    insult_count = int((df["insult"] >= 0.5).sum())

    return {
        "dataset": "Civil Comments",
        "total_records": len(df),
        "unique_texts": df["text"].nunique(),
        "toxic_threshold_0_5": toxic_count,
        "non_toxic_threshold_0_5": non_toxic_count,
        "severe_toxic_count": severe_toxic_count,
        "insult_count": insult_count,
        "text_length_words": {
            "min": int(lengths.min()),
            "max": int(lengths.max()),
            "mean": float(lengths.mean()),
            "median": float(lengths.median()),
        },
        "has_context": False,
        "has_images": False,
        "multimodal_type": "Text-Only (News Comment Threads)",
    }


def inspect_mustard():
    """Inspect MUStARD dialogue dataset."""
    p = PROJECT_ROOT / "data" / "raw" / "mustard" / "sarcasm_data.json"
    if not p.exists():
        return {"dataset": "MUStARD", "error": "Not found"}

    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)

    sarcastic_count = sum(1 for v in data.values() if v.get("sarcasm") is True)
    non_sarcastic_count = sum(1 for v in data.values() if v.get("sarcasm") is False)
    shows = Counter(v.get("show") for v in data.values())
    speakers = Counter(v.get("speaker") for v in data.values())
    context_lengths = [len(v.get("context", [])) for v in data.values()]

    return {
        "dataset": "MUStARD",
        "total_records": len(data),
        "sarcastic_count": sarcastic_count,
        "non_sarcastic_count": non_sarcastic_count,
        "show_distribution": dict(shows),
        "top_speakers": speakers.most_common(5),
        "context_turns": {
            "min": int(np.min(context_lengths)),
            "max": int(np.max(context_lengths)),
            "mean": float(np.mean(context_lengths)),
            "median": float(np.median(context_lengths)),
        },
        "has_context": True,
        "has_images": False,
        "multimodal_type": "Dialogue Audio/Video + Text Utterance & Context",
    }


def main():
    logger.info("=== Running Dataset Inspection & Quality Profiling ===")
    report = {
        "mmsd": inspect_mmsd(),
        "sarcnet": inspect_sarcnet(),
        "goemotions": inspect_goemotions(),
        "civil_comments": inspect_civil_comments(),
        "mustard": inspect_mustard(),
    }

    out_file = PROJECT_ROOT / "data" / "manifests" / "dataset_inspection_profile.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    logger.info(f"Dataset inspection profile saved to {out_file}")
    for k, v in report.items():
        logger.info(f"  {k.upper()}: {v.get('total_records', 'N/A')} records, Modality: {v.get('multimodal_type', 'N/A')}")


if __name__ == "__main__":
    main()
