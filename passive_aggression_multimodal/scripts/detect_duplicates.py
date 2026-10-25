"""Duplicate and data-leakage detection script."""
from __future__ import annotations

import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utilities.hash_utils import compute_file_sha256
from src.utilities.logger import setup_console_logger

logger = setup_console_logger("duplicate_detector")


def check_text_duplicates():
    """Detects exact duplicate texts within and across datasets."""
    logger.info("Analyzing text duplicates across datasets...")
    text_to_records = defaultdict(list)

    # 1. MMSD
    mmsd_raw = PROJECT_ROOT / "data" / "raw" / "mmsd"
    for split in ["mmsd_train.json", "mmsd_valid.json", "mmsd_test.json"]:
        p = mmsd_raw / split
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
                for item in data:
                    txt = item.get("text", "").strip()
                    if txt:
                        text_to_records[txt].append({
                            "dataset": "mmsd",
                            "split": split,
                            "id": item.get("image_id"),
                            "label": item.get("label"),
                        })

    # 2. SarcNet
    sarcnet_raw = PROJECT_ROOT / "data" / "raw" / "sarcnet"
    for split in ["SarcNetTrain.csv", "SarcNetVal.csv", "SarcNetTest.csv"]:
        p = sarcnet_raw / split
        if p.exists():
            with open(p, "r", encoding="gb18030", errors="replace") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    txt = row.get("Text", "").strip()
                    if txt:
                        text_to_records[txt].append({
                            "dataset": "sarcnet",
                            "split": split,
                            "id": row.get("Imagepath"),
                            "label": row.get("Multi_label"),
                        })

    # 3. GoEmotions
    goemotions_raw = PROJECT_ROOT / "data" / "raw" / "goemotions"
    for split in ["train.tsv", "dev.tsv", "test.tsv"]:
        p = goemotions_raw / split
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split("\t")
                    if parts:
                        txt = parts[0].strip()
                        if txt:
                            text_to_records[txt].append({
                                "dataset": "goemotions",
                                "split": split,
                                "id": parts[2] if len(parts) > 2 else None,
                                "label": parts[1] if len(parts) > 1 else None,
                            })

    total_unique_texts = len(text_to_records)
    duplicates = {t: recs for t, recs in text_to_records.items() if len(recs) > 1}
    cross_dataset_duplicates = {
        t: recs for t, recs in duplicates.items()
        if len(set(r["dataset"] for r in recs)) > 1
    }

    # Check for label contradictions in duplicates
    label_contradictions = []
    for t, recs in duplicates.items():
        labels = set(str(r["label"]) for r in recs if r["label"] is not None)
        if len(labels) > 1 and len(set(r["dataset"] for r in recs)) == 1:
            label_contradictions.append({
                "text": t,
                "occurrences": recs,
            })

    return {
        "total_unique_texts": total_unique_texts,
        "duplicate_text_count": len(duplicates),
        "cross_dataset_duplicate_count": len(cross_dataset_duplicates),
        "internal_label_contradiction_count": len(label_contradictions),
        "sample_cross_dataset_duplicates": [
            {"text": t[:80], "occurrences": recs}
            for t, recs in list(cross_dataset_duplicates.items())[:5]
        ],
        "sample_label_contradictions": label_contradictions[:5],
    }


def check_image_duplicates():
    """Detects exact duplicate images based on SHA-256 hash."""
    logger.info("Analyzing image hash duplicates...")
    mmsd_dir = PROJECT_ROOT / "data" / "images" / "mmsd"
    sarcnet_dir = PROJECT_ROOT / "data" / "images" / "sarcnet"

    hash_to_images = defaultdict(list)

    # MMSD sample (check first 2000 images for speed)
    if mmsd_dir.exists():
        files = list(mmsd_dir.glob("*.*"))[:2000]
        for f in files:
            h = compute_file_sha256(str(f))
            hash_to_images[h].append({"dataset": "mmsd", "path": f.name})

    # SarcNet (all 3335 images)
    if sarcnet_dir.exists():
        for f in sarcnet_dir.glob("*.*"):
            h = compute_file_sha256(str(f))
            hash_to_images[h].append({"dataset": "sarcnet", "path": f.name})

    duplicate_hashes = {h: imgs for h, imgs in hash_to_images.items() if len(imgs) > 1}
    cross_dataset_image_duplicates = {
        h: imgs for h, imgs in duplicate_hashes.items()
        if len(set(i["dataset"] for i in imgs)) > 1
    }

    return {
        "checked_images_count": sum(len(imgs) for imgs in hash_to_images.values()),
        "duplicate_image_hash_count": len(duplicate_hashes),
        "cross_dataset_image_matches": len(cross_dataset_image_duplicates),
        "sample_duplicate_images": list(duplicate_hashes.values())[:5],
    }


def main():
    logger.info("=== Running Duplicate & Leakage Audit ===")
    text_results = check_text_duplicates()
    image_results = check_image_duplicates()

    report = {
        "text_duplicates": text_results,
        "image_duplicates": image_results,
    }

    out_file = PROJECT_ROOT / "data" / "manifests" / "duplicate_audit_report.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    logger.info(f"Duplicate audit report saved to {out_file}")
    logger.info(f"Unique texts: {text_results['total_unique_texts']}, Duplicate text entries: {text_results['duplicate_text_count']}")
    logger.info(f"Cross-dataset duplicate texts: {text_results['cross_dataset_duplicate_count']}")
    logger.info(f"Duplicate image hashes: {image_results['duplicate_image_hash_count']}")


if __name__ == "__main__":
    main()
