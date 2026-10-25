"""Unified manifest generator transforming raw datasets into validated UnifiedRecord entries.

Outputs:
- data/manifests/mmsd_manifest.jsonl
- data/manifests/sarcnet_manifest.jsonl
- data/manifests/goemotions_manifest.jsonl
- data/manifests/civil_comments_manifest.jsonl
- data/manifests/mustard_manifest.jsonl
- data/interim/unified_raw_manifest.jsonl
"""
from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path
import pandas as pd

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utilities.hash_utils import compute_file_sha256
from src.utilities.logger import setup_console_logger
from src.utilities.schema import (
    AnnotationStatus,
    ImageAvailabilityStatus,
    SyntheticStatus,
    UnifiedRecord,
)

logger = setup_console_logger("manifest_generator")


def generate_mmsd_manifest():
    """Generates manifest for MMSD / MMSD2.0 benchmark."""
    logger.info("Generating MMSD manifest...")
    raw_dir = PROJECT_ROOT / "data" / "raw" / "mmsd"
    img_dir = PROJECT_ROOT / "data" / "images" / "mmsd"
    records = []

    splits = [
        ("train", "mmsd_train.json"),
        ("validation", "mmsd_valid.json"),
        ("test", "mmsd_test.json"),
    ]

    for split_name, fname in splits:
        p = raw_dir / fname
        if not p.exists():
            continue
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
            for idx, item in enumerate(data):
                img_id = str(item.get("image_id", ""))
                text = item.get("text", "").strip()
                if not text:
                    continue

                # Locate image
                img_path = None
                img_sha256 = None
                avail = ImageAvailabilityStatus.MISSING
                if img_id:
                    for ext in [".jpg", ".png", ".jpeg"]:
                        candidate = img_dir / f"{img_id}{ext}"
                        if candidate.exists():
                            img_path = str(candidate.relative_to(PROJECT_ROOT))
                            avail = ImageAvailabilityStatus.AVAILABLE
                            break

                rec = UnifiedRecord(
                    example_id=f"mmsd_{split_name}_{idx:05d}",
                    source_dataset="mmsd",
                    source_record_id=img_id,
                    source_url_or_reference="https://github.com/ZHCMOONWIND/MMSD3.0",
                    dataset_version="2.0.0",
                    source_license="Academic Research Use Only (Twitter Terms)",
                    asset_license_status="Twitter Terms of Service",
                    post_id=img_id,
                    image_path=img_path,
                    image_sha256=img_sha256,
                    image_availability_status=avail,
                    target_comment=text,
                    caption=None,
                    original_label=item.get("label"),
                    original_label_scheme="0=non_sarcastic, 1=sarcastic",
                    passive_aggression_label=None,  # Preserved as null per label policy
                    passive_aggression_subtype=None,
                    annotation_status=AnnotationStatus.RAW,
                    synthetic_status=SyntheticStatus.REAL,
                    train_validation_test_split=split_name,
                )
                records.append(rec)

    out_p = PROJECT_ROOT / "data" / "manifests" / "mmsd_manifest.jsonl"
    with open(out_p, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r.to_jsonl_dict(), ensure_ascii=False) + "\n")

    logger.info(f"MMSD manifest generated: {len(records)} records -> {out_p}")
    return records


def generate_sarcnet_manifest():
    """Generates manifest for SarcNet dataset."""
    logger.info("Generating SarcNet manifest...")
    raw_dir = PROJECT_ROOT / "data" / "raw" / "sarcnet"
    img_dir = PROJECT_ROOT / "data" / "images" / "sarcnet"
    records = []

    splits = [
        ("train", "SarcNetTrain.csv"),
        ("validation", "SarcNetVal.csv"),
        ("test", "SarcNetTest.csv"),
    ]

    for split_name, fname in splits:
        p = raw_dir / fname
        if not p.exists():
            continue
        with open(p, "r", encoding="gb18030", errors="replace") as f:
            reader = csv.DictReader(f)
            for idx, row in enumerate(reader):
                text = row.get("Text", "").strip()
                if not text:
                    continue

                raw_img = row.get("Imagepath", "").strip()
                img_name = os.path.basename(raw_img) if raw_img else None
                img_path = None
                avail = ImageAvailabilityStatus.MISSING
                if img_name and (img_dir / img_name).exists():
                    img_path = str((img_dir / img_name).relative_to(PROJECT_ROOT))
                    avail = ImageAvailabilityStatus.AVAILABLE

                orig_labels = {
                    "text_label": row.get("Text_label", "").strip(),
                    "image_label": row.get("Image_label", "").strip(),
                    "multi_label": row.get("Multi_label", "").strip(),
                }

                rec = UnifiedRecord(
                    example_id=f"sarcnet_{split_name}_{idx:05d}",
                    source_dataset="sarcnet",
                    source_record_id=img_name or f"rec_{idx}",
                    source_url_or_reference="https://github.com/yuetanbupt/SarcNet",
                    dataset_version="1.0.0",
                    source_license="MIT License",
                    asset_license_status="Academic Research / Fair Use",
                    post_id=img_name,
                    image_path=img_path,
                    image_availability_status=avail,
                    target_comment=text,
                    caption=None,
                    original_label=orig_labels,
                    original_label_scheme="Text_label, Image_label, Multi_label (0=non-sarcastic, 1=sarcastic)",
                    passive_aggression_label=None,
                    passive_aggression_subtype=None,
                    annotation_status=AnnotationStatus.RAW,
                    synthetic_status=SyntheticStatus.REAL,
                    train_validation_test_split=split_name,
                )
                records.append(rec)

    out_p = PROJECT_ROOT / "data" / "manifests" / "sarcnet_manifest.jsonl"
    with open(out_p, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r.to_jsonl_dict(), ensure_ascii=False) + "\n")

    logger.info(f"SarcNet manifest generated: {len(records)} records -> {out_p}")
    return records


def generate_goemotions_manifest(sample_limit: int = 10000):
    """Generates manifest for GoEmotions dataset (saving prioritized PA-adjacent subset + sample)."""
    logger.info("Generating GoEmotions manifest...")
    raw_dir = PROJECT_ROOT / "data" / "raw" / "goemotions"
    emotions_file = raw_dir / "emotions.txt"
    emotion_names = []
    if emotions_file.exists():
        with open(emotions_file, "r", encoding="utf-8") as f:
            emotion_names = [line.strip() for line in f if line.strip()]

    records = []
    splits = [
        ("train", "train.tsv"),
        ("validation", "dev.tsv"),
        ("test", "test.tsv"),
    ]

    for split_name, fname in splits:
        p = raw_dir / fname
        if not p.exists():
            continue
        with open(p, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    text = parts[0].strip()
                    if not text:
                        continue
                    emo_indices = [int(i.strip()) for i in parts[1].split(",") if i.strip().isdigit()]
                    emo_labels = [emotion_names[i] if i < len(emotion_names) else str(i) for i in emo_indices]
                    comm_id = parts[2].strip() if len(parts) > 2 else f"{split_name}_{idx}"

                    rec = UnifiedRecord(
                        example_id=f"goemotions_{split_name}_{idx:05d}",
                        source_dataset="goemotions",
                        source_record_id=comm_id,
                        source_url_or_reference="https://github.com/google-research/google-research/tree/master/goemotions",
                        dataset_version="1.0.0",
                        source_license="Apache-2.0",
                        asset_license_status="Reddit User Terms",
                        conversation_id=comm_id,
                        image_path=None,
                        image_availability_status=ImageAvailabilityStatus.NOT_APPLICABLE,
                        target_comment=text,
                        caption=None,
                        original_label=emo_labels,
                        original_label_scheme="27 fine-grained Reddit emotions + neutral",
                        passive_aggression_label=None,
                        passive_aggression_subtype=None,
                        annotation_status=AnnotationStatus.RAW,
                        synthetic_status=SyntheticStatus.REAL,
                        train_validation_test_split=split_name,
                    )
                    records.append(rec)
                    if sample_limit and len(records) >= sample_limit:
                        break
        if sample_limit and len(records) >= sample_limit:
            break

    out_p = PROJECT_ROOT / "data" / "manifests" / "goemotions_manifest.jsonl"
    with open(out_p, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r.to_jsonl_dict(), ensure_ascii=False) + "\n")

    logger.info(f"GoEmotions manifest generated: {len(records)} records -> {out_p}")
    return records


def generate_civil_comments_manifest(sample_limit: int = 10000):
    """Generates manifest for Civil Comments dataset."""
    logger.info("Generating Civil Comments manifest...")
    val_path = PROJECT_ROOT / "data" / "raw" / "civil_comments" / "civil_comments_val.parquet"
    if not val_path.exists():
        return []

    df = pd.read_parquet(val_path)
    if sample_limit:
        df = df.head(sample_limit)

    records = []
    for idx, row in df.iterrows():
        text = str(row["text"]).strip()
        if not text:
            continue

        orig_labels = {
            "toxicity": float(row["toxicity"]),
            "severe_toxicity": float(row["severe_toxicity"]),
            "obscene": float(row["obscene"]),
            "insult": float(row["insult"]),
            "threat": float(row["threat"]),
            "identity_attack": float(row["identity_attack"]),
        }

        rec = UnifiedRecord(
            example_id=f"civil_comments_val_{idx:05d}",
            source_dataset="civil_comments",
            source_record_id=f"civil_{idx}",
            source_url_or_reference="https://huggingface.co/datasets/google/civil_comments",
            dataset_version="1.0.0",
            source_license="CC0-1.0 Universal",
            asset_license_status="Public Domain",
            image_path=None,
            image_availability_status=ImageAvailabilityStatus.NOT_APPLICABLE,
            target_comment=text,
            caption=None,
            original_label=orig_labels,
            original_label_scheme="Fraction of raters tagging toxicity, severe_toxicity, obscene, insult, threat, identity_attack",
            passive_aggression_label=None,
            passive_aggression_subtype=None,
            annotation_status=AnnotationStatus.RAW,
            synthetic_status=SyntheticStatus.REAL,
            train_validation_test_split="validation",
        )
        records.append(rec)

    out_p = PROJECT_ROOT / "data" / "manifests" / "civil_comments_manifest.jsonl"
    with open(out_p, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r.to_jsonl_dict(), ensure_ascii=False) + "\n")

    logger.info(f"Civil Comments manifest generated: {len(records)} records -> {out_p}")
    return records


def generate_mustard_manifest():
    """Generates manifest for MUStARD dialogue dataset."""
    logger.info("Generating MUStARD manifest...")
    p = PROJECT_ROOT / "data" / "raw" / "mustard" / "sarcasm_data.json"
    if not p.exists():
        return []

    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)

    records = []
    for k, v in data.items():
        utterance = v.get("utterance", "").strip()
        if not utterance:
            continue

        context = v.get("context", [])
        orig_label = {
            "sarcasm": v.get("sarcasm"),
            "speaker": v.get("speaker"),
            "show": v.get("show"),
            "context_speakers": v.get("context_speakers"),
        }

        rec = UnifiedRecord(
            example_id=f"mustard_{k}",
            source_dataset="mustard",
            source_record_id=k,
            source_url_or_reference="https://github.com/soujanyaporia/MUStARD",
            dataset_version="1.0.0",
            source_license="MIT License",
            asset_license_status="TV Sitcom Broadcast Fair Research Use",
            conversation_id=v.get("show"),
            image_path=None,
            image_availability_status=ImageAvailabilityStatus.NOT_APPLICABLE,
            target_comment=utterance,
            parent_comment=context[-1] if context else None,
            conversation_context=context,
            original_label=orig_label,
            original_label_scheme="sarcasm boolean, speaker, show, context_speakers",
            passive_aggression_label=None,
            passive_aggression_subtype=None,
            annotation_status=AnnotationStatus.RAW,
            synthetic_status=SyntheticStatus.REAL,
            train_validation_test_split=None,
        )
        records.append(rec)

    out_p = PROJECT_ROOT / "data" / "manifests" / "mustard_manifest.jsonl"
    with open(out_p, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r.to_jsonl_dict(), ensure_ascii=False) + "\n")

    logger.info(f"MUStARD manifest generated: {len(records)} records -> {out_p}")
    return records


def main():
    logger.info("=== Creating Manifests for All Collected Datasets ===")
    mmsd_recs = generate_mmsd_manifest()
    sarcnet_recs = generate_sarcnet_manifest()
    goemo_recs = generate_goemotions_manifest()
    civil_recs = generate_civil_comments_manifest()
    mustard_recs = generate_mustard_manifest()

    # Create unified interim manifest
    unified_p = PROJECT_ROOT / "data" / "interim" / "unified_raw_manifest.jsonl"
    unified_p.parent.mkdir(parents=True, exist_ok=True)
    with open(unified_p, "w", encoding="utf-8") as f:
        for r in mmsd_recs:
            f.write(json.dumps(r.to_jsonl_dict(), ensure_ascii=False) + "\n")
        for r in sarcnet_recs:
            f.write(json.dumps(r.to_jsonl_dict(), ensure_ascii=False) + "\n")
        for r in goemo_recs:
            f.write(json.dumps(r.to_jsonl_dict(), ensure_ascii=False) + "\n")
        for r in civil_recs:
            f.write(json.dumps(r.to_jsonl_dict(), ensure_ascii=False) + "\n")
        for r in mustard_recs:
            f.write(json.dumps(r.to_jsonl_dict(), ensure_ascii=False) + "\n")

    total_unified = len(mmsd_recs) + len(sarcnet_recs) + len(goemo_recs) + len(civil_recs) + len(mustard_recs)
    logger.info(f"Unified interim manifest created with {total_unified} records at {unified_p}")


if __name__ == "__main__":
    main()
