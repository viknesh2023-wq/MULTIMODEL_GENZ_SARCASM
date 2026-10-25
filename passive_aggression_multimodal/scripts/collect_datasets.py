"""Reproducible dataset collection script for Multimodal Passive Aggression Detection.

Executes and verifies collection of all benchmark datasets:
- MMSD / MMSD2.0 / MMSD3.0
- SarcNet
- GoEmotions
- Civil Comments
- MUStARD
- iSarcasm
Logs all actions to logs/collection_log.jsonl.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_collection.gdrive_downloader import download_file_from_google_drive
from src.utilities.hash_utils import compute_file_sha256
from src.utilities.logger import log_event_jsonl, setup_console_logger

logger = setup_console_logger("dataset_collector")
LOG_FILE = str(PROJECT_ROOT / "logs" / "collection_log.jsonl")


def collect_mmsd():
    """Verify and collect MMSD/MMSD2.0/MMSD3.0 datasets."""
    logger.info("Collecting MMSD datasets...")
    raw_dir = PROJECT_ROOT / "data" / "raw" / "mmsd"
    raw_dir.mkdir(parents=True, exist_ok=True)
    images_dir = PROJECT_ROOT / "data" / "images" / "mmsd"

    # MMSD2.0 benchmark files from local source if present
    source_mmsd2 = Path(r"D:\NLP_PROJECT\06_MMSD2.0\DATA1")
    if source_mmsd2.exists():
        for src_name, dst_name in [
            ("train (1).json", "mmsd_train.json"),
            ("valid (1).json", "mmsd_valid.json"),
            ("test (1).json", "mmsd_test.json"),
        ]:
            src_file = source_mmsd2 / src_name
            dst_file = raw_dir / dst_name
            if src_file.exists() and not dst_file.exists():
                shutil.copy2(src_file, dst_file)
                logger.info(f"Copied {src_name} -> {dst_name}")

        # Link or check images
        src_img = source_mmsd2 / "dataset_image" / "dataset_image"
        if src_img.exists() and not images_dir.exists():
            try:
                # Create directory junction on Windows
                subprocess.run(
                    ["powershell", "-Command", f"New-Item -ItemType Junction -Path '{images_dir}' -Target '{src_img}'"],
                    check=True,
                    capture_output=True,
                )
                logger.info("Created NTFS junction for MMSD images.")
            except Exception as e:
                logger.warning(f"Could not create junction: {e}")

    # MMSD3 open source splits from GitHub
    mmsd3_urls = {
        "mmsd3_train_opensource.json": "https://raw.githubusercontent.com/ZHCMOONWIND/MMSD3.0/main/data/MMSD3/train_data_opensource.json",
        "mmsd3_val_opensource.json": "https://raw.githubusercontent.com/ZHCMOONWIND/MMSD3.0/main/data/MMSD3/val_data_opensource.json",
        "mmsd3_test_opensource.json": "https://raw.githubusercontent.com/ZHCMOONWIND/MMSD3.0/main/data/MMSD3/test_data_opensource.json",
    }
    import urllib.request
    for fname, url in mmsd3_urls.items():
        dst_path = raw_dir / fname
        if not dst_path.exists():
            logger.info(f"Downloading {fname} from {url}...")
            urllib.request.urlretrieve(url, dst_path)

    # Verification
    files_present = list(raw_dir.glob("*.json"))
    img_count = len(list(images_dir.glob("*.*"))) if images_dir.exists() else 0
    status = "DOWNLOADED_AND_VERIFIED" if files_present and img_count > 0 else "PARTIALLY_AVAILABLE"
    details = {
        "json_files": [f.name for f in files_present],
        "image_count": img_count,
        "raw_dir": str(raw_dir),
    }
    log_event_jsonl(LOG_FILE, "COLLECT_DATASET", "mmsd", status, details)
    logger.info(f"MMSD Status: {status} ({len(files_present)} json files, {img_count} images)")
    return status, details


def collect_sarcnet():
    """Verify and collect SarcNet dataset."""
    logger.info("Collecting SarcNet dataset...")
    raw_dir = PROJECT_ROOT / "data" / "raw" / "sarcnet"
    raw_dir.mkdir(parents=True, exist_ok=True)
    images_dir = PROJECT_ROOT / "data" / "images" / "sarcnet"
    images_dir.mkdir(parents=True, exist_ok=True)

    bundle_path = raw_dir / "sarcnet_bundle.zip"
    gdrive_id = "18m3KdDCXgkAlvTbjNhfftvU9LdhyDUTt"
    if not bundle_path.exists():
        logger.info(f"Downloading SarcNet bundle from Google Drive id={gdrive_id}...")
        success, msg = download_file_from_google_drive(gdrive_id, str(bundle_path))
        if not success:
            log_event_jsonl(LOG_FILE, "COLLECT_DATASET", "sarcnet", "REQUIRES_MANUAL_ACTION", {"error": msg})
            return "REQUIRES_MANUAL_ACTION", {"error": msg}

    # Extract if images not present
    if len(list(images_dir.glob("*.*"))) < 3000:
        import zipfile
        logger.info("Extracting SarcNet bundle...")
        with zipfile.ZipFile(bundle_path, "r") as z:
            for member in z.namelist():
                filename = os.path.basename(member)
                if not filename:
                    continue
                if member.endswith(".csv"):
                    with open(raw_dir / filename, "wb") as f_out, z.open(member) as f_in:
                        f_out.write(f_in.read())
                elif member.startswith("SarcNet Image-Text/Image/") and (member.endswith(".jpg") or member.endswith(".png")):
                    with open(images_dir / filename, "wb") as f_out, z.open(member) as f_in:
                        f_out.write(f_in.read())

    csv_files = list(raw_dir.glob("*.csv"))
    img_count = len(list(images_dir.glob("*.*")))
    status = "DOWNLOADED_AND_VERIFIED" if csv_files and img_count >= 3000 else "PARTIALLY_AVAILABLE"
    details = {
        "csv_files": [f.name for f in csv_files],
        "image_count": img_count,
        "bundle_size": bundle_path.stat().st_size if bundle_path.exists() else 0,
    }
    log_event_jsonl(LOG_FILE, "COLLECT_DATASET", "sarcnet", status, details)
    logger.info(f"SarcNet Status: {status} ({len(csv_files)} csv files, {img_count} images)")
    return status, details


def collect_goemotions():
    """Verify and collect GoEmotions dataset."""
    logger.info("Collecting GoEmotions dataset...")
    raw_dir = PROJECT_ROOT / "data" / "raw" / "goemotions"
    raw_dir.mkdir(parents=True, exist_ok=True)

    base_url = "https://raw.githubusercontent.com/google-research/google-research/master/goemotions/data"
    files = ["train.tsv", "dev.tsv", "test.tsv", "emotions.txt"]
    import urllib.request
    for fname in files:
        target = raw_dir / fname
        if not target.exists():
            # Check if available in parent
            parent_src = Path(r"D:\NLP_PROJECT\GoEmotions") / fname
            if parent_src.exists():
                shutil.copy2(parent_src, target)
            else:
                urllib.request.urlretrieve(f"{base_url}/{fname}", target)

    files_present = list(raw_dir.glob("*.tsv")) + list(raw_dir.glob("*.txt"))
    status = "DOWNLOADED_AND_VERIFIED" if len(files_present) >= 4 else "PARTIALLY_AVAILABLE"
    details = {"files": [f.name for f in files_present], "raw_dir": str(raw_dir)}
    log_event_jsonl(LOG_FILE, "COLLECT_DATASET", "goemotions", status, details)
    logger.info(f"GoEmotions Status: {status} ({len(files_present)} files)")
    return status, details


def collect_civil_comments():
    """Verify and collect Civil Comments dataset."""
    logger.info("Collecting Civil Comments dataset...")
    raw_dir = PROJECT_ROOT / "data" / "raw" / "civil_comments"
    raw_dir.mkdir(parents=True, exist_ok=True)
    val_file = raw_dir / "civil_comments_val.parquet"

    if not val_file.exists():
        import urllib.request
        url = "https://huggingface.co/datasets/google/civil_comments/resolve/main/data/validation-00000-of-00001.parquet"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req) as resp, open(val_file, "wb") as f_out:
            shutil.copyfileobj(resp, f_out)

    status = "DOWNLOADED_AND_VERIFIED" if val_file.exists() and val_file.stat().st_size > 1000000 else "PARTIALLY_AVAILABLE"
    details = {
        "files": [val_file.name] if val_file.exists() else [],
        "size_bytes": val_file.stat().st_size if val_file.exists() else 0,
    }
    log_event_jsonl(LOG_FILE, "COLLECT_DATASET", "civil_comments", status, details)
    logger.info(f"Civil Comments Status: {status}")
    return status, details


def collect_mustard():
    """Verify and collect MUStARD dataset."""
    logger.info("Collecting MUStARD dataset...")
    raw_dir = PROJECT_ROOT / "data" / "raw" / "mustard"
    raw_dir.mkdir(parents=True, exist_ok=True)
    sarcasm_json = raw_dir / "sarcasm_data.json"

    if not sarcasm_json.exists():
        import urllib.request
        url = "https://raw.githubusercontent.com/soujanyaporia/MUStARD/master/data/sarcasm_data.json"
        urllib.request.urlretrieve(url, sarcasm_json)

    status = "DOWNLOADED_AND_VERIFIED" if sarcasm_json.exists() and sarcasm_json.stat().st_size > 100000 else "PARTIALLY_AVAILABLE"
    details = {
        "files": [sarcasm_json.name] if sarcasm_json.exists() else [],
        "size_bytes": sarcasm_json.stat().st_size if sarcasm_json.exists() else 0,
    }
    log_event_jsonl(LOG_FILE, "COLLECT_DATASET", "mustard", status, details)
    logger.info(f"MUStARD Status: {status}")
    return status, details


def collect_isarcasm():
    """Verify and collect iSarcasm dataset."""
    logger.info("Collecting iSarcasm dataset...")
    raw_dir = PROJECT_ROOT / "data" / "raw" / "isarcasm"
    raw_dir.mkdir(parents=True, exist_ok=True)

    parent_src = Path(r"D:\NLP_PROJECT\isarcasm")
    if parent_src.exists():
        for f in parent_src.glob("*.csv"):
            dst = raw_dir / f.name
            if not dst.exists():
                shutil.copy2(f, dst)

    csv_files = list(raw_dir.glob("*.csv"))
    # iSarcasm has tweet IDs without text hydrated (due to Twitter redistribution terms)
    status = "PARTIALLY_AVAILABLE" if csv_files else "UNAVAILABLE"
    details = {
        "files": [f.name for f in csv_files],
        "note": "Original Twitter dataset distribution contains tweet IDs and sarcasm labels; text requires Twitter API hydration per Twitter terms.",
    }
    log_event_jsonl(LOG_FILE, "COLLECT_DATASET", "isarcasm", status, details)
    logger.info(f"iSarcasm Status: {status} ({len(csv_files)} csv files)")
    return status, details


def main():
    logger.info("=== Starting Dataset Collection Orchestration ===")
    results = {}
    results["mmsd"] = collect_mmsd()
    results["sarcnet"] = collect_sarcnet()
    results["goemotions"] = collect_goemotions()
    results["civil_comments"] = collect_civil_comments()
    results["mustard"] = collect_mustard()
    results["isarcasm"] = collect_isarcasm()

    logger.info("=== Collection Summary ===")
    for ds, (stat, det) in results.items():
        logger.info(f"  {ds.upper()}: {stat}")


if __name__ == "__main__":
    main()
