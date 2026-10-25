"""Verification script for all downloaded dataset assets.

Checks:
- File existence, non-zero size, readability
- Format validity (JSON parse, TSV parse, Parquet parse, Zip integrity)
- Computes SHA-256 hashes
- Logs verification details to logs/verification_log.jsonl
"""
from __future__ import annotations

import csv
import json
import os
import sys
import zipfile
from pathlib import Path
import pandas as pd
import pyarrow.parquet as pq

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utilities.hash_utils import compute_file_sha256
from src.utilities.logger import log_event_jsonl, setup_console_logger

logger = setup_console_logger("download_verifier")
LOG_FILE = str(PROJECT_ROOT / "logs" / "verification_log.jsonl")


def verify_file(filepath: Path, expected_format: str) -> dict:
    """Verifies a single file for integrity and format correctness."""
    res = {
        "path": str(filepath.relative_to(PROJECT_ROOT)),
        "size_bytes": filepath.stat().st_size,
        "is_valid": False,
        "error": None,
        "sha256": None,
        "record_count": None,
    }
    if res["size_bytes"] == 0:
        res["error"] = "Empty file (0 bytes)"
        return res

    res["sha256"] = compute_file_sha256(str(filepath))

    try:
        if expected_format == "json":
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
                res["record_count"] = len(data) if isinstance(data, (list, dict)) else 1
            res["is_valid"] = True
        elif expected_format == "tsv":
            with open(filepath, "r", encoding="utf-8") as f:
                reader = csv.reader(f, delimiter="\t")
                count = sum(1 for _ in reader)
                res["record_count"] = count
            res["is_valid"] = True
        elif expected_format == "csv_gb18030":
            # For SarcNet CSVs encoded in Chinese GB18030/GBK
            with open(filepath, "r", encoding="gb18030", errors="replace") as f:
                reader = csv.reader(f)
                header = next(reader)
                count = sum(1 for _ in reader)
                res["record_count"] = count
            res["is_valid"] = True
        elif expected_format == "csv":
            with open(filepath, "r", encoding="utf-8", errors="replace") as f:
                reader = csv.reader(f)
                count = sum(1 for _ in reader)
                res["record_count"] = count
            res["is_valid"] = True
        elif expected_format == "parquet":
            table = pq.read_table(str(filepath))
            res["record_count"] = table.num_rows
            res["is_valid"] = True
        elif expected_format == "zip":
            with zipfile.ZipFile(str(filepath), "r") as z:
                test_res = z.testzip()
                if test_res is None:
                    res["record_count"] = len(z.namelist())
                    res["is_valid"] = True
                else:
                    res["error"] = f"Corrupt zip member: {test_res}"
        elif expected_format == "txt":
            with open(filepath, "r", encoding="utf-8") as f:
                lines = [line.strip() for line in f if line.strip()]
                res["record_count"] = len(lines)
            res["is_valid"] = True
    except Exception as e:
        res["error"] = str(e)
        res["is_valid"] = False

    return res


def run_verification():
    logger.info("=== Starting Dataset Verification ===")
    verified_registry = {}

    targets = [
        # MMSD
        ("mmsd", PROJECT_ROOT / "data" / "raw" / "mmsd" / "mmsd_train.json", "json"),
        ("mmsd", PROJECT_ROOT / "data" / "raw" / "mmsd" / "mmsd_valid.json", "json"),
        ("mmsd", PROJECT_ROOT / "data" / "raw" / "mmsd" / "mmsd_test.json", "json"),
        ("mmsd", PROJECT_ROOT / "data" / "raw" / "mmsd" / "mmsd3_train_opensource.json", "json"),
        ("mmsd", PROJECT_ROOT / "data" / "raw" / "mmsd" / "mmsd3_val_opensource.json", "json"),
        ("mmsd", PROJECT_ROOT / "data" / "raw" / "mmsd" / "mmsd3_test_opensource.json", "json"),
        # SarcNet
        ("sarcnet", PROJECT_ROOT / "data" / "raw" / "sarcnet" / "sarcnet_bundle.zip", "zip"),
        ("sarcnet", PROJECT_ROOT / "data" / "raw" / "sarcnet" / "SarcNetTrain.csv", "csv_gb18030"),
        ("sarcnet", PROJECT_ROOT / "data" / "raw" / "sarcnet" / "SarcNetVal.csv", "csv_gb18030"),
        ("sarcnet", PROJECT_ROOT / "data" / "raw" / "sarcnet" / "SarcNetTest.csv", "csv_gb18030"),
        # GoEmotions
        ("goemotions", PROJECT_ROOT / "data" / "raw" / "goemotions" / "train.tsv", "tsv"),
        ("goemotions", PROJECT_ROOT / "data" / "raw" / "goemotions" / "dev.tsv", "tsv"),
        ("goemotions", PROJECT_ROOT / "data" / "raw" / "goemotions" / "test.tsv", "tsv"),
        ("goemotions", PROJECT_ROOT / "data" / "raw" / "goemotions" / "emotions.txt", "txt"),
        # Civil Comments
        ("civil_comments", PROJECT_ROOT / "data" / "raw" / "civil_comments" / "civil_comments_val.parquet", "parquet"),
        # MUStARD
        ("mustard", PROJECT_ROOT / "data" / "raw" / "mustard" / "sarcasm_data.json", "json"),
        # iSarcasm
        ("isarcasm", PROJECT_ROOT / "data" / "raw" / "isarcasm" / "isarcasm_train.csv", "csv"),
        ("isarcasm", PROJECT_ROOT / "data" / "raw" / "isarcasm" / "isarcasm_test.csv", "csv"),
    ]

    all_passed = True
    for ds_name, path, fmt in targets:
        if not path.exists():
            logger.warning(f"File missing: {path}")
            all_passed = False
            log_event_jsonl(LOG_FILE, "VERIFY_FILE", ds_name, "MISSING", {"path": str(path)})
            continue

        result = verify_file(path, fmt)
        status = "VERIFIED_VALID" if result["is_valid"] else "VERIFICATION_FAILED"
        if not result["is_valid"]:
            all_passed = False
        log_event_jsonl(LOG_FILE, "VERIFY_FILE", ds_name, status, result)
        logger.info(f"[{ds_name.upper()}] {path.name}: {status} (Records: {result['record_count']}, Size: {result['size_bytes']:,} bytes)")
        verified_registry[str(path.name)] = result

    # Save summary manifest
    manifest_path = PROJECT_ROOT / "data" / "manifests" / "verification_summary.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(verified_registry, f, indent=2)

    logger.info(f"Verification summary written to {manifest_path}")
    logger.info(f"All files passed verification: {all_passed}")
    return all_passed


if __name__ == "__main__":
    passed = run_verification()
    sys.exit(0 if passed else 1)
