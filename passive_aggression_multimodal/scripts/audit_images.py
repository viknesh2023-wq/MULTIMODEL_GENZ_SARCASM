"""Image auditing script verifying format, readability, dimensions, and missing references."""
from __future__ import annotations

import concurrent.futures
import csv
import json
import os
import sys
from collections import Counter
from pathlib import Path
from PIL import Image

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utilities.hash_utils import verify_image_file
from src.utilities.logger import setup_console_logger

logger = setup_console_logger("image_auditor")


def _check_image(f_path: Path):
    if f_path.stat().st_size == 0:
        return (False, "Zero bytes", None, None, f_path.name)
    is_valid, err, dims, fmt = verify_image_file(str(f_path))
    return (is_valid, err, dims, fmt, f_path.name)


def audit_image_directory(image_dir: Path, max_workers: int = 16) -> dict:
    """Audits image files in image_dir for corruption, dimensions, and formats using thread pool."""
    logger.info(f"Auditing image directory: {image_dir}")
    if not image_dir.exists():
        return {"exists": False, "error": f"Directory not found: {image_dir}"}

    all_files = [f for f in image_dir.iterdir() if f.is_file()]
    total_files = len(all_files)
    logger.info(f"Found {total_files} files in {image_dir.name}. Processing with {max_workers} threads...")

    valid_count = 0
    corrupt_count = 0
    zero_byte_count = 0
    formats = Counter()
    widths = []
    heights = []
    corrupted_files = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        for is_valid, err, dims, fmt, fname in executor.map(_check_image, all_files):
            if is_valid:
                valid_count += 1
                formats[fmt] += 1
                widths.append(dims[0])
                heights.append(dims[1])
            else:
                corrupt_count += 1
                if err == "Zero bytes":
                    zero_byte_count += 1
                corrupted_files.append((fname, err))

    return {
        "directory": str(image_dir.relative_to(PROJECT_ROOT)),
        "total_files": total_files,
        "valid_images": valid_count,
        "corrupt_images": corrupt_count,
        "zero_byte_images": zero_byte_count,
        "format_distribution": dict(formats),
        "dimension_stats": {
            "min_width": min(widths) if widths else 0,
            "max_width": max(widths) if widths else 0,
            "avg_width": float(sum(widths) / len(widths)) if widths else 0.0,
            "min_height": min(heights) if heights else 0,
            "max_height": max(heights) if heights else 0,
            "avg_height": float(sum(heights) / len(heights)) if heights else 0.0,
        },
        "sample_corrupted_files": corrupted_files[:10],
    }


def audit_record_image_references() -> dict:
    """Verifies that image IDs referenced in JSON/CSV datasets actually exist on disk."""
    logger.info("Auditing record image cross-references...")
    mmsd_img_dir = PROJECT_ROOT / "data" / "images" / "mmsd"
    sarcnet_img_dir = PROJECT_ROOT / "data" / "images" / "sarcnet"

    # Pre-index existing files on disk for O(1) lookup
    mmsd_disk_files = set(f.name for f in mmsd_img_dir.iterdir() if f.is_file()) if mmsd_img_dir.exists() else set()
    sarcnet_disk_files = set(f.name for f in sarcnet_img_dir.iterdir() if f.is_file()) if sarcnet_img_dir.exists() else set()

    # MMSD references
    mmsd_raw = PROJECT_ROOT / "data" / "raw" / "mmsd"
    mmsd_referenced = 0
    mmsd_found = 0
    mmsd_missing = 0
    for split in ["mmsd_train.json", "mmsd_valid.json", "mmsd_test.json"]:
        p = mmsd_raw / split
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
                for item in data:
                    img_id = item.get("image_id")
                    if img_id:
                        mmsd_referenced += 1
                        fname1 = f"{img_id}.jpg"
                        fname2 = f"{img_id}.png"
                        if fname1 in mmsd_disk_files or fname2 in mmsd_disk_files:
                            mmsd_found += 1
                        else:
                            mmsd_missing += 1

    # SarcNet references
    sarcnet_raw = PROJECT_ROOT / "data" / "raw" / "sarcnet"
    sarcnet_referenced = 0
    sarcnet_found = 0
    sarcnet_missing = 0
    for split in ["SarcNetTrain.csv", "SarcNetVal.csv", "SarcNetTest.csv"]:
        p = sarcnet_raw / split
        if p.exists():
            with open(p, "r", encoding="gb18030", errors="replace") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    img_path = row.get("Imagepath", "").strip()
                    if img_path:
                        sarcnet_referenced += 1
                        base = os.path.basename(img_path)
                        if base in sarcnet_disk_files:
                            sarcnet_found += 1
                        else:
                            sarcnet_missing += 1

    return {
        "mmsd": {
            "referenced_in_records": mmsd_referenced,
            "images_present_on_disk": mmsd_found,
            "missing_on_disk": mmsd_missing,
            "availability_rate": (mmsd_found / mmsd_referenced) if mmsd_referenced > 0 else 0.0,
        },
        "sarcnet": {
            "referenced_in_records": sarcnet_referenced,
            "images_present_on_disk": sarcnet_found,
            "missing_on_disk": sarcnet_missing,
            "availability_rate": (sarcnet_found / sarcnet_referenced) if sarcnet_referenced > 0 else 0.0,
        },
    }


def main():
    logger.info("=== Running Image Audit ===")
    mmsd_audit = audit_image_directory(PROJECT_ROOT / "data" / "images" / "mmsd")
    sarcnet_audit = audit_image_directory(PROJECT_ROOT / "data" / "images" / "sarcnet")
    cross_refs = audit_record_image_references()

    report = {
        "directories": {
            "mmsd": mmsd_audit,
            "sarcnet": sarcnet_audit,
        },
        "record_cross_references": cross_refs,
    }

    out_file = PROJECT_ROOT / "data" / "manifests" / "image_audit_summary.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info(f"Image audit summary saved to {out_file}")
    logger.info(f"MMSD: {mmsd_audit['valid_images']}/{mmsd_audit['total_files']} valid images (corrupt: {mmsd_audit['corrupt_images']})")
    logger.info(f"SarcNet: {sarcnet_audit['valid_images']}/{sarcnet_audit['total_files']} valid images (corrupt: {sarcnet_audit['corrupt_images']})")
    logger.info(f"MMSD Cross-Ref Availability: {cross_refs['mmsd']['availability_rate']*100:.2f}% ({cross_refs['mmsd']['images_present_on_disk']}/{cross_refs['mmsd']['referenced_in_records']})")
    logger.info(f"SarcNet Cross-Ref Availability: {cross_refs['sarcnet']['availability_rate']*100:.2f}% ({cross_refs['sarcnet']['images_present_on_disk']}/{cross_refs['sarcnet']['referenced_in_records']})")


if __name__ == "__main__":
    main()
