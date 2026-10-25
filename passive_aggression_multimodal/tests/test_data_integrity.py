"""Data integrity and raw-preservation tests."""
import json
from pathlib import Path
from PIL import Image
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_synthetic_data_integrity():
    syn_path = PROJECT_ROOT / "data" / "synthetic" / "synthetic_pilot_350.jsonl"
    assert syn_path.exists()
    seen_ids = set()
    with open(syn_path, "r", encoding="utf-8") as f:
        count = 0
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            count += 1
            ex_id = rec["example_id"]
            assert ex_id not in seen_ids, f"Duplicate example_id {ex_id}"
            seen_ids.add(ex_id)
            assert rec["synthetic_status"] == "SYNTHETIC_UNVERIFIED"
            assert rec["human_review_status"] == "AWAITING_HUMAN_REVIEW"
            assert rec["passive_aggression_label"] in ["passive_aggressive", "non_passive_aggressive", "ambiguous"]
        assert count == 350


def test_mmsd_image_decoding():
    img_dir = PROJECT_ROOT / "data" / "images" / "mmsd"
    assert img_dir.exists()
    images = list(img_dir.glob("*.jpg"))[:20]
    assert len(images) > 0
    for img_p in images:
        with Image.open(img_p) as img:
            assert img.size[0] > 0
            assert img.size[1] > 0
            assert img.format in ["JPEG", "PNG", "WEBP"]


def test_sarcnet_image_decoding():
    img_dir = PROJECT_ROOT / "data" / "images" / "sarcnet"
    assert img_dir.exists()
    images = list(img_dir.glob("*.jpg"))[:20]
    assert len(images) > 0
    for img_p in images:
        with Image.open(img_p) as img:
            assert img.size[0] > 0
            assert img.size[1] > 0


def test_raw_data_preservation():
    """Verify that original labels in raw manifest are not replaced by PA labels."""
    mmsd_manifest = PROJECT_ROOT / "data" / "manifests" / "mmsd_manifest.jsonl"
    with open(mmsd_manifest, "r", encoding="utf-8") as f:
        for _ in range(50):
            line = f.readline()
            rec = json.loads(line)
            assert rec["passive_aggression_label"] is None
            assert rec["original_label"] in [0, 1]
            assert rec["original_label_scheme"] is not None
