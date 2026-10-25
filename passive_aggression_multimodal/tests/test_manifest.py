"""Unit tests for dataset manifest generation and integrity."""
import json
from pathlib import Path
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def manifest_dir():
    return PROJECT_ROOT / "data" / "manifests"


def test_manifest_files_exist(manifest_dir):
    expected_manifests = [
        "mmsd_manifest.jsonl",
        "sarcnet_manifest.jsonl",
        "goemotions_manifest.jsonl",
        "civil_comments_manifest.jsonl",
        "mustard_manifest.jsonl",
    ]
    for m in expected_manifests:
        p = manifest_dir / m
        assert p.exists(), f"Manifest {m} does not exist"
        assert p.stat().st_size > 0, f"Manifest {m} is empty"


def test_manifest_line_parsing(manifest_dir):
    p = manifest_dir / "mmsd_manifest.jsonl"
    with open(p, "r", encoding="utf-8") as f:
        first_line = f.readline()
        data = json.loads(first_line)
        assert "example_id" in data
        assert data["source_dataset"] == "mmsd"
        assert "target_comment" in data
        assert data["annotation_status"] == "RAW"
        assert data["synthetic_status"] == "REAL"
        assert data["passive_aggression_label"] is None


def test_unified_raw_manifest_present():
    unified_p = PROJECT_ROOT / "data" / "interim" / "unified_raw_manifest.jsonl"
    assert unified_p.exists()
    assert unified_p.stat().st_size > 1000000  # Multi-megabyte file
