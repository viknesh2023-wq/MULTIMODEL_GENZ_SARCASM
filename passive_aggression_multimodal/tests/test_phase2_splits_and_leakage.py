"""Tests for Phase 2 grouped splits, duplicate detection, and leakage prevention."""
from __future__ import annotations

import json
from pathlib import Path
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_split_files_exist_and_registry_valid():
    splits_dir = PROJECT_ROOT / "data" / "splits"
    train_file = splits_dir / "grouped_train.jsonl"
    val_file = splits_dir / "grouped_validation.jsonl"
    test_file = splits_dir / "grouped_test.jsonl"
    reg_file = splits_dir / "split_registry.json"

    assert train_file.exists(), "grouped_train.jsonl must exist"
    assert val_file.exists(), "grouped_validation.jsonl must exist"
    assert test_file.exists(), "grouped_test.jsonl must exist"
    assert reg_file.exists(), "split_registry.json must exist"

    with open(reg_file, "r", encoding="utf-8") as f:
        registry = json.load(f)

    assert registry["random_seed"] == 42
    assert registry["train_count"] == 22376
    assert registry["validation_count"] == 2797
    assert registry["test_count"] == 2797
    assert registry["grouped_split_leakage"] == 0


def test_zero_leakage_across_splits():
    splits_dir = PROJECT_ROOT / "data" / "splits"

    split_data = {}
    for name in ["train", "validation", "test"]:
        path = splits_dir / f"grouped_{name}.jsonl"
        records = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
        split_data[name] = records

    # 1. Zero Record ID overlap
    train_ids = {r["example_id"] for r in split_data["train"]}
    val_ids = {r["example_id"] for r in split_data["validation"]}
    test_ids = {r["example_id"] for r in split_data["test"]}

    assert len(train_ids.intersection(val_ids)) == 0, "Train-Val ID leak detected"
    assert len(train_ids.intersection(test_ids)) == 0, "Train-Test ID leak detected"
    assert len(val_ids.intersection(test_ids)) == 0, "Val-Test ID leak detected"

    # 2. Zero Post ID overlap
    train_posts = {r["post_id"] for r in split_data["train"] if r.get("post_id")}
    val_posts = {r["post_id"] for r in split_data["validation"] if r.get("post_id")}
    test_posts = {r["post_id"] for r in split_data["test"] if r.get("post_id")}

    assert len(train_posts.intersection(val_posts)) == 0, "Post ID leak between Train and Val"
    assert len(train_posts.intersection(test_posts)) == 0, "Post ID leak between Train and Test"
    assert len(val_posts.intersection(test_posts)) == 0, "Post ID leak between Val and Test"

    # 3. Zero Exact Text overlap
    train_texts = {r["target_comment"].strip().lower() for r in split_data["train"]}
    val_texts = {r["target_comment"].strip().lower() for r in split_data["validation"]}
    test_texts = {r["target_comment"].strip().lower() for r in split_data["test"]}

    assert len(train_texts.intersection(val_texts)) == 0, "Exact text leak between Train and Val"
    assert len(train_texts.intersection(test_texts)) == 0, "Exact text leak between Train and Test"
    assert len(val_texts.intersection(test_texts)) == 0, "Exact text leak between Val and Test"
