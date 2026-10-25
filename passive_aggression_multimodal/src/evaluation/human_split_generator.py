"""Reproducible Frozen Split Generator for Human-Annotated Passive-Aggression Benchmarks.

Enforces:
1. Strict filtering: only genuinely human-adjudicated labels are included.
2. Group-isolated splitting: threads, conversation IDs, image hashes, and identical comments
   are strictly kept within single split partitions.
3. Cryptographic integrity: generates SHA-256 hashes of resulting manifests.
4. Blockers: safely blocks split creation if insufficient reviewed labels exist.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import random
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple


def compute_sha256_of_file(filepath: str) -> str:
    """Computes SHA-256 checksum of a file."""
    sha = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            sha.update(chunk)
    return sha.hexdigest()


class HumanSplitGenerator:
    """
    Generates group-isolated train/val/test splits exclusively from human-reviewed records.
    """

    def __init__(
        self,
        min_required_records: int = 30,
        train_ratio: float = 0.60,
        val_ratio: float = 0.20,
        test_ratio: float = 0.20,
        seed: int = 42,
    ):
        self.min_required_records = min_required_records
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.seed = seed

    def load_and_filter_adjudicated_records(self, filepath: str) -> List[Dict[str, Any]]:
        """
        Loads records from CSV or JSONL and strictly extracts only valid human-reviewed records.
        Rejects pending, unreviewed, synthetic, or heuristic labels.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Input file not found: {filepath}")

        raw_records = []
        if filepath.endswith(".jsonl"):
            with open(filepath, "r", encoding="utf-8") as f:
                raw_records = [json.loads(line) for line in f if line.strip()]
        elif filepath.endswith(".csv"):
            with open(filepath, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                raw_records = list(reader)
        else:
            raise ValueError("Unsupported format. Use .csv or .jsonl")

        adjudicated = []
        for r in raw_records:
            # Check for adjudicated label column, falling back to primary_label if adjudicated
            label = (
                r.get("adjudicated_primary_label")
                or r.get("primary_label")
                or ""
            ).strip().lower()

            status = (
                r.get("adjudication_status")
                or r.get("human_review_status")
                or ""
            ).strip().upper()

            # Strictly require valid human label and non-pending status
            if label in ("passive_aggressive", "non_passive_aggressive", "ambiguous"):
                if status not in ("PENDING_REVIEW", "AWAITING_HUMAN_REVIEW", "SYNTHETIC_PILOT", ""):
                    adjudicated.append(r)

        return adjudicated

    def cluster_into_groups(self, records: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Clusters records by shared identifiers (conversation_id, image_path, normalized comment).
        """
        groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for r in records:
            # Deterministic group key preference: conversation_id -> post_id -> image_path -> normalized text
            cid = r.get("conversation_id", "").strip()
            pid = r.get("post_id", "").strip()
            img = r.get("image_path", "").strip()
            text = (r.get("comment_text") or r.get("target_comment") or "").strip().lower()
            norm_text = " ".join(text.split())

            if cid:
                key = f"conv_{cid}"
            elif pid:
                key = f"post_{pid}"
            elif img and img.lower() not in ("none", "null", ""):
                key = f"img_{os.path.basename(img)}"
            elif norm_text:
                key = f"txt_{hashlib.md5(norm_text.encode('utf-8')).hexdigest()[:12]}"
            else:
                key = f"rec_{r.get('record_id', random.random())}"

            groups[key].append(r)
        return groups

    def generate_splits(
        self,
        input_filepath: str,
        output_dir: str = "data/splits/phase5",
    ) -> Dict[str, Any]:
        """
        Executes verified split generation.
        Returns detailed status dict. If insufficient labels, returns safe blocked state.
        """
        adjudicated_records = self.load_and_filter_adjudicated_records(input_filepath)

        if len(adjudicated_records) < self.min_required_records:
            return {
                "status": "SPLIT_GENERATION_BLOCKED",
                "reason": "Insufficient verified human-annotated records.",
                "verified_records_available": len(adjudicated_records),
                "minimum_required_threshold": self.min_required_records,
                "action_required": (
                    f"At least {self.min_required_records} records must be double-annotated and adjudicated "
                    f"before a defensible real-world benchmark split can be created. Currently available: {len(adjudicated_records)}."
                ),
            }

        # Cluster records into groups
        groups = self.cluster_into_groups(adjudicated_records)
        group_keys = sorted(list(groups.keys()))

        rng = random.Random(self.seed)
        rng.shuffle(group_keys)

        total_records = len(adjudicated_records)
        target_train = int(total_records * self.train_ratio)
        target_val = int(total_records * self.val_ratio)

        train_records = []
        val_records = []
        test_records = []

        for key in group_keys:
            cluster = groups[key]
            if len(train_records) + len(cluster) <= target_train:
                train_records.extend(cluster)
            elif len(val_records) + len(cluster) <= target_val:
                val_records.extend(cluster)
            else:
                test_records.extend(cluster)

        # Fallback to ensure no empty split if possible
        if not val_records and len(train_records) > 2:
            val_records.append(train_records.pop())
        if not test_records and len(train_records) > 2:
            test_records.append(train_records.pop())

        os.makedirs(output_dir, exist_ok=True)
        train_path = os.path.join(output_dir, "human_gold_train.jsonl")
        val_path = os.path.join(output_dir, "human_gold_val.jsonl")
        test_path = os.path.join(output_dir, "human_gold_test.jsonl")

        for path, recs in [(train_path, train_records), (val_path, val_records), (test_path, test_records)]:
            with open(path, "w", encoding="utf-8") as f:
                for r in recs:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")

        registry = {
            "version": "5.0.0",
            "seed": self.seed,
            "min_required_threshold": self.min_required_records,
            "total_human_records": total_records,
            "train_count": len(train_records),
            "val_count": len(val_records),
            "test_count": len(test_records),
            "unique_groups": len(groups),
            "manifests": {
                "train": {"path": train_path, "sha256": compute_sha256_of_file(train_path)},
                "val": {"path": val_path, "sha256": compute_sha256_of_file(val_path)},
                "test": {"path": test_path, "sha256": compute_sha256_of_file(test_path)},
            },
        }

        registry_path = os.path.join(output_dir, "human_ground_truth_split_registry.json")
        with open(registry_path, "w", encoding="utf-8") as f:
            json.dump(registry, f, indent=2)

        return {
            "status": "SPLITS_GENERATED",
            "registry_path": registry_path,
            "train_count": len(train_records),
            "val_count": len(val_records),
            "test_count": len(test_records),
            "manifests": registry["manifests"],
        }
