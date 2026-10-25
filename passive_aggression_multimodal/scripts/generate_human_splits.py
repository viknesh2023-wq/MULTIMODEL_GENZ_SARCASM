"""
Executable CLI script for Phase 5 Frozen Human Split Generation.
Safely checks for verified human-annotated labels, generates group-isolated splits,
and freezes SHA-256 manifest hashes.
"""
import argparse
import json
import os
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.evaluation.human_split_generator import HumanSplitGenerator


def main():
    parser = argparse.ArgumentParser(description="Phase 5 Frozen Human Split Generator")
    parser.add_argument(
        "--input_file",
        default="data/annotations/phase5/adjudication_queue.csv",
        help="Path to adjudicated annotations CSV/JSONL",
    )
    parser.add_argument(
        "--output_dir",
        default="data/splits/phase5",
        help="Output directory for frozen split manifests",
    )
    parser.add_argument(
        "--min_records",
        type=int,
        default=30,
        help="Minimum verified records required before generating splits",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for grouping")
    args = parser.parse_args()

    print("=" * 65)
    print("PHASE 5 -- FROZEN HUMAN GROUND-TRUTH SPLIT GENERATOR")
    print("=" * 65)
    print(f"Input Adjudication File:  {args.input_file}")
    print(f"Target Output Directory:  {args.output_dir}")
    print(f"Minimum Required Threshold: {args.min_records} records")

    generator = HumanSplitGenerator(
        min_required_records=args.min_records,
        seed=args.seed,
    )

    if not os.path.exists(args.input_file):
        print(f"\n[!] Input file does not exist: {args.input_file}")
        print("Status: BLOCKED_PENDING_HUMAN_ANNOTATION")
        sys.exit(1)

    result = generator.generate_splits(args.input_file, args.output_dir)

    print("\n--- SPLIT GENERATION RESULT ---")
    print(f"Status: {result['status']}")

    if result["status"] == "SPLIT_GENERATION_BLOCKED":
        print(f"Reason:                     {result['reason']}")
        print(f"Verified Records Available: {result['verified_records_available']}")
        print(f"Minimum Threshold Required: {result['minimum_required_threshold']}")
        print(f"\nAction Required:")
        print(f"  {result['action_required']}")
    else:
        print(f"Train Records:  {result['train_count']}")
        print(f"Val Records:    {result['val_count']}")
        print(f"Test Records:   {result['test_count']}")
        print(f"Registry Path:  {result['registry_path']}")
        print("\nManifest SHA-256 Hashes:")
        for split, info in result["manifests"].items():
            print(f"  - {split}: {info['path']} ({info['sha256'][:16]}...)")

    print("=" * 65)


if __name__ == "__main__":
    main()
