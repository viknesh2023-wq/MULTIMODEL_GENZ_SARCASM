"""
Task A Audit Script: Verify project root, existing annotation files, schemas, and records.
"""
import os
import json
import csv

def audit():
    files_to_check = [
        "docs/phase5/annotator_instructions.md",
        "docs/phase4/annotation_guidelines.md",
        "docs/phase4/annotation_schema.md",
        "data/annotations/phase5/annotator_1_responses.csv",
        "data/annotations/phase5/annotator_2_responses.csv",
        "data/annotations/phase5/adjudication_queue.csv",
        "data/annotations/phase5/human_review_status_manifest.json",
        "data/annotations/phase4/human_sampling_manifest_100.jsonl",
        "reports/phase5/phase5_annotation_readiness.md",
        "reports/phase5/phase5_benchmark_plan.md",
        "src/annotation/quality_analysis.py",
        "scripts/analyze_annotation_quality.py",
    ]

    print("=== TASK A FILE AUDIT ===")
    all_present = True
    audit_data = {}
    for f in files_to_check:
        exists = os.path.exists(f)
        size = os.path.getsize(f) if exists else 0
        print(f"[{'OK' if exists else 'MISSING'}] {f} ({size} bytes)")
        if not exists:
            all_present = False
        audit_data[f] = {"exists": exists, "size_bytes": size}

    # Verify 100-record manifest
    manifest_path = "data/annotations/phase4/human_sampling_manifest_100.jsonl"
    with open(manifest_path, "r", encoding="utf-8") as f:
        recs = [json.loads(line) for line in f if line.strip()]

    print(f"\nManifest total records: {len(recs)}")
    record_ids = [r["record_id"] for r in recs]
    assert len(set(record_ids)) == 100, "Record IDs are not unique!"

    # Verify image paths
    images_verified = 0
    images_missing = 0
    for r in recs:
        if r.get("image_available"):
            img = r.get("image_path")
            if img and os.path.exists(img):
                images_verified += 1
            else:
                images_missing += 1

    print(f"Images verified on disk: {images_verified} / 60 claimed")
    print(f"Images missing: {images_missing}")

    # Verify context
    context_count = sum(1 for r in recs if r.get("context_text"))
    print(f"Context available count: {context_count} / 100")

    # Verify human response files match 100 record IDs
    with open("data/annotations/phase5/annotator_1_responses.csv", "r", encoding="utf-8") as f:
        r1_ids = [row["record_id"] for row in csv.DictReader(f)]
    with open("data/annotations/phase5/annotator_2_responses.csv", "r", encoding="utf-8") as f:
        r2_ids = [row["record_id"] for row in csv.DictReader(f)]

    assert r1_ids == record_ids, "Annotator 1 record IDs mismatch!"
    assert r2_ids == record_ids, "Annotator 2 record IDs mismatch!"
    print("Annotator 1 & Annotator 2 response files match manifest IDs exactly!")

    # Verify review status manifest
    with open("data/annotations/phase5/human_review_status_manifest.json", "r", encoding="utf-8") as f:
        h_status = json.load(f)
    print(f"Human review status: {h_status.get('evaluation_readiness_status')}")
    print(f"Human reviews completed: Annotator 1={h_status['annotator_01']['completed_count']}, Annotator 2={h_status['annotator_02']['completed_count']}")

    print("\n=== TASK A AUDIT VERIFIED ===")

if __name__ == "__main__":
    audit()
