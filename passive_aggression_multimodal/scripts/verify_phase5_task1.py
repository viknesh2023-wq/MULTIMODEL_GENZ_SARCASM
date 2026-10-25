"""
Phase 5 Task 1 Verification Script:
Audit the 100-record sampling manifest, checking IDs, provenance, duplicates,
actual physical image decoding, and context availability.
"""
import json
import os
import sys
from collections import Counter
from PIL import Image

def audit_manifest():
    manifest_path = "data/annotations/phase4/human_sampling_manifest_100.jsonl"
    template_path = "data/annotations/phase4/annotation_template.csv"
    status_path = "data/annotations/phase4/human_review_status_manifest.json"

    print("=== AUDITING TASK 1 MANIFESTS ===")
    assert os.path.exists(manifest_path), f"Missing {manifest_path}"
    assert os.path.exists(template_path), f"Missing {template_path}"
    assert os.path.exists(status_path), f"Missing {status_path}"

    with open(manifest_path, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    print(f"Total records in sampling manifest: {len(records)}")

    # 1. Unique Record IDs
    record_ids = [r["record_id"] for r in records]
    unique_ids = set(record_ids)
    print(f"Unique record IDs: {len(unique_ids)} / {len(record_ids)}")
    assert len(unique_ids) == len(record_ids), "Duplicate record_id found!"

    # 2. Source Dataset Breakdown
    sources = Counter(r["source_dataset"].lower() for r in records)
    print("Source dataset breakdown:")
    for src, count in sources.items():
        print(f"  - {src}: {count}")

    # 3. Comment Duplication & Near-Duplication Check
    comments = [r["comment_text"] for r in records]
    norm_comments = [" ".join(c.lower().split()) for c in comments]
    unique_norm_comments = set(norm_comments)
    print(f"Unique normalized comments: {len(unique_norm_comments)} / {len(comments)}")
    assert len(unique_norm_comments) == len(comments), "Duplicate normalized comments detected!"

    # Check pair-wise word Jaccard similarity to catch near-duplicates
    high_sim_pairs = []
    for i in range(len(norm_comments)):
        set_i = set(norm_comments[i].split())
        for j in range(i + 1, len(norm_comments)):
            set_j = set(norm_comments[j].split())
            if not set_i or not set_j:
                continue
            jaccard = len(set_i.intersection(set_j)) / len(set_i.union(set_j))
            if jaccard > 0.80:
                high_sim_pairs.append((i, j, jaccard, comments[i], comments[j]))

    print(f"Near-duplicate pairs (Jaccard > 0.80): {len(high_sim_pairs)}")
    for p in high_sim_pairs:
        print(f"  Warning near-duplicate: [{p[0]} vs {p[1]} Jaccard={p[2]:.2f}] '{p[3]}' <-> '{p[4]}'")

    # 4. Modality Availability & Physical Image Verification
    has_image_count = 0
    physically_valid_images = 0
    unreadable_images = 0
    has_context_count = 0

    for r in records:
        img_path = r.get("image_path")
        if img_path and os.path.exists(img_path):
            has_image_count += 1
            try:
                with Image.open(img_path) as im:
                    im.verify()
                physically_valid_images += 1
            except Exception as e:
                unreadable_images += 1
                print(f"  Unreadable image for record {r['record_id']}: {img_path} ({e})")
        
        ctx = r.get("context_text")
        if ctx and len(ctx.strip()) > 0:
            has_context_count += 1

    print(f"Records claiming image available: {sum(1 for r in records if r.get('image_available'))}")
    print(f"Physically present images on disk: {has_image_count}")
    print(f"Successfully decoded images: {physically_valid_images}")
    print(f"Unreadable/corrupt images: {unreadable_images}")
    print(f"Records with conversation context: {has_context_count}")

    # 5. Check Human Status Manifest
    with open(status_path, "r", encoding="utf-8") as f:
        status_data = json.load(f)
    print(f"Status manifest state: {status_data.get('evaluation_readiness_status')}")
    print(f"Human reviews completed: {status_data.get('human_reviews_completed')}")

    results = {
        "total_records": len(records),
        "unique_record_ids": len(unique_ids),
        "source_breakdown": dict(sources),
        "exact_duplicates": len(comments) - len(unique_norm_comments),
        "near_duplicates_above_80": len(high_sim_pairs),
        "images_present_and_decoded": physically_valid_images,
        "images_unreadable": unreadable_images,
        "context_available_count": has_context_count,
        "status": "VALIDATED"
    }
    os.makedirs("artifacts/phase5", exist_ok=True)
    with open("artifacts/phase5/manifest_audit_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print("=== TASK 1 AUDIT COMPLETE ===")

if __name__ == "__main__":
    audit_manifest()
