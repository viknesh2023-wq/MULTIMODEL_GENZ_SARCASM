"""
Script to sample 100 diverse candidate records from real multimodal data
for the Phase 4 human annotation benchmark.
"""
import json
import os
import random
import csv

def main():
    random.seed(42)
    records_file = "data/processed/preprocessed_multimodal_records.jsonl"
    
    if not os.path.exists(records_file):
        print(f"Error: {records_file} does not exist!")
        return

    by_dataset = {}
    seen_texts = set()

    with open(records_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            src = rec.get("source_dataset", "unknown").lower()
            text = rec.get("normalized_target_comment") or rec.get("target_comment") or ""
            text = text.strip()
            
            # Filter criteria: length at least 15 chars, not empty, not duplicate
            if len(text) < 15:
                continue
            norm_text = " ".join(text.lower().split())
            if norm_text in seen_texts:
                continue
            seen_texts.add(norm_text)
            
            if src not in by_dataset:
                by_dataset[src] = []
            by_dataset[src].append(rec)

    print("Candidates per dataset after basic deduplication:")
    for k, v in by_dataset.items():
        print(f"  {k}: {len(v)} candidates")

    # Sample targets:
    # MMSD: 35
    # SarcNet: 25
    # MUStARD: 15
    # Civil Comments: 25
    targets = {
        "mmsd": 35,
        "sarcnet": 25,
        "mustard": 15,
        "civil_comments": 25
    }

    sampled_records = []
    for ds, target_cnt in targets.items():
        pool = []
        for k in by_dataset:
            if ds in k:
                pool.extend(by_dataset[k])
        
        random.shuffle(pool)
        selected = pool[:target_cnt]
        print(f"Sampled {len(selected)} from {ds} (target: {target_cnt})")
        sampled_records.extend(selected)

    random.shuffle(sampled_records)
    print(f"Total sampled candidate records: {len(sampled_records)}")

    out_jsonl = "data/annotations/phase4/human_sampling_manifest_100.jsonl"
    out_csv = "data/annotations/phase4/annotation_template.csv"
    out_status = "data/annotations/phase4/human_review_status_manifest.json"

    os.makedirs(os.path.dirname(out_jsonl), exist_ok=True)

    csv_rows = []
    manifest_rows = []

    for i, r in enumerate(sampled_records):
        rec_id = r.get("example_id", f"sample_{i+1:03d}")
        src = r.get("source_dataset", "unknown")
        text = r.get("target_comment") or r.get("normalized_target_comment") or ""
        img_path = r.get("image_path") or ""
        ctx = r.get("formatted_conversation_context") or r.get("conversation_context") or r.get("caption") or ""

        # Normalize image availability
        if img_path:
            img_avail = os.path.exists(img_path)
        else:
            img_avail = False

        record_entry = {
            "sample_index": i + 1,
            "record_id": rec_id,
            "source_dataset": src,
            "comment_text": text,
            "image_path": img_path,
            "image_available": img_avail,
            "context_text": ctx,
            "primary_label": None,
            "subtype": None,
            "image_role": None,
            "context_sufficiency": None,
            "confidence": None,
            "evidence_span": None,
            "annotator_id": None,
            "annotation_timestamp": None,
            "adjudication_status": "PENDING_REVIEW"
        }
        manifest_rows.append(record_entry)

        csv_rows.append({
            "sample_index": i + 1,
            "record_id": rec_id,
            "source_dataset": src,
            "comment_text": text,
            "image_available": img_avail,
            "context_text": ctx,
            "primary_label": "",
            "subtype": "",
            "image_role": "",
            "context_sufficiency": "",
            "confidence": "",
            "evidence_span": "",
            "annotator_id": "",
            "notes": ""
        })

    with open(out_jsonl, "w", encoding="utf-8") as f:
        for row in manifest_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {len(manifest_rows)} records to {out_jsonl}")

    fieldnames = [
        "sample_index", "record_id", "source_dataset", "comment_text",
        "image_available", "context_text", "primary_label", "subtype",
        "image_role", "context_sufficiency", "confidence", "evidence_span",
        "annotator_id", "notes"
    ]
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"Wrote {len(csv_rows)} records to {out_csv}")

    status_manifest = {
        "manifest_name": "human_sampling_manifest_100",
        "version": "4.0.0",
        "created_date": "2026-10-21",
        "total_candidate_samples": len(manifest_rows),
        "source_breakdown": {
            "mmsd": sum(1 for r in manifest_rows if "mmsd" in r["source_dataset"].lower()),
            "sarcnet": sum(1 for r in manifest_rows if "sarcnet" in r["source_dataset"].lower()),
            "mustard": sum(1 for r in manifest_rows if "mustard" in r["source_dataset"].lower()),
            "civil_comments": sum(1 for r in manifest_rows if "civil" in r["source_dataset"].lower())
        },
        "multimodal_samples_with_image": sum(1 for r in manifest_rows if r["image_available"]),
        "text_only_samples": sum(1 for r in manifest_rows if not r["image_available"]),
        "human_reviews_completed": 0,
        "inter_annotator_agreement_cohens_kappa": None,
        "adjudicated_instances": 0,
        "evaluation_readiness_status": "BLOCKED_PENDING_HUMAN_REVIEW",
        "status_reason": (
            "Sampling manifest and annotation guidelines v4.0.0 prepared with 100 diverse real social media instances. "
            "Zero genuine human reviews have been conducted yet. In accordance with Phase 4 audit rules, "
            "real-data passive aggression evaluation cannot proceed until independent human annotations are recorded and adjudicated."
        )
    }

    with open(out_status, "w", encoding="utf-8") as f:
        json.dump(status_manifest, f, indent=2)
    print(f"Wrote status manifest to {out_status}")

if __name__ == "__main__":
    main()
