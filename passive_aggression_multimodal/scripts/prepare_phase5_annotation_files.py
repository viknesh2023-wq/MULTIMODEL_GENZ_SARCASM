"""
Script to prepare Phase 5 dual independent annotation files and templates.
"""
import csv
import json
import os

def prepare_files():
    source_manifest = "data/annotations/phase4/human_sampling_manifest_100.jsonl"
    out_dir = "data/annotations/phase5"
    os.makedirs(out_dir, exist_ok=True)

    with open(source_manifest, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    headers = [
        "sample_index",
        "record_id",
        "source_dataset",
        "comment_text",
        "image_available",
        "image_path",
        "context_text",
        "primary_label",
        "subtype",
        "image_role",
        "context_sufficiency",
        "confidence",
        "evidence_span",
        "annotator_id",
        "annotation_timestamp",
        "adjudication_status",
        "notes"
    ]

    # Generate templates for Annotator 1 and Annotator 2
    for annotator_id, num in [("annotator_01", 1), ("annotator_02", 2)]:
        template_file = os.path.join(out_dir, f"annotator_{num}_template.csv")
        response_file = os.path.join(out_dir, f"annotator_{num}_responses.csv")

        rows = []
        for r in records:
            row = {
                "sample_index": r["sample_index"],
                "record_id": r["record_id"],
                "source_dataset": r["source_dataset"],
                "comment_text": r["comment_text"],
                "image_available": r["image_available"],
                "image_path": r.get("image_path", ""),
                "context_text": r.get("context_text", ""),
                "primary_label": "",
                "subtype": "",
                "image_role": "",
                "context_sufficiency": "",
                "confidence": "",
                "evidence_span": "",
                "annotator_id": annotator_id,
                "annotation_timestamp": "",
                "adjudication_status": "PENDING_REVIEW",
                "notes": ""
            }
            rows.append(row)

        with open(template_file, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            writer.writerows(rows)
        print(f"Created {template_file}")

        with open(response_file, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            writer.writerows(rows)
        print(f"Created {response_file}")

    # Generate Adjudication Template
    adj_headers = [
        "sample_index",
        "record_id",
        "source_dataset",
        "comment_text",
        "image_available",
        "image_path",
        "context_text",
        "annotator_01_primary_label",
        "annotator_01_subtype",
        "annotator_01_confidence",
        "annotator_02_primary_label",
        "annotator_02_subtype",
        "annotator_02_confidence",
        "agreement_type",
        "adjudicated_primary_label",
        "adjudicated_subtype",
        "adjudicated_image_role",
        "adjudicated_context_sufficiency",
        "adjudicator_id",
        "adjudication_timestamp",
        "adjudication_status",
        "adjudication_notes"
    ]
    adj_rows = []
    for r in records:
        adj_row = {
            "sample_index": r["sample_index"],
            "record_id": r["record_id"],
            "source_dataset": r["source_dataset"],
            "comment_text": r["comment_text"],
            "image_available": r["image_available"],
            "image_path": r.get("image_path", ""),
            "context_text": r.get("context_text", ""),
            "annotator_01_primary_label": "",
            "annotator_01_subtype": "",
            "annotator_01_confidence": "",
            "annotator_02_primary_label": "",
            "annotator_02_subtype": "",
            "annotator_02_confidence": "",
            "agreement_type": "PENDING_DUAL_ANNOTATION",
            "adjudicated_primary_label": "",
            "adjudicated_subtype": "",
            "adjudicated_image_role": "",
            "adjudicated_context_sufficiency": "",
            "adjudicator_id": "",
            "adjudication_timestamp": "",
            "adjudication_status": "AWAITING_HUMAN_REVIEW",
            "adjudication_notes": ""
        }
        adj_rows.append(adj_row)

    adj_file = os.path.join(out_dir, "adjudication_template.csv")
    with open(adj_file, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=adj_headers)
        writer.writeheader()
        writer.writerows(adj_rows)
    print(f"Created {adj_file}")

    # Generate Phase 5 Human Review Status Manifest
    status_manifest = {
        "manifest_name": "phase5_human_review_status_manifest",
        "version": "5.0.0",
        "created_date": "2026-10-21",
        "pilot_sample_size": len(records),
        "source_breakdown": {
            "mmsd": sum(1 for r in records if "mmsd" in r["source_dataset"].lower()),
            "sarcnet": sum(1 for r in records if "sarcnet" in r["source_dataset"].lower()),
            "mustard": sum(1 for r in records if "mustard" in r["source_dataset"].lower()),
            "civil_comments": sum(1 for r in records if "civil" in r["source_dataset"].lower())
        },
        "multimodal_samples_with_image": sum(1 for r in records if r["image_available"]),
        "text_only_samples": sum(1 for r in records if not r["image_available"]),
        "annotator_01": {
            "assigned_file": "data/annotations/phase5/annotator_1_responses.csv",
            "completed_count": 0,
            "status": "AWAITING_INPUT"
        },
        "annotator_02": {
            "assigned_file": "data/annotations/phase5/annotator_2_responses.csv",
            "completed_count": 0,
            "status": "AWAITING_INPUT"
        },
        "inter_annotator_agreement": {
            "cohens_kappa": None,
            "raw_agreement_percent": None,
            "target_kappa_threshold": 0.70,
            "status": "PENDING_DUAL_ANNOTATION"
        },
        "adjudication": {
            "completed_count": 0,
            "unanimous_count": 0,
            "disputed_count": 0,
            "status": "AWAITING_COMPLETED_REVIEWS"
        },
        "evaluation_readiness_status": "BLOCKED_PENDING_HUMAN_ANNOTATION",
        "status_reason": (
            "Dual independent annotation templates generated for Annotator 1 and Annotator 2. "
            "Zero genuine human reviews have been logged yet. In accordance with Phase 4/5 scientific rules, "
            "real-data passive aggression evaluation cannot proceed until independent human annotations are recorded and adjudicated."
        )
    }

    status_file = os.path.join(out_dir, "human_review_status_manifest.json")
    with open(status_file, "w", encoding="utf-8") as f:
        json.dump(status_manifest, f, indent=2)
    print(f"Created {status_file}")

if __name__ == "__main__":
    prepare_files()
