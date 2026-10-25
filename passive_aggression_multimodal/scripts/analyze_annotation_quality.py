"""
Executable script for Phase 5 annotation quality analysis and agreement audit.
Evaluates Annotator 1 and Annotator 2 response CSVs, computes Cohen's kappa,
generates the adjudication reconciliation file, and updates status manifests.
"""
import argparse
import json
import os
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.annotation.quality_analysis import (
    compute_inter_annotator_agreement,
    generate_adjudication_file,
    load_annotation_file,
)


def main():
    parser = argparse.ArgumentParser(description="Phase 5 Annotation Quality & Agreement Analyzer")
    parser.add_argument(
        "--file1",
        default="data/annotations/phase5/annotator_1_responses.csv",
        help="Path to Annotator 1 response CSV",
    )
    parser.add_argument(
        "--file2",
        default="data/annotations/phase5/annotator_2_responses.csv",
        help="Path to Annotator 2 response CSV",
    )
    parser.add_argument(
        "--adjudication_out",
        default="data/annotations/phase5/adjudication_queue.csv",
        help="Path to export adjudication queue",
    )
    parser.add_argument(
        "--status_manifest",
        default="data/annotations/phase5/human_review_status_manifest.json",
        help="Path to human review status manifest",
    )
    args = parser.parse_args()

    print("=" * 65)
    print("PHASE 5 -- ANNOTATION QUALITY & INTER-ANNOTATOR AGREEMENT AUDIT")
    print("=" * 65)

    if not os.path.exists(args.file1) or not os.path.exists(args.file2):
        print(f"Error: Annotation response files do not exist: {args.file1}, {args.file2}")
        sys.exit(1)

    records1 = load_annotation_file(args.file1)
    records2 = load_annotation_file(args.file2)

    print(f"Loaded Annotator 1 records: {len(records1)}")
    print(f"Loaded Annotator 2 records: {len(records2)}")

    results = compute_inter_annotator_agreement(records1, records2, target_kappa=0.70)

    print("\n--- AUDIT SUMMARY ---")
    print(f"Review Status:               {results['status']}")
    print(f"Annotator 1 Completed:      {results['annotator_1_completed']} / {results['total_samples']}")
    print(f"Annotator 2 Completed:      {results['annotator_2_completed']} / {results['total_samples']}")
    print(f"Both Completed (Evaluated):  {results['both_completed_count']}")
    print(f"Unanimous Agreement Count:  {results['unanimous_count']}")
    print(f"Disputed Count:             {results['disputed_count']}")
    print(f"Invalid Rows:               {results['invalid_count']}")

    if results["raw_agreement_percent"] is not None:
        print(f"Raw Agreement:              {results['raw_agreement_percent']:.2f}%")
    else:
        print("Raw Agreement:              PENDING (Awaiting human inputs)")

    if results["cohens_kappa"] is not None:
        print(f"Cohen's Kappa (kappa):      {results['cohens_kappa']:.4f}")
        print(f"Target Threshold (kappa >= 0.7): {'MET' if results['target_achieved'] else 'NOT MET'}")
    else:
        print("Cohen's Kappa (kappa):      PENDING (Target: >= 0.70)")

    if results.get("warnings"):
        print("\nWarnings:")
        for w in results["warnings"]:
            print(f"  [!] {w}")

    # Generate or update adjudication queue
    adj_count = generate_adjudication_file(records1, records2, args.adjudication_out)
    print(f"\nWrote adjudication queue ({adj_count} rows) to: {args.adjudication_out}")

    # Save artifact report
    os.makedirs("artifacts/phase5", exist_ok=True)
    report_path = "artifacts/phase5/agreement_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Saved detailed agreement audit to: {report_path}")

    # Update human review status manifest if it exists
    if os.path.exists(args.status_manifest):
        with open(args.status_manifest, "r", encoding="utf-8") as f:
            status_data = json.load(f)

        status_data["annotator_01"]["completed_count"] = results["annotator_1_completed"]
        status_data["annotator_01"]["status"] = "COMPLETED" if results["annotator_1_completed"] == results["total_samples"] else "IN_PROGRESS" if results["annotator_1_completed"] > 0 else "AWAITING_INPUT"
        status_data["annotator_02"]["completed_count"] = results["annotator_2_completed"]
        status_data["annotator_02"]["status"] = "COMPLETED" if results["annotator_2_completed"] == results["total_samples"] else "IN_PROGRESS" if results["annotator_2_completed"] > 0 else "AWAITING_INPUT"

        status_data["inter_annotator_agreement"]["cohens_kappa"] = results["cohens_kappa"]
        status_data["inter_annotator_agreement"]["raw_agreement_percent"] = results["raw_agreement_percent"]
        status_data["inter_annotator_agreement"]["status"] = results["status"]

        status_data["adjudication"]["unanimous_count"] = results["unanimous_count"]
        status_data["adjudication"]["disputed_count"] = results["disputed_count"]

        if results["status"] == "COMPUTED" and results["target_achieved"]:
            status_data["evaluation_readiness_status"] = "READY_FOR_ADJUDICATION"
        else:
            status_data["evaluation_readiness_status"] = "BLOCKED_PENDING_HUMAN_ANNOTATION"

        with open(args.status_manifest, "w", encoding="utf-8") as f:
            json.dump(status_data, f, indent=2)
        print(f"Updated status manifest: {args.status_manifest}")

    print("=" * 65)


if __name__ == "__main__":
    main()
