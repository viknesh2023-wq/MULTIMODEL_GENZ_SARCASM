#!/usr/bin/env python3
"""
scripts/analyze_phase6a_agreement.py

Phase 6A Inter-Annotator Agreement Engine.
Analyzes independent judgments from Annotator 1 and Annotator 2 on the Phase 6A pilot.

Calculates:
  - Completion status per annotator.
  - Raw percent agreement.
  - Cohen's Kappa (chance-corrected inter-rater reliability).
  - Subtype concordance.
  - Confusion matrix and discrepancy log.
  - Adjudication queue for disagreements.

If reviews are incomplete, reports status as PENDING without fabricating results.
"""

import argparse
import csv
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Paths
PHASE6A_DIR = Path("data/annotations/phase6a")
ANN1_FILE = PHASE6A_DIR / "annotator_1_responses.csv"
ANN2_FILE = PHASE6A_DIR / "annotator_2_responses.csv"
STATUS_FILE = PHASE6A_DIR / "annotation_status_manifest.json"
ADJUDICATION_QUEUE_FILE = PHASE6A_DIR / "adjudication_queue.csv"

VALID_PRIMARY_LABELS = {"passive_aggressive", "non_passive_aggressive", "ambiguous"}


def load_csv(path: Path) -> List[Dict[str, Any]]:
    with open(path, mode="r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def check_annotator_completion(rows: List[Dict[str, Any]]) -> Tuple[int, int, List[Dict[str, Any]]]:
    """Returns (total_records, completed_records, completed_rows)."""
    total = len(rows)
    completed = []
    for r in rows:
        label = (r.get("primary_label") or "").strip().lower()
        if label in VALID_PRIMARY_LABELS:
            completed.append(r)
    return total, len(completed), completed


def compute_cohens_kappa(labels1: List[str], labels2: List[str], categories: List[str]) -> Tuple[float, float, Dict[Tuple[str, str], int]]:
    """
    Computes observed agreement (P_o), expected agreement (P_e), and Cohen's Kappa (kappa).
    Also returns category-level confusion counts.
    """
    n = len(labels1)
    if n == 0:
        return 0.0, 0.0, {}

    # Observed agreement
    agreements = sum(1 for l1, l2 in zip(labels1, labels2) if l1 == l2)
    p_o = agreements / n

    # Marginals
    c1 = {c: sum(1 for l in labels1 if l == c) for c in categories}
    c2 = {c: sum(1 for l in labels2 if l == c) for c in categories}

    p_e = sum((c1[c] / n) * (c2[c] / n) for c in categories)

    if p_e == 1.0:
        kappa = 1.0
    else:
        kappa = (p_o - p_e) / (1.0 - p_e)

    # Confusion counts
    confusion: Dict[Tuple[str, str], int] = {}
    for l1, l2 in zip(labels1, labels2):
        pair = (l1, l2)
        confusion[pair] = confusion.get(pair, 0) + 1

    return p_o, kappa, confusion


def analyze_agreement(update_status: bool = True) -> Dict[str, Any]:
    if not ANN1_FILE.exists() or not ANN2_FILE.exists():
        raise FileNotFoundError(f"Missing response templates in {PHASE6A_DIR}")

    ann1_rows = load_csv(ANN1_FILE)
    ann2_rows = load_csv(ANN2_FILE)

    tot1, comp1, c_rows1 = check_annotator_completion(ann1_rows)
    tot2, comp2, c_rows2 = check_annotator_completion(ann2_rows)

    logger.info(f"Annotator 01 Completion: {comp1} / {tot1} records")
    logger.info(f"Annotator 02 Completion: {comp2} / {tot2} records")

    status_data = {}
    if STATUS_FILE.exists():
        with open(STATUS_FILE, mode="r", encoding="utf-8") as f:
            status_data = json.load(f)

    status_data["annotator_01"]["completed_count"] = comp1
    status_data["annotator_01"]["status"] = "COMPLETED" if comp1 == tot1 else ("IN_PROGRESS" if comp1 > 0 else "AWAITING_HUMAN_INPUT")

    status_data["annotator_02"]["completed_count"] = comp2
    status_data["annotator_02"]["status"] = "COMPLETED" if comp2 == tot2 else ("IN_PROGRESS" if comp2 > 0 else "AWAITING_HUMAN_INPUT")

    # Both annotators must complete at least 1 common record for evaluation
    ann1_by_id = {r["record_id"]: (r.get("primary_label") or "").strip().lower() for r in ann1_rows if (r.get("primary_label") or "").strip().lower() in VALID_PRIMARY_LABELS}
    ann2_by_id = {r["record_id"]: (r.get("primary_label") or "").strip().lower() for r in ann2_rows if (r.get("primary_label") or "").strip().lower() in VALID_PRIMARY_LABELS}

    common_ids = sorted(set(ann1_by_id.keys()) & set(ann2_by_id.keys()))

    if len(common_ids) < 10:  # Minimum statistical requirement
        logger.info(f"Insufficient completed common annotations ({len(common_ids)} records). Agreement analysis is PENDING human annotation.")
        status_data["inter_annotator_agreement"]["status"] = "AWAITING_COMPLETED_REVIEWS"
        status_data["inter_annotator_agreement"]["cohens_kappa"] = None
        status_data["inter_annotator_agreement"]["raw_agreement_percent"] = None
        status_data["evaluation_readiness_status"] = "BLOCKED_PENDING_HUMAN_ANNOTATION"
        
        if update_status:
            with open(STATUS_FILE, mode="w", encoding="utf-8") as f:
                json.dump(status_data, f, indent=2)
            logger.info(f"Updated status manifest at {STATUS_FILE}")

        return {
            "status": "PENDING_HUMAN_ANNOTATION",
            "common_records_count": len(common_ids),
            "annotator_01_completed": comp1,
            "annotator_02_completed": comp2,
            "cohens_kappa": None,
            "raw_agreement_percent": None
        }

    # If common records exist, calculate statistics
    categories = sorted(list(VALID_PRIMARY_LABELS))
    l1 = [ann1_by_id[rid] for rid in common_ids]
    l2 = [ann2_by_id[rid] for rid in common_ids]

    p_o, kappa, confusion = compute_cohens_kappa(l1, l2, categories)
    raw_pct = p_o * 100.0

    logger.info(f"Common Evaluated Records: {len(common_ids)}")
    logger.info(f"Raw Percent Agreement: {raw_pct:.2f}%")
    logger.info(f"Cohen's Kappa: {kappa:.4f}")

    # Export Disagreements to Adjudication Queue
    disagreements = []
    ann1_full_map = {r["record_id"]: r for r in ann1_rows}
    ann2_full_map = {r["record_id"]: r for r in ann2_rows}

    for rid in common_ids:
        r1 = ann1_full_map[rid]
        r2 = ann2_full_map[rid]
        if r1["primary_label"] != r2["primary_label"] or r1.get("subtype") != r2.get("subtype"):
            disagreements.append({
                "record_id": rid,
                "source_dataset": r1["source_dataset"],
                "comment_text": r1["comment_text"],
                "image_available": r1["image_available"],
                "ann1_primary_label": r1["primary_label"],
                "ann1_subtype": r1.get("subtype", ""),
                "ann1_evidence_span": r1.get("evidence_span", ""),
                "ann2_primary_label": r2["primary_label"],
                "ann2_subtype": r2.get("subtype", ""),
                "ann2_evidence_span": r2.get("evidence_span", ""),
                "adjudicated_primary_label": "",
                "adjudicated_subtype": "",
                "adjudication_rationale": "",
                "adjudicator_id": "",
                "adjudication_timestamp": "",
                "status": "PENDING_ADJUDICATION"
            })

    if disagreements:
        adj_fields = [
            "record_id", "source_dataset", "comment_text", "image_available",
            "ann1_primary_label", "ann1_subtype", "ann1_evidence_span",
            "ann2_primary_label", "ann2_subtype", "ann2_evidence_span",
            "adjudicated_primary_label", "adjudicated_subtype",
            "adjudication_rationale", "adjudicator_id", "adjudication_timestamp", "status"
        ]
        with open(ADJUDICATION_QUEUE_FILE, mode="w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=adj_fields)
            writer.writeheader()
            writer.writerows(disagreements)
        logger.info(f"Exported {len(disagreements)} disagreements to {ADJUDICATION_QUEUE_FILE}")

    # Update manifest
    status_data["inter_annotator_agreement"]["status"] = "CALCULATED"
    status_data["inter_annotator_agreement"]["cohens_kappa"] = round(kappa, 4)
    status_data["inter_annotator_agreement"]["raw_agreement_percent"] = round(raw_pct, 2)
    status_data["adjudication"]["completed_count"] = 0
    status_data["adjudication"]["unanimous_count"] = len(common_ids) - len(disagreements)
    status_data["adjudication"]["disputed_count"] = len(disagreements)
    status_data["adjudication"]["status"] = "AWAITING_ADJUDICATION" if disagreements else "COMPLETE_UNANIMOUS"

    target_thresh = status_data["inter_annotator_agreement"].get("target_kappa_threshold", 0.70)
    if kappa >= target_thresh and comp1 == tot1 and comp2 == tot2:
        status_data["evaluation_readiness_status"] = "PILOT_VALIDATED"
    else:
        status_data["evaluation_readiness_status"] = "BLOCKED_PENDING_ADJUDICATION_OR_COMPLETION"

    if update_status:
        with open(STATUS_FILE, mode="w", encoding="utf-8") as f:
            json.dump(status_data, f, indent=2)
        logger.info(f"Updated status manifest at {STATUS_FILE}")

    return {
        "status": "CALCULATED",
        "common_records_count": len(common_ids),
        "raw_agreement_percent": raw_pct,
        "cohens_kappa": kappa,
        "disagreements_count": len(disagreements)
    }


def main():
    parser = argparse.ArgumentParser(description="Analyze Phase 6A Inter-Annotator Agreement")
    parser.add_argument("--check-status", action="store_true", help="Check and report current completion status")
    parser.add_argument("--no-update", action="store_true", help="Do not write back to annotation_status_manifest.json")
    args = parser.parse_args()

    analyze_agreement(update_status=not args.no_update)


if __name__ == "__main__":
    main()
