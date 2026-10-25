"""Annotation Quality Analysis and Inter-Annotator Agreement Engine (Phase 5).

Provides schema validation, dual-annotator matching, Cohen's kappa calculation,
confusion matrices, adjudication reconciliation, and statistical diagnostics.
"""
from __future__ import annotations

import csv
import json
import os
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np
from sklearn.metrics import cohen_kappa_score, confusion_matrix

VALID_PRIMARY_LABELS: Set[str] = {
    "passive_aggressive",
    "non_passive_aggressive",
    "ambiguous",
}

VALID_SUBTYPES: Set[str] = {
    "fake_politeness",
    "indirect_criticism",
    "backhanded_compliment",
    "implied_blame",
    "sarcastic_agreement",
    "dismissiveness",
    "other_or_unclear",
}

VALID_IMAGE_ROLES: Set[str] = {
    "ESSENTIAL",
    "HELPFUL",
    "IRRELEVANT",
    "ABSENT",
}

VALID_CONTEXT_SUFFICIENCY: Set[str] = {
    "SUFFICIENT",
    "PARTIAL",
    "INSUFFICIENT",
}


def validate_annotation_row(row: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validates a single human annotation response row against schema rules.
    Returns:
        (is_valid, list_of_error_strings)
    """
    errors: List[str] = []
    rec_id = row.get("record_id", "").strip()
    if not rec_id:
        errors.append("Missing record_id")

    primary = row.get("primary_label", "").strip()
    if not primary:
        errors.append("Empty primary_label")
    elif primary not in VALID_PRIMARY_LABELS:
        errors.append(f"Invalid primary_label '{primary}'. Must be one of {VALID_PRIMARY_LABELS}")

    subtype = row.get("subtype", "").strip()
    if primary == "passive_aggressive":
        if not subtype:
            errors.append("Subtype is required when primary_label is passive_aggressive")
        elif subtype not in VALID_SUBTYPES:
            errors.append(f"Invalid subtype '{subtype}'. Must be one of {VALID_SUBTYPES}")
    elif subtype and subtype not in VALID_SUBTYPES:
        errors.append(f"Invalid subtype '{subtype}'. Must be empty or one of {VALID_SUBTYPES}")

    img_role = row.get("image_role", "").strip()
    if img_role and img_role not in VALID_IMAGE_ROLES:
        errors.append(f"Invalid image_role '{img_role}'. Must be one of {VALID_IMAGE_ROLES}")

    ctx_suff = row.get("context_sufficiency", "").strip()
    if ctx_suff and ctx_suff not in VALID_CONTEXT_SUFFICIENCY:
        errors.append(f"Invalid context_sufficiency '{ctx_suff}'. Must be one of {VALID_CONTEXT_SUFFICIENCY}")

    conf_str = str(row.get("confidence", "")).strip()
    if conf_str:
        try:
            conf = int(conf_str)
            if not (1 <= conf <= 5):
                errors.append(f"Confidence {conf} out of range [1, 5]")
        except ValueError:
            errors.append(f"Non-integer confidence '{conf_str}'")

    return (len(errors) == 0, errors)


def load_annotation_file(filepath: str) -> Dict[str, Dict[str, Any]]:
    """Loads CSV annotation file into a dict keyed by stable record_id."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Annotation file not found: {filepath}")

    records = {}
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rec_id = row.get("record_id", "").strip()
            if rec_id:
                records[rec_id] = row
    return records


def compute_inter_annotator_agreement(
    records1: Dict[str, Dict[str, Any]],
    records2: Dict[str, Dict[str, Any]],
    target_kappa: float = 0.70,
) -> Dict[str, Any]:
    """
    Computes rigorous inter-annotator metrics between two independent review files.
    """
    common_ids = sorted(list(set(records1.keys()).intersection(set(records2.keys()))))
    total_samples = max(len(records1), len(records2))

    completed1 = 0
    completed2 = 0
    valid_pairs = []
    y1_labels = []
    y2_labels = []
    disputed_ids = []
    unanimous_ids = []
    invalid_rows = []

    for rid in common_ids:
        r1 = records1[rid]
        r2 = records2[rid]

        p1 = r1.get("primary_label", "").strip()
        p2 = r2.get("primary_label", "").strip()

        if p1:
            completed1 += 1
            v1, errs1 = validate_annotation_row(r1)
            if not v1:
                invalid_rows.append({"annotator": "annotator_01", "record_id": rid, "errors": errs1})

        if p2:
            completed2 += 1
            v2, errs2 = validate_annotation_row(r2)
            if not v2:
                invalid_rows.append({"annotator": "annotator_02", "record_id": rid, "errors": errs2})

        # Pair is evaluated only if both annotators submitted a label
        if p1 and p2:
            v1, _ = validate_annotation_row(r1)
            v2, _ = validate_annotation_row(r2)
            if v1 and v2:
                valid_pairs.append(rid)
                y1_labels.append(p1)
                y2_labels.append(p2)
                if p1 == p2:
                    unanimous_ids.append(rid)
                else:
                    disputed_ids.append(rid)

    num_evaluated = len(valid_pairs)
    warnings = []

    if num_evaluated == 0:
        return {
            "status": "AWAITING_HUMAN_REVIEWS",
            "total_samples": total_samples,
            "annotator_1_completed": completed1,
            "annotator_2_completed": completed2,
            "both_completed_count": 0,
            "raw_agreement_percent": None,
            "cohens_kappa": None,
            "target_kappa_threshold": target_kappa,
            "target_achieved": False,
            "unanimous_count": 0,
            "disputed_count": 0,
            "invalid_count": len(invalid_rows),
            "invalid_details": invalid_rows,
            "warnings": ["Zero valid double-annotated rows found. Agreement cannot be computed."],
        }

    # Label space definition
    classes = ["passive_aggressive", "non_passive_aggressive", "ambiguous"]
    cm = confusion_matrix(y1_labels, y2_labels, labels=classes).tolist()
    raw_agreement = float(np.mean(np.array(y1_labels) == np.array(y2_labels))) * 100.0

    try:
        kappa = float(cohen_kappa_score(y1_labels, y2_labels, labels=classes))
    except Exception as e:
        kappa = None
        warnings.append(f"Kappa calculation error: {e}")

    # Statistical warnings
    if num_evaluated < 50:
        warnings.append(f"Sample size ({num_evaluated}) is under 50; kappa confidence intervals may be wide.")

    present_classes = set(y1_labels).union(set(y2_labels))
    missing_classes = set(classes).difference(present_classes)
    if missing_classes:
        warnings.append(f"Missing classes in evaluation sample: {list(missing_classes)}")

    target_met = (kappa is not None) and (kappa >= target_kappa)

    return {
        "status": "COMPUTED",
        "total_samples": total_samples,
        "annotator_1_completed": completed1,
        "annotator_2_completed": completed2,
        "both_completed_count": num_evaluated,
        "raw_agreement_percent": round(raw_agreement, 2),
        "cohens_kappa": round(kappa, 4) if kappa is not None else None,
        "target_kappa_threshold": target_kappa,
        "target_achieved": target_met,
        "unanimous_count": len(unanimous_ids),
        "disputed_count": len(disputed_ids),
        "disputed_record_ids": disputed_ids,
        "invalid_count": len(invalid_rows),
        "invalid_details": invalid_rows,
        "confusion_matrix": {
            "classes": classes,
            "matrix": cm,
        },
        "warnings": warnings,
    }


def generate_adjudication_file(
    records1: Dict[str, Dict[str, Any]],
    records2: Dict[str, Dict[str, Any]],
    output_path: str,
) -> int:
    """
    Populates an adjudication reconciliation file comparing Annotator 1 and Annotator 2.
    """
    common_ids = sorted(list(set(records1.keys()).intersection(set(records2.keys()))))
    rows = []

    fieldnames = [
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
        "adjudication_notes",
    ]

    for rid in common_ids:
        r1 = records1[rid]
        r2 = records2[rid]

        p1 = r1.get("primary_label", "").strip()
        p2 = r2.get("primary_label", "").strip()

        if not p1 and not p2:
            agr_type = "PENDING_DUAL_ANNOTATION"
            adj_status = "AWAITING_HUMAN_REVIEW"
            adj_prim = ""
            adj_sub = ""
        elif p1 and not p2:
            agr_type = "PARTIAL_ANNOTATOR_1_ONLY"
            adj_status = "AWAITING_ANNOTATOR_2"
            adj_prim = ""
            adj_sub = ""
        elif p2 and not p1:
            agr_type = "PARTIAL_ANNOTATOR_2_ONLY"
            adj_status = "AWAITING_ANNOTATOR_1"
            adj_prim = ""
            adj_sub = ""
        elif p1 == p2:
            agr_type = "UNANIMOUS_AGREEMENT"
            adj_status = "RESOLVED_UNANIMOUS"
            adj_prim = p1
            adj_sub = r1.get("subtype", "") or r2.get("subtype", "")
        else:
            agr_type = "DISPUTE_REQUIRES_ADJUDICATION"
            adj_status = "AWAITING_ADJUDICATION_CONFERENCE"
            adj_prim = ""
            adj_sub = ""

        row = {
            "sample_index": r1.get("sample_index", ""),
            "record_id": rid,
            "source_dataset": r1.get("source_dataset", ""),
            "comment_text": r1.get("comment_text", ""),
            "image_available": r1.get("image_available", ""),
            "image_path": r1.get("image_path", ""),
            "context_text": r1.get("context_text", ""),
            "annotator_01_primary_label": p1,
            "annotator_01_subtype": r1.get("subtype", ""),
            "annotator_01_confidence": r1.get("confidence", ""),
            "annotator_02_primary_label": p2,
            "annotator_02_subtype": r2.get("subtype", ""),
            "annotator_02_confidence": r2.get("confidence", ""),
            "agreement_type": agr_type,
            "adjudicated_primary_label": adj_prim,
            "adjudicated_subtype": adj_sub,
            "adjudicated_image_role": r1.get("image_role", "") if p1 == p2 else "",
            "adjudicated_context_sufficiency": r1.get("context_sufficiency", "") if p1 == p2 else "",
            "adjudicator_id": "consensus_auto" if p1 == p2 and p1 else "",
            "adjudication_timestamp": "",
            "adjudication_status": adj_status,
            "adjudication_notes": "Unanimous label accepted" if p1 == p2 and p1 else "",
        }
        rows.append(row)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return len(rows)
