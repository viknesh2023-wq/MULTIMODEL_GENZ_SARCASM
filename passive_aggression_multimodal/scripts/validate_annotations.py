"""Validation script ensuring all annotation and synthetic files comply with schema requirements."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from pydantic import ValidationError

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utilities.logger import setup_console_logger
from src.utilities.schema import (
    AnnotationStatus,
    ImageAvailabilityStatus,
    PASubtype,
    PrimaryPALabel,
    SyntheticStatus,
    UnifiedRecord,
)

logger = setup_console_logger("annotation_validator")


def validate_jsonl_file(filepath: Path) -> dict:
    """Validates every record in a JSONL file against UnifiedRecord specifications."""
    logger.info(f"Validating file: {filepath.name}")
    if not filepath.exists():
        return {"exists": False, "error": "File does not exist"}

    total_records = 0
    valid_records = 0
    invalid_records = 0
    errors = []

    primary_label_counter = {}
    subtype_counter = {}

    with open(filepath, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            total_records += 1
            try:
                data = json.loads(line)
                rec = UnifiedRecord.model_validate(data)
                valid_records += 1

                # Track distributions
                if rec.passive_aggression_label:
                    p = rec.passive_aggression_label
                    primary_label_counter[p] = primary_label_counter.get(p, 0) + 1
                if rec.passive_aggression_subtype:
                    s = rec.passive_aggression_subtype
                    subtype_counter[s] = subtype_counter.get(s, 0) + 1

            except (ValidationError, Exception) as e:
                invalid_records += 1
                errors.append({"line": line_num, "error": str(e)})

    return {
        "file": filepath.name,
        "total_records": total_records,
        "valid_records": valid_records,
        "invalid_records": invalid_records,
        "is_all_valid": (invalid_records == 0 and total_records > 0),
        "primary_label_distribution": primary_label_counter,
        "subtype_distribution": subtype_counter,
        "sample_errors": errors[:5],
    }


def main():
    logger.info("=== Running Annotation Schema Validation ===")
    targets = [
        PROJECT_ROOT / "data" / "synthetic" / "synthetic_pilot_350.jsonl",
        PROJECT_ROOT / "data" / "interim" / "unified_raw_manifest.jsonl",
    ]

    all_passed = True
    summary = {}
    for target in targets:
        if target.exists():
            res = validate_jsonl_file(target)
            summary[target.name] = res
            status_str = "PASSED" if res["is_all_valid"] else "FAILED"
            logger.info(f"[{status_str}] {target.name}: {res['valid_records']}/{res['total_records']} valid records.")
            if not res["is_all_valid"]:
                all_passed = False
        else:
            logger.warning(f"Target not found: {target}")

    report_p = PROJECT_ROOT / "data" / "manifests" / "annotation_validation_summary.json"
    with open(report_p, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Annotation validation summary written to {report_p}")
    logger.info(f"All evaluated files passed validation: {all_passed}")
    return all_passed


if __name__ == "__main__":
    passed = main()
    sys.exit(0 if passed else 1)
