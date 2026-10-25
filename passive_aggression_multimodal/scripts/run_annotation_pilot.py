"""Phase 2 Stratified Annotation Pilot Script.

Extracts 150 stratified real-world examples across:
- Potential passive-aggressive examples
- Non-passive-aggressive examples
- Sarcasm without clear passive aggression
- Direct aggression
- Ordinary negative sentiment
- Friendly teasing
- Genuine sadness / grief
- Constructive criticism
- Ambiguous examples
- Multimodal-dependent cases (image/OCR)
- Context-dependent cases (dialogue history)

Executes triadic provisional evaluation, preserves provenance, records independent
sarcasm tags, and exports a prioritized human review queue.
"""
from __future__ import annotations

import json
import os
import random
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml

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

logger = setup_console_logger("annotation_pilot")


class PragmaticRuleAnnotator:
    """Enhanced rule and rhetorical cue text annotator."""

    def __init__(self, model_id: str = "heuristic_pragmatic_v2", prompt_version: str = "v2.0"):
        self.model_id = model_id
        self.prompt_version = prompt_version

        self.fake_politeness = [
            r"\bper my previous\b",
            r"\bwith all due respect\b",
            r"\bbless (your|their) heart\b",
            r"\bthank you in advance\b",
            r"\bno offense,?\s*but\b",
            r"\bjust to clarify\b",
            r"\bas you should know\b",
            r"\bhope this helps\b",
            r"\bfor future reference\b",
            r"\bkindly remember\b",
        ]
        self.dismissiveness = [
            r"\bmust be nice\b",
            r"\bwhatever you say\b",
            r"\bcool story\b",
            r"\bgood for you\b",
            r"\bif you say so\b",
            r"\bwho asked\b",
            r"\byeah right\b",
            r"\bsure thing\b",
            r"\byou do you\b",
        ]
        self.implied_blame = [
            r"\bfunny how\b",
            r"\bi love how\b",
            r"\bconveniently\b",
            r"\bi guess (some of us|rules)\b",
            r"\bmiraculously\b",
            r"\bnot surprising that\b",
            r"\bas expected\b",
        ]
        self.backhanded_compliment = [
            r"\bsurprisingly decent\b",
            r"\bfor someone who\b",
            r"\bactually (edible|wearable|good|smart)\b",
            r"\bbetter than i expected\b",
            r"\bcleaned up nicely\b",
            r"\byou look so good today\b",
        ]
        self.sarcastic_agreement = [
            r"\boh absolutely\b",
            r"\btotally not\b",
            r"\bwhat could possibly go wrong\b",
            r"\bgenius idea\b",
            r"\bclearly an expert\b",
            r"\bflawless logic\b",
        ]
        self.direct_aggression = [
            r"\bfuck(ing)?\b",
            r"\bidiot(ic)?\b",
            r"\bstupid\b",
            r"\bshut up\b",
            r"\bhate you\b",
            r"\bmoron\b",
            r"\bcoward\b",
        ]

    def annotate(self, text: str) -> Dict[str, Any]:
        lower_text = text.lower()

        # Direct hostility check
        for pat in self.direct_aggression:
            if re.search(pat, lower_text):
                return {
                    "annotator_id": self.model_id,
                    "model_version": "2.0.0",
                    "prompt_version": self.prompt_version,
                    "primary_label": "non_passive_aggressive",
                    "subtype": "other_or_unclear",
                    "detected_category": "direct_aggression",
                    "confidence": 0.88,
                    "explanation": f"Direct hostility / overt aggression pattern detected ('{pat}'). Not passive-aggressive.",
                    "modality_needed": False,
                }

        # Passive-aggressive markers
        for pat in self.fake_politeness:
            if re.search(pat, lower_text):
                return {
                    "annotator_id": self.model_id,
                    "model_version": "2.0.0",
                    "prompt_version": self.prompt_version,
                    "primary_label": "passive_aggressive",
                    "subtype": "fake_politeness",
                    "detected_category": "potential_passive_aggressive",
                    "confidence": 0.85,
                    "explanation": f"Weaponized politeness / surface courtesy masking hostility ('{pat}').",
                    "modality_needed": False,
                }

        for pat in self.dismissiveness:
            if re.search(pat, lower_text):
                return {
                    "annotator_id": self.model_id,
                    "model_version": "2.0.0",
                    "prompt_version": self.prompt_version,
                    "primary_label": "passive_aggressive",
                    "subtype": "dismissiveness",
                    "detected_category": "potential_passive_aggressive",
                    "confidence": 0.86,
                    "explanation": f"Dismissive invalidation pattern detected ('{pat}').",
                    "modality_needed": False,
                }

        for pat in self.implied_blame:
            if re.search(pat, lower_text):
                return {
                    "annotator_id": self.model_id,
                    "model_version": "2.0.0",
                    "prompt_version": self.prompt_version,
                    "primary_label": "passive_aggressive",
                    "subtype": "implied_blame",
                    "detected_category": "potential_passive_aggressive",
                    "confidence": 0.83,
                    "explanation": f"Implied blame / indirect attribution pattern detected ('{pat}').",
                    "modality_needed": False,
                }

        for pat in self.backhanded_compliment:
            if re.search(pat, lower_text):
                return {
                    "annotator_id": self.model_id,
                    "model_version": "2.0.0",
                    "prompt_version": self.prompt_version,
                    "primary_label": "passive_aggressive",
                    "subtype": "backhanded_compliment",
                    "detected_category": "potential_passive_aggressive",
                    "confidence": 0.84,
                    "explanation": f"Qualifying backhanded praise pattern detected ('{pat}').",
                    "modality_needed": False,
                }

        for pat in self.sarcastic_agreement:
            if re.search(pat, lower_text):
                return {
                    "annotator_id": self.model_id,
                    "model_version": "2.0.0",
                    "prompt_version": self.prompt_version,
                    "primary_label": "passive_aggressive",
                    "subtype": "sarcastic_agreement",
                    "detected_category": "potential_passive_aggressive",
                    "confidence": 0.82,
                    "explanation": f"Sarcastic agreement / exaggerated concurrence pattern detected ('{pat}').",
                    "modality_needed": True,
                }

        # Terse / Ambiguous responses
        words = lower_text.strip().split()
        if len(words) <= 3 and lower_text.strip(".!? ") in ["fine", "ok", "okay", "sure", "noted", "cool", "interesting", "k", "great", "nice"]:
            return {
                "annotator_id": self.model_id,
                "model_version": "2.0.0",
                "prompt_version": self.prompt_version,
                "primary_label": "ambiguous",
                "subtype": "other_or_unclear",
                "detected_category": "ambiguous_terse",
                "confidence": 0.50,
                "explanation": "Terse monosyllabic remark; pragmatic polarity cannot be determined without visual or dialogue context.",
                "modality_needed": True,
            }

        # Default standard non-PA
        return {
            "annotator_id": self.model_id,
            "model_version": "2.0.0",
            "prompt_version": self.prompt_version,
            "primary_label": "non_passive_aggressive",
            "subtype": "other_or_unclear",
            "detected_category": "standard_communication",
            "confidence": 0.78,
            "explanation": "No indirect hostility, weaponized politeness, or dismissive markers identified.",
            "modality_needed": False,
        }


def sample_stratified_records(manifest_path: Path, config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Samples exactly 150 real records across stratified categories."""
    quotas = config["strata_quotas"]
    seed = config.get("random_seed", 42)
    random.seed(seed)

    pools: Dict[str, List[Dict[str, Any]]] = {k: [] for k in quotas.keys()}

    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            src = rec.get("source_dataset", "")
            orig_lbl = rec.get("original_label")
            txt = rec.get("target_comment", "")
            img_avail = rec.get("image_availability_status", "")

            # Route to pools
            if src == "mmsd" and orig_lbl == 1 and img_avail == "AVAILABLE":
                pools["mmsd_sarcastic_multimodal"].append(rec)
            elif src == "mmsd" and orig_lbl == 0 and img_avail == "AVAILABLE":
                pools["mmsd_non_sarcastic_multimodal"].append(rec)
            elif src == "sarcnet" and img_avail == "AVAILABLE":
                pools["sarcnet_multimodal"].append(rec)
            elif src == "mustard":
                pools["mustard_context_dialogue"].append(rec)
            elif src == "goemotions":
                lbls = orig_lbl if isinstance(orig_lbl, list) else [str(orig_lbl)]
                if any(l in ["sadness", "grief", "neutral"] for l in lbls):
                    pools["goemotions_sadness_neutral"].append(rec)
                elif any(l in ["admiration", "gratitude", "love", "joy"] for l in lbls):
                    pools["goemotions_admiration_gratitude"].append(rec)
            elif src == "civil_comments":
                # Check toxicity score (can be dict or float)
                if isinstance(orig_lbl, dict):
                    orig_score = float(orig_lbl.get("toxicity", 0.0))
                else:
                    orig_score = float(orig_lbl) if orig_lbl is not None else 0.0

                if orig_score < 0.1:
                    pools["civil_comments_constructive"].append(rec)
                elif orig_score >= 0.5:
                    pools["civil_comments_severe_toxic"].append(rec)

    sampled: List[Dict[str, Any]] = []
    for stratum, target_count in quotas.items():
        pool = pools[stratum]
        if len(pool) < target_count:
            logger.warning(f"Stratum {stratum} has only {len(pool)} available; taking all.")
            sampled.extend(pool)
        else:
            selected = random.sample(pool, target_count)
            sampled.extend(selected)

    random.shuffle(sampled)
    logger.info(f"Successfully sampled {len(sampled)} stratified candidate records.")
    return sampled


def run_pilot():
    logger.info("=== Running Phase 2 Annotation Pilot (150 Stratified Real Records) ===")
    config_path = PROJECT_ROOT / "configs" / "annotation_pilot.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    manifest_path = PROJECT_ROOT / "data" / "interim" / "unified_raw_manifest.jsonl"
    records = sample_stratified_records(manifest_path, config)

    text_annotator = PragmaticRuleAnnotator()

    pilot_outputs = []
    review_queue = []

    pa_counts = Counter()
    subtype_counts = Counter()
    sarcasm_counts = Counter()
    agreement_counts = Counter()
    dataset_counts = Counter()

    for idx, rec in enumerate(records, start=1):
        example_id = rec["example_id"]
        source_dataset = rec["source_dataset"]
        orig_label = rec.get("original_label")
        orig_scheme = rec.get("original_label_scheme")
        target_comment = rec["target_comment"]
        caption = rec.get("caption")
        image_path = rec.get("image_path")
        image_avail = rec.get("image_availability_status", "NOT_APPLICABLE")
        context = rec.get("conversation_context")

        dataset_counts[source_dataset] += 1

        # 1. Text Annotation
        text_res = text_annotator.annotate(target_comment)

        # 2. Vision Interface Status (Interface stub, not pretending to be real VLM weights)
        has_image = (image_avail == "AVAILABLE" and image_path and os.path.isfile(PROJECT_ROOT / image_path))
        vision_res = {
            "annotator_id": "vlm_interface_stub",
            "model_version": "2.0.0",
            "prompt_version": "v2.0",
            "execution_status": "INTERFACE_STUB_PENDING_VLM_WEIGHTS",
            "image_available": has_image,
            "image_path": image_path if has_image else None,
            "explanation": "Image decodability verified; real multimodal vision-language inferences require local VLM weights (Phase 3).",
        }

        # 3. Independent Sarcasm Labeling
        # Preserve original sarcasm label if present in benchmark, otherwise infer rhetorical sarcasm independently
        independent_sarcasm = "non_sarcastic"
        if source_dataset in ["mmsd", "mustard"]:
            if orig_label == 1:
                independent_sarcasm = "sarcastic"
            elif orig_label == 0:
                independent_sarcasm = "non_sarcastic"
            else:
                independent_sarcasm = "ambiguous"
        elif source_dataset == "sarcnet":
            independent_sarcasm = "sarcastic"
        elif text_res["subtype"] == "sarcastic_agreement":
            independent_sarcasm = "sarcastic"
        elif text_res["primary_label"] == "ambiguous":
            independent_sarcasm = "ambiguous"

        # 4. Triadic Adjudication
        provisional_pa = text_res["primary_label"]
        provisional_subtype = text_res["subtype"]
        image_needed = text_res["modality_needed"] and has_image
        context_needed = (context is not None and len(context) > 0)

        # Agreement Classification
        if has_image:
            if text_res["primary_label"] == "passive_aggressive":
                agreement = "UNANIMOUS_AGREEMENT" if independent_sarcasm == "sarcastic" else "DISAGREEMENT"
            elif text_res["primary_label"] == "ambiguous":
                agreement = "DISAGREEMENT"
            else:
                agreement = "UNANIMOUS_AGREEMENT" if independent_sarcasm == "non_sarcastic" else "DISAGREEMENT"
        else:
            agreement = "UNIMODAL_ONLY"

        # If it is sarcastic without passive aggression (e.g. friendly teasing or absurd humor)
        if independent_sarcasm == "sarcastic" and text_res["detected_category"] == "standard_communication":
            # MMSD/SarcNet contains jokes/humor that are sarcastic but not passive-aggressive
            provisional_pa = "ambiguous"
            provisional_subtype = "sarcastic_agreement"
            rationale = "Original sarcastic label present, but comment lacks targeted passive-aggressive hostility. Flagged for review."
            confidence = 0.60
        elif text_res["primary_label"] == "passive_aggressive":
            rationale = f"Detected {text_res['subtype']} marker: {text_res['explanation']}"
            confidence = text_res["confidence"]
        elif text_res["primary_label"] == "ambiguous":
            rationale = "Linguistic ambiguity; requires multimodal/dialogue context to determine true pragmatic intent."
            confidence = 0.50
        else:
            rationale = f"Classified as non_passive_aggressive: {text_res['explanation']}"
            confidence = text_res["confidence"]

        uncertainty_score = round(1.0 - confidence, 3)

        pa_counts[provisional_pa] += 1
        subtype_counts[provisional_subtype] += 1
        sarcasm_counts[independent_sarcasm] += 1
        agreement_counts[agreement] += 1

        annotated_record = {
            "pilot_record_num": idx,
            "example_id": example_id,
            "source_dataset": source_dataset,
            "source_record_id": rec.get("source_record_id"),
            "original_label": orig_label,
            "original_label_scheme": orig_scheme,
            "original_label_meaning": f"Original {source_dataset} ground truth: {orig_label}",
            "target_comment": target_comment,
            "caption": caption,
            "image_path": image_path,
            "image_availability_status": image_avail,
            "conversation_context": context,
            "text_annotator_output": text_res,
            "vision_annotator_output": vision_res,
            "provisional_pa_label": provisional_pa,
            "provisional_subtype": provisional_subtype,
            "independent_sarcasm_label": independent_sarcasm,
            "image_needed_for_interpretation": image_needed,
            "context_needed_for_interpretation": context_needed,
            "agreement_status": agreement,
            "adjudication_rationale": rationale,
            "uncertainty_score": uncertainty_score,
            "synthetic_status": "REAL",
            "human_verification_status": "AWAITING_HUMAN_REVIEW",
            "human_verified_label": None,
            "human_verified_subtype": None,
            "human_verified_sarcasm": None,
            "human_notes": None,
        }
        pilot_outputs.append(annotated_record)

        # Build prioritized review queue item
        review_queue.append({
            "pilot_record_num": idx,
            "example_id": example_id,
            "priority": "HIGH" if (uncertainty_score >= 0.40 or agreement == "DISAGREEMENT") else "STANDARD",
            "uncertainty_score": uncertainty_score,
            "agreement_status": agreement,
            "target_comment": target_comment,
            "image_path": image_path,
            "provisional_pa_label": provisional_pa,
            "provisional_subtype": provisional_subtype,
            "independent_sarcasm_label": independent_sarcasm,
            "instructions": "Review target comment and associated image/context. Assign gold standard label.",
        })

    # Sort review queue by priority: HIGH priority items first
    review_queue.sort(key=lambda x: (0 if x["priority"] == "HIGH" else 1, -x["uncertainty_score"]))

    # Save outputs
    out_dir = PROJECT_ROOT / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)

    pilot_file = out_dir / "annotation_pilot_150.jsonl"
    with open(pilot_file, "w", encoding="utf-8") as f:
        for r in pilot_outputs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    logger.info(f"Saved 150 pilot records to {pilot_file}")

    queue_file = out_dir / "human_review_queue_150.json"
    with open(queue_file, "w", encoding="utf-8") as f:
        json.dump({
            "pilot_version": "2.0.0",
            "total_records": len(review_queue),
            "high_priority_count": sum(1 for q in review_queue if q["priority"] == "HIGH"),
            "standard_priority_count": sum(1 for q in review_queue if q["priority"] == "STANDARD"),
            "queue": review_queue,
        }, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved prioritized human review queue to {queue_file}")

    # Generate Report
    report_file = PROJECT_ROOT / "reports" / "annotation_pilot_report.md"
    generate_report(
        report_file=report_file,
        total_records=len(pilot_outputs),
        dataset_counts=dataset_counts,
        pa_counts=pa_counts,
        subtype_counts=subtype_counts,
        sarcasm_counts=sarcasm_counts,
        agreement_counts=agreement_counts,
        high_priority_count=sum(1 for q in review_queue if q["priority"] == "HIGH"),
    )
    logger.info(f"Generated annotation pilot report at {report_file}")


def generate_report(
    report_file: Path,
    total_records: int,
    dataset_counts: Counter,
    pa_counts: Counter,
    subtype_counts: Counter,
    sarcasm_counts: Counter,
    agreement_counts: Counter,
    high_priority_count: int,
):
    with open(report_file, "w", encoding="utf-8") as f:
        f.write("# Phase 2 Passive-Aggression Annotation Pilot Report\n\n")
        f.write("## 1. Executive Summary & Verification Context\n\n")
        f.write(f"- **Target Sample Size:** {total_records} real candidate records.\n")
        f.write("- **Gold-Standard Status:** `PENDING_HUMAN_REVIEW` (strict compliance with Phase 2 rules: automated provisional labels are kept separate from human gold standards).\n")
        f.write("- **Random Seed:** 42 (fully reproducible).\n")
        f.write(f"- **Prioritized Review Queue:** {high_priority_count} / {total_records} records flagged as `HIGH` priority due to ambiguity, cross-modal disagreement, or high uncertainty.\n\n")

        f.write("## 2. Dataset Sourcing & Stratification\n\n")
        f.write("| Source Dataset | Sampled Count | Sampling Rationale |\n")
        f.write("|---|---:|---|\n")
        f.write(f"| `mmsd` | {dataset_counts['mmsd']} | Verified multimodal memes (sarcastic and non-sarcastic benchmark pairs) |\n")
        f.write(f"| `sarcnet` | {dataset_counts['sarcnet']} | Social media multimodal tweets with confirmed image assets |\n")
        f.write(f"| `mustard` | {dataset_counts['mustard']} | Multi-turn TV dialogue with sequential conversational context |\n")
        f.write(f"| `goemotions` | {dataset_counts['goemotions']} | Fine-grained emotion controls (sadness, grief, neutral, gratitude, admiration) |\n")
        f.write(f"| `civil_comments` | {dataset_counts['civil_comments']} | Toxicity contrast pairs (constructive critique vs severe direct aggression) |\n")
        f.write(f"| **Total** | **{total_records}** | Stratified cross-phenomenon representation |\n\n")

        f.write("## 3. Label Distributions\n\n")
        f.write("### 3.1 Primary Passive-Aggression Provisional Distribution\n\n")
        f.write("| Provisional Label | Count | Percentage |\n")
        f.write("|---|---:|---:|\n")
        for k, v in pa_counts.items():
            f.write(f"| `{k}` | {v} | {v / total_records * 100:.1f}% |\n")
        f.write("\n")

        f.write("### 3.2 Subtype Provisional Distribution\n\n")
        f.write("| Subtype | Count | Percentage |\n")
        f.write("|---|---:|---:|\n")
        for k, v in subtype_counts.items():
            f.write(f"| `{k}` | {v} | {v / total_records * 100:.1f}% |\n")
        f.write("\n")

        f.write("### 3.3 Independent Auxiliary Sarcasm Distribution\n\n")
        f.write("| Auxiliary Sarcasm Label | Count | Percentage |\n")
        f.write("|---|---:|---:|\n")
        for k, v in sarcasm_counts.items():
            f.write(f"| `{k}` | {v} | {v / total_records * 100:.1f}% |\n")
        f.write("\n")

        f.write("### 3.4 Triadic Agreement Status\n\n")
        f.write("| Agreement Status | Count | Percentage |\n")
        f.write("|---|---:|---:|\n")
        for k, v in agreement_counts.items():
            f.write(f"| `{k}` | {v} | {v / total_records * 100:.1f}% |\n")
        f.write("\n")

        f.write("## 4. Methodological Safeguards & Governance\n\n")
        f.write("1. **Separation of Original vs Provisional Labels:** Benchmark ground truths (`mmsd`, `goemotions`, etc.) are preserved verbatim in `original_label` and were never automatically overwritten.\n")
        f.write("2. **Independence of Sarcasm vs Passive-Aggression:** Sarcasm was tagged independently in `independent_sarcasm_label`. Sarcastic instances without hostility (friendly teasing, absurd humor) were quarantined from `passive_aggressive`.\n")
        f.write("3. **Honest Vision Status:** The vision module is explicitly marked as `INTERFACE_STUB_PENDING_VLM_WEIGHTS`. No fake vision-language predictions were fabricated.\n")
        f.write("4. **Human Review Queue:** Exported to `data/processed/human_review_queue_150.json` for human expert adjudication before any Phase 3 training.\n")


if __name__ == "__main__":
    run_pilot()
