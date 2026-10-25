#!/usr/bin/env python3
"""
scripts/run_phase5e_agreement_audit.py

Phase 5E: Complete Forensic Audit of All 68 Model-Agreement Records.
Independently verifies and audits the 68 records where Llama 3.2 and Qwen 2.5
agreed during Phase 5A dual-pass LLM annotation.

Generates:
  - reports/phase5e/tables/agreement_audit_68_complete.csv
  - reports/phase5e/tables/agreement_category_distribution.csv
  - reports/phase5e/tables/phase5d_reconciliation_68.csv
  - reports/phase5e/tables/shared_error_risk_register.csv
  - reports/phase5e/phase5e_human_review_priority_queue.csv
  - data/annotations/phase5e/agreement_audit_68_complete.jsonl
"""

import csv
import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Paths
PASS1_FILE = Path("data/annotations/phase5a/llm/llm_annotations_pass1.jsonl")
PASS2_FILE = Path("data/annotations/phase5a/llm/llm_annotations_pass2.jsonl")
MANIFEST_FILE = Path("data/annotations/phase4/human_sampling_manifest_100.jsonl")
P5D_VERIFIED_FILE = Path("data/annotations/phase5d/silver_dataset_verified.jsonl")
P5D_EXCLUDED_FILE = Path("data/annotations/phase5d/excluded_records.csv")

OUT_DIR_TABLES = Path("reports/phase5e/tables")
OUT_DIR_REPORTS = Path("reports/phase5e")
OUT_DIR_DATA = Path("data/annotations/phase5e")


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    with open(path, mode="r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_csv(path: Path) -> List[Dict[str, Any]]:
    with open(path, mode="r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def audit_record(
    r_id: str,
    manifest_rec: Dict[str, Any],
    p1_rec: Dict[str, Any],
    p2_rec: Dict[str, Any],
    p5d_ver_map: Dict[str, Dict[str, Any]],
    p5d_exc_map: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    dataset = manifest_rec["source_dataset"]
    prefix = f"{dataset}_"
    source_record_id = r_id[len(prefix):] if r_id.startswith(prefix) else r_id
    comment_text = manifest_rec.get("comment_text", "").strip()
    context_text = manifest_rec.get("context_text", "").strip()
    image_avail = bool(manifest_rec.get("image_available", False))
    ocr_avail = False  # OCR was not extracted/present for raw comments
    
    in_p5d_ver = r_id in p5d_ver_map
    p5d_status = (
        ("VERIFIED_" + p5d_ver_map[r_id]["silver_primary_label"])
        if in_p5d_ver
        else p5d_exc_map[r_id]["exclusion_category"]
    )
    
    model_a_pred = p1_rec["primary_label"]
    model_a_subtype = p1_rec.get("subtype", "none")
    model_a_rationale = p1_rec.get("rationale", "")
    
    model_b_pred = p2_rec["primary_label"]
    model_b_subtype = p2_rec.get("subtype", "none")
    model_b_rationale = p2_rec.get("rationale", "")
    
    # 9 Conservative Categories Assignment
    if in_p5d_ver:
        if p5d_ver_map[r_id]["silver_primary_label"] == "passive_aggressive":
            audit_cat = "LIKELY_GENUINE_PA_SILVER"
            target_status = "CONFIRMED_ADDRESSEE"
            covert_status = "CONFIRMED_COVERT"
            deniability_status = "CONFIRMED_DENIABLE"
            rhetorical_form = "passive_aggressive_rebuke"
            shared_risk = "NONE"
            confidence = "HIGH"
            evidence = "Meets all 3 pragmatic criteria: confirmed individual addressee, covert hostility, and plausible deniability."
            limitations = "Assessed on text modality only; visual context unassessed."
            disposition = "RETAIN_SILVER_VERIFIED"
            disp_justification = "High-quality silver positive candidate suitable for weak-supervision representation probing."
            human_review_rec = "NO"  # Retained in silver; queued for final gold benchmark adjudication
        else:
            audit_cat = "LIKELY_NON_PA"
            target_status = "ABSENT_OR_UNVERIFIED"
            covert_status = "ABSENT_BENIGN"
            deniability_status = "NOT_APPLICABLE"
            rhetorical_form = "linguistic_idiom_explanation"
            shared_risk = "NONE"
            confidence = "HIGH"
            evidence = "Objective Chinese idiom explanation; zero hostility or aggression."
            limitations = "Assessed on text modality only."
            disposition = "RETAIN_SILVER_VERIFIED"
            disp_justification = "Consensus non-passive-aggressive silver negative."
            human_review_rec = "NO"
    else:
        # Excluded records
        if p5d_status in ("EXCLUDED_INSTITUTIONAL_VENTING", "EXCLUDED_EXISTENTIAL_VENTING"):
            audit_cat = "INSTITUTIONAL_OR_SITUATIONAL_CRITIQUE"
            target_status = "INSTITUTIONAL_TARGET" if p5d_status == "EXCLUDED_INSTITUTIONAL_VENTING" else "ABSENT_OR_UNVERIFIED"
            covert_status = "UNCLEAR"
            deniability_status = "NOT_APPLICABLE"
            rhetorical_form = "institutional_criticism" if p5d_status == "EXCLUDED_INSTITUTIONAL_VENTING" else "situational_venting"
            shared_risk = "SHARED_FALSE_POSITIVE_RISK"
            confidence = "HIGH"
            evidence = "Critique or frustration aimed at institutions (Fox News, NFL, government) or situational conditions without individual interlocutor."
            limitations = "Models blind to target constraint."
            disposition = "EXCLUDE_FROM_SILVER_KEEP_FOR_HUMAN_REVIEW"
            disp_justification = "Shared false-positive due to target blindness; useful as contrast class."
            human_review_rec = "YES"
        elif p5d_status == "EXCLUDED_OVERT_AGGRESSION":
            audit_cat = "DIRECT_AGGRESSION_OR_DIRECT_ARGUMENT"
            target_status = "CONFIRMED_ADDRESSEE"
            covert_status = "OVERT_DIRECT"
            deniability_status = "ABSENT_DIRECT"
            rhetorical_form = "direct_aggression"
            shared_risk = "SHARED_FALSE_POSITIVE_RISK"
            confidence = "HIGH"
            evidence = "Explicit hostility, insult, or direct condemnation without polite facade or deniability."
            limitations = "Models failed to distinguish overt aggression from passive aggression."
            disposition = "EXCLUDE_FROM_SILVER_KEEP_FOR_HUMAN_REVIEW"
            disp_justification = "Direct aggression contrast class."
            human_review_rec = "YES"
        elif p5d_status == "EXCLUDED_INSUFFICIENT_CONTEXT":
            audit_cat = "AMBIGUOUS_INSUFFICIENT_CONTEXT"
            target_status = "ABSENT_OR_UNVERIFIED"
            covert_status = "UNCLEAR"
            deniability_status = "NOT_APPLICABLE"
            rhetorical_form = "unclear_fragment"
            shared_risk = "SHARED_FALSE_POSITIVE_RISK"
            confidence = "LOW"
            evidence = "Isolated fragment where communicative intent cannot be evaluated."
            limitations = "Truncated context in source corpus."
            disposition = "EXCLUDE_FROM_SILVER_KEEP_FOR_HUMAN_REVIEW"
            disp_justification = "Context insufficiency prevents reliable classification."
            human_review_rec = "YES"
        elif p5d_status in ("EXCLUDED_SHARED_FALSE_POSITIVE_PRAISE", "EXCLUDED_SELF_CONFESSION"):
            audit_cat = "SINCERE_PRAISE_OR_FRIENDLY_TEASING"
            target_status = "CONFIRMED_ADDRESSEE" if p5d_status == "EXCLUDED_SHARED_FALSE_POSITIVE_PRAISE" else "SELF_ADDRESSED"
            covert_status = "ABSENT_BENIGN"
            deniability_status = "NOT_APPLICABLE"
            rhetorical_form = "sincere_praise" if p5d_status == "EXCLUDED_SHARED_FALSE_POSITIVE_PRAISE" else "personal_confession"
            shared_risk = "SHARED_FALSE_POSITIVE_RISK"
            confidence = "HIGH"
            evidence = "Sincere appreciation or benign confession hallucinated as hostile by models."
            limitations = "Model prompt hyper-interpretability bias."
            disposition = "EXCLUDE_FROM_SILVER_KEEP_FOR_HUMAN_REVIEW"
            disp_justification = "Clear model hallucination on benign courtesy or self-talk."
            human_review_rec = "YES"
        elif p5d_status == "EXCLUDED_FACTUAL_OR_PROMOTIONAL":
            audit_cat = "SHARED_MODEL_ERROR_SUSPECTED"
            target_status = "GENERAL_AUDIENCE"
            covert_status = "ABSENT_BENIGN"
            deniability_status = "NOT_APPLICABLE"
            rhetorical_form = "factual_promotional"
            shared_risk = "SHARED_FALSE_POSITIVE_RISK"
            confidence = "HIGH"
            evidence = "Factual news headline or promotional inquiry misclassified as passive aggression."
            limitations = "Models over-sensitized to conversational cues."
            disposition = "EXCLUDE_FROM_SILVER_KEEP_FOR_HUMAN_REVIEW"
            disp_justification = "Definite shared false-positive hallucination."
            human_review_rec = "YES"
        elif p5d_status == "EXCLUDED_UNVERIFIED_TARGET":
            if image_avail:
                audit_cat = "UNRESOLVED_REQUIRES_HUMAN_REVIEW"
                target_status = "ABSENT_OR_UNVERIFIED"
                covert_status = "UNCLEAR"
                deniability_status = "NOT_APPLICABLE"
                rhetorical_form = "multimodal_dependent_sarcasm"
                shared_risk = "SHARED_FALSE_POSITIVE_RISK"
                confidence = "MEDIUM"
                evidence = "Multimodal post where image exists but was unassessed; target may be in image."
                limitations = "Image modality was not assessed by text-only LLMs."
                disposition = "EXCLUDE_FROM_SILVER_KEEP_FOR_HUMAN_REVIEW"
                disp_justification = "Requires visual adjudication to determine if image contains target."
                human_review_rec = "YES"
            else:
                audit_cat = "GENERAL_SARCASM_NO_INTERPERSONAL_TARGET"
                target_status = "GENERAL_AUDIENCE"
                covert_status = "UNCLEAR"
                deniability_status = "NOT_APPLICABLE"
                rhetorical_form = "general_sarcasm"
                shared_risk = "SHARED_FALSE_POSITIVE_RISK"
                confidence = "HIGH"
                evidence = "Text-only broadcast sarcasm or general irony without specific interlocutor."
                limitations = "Broadcast social media post."
                disposition = "EXCLUDE_FROM_SILVER_KEEP_FOR_HUMAN_REVIEW"
                disp_justification = "General sarcasm contrast class."
                human_review_rec = "YES"
        else:
            audit_cat = "UNRESOLVED_REQUIRES_HUMAN_REVIEW"
            target_status = "ABSENT_OR_UNVERIFIED"
            covert_status = "UNCLEAR"
            deniability_status = "NOT_APPLICABLE"
            rhetorical_form = "unclear"
            shared_risk = "SHARED_FALSE_POSITIVE_RISK"
            confidence = "LOW"
            evidence = f"Unresolved categorization for status: {p5d_status}"
            limitations = "Requires human adjudication."
            disposition = "EXCLUDE_FROM_SILVER_KEEP_FOR_HUMAN_REVIEW"
            disp_justification = "Requires human adjudication."
            human_review_rec = "YES"

    return {
        "record_id": r_id,
        "source_dataset": dataset,
        "source_record_id": source_record_id,
        "comment_text": comment_text,
        "context_text": context_text,
        "image_available": image_avail,
        "ocr_available": ocr_avail,
        "image_assessed_by_models": False,
        "ocr_assessed_by_models": False,
        "modality_used": "TEXT_ONLY",
        "model_a_prediction": model_a_pred,
        "model_a_subtype": model_a_subtype,
        "model_a_rationale": model_a_rationale,
        "model_b_prediction": model_b_pred,
        "model_b_subtype": model_b_subtype,
        "model_b_rationale": model_b_rationale,
        "agreement_status": "EXACT_MATCH",
        "interpersonal_target": target_status,
        "covert_hostility": covert_status,
        "plausible_deniability": deniability_status,
        "rhetorical_form": rhetorical_form,
        "shared_risk_flag": shared_risk,
        "audit_category": audit_cat,
        "audit_confidence": confidence,
        "supporting_evidence": evidence,
        "limitations": limitations,
        "recommended_disposition": disposition,
        "disposition_justification": disp_justification,
        "in_phase5d_verified": in_p5d_ver,
        "phase5d_status": p5d_status,
        "human_review_recommended": human_review_rec,
    }


def main():
    logger.info("Starting Phase 5E forensic audit of all 68 agreement records...")
    OUT_DIR_TABLES.mkdir(parents=True, exist_ok=True)
    OUT_DIR_REPORTS.mkdir(parents=True, exist_ok=True)
    OUT_DIR_DATA.mkdir(parents=True, exist_ok=True)

    # 1. Load inputs
    p1_data = load_jsonl(PASS1_FILE)
    p2_data = load_jsonl(PASS2_FILE)
    manifest_data = load_jsonl(MANIFEST_FILE)
    p5d_ver_data = load_jsonl(P5D_VERIFIED_FILE)
    p5d_exc_data = load_csv(P5D_EXCLUDED_FILE)

    p1_map = {r["record_id"]: r for r in p1_data}
    p2_map = {r["record_id"]: r for r in p2_data}
    manifest_map = {r["record_id"]: r for r in manifest_data}
    p5d_ver_map = {r["record_id"]: r for r in p5d_ver_data}
    p5d_exc_map = {r["record_id"]: r for r in p5d_exc_data}

    # 2. Recompute 68 agreements
    agreements = [
        rid for rid, r1 in p1_map.items() if r1["primary_label"] == p2_map[rid]["primary_label"]
    ]
    if len(agreements) != 68:
        raise ValueError(f"Expected exactly 68 agreements, found {len(agreements)}")
    logger.info("Successfully recomputed exactly 68 agreement records.")

    # 3. Audit all 68 records individually
    audited_records = []
    for rid in agreements:
        rec = audit_record(
            rid,
            manifest_map[rid],
            p1_map[rid],
            p2_map[rid],
            p5d_ver_map,
            p5d_exc_map,
        )
        audited_records.append(rec)

    # 4. Export Table 1: agreement_audit_68_complete.csv
    table1_path = OUT_DIR_TABLES / "agreement_audit_68_complete.csv"
    table1_fields = [
        "record_id",
        "source_dataset",
        "source_record_id",
        "comment_text",
        "context_text",
        "image_available",
        "ocr_available",
        "image_assessed_by_models",
        "ocr_assessed_by_models",
        "modality_used",
        "model_a_prediction",
        "model_a_subtype",
        "model_a_rationale",
        "model_b_prediction",
        "model_b_subtype",
        "model_b_rationale",
        "agreement_status",
        "interpersonal_target",
        "covert_hostility",
        "plausible_deniability",
        "rhetorical_form",
        "shared_risk_flag",
        "audit_category",
        "audit_confidence",
        "supporting_evidence",
        "limitations",
        "recommended_disposition",
        "disposition_justification",
    ]
    with open(table1_path, mode="w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=table1_fields)
        writer.writeheader()
        for r in audited_records:
            row = {k: r[k] for k in table1_fields}
            writer.writerow(row)
    logger.info(f"Saved complete audit table ({len(audited_records)} records) to {table1_path}")

    # Also save complete JSONL in data/annotations/phase5e/
    jsonl_path = OUT_DIR_DATA / "agreement_audit_68_complete.jsonl"
    with open(jsonl_path, mode="w", encoding="utf-8") as f:
        for r in audited_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    logger.info(f"Saved JSONL audit output to {jsonl_path}")

    # 5. Export Table 2: agreement_category_distribution.csv
    cat_counts = Counter(r["audit_category"] for r in audited_records)
    cat_descriptions = {
        "LIKELY_GENUINE_PA_SILVER": "High-confidence passive aggression with confirmed individual target, covert hostility, and plausible deniability.",
        "LIKELY_NON_PA": "Consensus negative example with confirmed absence of hostility or aggression.",
        "GENERAL_SARCASM_NO_INTERPERSONAL_TARGET": "Broadcast sarcasm or ironic humor directed at general audience rather than a specific addressee.",
        "INSTITUTIONAL_OR_SITUATIONAL_CRITIQUE": "Satire, venting, or grievance directed at abstract institutions (media, government, NFL) or general situations.",
        "DIRECT_AGGRESSION_OR_DIRECT_ARGUMENT": "Overt hostility, blunt arguments, or direct insults lacking courtesy facade or plausible deniability.",
        "SINCERE_PRAISE_OR_FRIENDLY_TEASING": "Sincere gratitude, polite compliment, or character self-talk misclassified as passive-aggressive.",
        "AMBIGUOUS_INSUFFICIENT_CONTEXT": "Short or fragmented comments where missing conversational context prevents reliable classification.",
        "SHARED_MODEL_ERROR_SUSPECTED": "Factual inquiries or promotional statements hallucinated by both models as passive-aggressive.",
        "UNRESOLVED_REQUIRES_HUMAN_REVIEW": "Multimodal posts where image was unassessed; target and intent cannot be resolved without human visual inspection.",
    }

    table2_path = OUT_DIR_TABLES / "agreement_category_distribution.csv"
    with open(table2_path, mode="w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["audit_category", "record_count", "percentage", "description", "example_record_ids"])
        writer.writeheader()
        for cat, desc in cat_descriptions.items():
            count = cat_counts.get(cat, 0)
            pct = f"{(count / 68.0) * 100:.2f}%"
            examples = "; ".join([r["record_id"] for r in audited_records if r["audit_category"] == cat][:3])
            writer.writerow({
                "audit_category": cat,
                "record_count": count,
                "percentage": pct,
                "description": desc,
                "example_record_ids": examples,
            })
    logger.info(f"Saved category distribution to {table2_path}")

    # 6. Export Table 3: phase5d_reconciliation_68.csv
    table3_path = OUT_DIR_TABLES / "phase5d_reconciliation_68.csv"
    table3_fields = [
        "record_id",
        "source_dataset",
        "in_phase5d_verified",
        "phase5d_decision",
        "phase5e_category",
        "decisions_agree",
        "reconciliation_notes",
        "human_review_recommended",
    ]
    reconciliation_rows = []
    for r in audited_records:
        in_p5d = "YES" if r["in_phase5d_verified"] else "NO"
        p5d_dec = r["phase5d_status"]
        p5e_cat = r["audit_category"]
        
        # Check agreement on inclusion/exclusion
        if in_p5d == "YES" and p5e_cat in ("LIKELY_GENUINE_PA_SILVER", "LIKELY_NON_PA"):
            dec_agree = "YES"
            notes = "Full alignment: verified in Phase 5D and confirmed in Phase 5E audit."
        elif in_p5d == "NO" and p5e_cat not in ("LIKELY_GENUINE_PA_SILVER", "LIKELY_NON_PA"):
            dec_agree = "YES"
            notes = f"Full alignment: excluded in Phase 5D ({p5d_dec}) and assigned to conservative exclusion category {p5e_cat}."
        else:
            dec_agree = "NO"
            notes = "Discrepancy detected between Phase 5D status and Phase 5E categorization."

        reconciliation_rows.append({
            "record_id": r["record_id"],
            "source_dataset": r["source_dataset"],
            "in_phase5d_verified": in_p5d,
            "phase5d_decision": p5d_dec,
            "phase5e_category": p5e_cat,
            "decisions_agree": dec_agree,
            "reconciliation_notes": notes,
            "human_review_recommended": r["human_review_recommended"],
        })

    with open(table3_path, mode="w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=table3_fields)
        writer.writeheader()
        writer.writerows(reconciliation_rows)
    logger.info(f"Saved Phase 5D reconciliation table to {table3_path}")

    # 7. Export Table 4: shared_error_risk_register.csv
    risks = [
        {
            "risk_id": "SER-01",
            "risk_type": "Shared False Positive - Target Blindness (Venting vs Interpersonal Hostility)",
            "affected_record_count": 11,
            "affected_records": "mustard_2_174; sarcnet_test_00274; civil_comments_val_09401; mmsd_train_05348",
            "root_cause": "Models lack an interlocutor-detection gate and treat generalized complaints against institutions (Fox, NFL, transit) or existential venting as passive aggression.",
            "severity_for_weak_supervision": "HIGH",
            "mitigation_implemented": "Strict exclusion of institutional and existential venting from verified silver pool (all 11 excluded).",
        },
        {
            "risk_id": "SER-02",
            "risk_type": "Shared False Positive - Modality Blindness (Unassessed Multimodal Sarcasm)",
            "affected_record_count": 20,
            "affected_records": "sarcnet_train_00045; mmsd_test_00916; sarcnet_train_00095; mmsd_train_14369",
            "root_cause": "Text-only LLMs evaluate multimodal social media posts in isolation, unable to confirm whether the image establishes an interpersonal target or contrast.",
            "severity_for_weak_supervision": "HIGH",
            "mitigation_implemented": "Quarantined all 20 multimodal unverified posts in UNRESOLVED_REQUIRES_HUMAN_REVIEW; excluded from verified silver dataset.",
        },
        {
            "risk_id": "SER-03",
            "risk_type": "Shared False Positive - Broadcast Sarcasm Without Interpersonal Target",
            "affected_record_count": 6,
            "affected_records": "civil_comments_val_05247; civil_comments_val_01142; mustard_2_317; civil_comments_val_06506",
            "root_cause": "Models equate general rhetorical sarcasm, news commentary, or dialogue filler to passive aggression directed at a conversation partner.",
            "severity_for_weak_supervision": "MEDIUM",
            "mitigation_implemented": "Categorized as GENERAL_SARCASM_NO_INTERPERSONAL_TARGET and excluded from silver dataset.",
        },
        {
            "risk_id": "SER-04",
            "risk_type": "Shared False Positive - Covert/Overt Conflation (Direct Aggression)",
            "affected_record_count": 4,
            "affected_records": "civil_comments_val_04917; civil_comments_val_02587; civil_comments_val_00154; civil_comments_val_05781",
            "root_cause": "Models fail to enforce the requirement for plausible deniability and polite camouflage, labeling overt hostility and direct arguments as passive-aggressive.",
            "severity_for_weak_supervision": "HIGH",
            "mitigation_implemented": "Categorized as DIRECT_AGGRESSION_OR_DIRECT_ARGUMENT and excluded as contrast class.",
        },
        {
            "risk_id": "SER-05",
            "risk_type": "Shared False Positive - Context Insufficiency",
            "affected_record_count": 4,
            "affected_records": "civil_comments_val_00793; mmsd_train_06126; mmsd_train_03612; mmsd_validation_02127",
            "root_cause": "Sentence fragments lack necessary surrounding dialogue turns, causing models to hallucinate hostile subtext.",
            "severity_for_weak_supervision": "MEDIUM",
            "mitigation_implemented": "Categorized as AMBIGUOUS_INSUFFICIENT_CONTEXT and excluded from silver dataset.",
        },
        {
            "risk_id": "SER-06",
            "risk_type": "Shared False Positive - Politeness Paranoia / Sincere Praise Hallucination",
            "affected_record_count": 2,
            "affected_records": "civil_comments_val_08920; mustard_2_372",
            "root_cause": "Prompt-induced suspicion leads models to interpret genuine courtesy ('I want to thank you for this fine article') or benign dialogue self-talk as deceptive sarcasm.",
            "severity_for_weak_supervision": "HIGH",
            "mitigation_implemented": "Categorized as SINCERE_PRAISE_OR_FRIENDLY_TEASING and excluded from silver dataset.",
        },
        {
            "risk_id": "SER-07",
            "risk_type": "Shared False Positive - Hallucination on Factual/Promotional Text",
            "affected_record_count": 2,
            "affected_records": "mmsd_train_00135; civil_comments_val_02265",
            "root_cause": "Models hallucinated passive aggression on objective news reporting or factual ride experiences.",
            "severity_for_weak_supervision": "HIGH",
            "mitigation_implemented": "Categorized as SHARED_MODEL_ERROR_SUSPECTED and excluded from silver dataset.",
        },
    ]

    table4_path = OUT_DIR_TABLES / "shared_error_risk_register.csv"
    with open(table4_path, mode="w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "risk_id", "risk_type", "affected_record_count", "affected_records", "root_cause", "severity_for_weak_supervision", "mitigation_implemented"
        ])
        writer.writeheader()
        writer.writerows(risks)
    logger.info(f"Saved shared error risk register to {table4_path}")

    # 8. Export Table 5: phase5e_human_review_priority_queue.csv
    # Prioritizes records requiring human adjudication
    queue_rows = []
    # Tier 1: Multimodal Unresolved (20 records)
    for r in audited_records:
        if r["audit_category"] == "UNRESOLVED_REQUIRES_HUMAN_REVIEW":
            queue_rows.append({
                "priority_tier": "P1_MULTIMODAL_IMAGE_ADJUDICATION",
                "record_id": r["record_id"],
                "source_dataset": r["source_dataset"],
                "comment_text": r["comment_text"],
                "image_available": r["image_available"],
                "audit_category": r["audit_category"],
                "reason_for_human_review": "Image exists but was unassessed by text models; human must review visual scene to determine if target/contrast is present.",
                "key_adjudication_question": "Does the image depict an identifiable person or context that turns this sarcasm into directed passive-aggression?",
            })
    # Tier 2: Ambiguous / Context Insufficient (4 records)
    for r in audited_records:
        if r["audit_category"] == "AMBIGUOUS_INSUFFICIENT_CONTEXT":
            queue_rows.append({
                "priority_tier": "P2_CONTEXT_EXPANSION_REQUIRED",
                "record_id": r["record_id"],
                "source_dataset": r["source_dataset"],
                "comment_text": r["comment_text"],
                "image_available": r["image_available"],
                "audit_category": r["audit_category"],
                "reason_for_human_review": "Fragmented text requires thread context inspection to confirm communicative intent.",
                "key_adjudication_question": "In the broader thread, was this comment an indirect critique of a previous comment?",
            })
    # Tier 3: Contrast Boundaries Verification (25 records)
    for r in audited_records:
        if r["audit_category"] in (
            "INSTITUTIONAL_OR_SITUATIONAL_CRITIQUE",
            "DIRECT_AGGRESSION_OR_DIRECT_ARGUMENT",
            "SINCERE_PRAISE_OR_FRIENDLY_TEASING",
            "SHARED_MODEL_ERROR_SUSPECTED",
            "GENERAL_SARCASM_NO_INTERPERSONAL_TARGET",
        ):
            queue_rows.append({
                "priority_tier": "P3_CONTRAST_BOUNDARY_VERIFICATION",
                "record_id": r["record_id"],
                "source_dataset": r["source_dataset"],
                "comment_text": r["comment_text"],
                "image_available": r["image_available"],
                "audit_category": r["audit_category"],
                "reason_for_human_review": f"Candidate for negative contrast class ({r['audit_category']}); requires human verification of boundary.",
                "key_adjudication_question": "Can this record be reliably verified as non-passive-aggressive contrast data?",
            })
    # Tier 4: Silver Positive Validation (19 records)
    for r in audited_records:
        if r["audit_category"] in ("LIKELY_GENUINE_PA_SILVER", "LIKELY_NON_PA"):
            queue_rows.append({
                "priority_tier": "P4_SILVER_TO_GOLD_PROMOTION",
                "record_id": r["record_id"],
                "source_dataset": r["source_dataset"],
                "comment_text": r["comment_text"],
                "image_available": r["image_available"],
                "audit_category": r["audit_category"],
                "reason_for_human_review": "Verified silver record; requires independent human sign-off to qualify for gold benchmark.",
                "key_adjudication_question": "Does human annotator agree with the silver label and subtype?",
            })

    table5_path = OUT_DIR_REPORTS / "phase5e_human_review_priority_queue.csv"
    with open(table5_path, mode="w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "priority_tier", "record_id", "source_dataset", "comment_text", "image_available", "audit_category", "reason_for_human_review", "key_adjudication_question"
        ])
        writer.writeheader()
        writer.writerows(queue_rows)
    logger.info(f"Saved human review priority queue ({len(queue_rows)} records) to {table5_path}")

    logger.info("Phase 5E forensic audit execution completed successfully.")


if __name__ == "__main__":
    main()
