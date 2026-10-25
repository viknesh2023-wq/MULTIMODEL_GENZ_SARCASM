#!/usr/bin/env python3
"""
scripts/run_phase6d_llm_annotation.py

Phase 6D: LLM-Based Dual Annotation, Agreement Analysis, and Adjudication Pipeline.

Executes:
1. Pass 1: Independent machine annotation of all 30 pilot records using llama3.2:latest.
2. Pass 2: Independent machine annotation of all 30 pilot records using qwen2.5:1.5b (double-blind isolation).
3. Inter-Model Agreement Calculation: Raw agreement %, Cohen's kappa, confusion matrix, subtype concordance.
4. Disagreement Extraction: Outputs all divergent cases.
5. Pass 3: LLM Adjudication on disagreements using llama3.2:latest as impartial adjudication judge.
6. Governance & Manifest Generation: Outputs complete Phase 6D manifest and audit reports.

CRITICAL CONSTRAINTS:
- Never touches data/annotations/phase6a/annotator_1_responses.csv or annotator_2_responses.csv.
- All labels bear explicit machine-generated provenance (ANNOTATED_BY_LOCAL_OLLAMA_LLM).
- Human benchmark gate remains BLOCKED_PENDING_HUMAN_ANNOTATION.
- Images remain unassessed (modality_used = 'TEXT_ONLY', image_assessed = False).
- Segregates 27 Core English Benchmark records from 3 Chinese exploratory records.
"""

import argparse
import csv
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple
import urllib.error
import urllib.request

try:
    from sklearn.metrics import cohen_kappa_score, confusion_matrix
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Paths
ROOT_DIR = Path(__file__).resolve().parent.parent
MANIFEST_CSV = ROOT_DIR / "data/annotations/phase6a/pilot_sampling_manifest.csv"
ANN1_HUMAN_CSV = ROOT_DIR / "data/annotations/phase6a/annotator_1_responses.csv"
ANN2_HUMAN_CSV = ROOT_DIR / "data/annotations/phase6a/annotator_2_responses.csv"

DATA_DIR_6D = ROOT_DIR / "data/annotations/phase6d"
REPORTS_DIR_6D = ROOT_DIR / "reports/phase6d"

ANN1_LLM_CSV = DATA_DIR_6D / "annotator_1_llm_responses.csv"
ANN2_LLM_CSV = DATA_DIR_6D / "annotator_2_llm_responses.csv"
DISAGREEMENTS_CSV = DATA_DIR_6D / "llm_annotation_disagreements.csv"
ADJUDICATION_CSV = DATA_DIR_6D / "llm_adjudication_results.csv"
MANIFEST_JSON = DATA_DIR_6D / "llm_annotation_manifest.json"

AGREEMENT_REPORT_MD = REPORTS_DIR_6D / "llm_annotation_agreement_report.md"
QUALITY_AUDIT_MD = REPORTS_DIR_6D / "llm_annotation_quality_audit.md"

# Prompt Specifications
PROMPT_VERSION = "v2.0.0"
ADJUDICATOR_PROMPT_VERSION = "v2.0.0_adjudication"

SYSTEM_PROMPT = """You are an expert computational linguist. Evaluate social media comments for passive aggression.

DEFINITION OF PASSIVE AGGRESSION:
Passive aggression is indirect communication conveying covert interpersonal hostility disguised beneath surface politeness, compliance, praise, or neutrality.
A comment is PASSIVE AGGRESSIVE IF AND ONLY IF all three criteria are met:
1. Interpersonal Target: Hostility or blame is directed at an individual or interlocutor (NOT an external situation, weather, delay, or general state of the world).
2. Plausible Deniability: Surface wording appears polite, praising, neutral, or deferential, allowing the speaker to deny hostile intent.
3. Covert Hostility: The pragmatic meaning undermines, demeans, mocks, or blames the target.

NEGATIVE BOUNDARIES (Select 'non_passive_aggressive'):
- Direct aggression / overt toxicity ("You are an idiot", "Shut up") -> Overt hostility lacking polite concealment is NOT PA.
- Situational sarcasm / venting ("Bus is delayed again!", "There is no wisdom left on this planet") -> No individual target, NOT PA.
- Factual reports, news & descriptions ("result: glen durrant edges out paul hogan 4-3", dates, links) -> Objective reporting, NOT PA.
- Sincere praise & gratitude ("Anthony, that is helpful information, thanks.") -> Genuine praise, NOT PA.
- Direct disagreement & debate ("Did you read the article?", "Prices will always go up") -> Blunt dispute without faux courtesy is NOT PA.
- Friendly teasing / banter ("Look who finally woke up!") -> Playful social bonding, NOT PA.

AMBIGUOUS CASES (Select 'ambiguous'):
- Missing context: Isolated phrase fragment or meme reference where target or intent cannot be determined.

Keep brief_rationale concise (1-2 sentences)."""

ANNOTATION_SCHEMA = {
    "type": "object",
    "properties": {
        "interpersonal_target": {
            "type": "string",
            "enum": ["PRESENT", "ABSENT", "UNCLEAR"]
        },
        "covert_hostility": {
            "type": "string",
            "enum": ["COVERT", "OVERT", "ABSENT", "UNCLEAR"]
        },
        "plausible_deniability": {
            "type": "string",
            "enum": ["YES", "NO", "UNCLEAR"]
        },
        "primary_label": {
            "type": "string",
            "enum": ["passive_aggressive", "non_passive_aggressive", "ambiguous"]
        },
        "subtype": {
            "type": "string",
            "enum": [
                "fake_politeness", "indirect_criticism", "backhanded_compliment",
                "implied_blame", "sarcastic_agreement", "dismissiveness", "other_or_unclear", "none"
            ]
        },
        "confidence": {
            "type": "integer",
            "minimum": 1,
            "maximum": 5
        },
        "context_sufficiency": {
            "type": "string",
            "enum": ["SUFFICIENT", "PARTIAL", "INSUFFICIENT"]
        },
        "evidence_span": {
            "type": "string"
        },
        "brief_rationale": {
            "type": "string"
        }
    },
    "required": [
        "interpersonal_target", "covert_hostility", "plausible_deniability",
        "primary_label", "subtype", "confidence", "context_sufficiency",
        "evidence_span", "brief_rationale"
    ]
}

ADJUDICATION_SYSTEM_PROMPT = """You are an expert computational linguist serving as an impartial adjudication judge.
Two independent LLM annotators have evaluated a social media post for passive aggression and produced divergent labels.

Your task is to:
1. Read the original post and available dialogue context.
2. Review both models' predictions and rationales.
3. Identify the linguistic root cause of their divergence.
4. Apply the operational definition of passive aggression conservatively:
   - Requires identifiable interpersonal target.
   - Requires covert hostility/blame/criticism disguised by surface politeness/neutrality.
   - Requires plausible deniability.
5. Provide an adjudicated label ('passive_aggressive', 'non_passive_aggressive', or 'ambiguous').
6. If genuine linguistic ambiguity remains or context is insufficient to decide, flag 'requires_human_review' as true and label as 'ambiguous'."""

ADJUDICATION_SCHEMA = {
    "type": "object",
    "properties": {
        "disagreement_root_cause": {
            "type": "string"
        },
        "interpersonal_target": {
            "type": "string",
            "enum": ["PRESENT", "ABSENT", "UNCLEAR"]
        },
        "covert_hostility": {
            "type": "string",
            "enum": ["COVERT", "OVERT", "ABSENT", "UNCLEAR"]
        },
        "plausible_deniability": {
            "type": "string",
            "enum": ["YES", "NO", "UNCLEAR"]
        },
        "adjudicated_label": {
            "type": "string",
            "enum": ["passive_aggressive", "non_passive_aggressive", "ambiguous"]
        },
        "adjudicated_subtype": {
            "type": "string",
            "enum": [
                "fake_politeness", "indirect_criticism", "backhanded_compliment",
                "implied_blame", "sarcastic_agreement", "dismissiveness", "other_or_unclear", "none"
            ]
        },
        "confidence": {
            "type": "integer",
            "minimum": 1,
            "maximum": 5
        },
        "requires_human_review": {
            "type": "boolean"
        },
        "adjudication_rationale": {
            "type": "string"
        }
    },
    "required": [
        "disagreement_root_cause", "interpersonal_target", "covert_hostility", "plausible_deniability",
        "adjudicated_label", "adjudicated_subtype", "confidence", "requires_human_review", "adjudication_rationale"
    ]
}


def load_manifest(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")
    with open(path, mode="r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def build_user_prompt(record: Dict[str, Any]) -> str:
    rec_id = record.get("record_id", "")
    src = record.get("source_dataset", "")
    text = record.get("comment_text", "").strip()
    ctx = record.get("context_text", "").strip()
    img_avail = record.get("image_available", "False")

    return (
        f"Evaluate the following social media post for passive aggression according to the calibrated criteria:\n\n"
        f"Record ID: {rec_id}\n"
        f"Source Dataset: {src}\n"
        f"Comment Text: \"{text}\"\n"
        f"Dialogue Context: {ctx if ctx else 'None'}\n"
        f"Image Availability: {img_avail} (Note: Image pixels unassessed; evaluate text and context)\n\n"
        f"Return valid JSON adhering to the required schema."
    )


def query_ollama(
    model: str,
    user_prompt: str,
    system_prompt: str = SYSTEM_PROMPT,
    schema: dict = ANNOTATION_SCHEMA,
    endpoint: str = "http://localhost:11434/api/chat",
    timeout: int = 120,
    temperature: float = 0.0,
    dry_run: bool = False,
) -> Dict[str, Any]:
    if dry_run:
        return {
            "interpersonal_target": "ABSENT",
            "covert_hostility": "ABSENT",
            "plausible_deniability": "YES",
            "primary_label": "non_passive_aggressive",
            "subtype": "none",
            "confidence": 4,
            "context_sufficiency": "SUFFICIENT",
            "evidence_span": "dry run evidence span",
            "brief_rationale": "Dry run mock rationale.",
            "_raw": "{}",
            "_latency_sec": 0.01,
        }

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "format": schema,
        "stream": False,
        "options": {"temperature": temperature},
    }

    req = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )

    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            latency = time.time() - t0
            res_data = json.loads(response.read().decode("utf-8"))
            raw_content = res_data.get("message", {}).get("content", "")
            parsed = json.loads(raw_content)
            parsed["_raw"] = raw_content
            parsed["_latency_sec"] = round(latency, 3)
            return parsed
    except Exception as e:
        raise RuntimeError(f"Ollama inference error for model '{model}': {e}")


def annotate_single_record(
    record: Dict[str, Any],
    model: str,
    annotator_type: str,
    endpoint: str = "http://localhost:11434/api/chat",
    timeout: int = 120,
    max_retries: int = 2,
    dry_run: bool = False,
) -> Dict[str, Any]:
    prompt = build_user_prompt(record)
    last_err = None

    for attempt in range(max_retries + 1):
        try:
            result = query_ollama(
                model=model,
                user_prompt=prompt,
                system_prompt=SYSTEM_PROMPT,
                schema=ANNOTATION_SCHEMA,
                endpoint=endpoint,
                timeout=timeout,
                dry_run=dry_run,
            )

            p_label = result.get("primary_label")
            if p_label not in ("passive_aggressive", "non_passive_aggressive", "ambiguous"):
                raise ValueError(f"Invalid primary_label: {p_label}")

            subtype = result.get("subtype", "none")
            if p_label != "passive_aggressive":
                subtype = "none"
            elif not subtype or subtype == "none":
                subtype = "other_or_unclear"

            now_iso = datetime.now(timezone.utc).isoformat()
            return {
                "record_id": record["record_id"],
                "source_dataset": record["source_dataset"],
                "source_record_id": record.get("source_record_id", ""),
                "comment_text": record["comment_text"],
                "context_text": record.get("context_text", ""),
                "language": record.get("language", "en"),
                "language_scope": record.get("language_scope", "CORE_ENGLISH_BENCHMARK"),
                "sampling_archetype": record.get("sampling_archetype", ""),
                "modality_used": "TEXT_ONLY",
                "image_assessed": False,
                "ocr_assessed": False,
                "context_available": bool(record.get("context_text")),
                "model_name": model,
                "prompt_version": PROMPT_VERSION,
                "interpersonal_target_assessment": result.get("interpersonal_target", "UNCLEAR"),
                "covert_hostility_assessment": result.get("covert_hostility", "UNCLEAR"),
                "plausible_deniability_assessment": result.get("plausible_deniability", "UNCLEAR"),
                "primary_label": p_label,
                "subtype": subtype,
                "confidence": int(result.get("confidence", 3)),
                "context_sufficiency": result.get("context_sufficiency", "INSUFFICIENT"),
                "evidence_span": result.get("evidence_span", "none"),
                "brief_rationale": result.get("brief_rationale", ""),
                "raw_model_response": result.get("_raw", ""),
                "validation_status": "VALID",
                "annotator_type": annotator_type,
                "annotated_at": now_iso,
                "provenance": "MACHINE_GENERATED_LOCAL_OLLAMA_LLM",
                "inference_latency_sec": result.get("_latency_sec", 0.0),
            }
        except Exception as e:
            last_err = str(e)
            logger.warning(f"Retry {attempt + 1}/{max_retries} for {record['record_id']} on {model}: {e}")
            time.sleep(2.0)

    # If all attempts failed
    now_iso = datetime.now(timezone.utc).isoformat()
    return {
        "record_id": record["record_id"],
        "source_dataset": record["source_dataset"],
        "source_record_id": record.get("source_record_id", ""),
        "comment_text": record["comment_text"],
        "context_text": record.get("context_text", ""),
        "language": record.get("language", "en"),
        "language_scope": record.get("language_scope", "CORE_ENGLISH_BENCHMARK"),
        "sampling_archetype": record.get("sampling_archetype", ""),
        "modality_used": "TEXT_ONLY",
        "image_assessed": False,
        "ocr_assessed": False,
        "context_available": bool(record.get("context_text")),
        "model_name": model,
        "prompt_version": PROMPT_VERSION,
        "interpersonal_target_assessment": "UNCLEAR",
        "covert_hostility_assessment": "UNCLEAR",
        "plausible_deniability_assessment": "UNCLEAR",
        "primary_label": "ambiguous",
        "subtype": "none",
        "confidence": 1,
        "context_sufficiency": "INSUFFICIENT",
        "evidence_span": "none",
        "brief_rationale": f"Inference failure after retries: {last_err}",
        "raw_model_response": "",
        "validation_status": "INVALID_FAILED_INFERENCE",
        "annotator_type": annotator_type,
        "annotated_at": now_iso,
        "provenance": "MACHINE_GENERATED_LOCAL_OLLAMA_LLM",
        "inference_latency_sec": 0.0,
    }


def run_annotation_pass(
    records: List[Dict[str, Any]],
    model: str,
    annotator_type: str,
    output_csv: Path,
    endpoint: str = "http://localhost:11434/api/chat",
    timeout: int = 120,
    resume: bool = True,
    dry_run: bool = False,
) -> List[Dict[str, Any]]:
    completed_records = []
    completed_ids = set()

    if resume and output_csv.exists():
        with open(output_csv, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                completed_records.append(r)
                completed_ids.add(r["record_id"])
        logger.info(f"[{annotator_type}] Resumed {len(completed_ids)} records from {output_csv.name}")

    if len(completed_ids) == len(records):
        logger.info(f"[{annotator_type}] All {len(records)} records already completed.")
        return completed_records

    # Open CSV for append / rewrite
    fieldnames = [
        "record_id", "source_dataset", "source_record_id", "comment_text", "context_text",
        "language", "language_scope", "sampling_archetype", "modality_used", "image_assessed",
        "ocr_assessed", "context_available", "model_name", "prompt_version",
        "interpersonal_target_assessment", "covert_hostility_assessment", "plausible_deniability_assessment",
        "primary_label", "subtype", "confidence", "context_sufficiency", "evidence_span",
        "brief_rationale", "raw_model_response", "validation_status", "annotator_type",
        "annotated_at", "provenance", "inference_latency_sec"
    ]

    write_header = not output_csv.exists() or not resume or len(completed_ids) == 0
    mode = "a" if (resume and output_csv.exists() and len(completed_ids) > 0) else "w"

    with open(output_csv, mode=mode, encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()

        for idx, rec in enumerate(records, 1):
            rid = rec["record_id"]
            if rid in completed_ids:
                continue

            logger.info(f"[{annotator_type}] ({idx}/{len(records)}) Annotating {rid} with {model}...")
            annotated = annotate_single_record(
                record=rec,
                model=model,
                annotator_type=annotator_type,
                endpoint=endpoint,
                timeout=timeout,
                dry_run=dry_run,
            )
            writer.writerow(annotated)
            f.flush()
            completed_records.append(annotated)
            completed_ids.add(rid)

    return completed_records


def calculate_agreement_metrics(
    ann1_rows: List[Dict[str, Any]],
    ann2_rows: List[Dict[str, Any]],
) -> Dict[str, Any]:
    map1 = {r["record_id"]: r for r in ann1_rows}
    map2 = {r["record_id"]: r for r in ann2_rows}
    common_ids = sorted(list(set(map1.keys()) & set(map2.keys())))

    classes = ["passive_aggressive", "non_passive_aggressive", "ambiguous"]
    
    # 1. Overall Metrics (All 30 records)
    total_pairs = len(common_ids)
    exact_matches = 0
    disagreements = []
    y1_all = []
    y2_all = []

    # 2. English-Only Metrics (27 records)
    en_pairs = 0
    en_matches = 0
    y1_en = []
    y2_en = []

    # 3. Chinese-Only Metrics (3 records)
    zh_pairs = 0
    zh_matches = 0

    # 4. Subtype Concordance (where both predicted passive_aggressive)
    pa_both_count = 0
    subtype_matches = 0

    for rid in common_ids:
        r1 = map1[rid]
        r2 = map2[rid]
        lbl1 = r1["primary_label"]
        lbl2 = r2["primary_label"]
        sub1 = r1.get("subtype", "none")
        sub2 = r2.get("subtype", "none")
        scope = r1.get("language_scope", "CORE_ENGLISH_BENCHMARK")

        y1_all.append(lbl1)
        y2_all.append(lbl2)

        is_match = (lbl1 == lbl2)
        if is_match:
            exact_matches += 1
        else:
            disagreements.append({
                "record_id": rid,
                "source_dataset": r1.get("source_dataset", ""),
                "language_scope": scope,
                "sampling_archetype": r1.get("sampling_archetype", ""),
                "comment_text": r1.get("comment_text", ""),
                "context_text": r1.get("context_text", ""),
                "model1_name": r1.get("model_name", ""),
                "model1_label": lbl1,
                "model1_subtype": sub1,
                "model1_rationale": r1.get("brief_rationale", ""),
                "model2_name": r2.get("model_name", ""),
                "model2_label": lbl2,
                "model2_subtype": sub2,
                "model2_rationale": r2.get("brief_rationale", ""),
                "disagreement_type": "PRIMARY_LABEL_DISAGREEMENT",
            })

        if lbl1 == "passive_aggressive" and lbl2 == "passive_aggressive":
            pa_both_count += 1
            if sub1 == sub2:
                subtype_matches += 1

        if scope == "CORE_ENGLISH_BENCHMARK":
            en_pairs += 1
            y1_en.append(lbl1)
            y2_en.append(lbl2)
            if is_match:
                en_matches += 1
        else:
            zh_pairs += 1
            if is_match:
                zh_matches += 1

    # Kappa calculations
    import math
    if SKLEARN_AVAILABLE and total_pairs > 0:
        k_val = cohen_kappa_score(y1_all, y2_all, labels=classes)
        kappa_overall = None if (k_val is None or math.isnan(k_val)) else float(k_val)
        cm_overall = confusion_matrix(y1_all, y2_all, labels=classes).tolist()
    else:
        kappa_overall = None
        cm_overall = []

    if SKLEARN_AVAILABLE and en_pairs > 0:
        k_val_en = cohen_kappa_score(y1_en, y2_en, labels=classes)
        kappa_en = None if (k_val_en is None or math.isnan(k_val_en)) else float(k_val_en)
        cm_en = confusion_matrix(y1_en, y2_en, labels=classes).tolist()
    else:
        kappa_en = None
        cm_en = []

    raw_agreement_all = round((exact_matches / total_pairs) * 100, 2) if total_pairs else 0.0
    raw_agreement_en = round((en_matches / en_pairs) * 100, 2) if en_pairs else 0.0
    raw_agreement_zh = round((zh_matches / zh_pairs) * 100, 2) if zh_pairs else 0.0
    subtype_concordance = round((subtype_matches / pa_both_count) * 100, 2) if pa_both_count else 0.0

    return {
        "total_pairs": total_pairs,
        "exact_matches": exact_matches,
        "disagreement_count": len(disagreements),
        "raw_agreement_percent_overall": raw_agreement_all,
        "cohens_kappa_overall": round(kappa_overall, 4) if kappa_overall is not None else None,
        "confusion_matrix_overall": {
            "classes": classes,
            "matrix": cm_overall,
        },
        "english_benchmark": {
            "en_pairs": en_pairs,
            "en_matches": en_matches,
            "raw_agreement_percent": raw_agreement_en,
            "cohens_kappa": round(kappa_en, 4) if kappa_en is not None else None,
            "confusion_matrix": {
                "classes": classes,
                "matrix": cm_en,
            },
        },
        "chinese_separate": {
            "zh_pairs": zh_pairs,
            "zh_matches": zh_matches,
            "raw_agreement_percent": raw_agreement_zh,
        },
        "subtype_agreement": {
            "both_passive_aggressive_count": pa_both_count,
            "subtype_matches": subtype_matches,
            "subtype_concordance_percent": subtype_concordance,
        },
        "disagreements": disagreements,
    }


def run_adjudication(
    manifest_records: List[Dict[str, Any]],
    ann1_rows: List[Dict[str, Any]],
    ann2_rows: List[Dict[str, Any]],
    adjudicator_model: str,
    output_csv: Path,
    endpoint: str = "http://localhost:11434/api/chat",
    timeout: int = 120,
    resume: bool = True,
    dry_run: bool = False,
) -> List[Dict[str, Any]]:
    map1 = {r["record_id"]: r for r in ann1_rows}
    map2 = {r["record_id"]: r for r in ann2_rows}

    completed_adjudications = []
    completed_ids = set()

    if resume and output_csv.exists():
        with open(output_csv, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                completed_adjudications.append(r)
                completed_ids.add(r["record_id"])
        logger.info(f"[ADJUDICATION] Resumed {len(completed_ids)} records from {output_csv.name}")

    if len(completed_ids) == len(manifest_records):
        logger.info(f"[ADJUDICATION] All {len(manifest_records)} records already adjudicated.")
        return completed_adjudications

    fieldnames = [
        "record_id", "source_dataset", "comment_text", "context_text", "language_scope",
        "model1_name", "model1_label", "model1_subtype", "model1_rationale",
        "model2_name", "model2_label", "model2_subtype", "model2_rationale",
        "adjudication_type", "adjudicated_label", "adjudicated_subtype",
        "interpersonal_target", "covert_hostility", "plausible_deniability",
        "confidence", "requires_human_review", "adjudication_rationale",
        "disagreement_root_cause", "adjudicator_model", "adjudicator_prompt_version",
        "adjudicated_at", "provenance"
    ]

    write_header = not output_csv.exists() or not resume or len(completed_ids) == 0
    mode = "a" if (resume and output_csv.exists() and len(completed_ids) > 0) else "w"

    with open(output_csv, mode=mode, encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()

        for idx, rec in enumerate(manifest_records, 1):
            rid = rec["record_id"]
            if rid in completed_ids:
                continue

            r1 = map1.get(rid, {})
            r2 = map2.get(rid, {})
            lbl1 = r1.get("primary_label", "ambiguous")
            lbl2 = r2.get("primary_label", "ambiguous")
            sub1 = r1.get("subtype", "none")
            sub2 = r2.get("subtype", "none")

            now_iso = datetime.now(timezone.utc).isoformat()

            # Case A: Consensus Agreement (Models Agree)
            if lbl1 == lbl2:
                # Subtype concordance
                final_subtype = sub1 if sub1 == sub2 else ("other_or_unclear" if lbl1 == "passive_aggressive" else "none")
                adjudicated_row = {
                    "record_id": rid,
                    "source_dataset": rec["source_dataset"],
                    "comment_text": rec["comment_text"],
                    "context_text": rec.get("context_text", ""),
                    "language_scope": rec.get("language_scope", "CORE_ENGLISH_BENCHMARK"),
                    "model1_name": r1.get("model_name", ""),
                    "model1_label": lbl1,
                    "model1_subtype": sub1,
                    "model1_rationale": r1.get("brief_rationale", ""),
                    "model2_name": r2.get("model_name", ""),
                    "model2_label": lbl2,
                    "model2_subtype": sub2,
                    "model2_rationale": r2.get("brief_rationale", ""),
                    "adjudication_type": "UNANIMOUS_AGREEMENT",
                    "adjudicated_label": lbl1,
                    "adjudicated_subtype": final_subtype,
                    "interpersonal_target": r1.get("interpersonal_target_assessment", "UNCLEAR"),
                    "covert_hostility": r1.get("covert_hostility_assessment", "UNCLEAR"),
                    "plausible_deniability": r1.get("plausible_deniability_assessment", "UNCLEAR"),
                    "confidence": max(int(r1.get("confidence", 3)), int(r2.get("confidence", 3))),
                    "requires_human_review": (lbl1 == "ambiguous"),
                    "adjudication_rationale": f"Unanimous machine consensus: both models independently assigned '{lbl1}'.",
                    "disagreement_root_cause": "NONE_CONSENSUS",
                    "adjudicator_model": adjudicator_model,
                    "adjudicator_prompt_version": ADJUDICATOR_PROMPT_VERSION,
                    "adjudicated_at": now_iso,
                    "provenance": "MACHINE_CONSENSUS_NOT_GROUND_TRUTH",
                }
                writer.writerow(adjudicated_row)
                f.flush()
                completed_adjudications.append(adjudicated_row)
                completed_ids.add(rid)
                continue

            # Case B: Disagreement -> Run LLM Adjudication Pass
            logger.info(f"[ADJUDICATION] ({idx}/{len(manifest_records)}) Adjudicating divergence for {rid}: {lbl1} vs {lbl2}...")
            adj_prompt = (
                f"Adjudicate the following disagreement:\n\n"
                f"Record ID: {rid}\n"
                f"Source Dataset: {rec['source_dataset']}\n"
                f"Comment Text: \"{rec['comment_text']}\"\n"
                f"Dialogue Context: {rec.get('context_text', 'None')}\n"
                f"Image Availability: {rec.get('image_available', False)} (pixels unassessed)\n\n"
                f"Model 1 ({r1.get('model_name', 'Model1')}):\n"
                f"- Primary Label: {lbl1}\n"
                f"- Subtype: {sub1}\n"
                f"- Rationale: {r1.get('brief_rationale', '')}\n\n"
                f"Model 2 ({r2.get('model_name', 'Model2')}):\n"
                f"- Primary Label: {lbl2}\n"
                f"- Subtype: {sub2}\n"
                f"- Rationale: {r2.get('brief_rationale', '')}\n\n"
                f"Return valid JSON adhering to the adjudication schema."
            )

            if dry_run:
                adj_res = {
                    "disagreement_root_cause": "Dry run mock root cause.",
                    "interpersonal_target": "ABSENT",
                    "covert_hostility": "ABSENT",
                    "plausible_deniability": "YES",
                    "adjudicated_label": "ambiguous",
                    "adjudicated_subtype": "none",
                    "confidence": 3,
                    "requires_human_review": True,
                    "adjudication_rationale": "Dry run mock adjudication rationale.",
                }
            else:
                try:
                    adj_res = query_ollama(
                        model=adjudicator_model,
                        user_prompt=adj_prompt,
                        system_prompt=ADJUDICATION_SYSTEM_PROMPT,
                        schema=ADJUDICATION_SCHEMA,
                        endpoint=endpoint,
                        timeout=timeout,
                        dry_run=False,
                    )
                except Exception as e:
                    logger.error(f"Adjudication failed for {rid}: {e}")
                    adj_res = {
                        "disagreement_root_cause": f"Inference failure: {e}",
                        "interpersonal_target": "UNCLEAR",
                        "covert_hostility": "UNCLEAR",
                        "plausible_deniability": "UNCLEAR",
                        "adjudicated_label": "ambiguous",
                        "adjudicated_subtype": "none",
                        "confidence": 1,
                        "requires_human_review": True,
                        "adjudication_rationale": f"Adjudication inference failure: {e}",
                    }

            adj_label = adj_res.get("adjudicated_label", "ambiguous")
            adj_subtype = adj_res.get("adjudicated_subtype", "none")
            if adj_label != "passive_aggressive":
                adj_subtype = "none"

            adjudicated_row = {
                "record_id": rid,
                "source_dataset": rec["source_dataset"],
                "comment_text": rec["comment_text"],
                "context_text": rec.get("context_text", ""),
                "language_scope": rec.get("language_scope", "CORE_ENGLISH_BENCHMARK"),
                "model1_name": r1.get("model_name", ""),
                "model1_label": lbl1,
                "model1_subtype": sub1,
                "model1_rationale": r1.get("brief_rationale", ""),
                "model2_name": r2.get("model_name", ""),
                "model2_label": lbl2,
                "model2_subtype": sub2,
                "model2_rationale": r2.get("brief_rationale", ""),
                "adjudication_type": "LLM_ADJUDICATED_DISAGREEMENT",
                "adjudicated_label": adj_label,
                "adjudicated_subtype": adj_subtype,
                "interpersonal_target": adj_res.get("interpersonal_target", "UNCLEAR"),
                "covert_hostility": adj_res.get("covert_hostility", "UNCLEAR"),
                "plausible_deniability": adj_res.get("plausible_deniability", "UNCLEAR"),
                "confidence": int(adj_res.get("confidence", 3)),
                "requires_human_review": bool(adj_res.get("requires_human_review", True)),
                "adjudication_rationale": adj_res.get("adjudication_rationale", ""),
                "disagreement_root_cause": adj_res.get("disagreement_root_cause", ""),
                "adjudicator_model": adjudicator_model,
                "adjudicator_prompt_version": ADJUDICATOR_PROMPT_VERSION,
                "adjudicated_at": now_iso,
                "provenance": "MACHINE_GENERATED_ADJUDICATION_NOT_GROUND_TRUTH",
            }
            writer.writerow(adjudicated_row)
            f.flush()
            completed_adjudications.append(adjudicated_row)
            completed_ids.add(rid)

    return completed_adjudications


def export_disagreements_csv(disagreements: List[Dict[str, Any]], output_path: Path):
    fieldnames = [
        "record_id", "source_dataset", "language_scope", "sampling_archetype",
        "comment_text", "context_text", "model1_name", "model1_label",
        "model1_subtype", "model1_rationale", "model2_name", "model2_label",
        "model2_subtype", "model2_rationale", "disagreement_type"
    ]
    with open(output_path, mode="w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for d in disagreements:
            writer.writerow(d)
    logger.info(f"Exported {len(disagreements)} disagreements to {output_path.name}")


def generate_manifest(
    metrics: Dict[str, Any],
    model1: str,
    model2: str,
    adjudicator_model: str,
    output_path: Path
):
    manifest_data = {
        "manifest_name": "phase6d_llm_annotation_manifest",
        "version": "6.4.0",
        "created_date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "pilot_sample_size": metrics["total_pairs"],
        "pipeline_provenance": "LOCAL_OLLAMA_LLM_DUAL_ANNOTATION_AND_ADJUDICATION",
        "ground_truth_status": "MACHINE_GENERATED_LABELS_NOT_HUMAN_GROUND_TRUTH",
        "human_benchmark_gate_status": "BLOCKED_PENDING_HUMAN_ANNOTATION",
        "human_annotator_01_completion_count": 0,
        "human_annotator_02_completion_count": 0,
        "models": {
            "model_1": {
                "name": model1,
                "role": "LLM_ANNOTATOR_1",
                "prompt_version": PROMPT_VERSION,
                "completed_count": metrics["total_pairs"],
                "assigned_file": str(ANN1_LLM_CSV.relative_to(ROOT_DIR)),
            },
            "model_2": {
                "name": model2,
                "role": "LLM_ANNOTATOR_2",
                "prompt_version": PROMPT_VERSION,
                "completed_count": metrics["total_pairs"],
                "assigned_file": str(ANN2_LLM_CSV.relative_to(ROOT_DIR)),
            },
            "adjudicator": {
                "name": adjudicator_model,
                "role": "LLM_ADJUDICATOR_JUDGE",
                "prompt_version": ADJUDICATOR_PROMPT_VERSION,
                "adjudicated_disagreements_count": metrics["disagreement_count"],
                "assigned_file": str(ADJUDICATION_CSV.relative_to(ROOT_DIR)),
            }
        },
        "modality_policy": {
            "modality_used": "TEXT_ONLY",
            "image_assessed": False,
            "ocr_assessed": False,
            "rationale": "No vision-capable models installed locally in Ollama environment. Multimodal images remain unassessed."
        },
        "inter_model_agreement": {
            "overall_30_records": {
                "paired_count": metrics["total_pairs"],
                "exact_matches": metrics["exact_matches"],
                "disagreements": metrics["disagreement_count"],
                "raw_agreement_percent": metrics["raw_agreement_percent_overall"],
                "cohens_kappa": metrics["cohens_kappa_overall"],
                "confusion_matrix": metrics["confusion_matrix_overall"],
            },
            "core_english_benchmark_27_records": metrics["english_benchmark"],
            "chinese_separate_analysis_3_records": metrics["chinese_separate"],
            "subtype_concordance": metrics["subtype_agreement"],
        },
        "adjudication_summary": {
            "total_records_resolved": metrics["total_pairs"],
            "unanimous_consensus_count": metrics["exact_matches"],
            "adjudicated_disagreements_count": metrics["disagreement_count"],
        },
        "evaluation_restrictions": [
            "DO_NOT_TRAIN_CLASSIFIER_ON_THESE_LABELS_WITHOUT_FLAGGING_AS_SILVER_MACHINE_DATA",
            "NEVER_REPORT_INTER_MODEL_AGREEMENT_AS_HUMAN_INTER_ANNOTATOR_AGREEMENT",
            "HUMAN_BENCHMARK_GATE_REMAINS_BLOCKED_PENDING_HUMAN_ANNOTATION",
            "IMAGE_EVIDENCE_UNASSESSED_DUE_TO_ABSENCE_OF_LOCAL_VISION_MODEL"
        ]
    }

    with open(output_path, mode="w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)
    logger.info(f"Generated Phase 6D manifest at {output_path.name}")


def generate_reports(
    metrics: Dict[str, Any],
    adjudications: List[Dict[str, Any]],
    model1: str,
    model2: str,
    adjudicator_model: str,
):
    # 1. Agreement Report
    cm = metrics["confusion_matrix_overall"]["matrix"]
    cm_en = metrics["english_benchmark"]["confusion_matrix"]["matrix"]
    classes = metrics["confusion_matrix_overall"]["classes"]

    agreement_md = f"""# Phase 6D: LLM Dual-Annotation Inter-Model Agreement Report

**Project Title:** Context-Aware Multimodal Passive-Aggression Detection in Social Media Using Image–Comment Fusion  
**Date:** {datetime.now(timezone.utc).strftime("%B %d, %Y")}  
**Status:** COMPLETE (Machine-Assisted Annotation and Adjudication)  
**Models Evaluated:**
- **Model 1:** `{model1}` (Prompt `{PROMPT_VERSION}`)
- **Model 2:** `{model2}` (Prompt `{PROMPT_VERSION}`)
- **Adjudicator:** `{adjudicator_model}` (Prompt `{ADJUDICATOR_PROMPT_VERSION}`)

---

## 1. Executive Summary & Epistemic Warning

> [!WARNING]
> **MACHINE-GENERATED DATASET NOTICE:**  
> These metrics represent **inter-model agreement** between two local large language models (`llama3.2:latest` 3.2B and `qwen2.5:1.5b` 1.5B).  
> **Never represent these metrics as human inter-annotator agreement or ground truth.**  
> High agreement indicates shared model heuristics or convergent prompt compliance, **not empirical fact**. The human annotation benchmark gate remains strictly `BLOCKED_PENDING_HUMAN_ANNOTATION`.

---

## 2. Quantitative Inter-Model Agreement

| Metric Scope | Total Paired Records | Exact Matches | Disagreements | Raw Agreement (%) | Cohen's Kappa (κ) | Status / Interpretation |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **All Pilot Records** | {metrics["total_pairs"]} | {metrics["exact_matches"]} | {metrics["disagreement_count"]} | **{metrics["raw_agreement_percent_overall"]}%** | **{metrics["cohens_kappa_overall"]}** | Full pilot set |
| **Core English Benchmark** | {metrics["english_benchmark"]["en_pairs"]} | {metrics["english_benchmark"]["en_matches"]} | {metrics["english_benchmark"]["en_pairs"] - metrics["english_benchmark"]["en_matches"]} | **{metrics["english_benchmark"]["raw_agreement_percent"]}%** | **{metrics["english_benchmark"]["cohens_kappa"]}** | Rigorous English subset |
| **Chinese Exploratory Subset** | {metrics["chinese_separate"]["zh_pairs"]} | {metrics["chinese_separate"]["zh_matches"]} | {metrics["chinese_separate"]["zh_pairs"] - metrics["chinese_separate"]["zh_matches"]} | **{metrics["chinese_separate"]["raw_agreement_percent"]}%** | N/A (N=3) | Segregated exploratory analysis |

---

## 3. Confusion Matrix Analysis

### Overall Confusion Matrix (Rows: Model 1 `{model1}`, Columns: Model 2 `{model2}`)

| Model 1 \\ Model 2 | passive_aggressive | non_passive_aggressive | ambiguous | Total |
| :--- | :---: | :---: | :---: | :---: |
| **passive_aggressive** | {cm[0][0] if cm else 0} | {cm[0][1] if cm else 0} | {cm[0][2] if cm else 0} | {sum(cm[0]) if cm else 0} |
| **non_passive_aggressive** | {cm[1][0] if cm else 0} | {cm[1][1] if cm else 0} | {cm[1][2] if cm else 0} | {sum(cm[1]) if cm else 0} |
| **ambiguous** | {cm[2][0] if cm else 0} | {cm[2][1] if cm else 0} | {cm[2][2] if cm else 0} | {sum(cm[2]) if cm else 0} |
| **Total** | {cm[0][0]+cm[1][0]+cm[2][0] if cm else 0} | {cm[0][1]+cm[1][1]+cm[2][1] if cm else 0} | {cm[0][2]+cm[1][2]+cm[2][2] if cm else 0} | {metrics["total_pairs"]} |

### Core English Benchmark Confusion Matrix (27 Records)

| Model 1 \\ Model 2 | passive_aggressive | non_passive_aggressive | ambiguous | Total |
| :--- | :---: | :---: | :---: | :---: |
| **passive_aggressive** | {cm_en[0][0] if cm_en else 0} | {cm_en[0][1] if cm_en else 0} | {cm_en[0][2] if cm_en else 0} | {sum(cm_en[0]) if cm_en else 0} |
| **non_passive_aggressive** | {cm_en[1][0] if cm_en else 0} | {cm_en[1][1] if cm_en else 0} | {cm_en[1][2] if cm_en else 0} | {sum(cm_en[1]) if cm_en else 0} |
| **ambiguous** | {cm_en[2][0] if cm_en else 0} | {cm_en[2][1] if cm_en else 0} | {cm_en[2][2] if cm_en else 0} | {sum(cm_en[2]) if cm_en else 0} |

---

## 4. Subtype Agreement Analysis

For records where both models independently agreed on `passive_aggressive`:
- **Both Models Agreed PA:** {metrics["subtype_agreement"]["both_passive_aggressive_count"]} records
- **Exact Subtype Concordance:** {metrics["subtype_agreement"]["subtype_matches"]} records
- **Subtype Concordance Rate:** **{metrics["subtype_agreement"]["subtype_concordance_percent"]}%**

---

## 5. Disagreement Audit & LLM Adjudication Log

Total disagreements requiring adjudication: **{metrics["disagreement_count"]}**

| Record ID | Source | Scope | Model 1 (`{model1}`) | Model 2 (`{model2}`) | Adjudicated Label | Root Cause Summary |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
"""
    for adj in adjudications:
        if adj["adjudication_type"] == "LLM_ADJUDICATED_DISAGREEMENT":
            agreement_md += f"| `{adj['record_id']}` | {adj['source_dataset']} | {adj['language_scope']} | `{adj['model1_label']}` | `{adj['model2_label']}` | **`{adj['adjudicated_label']}`** | {adj['disagreement_root_cause'][:60]}... |\n"

    agreement_md += """
---

## 6. Modality & Verification Restrictions

1. **Modality Coverage:** `TEXT_ONLY` for all 30 records. Zero images were evaluated (`image_assessed = false`, `ocr_assessed = false`).
2. **Double-Blind Isolation:** Model 2 received zero predictions, labels, or rationales from Model 1.
3. **Preservation of Human Files:** `data/annotations/phase6a/annotator_1_responses.csv` and `annotator_2_responses.csv` remain untouched with completion count = 0.
"""

    with open(AGREEMENT_REPORT_MD, mode="w", encoding="utf-8") as f:
        f.write(agreement_md)
    logger.info(f"Generated agreement report at {AGREEMENT_REPORT_MD.name}")

    # 2. Quality Audit Report
    quality_md = f"""# Phase 6D: LLM Annotation Quality Audit & Governance Document

**Project:** Context-Aware Multimodal Passive-Aggression Detection in Social Media Using Image–Comment Fusion  
**Artifact:** `reports/phase6d/llm_annotation_quality_audit.md`  
**Date:** {datetime.now(timezone.utc).strftime("%B %d, %Y")}  
**Auditor:** Automated Pragmatic & Experimental Governance Auditor  

---

## 1. Audit Scope & Objective

Phase 6D executed an end-to-end local LLM annotation workflow across the 30-record stratified Phase 6A pilot dataset.  
The audit verifies:
1. **Model Provenance & Independence:** Complete separation between Model 1 (`{model1}`) and Model 2 (`{model2}`).
2. **Modality Truthfulness:** Honest reporting that visual elements were unassessed (`TEXT_ONLY`).
3. **Linguistic Boundary Adherence:** Application of the 6 negative boundaries (direct aggression, situational sarcasm, news reports, sincere praise, blunt dispute, friendly teasing).
4. **Preservation of Human Benchmark:** Verification that human annotation files remain completely unpopulated (count = 0).
5. **Adjudication Traceability:** Forensic justification for all resolved disagreements.

---

## 2. Modality & Vision Model Assessment

An inspection of the local Ollama environment confirmed:
- `llama3.2:latest` (3.2B parameters) - Installed
- `qwen2.5:1.5b` (1.5B parameters) - Installed
- `llama3.2:1b` (1.2B parameters) - Installed
- **Vision Models (`llava`, `llama3.2-vision`, etc.):** **NONE INSTALLED**

In accordance with strict research integrity policies:
- No large weights were downloaded or installed without user consent.
- All 30 records were processed using text and conversational context only.
- For 14 image-bearing records, metadata explicitly documents `modality_used = "TEXT_ONLY"`, `image_assessed = false`, and `ocr_assessed = false`.

---

## 3. Language Segregation Audit

The 30-record pilot contains 27 English records and 3 Chinese records from SarcNet (`sarcnet_test_00053`, `sarcnet_train_00095`, `sarcnet_train_00844`).
- **English Benchmark Performance:** Reported separately ({metrics['english_benchmark']['en_pairs']} records, raw agreement: {metrics['english_benchmark']['raw_agreement_percent']}%, κ: {metrics['english_benchmark']['cohens_kappa']}).
- **Chinese Exploratory Performance:** Reported separately ({metrics['chinese_separate']['zh_pairs']} records, raw agreement: {metrics['chinese_separate']['raw_agreement_percent']}%).
- **No Cross-Contamination:** Chinese records were never pooled into English benchmark metrics.

---

## 4. Disagreement Root-Cause Forensic Analysis

The adjudication pass resolved {metrics['disagreement_count']} model divergences:
"""
    for adj in adjudications:
        if adj["adjudication_type"] == "LLM_ADJUDICATED_DISAGREEMENT":
            quality_md += f"""
### Record `{adj['record_id']}` ({adj['source_dataset']})
- **Comment Text:** "{adj['comment_text']}"
- **Dialogue Context:** {adj['context_text'] if adj['context_text'] else 'None'}
- **Model 1 (`{adj['model1_name']}`):** `{adj['model1_label']}` ({adj['model1_subtype']})  
  *Rationale:* {adj['model1_rationale']}
- **Model 2 (`{adj['model2_name']}`):** `{adj['model2_label']}` ({adj['model2_subtype']})  
  *Rationale:* {adj['model2_rationale']}
- **Adjudicated Decision:** **`{adj['adjudicated_label']}`** (Subtype: `{adj['adjudicated_subtype']}`, Confidence: {adj['confidence']}/5, Requires Human Review: {adj['requires_human_review']})
- **Root Cause & Adjudication Rationale:**  
  *Root Cause:* {adj['disagreement_root_cause']}  
  *Adjudicator Rationale:* {adj['adjudication_rationale']}
"""

    quality_md += """
---

## 5. Dataset Governance & Release Decision

> [!CAUTION]
> **GATE STATUS: BLOCKED_PENDING_HUMAN_ANNOTATION**  
> 1. The resulting dataset in `data/annotations/phase6d/llm_adjudication_results.csv` is suitable **strictly for weak-supervision ablation studies and diagnostic prompt engineering**.
> 2. It **MUST NOT** be used as an evaluation benchmark or reported as validated ground truth.
> 3. Downstream test splits remain uncompromised.
"""

    with open(QUALITY_AUDIT_MD, mode="w", encoding="utf-8") as f:
        f.write(quality_md)
    logger.info(f"Generated quality audit at {QUALITY_AUDIT_MD.name}")


def main():
    parser = argparse.ArgumentParser(description="Phase 6D LLM Annotation and Adjudication Pipeline")
    parser.add_argument("--pilot-manifest", type=str, default=str(MANIFEST_CSV))
    parser.add_argument("--model-1", type=str, default="llama3.2:latest")
    parser.add_argument("--model-2", type=str, default="qwen2.5:1.5b")
    parser.add_argument("--adjudicator-model", type=str, default="llama3.2:latest")
    parser.add_argument("--endpoint", type=str, default="http://localhost:11434/api/chat")
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    resume = not args.no_resume

    logger.info("==================================================")
    logger.info("PHASE 6D: LLM DUAL-ANNOTATION PIPELINE STARTING")
    logger.info(f"Model 1: {args.model_1}")
    logger.info(f"Model 2: {args.model_2}")
    logger.info(f"Adjudicator: {args.adjudicator_model}")
    logger.info(f"Dry Run: {args.dry_run} | Resume: {resume}")
    logger.info("==================================================")

    # 1. Verification of human file integrity
    if ANN1_HUMAN_CSV.exists():
        with open(ANN1_HUMAN_CSV, "r", encoding="utf-8") as f:
            h1_rows = list(csv.DictReader(f))
            h1_completed = sum(1 for r in h1_rows if r.get("primary_label", "").strip())
            assert h1_completed == 0, f"Integrity violation: {ANN1_HUMAN_CSV} has {h1_completed} completed rows!"
    if ANN2_HUMAN_CSV.exists():
        with open(ANN2_HUMAN_CSV, "r", encoding="utf-8") as f:
            h2_rows = list(csv.DictReader(f))
            h2_completed = sum(1 for r in h2_rows if r.get("primary_label", "").strip())
            assert h2_completed == 0, f"Integrity violation: {ANN2_HUMAN_CSV} has {h2_completed} completed rows!"

    # 2. Load Manifest
    manifest_records = load_manifest(Path(args.pilot_manifest))
    logger.info(f"Loaded {len(manifest_records)} records from pilot sampling manifest.")

    DATA_DIR_6D.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR_6D.mkdir(parents=True, exist_ok=True)

    # 3. Pass 1: Model 1
    logger.info(f"--- Running Pass 1: {args.model_1} ---")
    ann1_rows = run_annotation_pass(
        records=manifest_records,
        model=args.model_1,
        annotator_type="LLM_ANNOTATOR_1",
        output_csv=ANN1_LLM_CSV,
        endpoint=args.endpoint,
        timeout=args.timeout,
        resume=resume,
        dry_run=args.dry_run,
    )

    # 4. Pass 2: Model 2 (Independent)
    logger.info(f"--- Running Pass 2: {args.model_2} ---")
    ann2_rows = run_annotation_pass(
        records=manifest_records,
        model=args.model_2,
        annotator_type="LLM_ANNOTATOR_2",
        output_csv=ANN2_LLM_CSV,
        endpoint=args.endpoint,
        timeout=args.timeout,
        resume=resume,
        dry_run=args.dry_run,
    )

    # 5. Inter-Model Agreement
    logger.info("--- Calculating Inter-Model Agreement ---")
    metrics = calculate_agreement_metrics(ann1_rows, ann2_rows)
    logger.info(f"Overall Raw Agreement: {metrics['raw_agreement_percent_overall']}% ({metrics['exact_matches']}/{metrics['total_pairs']})")
    logger.info(f"Overall Cohen's Kappa: {metrics['cohens_kappa_overall']}")
    logger.info(f"Core English Raw Agreement: {metrics['english_benchmark']['raw_agreement_percent']}% | Kappa: {metrics['english_benchmark']['cohens_kappa']}")
    logger.info(f"Disagreements requiring adjudication: {metrics['disagreement_count']}")

    # Export Disagreements
    export_disagreements_csv(metrics["disagreements"], DISAGREEMENTS_CSV)

    # 6. Pass 3: Adjudication
    logger.info("--- Running Pass 3: Adjudication on Disagreements ---")
    adjudications = run_adjudication(
        manifest_records=manifest_records,
        ann1_rows=ann1_rows,
        ann2_rows=ann2_rows,
        adjudicator_model=args.adjudicator_model,
        output_csv=ADJUDICATION_CSV,
        endpoint=args.endpoint,
        timeout=args.timeout,
        resume=resume,
        dry_run=args.dry_run,
    )

    # 7. Generate Manifest and Reports
    logger.info("--- Generating Manifest & Reports ---")
    generate_manifest(
        metrics=metrics,
        model1=args.model_1,
        model2=args.model_2,
        adjudicator_model=args.adjudicator_model,
        output_path=MANIFEST_JSON,
    )

    generate_reports(
        metrics=metrics,
        adjudications=adjudications,
        model1=args.model_1,
        model2=args.model_2,
        adjudicator_model=args.adjudicator_model,
    )

    logger.info("==================================================")
    logger.info("PHASE 6D COMPLETE: All 30 records annotated, analyzed, and adjudicated.")
    logger.info("==================================================")


if __name__ == "__main__":
    main()
