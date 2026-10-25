"""
scripts/phase10/run_llm_annotation.py

Phase 10: Dual-Pass Independent LLM Annotation Runner.
Annotator A: llama3.2:latest
Annotator B: qwen2.5:1.5b

Features:
- Strict double-blind isolation (Annotator B never sees Annotator A's label)
- Native JSON Schema enforcement with Ollama
- In-memory resume caching (resumes interrupted runs without re-querying)
- Modality honesty (explicitly marks image_assessed = False, modality_used = ['text'])
- Standardized ISO timestamp and model provenance metadata
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Set
import urllib.error
import urllib.request

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Phase10-AnnotationRunner")

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT_DIR / "data/annotations/phase10"
SCHEMA_FILE = ROOT_DIR / "schemas/phase10/passive_aggression_annotation_schema.json"

PROMPT_VERSION = "v1.0.0"
SYSTEM_PROMPT = """You are an expert computational linguist. Evaluate social media comments for passive aggression using strict operational criteria.

OPERATIONAL DEFINITION:
Passive aggression is indirect communication conveying covert interpersonal hostility disguised beneath surface politeness, compliance, praise, or neutrality, with plausible deniability.

THREE-PRONG TEST:
1. Interpersonal Target: Hostility/criticism directed at an identifiable interlocutor, participant, or person.
2. Plausible Deniability: Surface wording appears polite, praising, or neutral, permitting denial of malice.
3. Covert Hostility: Pragmatic meaning conveys resentment, blame, contempt, or demeaning criticism.

A comment is 'passive_aggressive' ONLY IF ALL THREE CRITERIA ARE MET.

NEGATIVE CONTRASTS (Select 'non_passive_aggressive'):
- Direct aggression / overt toxicity ("You are an idiot", "Shut up") -> Overt hostility lacking polite disguise is NOT PA.
- Situational sarcasm / venting ("Bus is late again!", "Weather is terrible") -> No individual target is NOT PA.
- Factual reporting / descriptions (news summaries, scores, links) -> Objective reporting is NOT PA.
- Sincere praise & gratitude ("Thanks for the great advice!") -> Genuine appreciation is NOT PA.
- Direct disagreement ("I disagree with that point", "The data shows otherwise") -> Direct debate is NOT PA.
- Friendly teasing / banter ("Look who finally showed up!") -> Friendly social bonding is NOT PA.

AMBIGUOUS (Select 'ambiguous'):
- Missing context or meme fragment where intent cannot be verified.

Respond strictly in valid JSON format adhering to the schema."""

OLLAMA_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "interpersonal_target": {"type": "string", "enum": ["PRESENT", "ABSENT", "UNCLEAR"]},
        "plausible_deniability": {"type": "string", "enum": ["YES", "NO", "UNCLEAR"]},
        "covert_hostility": {"type": "string", "enum": ["COVERT", "OVERT", "ABSENT", "UNCLEAR"]},
        "label": {"type": "string", "enum": ["passive_aggressive", "non_passive_aggressive", "ambiguous"]},
        "subtype": {
            "type": "string",
            "enum": [
                "fake_politeness", "indirect_criticism", "backhanded_compliment",
                "implied_blame", "sarcastic_agreement", "dismissiveness", "other_or_unclear", "not_applicable"
            ]
        },
        "confidence_band": {"type": "string", "enum": ["high", "medium", "low"]},
        "evidence_span": {"type": "string"},
        "rationale": {"type": "string"},
        "context_sufficient": {"type": "boolean"}
    },
    "required": [
        "interpersonal_target", "plausible_deniability", "covert_hostility",
        "label", "subtype", "confidence_band", "evidence_span", "rationale", "context_sufficient"
    ]
}


SCHEMA_INSTRUCTION = """
Required JSON Output Format:
{
  "interpersonal_target": "PRESENT" | "ABSENT" | "UNCLEAR",
  "plausible_deniability": "YES" | "NO" | "UNCLEAR",
  "covert_hostility": "COVERT" | "OVERT" | "ABSENT" | "UNCLEAR",
  "label": "passive_aggressive" | "non_passive_aggressive" | "ambiguous",
  "subtype": "fake_politeness" | "indirect_criticism" | "backhanded_compliment" | "implied_blame" | "sarcastic_agreement" | "dismissiveness" | "other_or_unclear" | "not_applicable",
  "confidence_band": "high" | "medium" | "low",
  "evidence_span": "<short excerpt or empty>",
  "rationale": "<1-2 sentence evidence-based rationale>",
  "context_sufficient": true | false
}"""

SYSTEM_PROMPT_WITH_SCHEMA = SYSTEM_PROMPT + "\n\n" + SCHEMA_INSTRUCTION


def build_user_prompt(record: Dict[str, Any]) -> str:
    rec_id = record.get("record_id", "")
    src = record.get("source_dataset", "")
    text = record.get("comment_text", "").strip()
    ctx = record.get("context_text", "").strip()
    img_avail = record.get("image_available", False)

    return (
        f"Evaluate the following social media post for passive aggression according to the operational criteria:\n\n"
        f"Record ID: {rec_id}\n"
        f"Source Dataset: {src}\n"
        f"Comment Text: \"{text}\"\n"
        f"Dialogue Context: {ctx if ctx else 'None'}\n"
        f"Image Availability: {img_avail} (Note: Image pixels unassessed; evaluate text and context)\n\n"
        f"Return strictly valid JSON adhering to the required schema."
    )


def query_ollama(
    model: str,
    user_prompt: str,
    system_prompt: str = SYSTEM_PROMPT_WITH_SCHEMA,
    schema: Any = OLLAMA_JSON_SCHEMA,
    endpoint: str = "http://localhost:11434/api/chat",
    timeout: int = 90,
    temperature: float = 0.0,
    retries: int = 3,
    dry_run: bool = False,
) -> Dict[str, Any]:
    if dry_run:
        return {
            "interpersonal_target": "ABSENT",
            "plausible_deniability": "YES",
            "covert_hostility": "ABSENT",
            "label": "non_passive_aggressive",
            "subtype": "not_applicable",
            "confidence_band": "high",
            "evidence_span": "dry run evidence span",
            "rationale": "Dry run mock rationale.",
            "context_sufficient": True,
            "_latency_sec": 0.001,
            "_model": model,
        }

    # For qwen or smaller models, use format: "json" to bypass BNF grammar compilation overhead
    fmt = "json" if "qwen" in model.lower() else schema

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "format": fmt,
        "options": {
            "temperature": temperature,
            "seed": 42,
            "repeat_penalty": 1.15,
            "num_predict": 256,
        },
        "stream": False,
    }

    req_data = json.dumps(payload).encode("utf-8")
    for attempt in range(1, retries + 1):
        t0 = time.time()
        try:
            req = urllib.request.Request(
                endpoint,
                data=req_data,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                res_json = json.loads(resp.read().decode("utf-8"))
                dt = time.time() - t0
                content_str = res_json.get("message", {}).get("content", "").strip()
                parsed = json.loads(content_str)
                # Normalize keys and values
                parsed["_latency_sec"] = round(dt, 3)
                parsed["_model"] = model
                # Ensure required keys exist
                parsed.setdefault("label", "ambiguous")
                parsed.setdefault("subtype", "not_applicable")
                parsed.setdefault("confidence_band", "medium")
                parsed.setdefault("rationale", "No rationale supplied.")
                parsed.setdefault("context_sufficient", True)
                return parsed
        except Exception as e:
            logger.warning("[%s] Attempt %d failed: %s", model, attempt, e)
            if attempt == retries:
                logger.error("Exhausted retries on model %s: %s. Emitting fallback failure record.", model, e)
                return {
                    "interpersonal_target": "UNCLEAR",
                    "plausible_deniability": "UNCLEAR",
                    "covert_hostility": "UNCLEAR",
                    "label": "ambiguous",
                    "subtype": "other_or_unclear",
                    "confidence_band": "low",
                    "evidence_span": "",
                    "rationale": f"Inference failed after {retries} retries: {str(e)}",
                    "context_sufficient": False,
                    "_latency_sec": 0.0,
                    "_model": model,
                    "_error": str(e),
                }
            time.sleep(1.0 * attempt)

    return {
        "label": "ambiguous",
        "subtype": "other_or_unclear",
        "confidence_band": "low",
        "rationale": "Terminated unexpectedly.",
        "context_sufficient": False,
        "_latency_sec": 0.0,
        "_model": model,
    }


def load_cached_annotations(jsonl_path: Path) -> Dict[str, Dict[str, Any]]:
    cached = {}
    if jsonl_path.exists():
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    d = json.loads(line)
                    cached[d["record_id"]] = d
    return cached


def format_annotation_record(
    source_rec: Dict[str, Any],
    llm_out: Dict[str, Any],
    model_name: str,
) -> Dict[str, Any]:
    return {
        "record_id": source_rec["record_id"],
        "source_dataset": source_rec.get("source_dataset", "unknown"),
        "source_record_id": source_rec.get("source_record_id", source_rec["record_id"]),
        "text": source_rec["comment_text"],
        "caption": None,
        "image_path": source_rec.get("image_path") if source_rec.get("image_path") else None,
        "language": source_rec.get("language", "en"),
        "label": llm_out.get("label", "ambiguous"),
        "subtype": llm_out.get("subtype", "not_applicable"),
        "confidence_band": llm_out.get("confidence_band", "medium"),
        "evidence_spans": [llm_out.get("evidence_span", "")] if llm_out.get("evidence_span") else [],
        "rationale": llm_out.get("rationale", "No rationale provided."),
        "interpersonal_target_present": llm_out.get("interpersonal_target") == "PRESENT",
        "plausible_deniability_present": llm_out.get("plausible_deniability") == "YES",
        "covert_hostility_present": llm_out.get("covert_hostility") == "COVERT",
        "context_sufficient": bool(llm_out.get("context_sufficient", True)),
        "image_assessed": False,
        "ocr_assessed": False,
        "annotation_model": model_name,
        "model_version": model_name,
        "prompt_version": PROMPT_VERSION,
        "annotation_timestamp": datetime.now(timezone.utc).isoformat(),
        "provenance": "LLM_SILVER",
        "agreement_status": None,
        "adjudication_status": None,
        "exclusion_reason": None,
        "annotation_warnings": ["Text-only evaluation: Image pixels unassessed."],
        "_latency_sec": llm_out.get("_latency_sec", 0.0),
    }


def run_dual_pass_annotation(
    records: List[Dict[str, Any]],
    model_a: str = "llama3.2:latest",
    model_b: str = "qwen2.5:1.5b",
    dry_run: bool = False,
) -> Dict[str, Any]:
    file_a = DATA_DIR / "annotator_a.jsonl"
    file_b = DATA_DIR / "annotator_b.jsonl"

    cached_a = load_cached_annotations(file_a)
    cached_b = load_cached_annotations(file_b)

    logger.info("Starting Dual-Pass Annotation on %d records...", len(records))
    logger.info("Found %d cached for Annotator A, %d for Annotator B.", len(cached_a), len(cached_b))

    results_a: List[Dict[str, Any]] = list(cached_a.values())
    results_b: List[Dict[str, Any]] = list(cached_b.values())

    # Pass 1: Annotator A
    logger.info("--- Pass 1: Annotator A (%s) ---", model_a)
    t0_a = time.time()
    with open(file_a, "a", encoding="utf-8") as f_a:
        for idx, rec in enumerate(records, start=1):
            rec_id = rec["record_id"]
            if rec_id in cached_a:
                continue
            prompt = build_user_prompt(rec)
            llm_res = query_ollama(model=model_a, user_prompt=prompt, dry_run=dry_run)
            formatted = format_annotation_record(rec, llm_res, model_a)
            f_a.write(json.dumps(formatted) + "\n")
            f_a.flush()
            cached_a[rec_id] = formatted
            results_a.append(formatted)
            if idx % 10 == 0 or idx == len(records):
                logger.info("Annotator A: %d / %d completed.", idx, len(records))

    dur_a = time.time() - t0_a

    # Pass 2: Annotator B (Double-Blind Isolation)
    logger.info("--- Pass 2: Annotator B (%s) ---", model_b)
    t0_b = time.time()
    with open(file_b, "a", encoding="utf-8") as f_b:
        for idx, rec in enumerate(records, start=1):
            rec_id = rec["record_id"]
            if rec_id in cached_b:
                continue
            prompt = build_user_prompt(rec)
            llm_res = query_ollama(model=model_b, user_prompt=prompt, dry_run=dry_run)
            formatted = format_annotation_record(rec, llm_res, model_b)
            f_b.write(json.dumps(formatted) + "\n")
            f_b.flush()
            cached_b[rec_id] = formatted
            results_b.append(formatted)
            if idx % 10 == 0 or idx == len(records):
                logger.info("Annotator B: %d / %d completed.", idx, len(records))

    dur_b = time.time() - t0_b

    # Manifest
    manifest = {
        "phase": "Phase 10 — Automated Dual-Pass LLM Annotation",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "prompt_version": PROMPT_VERSION,
        "annotator_a": {
            "model": model_a,
            "total_records": len(results_a),
            "output_file": str(file_a.relative_to(ROOT_DIR)),
            "duration_sec": round(dur_a, 2),
        },
        "annotator_b": {
            "model": model_b,
            "total_records": len(results_b),
            "output_file": str(file_b.relative_to(ROOT_DIR)),
            "duration_sec": round(dur_b, 2),
        },
        "modality_honesty": {
            "image_assessed": False,
            "ocr_assessed": False,
            "modality_used": ["text"],
            "disclaimer": "Models were evaluated in text-only mode; image pixels were unassessed."
        },
        "evaluation_gate": "BLOCKED_PENDING_HUMAN_ANNOTATION",
        "silver_gate": "LLM_SILVER_DATASET_AVAILABLE_FOR_EXPLORATORY_RESEARCH",
    }

    manifest_file = DATA_DIR / "annotation_run_manifest.json"
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    logger.info("Annotation run manifest saved to %s", manifest_file)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Phase 10 dual-pass LLM annotation")
    parser.add_argument("--limit", type=int, default=100, help="Limit number of records to process")
    parser.add_argument("--dry-run", action="store_true", help="Run mock dry-run without Ollama")
    parser.add_argument("--model-a", type=str, default="llama3.2:latest")
    parser.add_argument("--model-b", type=str, default="qwen2.5:1.5b")
    args = parser.parse_args()

    eligible_csv = DATA_DIR / "eligible_records.csv"
    if not eligible_csv.exists():
        logger.error("Eligible records CSV missing! Run prepare_eligible_corpus.py first.")
        sys.exit(1)

    with open(eligible_csv, "r", encoding="utf-8") as f:
        records_to_run = list(csv.DictReader(f))[:args.limit]

    run_dual_pass_annotation(
        records_to_run,
        model_a=args.model_a,
        model_b=args.model_b,
        dry_run=args.dry_run,
    )
