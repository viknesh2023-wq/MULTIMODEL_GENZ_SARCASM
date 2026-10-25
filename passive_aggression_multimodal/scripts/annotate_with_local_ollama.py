"""
Phase 5A: Complete Local Ollama Annotation Automation Pipeline.

Processes candidate social media records through a local Ollama instance
with strict JSON schema validation, retry logic, resumable execution,
and dual-pass consistency checking.
"""
import argparse
import csv
import json
import os
import sys
import time
import urllib.request
import urllib.error
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

PROMPT_VERSION = "v1.0.0"

SYSTEM_PROMPT = """You are an expert computational linguist and pragmatic dialogue auditor. Your task is to annotate social media comments for PASSIVE AGGRESSION.

### 1. OPERATIONAL DEFINITION
Passive aggression is defined as:
INDIRECT COMMUNICATION THAT PROVIDES EVIDENCE OF COVERT INTERPERSONAL HOSTILITY, BLAME, RESENTMENT, OR CRITICISM BEHIND A SURFACE APPEARANCE OF POLITENESS, COMPLIANCE, OR NEUTRALITY.

Key hallmarks:
- Covert Hostility: The speaker seeks to demean, mock, insult, or fault the recipient.
- Plausible Deniability: Phrasing is formally polite or neutral, allowing the speaker to deny aggression if challenged.
- Pragmatic Incongruity: The hostile meaning arises from conversational context, backhanded caveats, or weaponized etiquette.

### 2. PRIMARY CLASSIFICATION LABELS (Choose Exactly One)
- "passive_aggressive": Clear pragmatic evidence of covert interpersonal hostility, implied blame, indirect criticism, or fake politeness.
- "non_passive_aggressive": Sincere statements, genuine praise, direct constructive feedback, friendly teasing/banter, venting about external objects, or OVERT DIRECT HOSTILITY.
- "ambiguous": The comment's intent cannot be reliably verified due to insufficient conversational context or inherently ambiguous phrasing.

### 3. CONTRAST BOUNDARIES (What is NOT Passive Aggression)
- DIRECT AGGRESSION / TOXICITY ("You are an idiot and completely useless"): NOT passive aggressive. It is overt and explicit; label as "non_passive_aggressive".
- SITUATIONAL SARCASM ("Great, the bus is delayed by an hour again!"): Sarcasm aimed at external misfortune or bad luck, not attacking an individual; label as "non_passive_aggressive".
- ORDINARY NEGATIVE SENTIMENT ("I feel overwhelmed and exhausted today"): Expression of internal distress without interpersonal hostility; label as "non_passive_aggressive".
- FRIENDLY TEASING ("Look who finally crawled out of bed!"): Affectionate bonding between friends; label as "non_passive_aggressive".

### 4. SUBTYPES (Select ONE if passive_aggressive; must be 'none' if non_passive_aggressive or ambiguous)
- "fake_politeness": Exaggerated formal courtesy or etiquette masking contempt.
- "indirect_criticism": Criticizing an individual without naming them directly or attacking general standards.
- "backhanded_compliment": A compliment paired with an insult, diminishing qualifier, or caveat.
- "implied_blame": Subtly shifting responsibility, fault, or guilt without direct accusation.
- "sarcastic_agreement": Feigned enthusiastic agreement to ridicule the recipient's stance.
- "dismissiveness": Stonewalling, minimization, or contemptuous brevity intended to belittle the other person.
- "other_or_unclear": Other forms of covert interpersonal hostility.
- "none": When primary_label is "non_passive_aggressive" or "ambiguous".

### 5. GROUNDING & EVIDENCE RULES
- Ground your judgment strictly in the provided comment text and dialogue context.
- Quote the exact triggering words in "evidence_span".
- Explain the pragmatic contradiction in "brief_rationale" (1-2 sentences).
- If context is too sparse to determine intent, select "ambiguous".
"""

ANNOTATION_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "brief_rationale": {
            "type": "string",
            "description": "Pragmatic reasoning: analyze whether covert interpersonal hostility exists, or if it is sincere, direct, or ambiguous."
        },
        "evidence_span": {
            "type": "string",
            "description": "Quoted words indicating intent, or 'none' if neutral."
        },
        "primary_label": {
            "type": "string",
            "enum": ["non_passive_aggressive", "ambiguous", "passive_aggressive"]
        },
        "subtype": {
            "type": "string",
            "enum": [
                "none", "fake_politeness", "indirect_criticism", "backhanded_compliment",
                "implied_blame", "sarcastic_agreement", "dismissiveness", "other_or_unclear"
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
        }
    },
    "required": [
        "brief_rationale", "evidence_span", "primary_label",
        "subtype", "confidence", "context_sufficiency"
    ]
}


def build_user_prompt(record: Dict[str, Any]) -> str:
    rec_id = record.get("record_id", "")
    src = record.get("source_dataset", "")
    text = record.get("comment_text", "").strip()
    ctx = record.get("context_text", "").strip()
    img_avail = record.get("image_available", False)

    return (
        f"Analyze the following social media post for passive aggression:\n\n"
        f"Record ID: {rec_id}\n"
        f"Source Dataset: {src}\n"
        f"Comment Text: \"{text}\"\n"
        f"Dialogue Context: {ctx if ctx else 'None'}\n"
        f"Image Availability: {img_avail} (Note: Image pixels unassessed; evaluate text and context)\n\n"
        f"DECISION RULES:\n"
        f"1. If comment is sincere, a news link, friendly banter, general chat, or direct criticism -> label as 'non_passive_aggressive' with subtype 'none'.\n"
        f"2. If intent is unclear or context is missing -> label as 'ambiguous' with subtype 'none'.\n"
        f"3. Only select 'passive_aggressive' if there is clear covert interpersonal hostility or blame masked by polite/neutral words."
    )


def query_ollama(
    model: str,
    user_prompt: str,
    system_prompt: str = SYSTEM_PROMPT,
    endpoint: str = "http://localhost:11434/api/chat",
    timeout: int = 45,
    temperature: float = 0.0,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Queries local Ollama chat API with schema constraints."""
    if dry_run:
        return {
            "primary_label": "non_passive_aggressive",
            "subtype": "none",
            "confidence": 4,
            "evidence_span": "dry run mock span",
            "brief_rationale": "Dry run mock rationale.",
            "context_sufficiency": "SUFFICIENT",
        }

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "format": ANNOTATION_JSON_SCHEMA,
        "stream": False,
        "options": {
            "temperature": temperature,
        },
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
            content = res_data.get("message", {}).get("content", "")
            parsed = json.loads(content)
            parsed["_latency_sec"] = round(latency, 3)
            return parsed
    except Exception as e:
        raise RuntimeError(f"Ollama inference error for model '{model}': {e}")


def annotate_record(
    record: Dict[str, Any],
    model: str,
    temperature: float = 0.0,
    max_retries: int = 2,
    dry_run: bool = False,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Runs annotation for a single record with validation and retries."""
    prompt = build_user_prompt(record)
    last_err = None

    for attempt in range(max_retries + 1):
        try:
            result = query_ollama(
                model=model,
                user_prompt=prompt,
                temperature=temperature,
                dry_run=dry_run,
            )

            # Strict field check
            p_label = result.get("primary_label")
            if p_label not in ("passive_aggressive", "non_passive_aggressive", "ambiguous"):
                raise ValueError(f"Invalid primary_label: {p_label}")

            subtype = result.get("subtype", "none")
            if p_label == "passive_aggressive" and (not subtype or subtype == "none"):
                subtype = "other_or_unclear"

            now_iso = datetime.now(timezone.utc).isoformat()
            annotated_record = {
                "record_id": record["record_id"],
                "source_dataset": record["source_dataset"],
                "comment_text": record["comment_text"],
                "image_available": record.get("image_available", False),
                "context_available": bool(record.get("context_text")),
                "primary_label": p_label,
                "subtype": subtype,
                "confidence": int(result.get("confidence", 3)),
                "evidence_span": result.get("evidence_span", ""),
                "brief_rationale": result.get("brief_rationale", ""),
                "context_sufficiency": result.get("context_sufficiency", "INSUFFICIENT"),
                "image_role": "UNASSESSED_VISION_MODEL_ABSENT" if record.get("image_available") else "ABSENT",
                "modality_used": "TEXT_ONLY",
                "model_name": model,
                "prompt_version": PROMPT_VERSION,
                "annotation_source": "LLM",
                "annotation_status": "LLM_PROVISIONAL",
                "inference_latency_sec": result.get("_latency_sec", 0.0),
                "timestamp": now_iso,
                "error": None,
            }
            return annotated_record, None
        except Exception as e:
            last_err = str(e)
            time.sleep(1.0)

    # If all attempts failed
    return None, last_err


def run_pass(
    records: List[Dict[str, Any]],
    output_path: str,
    failure_path: str,
    model: str,
    pass_name: str,
    temperature: float = 0.0,
    resume: bool = True,
    dry_run: bool = False,
) -> List[Dict[str, Any]]:
    """Executes an annotation pass over a list of records with incremental saving."""
    completed_ids = set()
    completed_records = []

    if resume and os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    completed_ids.add(item["record_id"])
                    completed_records.append(item)
        print(f"[{pass_name}] Resuming: {len(completed_ids)} records already completed in {output_path}")

    out_file = open(output_path, "a", encoding="utf-8")
    fail_file = open(failure_path, "a", encoding="utf-8")

    try:
        for idx, rec in enumerate(records):
            rid = rec["record_id"]
            if rid in completed_ids:
                continue

            print(f"[{pass_name}] ({idx + 1}/{len(records)}) Annotating {rid}...", end="", flush=True)
            annotated, err = annotate_record(
                record=rec,
                model=model,
                temperature=temperature,
                dry_run=dry_run,
            )

            if annotated is not None:
                out_file.write(json.dumps(annotated, ensure_ascii=False) + "\n")
                out_file.flush()
                completed_records.append(annotated)
                completed_ids.add(rid)
                print(f" DONE -> {annotated['primary_label']} ({annotated['subtype']}) [{annotated['inference_latency_sec']}s]")
            else:
                fail_entry = {
                    "record_id": rid,
                    "pass_name": pass_name,
                    "model": model,
                    "error": err,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
                fail_file.write(json.dumps(fail_entry, ensure_ascii=False) + "\n")
                fail_file.flush()
                print(f" FAILED: {err}")
    finally:
        out_file.close()
        fail_file.close()

    return completed_records


def compute_consistency(
    pass1: List[Dict[str, Any]],
    pass2: List[Dict[str, Any]],
    disagreements_path: str,
) -> Dict[str, Any]:
    """Computes repeated-run or cross-model consistency metrics."""
    map1 = {r["record_id"]: r for r in pass1}
    map2 = {r["record_id"]: r for r in pass2}
    common_ids = sorted(list(set(map1.keys()).intersection(set(map2.keys()))))

    matches = 0
    disagreements = []
    per_class_matches = {"passive_aggressive": 0, "non_passive_aggressive": 0, "ambiguous": 0}
    per_class_total = {"passive_aggressive": 0, "non_passive_aggressive": 0, "ambiguous": 0}

    for rid in common_ids:
        r1 = map1[rid]
        r2 = map2[rid]
        l1 = r1["primary_label"]
        l2 = r2["primary_label"]

        per_class_total[l1] = per_class_total.get(l1, 0) + 1

        if l1 == l2:
            matches += 1
            per_class_matches[l1] = per_class_matches.get(l1, 0) + 1
        else:
            disagreements.append({
                "record_id": rid,
                "comment_text": r1["comment_text"],
                "pass1_model": r1.get("model_name"),
                "pass1_label": l1,
                "pass1_subtype": r1.get("subtype"),
                "pass1_confidence": r1.get("confidence"),
                "pass1_rationale": r1.get("brief_rationale"),
                "pass2_model": r2.get("model_name"),
                "pass2_label": l2,
                "pass2_subtype": r2.get("subtype"),
                "pass2_confidence": r2.get("confidence"),
                "pass2_rationale": r2.get("brief_rationale"),
            })

    total = len(common_ids)
    raw_agreement = (matches / total * 100.0) if total > 0 else 0.0

    # Save disagreements CSV
    if disagreements:
        fieldnames = list(disagreements[0].keys())
        with open(disagreements_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(disagreements)

    return {
        "paired_records_count": total,
        "agreement_count": matches,
        "disagreement_count": len(disagreements),
        "raw_consistency_percent": round(raw_agreement, 2),
        "per_class_consistency": {
            k: round((per_class_matches[k] / per_class_total[k] * 100.0), 2) if per_class_total[k] > 0 else None
            for k in per_class_total
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Phase 5A Local Ollama Annotation Pipeline")
    parser.add_argument(
        "--input",
        default="data/annotations/phase4/human_sampling_manifest_100.jsonl",
        help="Input candidate manifest path",
    )
    parser.add_argument(
        "--output-dir",
        default="data/annotations/phase5a/llm",
        help="Directory to save LLM annotation artifacts",
    )
    parser.add_argument(
        "--model",
        default="llama3.2:latest",
        help="Primary Ollama model for Pass 1",
    )
    parser.add_argument(
        "--second-pass-model",
        default=None,
        help="Model for Pass 2 (defaults to same model as Pass 1)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum records to process (for testing or pilots)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=True,
        help="Resume from existing output files",
    )
    parser.add_argument(
        "--second-pass",
        action="store_true",
        default=False,
        help="Execute second independent annotation pass",
    )
    parser.add_argument(
        "--text-only",
        action="store_true",
        default=True,
        help="Enforce text-only evaluation",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Mock Ollama responses without calling API",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    start_time = datetime.now(timezone.utc)

    print("=" * 65)
    print("PHASE 5A -- LOCAL OLLAMA PASSIVE-AGGRESSION ANNOTATOR")
    print("=" * 65)
    print(f"Input Manifest:   {args.input}")
    print(f"Output Directory: {args.output_dir}")
    print(f"Primary Model:    {args.model}")
    print(f"Second Pass:      {args.second_pass} (Model: {args.second_pass_model or args.model})")
    print(f"Dry Run:          {args.dry_run}")

    if not os.path.exists(args.input):
        print(f"Error: Input file does not exist: {args.input}")
        sys.exit(1)

    with open(args.input, "r", encoding="utf-8") as f:
        records = [json.loads(l) for l in f if l.strip()]

    if args.limit:
        records = records[: args.limit]
    print(f"Total records to process: {len(records)}")

    pass1_out = os.path.join(args.output_dir, "llm_annotations_pass1.jsonl")
    pass2_out = os.path.join(args.output_dir, "llm_annotations_pass2.jsonl")
    fail_out = os.path.join(args.output_dir, "llm_annotation_failures.jsonl")
    disagree_out = os.path.join(args.output_dir, "llm_annotation_disagreements.csv")

    # Execute Pass 1
    print("\n--- EXECUTING PASS 1 ---")
    p1_results = run_pass(
        records=records,
        output_path=pass1_out,
        failure_path=fail_out,
        model=args.model,
        pass_name="PASS_1",
        temperature=0.0,
        resume=args.resume,
        dry_run=args.dry_run,
    )
    print(f"\nPass 1 Complete: {len(p1_results)} successful records.")

    # Execute Pass 2 if requested
    p2_results = []
    consistency_results = None
    if args.second_pass:
        p2_model = args.second_pass_model or args.model
        p2_temp = 0.2 if p2_model == args.model else 0.0
        print(f"\n--- EXECUTING PASS 2 (Model: {p2_model}, temp={p2_temp}) ---")
        p2_results = run_pass(
            records=records,
            output_path=pass2_out,
            failure_path=fail_out,
            model=p2_model,
            pass_name="PASS_2",
            temperature=p2_temp,
            resume=args.resume,
            dry_run=args.dry_run,
        )
        print(f"\nPass 2 Complete: {len(p2_results)} successful records.")
        consistency_results = compute_consistency(p1_results, p2_results, disagree_out)
        print(f"Consistency: {consistency_results['raw_consistency_percent']}% agreement ({consistency_results['agreement_count']}/{consistency_results['paired_records_count']})")
        print(f"Disagreements exported to: {disagree_out}")

    # Compute label distribution for Pass 1
    label_dist = {"passive_aggressive": 0, "non_passive_aggressive": 0, "ambiguous": 0}
    subtype_dist: Dict[str, int] = {}
    for r in p1_results:
        lbl = r["primary_label"]
        label_dist[lbl] = label_dist.get(lbl, 0) + 1
        sub = r["subtype"]
        if sub and sub != "none":
            subtype_dist[sub] = subtype_dist.get(sub, 0) + 1

    dist_path = os.path.join(args.output_dir, "llm_annotation_label_distribution.json")
    with open(dist_path, "w", encoding="utf-8") as f:
        json.dump({"primary_labels": label_dist, "subtypes": subtype_dist}, f, indent=2)

    end_time = datetime.now(timezone.utc)
    duration_total = (end_time - start_time).total_seconds()

    # Create run manifest
    run_manifest = {
        "pipeline_version": "5.0.0",
        "prompt_version": PROMPT_VERSION,
        "execution_date": start_time.strftime("%Y-%m-%d"),
        "start_time": start_time.isoformat(),
        "end_time": end_time.isoformat(),
        "total_duration_sec": round(duration_total, 2),
        "primary_model": args.model,
        "second_pass_executed": args.second_pass,
        "second_pass_model": args.second_pass_model or (args.model if args.second_pass else None),
        "records_attempted": len(records),
        "pass1_success_count": len(p1_results),
        "pass2_success_count": len(p2_results) if args.second_pass else 0,
        "dry_run": args.dry_run,
        "modality_used": "TEXT_ONLY",
        "visual_assessment_policy": "UNASSESSED_VISION_MODEL_ABSENT",
        "label_distribution_pass1": label_dist,
        "subtype_distribution_pass1": subtype_dist,
        "consistency": consistency_results,
        "output_files": {
            "pass1": pass1_out,
            "pass2": pass2_out if args.second_pass else None,
            "failures": fail_out,
            "disagreements": disagree_out if args.second_pass else None,
            "distribution": dist_path,
        },
    }

    manifest_path = os.path.join(args.output_dir, "llm_annotation_run_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(run_manifest, f, indent=2)
    print(f"\nSaved Run Manifest to: {manifest_path}")
    print("=" * 65)


if __name__ == "__main__":
    main()
