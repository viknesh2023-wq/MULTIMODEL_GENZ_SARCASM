"""
scripts/phase10/prepare_eligible_corpus.py

Builds a stratified, deduplicated eligible candidate corpus of 100 records for Phase 10
automated LLM annotation, auditing, and silver dataset creation.
Integrates:
- 30 Phase 6A pilot records
- 19 Phase 5D verified silver candidates
- Stratified sampling from MUStARD, Civil Comments, and MMSD to reach 100 diverse records
- Hard negatives: direct insults, situational sarcasm, sincere praise, factual descriptions
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Set

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Phase10-CorpusPrep")

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
RANDOM_SEED = 42

DATA_DIR = ROOT_DIR / "data"
OUT_ANN_DIR = DATA_DIR / "annotations/phase10"
OUT_PROC_DIR = DATA_DIR / "processed/phase10"
OUT_REP_DIR = ROOT_DIR / "reports/phase10"

OUT_ANN_DIR.mkdir(parents=True, exist_ok=True)
OUT_PROC_DIR.mkdir(parents=True, exist_ok=True)
OUT_REP_DIR.mkdir(parents=True, exist_ok=True)


def compute_hashes(text: str) -> tuple[str, str]:
    raw_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
    norm_text = re.sub(r"[^a-z0-9]", "", text.lower())
    norm_hash = hashlib.sha256(norm_text.encode("utf-8")).hexdigest()
    return raw_hash, norm_hash


def build_eligible_corpus(target_count: int = 100) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    seen_ids: Set[str] = set()
    seen_raw_hashes: Dict[str, str] = {}
    seen_norm_hashes: Dict[str, str] = {}
    duplicates_audit: List[Dict[str, Any]] = []

    def try_add(rec: Dict[str, Any], source_type: str) -> bool:
        rec_id = rec["record_id"]
        text = rec["comment_text"].strip()
        if not text:
            return False

        r_hash, n_hash = compute_hashes(text)

        if rec_id in seen_ids:
            duplicates_audit.append({
                "record_id": rec_id,
                "reason": "DUPLICATE_ID",
                "canonical_id": rec_id,
                "text": text[:50],
            })
            return False

        if r_hash in seen_raw_hashes:
            duplicates_audit.append({
                "record_id": rec_id,
                "reason": "EXACT_TEXT_DUPLICATE",
                "canonical_id": seen_raw_hashes[r_hash],
                "text": text[:50],
            })
            return False

        if n_hash in seen_norm_hashes:
            duplicates_audit.append({
                "record_id": rec_id,
                "reason": "NORMALIZED_TEXT_DUPLICATE",
                "canonical_id": seen_norm_hashes[n_hash],
                "text": text[:50],
            })
            return False

        seen_ids.add(rec_id)
        seen_raw_hashes[r_hash] = rec_id
        seen_norm_hashes[n_hash] = rec_id

        rec["raw_text_sha256"] = r_hash
        rec["normalized_text_sha256"] = n_hash
        rec["sample_source"] = source_type
        records.append(rec)
        return True

    # 1. Load Phase 6A Pilot Manifest (30 records)
    p6a_csv = DATA_DIR / "annotations/phase6a/pilot_sampling_manifest.csv"
    if p6a_csv.exists():
        with open(p6a_csv, "r", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                rec = {
                    "record_id": r["record_id"],
                    "source_dataset": r["source_dataset"],
                    "source_record_id": r.get("source_record_id", r["record_id"]),
                    "comment_text": r["comment_text"],
                    "context_text": r.get("context_text", ""),
                    "image_available": r.get("image_available", "False").lower() == "true",
                    "image_path": r.get("image_path", ""),
                    "language": r.get("language", "en"),
                    "archetype": r.get("sampling_archetype", "PILOT_CANDIDATE"),
                    "selection_notes": r.get("selection_rationale", "Phase 6A pilot sample"),
                }
                try_add(rec, "PHASE_6A_PILOT")

    logger.info("Loaded %d unique records from Phase 6A pilot.", len(records))

    # 2. Load Phase 5D Verified Silver (19 records)
    p5d_file = DATA_DIR / "annotations/phase5d/silver_dataset_verified.jsonl"
    if p5d_file.exists():
        with open(p5d_file, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                d = json.loads(line)
                rec = {
                    "record_id": d["record_id"],
                    "source_dataset": d.get("source_dataset", "unknown"),
                    "source_record_id": d.get("source_record_id", d["record_id"]),
                    "comment_text": d["comment_text"],
                    "context_text": d.get("context_text", ""),
                    "image_available": d.get("image_available", False),
                    "image_path": d.get("image_path", ""),
                    "language": "en",
                    "archetype": "PHASE_5D_SILVER_VERIFIED",
                    "selection_notes": d.get("audit_rationale", "Phase 5D silver verified record"),
                }
                try_add(rec, "PHASE_5D_SILVER")

    logger.info("Total records after Phase 5D addition: %d", len(records))

    # 3. Add Stratified Samples from Source Corpora to reach target_count
    # A. MUStARD (Dialogue Sarcasm / Contrast)
    mustard_p = DATA_DIR / "manifests/mustard_manifest.jsonl"
    if mustard_p.exists() and len(records) < target_count:
        with open(mustard_p, "r", encoding="utf-8") as f:
            for line in f:
                if len(records) >= target_count - 30:  # Reserve space for civil_comments and mmsd
                    break
                d = json.loads(line)
                ctx = "\n".join([f"[Context]: {c}" for c in d.get("conversation_context", [])])
                rec = {
                    "record_id": d["example_id"],
                    "source_dataset": "mustard",
                    "source_record_id": d["source_record_id"],
                    "comment_text": d["target_comment"],
                    "context_text": ctx,
                    "image_available": False,
                    "image_path": "",
                    "language": "en",
                    "archetype": "DIALOGUE_SITCOM_CANDIDATE",
                    "selection_notes": "MUStARD sitcom dialogue utterance with conversational turns",
                }
                try_add(rec, "MUSTARD_MANIFEST")

    # B. Civil Comments (Toxicity, Direct Aggression Negatives, News Discussions)
    cc_p = DATA_DIR / "manifests/civil_comments_manifest.jsonl"
    if cc_p.exists() and len(records) < target_count:
        with open(cc_p, "r", encoding="utf-8") as f:
            for line in f:
                if len(records) >= target_count - 15:
                    break
                d = json.loads(line)
                comment = d.get("target_comment", "").strip()
                # Exclude ultra-short fragments
                if len(comment.split()) < 4 or len(comment) > 300:
                    continue
                rec = {
                    "record_id": d["example_id"],
                    "source_dataset": "civil_comments",
                    "source_record_id": d["source_record_id"],
                    "comment_text": comment,
                    "context_text": "",
                    "image_available": False,
                    "image_path": "",
                    "language": "en",
                    "archetype": "CIVIL_COMMENTS_HARD_NEGATIVE_OR_CIVIL_DEBATE",
                    "selection_notes": "Civil Comments text with direct toxicity or constructive debate",
                }
                try_add(rec, "CIVIL_COMMENTS_MANIFEST")

    # C. MMSD (Situational Sarcasm & Image Multimodal Candidates)
    mmsd_p = DATA_DIR / "manifests/mmsd_manifest.jsonl"
    if mmsd_p.exists() and len(records) < target_count:
        with open(mmsd_p, "r", encoding="utf-8") as f:
            for line in f:
                if len(records) >= target_count:
                    break
                d = json.loads(line)
                comment = d.get("target_comment", "").strip()
                if len(comment.split()) < 4 or len(comment) > 250:
                    continue
                rec = {
                    "record_id": d["example_id"],
                    "source_dataset": "mmsd",
                    "source_record_id": d["source_record_id"],
                    "comment_text": comment,
                    "context_text": "",
                    "image_available": bool(d.get("image_path")),
                    "image_path": d.get("image_path", ""),
                    "language": "en",
                    "archetype": "MMSD_SITUATIONAL_SARCASM_MULTIMODAL",
                    "selection_notes": "MMSD Twitter post candidate with image link and situational context",
                }
                try_add(rec, "MMSD_MANIFEST")

    # Add sample index
    for idx, r in enumerate(records, start=1):
        r["sample_index"] = idx

    logger.info("Final unique eligible corpus count: %d records.", len(records))

    # Save eligible_records.csv
    eligible_csv = OUT_ANN_DIR / "eligible_records.csv"
    sampling_csv = OUT_ANN_DIR / "sampling_manifest.csv"
    fieldnames = [
        "sample_index", "record_id", "source_dataset", "source_record_id",
        "comment_text", "context_text", "image_available", "image_path",
        "language", "archetype", "sample_source", "selection_notes",
        "raw_text_sha256", "normalized_text_sha256",
    ]

    for p in [eligible_csv, sampling_csv]:
        with open(p, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in records:
                writer.writerow({k: r.get(k, "") for k in fieldnames})

    # Save duplicate audit
    dup_csv = OUT_PROC_DIR / "duplicate_audit.csv"
    with open(dup_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["record_id", "reason", "canonical_id", "text"])
        writer.writeheader()
        for d in duplicates_audit:
            writer.writerow(d)

    # Save duplicate group manifest
    grp_csv = OUT_PROC_DIR / "duplicate_group_manifest.csv"
    with open(grp_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["group_id", "canonical_record_id", "raw_hash", "normalized_hash"])
        writer.writeheader()
        for r in records:
            writer.writerow({
                "group_id": f"grp_{r['record_id']}",
                "canonical_record_id": r["record_id"],
                "raw_hash": r["raw_text_sha256"],
                "normalized_hash": r["normalized_text_sha256"],
            })

    # Generate Reports
    generate_eligibility_report(records, duplicates_audit)
    generate_deduplication_report(records, duplicates_audit)

    return records


def generate_eligibility_report(records: List[Dict[str, Any]], dups: List[Dict[str, Any]]) -> None:
    src_dist = {}
    lang_dist = {}
    img_dist = {"with_image": 0, "text_only": 0}
    for r in records:
        src_dist[r["source_dataset"]] = src_dist.get(r["source_dataset"], 0) + 1
        lang_dist[r["language"]] = lang_dist.get(r["language"], 0) + 1
        if r["image_available"]:
            img_dist["with_image"] += 1
        else:
            img_dist["text_only"] += 1

    md = f"""# Phase 10 Eligibility and Sampling Report
## Stratified Candidate Corpus Construction for Automated Annotation

**Project Title:** Context-Aware Multimodal Passive-Aggression Detection in Social Media Using Image–Comment Fusion  
**Artifact:** `reports/phase10/eligibility_and_sampling_report.md`  
**Date:** October 22, 2026  

---

## 1. Corpus Summary
- **Total Eligible Records Sampled:** {len(records)}
- **Deduplication Status:** 100% unique raw and normalized text SHA-256 hashes.
- **Excluded Duplicates Identified:** {len(dups)}

## 2. Source Dataset Distribution
| Source Dataset | Count | Percentage |
|---|---|---|
"""
    for s, c in sorted(src_dist.items(), key=lambda x: x[1], reverse=True):
        md += f"| **{s}** | {c} | {100.0 * c / len(records):.1f}% |\n"

    md += f"""
## 3. Modality Coverage
- **Text Only Records:** {img_dist['text_only']} ({100.0 * img_dist['text_only'] / len(records):.1f}%)
- **Image-Bearing Records:** {img_dist['with_image']} ({100.0 * img_dist['with_image'] / len(records):.1f}%)

## 4. Language Distribution
"""
    for l, c in lang_dist.items():
        md += f"- **{l}**: {c} records\n"

    md += """
## 5. Scientific Governance Boundary
- None of the source records inherit pre-existing sarcasm, toxicity, or emotion labels as passive-aggression truth.
- All records will be independently judged by dual LLMs according to the 3-prong operational definition.
"""
    with open(OUT_REP_DIR / "eligibility_and_sampling_report.md", "w", encoding="utf-8") as f:
        f.write(md)


def generate_deduplication_report(records: List[Dict[str, Any]], dups: List[Dict[str, Any]]) -> None:
    md = f"""# Phase 10 Leakage and Deduplication Audit Report
## Cryptographic Hash Verification and Partition Isolation

**Project Title:** Context-Aware Multimodal Passive-Aggression Detection in Social Media Using Image–Comment Fusion  
**Artifact:** `reports/phase10/leakage_and_deduplication_report.md`  
**Date:** October 22, 2026  

---

## 1. Deduplication Audit Results
- **Eligible Corpus Size:** {len(records)} unique records
- **Excluded Duplicate Instances:** {len(dups)}
- **Deduplication Methods Applied:**
  1. Record ID uniqueness check
  2. Cryptographic Raw Text SHA-256 hashing
  3. Normalized Alphanumeric Lowercase SHA-256 hashing
- **Leakage Prevention:** Every record belongs to an isolated `group_id` (`grp_<record_id>`) mapped in `data/processed/phase10/duplicate_group_manifest.csv`. No duplicate group will cross partition boundaries during silver dataset splitting.
"""
    with open(OUT_REP_DIR / "leakage_and_deduplication_report.md", "w", encoding="utf-8") as f:
        f.write(md)


if __name__ == "__main__":
    recs = build_eligible_corpus(target_count=100)
    logger.info("Successfully prepared Phase 10 eligible corpus with %d records.", len(recs))
