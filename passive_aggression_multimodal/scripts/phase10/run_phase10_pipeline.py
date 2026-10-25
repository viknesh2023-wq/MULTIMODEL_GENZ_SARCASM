"""
scripts/phase10/run_phase10_pipeline.py

Phase 10: Master End-to-End Orchestrator Pipeline.
Coordinates:
1. Eligible corpus preparation and cryptographic deduplication
2. Dual-pass independent LLM annotation (Meta Llama 3.2 vs Alibaba Qwen 2.5)
3. Agreement analysis, Cohen's kappa calculation, and disagreement taxonomy
4. Third-model automated adjudication and boundary-condition audits
5. Conservative silver-label acceptance filtering and modality segregation
6. Group-isolated, leakage-resistant dataset partitioning (train/val/diagnostic test)
7. Exploratory silver model training (TF-IDF + Logistic Regression baseline)
8. Generation of final reports, manifest registrations, and audit deliverables
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Phase10-Pipeline")

ROOT_DIR = Path(__file__).resolve().parent.parent.parent

# Import modular phase components
from scripts.phase10.prepare_eligible_corpus import run_corpus_preparation
from scripts.phase10.run_llm_annotation import run_dual_pass_annotation
from scripts.phase10.analyze_llm_agreement import run_agreement_analysis
from scripts.phase10.run_llm_adjudication import run_adjudication_pipeline
from scripts.phase10.build_silver_dataset import run_silver_dataset_construction
from scripts.phase10.train_pa_silver_baseline import run_training_and_evaluation


def run_pipeline(
    limit: int = 30,
    dry_run: bool = False,
    skip_annotation: bool = False,
    seed: int = 42,
) -> Dict[str, Any]:
    t0_pipeline = time.time()
    logger.info("================================================================================")
    logger.info("STARTING PHASE 10 FULL AUTOMATED PIPELINE (limit=%d, dry_run=%s)", limit, dry_run)
    logger.info("Primary Human Benchmark Gate: BLOCKED_PENDING_HUMAN_ANNOTATION")
    logger.info("Exploratory Silver Gate: LLM_SILVER_DATASET_AVAILABLE_FOR_EXPLORATORY_RESEARCH")
    logger.info("================================================================================")

    # 1. Corpus Preparation and Deduplication
    logger.info("\n>>> STEP 1: Corpus Preparation & Deduplication")
    corpus_stats = run_corpus_preparation()
    logger.info("Eligible corpus prepared: %d records.", corpus_stats["total_eligible"])

    # 2. Dual-Pass Independent Annotation
    if not skip_annotation:
        logger.info("\n>>> STEP 2: Dual-Pass LLM Annotation (Llama 3.2 & Qwen 2.5)")
        eligible_csv = ROOT_DIR / "data/annotations/phase10/eligible_records.csv"
        with open(eligible_csv, "r", encoding="utf-8") as f:
            records = list(csv.DictReader(f))[:limit]
        annotation_manifest = run_dual_pass_annotation(records, dry_run=dry_run)
    else:
        logger.info("\n>>> STEP 2: Dual-Pass LLM Annotation SKIPPED (reusing cached files).")

    # 3. Agreement Analysis
    logger.info("\n>>> STEP 3: Inter-Model Agreement & Quality Audit")
    agreement_results = run_agreement_analysis()

    # 4. Third-Model Adjudication & Audit
    logger.info("\n>>> STEP 4: Automated Adjudication of Disagreements")
    adjudication_manifest = run_adjudication_pipeline(dry_run=dry_run)

    # 5. Silver Dataset Acceptance & Leakage-Resistant Splits
    logger.info("\n>>> STEP 5: Silver Label Acceptance & Group-Isolated Splitting")
    split_manifest = run_silver_dataset_construction(seed=seed)

    # 6. Exploratory Model Training & Diagnostic Evaluation
    logger.info("\n>>> STEP 6: Exploratory Silver Model Training & Evaluation")
    eval_results = run_training_and_evaluation()

    total_duration = round(time.time() - t0_pipeline, 2)
    logger.info("\n================================================================================")
    logger.info("PHASE 10 PIPELINE COMPLETED SUCCESSFULLY in %.2f seconds.", total_duration)
    logger.info("================================================================================")

    summary = {
        "phase": "Phase 10 — Automated LLM Annotation & Silver Dataset Construction",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_duration_sec": total_duration,
        "eligible_corpus_size": corpus_stats["total_eligible"],
        "records_processed": limit,
        "agreement_results": agreement_results,
        "adjudication_manifest": adjudication_manifest,
        "split_manifest": split_manifest,
        "evaluation_results": eval_results["diagnostic_test_metrics"],
        "governance": {
            "human_benchmark_gate": "BLOCKED_PENDING_HUMAN_ANNOTATION",
            "silver_evaluation_gate": "LLM_SILVER_DIAGNOSTIC_EVALUATION_AVAILABLE",
            "human_files_untouched": True,
        },
    }

    results_file = ROOT_DIR / "reports/phase10/phase10_results.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info("Summary results registered at %s", results_file)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Phase 10 master end-to-end pipeline")
    parser.add_argument("--limit", type=int, default=30, help="Number of records to annotate")
    parser.add_argument("--dry-run", action="store_true", help="Run in dry-run mock mode")
    parser.add_argument("--skip-annotation", action="store_true", help="Skip annotation pass and reuse files")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for splitting")
    args = parser.parse_args()

    run_pipeline(
        limit=args.limit,
        dry_run=args.dry_run,
        skip_annotation=args.skip_annotation,
        seed=args.seed,
    )
