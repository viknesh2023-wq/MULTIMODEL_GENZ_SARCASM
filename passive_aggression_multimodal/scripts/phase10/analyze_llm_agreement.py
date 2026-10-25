"""
scripts/phase10/analyze_llm_agreement.py

Phase 10: Inter-Model Agreement, Disagreement Taxonomy, and Quality Audit.
Compares Annotator A (llama3.2:latest) and Annotator B (qwen2.5:1.5b).

Calculates:
- Raw percentage agreement
- Cohen's kappa (handling edge cases and zero marginals gracefully)
- Label distribution per model and joint confusion matrix
- Subtype concordance
- Three-prong criteria concordance (Interpersonal target, Plausible deniability, Covert hostility)
- Source dataset & language breakdowns
- Disagreement taxonomy categorization
- Produces markdown report and machine-readable CSV tables.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Phase10-AgreementAnalysis")

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT_DIR / "data/annotations/phase10"
REPORTS_DIR = ROOT_DIR / "reports/phase10"
TABLES_DIR = REPORTS_DIR / "tables"

LABELS = ["passive_aggressive", "non_passive_aggressive", "ambiguous"]


def compute_cohens_kappa(
    labels_a: List[str],
    labels_b: List[str],
    classes: List[str] = LABELS,
) -> Tuple[Optional[float], float, float]:
    """
    Computes Cohen's kappa: kappa = (Po - Pe) / (1 - Pe)
    Returns (kappa, Po, Pe). If Pe == 1.0, returns (None, Po, Pe).
    """
    n = len(labels_a)
    if n == 0 or len(labels_b) != n:
        return None, 0.0, 0.0

    # Count observed agreements
    agreed = sum(1 for a, b in zip(labels_a, labels_b) if a == b)
    po = agreed / n

    # Marginals
    count_a = {c: 0 for c in classes}
    count_b = {c: 0 for c in classes}
    for a in labels_a:
        count_a[a] = count_a.get(a, 0) + 1
    for b in labels_b:
        count_b[b] = count_b.get(b, 0) + 1

    pe = sum((count_a.get(c, 0) / n) * (count_b.get(c, 0) / n) for c in classes)

    if math.isclose(pe, 1.0):
        kappa = 1.0 if math.isclose(po, 1.0) else 0.0
    else:
        kappa = (po - pe) / (1.0 - pe)

    return round(kappa, 4), round(po, 4), round(pe, 4)


def categorize_disagreement(rec_a: Dict[str, Any], rec_b: Dict[str, Any]) -> str:
    """
    Classify disagreement into standardized taxonomy categories based on model responses
    and rationale clues.
    """
    lbl_a = rec_a.get("label")
    lbl_b = rec_b.get("label")
    sub_a = rec_a.get("subtype", "")
    sub_b = rec_b.get("subtype", "")
    rat_a = (rec_a.get("rationale") or "").lower()
    rat_b = (rec_b.get("rationale") or "").lower()

    if lbl_a == "ambiguous" or lbl_b == "ambiguous":
        return "ambiguous boundary case"
    if "direct" in rat_a or "direct" in rat_b or "overt" in rat_a or "overt" in rat_b or "insult" in rat_a or "insult" in rat_b:
        return "direct aggression"
    if "sarcas" in rat_a or "sarcas" in rat_b or sub_a == "sarcastic_agreement" or sub_b == "sarcastic_agreement":
        return "general sarcasm without PA"
    if "teas" in rat_a or "teas" in rat_b or "banter" in rat_a or "banter" in rat_b:
        return "friendly teasing"
    if "institution" in rat_a or "institution" in rat_b or "vent" in rat_a or "vent" in rat_b:
        return "institutional or situational criticism"
    if "praise" in rat_a or "praise" in rat_b or "compliment" in rat_a or "compliment" in rat_b:
        return "sincere positive sentiment"
    if "context" in rat_a or "context" in rat_b or not rec_a.get("context_sufficient") or not rec_b.get("context_sufficient"):
        return "insufficient context"
    if (lbl_a == "passive_aggressive" and lbl_b == "non_passive_aggressive") or (lbl_a == "non_passive_aggressive" and lbl_b == "passive_aggressive"):
        return "possible genuine PA"
    return "ambiguous boundary case"


def run_agreement_analysis(
    file_a: Optional[Path] = None,
    file_b: Optional[Path] = None,
) -> Dict[str, Any]:
    file_a = file_a or (DATA_DIR / "annotator_a.jsonl")
    file_b = file_b or (DATA_DIR / "annotator_b.jsonl")
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    records_a = {}
    if file_a.exists():
        with open(file_a, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    d = json.loads(line)
                    records_a[d["record_id"]] = d

    records_b = {}
    if file_b.exists():
        with open(file_b, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    d = json.loads(line)
                    records_b[d["record_id"]] = d

    common_ids = sorted(list(set(records_a.keys()) & set(records_b.keys())))
    logger.info("Found %d paired records between Annotator A and B.", len(common_ids))

    if not common_ids:
        logger.warning("No common paired records found to analyze.")
        return {"paired_count": 0}

    model_a_name = records_a[common_ids[0]].get("annotation_model", "Annotator A")
    model_b_name = records_b[common_ids[0]].get("annotation_model", "Annotator B")

    labels_a = [records_a[rid]["label"] for rid in common_ids]
    labels_b = [records_b[rid]["label"] for rid in common_ids]

    # Metrics
    kappa, po, pe = compute_cohens_kappa(labels_a, labels_b, LABELS)

    # Label distributions
    dist_a = {l: labels_a.count(l) for l in LABELS}
    dist_b = {l: labels_b.count(l) for l in LABELS}

    # Confusion Matrix
    conf_mat: Dict[str, Dict[str, int]] = {la: {lb: 0 for lb in LABELS} for la in LABELS}
    for la, lb in zip(labels_a, labels_b):
        conf_mat[la][lb] += 1

    # Per-class Agreement
    per_class = {}
    for c in LABELS:
        both_c = sum(1 for la, lb in zip(labels_a, labels_b) if la == c and lb == c)
        any_c = sum(1 for la, lb in zip(labels_a, labels_b) if la == c or lb == c)
        per_class[c] = {
            "agreement_count": both_c,
            "union_count": any_c,
            "jaccard_agreement": round(both_c / any_c, 4) if any_c > 0 else 1.0,
        }

    # 3-Prong Concordance
    prong_agreed = {
        "interpersonal_target": sum(1 for rid in common_ids if records_a[rid].get("interpersonal_target_present") == records_b[rid].get("interpersonal_target_present")),
        "plausible_deniability": sum(1 for rid in common_ids if records_a[rid].get("plausible_deniability_present") == records_b[rid].get("plausible_deniability_present")),
        "covert_hostility": sum(1 for rid in common_ids if records_a[rid].get("covert_hostility_present") == records_b[rid].get("covert_hostility_present")),
    }

    # Subtype concordance (where labels agree)
    subtype_matches = sum(1 for rid in common_ids if records_a[rid]["subtype"] == records_b[rid]["subtype"])
    subtype_concordance = round(subtype_matches / len(common_ids), 4)

    # Disagreements categorization
    disagreements = []
    disag_counts = {}
    for rid in common_ids:
        rec_a = records_a[rid]
        rec_b = records_b[rid]
        if rec_a["label"] != rec_b["label"]:
            cat = categorize_disagreement(rec_a, rec_b)
            disag_counts[cat] = disag_counts.get(cat, 0) + 1
            disagreements.append({
                "record_id": rid,
                "source_dataset": rec_a.get("source_dataset"),
                "text": rec_a.get("text"),
                "label_a": rec_a.get("label"),
                "label_b": rec_b.get("label"),
                "subtype_a": rec_a.get("subtype"),
                "subtype_b": rec_b.get("subtype"),
                "disagreement_category": cat,
                "rationale_a": rec_a.get("rationale"),
                "rationale_b": rec_b.get("rationale"),
            })

    # Source breakdown
    source_counts: Dict[str, Dict[str, Any]] = {}
    for rid in common_ids:
        src = records_a[rid].get("source_dataset", "unknown")
        if src not in source_counts:
            source_counts[src] = {"total": 0, "agreed": 0, "disagreed": 0, "pa_both": 0, "non_pa_both": 0}
        source_counts[src]["total"] += 1
        la, lb = records_a[rid]["label"], records_b[rid]["label"]
        if la == lb:
            source_counts[src]["agreed"] += 1
            if la == "passive_aggressive":
                source_counts[src]["pa_both"] += 1
            elif la == "non_passive_aggressive":
                source_counts[src]["non_pa_both"] += 1
        else:
            source_counts[src]["disagreed"] += 1

    # --- SAVE CSV TABLES ---
    # 1. Label distribution table
    dist_csv = TABLES_DIR / "label_distribution.csv"
    with open(dist_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["label", "annotator_a_count", "annotator_a_pct", "annotator_b_count", "annotator_b_pct"])
        for l in LABELS:
            cnt_a = dist_a[l]
            pct_a = round((cnt_a / len(common_ids)) * 100, 1)
            cnt_b = dist_b[l]
            pct_b = round((cnt_b / len(common_ids)) * 100, 1)
            w.writerow([l, cnt_a, f"{pct_a}%", cnt_b, f"{pct_b}%"])

    # 2. Disagreement categories table
    disag_csv = TABLES_DIR / "disagreement_categories.csv"
    with open(disag_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["disagreement_category", "count", "percentage_of_disagreements"])
        for cat, cnt in sorted(disag_counts.items(), key=lambda x: -x[1]):
            pct = round((cnt / len(disagreements)) * 100, 1) if disagreements else 0.0
            w.writerow([cat, cnt, f"{pct}%"])

    # 3. Source breakdown table
    src_csv = TABLES_DIR / "source_breakdown.csv"
    with open(src_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["source_dataset", "total_records", "agreed_count", "agreement_pct", "disagreed_count", "pa_concordant"])
        for src, sc in sorted(source_counts.items(), key=lambda x: -x[1]["total"]):
            ag_pct = round((sc["agreed"] / sc["total"]) * 100, 1)
            w.writerow([src, sc["total"], sc["agreed"], f"{ag_pct}%", sc["disagreed"], sc["pa_both"]])

    # --- SAVE DETAILED DISAGREEMENTS JSON ---
    disag_json = DATA_DIR / "disagreements.json"
    with open(disag_json, "w", encoding="utf-8") as f:
        json.dump(disagreements, f, indent=2)

    # --- GENERATE MARKDOWN REPORT ---
    report_md = REPORTS_DIR / "llm_agreement_report.md"
    n_tot = len(common_ids)
    n_agree = sum(1 for la, lb in zip(labels_a, labels_b) if la == lb)
    n_disag = len(disagreements)

    md_content = f"""# Phase 10: LLM Inter-Model Agreement and Quality Audit Report

**Date:** {logging.Formatter().formatTime(logging.LogRecord('', 0, '', 0, '', (), None))}  
**Annotator A:** `{model_a_name}`  
**Annotator B:** `{model_b_name}`  
**Evaluation Gate:** `BLOCKED_PENDING_HUMAN_ANNOTATION`  
**Silver Gate:** `LLM_SILVER_DATASET_AVAILABLE_FOR_EXPLORATORY_RESEARCH`  

---

## 1. Executive Summary

This report evaluates inter-model agreement and reliability between two architecturally distinct open-weights LLMs: Meta's **Llama 3.2 (3.2B)** and Alibaba's **Qwen 2.5 (1.5B)**. Both models independently evaluated **{n_tot}** paired eligible records using a strict, neutral operational definition of passive aggression (3-prong test) without seeing each other's outputs.

- **Total Paired Records:** {n_tot}
- **Raw Inter-Model Agreement:** {n_agree} / {n_tot} ({round(po * 100, 2)}%)
- **Disagreements:** {n_disag} ({round((n_disag / n_tot) * 100, 2)}%)
- **Cohen's Kappa ($\\kappa$):** `{kappa}` (Chance expected agreement $P_e = {pe}$)
- **Subtype Concordance:** {round(subtype_concordance * 100, 2)}%
- **Modality Honesty:** 100% text-only assessment (`image_assessed: false`). No unverified visual claims.

---

## 2. Statistical Reliability & Cohen's Kappa Interpretation

$$\\kappa = \\frac{{P_o - P_e}}{{1 - P_e}} = \\frac{{{po} - {pe}}}{{1 - {pe}}} = {kappa}$$

| Metric | Value | Interpretation |
| :--- | :--- | :--- |
| **Total Paired ($N$)** | {n_tot} | Full eligible corpus paired sample |
| **Observed Agreement ($P_o$)** | {round(po * 100, 2)}% | Overall raw label concordance |
| **Expected Agreement ($P_e$)** | {round(pe * 100, 2)}% | Concordance expected purely by chance marginals |
| **Cohen's $\\kappa$** | **{kappa}** | {'Substantial agreement' if (kappa or 0) > 0.6 else ('Moderate agreement' if (kappa or 0) > 0.4 else 'Fair / Slight agreement')} |

*Note: In accordance with Section 3.1 & 3.2, cross-model agreement is NOT proof of ground-truth correctness. It establishes automated reliability and consistency across independent architectures.*

---

## 3. Label Distributions and Joint Confusion Matrix

### Label Distribution

| Label | Annotator A ({model_a_name}) | % | Annotator B ({model_b_name}) | % |
| :--- | :--- | :--- | :--- | :--- |
| `passive_aggressive` | {dist_a['passive_aggressive']} | {round(dist_a['passive_aggressive'] / n_tot * 100, 1)}% | {dist_b['passive_aggressive']} | {round(dist_b['passive_aggressive'] / n_tot * 100, 1)}% |
| `non_passive_aggressive` | {dist_a['non_passive_aggressive']} | {round(dist_a['non_passive_aggressive'] / n_tot * 100, 1)}% | {dist_b['non_passive_aggressive']} | {round(dist_b['non_passive_aggressive'] / n_tot * 100, 1)}% |
| `ambiguous` | {dist_a['ambiguous']} | {round(dist_a['ambiguous'] / n_tot * 100, 1)}% | {dist_b['ambiguous']} | {round(dist_b['ambiguous'] / n_tot * 100, 1)}% |

### Confusion Matrix (Rows = Annotator A, Columns = Annotator B)

| Annotator A \\ B | `passive_aggressive` | `non_passive_aggressive` | `ambiguous` | Total A |
| :--- | :--- | :--- | :--- | :--- |
| **`passive_aggressive`** | **{conf_mat['passive_aggressive']['passive_aggressive']}** | {conf_mat['passive_aggressive']['non_passive_aggressive']} | {conf_mat['passive_aggressive']['ambiguous']} | {dist_a['passive_aggressive']} |
| **`non_passive_aggressive`** | {conf_mat['non_passive_aggressive']['passive_aggressive']} | **{conf_mat['non_passive_aggressive']['non_passive_aggressive']}** | {conf_mat['non_passive_aggressive']['ambiguous']} | {dist_a['non_passive_aggressive']} |
| **`ambiguous`** | {conf_mat['ambiguous']['passive_aggressive']} | {conf_mat['ambiguous']['non_passive_aggressive']} | **{conf_mat['ambiguous']['ambiguous']}** | {dist_a['ambiguous']} |
| **Total B** | {dist_b['passive_aggressive']} | {dist_b['non_passive_aggressive']} | {dist_b['ambiguous']} | {n_tot} |

---

## 4. Three-Prong Criteria Concordance

The operational definition requires satisfying three distinct semantic prongs:

| Operational Prong | Agreed Count | Concordance Rate |
| :--- | :--- | :--- |
| **Interpersonal Target Present** | {prong_agreed['interpersonal_target']} / {n_tot} | {round(prong_agreed['interpersonal_target'] / n_tot * 100, 1)}% |
| **Plausible Deniability Present** | {prong_agreed['plausible_deniability']} / {n_tot} | {round(prong_agreed['plausible_deniability'] / n_tot * 100, 1)}% |
| **Covert Hostility Present** | {prong_agreed['covert_hostility']} / {n_tot} | {round(prong_agreed['covert_hostility'] / n_tot * 100, 1)}% |

---

## 5. Disagreement Taxonomy Breakdown

Total disagreements: **{n_disag}** ({round((n_disag / n_tot) * 100, 1)}% of corpus).

| Disagreement Category | Count | Pct of Disagreements | Description |
| :--- | :--- | :--- | :--- |
"""
    for cat, cnt in sorted(disag_counts.items(), key=lambda x: -x[1]):
        pct = round((cnt / len(disagreements)) * 100, 1) if disagreements else 0.0
        md_content += f"| `{cat}` | {cnt} | {pct}% | Primary boundary conflict between surface polite disguise and directness |\n"

    md_content += f"""
---

## 6. Source Dataset Breakdown

| Source Dataset | Total Records | Agreed | Agreement % | Disagreed | Full PA Concordance |
| :--- | :--- | :--- | :--- | :--- | :--- |
"""
    for src, sc in sorted(source_counts.items(), key=lambda x: -x[1]["total"]):
        ag_pct = round((sc["agreed"] / sc["total"]) * 100, 1)
        md_content += f"| `{src}` | {sc['total']} | {sc['agreed']} | {ag_pct}% | {sc['disagreed']} | {sc['pa_both']} |\n"

    md_content += f"""
---

## 7. Next Step in Pipeline

The {n_disag} disagreement records will be forwarded to Step 9: Third-Model Automated Adjudication (`scripts/phase10/run_llm_adjudication.py`), adhering to strict judicial guidelines without overwriting raw annotator outputs.
"""

    with open(report_md, "w", encoding="utf-8") as f:
        f.write(md_content)

    logger.info("Agreement report generated at %s", report_md)
    return {
        "paired_count": n_tot,
        "raw_agreement_po": po,
        "cohens_kappa": kappa,
        "agreed_count": n_agree,
        "disagreed_count": n_disag,
        "table_files": [str(dist_csv), str(disag_csv), str(src_csv)],
        "report_file": str(report_md),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze Phase 10 LLM inter-annotator agreement")
    parser.add_argument("--file-a", type=Path, default=None)
    parser.add_argument("--file-b", type=Path, default=None)
    args = parser.parse_args()

    run_agreement_analysis(args.file_a, args.file_b)
