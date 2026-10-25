import csv
import json
from pathlib import Path

MANIFEST_100_FILE = Path("data/annotations/phase4/human_sampling_manifest_100.jsonl")
P1_FILE = Path("data/annotations/phase5a/llm/llm_annotations_pass1.jsonl")
P2_FILE = Path("data/annotations/phase5a/llm/llm_annotations_pass2.jsonl")
P5D_VER_FILE = Path("data/annotations/phase5d/silver_dataset_verified.jsonl")
P5D_EXC_FILE = Path("data/annotations/phase5d/excluded_records.csv")
P5E_AGREE_FILE = Path("reports/phase5e/tables/agreement_audit_68_complete.csv")

OUT_DIR = Path("data/annotations/phase6a")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Load references
manifest_map = {json.loads(line)["record_id"]: json.loads(line) for line in MANIFEST_100_FILE.read_text(encoding="utf-8").strip().split("\n")}
p1_map = {json.loads(line)["record_id"]: json.loads(line) for line in P1_FILE.read_text(encoding="utf-8").strip().split("\n")}
p2_map = {json.loads(line)["record_id"]: json.loads(line) for line in P2_FILE.read_text(encoding="utf-8").strip().split("\n")}
p5d_ver_map = {json.loads(line)["record_id"]: json.loads(line) for line in P5D_VER_FILE.read_text(encoding="utf-8").strip().split("\n")}
p5d_exc_map = {r["record_id"]: r for r in csv.DictReader(P5D_EXC_FILE.open(encoding="utf-8"))}
p5e_agree_map = {r["record_id"]: r for r in csv.DictReader(P5E_AGREE_FILE.open(encoding="utf-8"))}

# Define the 30 sampled records with sampling archetype and rationale
selected_specs = [
    # 1. LLM Silver-Positive Candidates (5 records)
    ("mustard_2_490", "LLM_SILVER_POSITIVE_CANDIDATE", "Classic sitcom backhanded suggestion with conversational context ('Collect all five')."),
    ("civil_comments_val_06315", "LLM_SILVER_POSITIVE_CANDIDATE", "Textbook weaponized faux courtesy ('What a sweetheart') directed at an online interlocutor."),
    ("mustard_1_8407", "LLM_SILVER_POSITIVE_CANDIDATE", "Sarcastic agreement paired with absurd scenario to mock friend's suggestion."),
    ("civil_comments_val_07561", "LLM_SILVER_POSITIVE_CANDIDATE", "Polite opening masking backhanded ridicule of user's handle ('eye test')."),
    ("mmsd_train_10780", "LLM_SILVER_POSITIVE_CANDIDATE", "Multimodal tweet with surface praise ('taking pride') and backhanded modifier implying failure."),

    # 2. Non-PA Contrast Candidates (5 records)
    ("sarcnet_test_00053", "NON_PA_CONTRAST_CANDIDATE", "Consensus negative example explaining a Chinese idiom; zero hostility or aggression."),
    ("civil_comments_val_08920", "NON_PA_CONTRAST_CANDIDATE", "Sincere praise ('thank you for this fine article'); critical false-positive contrast example."),
    ("mustard_2_372", "NON_PA_CONTRAST_CANDIDATE", "Sitcom character self-confession about smoking; no interpersonal aggression."),
    ("mustard_1_1470", "NON_PA_CONTRAST_CANDIDATE", "Factual technical dialogue reporting radio static; Model A/B disagreement."),
    ("mmsd_test_00104", "NON_PA_CONTRAST_CANDIDATE", "Playful teasing ('good one panda'); signals social closeness without hostility; Model A/B disagreement."),

    # 3. Direct Aggression and Direct Disagreement (5 records)
    ("civil_comments_val_04917", "DIRECT_AGGRESSION_OR_DISAGREEMENT", "Blunt political debate attacking GOP policy on Social Security lacking politeness facade."),
    ("civil_comments_val_02587", "DIRECT_AGGRESSION_OR_DISAGREEMENT", "Direct confrontational argumentation ('Bob, Trump has done a fine job...')."),
    ("civil_comments_val_00154", "DIRECT_AGGRESSION_OR_DISAGREEMENT", "Overt verbal aggression lacking plausible deniability; contrast class for overt toxicity."),
    ("civil_comments_val_05781", "DIRECT_AGGRESSION_OR_DISAGREEMENT", "Direct hostile attack on interlocutor; direct hostility contrast class."),
    ("mmsd_train_02546", "DIRECT_AGGRESSION_OR_DISAGREEMENT", "Direct dispute ('not all girls ....'); Model A/B disagreement on ambiguous boundary."),

    # 4. General Sarcasm and Institutional Critique (5 records)
    ("mustard_2_174", "GENERAL_SARCASM_INSTITUTIONAL_CRITIQUE", "Satire aimed at television institution/workplace ('Satan\\'s minions at work again?')."),
    ("sarcnet_test_00274", "GENERAL_SARCASM_INSTITUTIONAL_CRITIQUE", "English political critique directed at Federal Govt and MSM without individual addressee."),
    ("civil_comments_val_09401", "GENERAL_SARCASM_INSTITUTIONAL_CRITIQUE", "Discussion of NFL policies; institutional grievance contrast class."),
    ("mmsd_train_05348", "GENERAL_SARCASM_INSTITUTIONAL_CRITIQUE", "Pop-culture existential venting ('i should leave this planet morty'); no addressee."),
    ("civil_comments_val_00462", "GENERAL_SARCASM_INSTITUTIONAL_CRITIQUE", "Media critique aimed at Fox News; institutional satire contrast class."),

    # 5. Ambiguous Examples Requiring Context (5 records)
    ("civil_comments_val_00793", "AMBIGUOUS_CONTEXT_DEPENDENT", "Dangling fragment ('...... but would immediately respond') lacking surrounding thread context."),
    ("mmsd_train_06126", "AMBIGUOUS_CONTEXT_DEPENDENT", "Ambiguous statement ('so glad i have friends willing to help') requiring context to determine sarcasm."),
    ("mmsd_train_03612", "AMBIGUOUS_CONTEXT_DEPENDENT", "Truncated tweet fragment testing context insufficiency boundary."),
    ("mustard_2_111", "AMBIGUOUS_CONTEXT_DEPENDENT", "Sitcom exclamation ('What was that?!') ambiguous without visual/dialogue context."),
    ("mmsd_validation_02127", "AMBIGUOUS_CONTEXT_DEPENDENT", "Short phrase fragment requiring surrounding narrative grounding."),

    # 6. Image-Containing Multimodal Posts (5 records)
    ("mmsd_validation_01859", "MULTIMODAL_IMAGE_DEPENDENT", "Pogo stick backflip stunt tweet where image provides crucial situational ground truth."),
    ("mmsd_test_00916", "MULTIMODAL_IMAGE_DEPENDENT", "Bot-accusation tweet where accompanying screenshot provides evidence of follower count."),
    ("sarcnet_train_00095", "MULTIMODAL_IMAGE_DEPENDENT", "Chinese Weibo post accompanied by photo where visual scene clarifies the ironic target."),
    ("mmsd_train_14369", "MULTIMODAL_IMAGE_DEPENDENT", "Dialogue referencing person whose identity/demeanor is captured in the image."),
    ("sarcnet_train_00844", "MULTIMODAL_IMAGE_DEPENDENT", "Feigned ignorance Weibo post ('#鼠鼠我鸭#') where image of food object establishes irony.")
]

assert len(selected_specs) == 30, f"Expected exactly 30 records, got {len(selected_specs)}"

# 1. Build Pilot Sampling Manifest
manifest_rows = []
for idx, (rid, archetype, rationale) in enumerate(selected_specs, 1):
    m = manifest_map[rid]
    dataset = m["source_dataset"]
    prefix = f"{dataset}_"
    source_rid = rid[len(prefix):] if rid.startswith(prefix) else rid
    text = m["comment_text"]
    context = m.get("context_text", "")
    img_avail = bool(m.get("image_available", False))
    img_path = m.get("image_path", "")
    
    # Language check
    is_chinese = any('\u4e00' <= char <= '\u9fff' for char in text)
    lang = "zh" if is_chinese else "en"
    lang_scope = "NON_ENGLISH_SEPARATE_ANALYSIS" if is_chinese else "CORE_ENGLISH_BENCHMARK"
    
    # Model agreement status from Phase 5A
    p1_label = p1_map[rid]["primary_label"]
    p2_label = p2_map[rid]["primary_label"]
    agree_status = "EXACT_MATCH" if p1_label == p2_label else "DISAGREEMENT"
    
    # Phase 5D & Phase 5E status
    in_p5d_ver = rid in p5d_ver_map
    p5d_st = f"VERIFIED_{p5d_ver_map[rid]['silver_primary_label']}" if in_p5d_ver else p5d_exc_map[rid]["exclusion_category"]
    p5e_cat = p5e_agree_map[rid]["audit_category"] if rid in p5e_agree_map else "DISAGREEMENT_IN_P5A"

    manifest_rows.append({
        "sample_index": idx,
        "record_id": rid,
        "source_dataset": dataset,
        "source_record_id": source_rid,
        "comment_text": text,
        "context_text": context,
        "image_available": img_avail,
        "image_path": img_path,
        "language": lang,
        "language_scope": lang_scope,
        "sampling_archetype": archetype,
        "selection_rationale": rationale,
        "phase5a_agreement_status": agree_status,
        "phase5d_status": p5d_st,
        "phase5e_category": p5e_cat
    })

manifest_fields = [
    "sample_index", "record_id", "source_dataset", "source_record_id",
    "comment_text", "context_text", "image_available", "image_path",
    "language", "language_scope", "sampling_archetype", "selection_rationale",
    "phase5a_agreement_status", "phase5d_status", "phase5e_category"
]

manifest_csv_path = OUT_DIR / "pilot_sampling_manifest.csv"
with open(manifest_csv_path, mode="w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=manifest_fields)
    writer.writeheader()
    writer.writerows(manifest_rows)
print(f"Created {manifest_csv_path} with {len(manifest_rows)} records.")

# 2. Build Annotator 1 and Annotator 2 Response Templates (Double-Blind)
annotator_fields = [
    "sample_index", "record_id", "source_dataset", "comment_text",
    "context_text", "image_available", "image_path", "language", "language_scope",
    "primary_label", "subtype", "target_evidence", "covert_hostility_evidence",
    "plausible_deniability_evidence", "context_sufficiency", "image_reviewed",
    "image_role", "confidence", "evidence_span", "annotator_id",
    "annotation_timestamp", "review_status", "notes"
]

for ann_id, filename in [("annotator_01", "annotator_1_responses.csv"), ("annotator_02", "annotator_2_responses.csv")]:
    ann_rows = []
    for r in manifest_rows:
        ann_rows.append({
            "sample_index": r["sample_index"],
            "record_id": r["record_id"],
            "source_dataset": r["source_dataset"],
            "comment_text": r["comment_text"],
            "context_text": r["context_text"],
            "image_available": r["image_available"],
            "image_path": r["image_path"],
            "language": r["language"],
            "language_scope": r["language_scope"],
            "primary_label": "",
            "subtype": "",
            "target_evidence": "",
            "covert_hostility_evidence": "",
            "plausible_deniability_evidence": "",
            "context_sufficiency": "",
            "image_reviewed": "",
            "image_role": "",
            "confidence": "",
            "evidence_span": "",
            "annotator_id": ann_id,
            "annotation_timestamp": "",
            "review_status": "PENDING_REVIEW",
            "notes": ""
        })
    
    ann_path = OUT_DIR / filename
    with open(ann_path, mode="w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=annotator_fields)
        writer.writeheader()
        writer.writerows(ann_rows)
    print(f"Created {ann_path} for {ann_id}.")

# 3. Build Annotation Status Manifest JSON
status_manifest = {
    "manifest_name": "phase6a_annotation_status_manifest",
    "version": "6.0.0",
    "created_date": "2026-10-21",
    "pilot_sample_size": 30,
    "archetype_distribution": {
        "LLM_SILVER_POSITIVE_CANDIDATE": 5,
        "NON_PA_CONTRAST_CANDIDATE": 5,
        "DIRECT_AGGRESSION_OR_DISAGREEMENT": 5,
        "GENERAL_SARCASM_INSTITUTIONAL_CRITIQUE": 5,
        "AMBIGUOUS_CONTEXT_DEPENDENT": 5,
        "MULTIMODAL_IMAGE_DEPENDENT": 5
    },
    "source_dataset_distribution": {
        "mustard": 6,
        "civil_comments": 10,
        "mmsd": 10,
        "sarcnet": 4
    },
    "language_coverage": {
        "CORE_ENGLISH_BENCHMARK": 27,
        "NON_ENGLISH_SEPARATE_ANALYSIS": 3
    },
    "modality_coverage": {
        "text_only": 16,
        "image_available": 14
    },
    "phase5a_agreement_distribution": {
        "EXACT_MATCH": 22,
        "DISAGREEMENT": 8
    },
    "annotator_01": {
        "assigned_file": "data/annotations/phase6a/annotator_1_responses.csv",
        "completed_count": 0,
        "status": "AWAITING_HUMAN_INPUT"
    },
    "annotator_02": {
        "assigned_file": "data/annotations/phase6a/annotator_2_responses.csv",
        "completed_count": 0,
        "status": "AWAITING_HUMAN_INPUT"
    },
    "inter_annotator_agreement": {
        "cohens_kappa": None,
        "raw_agreement_percent": None,
        "target_kappa_threshold": 0.70,
        "status": "AWAITING_COMPLETED_REVIEWS"
    },
    "adjudication": {
        "completed_count": 0,
        "unanimous_count": 0,
        "disputed_count": 0,
        "status": "AWAITING_COMPLETED_REVIEWS"
    },
    "evaluation_readiness_status": "BLOCKED_PENDING_HUMAN_ANNOTATION",
    "status_reason": "Phase 6A pilot templates prepared for Annotator 1 and Annotator 2 across 30 stratified records. Zero human reviews have been logged yet. In accordance with Phase 4/5/6 scientific rules, benchmark evaluation remains blocked until independent human annotations are recorded and adjudicated."
}

manifest_json_path = OUT_DIR / "annotation_status_manifest.json"
with open(manifest_json_path, mode="w", encoding="utf-8") as f:
    json.dump(status_manifest, f, indent=2)
print(f"Created {manifest_json_path}.")
