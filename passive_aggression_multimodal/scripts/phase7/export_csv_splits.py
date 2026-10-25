import csv
import json
from pathlib import Path
import shutil

ROOT_DIR = Path(__file__).resolve().parent.parent.parent

def main():
    # 1. Copy label_eligibility
    reg_csv = ROOT_DIR / "data/processed/phase7/label_eligibility_registry.csv"
    elig_csv = ROOT_DIR / "data/processed/phase7/label_eligibility.csv"
    shutil.copyfile(reg_csv, elig_csv)
    print("Copied label_eligibility.csv")

    # 2. Convert sarcasm splits to CSV
    splits_dir = ROOT_DIR / "data/splits/phase7"
    for split_name in ["train", "val", "test"]:
        jl_path = splits_dir / f"phase7_multimodal_sarcasm_{split_name}.jsonl"
        csv_path = splits_dir / f"sarcasm_auxiliary_{split_name}.csv"
        with open(jl_path, "r", encoding="utf-8") as f_in, open(csv_path, "w", encoding="utf-8", newline="") as f_out:
            writer = None
            for line in f_in:
                r = json.loads(line)
                r["primary_label"] = "sarcastic" if str(r.get("original_label")) in ("1", "True") else "non_sarcastic"
                r["label_provenance"] = "ORIGINAL_SOURCE_LABEL"
                r["eligible_for_experiment"] = "ELIGIBLE_FOR_AUXILIARY_TASK"
                if writer is None:
                    writer = csv.DictWriter(f_out, fieldnames=list(r.keys()))
                    writer.writeheader()
                writer.writerow(r)
        print(f"Exported sarcasm_auxiliary_{split_name}.csv")

    # 3. Convert weak supervision splits to CSV
    silver_items = []
    silver_path = ROOT_DIR / "data/annotations/phase5d/silver_dataset_verified.jsonl"
    if silver_path.exists():
        with open(silver_path, "r", encoding="utf-8") as f:
            silver_items = [json.loads(line) for line in f]

        # 14 train, 3 val, 3 test
        w_train = silver_items[:14]
        w_val = silver_items[14:17]
        w_test = silver_items[17:20]

        for name, s_list in [("train", w_train), ("val", w_val), ("test", w_test)]:
            csv_path = splits_dir / f"weak_supervision_{name}.csv"
            with open(csv_path, "w", encoding="utf-8", newline="") as f_out:
                writer = None
                for idx, item in enumerate(s_list):
                    rid = item.get("record_id", f"silver_{idx}")
                    row = {
                        "record_id": rid,
                        "source_dataset": item.get("source_dataset", "mmsd"),
                        "text": item.get("text", ""),
                        "image_path": item.get("image_path", ""),
                        "primary_label": item.get("verified_label", "passive_aggressive"),
                        "label_provenance": "MACHINE_GENERATED_LOCAL_OLLAMA_LLM",
                        "assigned_split": name,
                        "split_group_id": f"silver_grp_{rid}",
                        "eligible_for_experiment": "ELIGIBLE_FOR_EXPLORATORY_WEAK_SUPERVISION",
                        "exclusion_reason": "",
                    }
                    if writer is None:
                        writer = csv.DictWriter(f_out, fieldnames=list(row.keys()))
                        writer.writeheader()
                    writer.writerow(row)
            print(f"Exported weak_supervision_{name}.csv")

if __name__ == "__main__":
    main()
