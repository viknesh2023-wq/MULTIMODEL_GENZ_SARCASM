import json
import csv

manifest = [json.loads(line) for line in open('data/annotations/phase4/human_sampling_manifest_100.jsonl', encoding='utf-8')]
p1 = {json.loads(line)['record_id']: json.loads(line) for line in open('data/annotations/phase5a/llm/llm_annotations_pass1.jsonl', encoding='utf-8').read().strip().split('\n')}
p2 = {json.loads(line)['record_id']: json.loads(line) for line in open('data/annotations/phase5a/llm/llm_annotations_pass2.jsonl', encoding='utf-8').read().strip().split('\n')}
p5d_ver = {json.loads(l)['record_id']: json.loads(l) for l in open('data/annotations/phase5d/silver_dataset_verified.jsonl', encoding='utf-8').read().strip().split('\n')}
p5e_agree = {r['record_id']: r for r in csv.DictReader(open('reports/phase5e/tables/agreement_audit_68_complete.csv', encoding='utf-8'))}

# Look at disagreements
disagreements = [r for r in manifest if p1[r['record_id']]['primary_label'] != p2[r['record_id']]['primary_label']]
print(f"Total disagreements in manifest: {len(disagreements)}")

for d in disagreements[:10]:
    rid = d['record_id']
    print(f"Disagreement {rid} ({d['source_dataset']}) | Llama: {p1[rid]['primary_label']} | Qwen: {p2[rid]['primary_label']} | Text: {d['comment_text'][:60].encode('ascii', 'replace').decode('ascii')}")
