import json
import csv
from pathlib import Path

p1 = {json.loads(line)['record_id']: json.loads(line) for line in open('data/annotations/phase5a/llm/llm_annotations_pass1.jsonl', encoding='utf-8').read().strip().split('\n')}
p2 = {json.loads(line)['record_id']: json.loads(line) for line in open('data/annotations/phase5a/llm/llm_annotations_pass2.jsonl', encoding='utf-8').read().strip().split('\n')}
manifest = {json.loads(line)['record_id']: json.loads(line) for line in open('data/annotations/phase4/human_sampling_manifest_100.jsonl', encoding='utf-8').read().strip().split('\n')}
p5d_ver = {json.loads(l)['record_id']: json.loads(l) for l in open('data/annotations/phase5d/silver_dataset_verified.jsonl', encoding='utf-8').read().strip().split('\n')}
p5d_exc = {r['record_id']: r for r in csv.DictReader(open('data/annotations/phase5d/excluded_records.csv', encoding='utf-8'))}

agreements = [rid for rid in p1 if p1[rid]['primary_label'] == p2[rid]['primary_label']]

out_lines = []
for rid in agreements:
    m = manifest[rid]
    r1 = p1[rid]
    r2 = p2[rid]
    in_ver = rid in p5d_ver
    p5d_status = 'VERIFIED_' + p5d_ver[rid]['silver_primary_label'] if in_ver else p5d_exc[rid]['exclusion_category']
    out_lines.append('='*70)
    out_lines.append(f"ID: {rid} | Dataset: {m['source_dataset']} | P5D: {p5d_status}")
    out_lines.append(f"Text: {m['comment_text']}")
    out_lines.append(f"Context: {m.get('context_text', '')}")
    out_lines.append(f"Image available: {m.get('image_available', False)}")
    out_lines.append(f"Llama: {r1['primary_label']} ({r1.get('subtype', '')}) | Rationale: {r1.get('rationale', '')}")
    out_lines.append(f"Qwen: {r2['primary_label']} ({r2.get('subtype', '')}) | Rationale: {r2.get('rationale', '')}")

Path('scratch/inspect_68_records.txt').write_text('\n'.join(out_lines), encoding='utf-8')
print(f"Dumped {len(agreements)} records to scratch/inspect_68_records.txt")
