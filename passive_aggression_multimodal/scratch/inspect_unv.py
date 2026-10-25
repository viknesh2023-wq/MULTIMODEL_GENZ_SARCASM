import json

with open('scratch/records_68_full.json', encoding='utf-8') as f:
    records = json.load(f)

unv = [r for r in records if r['p5d_status'] == 'EXCLUDED_UNVERIFIED_TARGET']

for r in unv:
    has_img = r['image_available']
    rid = r['record_id']
    ds = r['source_dataset']
    txt = r['comment_text'].replace('\n', ' ')[:60]
    # encode safe print
    print(f"{rid:26} | img={str(has_img):5} | ds={ds:15} | text: {txt.encode('ascii', 'replace').decode('ascii')}")
