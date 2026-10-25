"""
Test Prompt v2.0.0 on a documented 12-record diagnostic subset.

Processes all records through Llama 3.2 first, then all records through Qwen 2.5
to avoid VRAM model swapping thrash.
"""
import json
import os
import sys
import time
import urllib.request
import urllib.error

sys.stdout.reconfigure(encoding="utf-8")

PROMPT_V2_SYSTEM = """You are an expert computational linguist and pragmatic dialogue auditor. Your task is to objectively evaluate social media comments and distinguish genuine PASSIVE AGGRESSION from other forms of communication (direct aggression, situational sarcasm, news reports, sincere praise, or ordinary debate).

### 1. OPERATIONAL DEFINITION OF PASSIVE AGGRESSION
Passive aggression is strictly defined as:
INDIRECT COMMUNICATION THAT CONVEYS COVERT INTERPERSONAL HOSTILITY, BLAME, RESENTMENT, OR CRITICISM DISGUISED BENEATH A SURFACE APPEARANCE OF POLITENESS, COMPLIANCE, PRAISE, OR INNOCUOUS NEUTRALITY.

A comment is PASSIVE AGGRESSIVE IF AND ONLY IF all three criteria are met:
1. Interpersonal Target: Hostility or blame is directed at an individual or conversation partner (NOT an external situation, weather, transit delay, or general state of the world).
2. Plausible Deniability: The literal words appear polite, praising, neutral, or deferential on the surface, allowing the speaker to deny bad intent if questioned ("I was just paying you a compliment!").
3. Covert Hostile Intent: The pragmatic meaning undermines, demeans, mocks, or blames the target.

### 2. STRICT CONTRAST BOUNDARIES (What is NOT Passive Aggression -> Select 'non_passive_aggressive')
- DIRECT AGGRESSION & TOXICITY ("You are an idiot", "Shut up, your point makes no sense"):
  Overt hostility is NOT passive aggressive because it lacks polite concealment or plausible deniability. Label as "non_passive_aggressive".
- SITUATIONAL SARCASM & VENTING ("Great, the bus is delayed again!", "There is no wisdom left on this planet"):
  Frustration aimed at inanimate objects, institutions, bad luck, or the universe without an interpersonal attack. Label as "non_passive_aggressive".
- FACTUAL NEWS, HEADLINES & DESCRIPTIONS ("result: glen durrant edges out paul hogan 4-3", airport codes, dates):
  Objective reporting of events or facts. Label as "non_passive_aggressive".
- SINCERE PRAISE & POSITIVE SENTIMENT ("I love this photo!", "Great job everyone!"):
  Genuine admiration without a backhanded sting or insult. Label as "non_passive_aggressive".
- DIRECT DISAGREEMENT, BLUNT STATEMENTS & RHETORICAL QUESTIONS ("Did you read the article?", "Prices will always go up"):
  Blunt disagreement or factual correction without faux courtesy is direct debate. Label as "non_passive_aggressive".
- FRIENDLY TEASING / BANTER ("Look who finally woke up!"):
  Playful social interaction between peers without malicious intent. Label as "non_passive_aggressive".

### 3. AMBIGUOUS CASES (When to Select 'ambiguous')
- INSUFFICIENT CONTEXT: If the comment is an isolated phrase ("not all girls...", "* gasp * oh no"), a meme reference without image, or where target and intent cannot be reasonably determined without missing context, label as "ambiguous".

### 4. PRIMARY LABELS (Choose Exactly One)
- "passive_aggressive": Clear evidence of covert interpersonal hostility masked by faux politeness/neutrality.
- "non_passive_aggressive": Direct hostility, situational sarcasm, factual reports, sincere praise, blunt argument, or friendly teasing.
- "ambiguous": Unclear intent due to missing conversational context or unassessed image.

### 5. SUBTYPES (Select ONE if passive_aggressive; must be 'none' if non_passive_aggressive or ambiguous)
- "fake_politeness": Exaggerated formal courtesy masking contempt ("Per my previous email...").
- "indirect_criticism": Criticizing an individual without naming them directly ("Interesting how some people never clean up...").
- "backhanded_compliment": A compliment paired with an insult or diminishing caveat ("You look surprisingly competent today!").
- "implied_blame": Subtly shifting guilt onto an individual under neutral phrasing ("Must be nice to have so much free time while others work...").
- "sarcastic_agreement": Feigned enthusiastic agreement to mock the recipient's stance ("Oh absolutely, you know better than the doctor!").
- "dismissiveness": Minimization or contemptuous brevity intended to belittle ("Whatever makes you feel better", "K.").
- "other_or_unclear": Other forms of covert interpersonal hostility.
- "none": When primary_label is "non_passive_aggressive" or "ambiguous".

### 6. PRAGMATIC REASONING PROTOCOL
In your "brief_rationale", you must explicitly analyze:
1. The literal surface meaning.
2. Whether an interpersonal target is being attacked.
3. Whether the attack is covert (disguised by surface politeness) or overt/absent.
"""

SCHEMA_V2 = {
    "type": "object",
    "properties": {
        "brief_rationale": {
            "type": "string",
            "description": "Pragmatic reasoning: surface meaning, interpersonal target, and covert vs overt/absent hostility."
        },
        "evidence_span": {
            "type": "string",
            "description": "Quoted words indicating covert hostility, or 'none' if non-PA / neutral."
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

def build_v2_user_prompt(record: dict) -> str:
    rec_id = record.get("record_id", "")
    src = record.get("source_dataset", "")
    text = record.get("comment_text", "").strip()
    ctx = record.get("context_text", "").strip()
    img_avail = record.get("image_available", False)

    return (
        f"Evaluate the following social media post for passive aggression according to the calibrated criteria:\n\n"
        f"Record ID: {rec_id}\n"
        f"Source Dataset: {src}\n"
        f"Comment Text: \"{text}\"\n"
        f"Dialogue Context: {ctx if ctx else 'None'}\n"
        f"Image Availability: {img_avail} (Note: Image pixels unassessed; evaluate text and context)\n\n"
        f"Remember:\n"
        f"- News headlines, factual statements, direct insults, and sincere praise are NOT passive-aggressive.\n"
        f"- Do NOT assume covert hostility unless literal wording is polite/neutral but pragmatic intent is hostile towards an individual.\n"
        f"- If context is missing to understand the target or tone, select 'ambiguous'.\n\n"
        f"Return valid JSON adhering to the required schema."
    )

def query_ollama(model: str, user_prompt: str, system_prompt: str = PROMPT_V2_SYSTEM, retries: int = 3) -> dict:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "format": SCHEMA_V2,
        "stream": False,
        "options": {"temperature": 0.0},
    }
    
    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                "http://localhost:11434/api/chat",
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=90) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                content = data.get("message", {}).get("content", "")
                return json.loads(content)
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2)
            else:
                raise e

def main():
    target_ids = [
        "mmsd_test_02285",         # Sports news
        "mmsd_test_01815",         # Sincere admiration
        "civil_comments_val_03516", # Blunt assertion ("prices will always go up")
        "civil_comments_val_02472", # Rhetorical question ("Did you read the article?")
        "mmsd_test_00104",         # "good one panda . ( insert )"
        "mmsd_train_05348",        # Existential venting ("i should leave this planet morty")
        "mmsd_test_00277",         # Missing context ("* gasp * oh no ... not the grandpa ....")
        "mmsd_train_02546",        # Trailing ellipsis ("not all girls ....")
        "mustard_2_490",           # Real PA ("Maybe it's a contest, you know? Like 'Collect all five'")
        "sarcnet_test_00274",       # Real PA ("Maybe now someone in the Federal Govt and MSM will start paying attention to them.")
        "mmsd_train_16580",        # Baseline Non-PA ("2017012 1 haneda airport")
        "mmsd_validation_02047",   # Baseline Non-PA ("the glimmer twins")
    ]

    sampling_path = "data/annotations/phase4/human_sampling_manifest_100.jsonl"
    p1_path = "data/annotations/phase5a/llm/llm_annotations_pass1.jsonl"
    p2_path = "data/annotations/phase5a/llm/llm_annotations_pass2.jsonl"

    with open(sampling_path, "r", encoding="utf-8") as f:
        samples = {r["record_id"]: r for r in (json.loads(l) for l in f if l.strip())}
    with open(p1_path, "r", encoding="utf-8") as f:
        p1 = {r["record_id"]: r for r in (json.loads(l) for l in f if l.strip())}
    with open(p2_path, "r", encoding="utf-8") as f:
        p2 = {r["record_id"]: r for r in (json.loads(l) for l in f if l.strip())}

    print("=== PASS 1: LLAMA 3.2 UNDER PROMPT V2 ===")
    llama_results = {}
    for i, rid in enumerate(target_ids):
        rec = samples[rid]
        prompt = build_v2_user_prompt(rec)
        print(f"[{i+1}/{len(target_ids)}] Llama 3.2: {rid}...", end=" ", flush=True)
        res = query_ollama("llama3.2:latest", prompt)
        llama_results[rid] = res
        print(f"{res.get('primary_label')} ({res.get('subtype')})")

    print("\n=== PASS 2: QWEN 2.5 UNDER PROMPT V2 ===")
    qwen_results = {}
    for i, rid in enumerate(target_ids):
        rec = samples[rid]
        prompt = build_v2_user_prompt(rec)
        print(f"[{i+1}/{len(target_ids)}] Qwen 2.5: {rid}...", end=" ", flush=True)
        res = query_ollama("qwen2.5:1.5b", prompt)
        qwen_results[rid] = res
        print(f"{res.get('primary_label')} ({res.get('subtype')})")

    comparison = []
    for rid in target_ids:
        rec = samples[rid]
        v1_l = p1.get(rid, {})
        v1_q = p2.get(rid, {})
        v2_l = llama_results.get(rid, {})
        v2_q = qwen_results.get(rid, {})

        comparison.append({
            "record_id": rid,
            "comment_text": rec["comment_text"],
            "v1_llama": f"{v1_l.get('primary_label')} ({v1_l.get('subtype')})",
            "v2_llama": f"{v2_l.get('primary_label')} ({v2_l.get('subtype')})",
            "v2_llama_rationale": v2_l.get("brief_rationale"),
            "v1_qwen": f"{v1_q.get('primary_label')} ({v1_q.get('subtype')})",
            "v2_qwen": f"{v2_q.get('primary_label')} ({v2_q.get('subtype')})",
            "v2_qwen_rationale": v2_q.get("brief_rationale"),
        })

    out_path = "data/annotations/phase5b/prompt_v2_subset_test.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2, ensure_ascii=False)
    print(f"\nSaved comparison to {out_path}")

if __name__ == "__main__":
    main()
