import urllib.request
import json
import time

def test():
    schema = {
        "type": "object",
        "properties": {
            "primary_label": {"type": "string", "enum": ["passive_aggressive", "non_passive_aggressive", "ambiguous"]},
            "subtype": {"type": "string", "enum": ["fake_politeness", "indirect_criticism", "backhanded_compliment", "implied_blame", "sarcastic_agreement", "dismissiveness", "other_or_unclear", "none"]},
            "confidence": {"type": "integer", "minimum": 1, "maximum": 5},
            "evidence_span": {"type": "string"},
            "brief_rationale": {"type": "string"},
            "context_sufficiency": {"type": "string", "enum": ["SUFFICIENT", "PARTIAL", "INSUFFICIENT"]}
        },
        "required": ["primary_label", "subtype", "confidence", "evidence_span", "brief_rationale", "context_sufficiency"]
    }

    user_text = (
        "Analyze the following social media comment for passive aggression:\n\n"
        "Comment: \"Speaking of children who need protection: http://news.nationalpost.com/news/...\"\n"
        "Context: None\n\n"
        "INSTRUCTIONS:\n"
        "1. Is this comment expressing covert interpersonal hostility or resentment toward someone?\n"
        "2. If it is merely sharing a news article, venting, direct criticism, or ordinary conversation, label as 'non_passive_aggressive' with subtype 'none'.\n"
        "3. If intent is unclear, label as 'ambiguous' with subtype 'none'.\n"
        "4. Only select 'passive_aggressive' if there is clear evidence of covert, indirect hostility."
    )

    payload = {
        "model": "llama3.2:latest",
        "messages": [
            {
                "role": "system",
                "content": "You are a linguistic annotator. Classify social media text strictly according to pragmatic evidence. Output JSON adhering to schema."
            },
            {
                "role": "user",
                "content": user_text
            }
        ],
        "format": schema,
        "stream": False,
        "options": {"temperature": 0.0}
    }

    req = urllib.request.Request(
        "http://localhost:11434/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )

    t0 = time.time()
    res = urllib.request.urlopen(req, timeout=30)
    dur = time.time() - t0
    data = json.loads(res.read().decode("utf-8"))
    content = data.get("message", {}).get("content", "")
    print(f"Duration: {dur:.2f}s")
    print("Parsed JSON:", json.loads(content))

if __name__ == "__main__":
    test()
