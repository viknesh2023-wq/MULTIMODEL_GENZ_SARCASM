import urllib.request
import json
import time

def test():
    schema = {
        "type": "object",
        "properties": {
            "brief_rationale": {
                "type": "string",
                "description": "Analyze if the comment expresses covert interpersonal hostility or if it is just a news link / direct comment / ambiguous."
            },
            "evidence_span": {
                "type": "string",
                "description": "Exact quoted words demonstrating the intent, or 'none' if neutral."
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
            "confidence": {"type": "integer", "minimum": 1, "maximum": 5},
            "context_sufficiency": {"type": "string", "enum": ["SUFFICIENT", "PARTIAL", "INSUFFICIENT"]}
        },
        "required": ["brief_rationale", "evidence_span", "primary_label", "subtype", "confidence", "context_sufficiency"]
    }

    user_text = (
        "Analyze the following social media comment for passive aggression:\n\n"
        "Comment: \"Speaking of children who need protection: http://news.nationalpost.com/news/...\"\n"
        "Context: None\n\n"
        "Guidelines:\n"
        "- If it is just sharing a link, news, or general talk, it is 'non_passive_aggressive' with subtype 'none'.\n"
        "- If context is insufficient to determine intent, it is 'ambiguous'.\n"
        "- Only label 'passive_aggressive' if there is clear covert interpersonal hostility."
    )

    payload = {
        "model": "llama3.2:latest",
        "messages": [
            {
                "role": "system",
                "content": "You are a linguistic annotator. First analyze the pragmatic evidence, then select the appropriate primary label and subtype."
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
