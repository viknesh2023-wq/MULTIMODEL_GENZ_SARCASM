import urllib.request
import json
import time

def test_constrained():
    schema = {
        "type": "object",
        "properties": {
            "primary_label": {
                "type": "string",
                "enum": ["passive_aggressive", "non_passive_aggressive", "ambiguous"]
            },
            "subtype": {
                "type": "string",
                "enum": [
                    "fake_politeness", "indirect_criticism", "backhanded_compliment",
                    "implied_blame", "sarcastic_agreement", "dismissiveness",
                    "other_or_unclear", "none"
                ]
            },
            "confidence": {
                "type": "integer",
                "minimum": 1,
                "maximum": 5
            },
            "evidence_span": {
                "type": "string"
            },
            "brief_rationale": {
                "type": "string"
            }
        },
        "required": ["primary_label", "subtype", "confidence", "evidence_span", "brief_rationale"]
    }

    payload = {
        "model": "qwen2.5:1.5b",
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a linguistic annotator analyzing social media comments for passive aggression. "
                    "Passive aggression is indirect communication expressing covert interpersonal hostility, blame, resentment, or criticism behind a veneer of politeness or neutrality. "
                    "Classify into: passive_aggressive, non_passive_aggressive, or ambiguous. "
                    "If passive_aggressive, select the dominant subtype. If non_passive_aggressive, subtype must be 'none'."
                )
            },
            {
                "role": "user",
                "content": "Analyze comment: 'Per my previous email which was clearly ignored, thank you so much for finally joining the meeting.' Context: None."
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
    print(f"Status: OK, Duration: {dur:.2f}s")
    print("Raw Content:", content)
    parsed = json.loads(content)
    print("Parsed JSON:", parsed)

def test_llama():
    schema = {
        "type": "object",
        "properties": {
            "primary_label": {
                "type": "string",
                "enum": ["passive_aggressive", "non_passive_aggressive", "ambiguous"]
            },
            "subtype": {
                "type": "string",
                "enum": [
                    "fake_politeness", "indirect_criticism", "backhanded_compliment",
                    "implied_blame", "sarcastic_agreement", "dismissiveness",
                    "other_or_unclear", "none"
                ]
            },
            "confidence": {
                "type": "integer",
                "minimum": 1,
                "maximum": 5
            },
            "evidence_span": {
                "type": "string"
            },
            "brief_rationale": {
                "type": "string"
            }
        },
        "required": ["primary_label", "subtype", "confidence", "evidence_span", "brief_rationale"]
    }

    payload = {
        "model": "llama3.2:latest",
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a linguistic annotator analyzing social media comments for passive aggression. "
                    "Passive aggression is indirect communication expressing covert interpersonal hostility, blame, resentment, or criticism behind polite or neutral words. "
                    "Classify into: passive_aggressive, non_passive_aggressive, or ambiguous. "
                    "If passive_aggressive, select the dominant subtype from: fake_politeness, indirect_criticism, backhanded_compliment, implied_blame, sarcastic_agreement, dismissiveness, other_or_unclear. If non_passive_aggressive, subtype must be 'none'."
                )
            },
            {
                "role": "user",
                "content": "Analyze comment: 'Warmest regards to whoever felt inspired to leave all their pots for the cleaning fairy to bless 💕' Context: Shared kitchen."
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
    res = urllib.request.urlopen(req, timeout=90)
    dur = time.time() - t0
    data = json.loads(res.read().decode("utf-8"))
    content = data.get("message", {}).get("content", "")
    print(f"Llama 3.2 Status: OK, Duration: {dur:.2f}s")
    parsed = json.loads(content)
    print("Parsed JSON:", parsed)

if __name__ == "__main__":
    test_llama()
