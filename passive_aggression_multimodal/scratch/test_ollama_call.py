import urllib.request
import json
import time

def test_chat():
    payload = {
        "model": "qwen2.5:1.5b",
        "messages": [
            {
                "role": "system",
                "content": "You are a linguistic annotator. Output valid JSON only with keys: primary_label, subtype, confidence, brief_rationale."
            },
            {
                "role": "user",
                "content": "Analyze comment: 'Per my previous email, thank you so much for finally showing up.' Context: None."
            }
        ],
        "format": "json",
        "stream": False,
        "options": {"temperature": 0.0}
    }

    req = urllib.request.Request(
        "http://localhost:11434/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )

    t0 = time.time()
    try:
        res = urllib.request.urlopen(req, timeout=90)
        dur = time.time() - t0
        data = json.loads(res.read().decode("utf-8"))
        content = data.get("message", {}).get("content", "")
        print(f"Status: OK, Duration: {dur:.2f}s")
        print("Raw Content:", content)
        parsed = json.loads(content)
        print("Parsed JSON:", parsed)
    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    test_chat()
