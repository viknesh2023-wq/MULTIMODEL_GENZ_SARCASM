"""
Local Explanation Reasoner Service (Milestone 6)
Project: Gen-Z Multimodal Sarcasm Detection System
Hardware: CPU Offload / NVIDIA GeForce RTX 4050 GPU VRAM Safety

Provides evidence-grounded, human-readable explanations of multimodal sarcasm predictions:
- Grounds explanations strictly in model outputs, text cues, and verified visual descriptions.
- Never modifies the classifier's predicted label, probabilities, or confidence scores.
- Operates primarily via local instruction LLM (Ollama llama3.2:1b / qwen2.5:1.5b) on system CPU.
- Implements a deterministic, zero-dependency Cognitive Reasoner fallback.
- Guarantees complete VRAM safety by isolating LLM execution from GPU VRAM.
"""

import json
import time
import logging
import urllib.request
import urllib.error
from typing import Dict, Any, List, Optional
from PIL import Image
import torch

logger = logging.getLogger(__name__)

OLLAMA_API_BASE = "http://127.0.0.1:11434"
DEFAULT_OLLAMA_MODEL = "llama3.2:1b"
FALLBACK_OLLAMA_MODEL = "qwen2.5:1.5b"
OLLAMA_TIMEOUT_SECONDS = 20.0


class ExplanationService:
    """
    Manages local explanation synthesis with zero GPU VRAM impact.
    """
    def __init__(self, ollama_api_base: str = OLLAMA_API_BASE):
        self.ollama_api_base = ollama_api_base
        self.ollama_available = False
        self.active_ollama_model: Optional[str] = None
        self._force_offline = False
        self._check_ollama_status()

    def _check_ollama_status(self) -> None:
        """Probe local Ollama service and discover installed models."""
        if self._force_offline:
            self.ollama_available = False
            self.active_ollama_model = None
            return
        try:
            req = urllib.request.Request(f"{self.ollama_api_base}/api/tags", method="GET")
            with urllib.request.urlopen(req, timeout=2.0) as resp:
                data = json.loads(resp.read().decode())
                model_names = [m.get("name", "").split(":")[0] for m in data.get("models", [])]
                full_names = [m.get("name", "") for m in data.get("models", [])]
                
                # Check for llama3.2:1b, qwen2.5:1.5b, or llama3.2
                if any("llama3.2:1b" in fn for fn in full_names):
                    self.active_ollama_model = "llama3.2:1b"
                    self.ollama_available = True
                elif any("qwen2.5:1.5b" in fn for fn in full_names):
                    self.active_ollama_model = "qwen2.5:1.5b"
                    self.ollama_available = True
                elif any("llama3.2" in fn for fn in full_names):
                    self.active_ollama_model = "llama3.2"
                    self.ollama_available = True
                else:
                    self.ollama_available = False
        except Exception as exc:
            logger.debug(f"Ollama local service check: {exc}")
            self.ollama_available = False
            self.active_ollama_model = None

    def get_status(self) -> Dict[str, Any]:
        self._check_ollama_status()
        return {
            "ollama_available": self.ollama_available,
            "active_model": self.active_ollama_model,
            "fallback_reasoner": "deterministic_cognitive_reasoner",
            "runtime": "local_cpu",
            "vram_overhead": "0 MB (CPU execution)",
        }

    def generate_visual_description(self, image: Image.Image) -> Optional[str]:
        """
        Derives an authentic visual observation using the resident InternVL3-2B model.
        Returns a concise 1-sentence factual description of the image content.
        """
        try:
            from app.services.multimodal_service import get_multimodal_manager
            mm_manager = get_multimodal_manager()
            if not mm_manager.is_ready() or mm_manager.vlm_model is None:
                return None

            from transformers import AutoTokenizer
            from torchvision import transforms

            # Use lightweight transforms for 1-tile factual visual query
            IMAGENET_MEAN = (0.485, 0.456, 0.406)
            IMAGENET_STD = (0.229, 0.224, 0.225)
            tr = transforms.Compose([
                transforms.Lambda(lambda img: img.convert("RGB")),
                transforms.Resize((448, 448), interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ])

            tokenizer = AutoTokenizer.from_pretrained("OpenGVLab/InternVL3-2B", trust_remote_code=True, use_fast=False)
            pv = tr(image).unsqueeze(0).to(mm_manager.device, dtype=torch.bfloat16)

            gen_config = dict(max_new_tokens=40, do_sample=False)
            prompt = "<image>\nBriefly describe what is in this image in one factual sentence."

            with mm_manager.inference_lock:
                with torch.inference_mode():
                    desc = mm_manager.vlm_model.chat(tokenizer, pv, prompt, gen_config)
            
            clean_desc = desc.strip().replace("\n", " ")
            return clean_desc if clean_desc else None

        except Exception as exc:
            logger.warning(f"Could not derive visual description from InternVL: {exc}")
            return None

    def generate_explanation(
        self,
        text: str,
        prediction: Dict[str, Any],
        sentiment: Dict[str, Any],
        emotions: Dict[str, Any],
        visual_evidence: Optional[str] = None,
        image_metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Synthesizes a structured explanation from real prediction signals.
        Tries local Ollama on CPU first, falls back to Cognitive Reasoner on failure.
        """
        t0 = time.time()
        self._check_ollama_status()

        # Attempt Ollama LLM Reasoner if service is online
        if self.ollama_available and self.active_ollama_model:
            try:
                explanation = self._query_ollama(
                    text=text,
                    prediction=prediction,
                    sentiment=sentiment,
                    emotions=emotions,
                    visual_evidence=visual_evidence,
                )
                if explanation and self._validate_schema(explanation):
                    elapsed_ms = round((time.time() - t0) * 1000, 2)
                    return {
                        "status": "success",
                        "explanation": explanation,
                        "prediction_reference": {
                            "label": prediction["label"],
                            "class_id": prediction["class_id"],
                            "probabilities": prediction["probabilities"],
                            "confidence": prediction["confidence"],
                        },
                        "reasoner": {
                            "model": f"Ollama {self.active_ollama_model}",
                            "runtime": "local_cpu",
                            "latency_ms": elapsed_ms,
                        }
                    }
            except Exception as exc:
                logger.warning(f"Ollama generation failed or timed out ({exc}). Engaging Cognitive Reasoner fallback.")

        # Fallback Reasoner (Deterministic, Instant, Zero VRAM, Fully Grounded)
        explanation = self._cognitive_fallback_reasoner(
            text=text,
            prediction=prediction,
            sentiment=sentiment,
            emotions=emotions,
            visual_evidence=visual_evidence,
            image_metadata=image_metadata,
        )
        elapsed_ms = round((time.time() - t0) * 1000, 2)

        return {
            "status": "success",
            "explanation": explanation,
            "prediction_reference": {
                "label": prediction["label"],
                "class_id": prediction["class_id"],
                "probabilities": prediction["probabilities"],
                "confidence": prediction["confidence"],
            },
            "reasoner": {
                "model": "Deterministic Cognitive Reasoner (Local Grounded)",
                "runtime": "local_cpu",
                "latency_ms": elapsed_ms,
            }
        }

    def _query_ollama(
        self,
        text: str,
        prediction: Dict[str, Any],
        sentiment: Dict[str, Any],
        emotions: Dict[str, Any],
        visual_evidence: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Constructs an evidence-constrained prompt and requests structured JSON from Ollama."""
        top_emo_str = ", ".join([
            f"{e['emotion']} ({e['probability']*100:.1f}%)"
            for e in emotions.get("top_emotions", [])[:3]
        ])

        visual_str = visual_evidence if visual_evidence else "Image features extracted across CLIP ViT-B/32 and InternVL3-2B representations."

        system_prompt = (
            "You are a scientific Explainable AI (XAI) reasoner for a Multimodal Sarcasm Detection System.\n"
            "Synthesize an evidence-grounded explanation for the multimodal classification.\n"
            "DO NOT change the model prediction. DO NOT invent facts. Use cautious, objective language.\n\n"
            f"[USER TEXT]: \"{text}\"\n"
            f"[MODEL PREDICTION]: {prediction['label']} (Confidence: {prediction['confidence']*100:.1f}%)\n"
            f"[SENTIMENT SIGNALS]: Negative: {sentiment['negative']*100:.1f}%, Neutral: {sentiment['neutral']*100:.1f}%, Positive: {sentiment['positive']*100:.1f}%\n"
            f"[TOP EMOTION SIGNALS]: {top_emo_str}\n"
            f"[VISUAL OBSERVATIONS]: {visual_str}\n\n"
            "Return ONLY valid JSON matching this schema:\n"
            "{\n"
            "  \"summary\": \"1-2 sentence overview of the linguistic or multimodal cues supporting the prediction.\",\n"
            "  \"text_evidence\": [\"1-2 specific lexical cues, punctuation, or phrase contrasts in the text\"],\n"
            "  \"visual_evidence\": [\"1-2 factual visual observations that contrast or contextualize the text\"],\n"
            "  \"supporting_signals\": [\"1-2 references to the model's sentiment and emotion probability distributions\"],\n"
            "  \"limitations\": [\"1 statement noting that correlation does not guarantee intent or noting ambiguities\"]\n"
            "}"
        )

        payload = {
            "model": self.active_ollama_model,
            "prompt": system_prompt,
            "format": "json",
            "stream": False,
            "options": {
                "num_gpu": 0,  # Strict CPU pinning to protect GPU memory
                "temperature": 0.2,
                "num_predict": 300,
            }
        }

        req = urllib.request.Request(
            f"{OLLAMA_API_BASE}/api/generate",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"}
        )

        with urllib.request.urlopen(req, timeout=OLLAMA_TIMEOUT_SECONDS) as resp:
            data = json.loads(resp.read().decode())
            response_text = data.get("response", "")
            parsed = json.loads(response_text)

            # Strict normalization of schema types
            if isinstance(parsed.get("summary"), list):
                parsed["summary"] = " ".join(str(s) for s in parsed["summary"])
            elif not isinstance(parsed.get("summary"), str):
                parsed["summary"] = str(parsed.get("summary", ""))

            for list_key in ["text_evidence", "visual_evidence", "supporting_signals", "limitations"]:
                val = parsed.get(list_key)
                if isinstance(val, str):
                    parsed[list_key] = [val.strip()]
                elif isinstance(val, list):
                    parsed[list_key] = [str(item).strip() for item in val if str(item).strip()]
                elif val is not None:
                    parsed[list_key] = [str(val)]
                else:
                    parsed[list_key] = []

            return parsed

    def _cognitive_fallback_reasoner(
        self,
        text: str,
        prediction: Dict[str, Any],
        sentiment: Dict[str, Any],
        emotions: Dict[str, Any],
        visual_evidence: Optional[str] = None,
        image_metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Deterministic, rule-grounded explanation generator.
        Provides robust explanations grounded in lexical patterns and affective vectors.
        """
        is_sarcastic = prediction["label"].lower() == "sarcastic"
        conf = prediction["confidence"]
        neg_sent = sentiment.get("negative", 0.0)
        pos_sent = sentiment.get("positive", 0.0)
        neu_sent = sentiment.get("neutral", 0.0)
        
        top_emos = emotions.get("top_emotions", [])
        top_emo_names = [e["emotion"] for e in top_emos[:3]]
        
        # Analyze lexical cues
        lower_text = text.lower()
        pos_markers = [w for w in ["great", "wonderful", "love", "amazing", "brilliant", "perfect", "awesome", "fantastic", "thanks", "thank you"] if w in lower_text]
        neg_markers = [w for w in ["delay", "broken", "worst", "fail", "terrible", "hate", "traffic", "crash", "waiting", "tire", "monday", "stuck"] if w in lower_text]
        punct_markers = []
        if "!" in text:
            punct_markers.append("exclamation emphasis ('!')")
        if "..." in text or "…" in text:
            punct_markers.append("suspension / trailing ellipsis ('...')")
        if "?" in text:
            punct_markers.append("rhetorical question marker ('?')")

        # Synthesize Text Evidence
        text_evidence = []
        if pos_markers and (neg_markers or neg_sent > 0.4 or "annoyance" in top_emo_names or "disapproval" in top_emo_names):
            text_evidence.append(
                f"Lexical incongruity: Surface-positive wording ({', '.join(f"'{p}'" for p in pos_markers[:2])}) is juxtaposed with negative contextual cues ({', '.join(f"'{n}'" for n in neg_markers[:2]) if neg_markers else 'underlying negative affect'})."
            )
        elif pos_markers and is_sarcastic:
            text_evidence.append(
                f"Intensified surface praise ({', '.join(f"'{p}'" for p in pos_markers[:2])}) commonly functions as ironic contrast in social commentary."
            )
        else:
            text_evidence.append(
                f"Utterance presents { 'direct literal framing' if not is_sarcastic else 'subtle affective framing'} without overt lexical contradiction."
            )

        if punct_markers:
            text_evidence.append(f"Typographical markers: Detected {', '.join(punct_markers)}.")

        # Synthesize Visual Evidence
        visual_evidence_list = []
        if visual_evidence:
            visual_evidence_list.append(f"Observed visual scene: {visual_evidence}")
        elif image_metadata:
            dims = image_metadata.get("dimensions", "Standard")
            visual_evidence_list.append(f"Visual media encoded at {dims} resolution across CLIP ViT-B/32 global embeddings and InternVL3-2B localized tiles.")
        else:
            visual_evidence_list.append("Visual modality provides scene context extracted via dual CLIP and InternVL3-2B visual towers.")

        # Synthesize Supporting Signals
        dominant_sent = "Positive" if pos_sent > max(neg_sent, neu_sent) else ("Negative" if neg_sent > neu_sent else "Neutral")
        supporting_signals = [
            f"Sentiment polarity distribution: {dominant_sent} dominant (Negative: {neg_sent*100:.1f}%, Neutral: {neu_sent*100:.1f}%, Positive: {pos_sent*100:.1f}%).",
            f"Primary detected GoEmotions: {', '.join([f'{e['emotion'].capitalize()} ({e['probability']*100:.1f}%)' for e in top_emos[:3]]) if top_emos else 'Standard distribution'}."
        ]

        # Summary
        if is_sarcastic:
            summary = (
                f"The system classifies this instance as sarcastic with {conf*100:.1f}% confidence. "
                f"The prediction is supported by affective dissonance between surface lexical cues and underlying {top_emo_names[0] if top_emo_names else 'negative'} emotional polarity."
            )
        else:
            summary = (
                f"The system classifies this instance as non-sarcastic with {conf*100:.1f}% confidence. "
                f"The linguistic and visual evidence exhibits congruent literal valence without significant ironic discordance."
            )

        # Limitations
        limitations = [
            "Statistical feature correlation does not establish subjective communicative intent.",
            "Visual-textual interpretation may vary across cultural idioms, slang evolution, and offline context."
        ]

        return {
            "summary": summary,
            "text_evidence": text_evidence,
            "visual_evidence": visual_evidence_list,
            "supporting_signals": supporting_signals,
            "limitations": limitations,
        }

    def _validate_schema(self, explanation: Any) -> bool:
        """Verifies that the generated explanation contains required keys and non-empty values."""
        if not isinstance(explanation, dict):
            return False
        required_keys = ["summary", "text_evidence", "visual_evidence", "supporting_signals", "limitations"]
        for k in required_keys:
            if k not in explanation or not explanation[k]:
                return False
        return True


# Singleton instance
_explanation_service = ExplanationService()

def get_explanation_service() -> ExplanationService:
    return _explanation_service
