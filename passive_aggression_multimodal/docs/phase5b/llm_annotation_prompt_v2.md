# Phase 5B Versioned LLM Annotation Prompt Specification (v2.0.0)

**Project Title:** Context-Aware Multimodal Passive-Aggression Detection in Social Media Using Image–Comment Fusion  
**Artifact:** `docs/phase5b/llm_annotation_prompt_v2.md`  
**Prompt Version:** `v2.0.0`  
**Date:** October 21, 2026  
**Status:** Calibrated Contrastive Prompt for Quality Audit  

---

## 1. Motivation for Revision from v1.0.0

An audit of the Phase 5A annotations revealed that `llama3.2:latest` assigned `passive_aggressive` to **96%** of candidate records, while `qwen2.5:1.5b` assigned it to **68%**. Detailed inspection demonstrated severe pragmatic over-attribution:
1. **Factual news and sports reports** (e.g., tennis match results) were classified as `implied_blame`.
2. **Sincere admiration** (e.g., "also i'm in love with this pic") was classified as `backhanded_compliment` via hallucinated obligation.
3. **Situational venting and pop-culture quotes** (e.g., Rick and Morty quotes) were conflated with interpersonal attacks.
4. **Direct, blunt debate remarks and rhetorical pushback** were miscategorized as covert hostility.

Prompt v2.0.0 introduces explicit negative boundaries, calibrates task expectations, and forces an overt-vs-covert boundary analysis during Chain-of-Thought reasoning.

---

## 2. System Prompt Specification (v2.0.0)

```text
You are an expert computational linguist and pragmatic dialogue auditor. Your task is to objectively evaluate social media comments and distinguish genuine PASSIVE AGGRESSION from other forms of communication (direct aggression, situational sarcasm, news reports, sincere praise, or ordinary debate).

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
In your "brief_rationale", you must explicitly state:
1. The literal surface meaning.
2. Whether an interpersonal target is being attacked.
3. Whether the attack is covert (disguised by surface politeness) or overt/absent.
```

---

## 3. User Prompt Template (v2.0.0)

```text
Evaluate the following social media post for passive aggression according to the calibrated criteria:

Record ID: {record_id}
Source Dataset: {source_dataset}
Comment Text: "{comment_text}"
Dialogue Context: {context_text}
Image Availability: {image_available} (Note: Image pixels unassessed; evaluate text and context)

Remember:
- News headlines, factual statements, direct insults, and sincere praise are NOT passive-aggressive.
- Do NOT assume covert hostility unless literal wording is polite/neutral but pragmatic intent is hostile towards an individual.
- If context is missing to understand the target or tone, select 'ambiguous'.

Return valid JSON adhering to the required schema.
```

---

## 4. JSON Schema Contract (v2.0.0)

```json
{
  "type": "object",
  "properties": {
    "brief_rationale": {
      "type": "string",
      "description": "Analyze: (1) surface meaning, (2) presence of interpersonal target, (3) covert vs overt vs absent hostility."
    },
    "evidence_span": {
      "type": "string",
      "description": "Exact quoted words showing covert hostility, or 'none' if non-PA / neutral."
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
```
