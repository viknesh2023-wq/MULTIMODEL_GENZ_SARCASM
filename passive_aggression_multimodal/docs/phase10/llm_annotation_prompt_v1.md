# Phase 10 Neutral LLM Annotation Protocol (Version v1.0.0)

## 1. Operational Definition of Passive Aggression

Passive aggression is defined as:
> **Indirect communication that provides evidence of interpersonal hostility, resentment, blame, or criticism, disguised beneath surface politeness, compliance, praise, or neutrality, while retaining some degree of plausible deniability.**

A statement is classified as `passive_aggressive` **if and only if** all three prongs of the test are satisfied:

1. **Prong 1: Interpersonal Target Present**
   - The comment is directed at an identifiable interlocutor, specific individual, or participant in the interaction.
   - It is *not* directed at an abstract concept, weather, traffic, delay, or the world in general.
2. **Prong 2: Plausible Deniability Present**
   - The surface wording is softened, polite, praising, or seemingly neutral, allowing the speaker to deny malicious or hostile intent if challenged.
   - Overtly hostile, abusive, or explicitly threatening language lacks plausible deniability.
3. **Prong 3: Covert Hostility Present**
   - The pragmatic implication carries blame, criticism, resentment, mockery, or condescension rather than genuine warmth or objective disagreement.

---

## 2. Distinct Negative Contrast Boundaries (`non_passive_aggressive`)

The annotator must actively distinguish passive aggression from related communicative phenomena:

- **Direct Aggression / Overt Toxicity:** Explicit insults ("You are stupid", "Shut up") lack surface courtesy or plausible deniability $\to$ `non_passive_aggressive`.
- **Situational Sarcasm / Venting:** Irony directed at external circumstances ("Great, another train delay!", "Rain on my wedding day") lacks an interpersonal target $\to$ `non_passive_aggressive`.
- **Factual Reporting & News:** Objective sports results, news quotes, dates, statistics, and descriptive summaries $\to$ `non_passive_aggressive`.
- **Sincere Praise & Gratitude:** Genuine compliments, thank-yous, and words of encouragement without barbed undertones $\to$ `non_passive_aggressive`.
- **Constructive / Direct Disagreement:** Blunt disputes, technical counterarguments, and factual debate without faux politeness $\to$ `non_passive_aggressive`.
- **Friendly Teasing / Playful Banter:** In-group jokes and warm teasing between friends without hostile intent $\to$ `non_passive_aggressive`.
- **Institutional / Systemic Critique:** Frustration aimed at corporations, governments, or abstract systems rather than an individual target $\to$ `non_passive_aggressive`.

---

## 3. Ambiguity & Missing Context Rules (`ambiguous`)

A record must be labeled `ambiguous` when:
1. The text is an isolated fragment, hashtag list, or meme reference whose conversational target cannot be established.
2. The comment hinges fundamentally on an unassessed image (e.g. text says "She looks thrilling" but the image is not assessed).
3. The pragmatic intent is genuinely balanced between sincere praise and indirect criticism, with insufficient evidence to resolve the boundary.

---

## 4. System Prompt for Ollama Annotators

```text
You are an expert computational linguist. Evaluate social media comments for passive aggression using strict operational criteria.

OPERATIONAL DEFINITION:
Passive aggression is indirect communication conveying covert interpersonal hostility disguised beneath surface politeness, compliance, praise, or neutrality, with plausible deniability.

THREE-PRONG TEST:
1. Interpersonal Target: Hostility/criticism directed at a specific person or participant.
2. Plausible Deniability: Surface wording appears polite, praising, or neutral, permitting denial of malice.
3. Covert Hostility: Pragmatic meaning conveys resentment, blame, contempt, or demeaning criticism.

A comment is 'passive_aggressive' ONLY IF ALL THREE CRITERIA ARE MET.

NEGATIVE CONTRASTS (Select 'non_passive_aggressive'):
- Direct aggression / insults ("You're an idiot", "Shut up") -> Overt hostility lacking polite disguise is NOT PA.
- Situational sarcasm / venting ("Bus is late again!", "Weather is terrible") -> No individual target is NOT PA.
- Factual reporting / descriptions (news summaries, scores, links) -> Objective reporting is NOT PA.
- Sincere praise & gratitude ("Thanks for the great advice!") -> Genuine appreciation is NOT PA.
- Direct disagreement ("I disagree with that point", "The data shows otherwise") -> Direct debate is NOT PA.
- Friendly teasing / banter ("Look who finally showed up!") -> Friendly social bonding is NOT PA.

AMBIGUOUS (Select 'ambiguous'):
- Missing context or meme fragment where intent cannot be verified.

Respond strictly in valid JSON format.
```

---

## 5. Structured JSON Output Schema for Ollama

```json
{
  "type": "object",
  "properties": {
    "interpersonal_target": { "type": "string", "enum": ["PRESENT", "ABSENT", "UNCLEAR"] },
    "plausible_deniability": { "type": "string", "enum": ["YES", "NO", "UNCLEAR"] },
    "covert_hostility": { "type": "string", "enum": ["COVERT", "OVERT", "ABSENT", "UNCLEAR"] },
    "label": { "type": "string", "enum": ["passive_aggressive", "non_passive_aggressive", "ambiguous"] },
    "subtype": {
      "type": "string",
      "enum": [
        "fake_politeness", "indirect_criticism", "backhanded_compliment",
        "implied_blame", "sarcastic_agreement", "dismissiveness", "other_or_unclear", "not_applicable"
      ]
    },
    "confidence_band": { "type": "string", "enum": ["high", "medium", "low"] },
    "evidence_span": { "type": "string" },
    "rationale": { "type": "string" },
    "context_sufficient": { "type": "boolean" }
  },
  "required": [
    "interpersonal_target", "plausible_deniability", "covert_hostility",
    "label", "subtype", "confidence_band", "evidence_span", "rationale", "context_sufficient"
  ]
}
```
