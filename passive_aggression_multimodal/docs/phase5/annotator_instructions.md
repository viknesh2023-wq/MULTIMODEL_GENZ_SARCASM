# Phase 5 Human Annotator Instructions & Guidelines

**Project Title:** Context-Aware Multimodal Passive-Aggression Detection in Social Media Using Image–Comment Fusion  
**Artifact:** `docs/phase5/annotator_instructions.md`  
**Target Audience:** Independent Human Annotators (Annotator 1 & Annotator 2)  
**Version:** 5.0.0 (Gold-Standard Real Data Protocol)  

---

## 1. Overview & Your Crucial Role

Welcome to the independent annotation pilot for Multimodal Passive-Aggression Detection.

Your task is to review social media posts (including comments, accompanying images, and dialogue context where available) and determine whether the comment exhibits **passive aggression**.

> **SCIENTIFIC INTEGRITY RULE:**  
> You and your co-annotator must work **completely independently**. Do not discuss specific items, share spreadsheets, or compare decisions until the formal adjudication phase. True inter-annotator agreement can only be measured from independent judgments.

---

## 2. Core Operational Definition

Passive aggression is defined as:
> **Indirect communication that provides evidence of interpersonal hostility, resentment, blame, or criticism, delivered covertly with plausible deniability, rather than expressing direct hostility, pure sarcasm, or ordinary negative emotion.**

### Key Hallmarks:
1. **Covert Hostility:** The speaker intends to criticize, mock, demean, or register resentment toward an individual or group.
2. **Plausible Deniability:** The literal phrasing sounds polite, compliant, cheerful, or neutral on the surface ("Per my previous email...", "Great parking job!"), allowing the speaker to deny aggressive intent if challenged.
3. **Pragmatic Incongruity:** The aggression emerges from the clash between the literal words and the situational context (e.g., complimenting a dish that is visibly burnt, praising someone for arriving hours late).

---

## 3. Strict Boundary Distinctions (What Passive Aggression Is NOT)

To avoid category inflation, apply these boundaries strictly:

| Phenomenon | Why It Is NOT Passive Aggression | Example | Correct Primary Label |
|---|---|---|---|
| **Direct Toxicity / Verbal Abuse** | Hostility is explicit, overt, and impossible to deny. | *"You are an incompetent idiot."* | `non_passive_aggressive` |
| **Pure Sarcasm (No Personal Target)** | Inversion of meaning aimed at situations or external misfortune, not attacking a person. | *"Oh fantastic, the subway broke down again!"* | `non_passive_aggressive` |
| **Ordinary Negative Sentiment** | Genuine sadness, grief, anxiety, or exhaustion without interpersonal hostility. | *"I feel so drained today, nothing went right."* | `non_passive_aggressive` |
| **Friendly Teasing / Banter** | Joking between friends that signals social closeness and affection. | *"Look who finally woke up from hibernation!"* | `non_passive_aggressive` |
| **Constructive / Direct Disagreement** | Professional, respectful, or direct objection without covert blame. | *"I disagree with this conclusion based on Section 3."* | `non_passive_aggressive` |
| **Unverifiable Intent / Missing Context** | The phrase could be sincere or sarcastic, but available context is too sparse to know. | *"Sure, whatever you say."* (without tone/context) | `ambiguous` |

---

## 4. Primary Classification Schema

For every post, select exactly one `primary_label`:

1. **`passive_aggressive`** (Code `0`):
   - Select when there is clear pragmatic evidence of covert interpersonal hostility, implied blame, indirect criticism, or weaponized politeness.
2. **`non_passive_aggressive`** (Code `1`):
   - Select for neutral statements, genuine compliments, direct feedback, friendly banter, venting about external objects, or overt direct aggression.
3. **`ambiguous`** (Code `2`):
   - Select when intent cannot reasonably be verified due to lack of conversational background or ambiguous phrasing. **Do not force a binary choice if context is missing.**

---

## 5. Secondary Subtype Taxonomy (Required if Passive Aggressive)

If you assign `primary_label = passive_aggressive`, assign one dominant `subtype`:

1. **`fake_politeness`**: Exaggerated formality, bureaucratic etiquette, or honorifics masking contempt (*"With all due respect...", "As per my previous email which was clearly ignored..."*).
2. **`indirect_criticism`**: Fault-finding without directly naming the individual or by criticizing a general group (*"Interesting how some people never clean their mess in the kitchen..."*).
3. **`backhanded_compliment`**: A compliment containing a backhanded insult or diminishing caveat (*"You look surprisingly professional today!"*).
4. **`implied_blame`**: Shifting responsibility or guilt onto the reader without direct accusation (*"Must be nice to have so much free time while others work late"*).
5. **`sarcastic_agreement`**: Feigned enthusiasm or agreement to highlight the absurdity or failure of the other person (*"Oh absolutely, you definitely know better than the doctor!"*).
6. **`dismissiveness`**: Stonewalling, minimization, or contemptuous brevity intended to belittle the other person's effort (*"Whatever makes you feel better", "K."*).
7. **`other_or_unclear`**: Passive aggression that does not neatly fit the above categories.

---

## 6. How to Evaluate Multimodal Information

1. **When an image is present (`image_available = True`):**
   - Open the image from the given `image_path`.
   - Inspect the visual scene: Does the image contradict the comment?
   - Assign `image_role`:
     - `ESSENTIAL`: Comment is neutral or literal without the image; the image reveals the contradiction.
     - `HELPFUL`: Image adds clarity, but text already contains indirect hostility.
     - `IRRELEVANT`: Image does not add or alter interpersonal meaning.
     - `ABSENT`: No image exists.
2. **When conversation context is present (`context_text`):**
   - Read the preceding conversational turns.
   - Assign `context_sufficiency`:
     - `SUFFICIENT`: Context clearly reveals the relationship, topic, and intent.
     - `PARTIAL`: Some context is provided, but some ambiguity remains.
     - `INSUFFICIENT`: Isolated comment with no contextual grounding.

---

## 7. Step-by-Step Spreadsheet Guide

1. Open your assigned response file:
   - Annotator 1: `data/annotations/phase5/annotator_1_responses.csv`
   - Annotator 2: `data/annotations/phase5/annotator_2_responses.csv`
2. For each row (100 total rows):
   - Read `comment_text`, check `image_path` (if `image_available == True`), and check `context_text`.
   - Fill in:
     - `primary_label`: `passive_aggressive`, `non_passive_aggressive`, or `ambiguous`.
     - `subtype`: One of the 7 subtypes (required if PA, leave blank otherwise).
     - `image_role`: `ESSENTIAL`, `HELPFUL`, `IRRELEVANT`, or `ABSENT`.
     - `context_sufficiency`: `SUFFICIENT`, `PARTIAL`, or `INSUFFICIENT`.
     - `confidence`: Integer `1` (guessing), `2` (low), `3` (moderate), `4` (high), `5` (certain).
     - `evidence_span`: The key words or phrase that revealed the intent (e.g., *"Must be nice"*).
     - `annotator_id`: `annotator_01` or `annotator_02`.
     - `annotation_timestamp`: Current date/time (e.g. `2026-10-21T14:30:00Z`).
     - `adjudication_status`: Change from `PENDING_REVIEW` to `COMPLETED`.
     - `notes`: Optional observations.
3. Save your file in UTF-8 CSV format.
4. Notify the Lead ML Auditor once your 100 rows are completed.
