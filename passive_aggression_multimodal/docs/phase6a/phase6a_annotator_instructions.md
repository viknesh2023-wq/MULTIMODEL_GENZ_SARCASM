# Phase 6A: Human Annotator Instructions & Guidelines

**Project Title:** Context-Aware Multimodal Passive-Aggression Detection in Social Media Using Image–Comment Fusion  
**Artifact:** `docs/phase6a/phase6a_annotator_instructions.md`  
**Target Audience:** Independent Human Annotators (Annotator 1 & Annotator 2)  
**Version:** 6.0.0 (Double-Blind Gold-Standard Protocol)  

---

## 1. Welcome and Your Essential Role

Thank you for contributing to this research project. Your task is to independently review **30 real-world social media records** and evaluate whether each post exhibits **passive aggression**.

Because automated AI models (such as LLMs) suffer from severe false-positive biases, your human judgments represent the sole source of ground-truth evidence for this scientific study.

> **CRITICAL SCIENTIFIC DIRECTIVE:**  
> **Work 100% Independently.**  
> - Annotator 1 must only edit: `data/annotations/phase6a/annotator_1_responses.csv`.  
> - Annotator 2 must only edit: `data/annotations/phase6a/annotator_2_responses.csv`.  
> - Do not discuss specific items, share your spreadsheet, or compare answers until both of you have submitted your completed reviews.

---

## 2. Core Operational Definition: The Three-Prong Test

For a comment to be classified as **`passive_aggressive`**, it must satisfy all three prongs:

```
                ┌──────────────────────────────────────────────┐
                │        THE THREE-PRONG PRAGMATIC TEST        │
                └──────────────────────────────────────────────┘
                                        │
           ┌────────────────────────────┼────────────────────────────┐
           ▼                            ▼                            ▼
  [1. Interpersonal Target]    [2. Covert Hostility]      [3. Plausible Deniability]
  Directed at a specific       Intent to mock, belittle,  Literal phrasing appears
  conversation partner,        criticize, or assign blame polite, compliant, cheerful,
  interlocutor, or group.      indirectly.                or neutral ("Just saying!").
```

### Contrast Classes (What Passive Aggression is NOT):
To keep our benchmark scientifically rigorous, **do not** classify these contrast phenomena as passive-aggressive:
1. **Direct Aggression / Verbal Abuse:** Explicit insults, profanity, or blatant condemnation (e.g. *"You are a complete idiot"*). Hostility is obvious and cannot be denied $\rightarrow$ Label as `non_passive_aggressive`.
2. **General Sarcasm / Pure Irony:** Sarcastic venting about external situations, the weather, traffic, or life (e.g. *"Oh great, another flat tire"*). No individual is being attacked $\rightarrow$ Label as `non_passive_aggressive`.
3. **Institutional Criticism:** Complaining about or mocking broad abstract entities like government agencies, news networks, or the NFL without attacking an interlocutor $\rightarrow$ Label as `non_passive_aggressive`.
4. **Sincere Praise / Compliments:** Genuine expressions of gratitude or admiration (e.g. *"I want to thank you for this fine article"*). Do not look for hidden conspiracy where there is none $\rightarrow$ Label as `non_passive_aggressive`.
5. **Friendly Teasing / Affectionate Banter:** Playful jokes between friends signaling closeness (e.g. *"Look who finally woke up!"*) $\rightarrow$ Label as `non_passive_aggressive`.
6. **Ambiguous Context / Dangling Fragments:** When a comment is an incomplete sentence or where the intent is completely impossible to decipher from the available context $\rightarrow$ Label as `ambiguous`.

---

## 3. Step-by-Step Annotation Procedure

### Step 1: Open Your Designated Template
- **Annotator 1:** Open [data/annotations/phase6a/annotator_1_responses.csv](file:///d:/multimodel_passive_aggression/data/annotations/phase6a/annotator_1_responses.csv) in VS Code, Excel, or your preferred CSV editor.
- **Annotator 2:** Open [data/annotations/phase6a/annotator_2_responses.csv](file:///d:/multimodel_passive_aggression/data/annotations/phase6a/annotator_2_responses.csv).

### Step 2: Read the Comment and Context
- Read the `comment_text` carefully.
- Read `context_text` to understand the conversational turns preceding the comment (if available).

### Step 3: Inspect the Image (If `image_available == True`)
- Look at the `image_path` column (e.g. `data/images/mmsd/820050180577730560.jpg`).
- Open and inspect the image file in the project directory.
- Determine whether the image provides crucial context (e.g., showing a failure, displaying a person's photo, or establishing visual contrast).
- Record `image_reviewed`: `YES` (or `NOT_APPLICABLE` if no image exists).
- Record `image_role`:
  - `ESSENTIAL`: The passive aggression cannot be detected without seeing the image.
  - `HELPFUL`: The image clarifies or strengthens the signal, but text has hints.
  - `IRRELEVANT`: The image is decorative or unrelated to the comment.
  - `ABSENT`: No image exists for this record.

### Step 4: Record Your Core Label
- Fill in `primary_label`:
  - `passive_aggressive`
  - `non_passive_aggressive`
  - `ambiguous`

### Step 5: If Passive-Aggressive, Select a Subtype
- Fill in `subtype`:
  - `fake_politeness`: Weaponized formal courtesy masking hostility (e.g. *"Per my previous email, as you clearly failed to read..."*).
  - `indirect_criticism`: Implicit fault-finding or disparagement.
  - `backhanded_compliment`: Diminishing praise or subtle put-down (e.g. *"Twenty-one hours, you're a hero"*).
  - `implied_blame`: Indirectly shifting fault onto the interlocutor.
  - `sarcastic_agreement`: Feigned compliance to mock an idea.
  - `dismissiveness`: Stonewalling or contemptuous brevity.
  - `other_or_unclear`: Other indirect aggressive forms.

### Step 6: Provide Brief Evidence Notes
- `target_evidence`: Who is the comment attacking? (e.g. "Addressee in conversation", "Poster", "None").
- `covert_hostility_evidence`: How is the hostility expressed?
- `plausible_deniability_evidence`: Why could the speaker claim innocence?
- `context_sufficiency`: Is context `SUFFICIENT`, `PARTIAL`, or `INSUFFICIENT`?
- `confidence`: `HIGH`, `MEDIUM`, or `LOW`.
- `evidence_span`: Paste the exact words that demonstrate the covert hostility.

### Step 7: Finalize Row Metadata
- `review_status`: Change from `PENDING_REVIEW` to `COMPLETED`.
- `annotation_timestamp`: Enter the date/time of completion (e.g. `2026-10-21 23:00`).

---

## 4. Submission & Quality Verification

Once you have completed all 30 rows in your file:
1. Save your CSV file using UTF-8 encoding.
2. Run the status verification script:
   ```bash
   python scripts/analyze_phase6a_agreement.py --check-status
   ```
3. Notify the study coordinator that your independent responses have been logged.
