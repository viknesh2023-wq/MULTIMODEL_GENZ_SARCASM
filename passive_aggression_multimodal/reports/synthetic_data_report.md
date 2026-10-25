# Phase 1 Synthetic Data Pilot Report

**Project Title:** Context-Aware Multimodal Passive-Aggression Detection in Social Media Using Image–Comment Fusion  
**Date of Audit:** October 21, 2026  
**Status:** Generated, Schema-Validated, Marked `SYNTHETIC_UNVERIFIED`  
**Dataset Artifact:** `data/synthetic/synthetic_pilot_350.jsonl`  

---

## 1. Executive Summary

A controlled synthetic pilot dataset comprising **350 structured scenarios** was designed and generated during Phase 1 (`scripts/generate_synthetic_pilot.py`). The objective is to provide a balanced, high-resolution testbed for developing the multimodal annotation pipeline and evaluating baseline contrastive behaviors without contaminating the real-world evaluation benchmark.

```
Total Synthetic Scenarios:       350
Primary Label Breakdown:
  - passive_aggressive:          180 (51.4%)
  - non_passive_aggressive:      145 (41.4%)
  - ambiguous:                    25 (7.2%)
Verification Status:             SYNTHETIC_UNVERIFIED (100%)
Human Review Status:             AWAITING_HUMAN_REVIEW (100%)
```

---

## 2. Archetype and Subtype Distribution

| Archetype / Category | Target Class | Subtype Label | Record Count | Multimodal Role |
| :--- | :--- | :--- | :--- | :--- |
| **Fake Politeness** | `passive_aggressive` | `fake_politeness` | 30 | Masked hostility behind courteous phrasing |
| **Indirect Criticism** | `passive_aggressive` | `indirect_criticism` | 30 | Snide rhetorical critique avoiding direct confrontation |
| **Backhanded Compliment** | `passive_aggressive` | `backhanded_compliment` | 30 | Praise conditioned on an insulting baseline |
| **Implied Blame** | `passive_aggressive` | `implied_blame` | 30 | Shifting responsibility subtly without overt accusation |
| **Sarcastic Agreement** | `passive_aggressive` | `sarcastic_agreement` | 30 | Feigned agreement mocking the poster's assertion |
| **Dismissiveness** | `passive_aggressive` | `dismissiveness` | 30 | Invalidation of poster pride ("must be nice") |
| **Direct Criticism** | `non_passive_aggressive` | `other_or_unclear` | 29 | Overt, non-indirect confrontation (Contrast) |
| **Ordinary Negativity** | `non_passive_aggressive` | `other_or_unclear` | 29 | Frustration with weather, traffic, events (Contrast) |
| **Genuine Sadness** | `non_passive_aggressive` | `other_or_unclear` | 29 | Sincere grief, sorrow, mourning (Contrast) |
| **Friendly Teasing** | `non_passive_aggressive` | `other_or_unclear` | 29 | Playful affectionate banter among friends (Contrast) |
| **Constructive Disagreement** | `non_passive_aggressive` | `other_or_unclear` | 29 | Technical / objective debate without rancor (Contrast) |
| **Ambiguous Scenarios** | `ambiguous` | `other_or_unclear` | 25 | Underspecified tone, terse replies ("Noted.") |

---

## 3. Multimodal Dependency Analysis

Each synthetic scenario explicitly encodes its multimodal dependency relationship:
1. **`image_necessary` (190 records):** The text comment appears polite, neutral, or enthusiastic in isolation, but reverses into passive aggression or clear irony only when evaluated against the visual incongruity.
2. **`image_irrelevant` (135 records):** The text comment contains unambiguous indirect hostility or weaponized phrasing regardless of the image content.
3. **`context_dependent` (25 records):** The comment relies on preceding dialogue turns or thread history to resolve ambiguity.

---

## 4. Case Studies from Synthetic Pilot

### Case Study A: Multimodal Reversal (Fake Politeness + Image Incongruity)
- **Example ID:** `synthetic_pilot_0001`
- **Image Description:** *A meticulously baked artisan sourdough loaf with uneven scoring and a collapsed side.*
- **Post Caption:** *"Baked my first artisan sourdough after 36 hours of fermentation!"*
- **Target Comment:** *"Per my previous email, following basic instructions usually yields an even crumb, but thank you so much for sharing your unique effort!"*
- **Primary Label:** `passive_aggressive` | **Subtype:** `fake_politeness`
- **Linguistic Rationale:** The comment weaponizes corporate boilerplate ("Per my previous email", "thank you so much") to deliver a condescending critique of baking failure visible in the image.

### Case Study B: Contrast Class (Direct Criticism vs. Passive Aggression)
- **Example ID:** `synthetic_pilot_0031`
- **Image Description:** *A dangerous poorly wired electrical plug with exposed copper wires.*
- **Post Caption:** *"DIY electrical repair done!"*
- **Target Comment:** *"Stop immediately. That is an extreme fire hazard and illegal wiring."*
- **Primary Label:** `non_passive_aggressive` | **Subtype:** `other_or_unclear`
- **Linguistic Rationale:** Overt, transparent, safety-critical directive with zero indirectness or masked hostility. Direct criticism is not passive aggression.

### Case Study C: Genuine Ambiguity (Under-specified Intent)
- **Example ID:** `synthetic_pilot_0056`
- **Image Description:** *A photo of a person wearing a neon lime winter coat.*
- **Post Caption:** *"Winter coat arrived!"*
- **Target Comment:** *"Very bold color choice."*
- **Primary Label:** `ambiguous` | **Subtype:** `other_or_unclear`
- **Linguistic Rationale:** "Very bold color choice" can denote sincere stylistic appreciation, neutral observation, or covert judgment. Without prosodic cues, vocal inflection, or known social relationship, the communication must remain labeled as ambiguous.

---

## 5. Governance and Verification Protocol

All 350 records are stored in `data/synthetic/synthetic_pilot_350.jsonl` with:
- `synthetic_status: "SYNTHETIC_UNVERIFIED"`
- `annotation_status: "SYNTHETIC_PILOT"`
- `human_review_status: "AWAITING_HUMAN_REVIEW"`
- Under the data governance policy, synthetic records are sequestered from real-world benchmark evaluation splits and will only be promoted to `SYNTHETIC_VERIFIED` after double-blind human auditor review in Phase 2.
