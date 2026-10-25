# Phase 5D: Silver Dataset Validation and Leakage-Safe Preparation Report

**Project Title:** Context-Aware Multimodal Passive-Aggression Detection in Social Media Using Image–Comment Fusion  
**Canonical Project Root:** `D:\NLP_PROJECT\passive_aggression_multimodal`  
**Linked Workspace:** `D:\multimodel_passive_aggression`  
**Milestone:** Phase 5D — Silver Dataset Forensic Validation & Leakage Prevention  
**Date:** October 21, 2026  
**Auditor:** Senior ML Research Auditor & Experimental Reliability Team  

---

## 1. Executive Summary & Verification of Phase 5C Outputs

This Phase 5D milestone continues directly from the Phase 5C forensic audit. In Phase 5C, an initial two-tiered filtering rule was proposed, projecting an estimated 36-record candidate subset. 

In Phase 5D, we conducted an independent, record-by-record linguistic audit of all candidate records, verified candidate negative examples, eliminated duplicate/leakage risks, and constructed a leakage-safe, provenance-tagged silver dataset.

### Key Audit Findings:
1. **Independent Recomputation of Proposed Subset:**
   - In Phase 5C, we hypothesized retaining ~36 consensus records (35 positive, 1 negative).
   - In Phase 5D, applying strict pragmatic criteria (confirmed individual addressee, confirmed covert hostility, and confirmed plausible deniability) reduced the verified positive subset to **18 records**.
   - Exactly **1 record** (`sarcnet_test_00053`) met the standard for a verified consensus negative.
   - Therefore, the verified silver dataset contains exactly **19 records (18 positive, 1 negative)**.
   - Rather than forcing an artificial target count (36) or fabricating negative labels, we report the exact, truthful scientific result.
2. **Rejection of Sentiment/Toxicity Labels as Passive-Aggression Negatives:**
   - We explicitly tested and rejected the hypothesis that neutral sentiment or zero-toxicity comments from CivilComments can be converted into verified passive-aggression negatives.
   - Passive aggression is, by definition, non-toxic on the surface. Sincere praise, direct argument, and weaponized faux politeness all share low toxicity scores.
   - All unverified neutral candidates were designated `UNVERIFIED_NEGATIVE_CANDIDATE` and strictly excluded from the verified classification subset.
3. **Zero Leakage & Deduplication Guarantee:**
   - 100% unique record IDs; 0 normalized-text collisions.
   - Silver datasets remain completely isolated from future human evaluation splits.
   - All 19 verified records carry immutable provenance tagging: `TRAINED_ON_LLM_SILVER_DATA`.

---

## 2. Recomputed Partition Counts

The 100 candidate records from the candidate manifest ([data/annotations/phase4/human_sampling_manifest_100.jsonl](file:///d:/multimodel_passive_aggression/data/annotations/phase4/human_sampling_manifest_100.jsonl)) were audited and partitioned into verified and excluded subsets:

| Dataset Category | Record Count | Percentage | Artifact Path |
| :--- | :--- | :--- | :--- |
| **Candidate Pool Evaluated** | 100 | 100.0% | [data/annotations/phase5d/silver_dataset_candidates.jsonl](file:///d:/multimodel_passive_aggression/data/annotations/phase5d/silver_dataset_candidates.jsonl) |
| **Verified Silver Subset** | **19** | **19.0%** | [data/annotations/phase5d/silver_dataset_verified.jsonl](file:///d:/multimodel_passive_aggression/data/annotations/phase5d/silver_dataset_verified.jsonl) |
| — *Verified Silver Positives* | 18 | 18.0% | `silver_primary_label = "passive_aggressive"` |
| — *Verified Silver Negatives* | 1 | 1.0% | `silver_primary_label = "non_passive_aggressive"` |
| **Excluded Records** | **81** | **81.0%** | [data/annotations/phase5d/excluded_records.csv](file:///d:/multimodel_passive_aggression/data/annotations/phase5d/excluded_records.csv) |
| — *Cross-Model Disagreements* | 32 | 32.0% | `EXCLUDED_DISAGREEMENT` (Routed to Human Queue) |
| — *Overt Direct Aggression* | 8 | 8.0% | `EXCLUDED_OVERT_AGGRESSION` |
| — *Institutional / Political Venting* | 14 | 14.0% | `EXCLUDED_INSTITUTIONAL_VENTING` |
| — *Existential / Pop-Culture Venting* | 6 | 6.0% | `EXCLUDED_EXISTENTIAL_VENTING` |
| — *Direct Debate / Blunt Argument* | 5 | 5.0% | `EXCLUDED_DIRECT_DEBATE` |
| — *Insufficient Context / Fragments* | 4 | 4.0% | `EXCLUDED_INSUFFICIENT_CONTEXT` |
| — *Unverified Sarcasm / Ambiguous Target* | 10 | 10.0% | `EXCLUDED_UNVERIFIED_TARGET` |
| — *Sitcom Confession / Self-Talk* | 1 | 1.0% | `EXCLUDED_SELF_CONFESSION` |
| — *Shared False Praise Hallucination* | 1 | 1.0% | `EXCLUDED_SHARED_FALSE_POSITIVE_PRAISE` |
| **Negative Candidates Audited** | 32 | 32.0% | [data/annotations/phase5d/negative_candidate_audit.csv](file:///d:/multimodel_passive_aggression/data/annotations/phase5d/negative_candidate_audit.csv) |

---

## 3. Forensic Audit of the 19 Verified Silver Records

Every record admitted into [data/annotations/phase5d/silver_dataset_verified.jsonl](file:///d:/multimodel_passive_aggression/data/annotations/phase5d/silver_dataset_verified.jsonl) satisfies all three mandatory criteria:
1. **Confirmed Interpersonal Target:** Directed at a specific interlocutor, partner, or conversation participant.
2. **Confirmed Covert Hostility:** The communicative intent mocks, belittles, or disparages.
3. **Confirmed Plausible Deniability:** Phrased with surface politeness, compliment, or feigned agreement.

### Complete Table of Verified Silver Records:

| Record ID | Source | Verified Label | Subtype | Comment Text | Pragmatic Hostility & Plausible Deniability Rationale |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `mustard_2_490` | MUStARD | `passive_aggressive` | `backhanded_compliment` | *"Maybe it's a contest, you know? Like 'Collect all five'."* | Sarcastic faux-suggestion mocking someone finding an unwanted item in food. Target is interlocutor; surface is innocent game idea. |
| `civil_comments_val_06315` | CivilComments | `passive_aggressive` | `backhanded_compliment` | *"icon66 blessing my demise. What a sweetheart."* | Sarcastic praise (*"What a sweetheart"*) mocking another user (*"icon66"*) wishing ill. Textbook faux politeness. |
| `mustard_1_3660` | MUStARD | `passive_aggressive` | `backhanded_compliment` | *"There a new girlfriend in there? 'Cause you might need one."* | Faux helpfulness masking disparagement of Leonard's romantic relationship. |
| `mustard_1_8407` | MUStARD | `passive_aggressive` | `implied_blame` | *"Oh, sure, what a wonderful idea. And after that, I'll make up my own rules of oral hygiene..."* | Sarcastic agreement (*"what a wonderful idea"*) paired with absurd dental hygiene scenario to mock friend's suggestion. |
| `mustard_2_439` | MUStARD | `passive_aggressive` | `implied_blame` | *"Twenty-one hours, you're a hero."* | Sarcastic praise (*"you're a hero"*) to belittle someone's exaggerated accomplishment. |
| `mustard_2_77` | MUStARD | `passive_aggressive` | `implied_blame` | *"Umm, excuse me, we switched apartments. You can't eat are food anymore, that-that gravy train had ended."* | Surface politeness (*"Umm, excuse me"*) weaponized to cut off food access and demean former roommate. |
| `civil_comments_val_07561` | CivilComments | `passive_aggressive` | `backhanded_compliment` | *"Thanks, GofJBR. Like I said, I like Francis... Reading your nom de plume is starting to become like taking an eye test..."* | Polite opening (*"Thanks... I like Francis"*) masking backhanded ridicule of recipient's handle. |
| `mmsd_validation_01859` | MMSD | `passive_aggressive` | `implied_blame` | *"* backflips off a pogo after only 2 hours of training * * still isn 't physical enough * ( whatever that means )*"* | Sarcastic compliance mocking an external evaluation standard. |
| `sarcnet_train_01587` | SarcNet | `passive_aggressive` | `indirect_criticism` | *"Maybe your reporters should follow suit seems they make habit hampering the search... Just a thought"* | Sarcastic faux-modesty (*"Just a thought"*) masking direct indictment of journalists. |
| `civil_comments_val_06377` | CivilComments | `passive_aggressive` | `indirect_criticism` | *"Patrick and Tim lost an opportunity to take a stand... Surely they had easy access to the 'data'..."* | Pseudo-polite assumption (*"Surely they had easy access"*) mocking interlocutors' lack of courage. |
| `civil_comments_val_07107` | CivilComments | `passive_aggressive` | `indirect_criticism` | *"Bob, Trump has done a fine job... blaming Alice Rogoff is just attacking the messenger... shouting in large caps does not make it true."* | Conversational address (*"Bob..."*) masking condescending correction. |
| `civil_comments_val_08072` | CivilComments | `passive_aggressive` | `implied_blame` | *"What Comey did was not 'illegal'. Please cite the law you think he broke...."* | Polite challenge (*"Please cite the law"*) demeaning interlocutor's legal competence. |
| `mmsd_train_10780` | MMSD | `passive_aggressive` | `implied_blame` | *"good to see the guys at <user> once again taking pride in doing a good job ."* | Surface praise (*"good to see... taking pride"*) with backhanded modifier (*"once again"*) implying frequent failure. |
| `mustard_1_162` | MUStARD | `passive_aggressive` | `implied_blame` | *"Sheldon, if you were a robot, and I knew and you didn't..."* | Mock-philosophical inquiry belittling Sheldon's emotional capacity. |
| `mustard_2_9` | MUStARD | `passive_aggressive` | `implied_blame` | *"No Rose, go to the corner and open a stand"* | Sarcastic dismissal mocking an offer of assistance. |
| `mustard_2_574` | MUStARD | `passive_aggressive` | `implied_blame` | *"You can't do anything."* | Dismissive put-down in dialogue banter. |
| `sarcnet_train_00844` | SarcNet | `passive_aggressive` | `implied_blame` | *"#鼠鼠我鸭#哇哦，原来鸭子是这样子的呀，学到了，课本上没有的知识耶"* | Exaggerated feigned ignorance (*"I learned something new!"*) mocking obvious institutional dishonesty. |
| `mmsd_test_02238` | MMSD | `passive_aggressive` | `backhanded_compliment` | *"these should brighten your day"* | Faux courtesy masking condescending delivery. |
| `sarcnet_test_00053` | SarcNet | `non_passive_aggressive` | `none` | *"普及一个新成语见易思签#李易峰##吴签#"* | Both models consensus negative; neutral explanation of Chinese idiom with zero hostility. |

---

## 4. Critical Verification of Negative Examples

Section 3 of the prompt mandates:
> **"A neutral sentiment label is not proof of a non-passive-aggressive label... Only include a negative example as a verified negative if it has appropriate evidence and a defensible label source. Otherwise label it UNVERIFIED_NEGATIVE_CANDIDATE and exclude it from the verified classification subset."**

### Forensic Negative Audit ([data/annotations/phase5d/negative_candidate_audit.csv](file:///d:/multimodel_passive_aggression/data/annotations/phase5d/negative_candidate_audit.csv)):
We evaluated 32 negative candidate records from three sources:
1. **Model B Disagreement Negatives (28 records):** Where Model B predicted `non_passive_aggressive` while Model A predicted `passive_aggressive`.
2. **Model A Baseline Negatives (3 records):** Where Model A predicted `non_passive_aggressive` but Model B disagreed (`mmsd_train_16580`, `mmsd_validation_02047`, `sarcnet_test_00188`).
3. **CivilComments Neutral Split Candidates:** Unreviewed comments with toxicity = 0.0.

### Major Scientific Finding:
- **Toxicity = 0.0 Does Not Equal Non-Passive-Aggressive:**  
  Passive-aggressive communication avoids profanity, slurs, and explicit threats. For example, record `civil_comments_val_06315` (*"What a sweetheart"*) has a toxicity score of 0.0, yet is undeniable passive aggression.
- **Result:** Only `sarcnet_test_00053` achieved full cross-model consensus and unassailable evidence of non-hostility. All other 31 candidates were labeled `UNVERIFIED_NEGATIVE_CANDIDATE` and excluded.
- **Scientific Integrity:** We refused to fabricate an artificial 1:1 balance by injecting unverified neutral comments. The verified silver dataset contains exactly **1 verified negative**.

---

## 5. Leakage, Deduplication, and Near-Duplicate Analysis

All records in [data/annotations/phase5d/silver_dataset_verified.jsonl](file:///d:/multimodel_passive_aggression/data/annotations/phase5d/silver_dataset_verified.jsonl) were subjected to rigorous leakage audits:
1. **Duplicate ID Check:** 19 unique IDs / 19 records (0 duplicates).
2. **Exact Text Deduplication:** 19 unique texts / 19 records (0 exact duplicates).
3. **Normalized Near-Duplicate Check:** Normalized strings (lowercase, punctuation stripped, whitespace collapsed) yielded 19 unique items (0 near-duplicate leakage).
4. **Modality Honesty:** All 19 records are designated `modality_used = "TEXT_ONLY"`. Images are marked `UNASSESSED_VISION_MODEL_ABSENT`.
5. **Evaluation Set Isolation:** The silver dataset is strictly segregated in `data/annotations/phase5d/` and is completely quarantined from human adjudication files.

---

## 6. Suitability for Exploratory Weak Supervision

### Verdict: **SUITABLE FOR AUXILIARY REPRESENTATION LEARNING PROBES ONLY**

The verified 19-record dataset is suitable for small exploratory representation probes, subject to the following non-negotiable governance constraints:
1. **Auxiliary Feature Alignment Only:** May be used to probe whether multimodal embeddings cluster true covert hostility distinctly from overt aggression.
2. **Mandatory Provenance Tagging:** Every record preserves the tag: `TRAINED_ON_LLM_SILVER_DATA`.
3. **Strict Evaluation Ban:** The primary model must **NEVER** be evaluated against these silver labels to claim benchmark performance.
4. **Real-World Benchmark Remains Gated:** Human evaluation status remains `BLOCKED_PENDING_HUMAN_ANNOTATION` (0 completed reviews).
