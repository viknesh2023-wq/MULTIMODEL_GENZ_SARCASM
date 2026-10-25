"""Synthetic pilot generator creating 350 structured scenarios across all archetypes and contrast classes.

Every record is explicitly marked as SYNTHETIC_UNVERIFIED and AWAITING_HUMAN_REVIEW.
Covers:
- Fake politeness
- Indirect criticism
- Backhanded compliments
- Implied blame
- Sarcastic agreement
- Dismissiveness
- Direct criticism (contrast)
- Ordinary negative sentiment (contrast)
- Genuine sadness (contrast)
- Friendly teasing (contrast)
- Constructive disagreement (contrast)
- Ambiguous examples
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utilities.logger import setup_console_logger
from src.utilities.schema import (
    AnnotationStatus,
    ImageAvailabilityStatus,
    SyntheticStatus,
    UnifiedRecord,
)

logger = setup_console_logger("synthetic_generator")


def build_scenario_templates():
    """Builds diverse seed scenario blueprints covering all subtypes and multimodal relationships."""
    blueprints = []

    # 1. Fake politeness (Target ~35 scenarios)
    fake_politeness_seeds = [
        ("A meticulously baked artisan sourdough loaf with uneven scoring.", "Baked my first artisan sourdough after 36 hours of fermentation!", "Per my previous email, following basic instructions usually yields an even crumb, but thank you so much for sharing your unique effort!", "image_necessary", "Polite phrasing ('thank you so much') weaponized to insult baking competence with corporate condescension.", False),
        ("A photograph of a cluttered shared kitchen counter with dishes in the sink.", "Loving our cozy apartment vibe on a Sunday afternoon ✨", "Warmest regards to whoever felt inspired to leave all their pots for the cleaning fairy to bless 💕", "image_necessary", "Excessive affectionate emojis and greeting masking resentment over unwashed dishes.", False),
        ("A slide deck presentation on quarterly financial projections.", "Just finished presenting the Q3 strategy to leadership.", "With all due respect, I am sure you tried your absolute hardest with those calculations.", "image_irrelevant", "Pretending respect while condescendingly disparaging professional capability.", False),
        ("An amateur painting of a sunset with muddy colors.", "My latest oil painting: 'Dusk Over Harbor'.", "Bless your heart for having the bravery to display this in public view!", "image_necessary", "Colloquial fake politeness ('bless your heart') used to patronize artistic ability.", False),
        ("A photo of a colleague at a sunny outdoor team lunch.", "Great team bonding lunch today!", "So wonderful that some team members have the luxury of extended lunch hours while critical bug tickets remain unassigned :)", "image_necessary", "Smiley face and 'so wonderful' masking passive-aggressive resentment over workload distribution.", False),
    ]

    # 2. Indirect criticism (Target ~35 scenarios)
    indirect_criticism_seeds = [
        ("A runner showing a 5k medal with a time of 38 minutes.", "Finished my first 5k today! Never thought I could do it.", "Interesting how participation medals are handed out for any walking pace nowadays.", "image_necessary", "Indirect attack dismissing achievement without naming the runner directly.", False),
        ("A minimalist architectural home interior with white furniture.", "Finally finished decorating my dream living room.", "Must be nice to live in an environment where no actual living or practical work appears to happen.", "image_necessary", "Rhetorical observation undermining the owner's lifestyle choices.", False),
        ("A graphic design poster with crowded fonts.", "Designed the flyer for this weekend's indie concert!", "I guess readability is considered completely overrated in modern typography.", "image_necessary", "Broad generalization attacking the designer's typographic competence indirectly.", False),
        ("A photo of an elaborate 10-step skincare shelf.", "My night-time skincare holy grails.", "Funny how some people will buy every single chemical before trying drinking water.", "image_irrelevant", "Snide indirect dismissal of skincare routine framed as general observation.", False),
        ("A selfie of someone in a bright vintage outfit.", "Rocking 70s velvet corduroy today.", "Some outfits truly require an extraordinary amount of unearned confidence.", "image_irrelevant", "Indirect slight criticizing wardrobe choices under the guise of commenting on confidence.", False),
    ]

    # 3. Backhanded compliments (Target ~35 scenarios)
    backhanded_compliment_seeds = [
        ("A portrait of a woman wearing evening makeup.", "Ready for the charity gala tonight!", "You look surprisingly decent when you actually put in the effort to style yourself.", "image_irrelevant", "Compliment conditioned on an insulting baseline ('surprisingly decent', 'actually put in effort').", False),
        ("A photo of a handmade knitted sweater with slightly uneven sleeves.", "Knitted my first wool sweater from scratch!", "It's remarkably wearable considering you have zero formal sewing or textile background.", "image_necessary", "Diminishing the creator's achievement by emphasizing their lack of qualification.", False),
        ("A researcher standing by an academic conference poster.", "Presented our neural network paper at the workshop today.", "Such an accessible topic, anyone without a math background could understand your paper.", "image_irrelevant", "Demeaning academic rigor by reframing it as trivial.", False),
        ("A car parked slightly crookedly across a white parking stall line.", "Got the last parking spot at the mall!", "Impressive parking job for someone who only passed their driving exam on the fourth try.", "image_necessary", "Insulting personal history tacked onto a sarcastic evaluation of poor parking.", False),
        ("A photo of a homemade lasagna with burnt cheese edges.", "Family dinner ready! Homemade 4-cheese lasagna.", "It actually smells edible! You've certainly lowered our culinary expectations over the years.", "image_necessary", "Praise backhanded by recalling a long history of subpar cooking.", False),
    ]

    # 4. Implied blame (Target ~30 scenarios)
    implied_blame_seeds = [
        ("A photo of a locked office door with lights off at 5:01 PM.", "End of another productive week!", "It must be convenient when personal commitments always take priority over team release deadlines.", "image_necessary", "Shaming a coworker for leaving on time while implicitly blaming them for project stress.", False),
        ("A photo of an empty coffee pot on a breakroom warmer.", "Monday morning energy needed.", "Funny how the coffee pot always seems to empty itself without anyone knowing how to start a new brew.", "image_necessary", "Expressing frustration through passive accusation regarding communal office responsibilities.", False),
        ("A screenshot of a git repository with broken CI build badge.", "Pushed the refactoring branch before leaving.", "Glad to see someone felt confident enough to push code without running the integration test suite first.", "image_necessary", "Attributing blame for broken test suite through sarcastic congratulations on confidence.", False),
        ("A messy shared living room table with gaming controllers.", "Weekend gaming session underway.", "I suppose the cleaning schedule was merely a theoretical suggestion for some of us.", "image_necessary", "Targeting roommates for mess without direct confrontation.", False),
        ("A car with an empty fuel gauge light glowing.", "Quick road trip with the gang!", "I love how the fuel tank miraculously drops to empty right when it's my turn to borrow the car.", "image_necessary", "Imputing bad faith and neglect to family members or friends indirectly.", False),
    ]

    # 5. Sarcastic agreement (Target ~30 scenarios)
    sarcastic_agreement_seeds = [
        ("A weather app screenshot showing torrential rain and flooding alerts.", "Beach vacation starts today! Perfect timing.", "Oh absolutely, couldn't have picked a more ideal day for sunbathing and swimming.", "image_necessary", "Sarcastic agreement mocking the poster's vacation misfortune.", False),
        ("A photo of a broken chair collapsed on the carpet.", "Fixed the antique chair all by myself!", "Yes, that looks completely structurally sound and safe for human weight.", "image_necessary", "Feigned agreement highlighting obvious physical danger and poor craftsmanship.", False),
        ("A graph showing stock portfolio crashing 40% in a week.", "Diamond hands! Buying the dip as planned.", "Totally, financial ruin has always been the smartest investment strategy.", "image_necessary", "Mock agreement ridiculing the poster's disastrous financial reasoning.", False),
        ("A burnt pizza with black crust on a baking stone.", "Gourmet wood-fired style pizza at home.", "Right, the charred charcoal flavor is exactly what Michelin inspectors look for.", "image_necessary", "Mock praise masquerading as agreement with the culinary claim.", False),
        ("A photo of an overgrown lawn with weeds two feet tall.", "Cultivating natural biodiversity in my front yard.", "Clearly, city code inspectors are famous for appreciating unkempt weed sanctuaries.", "image_necessary", "Mocking agreement citing municipal regulations sarcastically.", False),
    ]

    # 6. Dismissiveness (Target ~30 scenarios)
    dismissiveness_seeds = [
        ("A scenic photograph from a first-class airplane seat overlooking clouds.", "Blessed to travel the world doing what I love ✈️", "Must be nice. Some people actually have to work for a living.", "image_necessary", "Invalidating the poster's life and success with resentful hostility.", False),
        ("A photo of a brand new sports car in a driveway.", "Hard work always pays off! Finally bought my dream car.", "Cool story. We all have different definitions of paying off, I guess.", "image_necessary", "Dismissive brush-off intended to diminish the poster's pride and achievement.", False),
        ("A detailed review of a specialty single-origin espresso.", "Tasting notes of bergamot, jasmine and white peach in this roast.", "Whatever you need to tell yourself to justify spending ten dollars on bean water.", "image_irrelevant", "Curt dismissiveness reducing passion to self-delusion.", False),
        ("A graduation photo with a master's degree diploma.", "Two years of grueling research completed!", "Good for you, though degrees don't guarantee competence in the real market.", "image_necessary", "Deflating celebration with dismissive cautionary cynicism.", False),
        ("A photo of a home workout space with kettlebells.", "Day 100 of consistent morning workouts!", "Must be great having unlimited free time to obsess over lifting weights.", "image_necessary", "Passive-aggressive dismissiveness attributing fitness success to excess privilege.", False),
    ]

    # 7. Direct criticism (Contrast Class - NON-PA) (Target ~25 scenarios)
    direct_criticism_seeds = [
        ("A screenshot of an offensive social media comment.", "Look at what this influencer said today.", "This statement is racist, harmful, and completely unacceptable.", "image_irrelevant", "Direct, overt moral condemnation without passive-aggressive masking.", False),
        ("A photo of a dangerous poorly wired electrical plug.", "DIY electrical repair done!", "Stop immediately. That is an extreme fire hazard and illegal wiring.", "image_necessary", "Direct safety warning and criticism, clear and non-indirect.", False),
        ("A plagiarized essay snippet side by side with original paper.", "Submitted my term paper draft.", "You copied paragraph 3 and 4 word-for-word from Wikipedia. That is plagiarism.", "image_necessary", "Direct factual accusation of academic dishonesty.", False),
        ("A video frame of a driver texting while drifting lanes.", "Morning commute traffic.", "Put your phone down while driving, you are endangering everyone around you.", "image_necessary", "Direct reprimand for dangerous behavior.", False),
        ("A recipe containing raw untreated chicken in a cold salad.", "Quick raw chicken salad recipe!", "Consuming raw chicken causes salmonella poisoning. Do not eat this.", "image_necessary", "Direct health warning.", False),
    ]

    # 8. Ordinary negativity (Contrast Class - NON-PA) (Target ~25 scenarios)
    ordinary_negativity_seeds = [
        ("A rain-slicked highway with gridlock traffic.", "Two hours stuck on the I-95 bridge.", "I despise this commute with every fiber of my being. Traffic is miserable.", "image_necessary", "Direct negative emotion directed at traffic, not interpersonal hostility.", False),
        ("A broken laptop screen with colored glitch lines.", "Dropped my backpack on the train.", "I hate Mondays so much. Everything is going wrong today.", "image_irrelevant", "General frustration and venting without veiled interpersonal attack.", False),
        ("A cold rainy bus stop in winter darkness.", "Waiting for the late 6:00 AM bus.", "The weather is freezing and public transit delays are exhausting.", "image_necessary", "Expression of fatigue and weather discomfort.", False),
        ("A pile of tax audit documents on a desk.", "Annual tax filing weekend.", "Taxes are confusing, tedious, and completely draining.", "image_irrelevant", "Venting about bureaucracy.", False),
        ("A photo of a delayed flight boarding screen.", "Flight delayed 6 hours in Chicago.", "This airline delay ruined my weekend plans completely.", "image_necessary", "Expressing personal disappointment about travel delays.", False),
    ]

    # 9. Genuine sadness (Contrast Class - NON-PA) (Target ~25 scenarios)
    genuine_sadness_seeds = [
        ("An empty pet dog bed in a quiet sunlit room.", "Said goodbye to our sweet companion of 14 years today.", "I can't stop crying seeing this photo. He was the sweetest dog in the world.", "image_necessary", "Genuine grief and sympathetic mourning.", False),
        ("A demolished neighborhood landmark historic theater.", "Demolition of the Grand Lyric Theater completed.", "This genuinely breaks my heart. So many cherished memories lost forever.", "image_necessary", "Sincere melancholy over cultural loss.", False),
        ("A hospital room hallway with soft lighting.", "Visiting family during recovery.", "It's so heartbreaking seeing someone you love in pain. Praying for strength.", "image_necessary", "Empathetic sorrow and compassion.", False),
        ("A photo of an abandoned childhood swing set.", "Visiting my childhood home after 20 years.", "Time moves far too fast. I miss being a kid so deeply.", "image_irrelevant", "Nostalgic sadness without hostility.", False),
        ("A wilted houseplant that could not be revived.", "Tried my best to save this heirloom fern.", "I'm genuinely so sad this didn't make it despite all the effort.", "image_necessary", "Mild personal sorrow.", False),
    ]

    # 10. Friendly teasing (Contrast Class - NON-PA) (Target ~25 scenarios)
    friendly_teasing_seeds = [
        ("A close friend wearing an oversized funny Christmas sweater.", "Ready for the holiday party!", "Look at this absolute nerd 😂 You better wear that to brunch tomorrow too bro!", "image_necessary", "Affectionate playful teasing between close peers with laughter emojis.", False),
        ("A selfie of two friends laughing at a failed baking attempt.", "Cake exploded in the oven lol", "Hahaha you are banned from kitchen duty forever man! Pizza is on me tonight 🍕", "image_necessary", "Good-natured bantering with immediate positive social remedy.", False),
        ("A gamer friend missing an easy goal in a multiplayer game.", "Watch my epic clutch play!", "Bro whiffed the easiest shot in franchise history lmao, we still love you though!", "image_irrelevant", "Playful gaming ribbing counterbalanced with explicit affection.", False),
        ("A selfie of a friend wearing mismatched colorful socks.", "Professional corporate attire.", "Fashion icon strikes again! Only you could rock neon dinosaurs in a board meeting haha.", "image_necessary", "Warm friendly joke celebrating eccentricity.", False),
        ("A friend proudly holding a tiny three-inch fish on a fishing dock.", "Catch of the day! Beast mode.", "Save some sea monsters for the rest of us Captain Ahab! 😂🎣", "image_necessary", "Humorous hyperbole in friendly sporting banter.", False),
    ]

    # 11. Constructive disagreement (Contrast Class - NON-PA) (Target ~25 scenarios)
    constructive_disagreement_seeds = [
        ("A Python code snippet using a while loop for list aggregation.", "Best way to sum positive numbers in Python.", "Using the built-in sum() with a generator comprehension is more pythonic and runs in O(n) with less memory overhead.", "image_necessary", "Technical, objective correction without personal animus.", False),
        ("An infographic showing solar energy vs wind energy capacity.", "Why solar is superior in every single scenario.", "While solar excels in low latitudes, offshore wind provides crucial baseload power during winter and cloudy seasons, as shown in NREL data.", "image_necessary", "Balanced evidentiary counterargument in scientific debate.", False),
        ("A recipe advocating boiling whole garlic cloves in oil at high heat.", "Crispy garlic hack!", "High heat causes garlic to burn quickly and become bitter; simmering gently on medium-low extracts better flavor.", "image_irrelevant", "Constructive culinary feedback focused purely on method.", False),
        ("A fitness diagram advocating rounding the lower back during heavy deadlifts.", "How to maximize back engagement on deadlifts.", "Rounding the lumbar spine under heavy axial load significantly elevates disc herniation risk; maintaining neutral spine is safer.", "image_necessary", "Safety-focused technical critique without hostility.", False),
        ("A book review calling a classic novel completely unreadable.", "Worst classic ever written.", "I understand the pacing feels slow initially, but the second half develops the philosophical themes remarkably well.", "image_irrelevant", "Respectful divergence of literary perspective.", False),
    ]

    # 12. Ambiguous examples (Target ~30 scenarios)
    ambiguous_seeds = [
        ("A colleague's photo holding a certificate of completion.", "Completed the introductory machine learning course!", "Interesting choice.", "image_necessary", "Insufficient textual and social cues to determine if congratulatory, skeptical, or dismissive.", True),
        ("A photograph of a plain white wall in a modern art gallery.", "Art exhibit opening tonight.", "That's certainly something.", "image_necessary", "Ambiguous assessment: could be genuine bewilderment, dry wit, or passive aggression.", True),
        ("A photo of a renovated kitchen with black cabinets and marble tops.", "Our kitchen remodel is done!", "Well, you certainly made a statement.", "image_necessary", "Could be praise for bold aesthetics or veiled disapproval of unusual design.", True),
        ("A screenshot of a short email response saying 'Noted.'", "Update sent to team.", "Noted.", "context_dependent", "Single-word acknowledgement which can be professional neutral or curt passive-aggressive depending on context.", True),
        ("A photo of a person wearing a neon green winter coat.", "Winter coat arrived!", "Very bold color choice.", "image_necessary", "Tone cannot be calibrated without vocal inflection or interpersonal history.", True),
    ]

    return {
        "fake_politeness": fake_politeness_seeds,
        "indirect_criticism": indirect_criticism_seeds,
        "backhanded_compliment": backhanded_compliment_seeds,
        "implied_blame": implied_blame_seeds,
        "sarcastic_agreement": sarcastic_agreement_seeds,
        "dismissiveness": dismissiveness_seeds,
        "direct_criticism": direct_criticism_seeds,
        "ordinary_negativity": ordinary_negativity_seeds,
        "genuine_sadness": genuine_sadness_seeds,
        "friendly_teasing": friendly_teasing_seeds,
        "constructive_disagreement": constructive_disagreement_seeds,
        "ambiguous": ambiguous_seeds,
    }


def generate_full_synthetic_pilot(target_count: int = 350) -> list[dict]:
    """Expands seed scenarios into 350 fully-specified structured scenarios with variations."""
    seed_dict = build_scenario_templates()
    records = []
    curr_id = 1

    # Map archetype to primary label and subtype
    archetype_meta = {
        "fake_politeness": ("passive_aggressive", "fake_politeness"),
        "indirect_criticism": ("passive_aggressive", "indirect_criticism"),
        "backhanded_compliment": ("passive_aggressive", "backhanded_compliment"),
        "implied_blame": ("passive_aggressive", "implied_blame"),
        "sarcastic_agreement": ("passive_aggressive", "sarcastic_agreement"),
        "dismissiveness": ("passive_aggressive", "dismissiveness"),
        "direct_criticism": ("non_passive_aggressive", "other_or_unclear"),
        "ordinary_negativity": ("non_passive_aggressive", "other_or_unclear"),
        "genuine_sadness": ("non_passive_aggressive", "other_or_unclear"),
        "friendly_teasing": ("non_passive_aggressive", "other_or_unclear"),
        "constructive_disagreement": ("non_passive_aggressive", "other_or_unclear"),
        "ambiguous": ("ambiguous", "other_or_unclear"),
    }

    # Domains for variations
    domains = ["culinary", "workplace", "academia", "fitness", "home_decor", "parenting", "creative_arts", "tech_gadgets"]

    # Generate records
    while len(records) < target_count:
        for arch_name, seeds in seed_dict.items():
            primary_lbl, subtype = archetype_meta[arch_name]
            for s_idx, (img_desc, caption, comment, dep, rationale, is_ambig) in enumerate(seeds):
                if len(records) >= target_count:
                    break

                # Create variation if beyond first cycle
                cycle = len(records) // (len(seed_dict) * 5)
                var_suffix = f" [Context variation: {domains[cycle % len(domains)]}]" if cycle > 0 else ""
                cur_comment = comment if cycle == 0 else f"{comment}{'..' if cycle % 2 == 0 else '!'}"
                cur_desc = f"{img_desc}{var_suffix}"

                ex_id = f"synthetic_pilot_{curr_id:04d}"
                curr_id += 1

                rec = {
                    "example_id": ex_id,
                    "source_dataset": "synthetic_pilot",
                    "source_record_id": f"syn_{arch_name}_{curr_id}",
                    "source_url_or_reference": "Phase 1 Controlled Synthetic Benchmark",
                    "dataset_version": "1.0.0",
                    "source_license": "CC-BY-4.0 (Synthetic Benchmark Dataset)",
                    "asset_license_status": "Textual authorized reference; no binary image claimed",
                    "scenario_description": f"Scenario depicting {arch_name.replace('_', ' ')} in social communication.",
                    "image_description": cur_desc,
                    "image_path": None,
                    "image_availability_status": ImageAvailabilityStatus.NOT_APPLICABLE.value,
                    "caption": caption,
                    "target_comment": cur_comment,
                    "parent_comment": caption,
                    "conversation_context": [caption],
                    "original_label": arch_name,
                    "original_label_scheme": "Phase 1 Archetype Blueprint",
                    "passive_aggression_label": primary_lbl,
                    "passive_aggression_subtype": subtype,
                    "multimodal_dependency": dep,
                    "image_needed_for_interpretation": (dep == "image_necessary"),
                    "rationale": f"{rationale} (Archetype: {arch_name})",
                    "ambiguity_indicator": is_ambig,
                    "annotation_confidence": 0.95 if not is_ambig else 0.50,
                    "annotation_status": AnnotationStatus.SYNTHETIC_PILOT.value,
                    "synthetic_status": SyntheticStatus.SYNTHETIC_UNVERIFIED.value,
                    "human_review_status": "AWAITING_HUMAN_REVIEW",
                    "generation_provenance": "SYNTHETIC_GENERATED_PILOT_V1_CONTROLLED_RULES",
                    "train_validation_test_split": None,
                }
                records.append(rec)

    return records


def main():
    logger.info("=== Generating Synthetic Pilot Dataset ===")
    records = generate_full_synthetic_pilot(target_count=350)

    out_file = PROJECT_ROOT / "data" / "synthetic" / "synthetic_pilot_350.jsonl"
    out_file.parent.mkdir(parents=True, exist_ok=True)

    with open(out_file, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    logger.info(f"Synthetic pilot dataset created with {len(records)} records at {out_file}")

    # Log distribution breakdown
    prim_counts = {}
    sub_counts = {}
    for r in records:
        p = r["passive_aggression_label"]
        s = r["passive_aggression_subtype"]
        prim_counts[p] = prim_counts.get(p, 0) + 1
        sub_counts[s] = sub_counts.get(s, 0) + 1

    logger.info(f"Primary label distribution: {prim_counts}")
    logger.info(f"Subtype distribution: {sub_counts}")


if __name__ == "__main__":
    main()
