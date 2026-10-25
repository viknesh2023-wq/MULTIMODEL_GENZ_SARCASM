"""Script preprocessing social media comments, captions, and context across datasets.

Reads unified records, performs non-destructive pragmatic normalization,
extracts rhetorical features, and saves output to data/processed/preprocessed_multimodal_records.jsonl.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.context_preprocessor import ContextPreprocessor
from src.preprocessing.text_preprocessor import PragmaticTextPreprocessor
from src.utilities.logger import setup_console_logger

logger = setup_console_logger("text_preprocessor_runner")


def run_text_preprocessing(input_path: Path, output_path: Path):
    logger.info(f"Starting text and context preprocessing on {input_path}...")
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        return

    tp = PragmaticTextPreprocessor(normalize_urls=True, normalize_mentions=False)
    cp = ContextPreprocessor(text_preprocessor=tp)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_records = 0
    emoji_count = 0
    caps_count = 0
    punct_count = 0

    with open(input_path, "r", encoding="utf-8") as f_in, open(output_path, "w", encoding="utf-8") as f_out:
        for line in f_in:
            if not line.strip():
                continue
            data = json.loads(line)
            total_records += 1

            # Target comment
            raw_target = data.get("target_comment", "")
            norm_target = tp.preprocess(raw_target)
            features = tp.extract_pragmatic_features(raw_target)

            if features["has_emojis"]:
                emoji_count += 1
            if features["has_all_caps"]:
                caps_count += 1
            if features["has_repeated_punct"]:
                punct_count += 1

            # Caption
            raw_caption = data.get("caption")
            norm_caption = tp.preprocess(raw_caption) if raw_caption else None

            # Context
            raw_parent = data.get("parent_comment")
            norm_parent = tp.preprocess(raw_parent) if raw_parent else None
            conv_turns = data.get("conversation_context")
            formatted_dialogue = cp.format_dialogue_turns(conv_turns) if conv_turns else None

            # Combined context prefix
            ctx_prefix, _ = cp.build_context_aware_input(
                target_comment=raw_target,
                caption=raw_caption,
                parent_comment=raw_parent,
                conversation_context=conv_turns,
                ocr_text=data.get("image_ocr_text"),
            )

            # Store derived fields without altering raw fields
            data["normalized_target_comment"] = norm_target
            data["normalized_caption"] = norm_caption
            data["normalized_parent_comment"] = norm_parent
            data["formatted_conversation_context"] = formatted_dialogue
            data["context_prefix_for_transformer"] = ctx_prefix
            data["pragmatic_features"] = features

            f_out.write(json.dumps(data, ensure_ascii=False) + "\n")

    logger.info(f"Finished text preprocessing for {total_records} records.")
    logger.info(f"Pragmatic markers: Emojis={emoji_count} ({emoji_count/total_records*100:.1f}%), "
                f"ALL CAPS={caps_count} ({caps_count/total_records*100:.1f}%), "
                f"Repeated Punctuation={punct_count} ({punct_count/total_records*100:.1f}%)")
    logger.info(f"Saved derived preprocessed records to {output_path}")


def main():
    logger.info("=== Running Text & Context Preprocessing Pipeline ===")
    input_file = PROJECT_ROOT / "data" / "interim" / "unified_raw_manifest.jsonl"
    output_file = PROJECT_ROOT / "data" / "processed" / "preprocessed_multimodal_records.jsonl"
    run_text_preprocessing(input_file, output_file)


if __name__ == "__main__":
    main()
