"""Context preprocessor for threaded replies, parent comments, and multi-turn dialogues."""
from __future__ import annotations

from typing import List, Optional, Tuple
from src.preprocessing.text_preprocessor import PragmaticTextPreprocessor


class ContextPreprocessor:
    """Preprocesses and formats conversational and parent-comment context."""

    def __init__(self, text_preprocessor: Optional[PragmaticTextPreprocessor] = None, max_context_words: int = 128):
        self.tp = text_preprocessor or PragmaticTextPreprocessor()
        self.max_context_words = max_context_words

    def format_dialogue_turns(self, turns: Optional[List[str]], speakers: Optional[List[str]] = None) -> str:
        """
        Formats a list of context turns chronologically with optional speaker tags.
        Truncates from the left (oldest turns) to retain immediate conversational precedence.
        """
        if not turns:
            return ""

        formatted_turns = []
        for idx, turn in enumerate(turns):
            clean_turn = self.tp.preprocess(turn)
            if not clean_turn:
                continue
            if speakers and idx < len(speakers):
                formatted_turns.append(f"[{speakers[idx]}]: {clean_turn}")
            else:
                formatted_turns.append(f"[Turn {idx+1}]: {clean_turn}")

        if not formatted_turns:
            return ""

        # Check total words; truncate from beginning if exceeds max_context_words
        combined = " \n ".join(formatted_turns)
        words = combined.split()
        if len(words) > self.max_context_words:
            truncated = words[-self.max_context_words:]
            combined = "... " + " ".join(truncated)

        return combined

    def build_context_aware_input(
        self,
        target_comment: str,
        caption: Optional[str] = None,
        parent_comment: Optional[str] = None,
        conversation_context: Optional[List[str]] = None,
        ocr_text: Optional[str] = None,
    ) -> Tuple[str, str]:
        """
        Constructs context-aware representation for sequence-pair transformer inputs:
        Returns: (context_prefix, comment_text)
        Where context_prefix bundles [Caption: ...] [In-Image Text: ...] [Thread Context: ...]
        """
        clean_comment = self.tp.preprocess(target_comment)
        context_parts = []

        if caption:
            clean_cap = self.tp.preprocess(caption)
            if clean_cap:
                context_parts.append(f"[Post Caption]: {clean_cap}")

        if ocr_text:
            clean_ocr = self.tp.preprocess(ocr_text)
            if clean_ocr:
                context_parts.append(f"[Visual Text]: {clean_ocr}")

        if parent_comment and not conversation_context:
            clean_parent = self.tp.preprocess(parent_comment)
            if clean_parent:
                context_parts.append(f"[Parent]: {clean_parent}")

        if conversation_context:
            formatted_dialogue = self.format_dialogue_turns(conversation_context)
            if formatted_dialogue:
                context_parts.append(f"[Thread]: {formatted_dialogue}")

        context_prefix = " | ".join(context_parts) if context_parts else ""
        return context_prefix, clean_comment
