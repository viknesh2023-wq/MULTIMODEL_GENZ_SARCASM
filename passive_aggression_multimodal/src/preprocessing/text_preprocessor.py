"""Pragmatic text preprocessor for social media comments and captions.

Designed specifically for passive-aggression and irony detection:
- Preserves emojis, emoticons, and sentiment-bearing punctuation (???, !!!, ...).
- Preserves negation tokens (not, never, no, n't, hardly) and capitalization cues (ALL CAPS, SpOnGeBoB case).
- Normalizes social URLs and user mentions while retaining structural pragmatic information.
- Prevents destructive stemming, lemmatization, and stopword stripping that destroy pragmatic nuances.
"""
from __future__ import annotations

import html
import re
import unicodedata
from typing import Dict, Optional


class PragmaticTextPreprocessor:
    """Non-destructive social media text preprocessor."""

    def __init__(self, normalize_urls: bool = True, normalize_mentions: bool = False):
        self.normalize_urls = normalize_urls
        self.normalize_mentions = normalize_mentions

        # Regex patterns
        self.url_pattern = re.compile(r"https?://\S+|www\.\S+")
        self.mention_pattern = re.compile(r"@[\w_]+")
        self.multi_space_pattern = re.compile(r"[ \t]+")
        self.multi_newline_pattern = re.compile(r"\n\s*\n+")

    def clean_encoding(self, text: str) -> str:
        """Fixes unicode normalization and unescapes HTML entities."""
        if not text:
            return ""
        # Unescape HTML entities (&amp;, &lt;, etc.)
        t = html.unescape(text)
        # Unicode NFKC normalization (retains emoji characters while standardizing ligatures)
        t = unicodedata.normalize("NFKC", t)
        return t

    def extract_pragmatic_features(self, text: str) -> Dict[str, any]:
        """Extracts pragmatic surface features useful for downstream classifier attention."""
        if not text:
            return {
                "has_all_caps": False,
                "has_repeated_punct": False,
                "has_emojis": False,
                "char_length": 0,
                "word_count": 0,
            }

        words = text.split()
        all_caps_words = [w for w in words if len(w) > 1 and w.isupper() and w.isalpha()]
        has_repeated_punct = bool(re.search(r"([!?.]){2,}", text))
        # Simple unicode range check for emojis
        has_emojis = any(
            ("\U0001F600" <= c <= "\U0001F64F") or  # Emoticons
            ("\U0001F300" <= c <= "\U0001F5FF") or  # Misc symbols and pictographs
            ("\U0001F680" <= c <= "\U0001F6FF") or  # Transport and map
            ("\U0001F900" <= c <= "\U0001F9FF")     # Supplemental symbols
            for c in text
        )

        return {
            "has_all_caps": len(all_caps_words) > 0,
            "all_caps_count": len(all_caps_words),
            "has_repeated_punct": has_repeated_punct,
            "has_emojis": has_emojis,
            "char_length": len(text),
            "word_count": len(words),
        }

    def preprocess(self, text: Optional[str]) -> str:
        """
        Executes normalized cleaning:
        - Decodes entities
        - Normalizes URLs to <URL>
        - Standardizes multi-spaces while preserving linebreaks, emojis, punctuation, and casing
        """
        if text is None:
            return ""

        t = self.clean_encoding(text.strip())
        if not t:
            return ""

        # Normalize URLs
        if self.normalize_urls:
            t = self.url_pattern.sub("<URL>", t)

        # Normalize Mentions if requested
        if self.normalize_mentions:
            t = self.mention_pattern.sub("<USER>", t)

        # Normalize excessive whitespace but preserve single punctuation repetitions (e.g. '...', '???')
        t = self.multi_space_pattern.sub(" ", t)
        t = self.multi_newline_pattern.sub("\n", t)

        return t.strip()


TextPreprocessor = PragmaticTextPreprocessor


def get_default_preprocessor() -> PragmaticTextPreprocessor:
    return PragmaticTextPreprocessor()

