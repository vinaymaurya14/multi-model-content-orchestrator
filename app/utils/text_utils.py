"""General-purpose text utilities used across the application."""

from __future__ import annotations

import re
from typing import List


def truncate(text: str, max_length: int = 500, suffix: str = "...") -> str:
    """Truncate *text* to *max_length* characters, appending *suffix* if cut."""
    if len(text) <= max_length:
        return text
    return text[: max_length - len(suffix)].rsplit(" ", 1)[0] + suffix


def word_count(text: str) -> int:
    """Return the number of whitespace-separated words in *text*."""
    return len(text.split())


def estimate_tokens(text: str) -> int:
    """Rough token estimate (~4 characters per token for English)."""
    return max(1, len(text) // 4)


def extract_keywords(text: str, top_n: int = 10) -> List[str]:
    """Extract the most frequent significant words from *text*."""
    stop_words = {
        "the", "and", "for", "are", "but", "not", "you", "all", "can",
        "had", "her", "was", "one", "our", "out", "this", "that", "with",
        "have", "from", "they", "been", "said", "each", "which", "their",
        "will", "other", "about", "many", "then", "them", "these", "some",
        "would", "into", "more", "your", "what", "when", "make", "like",
    }
    words = re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())
    freq: dict[str, int] = {}
    for w in words:
        if w not in stop_words:
            freq[w] = freq.get(w, 0) + 1
    sorted_words = sorted(freq, key=freq.get, reverse=True)  # type: ignore[arg-type]
    return sorted_words[:top_n]


def slugify(text: str, max_length: int = 60) -> str:
    """Convert *text* to a URL-friendly slug."""
    slug = re.sub(r"[^\w\s-]", "", text.lower())
    slug = re.sub(r"[\s_]+", "-", slug).strip("-")
    return slug[:max_length]


def strip_markdown(text: str) -> str:
    """Remove common Markdown formatting from *text*."""
    text = re.sub(r"#{1,6}\s*", "", text)          # headings
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)  # bold
    text = re.sub(r"\*([^*]+)\*", r"\1", text)      # italic
    text = re.sub(r"`([^`]+)`", r"\1", text)        # inline code
    text = re.sub(r"^\s*[\-\*]\s", "", text, flags=re.MULTILINE)  # list markers
    return text.strip()
