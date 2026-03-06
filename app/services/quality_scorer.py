"""Unified content quality scoring.

Evaluates generated content across multiple dimensions and returns a
composite 0-100 score.  Works entirely offline using NLTK and heuristics --
no API calls required.

Dimensions
----------
- coherence    : sentence flow and structural quality
- relevance    : overlap between prompt keywords and output
- readability  : Flesch-Kincaid grade level mapped to a 0-100 score
- completeness : length adequacy and structural markers
- seo_score    : keyword density, heading presence, list usage
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Dict, List

# We use NLTK only when available; graceful fallback otherwise.
try:
    import nltk
    from nltk.tokenize import sent_tokenize, word_tokenize

    # Ensure required data is downloaded (no-op if already present)
    for _pkg in ("punkt", "punkt_tab", "averaged_perceptron_tagger",
                 "averaged_perceptron_tagger_eng", "stopwords"):
        try:
            nltk.data.find(f"tokenizers/{_pkg}" if "punkt" in _pkg else f"taggers/{_pkg}" if "tagger" in _pkg else f"corpora/{_pkg}")
        except LookupError:
            nltk.download(_pkg, quiet=True)

    _NLTK_AVAILABLE = True
except Exception:
    _NLTK_AVAILABLE = False


@dataclass
class QualityResult:
    """Quality assessment result."""
    overall: float
    coherence: float
    relevance: float
    readability: float
    completeness: float
    seo_score: float
    breakdown: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "overall": self.overall,
            "coherence": self.coherence,
            "relevance": self.relevance,
            "readability": self.readability,
            "completeness": self.completeness,
            "seo_score": self.seo_score,
        }


class QualityScorer:
    """Score content quality on a 0-100 scale."""

    # Weights for each dimension in the composite score
    WEIGHTS = {
        "coherence": 0.20,
        "relevance": 0.25,
        "readability": 0.20,
        "completeness": 0.15,
        "seo_score": 0.20,
    }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score(self, text: str, prompt: str = "") -> QualityResult:
        """Compute quality scores for *text* given the original *prompt*."""
        if not text or not text.strip():
            return QualityResult(0, 0, 0, 0, 0, 0)

        coherence = self._coherence(text)
        relevance = self._relevance(text, prompt)
        readability = self._readability(text)
        completeness = self._completeness(text)
        seo = self._seo_score(text, prompt)

        overall = (
            self.WEIGHTS["coherence"] * coherence
            + self.WEIGHTS["relevance"] * relevance
            + self.WEIGHTS["readability"] * readability
            + self.WEIGHTS["completeness"] * completeness
            + self.WEIGHTS["seo_score"] * seo
        )

        return QualityResult(
            overall=round(min(100, max(0, overall)), 2),
            coherence=round(coherence, 2),
            relevance=round(relevance, 2),
            readability=round(readability, 2),
            completeness=round(completeness, 2),
            seo_score=round(seo, 2),
            breakdown=dict(self.WEIGHTS),
        )

    # ------------------------------------------------------------------
    # Dimension scorers
    # ------------------------------------------------------------------

    def _coherence(self, text: str) -> float:
        """Measure structural coherence via sentence count, avg sentence
        length, and transition-word usage."""
        sentences = self._sentences(text)
        if len(sentences) < 1:
            return 20.0

        # Sentence-length variance (lower = more consistent = better)
        lengths = [len(s.split()) for s in sentences]
        avg_len = sum(lengths) / len(lengths)
        variance = sum((l - avg_len) ** 2 for l in lengths) / len(lengths)
        std = math.sqrt(variance)

        # Ideal avg sentence length: 10-25 words
        length_score = 100 if 10 <= avg_len <= 25 else max(0, 100 - abs(avg_len - 17) * 4)
        # Penalise high variance
        consistency_score = max(0, 100 - std * 3)

        # Transition words
        transitions = {
            "however", "therefore", "furthermore", "moreover", "additionally",
            "consequently", "meanwhile", "nevertheless", "instead", "thus",
            "indeed", "specifically", "in addition", "for example", "as a result",
            "on the other hand", "in contrast", "similarly",
        }
        lower = text.lower()
        transition_count = sum(1 for t in transitions if t in lower)
        transition_score = min(100, transition_count * 15)

        return (length_score * 0.4 + consistency_score * 0.3 + transition_score * 0.3)

    def _relevance(self, text: str, prompt: str) -> float:
        """Measure keyword overlap between prompt and generated text."""
        if not prompt:
            return 70.0  # No prompt to compare against; give baseline

        prompt_words = set(self._significant_words(prompt.lower()))
        text_words = set(self._significant_words(text.lower()))

        if not prompt_words:
            return 70.0

        overlap = prompt_words & text_words
        coverage = len(overlap) / len(prompt_words)
        return min(100, coverage * 100 + 10)  # +10 baseline

    def _readability(self, text: str) -> float:
        """Flesch-Kincaid readability mapped to a 0-100 quality score."""
        sentences = self._sentences(text)
        words = self._words(text)
        if not words or not sentences:
            return 30.0

        syllable_count = sum(self._count_syllables(w) for w in words)
        asl = len(words) / len(sentences)  # avg sentence length
        asw = syllable_count / len(words)  # avg syllables per word

        # Flesch Reading Ease
        fre = 206.835 - 1.015 * asl - 84.6 * asw
        # Clamp to 0-100
        fre = max(0, min(100, fre))

        # Very low (hard to read) or very high (too simple) are both suboptimal.
        # Map to quality: ideal FRE for general content is ~50-70.
        if 40 <= fre <= 80:
            return 85 + (fre - 40) * 0.375  # 85-100
        elif fre > 80:
            return max(60, 100 - (fre - 80) * 1.5)
        else:
            return max(30, fre + 20)

    def _completeness(self, text: str) -> float:
        """Heuristic completeness score based on length and structural markers."""
        word_count = len(self._words(text))
        sentence_count = len(self._sentences(text))

        # Length score: ramp up to ~300 words, plateau until 2000
        if word_count < 20:
            length_score = 20
        elif word_count < 100:
            length_score = 40 + (word_count - 20) * 0.5
        elif word_count < 300:
            length_score = 80 + (word_count - 100) * 0.1
        else:
            length_score = 100

        # Structure markers (headings, lists, paragraphs)
        has_headings = bool(re.search(r"^#{1,3}\s", text, re.MULTILINE))
        has_lists = bool(re.search(r"^[\-\*\d]\s", text, re.MULTILINE))
        has_paragraphs = text.count("\n\n") >= 2

        structure_score = 40
        if has_headings:
            structure_score += 25
        if has_lists:
            structure_score += 20
        if has_paragraphs:
            structure_score += 15

        return length_score * 0.5 + min(100, structure_score) * 0.5

    def _seo_score(self, text: str, prompt: str) -> float:
        """SEO-friendliness: keyword presence, heading structure, list usage,
        and keyword density."""
        score = 40.0  # baseline

        lower = text.lower()

        # Heading presence
        headings = re.findall(r"^#{1,3}\s+(.+)", text, re.MULTILINE)
        if headings:
            score += min(20, len(headings) * 5)

        # Bullet / numbered lists
        lists = re.findall(r"^[\-\*]\s", text, re.MULTILINE)
        if lists:
            score += min(10, len(lists) * 2)

        # Keyword density from prompt
        if prompt:
            keywords = self._significant_words(prompt.lower())
            word_count = max(1, len(self._words(text)))
            keyword_occurrences = sum(lower.count(kw) for kw in keywords)
            density = keyword_occurrences / word_count
            # Ideal density 1-3%
            if 0.01 <= density <= 0.03:
                score += 20
            elif density > 0:
                score += 10

        # Bold / emphasis usage
        bold_count = len(re.findall(r"\*\*[^*]+\*\*", text))
        if bold_count:
            score += min(10, bold_count * 3)

        return min(100, score)

    # ------------------------------------------------------------------
    # Text utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _sentences(text: str) -> List[str]:
        if _NLTK_AVAILABLE:
            try:
                return sent_tokenize(text)
            except Exception:
                pass
        # Fallback: split on period/question/exclamation followed by space or end
        return [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]

    @staticmethod
    def _words(text: str) -> List[str]:
        if _NLTK_AVAILABLE:
            try:
                return word_tokenize(text)
            except Exception:
                pass
        return re.findall(r'\b\w+\b', text)

    @staticmethod
    def _significant_words(text: str) -> List[str]:
        """Return words longer than 3 chars, excluding common stop words."""
        stop = {
            "the", "and", "for", "are", "but", "not", "you", "all", "can",
            "had", "her", "was", "one", "our", "out", "this", "that", "with",
            "have", "from", "they", "been", "said", "each", "which", "their",
            "will", "other", "about", "many", "then", "them", "these", "some",
            "would", "into", "more", "your", "what", "when", "make", "like",
        }
        words = re.findall(r'\b\w+\b', text)
        return [w for w in words if len(w) > 3 and w not in stop]

    @staticmethod
    def _count_syllables(word: str) -> int:
        """Approximate syllable count using a vowel-group heuristic."""
        word = word.lower().strip()
        if not word:
            return 1
        count = len(re.findall(r'[aeiouy]+', word))
        # Adjust for silent e
        if word.endswith('e') and count > 1:
            count -= 1
        return max(1, count)
