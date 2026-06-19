"""
semantic/sentence_ranker.py
Scores and ranks sentences using:
  - Centroid similarity  : cosine distance to document centroid (informativeness)
  - Position bias        : earlier sentences get a slight boost
  - Length penalty       : penalise very short or very long sentences

The top-K sentences are returned in their ORIGINAL ORDER (not score order)
so the extractive passage reads naturally when fed to BART.
"""

import logging
from typing import Optional
from collections import Counter
import numpy as np
import re
from semantic.embedding_service import EmbeddingService

import config

logger = logging.getLogger(__name__)

POSITION_DECAY = config.RANKER_POSITION_DECAY
MIN_WORDS      = 5      # sentences shorter than this get a hard penalty

# ── Sentence quality patterns (module-level to avoid recompilation) ───────────
# Hard-reject: fragment that starts with a preposition/conjunction (no subject)
_FRAG_START = re.compile(
    r'^(between|among|from|to|with|without|including|'
    r'such as|as well as|along with|in addition|'
    r'as a result|in contrast|on the other hand)\b',
    re.IGNORECASE,
)
# Hard-reject: single uppercase letter + period opener (leftover abbreviation half)
_ABBREV_OPENER = re.compile(r'^[A-Z]\. ')
# Hard-reject: boilerplate navigation / footer lines
_BOILERPLATE_START = re.compile(
    r'^\s*(back to|return to|home|about|contact|privacy|terms|cookies|sitemap|search|menu)\b',
    re.IGNORECASE,
)


class SentenceRanker:
    """Ranks sentences using RoBERTa embeddings + scoring heuristics."""

    def __init__(self, embedding_service: EmbeddingService):
        self.embedder = embedding_service

    # ── Public API ────────────────────────────────────────────────────────────
    def rank_and_select(
        self,
        sentences: list[str],
        target_words: int = 200,
        top_k: Optional[int] = None,
    ) -> list[str]:
        """
        Score all sentences, return the top-scoring sentences in document order until target word count is reached.

        Args:
            sentences    : List of sentence strings.
            target_words : The target word count budget.

        Returns:
            Ordered list of top sentence strings.
        """
        if not sentences:
            return []

        # Precompute heading context on original sentence stream (for section weighting)
        heading_labels = [
            self._heading_label(s) if self._is_heading_like(s) else ""
            for s in sentences
        ]

        # Sentence-quality filtering + downweighting before embeddings/scoring
        filtered_sentences = []
        kept_original_indices = []
        quality_weights = []
        removed_count = 0
        for i, s in enumerate(sentences):
            keep, weight = self._sentence_quality(s)
            if not keep:
                removed_count += 1
                continue
            filtered_sentences.append(s)
            kept_original_indices.append(i)
            quality_weights.append(weight)

        if removed_count:
            logger.info("Sentence quality filter removed %d/%d sentences before ranking.", removed_count, len(sentences))

        if not filtered_sentences:
            return []

        embeddings = self.embedder.embed(filtered_sentences)
        base_scores = self._score(filtered_sentences, embeddings)
        # Normalize embeddings for MMR
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-9
        normed = embeddings / norms
        # Apply sentence-quality weights + section-priority weights
        sec_weights = np.array(
            [self._section_weight(sentences, kept_original_indices[j], heading_labels) for j in range(len(filtered_sentences))],
            dtype=float,
        )
        scores = base_scores * np.array(quality_weights, dtype=float) * sec_weights

        selected_indices = []
        current_words = 0

        # Tighten selection for technical docs: never keep "most" of the sentences
        n_total = len(filtered_sentences)
        if n_total >= 15:
            max_frac = config.RANKER_MAX_SENT_FRAC_TECHDOC
            max_abs = config.RANKER_MAX_SENTENCES_TECHDOC
            max_to_select = max(10, min(int(n_total * max_frac), max_abs))
        else:
            max_to_select = n_total

        if top_k is not None:
            # Legacy mode: caller wants up to N sentences regardless of target word budget.
            # Also keep the existing long-doc caps from configuration.
            max_to_select = min(max_to_select, max(1, int(top_k)))

        # If top_k is provided (legacy evaluation mode), we stop by sentence count
        # rather than word-budget. This keeps evaluation/evaluate.py behavior stable.
        target_words_budget = float("inf") if top_k is not None else target_words

        lambda_mmr = getattr(config, "RANKER_MMR_LAMBDA", 0.7)
        unselected_indices = list(range(n_total))

        # Iterative HLA-MMR Selection Loop
        while len(selected_indices) < max_to_select and unselected_indices:
            # Vectorized MMR scoring
            unsel_embs = normed[unselected_indices]
            if selected_indices:
                sel_embs = normed[selected_indices]
                sims = unsel_embs @ sel_embs.T          # (U, K)
                max_sims = sims.max(axis=1)             # (U,)
            else:
                max_sims = np.zeros(len(unselected_indices))

            mmr_scores = lambda_mmr * scores[unselected_indices] - (1.0 - lambda_mmr) * max_sims
            # Pick the sentence with the highest MMR score
            best_idx_in_unselected = int(np.argmax(mmr_scores))
            best_idx = unselected_indices[best_idx_in_unselected]
            
            sentence_words = len(filtered_sentences[best_idx].split())
            
            # If adding this sentence pushes us way over budget, we stop, unless we haven't selected anything yet.
            # Relaxed: Allow going 10% over budget to avoid BART hallucination
            if current_words + sentence_words > target_words_budget * 1.10 and current_words > 0:
                break
                
            selected_indices.append(best_idx)
            unselected_indices.pop(best_idx_in_unselected)
            current_words += sentence_words

        # Re-sort the final selected indices back to original document order
        top_indices_filtered = sorted(selected_indices)
        original_indices = [kept_original_indices[i] for i in top_indices_filtered]
        selected = [sentences[i] for i in original_indices]

        logger.info(
            "Selected %d sentences (%d words). Cap=%d. Target Words=%d.",
            len(selected),
            current_words,
            max_to_select,
            target_words,
        )
        return selected

    # ── Private helpers ───────────────────────────────────────────────────────
    def _score(self, sentences: list[str], embeddings: np.ndarray) -> np.ndarray:
        """Composite scoring: centroid similarity + position bias + length score."""
        n = len(sentences)

        # 1. Centroid similarity (document-level) — normalized embeddings
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-9
        normed = embeddings / norms
        centroid = normed.mean(axis=0)
        centroid /= np.linalg.norm(centroid) + 1e-9
        centroid_sim = (normed @ centroid).flatten()         # (N,)

        # 2. Position bias — sentences near the start tend to be informative
        position_scores = np.array([POSITION_DECAY ** i for i in range(n)])

        # 3. Length score — prefer medium-length sentences
        lengths = np.array([len(s.split()) for s in sentences])
        length_scores = np.where(lengths < MIN_WORDS, 0.0,
                        np.where(lengths < 10,        0.5,
                        np.where(lengths < 40,        1.0, 0.8)))

        # 4. Keyword density proxy (using term frequencies)
        all_words = [w for s in sentences for w in re.findall(r'\b\w+\b', s.lower())]
        word_counts = Counter(all_words)
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'it', 'that', 'this', 'as'}
        
        keyword_scores = []
        for s in sentences:
            words = re.findall(r'\b\w+\b', s.lower())
            score = sum(word_counts[w] for w in words if w not in stop_words)
            keyword_scores.append(score / (len(words) + 1e-9))
        keyword_scores = np.array(keyword_scores)
        if keyword_scores.max() > 0: keyword_scores /= keyword_scores.max()

        # 5. Structural weight (e.g. quotes or title-cased words imply importance)
        struct_scores = []
        for s in sentences:
            score = 0.5 if s.strip().startswith('"') or s.strip().startswith("'") else 0.0
            score -= 0.5 if s.isupper() else 0.0
            title_words = sum(1 for w in s.split() if w.istitle())
            score += min(0.5, title_words / (len(s.split()) + 1e-9))
            struct_scores.append(score)
        struct_scores = np.array(struct_scores)

        # Weighted combination
        scores = getattr(config, 'RANKER_CENTROID_WEIGHT', 0.5) * centroid_sim + \
                 getattr(config, 'RANKER_POSITION_WEIGHT', 0.15) * position_scores + \
                 getattr(config, 'RANKER_LENGTH_WEIGHT', 0.1) * length_scores + \
                 getattr(config, 'RANKER_KEYWORD_WEIGHT', 0.15) * keyword_scores + \
                 getattr(config, 'RANKER_STRUCTURAL_WEIGHT', 0.1) * struct_scores
        return scores

    def _sentence_quality(self, s: str) -> tuple[bool, float]:
        """
        Returns (keep, weight). Weight is a soft downweight (0..1.0) applied to score.
        Acts as a second line of defence after _join_ocr_lines in the cleaner.
        """
        t = (s or "").strip()
        if not t:
            return False, 0.0

        word_tokens = re.findall(r"\b\w+\b", t)

        # ── Hard reject: too short to carry meaning ───────────────────────────
        # Relaxed from 6 → 4 words: Allow shorter sentences (e.g., "Results show improvement.")
        if len(word_tokens) < 4:
            return False, 0.0
        if len(t) < 20:  # Relaxed from 30 → 20 characters
            return False, 0.0

        # ── Hard reject: single-letter abbreviation fragment opener ───────────
        if _ABBREV_OPENER.match(t):
            return False, 0.0

        # ── Hard reject: starts with a continuation word (no subject) ─────────
        if _FRAG_START.match(t):
            return False, 0.0

        # ── Symbol density check ──────────────────────────────────────────────
        non_alnum = sum(1 for ch in t if not ch.isalnum() and not ch.isspace())
        non_alnum_ratio = non_alnum / max(1, len(t))
        if non_alnum_ratio > 0.50:  # Relaxed from 0.40 → 0.50 (allow more symbols)
            return False, 0.0
        weight = 1.0
        if non_alnum_ratio > 0.30:  # Relaxed from 0.22 → 0.30
            weight *= 0.7  # Relaxed from 0.6 → 0.7

        # ── Boilerplate / navigation ──────────────────────────────────────────
        lower = t.lower()
        if _BOILERPLATE_START.match(t):
            return False, 0.0
        # Relaxed: Allow URLs in sentences (they might be important references)
        # Removed: if lower.count("http") >= 1 or lower.count("www.") >= 1:
        if lower.count("copyright") >= 1 or lower.count("all rights reserved") >= 1:
            return False, 0.0
        if lower.count("disclaimer") >= 1 or lower.count("privacy policy") >= 1:
            return False, 0.0

        # ── Soft downweights ──────────────────────────────────────────────────
        if re.match(r"^\s*(team|members?|contributors?|faculty|department|institution|university|college)\b", lower):
            weight *= 0.65  # Relaxed from 0.55 → 0.65 (less aggressive penalty)

        # Relaxed code detection: Don't hard-reject code, just downweight it
        if re.search(r"[{}\[\]<>]|(::)|\b(def|class|import|from|return|#include|public|private|static|void)\b", t):
            if non_alnum_ratio > 0.30 and len(re.findall(r"\b[a-zA-Z]{3,}\b", t)) < 3:
                return False, 0.0
            weight *= 0.8  # Relaxed from 0.7 → 0.8 (less aggressive penalty)

        # Penalise excessive word repetition
        uniq = len(set(w.lower() for w in word_tokens))
        if uniq / max(1, len(word_tokens)) < 0.50:  # Relaxed from 0.55 → 0.50
            weight *= 0.75  # Relaxed from 0.65 → 0.75

        return True, float(max(0.1, min(1.0, weight)))

    def _is_heading_like(self, s: str) -> bool:
        t = (s or "").strip()
        if not t:
            return False
        if t.endswith(":") and len(t.split()) <= 8 and len(t) <= 70:
            return True
        if t.isupper() and 2 <= len(t.split()) <= 8 and len(t) <= 70:
            return True
        # Title-ish short lines
        if len(t.split()) <= 8 and len(t) <= 70:
            titleish = sum(1 for w in t.split() if w[:1].isupper()) / max(1, len(t.split()))
            if titleish >= 0.7 and t[-1] not in ".!?":
                return True
        return False

    def _heading_label(self, s: str) -> str:
        t = (s or "").strip().lower()
        if not t:
            return ""
        # prioritize technical sections
        if "problem" in t:
            return "problem"
        if "method" in t or "methodology" in t or "proposed system" in t:
            return "method"
        if "result" in t or "performance" in t or "evaluation" in t:
            return "results"
        if "observation" in t:
            return "observations"
        if "next step" in t or "future work" in t:
            return "next_steps"
        # de-prioritize metadata-ish sections
        if "team" in t or "member" in t:
            return "team"
        if "faculty" in t or "department" in t or "institution" in t or "university" in t or "college" in t:
            return "institution"
        return ""

    def _section_weight(self, sentences: list[str], original_idx: int, heading_labels: list[str]) -> float:
        """
        Boost sentences under high-value headings; downweight under low-value headings.
        Uses nearest previous heading within a small window.
        """
        # find nearest previous heading within last 3 sentences
        label = ""
        for j in range(original_idx - 1, max(-1, original_idx - 4), -1):
            if heading_labels[j]:
                label = heading_labels[j]
                break
        if not label:
            return 1.0

        boosts = {
            "problem": 1.25,
            "method": 1.25,
            "results": 1.25,
            "observations": 1.15,
            "next_steps": 1.2,
        }
        penalties = {
            "team": 0.7,
            "institution": 0.8,
        }
        if label in boosts:
            return boosts[label]
        if label in penalties:
            return penalties[label]
        return 1.0

    # Store normed embeddings for MMR loop
    _normed_embeddings = None