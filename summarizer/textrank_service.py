"""
summarizer/textrank_service.py
Extractive summarization using TextRank algorithm.
"""

import logging
from typing import Tuple, Dict
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pysbd

logger = logging.getLogger(__name__)


class TextRankService:
    """TextRank extractive summarization."""

    def __init__(self):
        self._segmenter = pysbd.Segmenter(language="en", clean=False)
        logger.info("TextRank service initialized")

    def summarize(
        self,
        text: str,
        length: str = "medium",
        target_word_count: int = 0,
        compression_ratio: float = 0.25,
    ) -> Tuple[str, Dict]:
        """
        Generate extractive summary using TextRank.

        Args:
            text: Input text to summarize
            length: 'short' | 'medium' | 'long'
            target_word_count: Desired output word count; 0 = use compression_ratio
            compression_ratio: Fraction of sentences to extract (0.0-1.0)

        Returns:
            (summary_str, metadata_dict)
        """
        meta = {"sentences_total": 0, "sentences_selected": 0, "compression_ratio": 0}

        if not text.strip():
            return "", meta

        # Segment into sentences
        sentences = self._segmenter.segment(text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not sentences:
            return "", meta

        meta["sentences_total"] = len(sentences)

        # Determine number of sentences to extract
        if target_word_count > 0:
            # Estimate: ~4.5 words per sentence on average
            target_sentences = max(1, int(target_word_count / 4.5))
            num_sentences = min(target_sentences, len(sentences))
        else:
            num_sentences = max(1, int(len(sentences) * compression_ratio))

        if num_sentences >= len(sentences):
            return text, {**meta, "sentences_selected": len(sentences), "compression_ratio": 1.0}

        # Build TF-IDF matrix
        try:
            vectorizer = TfidfVectorizer(
                max_features=100,
                stop_words="english",
                lowercase=True,
                min_df=1,
            )
            tfidf_matrix = vectorizer.fit_transform(sentences)
        except Exception as e:
            logger.warning(f"TF-IDF vectorization failed: {e}. Returning first sentences.")
            summary = " ".join(sentences[:num_sentences])
            return summary, {**meta, "sentences_selected": num_sentences, "compression_ratio": num_sentences / len(sentences)}

        # Compute similarity matrix
        similarity_matrix = cosine_similarity(tfidf_matrix)
        np.fill_diagonal(similarity_matrix, 0)

        # TextRank scoring: sum of similarities
        scores = similarity_matrix.sum(axis=1)

        # Select top-scoring sentences in original order
        top_indices = np.argsort(scores)[-num_sentences:]
        top_indices = sorted(top_indices)

        summary = " ".join(sentences[i] for i in top_indices)

        meta["sentences_selected"] = num_sentences
        meta["compression_ratio"] = num_sentences / len(sentences)

        logger.info(
            f"TextRank: {len(sentences)} sentences → {num_sentences} selected "
            f"({meta['compression_ratio']:.1%} compression)"
        )

        return summary, meta

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass
