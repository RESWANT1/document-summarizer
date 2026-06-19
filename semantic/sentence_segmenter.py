"""
semantic/sentence_segmenter.py
Splits cleaned text into individual sentences using NLTK's Punkt tokenizer.
Filters out very short or noisy fragments.
"""

import logging
import pysbd
import config

logger = logging.getLogger(__name__)

# Initialize PySBD Segmenter (SOTA for academic/technical texts)
segmenter = pysbd.Segmenter(language="en", clean=False)

MIN_SENTENCE_LENGTH = config.MIN_SENTENCE_LENGTH
MAX_SENTENCE_LENGTH = config.MAX_SENTENCE_LENGTH


def segment_sentences(text: str) -> list[str]:
    """
    Tokenise text into sentences.

    Returns:
        List of sentence strings, filtered and cleaned.
    """
    if not text.strip():
        return []

    raw_sentences = segmenter.segment(text)
    sentences = []

    for sent in raw_sentences:
        sent = sent.strip()
        if len(sent) < MIN_SENTENCE_LENGTH:
            continue
        # If absurdly long, split on newlines as a fallback
        if len(sent) > MAX_SENTENCE_LENGTH:
            sub_parts = [s.strip() for s in sent.split("\n") if len(s.strip()) >= MIN_SENTENCE_LENGTH]
            sentences.extend(sub_parts)
        else:
            sentences.append(sent)

    logger.info("Segmented into %d sentences.", len(sentences))
    return sentences