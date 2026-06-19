"""
summarizer/hla_mmr_service.py
Hybrid Lexical-Abstractive with Maximal Marginal Relevance (HLA-MMR).
Combines extractive ranking with abstractive generation for better quality.
"""

import logging
from typing import Tuple, Dict
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pysbd
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

import config

logger = logging.getLogger(__name__)

MODEL_NAME = config.SUMMARIZER_MODEL
MAX_INPUT_TOKENS = config.BART_MAX_INPUT_TOKENS


class HLAMMRService:
    """Hybrid Lexical-Abstractive with Maximal Marginal Relevance."""

    def __init__(self):
        self._segmenter = pysbd.Segmenter(language="en", clean=False)
        
        # Initialize BART for abstractive generation
        logger.info("Loading BART model for HLA-MMR...")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, resume_download=True)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, resume_download=True)
        
        if torch.cuda.is_available():
            self.model = self.model.to("cuda:0")
            device_id = 0
            device_label = "GPU"
        else:
            device_id = -1
            device_label = "CPU"
            logger.warning("CUDA not available — running on CPU. Expect slower inference.")

        self._pipe = pipeline(
            "summarization",
            model=self.model,
            tokenizer=self.tokenizer,
            device=device_id,
        )
        logger.info(f"HLA-MMR service initialized (Device: {device_label})")

    def summarize(
        self,
        text: str,
        length: str = "medium",
        target_word_count: int = 0,
        extraction_ratio: float = 0.5,
        mmr_lambda: float = 0.6,
        centroid_weight: float = 0.6,
        position_weight: float = 0.1,
    ) -> Tuple[str, Dict]:
        """
        Generate hybrid summary using HLA-MMR.

        Args:
            text: Input text to summarize
            length: 'short' | 'medium' | 'long'
            target_word_count: Desired output word count
            extraction_ratio: Fraction of sentences to extract (0.0-1.0)
            mmr_lambda: Balance between saliency (1.0) and diversity (0.0)
            centroid_weight: Weight for centroid similarity in MMR
            position_weight: Weight for position bias (earlier sentences scored higher)

        Returns:
            (summary_str, metadata_dict)
        """
        meta = {
            "sentences_total": 0,
            "sentences_extracted": 0,
            "extraction_ratio": 0,
            "method": "HLA-MMR",
        }

        if not text.strip():
            return "", meta

        # Step 1: Segment into sentences
        sentences = self._segmenter.segment(text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not sentences:
            return "", meta

        meta["sentences_total"] = len(sentences)

        # Step 2: Determine extraction count
        if target_word_count > 0:
            target_sentences = max(1, int(target_word_count / 4.5))
            num_extract = min(target_sentences, len(sentences))
        else:
            # Extract 50% of sentences for better context preservation
            num_extract = max(3, int(len(sentences) * 0.5))

        if num_extract >= len(sentences):
            return text, {**meta, "sentences_extracted": len(sentences), "extraction_ratio": 1.0}

        # Step 3: Build TF-IDF matrix for MMR
        try:
            vectorizer = TfidfVectorizer(
                max_features=100,
                stop_words="english",
                lowercase=True,
                min_df=1,
            )
            tfidf_matrix = vectorizer.fit_transform(sentences)
        except Exception as e:
            logger.warning(f"TF-IDF vectorization failed: {e}. Using position-based extraction.")
            # Fallback: select sentences by position
            indices = np.linspace(0, len(sentences) - 1, num_extract, dtype=int)
            extracted = " ".join(sentences[i] for i in sorted(indices))
            return extracted, {**meta, "sentences_extracted": num_extract, "extraction_ratio": num_extract / len(sentences)}

        # Step 4: Compute centroid (average TF-IDF vector)
        centroid = tfidf_matrix.mean(axis=0)
        # Convert to dense array properly
        if hasattr(centroid, 'A1'):
            centroid_dense = centroid.A1
        elif hasattr(centroid, 'toarray'):
            centroid_dense = centroid.toarray().flatten()
        else:
            centroid_dense = np.asarray(centroid).flatten()

        # Step 5: MMR-based sentence selection
        similarity_matrix = cosine_similarity(tfidf_matrix)
        np.fill_diagonal(similarity_matrix, 0)

        selected_indices = []
        remaining_indices = set(range(len(sentences)))

        for _ in range(num_extract):
            if not remaining_indices:
                break

            best_idx = None
            best_score = -np.inf

            for idx in remaining_indices:
                # Convert sparse matrix row to dense array
                tfidf_row = tfidf_matrix[idx]
                if hasattr(tfidf_row, 'A1'):
                    tfidf_dense = tfidf_row.A1
                elif hasattr(tfidf_row, 'toarray'):
                    tfidf_dense = tfidf_row.toarray().flatten()
                else:
                    tfidf_dense = np.asarray(tfidf_row).flatten()

                # Saliency: similarity to centroid
                saliency = np.dot(tfidf_dense, centroid_dense) / (
                    np.linalg.norm(tfidf_dense) * np.linalg.norm(centroid_dense) + 1e-10
                )

                # Diversity: average dissimilarity to already selected sentences
                if selected_indices:
                    diversity = 1 - similarity_matrix[idx, selected_indices].mean()
                else:
                    diversity = 1.0

                # Position bias: earlier sentences get higher scores
                position_score = 1.0 - (idx / len(sentences)) * position_weight

                # MMR score
                mmr_score = (mmr_lambda * saliency + (1 - mmr_lambda) * diversity) * position_score

                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = idx

            if best_idx is not None:
                selected_indices.append(best_idx)
                remaining_indices.remove(best_idx)

        # Step 6: Reorder selected sentences to maintain original order
        selected_indices = sorted(selected_indices)
        extracted_text = " ".join(sentences[i] for i in selected_indices)

        meta["sentences_extracted"] = len(selected_indices)
        meta["extraction_ratio"] = len(selected_indices) / len(sentences)

        logger.info(
            f"HLA-MMR extraction: {len(sentences)} sentences → {len(selected_indices)} selected "
            f"({meta['extraction_ratio']:.1%})"
        )

        # Step 7: Abstractive refinement with BART
        try:
            # Determine generation length
            length_constraints = {
                "short": {"min": 50, "max": 150},
                "medium": {"min": 100, "max": 280},
                "long": {"min": 150, "max": 450},
            }
            constraints = length_constraints.get(length, length_constraints["medium"])

            result = self._pipe(
                extracted_text,
                min_length=constraints["min"],
                max_length=constraints["max"],
                do_sample=False,
                num_beams=4,
                early_stopping=True,
                no_repeat_ngram_size=3,
                repetition_penalty=1.0,
                length_penalty=1.0,
                truncation=True,
            )
            abstractive_summary = result[0]["summary_text"].strip()
            logger.info(f"HLA-MMR abstractive refinement: {len(abstractive_summary.split())} words")
            return abstractive_summary, meta

        except Exception as e:
            logger.warning(f"BART refinement failed: {e}. Returning extracted text.")
            return extracted_text, meta

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self._cleanup()

    def _cleanup(self):
        """Clean up resources."""
        if self.model is not None:
            try:
                del self.model
            except Exception:
                pass
        if self.tokenizer is not None:
            try:
                del self.tokenizer
            except Exception:
                pass
        if self._pipe is not None:
            try:
                del self._pipe
            except Exception:
                pass
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
