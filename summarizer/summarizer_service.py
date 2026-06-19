"""
summarizer/summarizer_service.py
Abstractive summarization using facebook/bart-large-cnn.
Accepts extractive pre-selected text and generates a fluent summary.
"""

import logging
import re
import torch
import pysbd
import time
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

import config

logger = logging.getLogger(__name__)

MODEL_NAME       = config.SUMMARIZER_MODEL
MAX_INPUT_TOKENS = config.BART_MAX_INPUT_TOKENS
TOKENS_PER_WORD  = getattr(config, "BART_TOKENS_PER_WORD", 1.3)

# Per-mode length constraints (in output tokens, ~0.75 tokens per word)
# Increased limits to allow longer, more detailed summaries
LENGTH_CONSTRAINTS = {
    "short":  {"min": 50,  "max": 150},
    "medium": {"min": 100, "max": 280},
    "long":   {"min": 150, "max": 450},
}


def _clean_generated(text: str) -> str:
    """Remove stray bullet markers that BART occasionally copies from input."""
    text = re.sub(r"(?m)^\s*[-•]\s+", " ", text)      # stray bullet markers
    text = re.sub(r" {2,}", " ", text)                 # double spaces
    return text.strip()



from semantic.embedding_service import EmbeddingService
import numpy as np

class SummarizerService:
    """Wrapper around BART for abstractive summarization, with optional MMR reranking for novelty."""

    def __init__(self, model_name: str = MODEL_NAME, max_retries: int = 3):
        logger.info("Loading summarization model: %s …", model_name)
        
        self.tokenizer = None
        self.model = None
        self._segmenter = None
        self._pipe = None
        
        try:
            # Retry logic for large model downloads
            for attempt in range(1, max_retries + 1):
                try:
                    self.tokenizer = AutoTokenizer.from_pretrained(model_name, resume_download=True)
                    self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name, resume_download=True)
                    break
                except Exception as e:
                    if attempt < max_retries:
                        wait_time = attempt * 5
                        logger.warning("Model download failed (attempt %d/%d): %s. Retrying in %ds...", 
                                     attempt, max_retries, str(e)[:100], wait_time)
                        time.sleep(wait_time)
                    else:
                        logger.error("Failed to load model after %d attempts", max_retries)
                        raise RuntimeError(f"Could not load {model_name} after {max_retries} attempts. "
                                         f"Check your internet connection or try downloading manually.") from e
            
            self._segmenter = pysbd.Segmenter(language="en", clean=False)

            if torch.cuda.is_available():
                device_id = 0
                self.model = self.model.to(f"cuda:{device_id}")
                torch.cuda.empty_cache()
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
            logger.info("Summarization model ready. (Device: %s)", device_label)
        except Exception as e:
            # Clean up any partially initialized resources
            self._cleanup()
            raise

    # ── Public API ────────────────────────────────────────────────────────────
    def summarize(
        self,
        text: str,
        length: str = "medium",
        target_word_count: int = 0,
        novelty_rerank: bool = False,
        num_candidates: int = 5,
        mmr_lambda: float = 0.7,
    ) -> tuple[str, dict]:
        """
        Generate an abstractive summary of the given text.

        Args:
            text             : Pre-selected extractive passage (plain prose).
            length           : 'short' | 'medium' | 'long'.
            target_word_count: Desired output word count; 0 = use mode defaults.

        Returns:
            (summary_str, metadata_dict)
        """

        meta = {"chunk_count": 1, "avg_chunk_tokens": 0, "multipass": False, "novelty": novelty_rerank}

        if not text.strip():
            return "", meta

        if novelty_rerank:
            return self.summarize_with_novelty(
                text, length, target_word_count, num_candidates, mmr_lambda, meta
            )
    def summarize_with_novelty(
        self,
        text: str,
        length: str = "medium",
        target_word_count: int = 0,
        num_candidates: int = 5,
        mmr_lambda: float = 0.7,
        meta: dict = None,
    ) -> tuple[str, dict]:
        """
        Generate multiple BART summaries and rerank with MMR for novelty/diversity.
        """
        if meta is None:
            meta = {}
        # Use the same chunking logic as before (single chunk for simplicity)
        sentences = self._segmenter.segment(text)
        chunk_text = " ".join(s.strip() for s in sentences)
        chunk_token_count = len(self.tokenizer.encode(chunk_text, add_special_tokens=False))
        constraints = LENGTH_CONSTRAINTS.get(length, LENGTH_CONSTRAINTS["medium"])
        if target_word_count > 0:
            per_chunk_words = target_word_count
            per_chunk_tokens = int(per_chunk_words * TOKENS_PER_WORD)
            min_gen_length = max(constraints["min"], int(per_chunk_tokens * 0.80))
            max_gen_length = min(constraints["max"], int(per_chunk_tokens * 1.20))
            max_gen_length = max(max_gen_length, min_gen_length + 20)
            max_gen_length = min(max_gen_length, int(chunk_token_count * 0.90) + 30)
        else:
            max_gen_length = min(constraints["max"], chunk_token_count + 40)
            min_gen_length = min(constraints["min"], max_gen_length - 20)
        min_gen_length = max(10, min_gen_length)
        max_gen_length = max(min_gen_length + 10, max_gen_length)

        # Generate multiple candidates using beam search
        try:
            outputs = self._pipe(
                chunk_text,
                min_length=min_gen_length,
                max_length=max_gen_length,
                num_beams=max(num_candidates, 4),
                num_return_sequences=num_candidates,
                do_sample=False,
                early_stopping=True,
                no_repeat_ngram_size=3,
                repetition_penalty=config.BART_REPETITION_PENALTY,
                length_penalty=config.BART_LENGTH_PENALTY,
                truncation=True,
            )
        except Exception as e:
            # Fallback to single summary if error
            logger.warning(f"Novelty rerank failed, falling back: {e}")
            return self.summarize(text, length, target_word_count, novelty_rerank=False)

        candidates = [_clean_generated(o["summary_text"]) for o in outputs]
        # Remove duplicates
        candidates = list(dict.fromkeys(candidates))
        if len(candidates) == 1:
            return candidates[0], meta

        # Compute embeddings for candidates and source
        embedder = EmbeddingService()
        cand_embeds = embedder.embed(candidates)
        src_embed = embedder.embed([chunk_text])[0]

        # MMR reranking
        selected = []
        selected_idx = []
        candidate_indices = list(range(len(candidates)))
        # Start with the most relevant (highest similarity to source)
        sims = np.dot(cand_embeds, src_embed)
        first_idx = int(np.argmax(sims))
        selected.append(candidates[first_idx])
        selected_idx.append(first_idx)
        candidate_indices.remove(first_idx)

        # For the rest, select by MMR
        for _ in range(1, min(3, len(candidates))):
            mmr_scores = []
            for idx in candidate_indices:
                relevance = np.dot(cand_embeds[idx], src_embed)
                diversity = max(
                    [np.dot(cand_embeds[idx], cand_embeds[j]) for j in selected_idx] or [0]
                )
                mmr = mmr_lambda * relevance - (1 - mmr_lambda) * diversity
                mmr_scores.append(mmr)
            next_idx = candidate_indices[int(np.argmax(mmr_scores))]
            selected.append(candidates[next_idx])
            selected_idx.append(next_idx)
            candidate_indices.remove(next_idx)

        # Join selected summaries (or just use the top one)
        final_summary = selected[0]
        meta["novelty_candidates"] = candidates
        meta["novelty_selected"] = selected
        return final_summary, meta

        # ── 1. Sentence-aware chunking ────────────────────────────────────────
        sentences = self._segmenter.segment(text)

        chunks: list[list[str]] = []
        current_chunk: list[str] = []
        current_length = 0

        for sentence in sentences:
            tokens = self.tokenizer.encode(sentence, add_special_tokens=False)
            token_count = len(tokens)
            if current_length + token_count > MAX_INPUT_TOKENS and current_chunk:
                chunks.append(current_chunk)
                current_chunk = [sentence]
                current_length = token_count
            else:
                current_chunk.append(sentence)
                current_length += token_count

        if current_chunk:
            chunks.append(current_chunk)

        n_chunks = len(chunks)
        meta["chunk_count"] = n_chunks

        if n_chunks > 1:
            logger.info("Input split into %d sentence-aware chunks.", n_chunks)
        else:
            logger.info("Input token size is within limits. Passing intact.")

        if n_chunks > 0:
            total_tokens = sum(
                len(self.tokenizer.encode(" ".join(c), add_special_tokens=False))
                for c in chunks
            )
            meta["avg_chunk_tokens"] = round(total_tokens / n_chunks)

        # ── 2. Per-chunk generation ───────────────────────────────────────────
        constraints = LENGTH_CONSTRAINTS.get(length, LENGTH_CONSTRAINTS["medium"])
        chunk_summaries: list[str] = []

        for idx, chunk_list in enumerate(chunks):
            # Pass prose directly to BART — bullet wrapping causes fragment outputs.
            chunk_text = " ".join(s.strip() for s in chunk_list)
            chunk_token_count = len(
                self.tokenizer.encode(chunk_text, add_special_tokens=False)
            )

            if target_word_count > 0:
                # Distribute target word budget across chunks, convert words→tokens
                per_chunk_words = target_word_count / n_chunks
                per_chunk_tokens = int(per_chunk_words * TOKENS_PER_WORD)

                # Keep generation tightly bounded: min = 80% of target, max = 120%
                min_gen_length = max(constraints["min"], int(per_chunk_tokens * 0.80))
                max_gen_length = min(constraints["max"], int(per_chunk_tokens * 1.20))

                # Safety: max must always be > min, and never exceed input length × 0.9
                max_gen_length = max(max_gen_length, min_gen_length + 20)
                max_gen_length = min(max_gen_length, int(chunk_token_count * 0.90) + 30)
            else:
                # Mode-only defaults: cap max at input size so BART doesn't repeat
                max_gen_length = min(constraints["max"], chunk_token_count + 40)
                min_gen_length = min(constraints["min"], max_gen_length - 20)

            # Ensure hard lower bound
            min_gen_length = max(10, min_gen_length)
            max_gen_length = max(min_gen_length + 10, max_gen_length)

            if idx == 0:
                logger.info(
                    "BART gen params — min_length: %d  max_length: %d  chunk_tokens: %d",
                    min_gen_length, max_gen_length, chunk_token_count,
                )

            try:
                result = self._pipe(
                    chunk_text,
                    min_length=min_gen_length,
                    max_length=max_gen_length,
                    # Beam search (do_sample=False) produces the most coherent prose
                    do_sample=False,
                    num_beams=4,
                    early_stopping=True,
                    no_repeat_ngram_size=3,  # Back to 3 for better fluency
                    repetition_penalty=config.BART_REPETITION_PENALTY,  # Use config value
                    length_penalty=config.BART_LENGTH_PENALTY,
                    truncation=True,  # Ensure input is properly truncated
                )
            except RuntimeError as e:
                if "CUDA" in str(e):
                    logger.error("CUDA error during generation, clearing cache and retrying: %s", str(e)[:200])
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    # Retry with safer parameters
                    result = self._pipe(
                        chunk_text,
                        min_length=min_gen_length,
                        max_length=max_gen_length,
                        do_sample=False,
                        num_beams=2,
                        early_stopping=True,
                        no_repeat_ngram_size=3,
                        repetition_penalty=config.BART_REPETITION_PENALTY,
                        length_penalty=config.BART_LENGTH_PENALTY,
                        truncation=True,
                    )
                else:
                    raise
            chunk_summary = _clean_generated(result[0]["summary_text"])
            chunk_summaries.append(chunk_summary)
            logger.info(
                "Chunk %d/%d generated: %d chars.",
                idx + 1, n_chunks, len(chunk_summary),
            )

        # ── 3. Merge chunks ───────────────────────────────────────────────────
        # Join ensuring each chunk ends with a sentence-terminal punctuation mark
        merged_parts = []
        for s in chunk_summaries:
            s = s.strip()
            if s and s[-1] not in ".!?":
                s += "."
            merged_parts.append(s)
        merged_summary = " ".join(merged_parts)

        # ── 4. Optional Pass-2 compression ───────────────────────────────────
        # Only trigger when multi-chunk AND output is significantly over target
        effective_target = target_word_count if target_word_count > 0 else int(constraints["max"] / TOKENS_PER_WORD)
        if n_chunks > 1 and len(merged_summary.split()) > int(effective_target * 1.25):
            logger.info(
                "Pass 2: merging %d words → ~%d words.", len(merged_summary.split()), effective_target
            )
            meta["multipass"] = True
            final_tokens = int(effective_target * TOKENS_PER_WORD)
            try:
                pass2_result = self._pipe(
                    merged_summary,
                    min_length=max(10, int(final_tokens * 0.85)),
                    max_length=max(40, int(final_tokens * 1.15)),
                    do_sample=False,
                    num_beams=4,
                    early_stopping=True,
                    no_repeat_ngram_size=3,
                    repetition_penalty=config.BART_REPETITION_PENALTY,
                    length_penalty=config.BART_LENGTH_PENALTY,
                    truncation=True,
                )
            except RuntimeError as e:
                if "CUDA" in str(e):
                    logger.warning("CUDA error in pass 2, skipping compression: %s", str(e)[:100])
                    return merged_summary, meta
                raise
            final_summary = _clean_generated(pass2_result[0]["summary_text"])
            logger.info(
                "Pass 2 complete: %d → %d words.",
                len(merged_summary.split()), len(final_summary.split()),
            )
            return final_summary, meta

        logger.info(
            "Final abstractive summary: %d chars / %d words.",
            len(merged_summary), len(merged_summary.split()),
        )
        return merged_summary, meta
    
    def __enter__(self):
        return self

    def __exit__(self, *_):
        self._cleanup()
    
    def _cleanup(self):
        """Clean up resources to prevent memory leaks."""
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
            if torch is not None and torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
