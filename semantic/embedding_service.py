"""
semantic/embedding_service.py
Generates dense sentence embeddings using a RoBERTa-based SentenceTransformer model.
Model: 'roberta-base-nli-mean-tokens' (sentence-transformers hub)
"""

import logging
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

import config

logger = logging.getLogger(__name__)

MODEL_NAME = config.EMBEDDING_MODEL


class EmbeddingService:
    """Singleton wrapper around a SentenceTransformer model."""

    def __init__(self, model_name: str = MODEL_NAME):
        logger.info("Loading embedding model: %s …", model_name)
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
        if device_str == "cpu":
            logger.warning("CUDA is not available. Using CPU (slower). For GPU support, install CUDA PyTorch: "
                          "`pip install torch --index-url https://download.pytorch.org/whl/cu121`")
        self.model = SentenceTransformer(model_name, device=device_str)
        logger.info("Embedding model ready. (Device: %s)", device_str.upper())

    def embed(self, sentences: list[str]) -> np.ndarray:
        """
        Encode a list of sentences into L2-normalised embeddings.

        Args:
            sentences: List of sentence strings.

        Returns:
            numpy array of shape (N, embedding_dim), float32, L2-normalised.
        """
        if not sentences:
            return np.array([])

        embeddings = self.model.encode(
            sentences,
            batch_size=32,
            show_progress_bar=False,
            normalize_embeddings=True,   # L2 normalise → cosine similarity = dot product
            convert_to_numpy=True,
        )
        return embeddings.astype(np.float32)