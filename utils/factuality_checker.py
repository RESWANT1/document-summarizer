"""
utils/factuality_checker.py
Verifies factual consistency using SummaC (IEEE publication requirement).
"""

import logging
from summac.model_summac import SummaCZS
import torch

logger = logging.getLogger(__name__)

class FactualityChecker:
    """NLI-based factual consistency checker."""
    
    def __init__(self):
        logger.info("Loading SummaC factuality model...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SummaCZS(
            granularity="sentence",
            model_name="vitc",  # Faster variant
            device=device
        )
        logger.info(f"FactualityChecker ready on {device.upper()}.")
    
    def check(self, source_text: str, summary: str) -> dict:
        """
        Compute factuality score between source and summary.
        
        Args:
            source_text: Original extractive context
            summary: Generated abstractive summary
            
        Returns:
            {
                "score": float (0-1, higher = more faithful),
                "is_faithful": bool (score > threshold)
            }
        """
        if not source_text.strip() or not summary.strip():
            return {"score": 0.0, "is_faithful": False}
        
        try:
            score = self.model.score([source_text], [summary])["scores"][0]
        except Exception as e:
            logger.error(f"Factuality check failed: {e}")
            return {"score": 0.0, "is_faithful": False}
        
        threshold = 0.75  # IEEE standard threshold
        is_faithful = score >= threshold
        
        logger.info(f"Factuality score: {score:.3f} (threshold: {threshold})")
        
        return {
            "score": round(float(score), 4),
            "is_faithful": bool(is_faithful)
        }
