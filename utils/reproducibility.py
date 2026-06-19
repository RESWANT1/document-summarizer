"""
utils/reproducibility.py
Ensures deterministic behavior for IEEE reproducibility standards.
"""

import random
import numpy as np
import torch
import os
import logging

logger = logging.getLogger(__name__)

def set_seed(seed: int = 42):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    logger.info(f"Random seed set to {seed} (deterministic mode enabled)")

def log_environment():
    """Log system environment for reproducibility."""
    import sys
    import transformers
    import sentence_transformers
    
    env_info = {
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else "N/A",
        "transformers_version": transformers.__version__,
        "sentence_transformers_version": sentence_transformers.__version__,
    }
    
    logger.info("Environment Info:")
    for key, value in env_info.items():
        logger.info(f"  {key}: {value}")
    
    return env_info
