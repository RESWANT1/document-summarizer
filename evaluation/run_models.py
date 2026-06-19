"""
evaluation/run_models.py
Run TextRank, BART, and HLA-MMR on arXiv and multi-format datasets.
"""

import json
import logging
from pathlib import Path
from tqdm import tqdm
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from summarizer.summarizer_service import SummarizerService
from summarizer.textrank_service import TextRankService
from summarizer.hla_mmr_service import HLAMMRService

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

DATASETS_DIR = Path(__file__).parent.parent / "datasets"
OUTPUTS_DIR = Path(__file__).parent / "model_outputs"
OUTPUTS_DIR.mkdir(exist_ok=True)

def run_arxiv_evaluation():
    """Run all models on arXiv papers."""
    logger.info("\n" + "="*60)
    logger.info("STEP 2: Running Models on arXiv")
    logger.info("="*60)
    
    with open(DATASETS_DIR / "arxiv_samples.json") as f:
        samples = json.load(f)
    
    results = {}
    
    # TextRank
    logger.info("\n[TextRank] Processing arXiv...")
    with TextRankService() as textrank:
        for sample in tqdm(samples, desc="TextRank"):
            doc_id = sample["id"]
            text = sample["intro_conclusion"]
            
            try:
                summary, _ = textrank.summarize(text, length="medium")
                with open(OUTPUTS_DIR / f"{doc_id}_textrank.txt", "w") as f:
                    f.write(summary)
                results[doc_id] = {"textrank_len": len(summary.split())}
            except Exception as e:
                logger.warning(f"TextRank failed for {doc_id}: {e}")
                results[doc_id] = {"textrank_len": 0}
    
    # BART
    logger.info("\n[BART] Processing arXiv...")
    with SummarizerService() as summarizer:
        for sample in tqdm(samples, desc="BART"):
            doc_id = sample["id"]
            text = sample["intro_conclusion"]
            
            try:
                summary, _ = summarizer.summarize(text, length="medium")
                with open(OUTPUTS_DIR / f"{doc_id}_bart.txt", "w") as f:
                    f.write(summary)
                if doc_id not in results:
                    results[doc_id] = {}
                results[doc_id]["bart_len"] = len(summary.split())
            except Exception as e:
                logger.warning(f"BART failed for {doc_id}: {e}")
                if doc_id not in results:
                    results[doc_id] = {}
                results[doc_id]["bart_len"] = 0
    
    # HLA-MMR
    logger.info("\n[HLA-MMR] Processing arXiv...")
    with HLAMMRService() as hla_mmr:
        for sample in tqdm(samples, desc="HLA-MMR"):
            doc_id = sample["id"]
            text = sample["intro_conclusion"]
            
            try:
                summary, _ = hla_mmr.summarize(text, length="medium")
                with open(OUTPUTS_DIR / f"{doc_id}_hla_mmr.txt", "w") as f:
                    f.write(summary)
                if doc_id not in results:
                    results[doc_id] = {}
                results[doc_id]["hla_mmr_len"] = len(summary.split())
            except Exception as e:
                logger.warning(f"HLA-MMR failed for {doc_id}: {e}")
                if doc_id not in results:
                    results[doc_id] = {}
                results[doc_id]["hla_mmr_len"] = 0
    
    logger.info(f"✓ arXiv: {len(samples)} documents processed")
    return results

def run_multiformat_evaluation():
    """Run all models on multi-format inputs."""
    logger.info("\n" + "="*60)
    logger.info("STEP 2B: Running Models on Multi-Format Inputs")
    logger.info("="*60)
    
    with open(DATASETS_DIR / "multiformat_samples.json") as f:
        samples = json.load(f)
    
    results = {}
    
    # TextRank
    logger.info("\n[TextRank] Processing multi-format...")
    with TextRankService() as textrank:
        for fmt, sample in samples.items():
            if sample is None:
                logger.warning(f"⚠ {fmt.upper()} sample not found")
                continue
            
            text = sample.get("text", "")
            if not text.strip():
                logger.warning(f"⚠ {fmt.upper()} text is empty")
                continue
            
            try:
                summary, _ = textrank.summarize(text, length="medium")
                with open(OUTPUTS_DIR / f"multiformat_{fmt}_textrank.txt", "w") as f:
                    f.write(summary)
                results[fmt] = {"file": sample.get("file", "unknown"), "textrank_len": len(summary.split())}
            except Exception as e:
                logger.warning(f"TextRank failed for {fmt}: {e}")
                results[fmt] = {"file": sample.get("file", "unknown"), "textrank_len": 0}
    
    # BART
    logger.info("\n[BART] Processing multi-format...")
    with SummarizerService() as summarizer:
        for fmt, sample in samples.items():
            if sample is None:
                continue
            
            text = sample.get("text", "")
            if not text.strip():
                continue
            
            try:
                summary, _ = summarizer.summarize(text, length="medium")
                with open(OUTPUTS_DIR / f"multiformat_{fmt}_bart.txt", "w") as f:
                    f.write(summary)
                if fmt not in results:
                    results[fmt] = {"file": sample.get("file", "unknown")}
                results[fmt]["bart_len"] = len(summary.split())
            except Exception as e:
                logger.warning(f"BART failed for {fmt}: {e}")
                if fmt not in results:
                    results[fmt] = {"file": sample.get("file", "unknown")}
                results[fmt]["bart_len"] = 0
    
    # HLA-MMR
    logger.info("\n[HLA-MMR] Processing multi-format...")
    with HLAMMRService() as hla_mmr:
        for fmt, sample in samples.items():
            if sample is None:
                continue
            
            text = sample.get("text", "")
            if not text.strip():
                continue
            
            try:
                summary, _ = hla_mmr.summarize(text, length="medium")
                with open(OUTPUTS_DIR / f"multiformat_{fmt}_hla_mmr.txt", "w") as f:
                    f.write(summary)
                if fmt not in results:
                    results[fmt] = {"file": sample.get("file", "unknown")}
                results[fmt]["hla_mmr_len"] = len(summary.split())
            except Exception as e:
                logger.warning(f"HLA-MMR failed for {fmt}: {e}")
                if fmt not in results:
                    results[fmt] = {"file": sample.get("file", "unknown")}
                results[fmt]["hla_mmr_len"] = 0
    
    return results

if __name__ == "__main__":
    arxiv_results = run_arxiv_evaluation()
    multiformat_results = run_multiformat_evaluation()
    
    logger.info("\n✓ All models executed!")
    logger.info(f"  Outputs saved to: {OUTPUTS_DIR}")
