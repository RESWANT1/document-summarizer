"""
evaluation/compute_metrics.py
Compute ROUGE and SummaC metrics for all models.
"""

import json
import logging
from pathlib import Path
from tqdm import tqdm
import pandas as pd
from rouge_score import rouge_scorer
from summac.model_summac import SummaCZS

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

DATASETS_DIR = Path(__file__).parent.parent / "datasets"
OUTPUTS_DIR = Path(__file__).parent / "model_outputs"
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

class MetricsComputer:
    def __init__(self):
        self.rouge_scorer = rouge_scorer.RougeScorer(
            ['rouge1', 'rouge2', 'rougeL'], 
            use_stemmer=True
        )
        logger.info("Loading SummaC...")
        self.summac = SummaCZS(granularity="sentence", model_name="vitc")
    
    def compute_rouge(self, reference: str, summary: str) -> dict:
        """Compute ROUGE scores."""
        scores = self.rouge_scorer.score(reference, summary)
        return {
            "rouge1": round(scores['rouge1'].fmeasure, 4),
            "rouge2": round(scores['rouge2'].fmeasure, 4),
            "rougeL": round(scores['rougeL'].fmeasure, 4),
        }
    
    def compute_summac(self, source: str, summary: str) -> float:
        """Compute SummaC faithfulness score."""
        try:
            score = self.summac.score([source], [summary])["scores"][0]
            return round(float(score), 4)
        except Exception as e:
            logger.error(f"SummaC error: {e}")
            return 0.0

def evaluate_arxiv():
    """Evaluate all models on arXiv."""
    logger.info("\n" + "="*60)
    logger.info("STEP 3: Computing Metrics for arXiv")
    logger.info("="*60)
    
    with open(DATASETS_DIR / "arxiv_samples.json") as f:
        samples = json.load(f)
    
    computer = MetricsComputer()
    results = {
        "textrank": {"rouge1": [], "rouge2": [], "rougeL": [], "summac": []},
        "bart": {"rouge1": [], "rouge2": [], "rougeL": [], "summac": []},
        "hla_mmr": {"rouge1": [], "rouge2": [], "rougeL": [], "summac": []},
    }
    
    for sample in tqdm(samples, desc="arXiv Metrics"):
        doc_id = sample["id"]
        text = sample["intro_conclusion"]
        reference = sample["reference_abstract"]
        
        # TextRank
        summary_file = OUTPUTS_DIR / f"{doc_id}_textrank.txt"
        if summary_file.exists():
            with open(summary_file) as f:
                summary = f.read()
            rouge = computer.compute_rouge(reference, summary)
            results["textrank"]["rouge1"].append(rouge["rouge1"])
            results["textrank"]["rouge2"].append(rouge["rouge2"])
            results["textrank"]["rougeL"].append(rouge["rougeL"])
            summac = computer.compute_summac(text, summary)
            results["textrank"]["summac"].append(summac)
        
        # BART
        summary_file = OUTPUTS_DIR / f"{doc_id}_bart.txt"
        if summary_file.exists():
            with open(summary_file) as f:
                summary = f.read()
            rouge = computer.compute_rouge(reference, summary)
            results["bart"]["rouge1"].append(rouge["rouge1"])
            results["bart"]["rouge2"].append(rouge["rouge2"])
            results["bart"]["rougeL"].append(rouge["rougeL"])
            summac = computer.compute_summac(text, summary)
            results["bart"]["summac"].append(summac)
        
        # HLA-MMR
        summary_file = OUTPUTS_DIR / f"{doc_id}_hla_mmr.txt"
        if summary_file.exists():
            with open(summary_file) as f:
                summary = f.read()
            rouge = computer.compute_rouge(reference, summary)
            results["hla_mmr"]["rouge1"].append(rouge["rouge1"])
            results["hla_mmr"]["rouge2"].append(rouge["rouge2"])
            results["hla_mmr"]["rougeL"].append(rouge["rougeL"])
            summac = computer.compute_summac(text, summary)
            results["hla_mmr"]["summac"].append(summac)
    
    # Compute averages
    arxiv_table = []
    for model in ["textrank", "bart", "hla_mmr"]:
        if results[model]["rouge1"]:
            arxiv_table.append({
                "Model": model.upper(),
                "ROUGE-1": round(sum(results[model]["rouge1"]) / len(results[model]["rouge1"]), 4),
                "ROUGE-2": round(sum(results[model]["rouge2"]) / len(results[model]["rouge2"]), 4),
                "ROUGE-L": round(sum(results[model]["rougeL"]) / len(results[model]["rougeL"]), 4),
                "SummaC": round(sum(results[model]["summac"]) / len(results[model]["summac"]), 4),
            })
    
    df_arxiv = pd.DataFrame(arxiv_table)
    logger.info("\narXiv Results:")
    logger.info(df_arxiv.to_string(index=False))
    
    df_arxiv.to_csv(RESULTS_DIR / "arxiv_results.csv", index=False)
    
    return df_arxiv

if __name__ == "__main__":
    df_arxiv = evaluate_arxiv()
    
    logger.info("\n✓ Metrics computed!")
    logger.info(f"  Results saved to: {RESULTS_DIR}")
