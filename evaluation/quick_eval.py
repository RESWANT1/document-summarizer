"""
evaluation/quick_eval.py
Quick evaluation script to run models and compute metrics.
"""

import json
import logging
from pathlib import Path
from tqdm import tqdm
import pandas as pd
from rouge_score import rouge_scorer
from summac.model_summac import SummaCZS
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from summarizer.summarizer_service import SummarizerService
from summarizer.textrank_service import TextRankService
from summarizer.hla_mmr_service import HLAMMRService

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

DATASETS_DIR = Path(__file__).parent.parent / "datasets"
OUTPUTS_DIR = Path(__file__).parent / "model_outputs"
RESULTS_DIR = Path(__file__).parent / "results"
OUTPUTS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

def run_models():
    """Run all models on arXiv."""
    print("\n[Running Models on arXiv...]")
    
    with open(DATASETS_DIR / "arxiv_samples.json") as f:
        samples = json.load(f)
    
    # TextRank
    print("TextRank...")
    with TextRankService() as textrank:
        for sample in samples:
            doc_id = sample["id"]
            text = sample["intro_conclusion"]
            summary, _ = textrank.summarize(text, length="medium")
            with open(OUTPUTS_DIR / f"{doc_id}_textrank.txt", "w") as f:
                f.write(summary)
    
    # BART
    print("BART...")
    with SummarizerService() as summarizer:
        for sample in samples:
            doc_id = sample["id"]
            text = sample["intro_conclusion"]
            summary, _ = summarizer.summarize(text, length="medium")
            with open(OUTPUTS_DIR / f"{doc_id}_bart.txt", "w") as f:
                f.write(summary)
    
    # HLA-MMR
    print("HLA-MMR...")
    with HLAMMRService() as hla_mmr:
        for sample in samples:
            doc_id = sample["id"]
            text = sample["intro_conclusion"]
            summary, _ = hla_mmr.summarize(text, length="medium")
            with open(OUTPUTS_DIR / f"{doc_id}_hla_mmr.txt", "w") as f:
                f.write(summary)

def compute_metrics():
    """Compute metrics for all models."""
    print("\n[Computing Metrics...]")
    
    with open(DATASETS_DIR / "arxiv_samples.json") as f:
        samples = json.load(f)
    
    rouge_scorer_obj = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    summac = SummaCZS(granularity="sentence", model_name="vitc")
    
    results = {
        "textrank": {"rouge1": [], "rouge2": [], "rougeL": [], "summac": []},
        "bart": {"rouge1": [], "rouge2": [], "rougeL": [], "summac": []},
        "hla_mmr": {"rouge1": [], "rouge2": [], "rougeL": [], "summac": []},
    }
    
    for sample in tqdm(samples, desc="Computing metrics"):
        doc_id = sample["id"]
        text = sample["intro_conclusion"]
        reference = sample["reference_abstract"]
        
        for model in ["textrank", "bart", "hla_mmr"]:
            summary_file = OUTPUTS_DIR / f"{doc_id}_{model}.txt"
            if summary_file.exists():
                with open(summary_file) as f:
                    summary = f.read()
                
                # ROUGE
                scores = rouge_scorer_obj.score(reference, summary)
                results[model]["rouge1"].append(round(scores['rouge1'].fmeasure, 4))
                results[model]["rouge2"].append(round(scores['rouge2'].fmeasure, 4))
                results[model]["rougeL"].append(round(scores['rougeL'].fmeasure, 4))
                
                # SummaC
                try:
                    score = summac.score([text], [summary])["scores"][0]
                    results[model]["summac"].append(round(float(score), 4))
                except:
                    results[model]["summac"].append(0.0)
    
    # Create table
    table = []
    for model in ["textrank", "bart", "hla_mmr"]:
        if results[model]["rouge1"]:
            table.append({
                "Model": model.upper(),
                "ROUGE-1": round(sum(results[model]["rouge1"]) / len(results[model]["rouge1"]), 4),
                "ROUGE-2": round(sum(results[model]["rouge2"]) / len(results[model]["rouge2"]), 4),
                "ROUGE-L": round(sum(results[model]["rougeL"]) / len(results[model]["rougeL"]), 4),
                "SummaC": round(sum(results[model]["summac"]) / len(results[model]["summac"]), 4),
            })
    
    df = pd.DataFrame(table)
    print("\nResults:")
    print(df.to_string(index=False))
    
    # Save LaTeX
    latex = df.to_latex(index=False, float_format=lambda x: f"{x:.4f}")
    with open(RESULTS_DIR / "table_arxiv.tex", "w") as f:
        f.write("\\begin{table}[h]\n")
        f.write("\\centering\n")
        f.write("\\caption{arXiv Evaluation Results (TextRank, BART, HLA-MMR)}\n")
        f.write("\\label{tab:arxiv_results}\n")
        f.write(latex)
        f.write("\\end{table}\n")
    
    print("\nResults saved to evaluation/results/table_arxiv.tex")

if __name__ == "__main__":
    run_models()
    compute_metrics()
