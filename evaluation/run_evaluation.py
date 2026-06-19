"""
evaluation/run_evaluation.py
Master script to run the complete evaluation pipeline.
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def main():
    """Execute full evaluation pipeline."""
    
    print("\n" + "="*70)
    print("EVALUATION PIPELINE FOR SUMMARIZATION MODELS")
    print("="*70)
    
    # STEP 1: Prepare Datasets
    print("\n[STEP 1/6] Preparing Datasets...")
    try:
        from evaluation.prepare_datasets import prepare_arxiv, prepare_multiformat
        arxiv_samples = prepare_arxiv()
        multiformat_samples = prepare_multiformat()
        print("[OK] STEP 1 Complete: Datasets prepared")
    except Exception as e:
        print(f"[ERROR] STEP 1 Failed: {e}")
        return
    
    # STEP 2: Run Models
    print("\n[STEP 2/6] Running Models...")
    try:
        from evaluation.run_models import run_arxiv_evaluation, run_multiformat_evaluation
        arxiv_results = run_arxiv_evaluation()
        multiformat_results = run_multiformat_evaluation()
        print("[OK] STEP 2 Complete: All models executed")
    except Exception as e:
        print(f"[ERROR] STEP 2 Failed: {e}")
        print("Continuing to next step...")
    
    # STEP 3: Compute Metrics
    print("\n[STEP 3/6] Computing Metrics...")
    try:
        from evaluation.compute_metrics import evaluate_arxiv
        df_arxiv = evaluate_arxiv()
        print("[OK] STEP 3 Complete: Metrics computed")
    except Exception as e:
        print(f"[ERROR] STEP 3 Failed: {e}")
        print("Continuing to next step...")
    
    # STEP 4-6: Generate Report
    print("\n[STEP 4-6/6] Generating Report...")
    try:
        from evaluation.generate_report import generate_latex_tables, generate_multiformat_table, generate_results_section
        df_arxiv = generate_latex_tables()
        df_multiformat = generate_multiformat_table()
        results_section = generate_results_section()
        print("[OK] STEP 4-6 Complete: Report generated")
    except Exception as e:
        print(f"[ERROR] STEP 4-6 Failed: {e}")
    
    # Summary
    print("\n" + "="*70)
    print("EVALUATION COMPLETE!")
    print("="*70)
    print("\nGenerated Results for IEEE Paper:")
    print("  Results Directory: evaluation/results/")
    print("  LaTeX Tables:")
    print("     - table_arxiv.tex")
    print("     - table_multiformat.tex")
    print("  Paper Section:")
    print("     - results_section.tex")
    print("\nNext Steps:")
    print("  1. Copy table_arxiv.tex to your IEEE paper")
    print("  2. Copy table_multiformat.tex to your IEEE paper")
    print("  3. Copy results_section.tex to your IEEE paper")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
