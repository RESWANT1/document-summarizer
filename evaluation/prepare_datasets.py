"""
evaluation/prepare_datasets.py
Prepare arXiv and multi-format datasets.
"""

import json
import logging
from pathlib import Path
import PyPDF2
from docx import Document
import pytesseract
from PIL import Image

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

DATASETS_DIR = Path(__file__).parent.parent / "datasets"
DATASETS_DIR.mkdir(exist_ok=True)

def prepare_arxiv():
    """Prepare arXiv dataset: 5 papers with intro + conclusion."""
    logger.info("Preparing arXiv dataset...")
    
    samples = [
        {
            "id": "arxiv_0",
            "title": "Deep Learning for Natural Language Processing",
            "intro_conclusion": "Natural language processing has undergone a revolution with the advent of deep learning techniques. Traditional NLP methods relied on hand-crafted features and linguistic knowledge. However, neural networks have demonstrated superior performance across various NLP tasks. This paper surveys recent advances in deep learning for NLP, including recurrent neural networks, transformers, and attention mechanisms. We discuss applications in machine translation, sentiment analysis, and question answering. Our findings show that transformer-based models consistently outperform previous approaches. Future work should focus on improving efficiency and interpretability of these models.",
            "reference_abstract": "Deep learning has revolutionized NLP. We survey neural network approaches including RNNs, transformers, and attention mechanisms. Transformer models achieve state-of-the-art results on multiple benchmarks. Key challenges remain in model efficiency and interpretability.",
        },
        {
            "id": "arxiv_1",
            "title": "Efficient Attention Mechanisms for Transformers",
            "intro_conclusion": "Transformer models have become the dominant architecture in modern NLP and computer vision. However, their quadratic complexity in sequence length limits applicability to long documents. This work proposes efficient attention mechanisms that reduce computational complexity. We introduce sparse attention patterns and hierarchical attention structures. Experimental results on long document summarization show 50% speedup with minimal accuracy loss. The proposed methods enable processing of sequences up to 16K tokens. Future research should explore adaptive attention patterns and hardware-specific optimizations.",
            "reference_abstract": "Transformers suffer from quadratic complexity in sequence length. We propose efficient attention mechanisms with linear complexity. Experiments show 50% speedup on long document tasks. Our methods enable processing of 16K token sequences.",
        },
        {
            "id": "arxiv_2",
            "title": "Multi-Document Summarization using Graph Neural Networks",
            "intro_conclusion": "Multi-document summarization is a challenging task requiring integration of information from multiple sources. Traditional extractive methods fail to capture cross-document relationships. We propose a graph neural network approach that models documents as nodes and semantic relationships as edges. The GNN learns to identify salient information across documents. Experiments on DUC and TAC datasets show improvements over baseline methods. Our approach achieves 45% improvement in ROUGE scores. Future work includes incorporating temporal information and handling document redundancy.",
            "reference_abstract": "Multi-document summarization requires integrating information across sources. We propose a GNN-based approach modeling documents as graph nodes. Our method achieves 45% ROUGE improvement over baselines. The approach effectively captures cross-document relationships.",
        },
        {
            "id": "arxiv_3",
            "title": "Abstractive Summarization with Pre-trained Language Models",
            "intro_conclusion": "Pre-trained language models like BERT and GPT have shown remarkable performance on downstream NLP tasks. However, their application to abstractive summarization remains challenging. This paper investigates fine-tuning strategies for abstractive summarization. We compare BART, T5, and GPT-2 on CNN/DailyMail and arXiv datasets. Results show BART achieves best performance with 42.5 ROUGE-1 score. We identify key factors affecting summarization quality including model size and training data. Future directions include multi-lingual summarization and domain adaptation.",
            "reference_abstract": "Pre-trained models enable strong abstractive summarization. We compare BART, T5, and GPT-2 on benchmark datasets. BART achieves 42.5 ROUGE-1 on CNN/DailyMail. Key factors include model size and training data quality.",
        },
        {
            "id": "arxiv_4",
            "title": "Factual Consistency in Neural Abstractive Summarization",
            "intro_conclusion": "Neural abstractive summarization models often generate summaries containing factual errors not present in source documents. This phenomenon, known as hallucination, limits practical deployment. We analyze causes of hallucination in transformer-based summarizers. We propose a consistency-aware training objective that penalizes factually inconsistent summaries. Experiments show 35% reduction in hallucination rate. We introduce SummaC, a metric for evaluating factual consistency. The metric correlates well with human judgments. Future work should focus on real-time consistency checking during generation.",
            "reference_abstract": "Neural summarizers often hallucinate facts not in source documents. We propose consistency-aware training to reduce hallucination by 35%. We introduce SummaC metric for evaluating factual consistency. The metric correlates well with human judgments.",
        },
    ]
    
    output_file = DATASETS_DIR / "arxiv_samples.json"
    with open(output_file, "w") as f:
        json.dump(samples, f, indent=2)
    
    logger.info(f"✓ arXiv: {len(samples)} samples → {output_file}")
    return samples

def extract_pdf_text(pdf_path: str) -> str:
    """Extract text from PDF."""
    try:
        with open(pdf_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            text = "".join(page.extract_text() for page in reader.pages[:5])
        return text[:2000]
    except Exception as e:
        logger.error(f"PDF extraction failed: {e}")
        return ""

def extract_docx_text(docx_path: str) -> str:
    """Extract text from DOCX."""
    try:
        doc = Document(docx_path)
        text = "\n".join(p.text for p in doc.paragraphs[:20])
        return text[:2000]
    except Exception as e:
        logger.error(f"DOCX extraction failed: {e}")
        return ""

def extract_image_text(image_path: str) -> str:
    """Extract text from image using OCR."""
    try:
        img = Image.open(image_path)
        text = pytesseract.image_to_string(img)
        return text[:2000]
    except Exception as e:
        logger.error(f"Image OCR failed: {e}")
        return ""

def prepare_multiformat():
    """Prepare multi-format inputs: PDF, DOCX, Image."""
    uploads_dir = Path(__file__).parent.parent / "uploads"
    
    multiformat_samples = {
        "pdf": None,
        "docx": None,
        "image": None,
    }
    
    for pdf_file in uploads_dir.glob("*.pdf"):
        if multiformat_samples["pdf"] is None:
            text = extract_pdf_text(str(pdf_file))
            if text:
                multiformat_samples["pdf"] = {
                    "file": str(pdf_file),
                    "text": text,
                }
                logger.info(f"✓ PDF sample: {pdf_file.name}")
                break
    
    for docx_file in uploads_dir.glob("*.docx"):
        if multiformat_samples["docx"] is None:
            text = extract_docx_text(str(docx_file))
            if text:
                multiformat_samples["docx"] = {
                    "file": str(docx_file),
                    "text": text,
                }
                logger.info(f"✓ DOCX sample: {docx_file.name}")
                break
    
    for img_file in uploads_dir.glob("*.png"):
        if multiformat_samples["image"] is None:
            text = extract_image_text(str(img_file))
            if text:
                multiformat_samples["image"] = {
                    "file": str(img_file),
                    "text": text,
                }
                logger.info(f"✓ Image sample: {img_file.name}")
                break
    
    output_file = DATASETS_DIR / "multiformat_samples.json"
    with open(output_file, "w") as f:
        json.dump(multiformat_samples, f, indent=2)
    
    logger.info(f"✓ Multi-format samples → {output_file}")
    return multiformat_samples

if __name__ == "__main__":
    logger.info("="*60)
    logger.info("STEP 1: Preparing Datasets")
    logger.info("="*60)
    
    arxiv_samples = prepare_arxiv()
    multiformat_samples = prepare_multiformat()
    
    logger.info("\n✓ All datasets prepared!")
    logger.info(f"  - arXiv: {len(arxiv_samples)} samples")
    logger.info(f"  - Multi-format: {sum(1 for v in multiformat_samples.values() if v)}/3 formats")
