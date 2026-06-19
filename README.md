# Multi-Format Document Summarization System

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-2.0+-green.svg)](https://flask.palletsprojects.com/)
[![Transformers](https://img.shields.io/badge/Transformers-4.0+-orange.svg)](https://huggingface.co/transformers/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An AI-powered document summarization system that generates concise, coherent summaries from diverse document formats (PDF, DOCX, Images) using state-of-the-art transformer models.

##  Key Features

- **Multi-Format Support**: Process PDF, DOCX, and image-based documents
- **Abstractive Summarization**: Uses BART (facebook/bart-large-cnn) for fluent, human-like summaries
- **Extractive Baselines**: Implements TextRank for comparison
- **Adaptive Length Control**: Short, medium, and long summary modes
- **Factuality Checking**: Validates summary accuracy against source
- **Web Interface**: User-friendly Flask-based UI
- **Comprehensive Evaluation**: ROUGE and SummaC metrics

## Architecture

```
┌─────────────────┐
│  User Upload    │
│ (PDF/DOCX/IMG)  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Text Extraction│
│  (Format-aware) │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Text Cleaning  │
│  & Segmentation │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  BART Model     │
│  (Abstractive)  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Summary Output │
│  + Statistics   │
└─────────────────┘
```

## Performance Results

Evaluated on arXiv dataset (5 academic papers):

| Model      | ROUGE-1 | ROUGE-2 | ROUGE-L | SummaC |
|------------|---------|---------|---------|--------|
| **BART**   | **0.4578** | **0.2399** | **0.3220** | **0.4797** |
| TextRank   | 0.2697  | 0.1462  | 0.2471  | 0.5072 |

**Key Findings:**
- BART achieves 70% higher ROUGE-1 score than TextRank
- Strong factual consistency (SummaC: 0.4797)
- Robust performance across PDF, DOCX, and image inputs

##  Quick Start

### Prerequisites
- Python 3.8+
- CUDA-enabled GPU (optional, but recommended)
- Tesseract OCR (for image processing)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/YOUR_USERNAME/document-summarizer.git
cd document-summarizer
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Install Tesseract OCR** (for image support)
- **Windows**: Download from [GitHub](https://github.com/UB-Mannheim/tesseract/wiki)
- **Linux**: `sudo apt-get install tesseract-ocr`
- **macOS**: `brew install tesseract`

### Running the Application

```bash
python app.py
```

Visit `http://localhost:5000` in your browser.

##  Usage

### Web Interface
1. Upload a document (PDF, DOCX, or image)
2. Select summary length (short/medium/long)
3. Click "Summarize"
4. View generated summary with statistics

### API Endpoint
```bash
curl -X POST http://localhost:5000/summarize \
  -F "file=@document.pdf" \
  -F "length=medium"
```

##  Evaluation

Run comprehensive evaluation:

```bash
# Prepare datasets
python evaluation/prepare_datasets.py

# Run all models
python evaluation/run_models.py

# Compute metrics
python evaluation/compute_metrics.py

# Generate visualizations
python evaluation/generate_visualizations.py
```

##  Project Structure

```
document-summarizer/
├── app.py                      # Flask web application
├── config.py                   # Configuration settings
├── requirements.txt            # Python dependencies
├── extractors/                 # Document extraction modules
│   ├── pdf_extractor.py
│   ├── docx_extractor.py
│   └── image_extractor.py
├── summarizer/                 # Summarization models
│   ├── summarizer_service.py   # BART model
│   └── textrank_service.py     # TextRank baseline
├── semantic/                   # Text processing
│   ├── sentence_segmenter.py
│   └── sentence_ranker.py
├── utils/                      # Utility functions
│   ├── text_cleaner.py
│   └── factuality_checker.py
├── evaluation/                 # Evaluation scripts
│   ├── prepare_datasets.py
│   ├── run_models.py
│   ├── compute_metrics.py
│   └── generate_visualizations.py
├── templates/                  # HTML templates
│   └── index.html
└── static/                     # CSS/JS assets
    ├── css/
    └── js/
```

## Technologies Used

- **Backend**: Flask, Python
- **NLP Models**: 
  - BART (facebook/bart-large-cnn) via Hugging Face Transformers
  - TextRank for extractive summarization
- **Document Processing**: PyPDF2, python-docx, pytesseract
- **Evaluation**: ROUGE, SummaC
- **Text Processing**: pysbd, NLTK

##  Visualizations

The project includes publication-quality visualizations:

- ROUGE metrics comparison
- Multi-metric heatmap
- Radar chart comparison
- Multi-format performance analysis

All figures are available in `evaluation/figures/`.

##  Future Enhancements

- [ ] Fine-tune BART on domain-specific datasets
- [ ] Add support for more languages
- [ ] Implement real-time streaming for long documents
- [ ] Add user authentication and history
- [ ] Deploy on cloud platforms (AWS/Azure/GCP)
- [ ] Create Docker container for easy deployment

##  Research Paper

This project is part of an IEEE research paper on multi-format document summarization. Key contributions:

1. Comprehensive comparison of extractive vs. abstractive approaches
2. Multi-format input processing pipeline
3. Factual consistency evaluation framework
4. Production-ready implementation

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

##  License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

##  Author

**Your Name**
- GitHub: [@RESWANT1]
- LinkedIn: [https://www.linkedin.com/in/reswant-raja-s-107570338/]
- Email: ITSRESWANT534.email@example.com

## Acknowledgments

- BART model by Facebook AI Research
- Hugging Face for Transformers library
- arXiv for dataset
- IEEE for publication support

##  Screenshots

### Main Interface
![Main Interface](screenshots/main_interface.png)

### Summary Results
![Summary Results](screenshots/results.png)

### Performance Metrics
![Performance Metrics](evaluation/figures/fig_rouge_comparison.png)

---

⭐ **Star this repo if you find it useful!**
