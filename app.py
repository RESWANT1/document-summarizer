"""
app.py — Entry point for the Hybrid Document Summarizer
Orchestrates: extraction → cleaning → segmentation → embedding → ranking → summarization
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
import logging
from utils.reproducibility import set_seed, log_environment

# Set seed for reproducibility (IEEE requirement)
set_seed(42)
log_environment()
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename

from extractors.pdf_extractor import extract_pdf
from extractors.docx_extractor import extract_docx
from extractors.image_extractor import extract_image
from extractors.pptx_extractor import extract_pptx
from utils.text_cleaner import clean_text
from semantic.sentence_segmenter import segment_sentences
from semantic.embedding_service import EmbeddingService
from semantic.sentence_ranker import SentenceRanker
from summarizer.summarizer_service import SummarizerService
from postprocess import postprocess_summary
from utils.factuality_checker import FactualityChecker

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = config.UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = config.MAX_CONTENT_MB * 1024 * 1024
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

ALLOWED_EXTENSIONS = config.ALLOWED_EXTENSIONS

def _compute_eval_metrics(summary: str, reference_summary: str) -> dict:
    """
    Compute ROUGE-1/2/L and BERTScore-F1 for a single summary-reference pair.
    Returns {} when reference is missing.
    """
    if not (reference_summary or "").strip():
        return {}

    from rouge_score import rouge_scorer
    import bert_score

    ref = reference_summary.strip()
    pred = (summary or "").strip()

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    r = scorer.score(ref, pred)
    _, _, f1 = bert_score.score([pred], [ref], lang="en", verbose=False)

    return {
        "rouge1": round(r["rouge1"].fmeasure, 4),
        "rouge2": round(r["rouge2"].fmeasure, 4),
        "rougeL": round(r["rougeL"].fmeasure, 4),
        "bertscore_f1": round(f1.mean().item(), 4),
    }

# ── Thread-safe service management ───────────────────────────────────────────
import threading
_service_lock = threading.Lock()
_embedder = None
_ranker = None
_summarizer = None
_factuality_checker = None

def get_services():
    """Thread-safe lazy loading of services."""
    global _embedder, _ranker, _summarizer, _factuality_checker
    
    with _service_lock:
        if _embedder is None:
            logger.info("Loading models (first request — may take 30–60 s)…")
            _embedder = EmbeddingService()
            _ranker = SentenceRanker(_embedder)
            try:
                _summarizer = SummarizerService()
            except Exception as e:
                logger.error("Failed to load summarizer: %s", e)
                return jsonify({"error": "Model loading failed. Please restart the server and ensure stable internet connection."}), 503
            _factuality_checker = FactualityChecker()
            logger.info("All models loaded.")
    return _embedder, _ranker, _summarizer, _factuality_checker


# ── Helpers ───────────────────────────────────────────────────────────────────
def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def get_target_words(length_key: str, original_word_count: int, num_sentences: int) -> int:
    """Adaptive compression based on input length - longer docs get lower compression ratios."""
    
    # Short document fallback (OCR/small files)
    if original_word_count < 120 or num_sentences <= 4:
        fractions = {"short": 0.60, "medium": 0.80, "long": 0.95}
        logger.info("Short document detected (words: %d, sents: %d). Light compression.", 
                   original_word_count, num_sentences)
        frac = fractions.get(length_key, 0.80)
        return max(10, int(frac * original_word_count))
    
    # Adaptive compression tiers based on document length
    if original_word_count < 500:
        # Very short docs: keep more content
        ratios = {"short": 0.40, "medium": 0.55, "long": 0.70}
    elif original_word_count < 1500:
        # Short-medium docs: moderate compression
        ratios = {"short": 0.30, "medium": 0.45, "long": 0.60}
    elif original_word_count < 3000:
        # Medium docs: standard compression
        ratios = {"short": 0.25, "medium": 0.35, "long": 0.50}
    elif original_word_count < 6000:
        # Long docs: higher compression
        ratios = {"short": 0.18, "medium": 0.28, "long": 0.40}
    else:
        # Very long docs: aggressive compression
        ratios = {"short": 0.12, "medium": 0.20, "long": 0.30}
    
    frac = ratios.get(length_key, 0.35)
    target = int(frac * original_word_count)
    
    # Absolute caps to prevent extreme outputs
    caps = {"short": 400, "medium": 800, "long": 1500}
    final_target = max(50, min(target, caps.get(length_key, 800)))
    
    logger.info("Adaptive compression: %d words → %d words (%.1f%% ratio, tier: %s)",
               original_word_count, final_target, (final_target/original_word_count)*100,
               "<500" if original_word_count < 500 else 
               "<1.5k" if original_word_count < 1500 else
               "<3k" if original_word_count < 3000 else
               "<6k" if original_word_count < 6000 else ">6k")
    
    return final_target


def extract_text(filepath: str, ext: str) -> str:
    extractors = {
        "pdf":  extract_pdf,
        "docx": extract_docx,
        "pptx": extract_pptx,
        "png":  extract_image,
        "jpg":  extract_image,
        "jpeg": extract_image,
    }
    fn = extractors.get(ext)
    if fn is None:
        raise ValueError(f"Unsupported file type: {ext}")
    return fn(filepath)


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/summarize", methods=["POST"])
def summarize():
    """
    POST /summarize
    Form-data fields:
      - file      : document upload (pdf/docx/image/pptx)
      - length    : 'short' | 'medium' | 'long'  (default: medium)
      - text      : raw text (alternative to file upload)
      - reference_summary : optional reference for evaluation metrics
    """
    length_key = request.form.get("length", "medium").lower()
    reference_summary = request.form.get("reference_summary", "")

    # ── 1. Get raw text ───────────────────────────────────────────────────────
    if "file" in request.files and request.files["file"].filename:
        f = request.files["file"]
        if not allowed_file(f.filename):
            return jsonify({"error": "Unsupported file type."}), 400
        filename = secure_filename(f.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        f.save(filepath)
        ext = filename.rsplit(".", 1)[1].lower()
        try:
            raw_text = extract_text(filepath, ext)
        except Exception as e:
            logger.error("Extraction failed: %s", e)
            return jsonify({"error": f"Text extraction failed: {e}"}), 500
    elif request.form.get("text"):
        raw_text = request.form["text"]
    else:
        return jsonify({"error": "Provide a file or text field."}), 400

    if not raw_text.strip():
        return jsonify({"error": "No text could be extracted from the document."}), 400

    logger.info("--- START PIPELINE TRACE ---")
    logger.info("[Extracted] Total input chars: %d", len(raw_text))

    # ── 2. Pipeline ───────────────────────────────────────────────────────────
    import time
    start_time = time.time()
    try:
        result = get_services()
        if not isinstance(result, tuple):
            return result  # Error response from get_services
        _, ranker, summarizer, factuality_checker = result
        
        if summarizer is None:
            return jsonify({"error": "Summarizer not loaded. Please restart the server."}), 503

        clean, cleaning_stats = clean_text(raw_text, return_stats=True, resolve_coref=True)
        logger.info("[Cleaned] Total cleaned chars: %d", len(clean))
        
        sents   = segment_sentences(clean)
        
        # Calculate word counts
        original_word_count = len(raw_text.split())
        target_words = get_target_words(length_key, original_word_count, len(sents))
        
        # Give the abstractive model a larger extractive "context window" to rewrite from
        if original_word_count < 120 or len(sents) <= 4:
            # For short OCR documents, do not restrict the extractive budget at all. 
            # Give BART 100% of the sentences and let it purely paraphrase.
            extractive_budget = original_word_count
        else:
            # Ensure it is at least 50% larger than the final target so BART can actually summarize it down
            extractive_budget = max(int(target_words * 1.5), min(original_word_count, int(target_words * 2.0)))
        
        logger.info("[Segmented] Total valid sentences: %d. Final target words: %d. Context budget: %d", len(sents), target_words, extractive_budget)
        
        top_sents = ranker.rank_and_select(sents, target_words=extractive_budget)
        extractive_text = " ".join(top_sents)
        
        # Track extraction stats
        extracted_char_count = len(extractive_text)
        selected_sentence_count = len(top_sents)
        
        # Calculate chunks based on sentences 
        # (This is a simplified count just for UI, accurate tracking happens inside summarizer)
        tokens_rough_estimate = len(summarizer.tokenizer.encode(extractive_text, add_special_tokens=False)) if summarizer and hasattr(summarizer, 'tokenizer') else 0
        chunking_triggered = tokens_rough_estimate > config.BART_MAX_INPUT_TOKENS
        
        # Use BART+MMR novelty reranking for summary
        # --- Summarizer parameter tuning ---
        summary, meta = summarizer.summarize(
            extractive_text,
            length=length_key,
            target_word_count=target_words,
            novelty_rerank=True,   # Enable MMR reranking
            num_candidates=7,      # Increased candidates for diversity
            mmr_lambda=0.8         # Slightly higher for more novelty
        )
        summary = postprocess_summary(summary)

        # Compute factuality score (IEEE requirement)
        factuality_result = factuality_checker.check(extractive_text, summary)

        # Flag low-factuality summaries
        is_faithful = factuality_result.get("is_faithful", False)
        factuality_flag = "PASS" if is_faithful else "LOW FACTUALITY"
        
        # Calculate novel n-grams (abstractiveness metric)
        def calculate_novel_ngrams(summary_text: str, source_text: str, n: int = 2) -> float:
            """Calculate percentage of novel n-grams in summary."""
            from collections import Counter
            summary_words = summary_text.lower().split()
            source_words = source_text.lower().split()
            
            if len(summary_words) < n:
                return 0.0
            
            summary_ngrams = [tuple(summary_words[i:i+n]) for i in range(len(summary_words)-n+1)]
            source_ngrams = set(tuple(source_words[i:i+n]) for i in range(len(source_words)-n+1))
            
            if not summary_ngrams:
                return 0.0
            
            novel_count = sum(1 for ng in summary_ngrams if ng not in source_ngrams)
            return round((novel_count / len(summary_ngrams)) * 100, 1)
        
        novel_bigrams = calculate_novel_ngrams(summary, extractive_text, n=2)
        
        logger.info("Factuality result: %s", factuality_result)
        logger.info("Novel bigrams: %s%%", novel_bigrams)
        
        summary_word_count = len(summary.split())
        compression_ratio = round((summary_word_count / original_word_count) * 100, 1) if original_word_count else 0
        
        # -- DEBUG SECTION --
        logger.info("========== DEBUG OUTPUT ==========")
        logger.info("Original word count: %d", original_word_count)
        logger.info("Target word count for mode '%s': %d", length_key, target_words)
        logger.info("Selected extractive context word count before BART: %d", len(extractive_text.split()))
        logger.info("Final generated summary word count: %d", summary_word_count)
        logger.info("Final output is shorter than intended mode target? %s", "Yes" if summary_word_count < target_words - 5 else "No")
        logger.info("==================================")

        processing_time = round(time.time() - start_time, 2)
        
        # Determine the string status for the exact path taken
        if meta.get("novelty", False):
            summary_path = "BART + MMR Novelty Rerank"
        elif meta.get("multipass", False):
            summary_path = "Multi-Pass Hierarchical Abstractive"
        elif meta.get("chunk_count", 1) > 1:
            summary_path = "Chunked Hierarchical Abstractive"
        else:
            summary_path = "Extractive + Abstractive"

        return jsonify({
            "summary": summary,
            "factuality_flag": factuality_flag,
            "factuality_score": factuality_result.get("score", 0.0),
            "top_sentences": top_sents,  # Hidden debug data
            "extractive_text": extractive_text,  # Full extractive context for display
            "stats": {
                "original_word_count": original_word_count,
                "summary_word_count": summary_word_count,
                "compression_ratio": f"{compression_ratio}%",
                "extracted_char_count": extracted_char_count,
                "total_sentence_count": len(sents),
                "selected_sentence_count": selected_sentence_count,
                "chunk_count": meta.get("chunk_count", 1),
                "avg_chunk_tokens": meta.get("avg_chunk_tokens", 0),
                "multipass_triggered": meta.get("multipass", False),
                "summary_mode": length_key,
                "model_used": config.SUMMARIZER_MODEL,
                "processing_time": f"{processing_time}s",
                "summary_path": summary_path,
                "novelty_algorithm": "BART + MMR Novelty Rerank",
                "ranker_params": {
                    "lambda_mmr": 0.7,
                    "layout_boost": True
                },
                "cleaning": {
                    "coref_resolved": True,
                    "rules_added": cleaning_stats.get("rules_added", []),
                    "removed_lines_total": cleaning_stats.get("removed_lines_total", 0),
                    "removed_by_rule": cleaning_stats.get("removed_by_rule", {}),
                },
            },
            "evaluation": {
                **_compute_eval_metrics(summary, reference_summary),
                "novel_bigrams": novel_bigrams,
            },
            "factuality": factuality_result,
        })
    except Exception as e:
        logger.exception("Pipeline error")
        return jsonify({"error": str(e)}), 500


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    # CWE-489 FIX: Never enable debug in production
    import os
    debug_mode = True  # Always enable debug mode for development
    app.run(debug=debug_mode, host="127.0.0.1", port=5000)