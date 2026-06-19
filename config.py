# config.py — Central configuration for all components

# ── Model settings ────────────────────────────────────────────────────────────
EMBEDDING_MODEL   = "BAAI/bge-large-en-v1.5"
SUMMARIZER_MODEL  = "facebook/bart-large-cnn"  # Switched back from LED due to CUDA instability

# ── Length control ────────────────────────────────────────────────────────────
# CNN/DailyMail has very short reference summaries (~3-4 sentences)
# Using standard compression ratio for news summarization
LENGTH_FRACTIONS = {
    "short":  0.20,
    "medium": 0.25,  # Standard for CNN/DailyMail benchmark
    "long":   0.35,
}

# ── BART generation parameters ────────────────────────────────────────────────
# min/max lengths are dynamically derived from mode target words in summarizer_service.py.
BART_NUM_BEAMS         = 4     # Beam width: good coherence / speed balance
BART_NO_REPEAT_NGRAM   = 3     # Prevents obvious repetition without over-constraining
BART_LENGTH_PENALTY    = 1.0   # Changed from 1.1 to reduce over-paraphrasing
BART_MAX_INPUT_TOKENS  = 1024  # BART-large max context (switched back from LED's 16k)
BART_REPETITION_PENALTY = 1.0  # Reduced from 1.2 to allow more faithful content reproduction

# ── Text cleaner ──────────────────────────────────────────────────────────────
MIN_SENTENCE_LENGTH = 25   # chars — filters pure noise/single-word fragments
MAX_SENTENCE_LENGTH = 550  # chars — force-splits absurdly long run-ons

# ── Ranker weights ────────────────────────────────────────────────────────────
RANKER_CENTROID_WEIGHT  = 0.6   # Increased from 0.5 to prioritize salient content
RANKER_POSITION_WEIGHT  = 0.2   # Increased from 0.15 to favor important early sentences
RANKER_LENGTH_WEIGHT    = 0.05  # Decreased from 0.1 to reduce length bias
RANKER_KEYWORD_WEIGHT   = 0.1   # Decreased from 0.15
RANKER_STRUCTURAL_WEIGHT= 0.05  # Decreased from 0.1 (less relevant for news)
RANKER_POSITION_DECAY   = 0.90  # Reduced from 0.95 to give more weight to early content
RANKER_MMR_LAMBDA       = 0.85  # Increased from 0.7 - prioritize saliency over diversity

# ── Ranker selection tightening (long docs) ───────────────────────────────────
# Prevent extractive stage from keeping most sentences in long OCR docs.
RANKER_MAX_SENT_FRAC_LONGDOC = 0.35
RANKER_MAX_SENTENCES_LONGDOC = 22

# ── Ranker selection tightening (technical docs, ~15+ sentences) ──────────────
RANKER_MAX_SENT_FRAC_TECHDOC = 0.85  # Increased to 0.85 to prevent BART hallucination
RANKER_MAX_SENTENCES_TECHDOC = 80    # Increased from 70 to allow more sentences

# ── Evaluation ────────────────────────────────────────────────────────────────
EVAL_DATASET       = "cnn_dailymail"
EVAL_DATASET_VER   = "3.0.0"
EVAL_NUM_SAMPLES   = 500
EVAL_TOP_K         = 8       # medium length

# ── Flask ─────────────────────────────────────────────────────────────────────
UPLOAD_FOLDER      = "uploads"
MAX_CONTENT_MB     = 32
ALLOWED_EXTENSIONS = {"pdf", "docx", "pptx", "png", "jpg", "jpeg"}