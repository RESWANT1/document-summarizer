"""
postprocess.py
Light post-processing applied to the final BART summary before returning to the user.
  - Capitalise first letter of summary and each sentence
  - Ensure ending punctuation
  - Fix minor formatting artefacts without breaking abbreviations (U.S., Jan 2021, etc.)
"""

import re


# Matches a space before ?,!," ONLY when that space is NOT preceded by a single
# uppercase letter (which would be an abbreviation like "U. S.")
_SPACE_BEFORE_CLOSING_PUNCT = re.compile(r'(?<![A-Z])\s+([?!"])')

# Sentence boundary: period/!/? followed by 1+ spaces and a lowercase letter,
# BUT only when the period is preceded by 2+ chars (not a single-letter abbreviation).
# This prevents "U.S. between" → "U.S. Between".
_SENT_BOUNDARY = re.compile(r'(?<=[a-zA-Z]{2})([.!?])\s+([a-z])')


def _capitalize_sentences(text: str) -> str:
    """Capitalise the first letter after each genuine sentence-ending punctuation."""
    def _cap(m: re.Match) -> str:
        return m.group(1) + " " + m.group(2).upper()
    return _SENT_BOUNDARY.sub(_cap, text)


def postprocess_summary(text: str) -> str:
    """Clean up the BART output summary."""
    if not text:
        return ""

    text = text.strip()
    # Remove hallucinated URLs and "For more information" spam
    text = re.sub(r'For more (information|details|info)[^.!?]*[.!?]', '', text, flags=re.IGNORECASE)
    # Remove all URL patterns (including malformed ones)
    text = re.sub(r'https?://[^\s]*', '', text)
    text = re.sub(r'www\.[^\s]*', '', text)
    text = re.sub(r'\b[a-zA-Z0-9-]+\.(com|org|gov|net|edu|html|php)[^\s]*', '', text)
    # Remove "visit:" patterns
    text = re.sub(r'visit:?\s*[^\s.!?]*', '', text, flags=re.IGNORECASE)
    # Remove domain-like patterns (barc-project.org, BART-Project.gov, etc.)
    text = re.sub(r'\b[A-Z]?[a-z]*-?[Pp]roject\.[a-z]{2,4}[^\s.!?]*', '', text)
    # Remove standalone domain fragments
    text = re.sub(r'\b[a-zA-Z0-9-]+\.org/[^\s]*', '', text)
    text = re.sub(r'\b[a-zA-Z0-9-]+\.gov/[^\s]*', '', text)
    text = re.sub(r'\b[a-zA-Z0-9-]+\.com/[^\s]*', '', text)

    # Capitalise first character of the whole summary
    text = text[0].upper() + text[1:] if len(text) > 1 else text.upper()

    # Capitalise first word of each subsequent sentence
    text = _capitalize_sentences(text)

    # Ensure the summary ends with sentence-terminal punctuation
    if text and text[-1] not in ".!?":
        text += "."

    # ── Formatting artefacts ──────────────────────────────────────────────────
    # Collapse runs of periods (ellipsis artefacts → single period)
    text = re.sub(r'\.{2,}', '.', text)

    # Remove space before ?,!" but NOT before . (would break "U.S.", "e.g.", dates)
    text = _SPACE_BEFORE_CLOSING_PUNCT.sub(r'\1', text)

    # Snap floating quotes: " word " → "word"
    text = re.sub(r'([\'"])+(.+?)\s+([\'"])', r'\1\2\3', text)

    # Collapse double spaces (after URL removal)
    text = re.sub(r'\s{2,}', ' ', text)
    
    # Remove trailing fragments after URL removal
    text = re.sub(r'\s+[.!?]', '.', text)

    # --- Grammar and coherence fixes ---
    # Fix common grammar issues: lowercase after period, missing spaces, etc.
    text = re.sub(r'([.!?])([A-Za-z])', r'\1 \2', text)  # Ensure space after punctuation
    text = re.sub(r'\bi\s+am\b', 'I am', text)
    text = re.sub(r'\bi\s+was\b', 'I was', text)
    text = re.sub(r'\bi\s+have\b', 'I have', text)
    text = re.sub(r'\bi\s+will\b', 'I will', text)
    # Remove repeated words (e.g., "the the")
    text = re.sub(r'\b(\w+) \1\b', r'\1', text)
    # Remove stray single characters at the start of lines
    text = re.sub(r'^\w\s+', '', text, flags=re.MULTILINE)

    # --- Remove repeated phrases/sentences ---
    # Split into sentences, remove duplicates while preserving order
    import itertools
    import collections
    sentences = re.split(r'(?<=[.!?])\s+', text)
    seen = set()
    unique_sentences = []
    for s in sentences:
        s_clean = s.strip().lower()
        if s_clean and s_clean not in seen:
            seen.add(s_clean)
            unique_sentences.append(s.strip())
    # Join back into a single paragraph
    text = ' '.join(unique_sentences)
    # Remove repeated n-grams (3+ words)
    def remove_repeated_ngrams(text, n=5):
        words = text.split()
        ngrams = collections.defaultdict(list)
        i = 0
        while i <= len(words) - n:
            ng = tuple(words[i:i+n])
            if ng in ngrams and ngrams[ng][-1] < i - n:
                # Remove this n-gram occurrence
                del words[i:i+n]
                continue
            ngrams[ng].append(i)
            i += 1
        return ' '.join(words)
    text = remove_repeated_ngrams(text, n=5)

    # Final cleanup: collapse spaces, ensure single period at end
    text = re.sub(r'\s{2,}', ' ', text).strip()
    if text and text[-1] not in ".!?":
        text += "."
    return text