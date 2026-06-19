"""
utils/text_cleaner.py
Cleans and normalises raw extracted text before segmentation.
Steps:
  1. Collapse excessive whitespace / newlines
  2. Remove control characters
  3. Fix common OCR artefacts (ligatures, broken hyphens)
  4. Strip headers/footers patterns (page numbers, URLs)
  5. Normalise unicode
"""

import re
import unicodedata
import logging

try:
    from fastcoref import FCoref
    HAS_FASTCOREF = True
except ImportError:
    HAS_FASTCOREF = False

logger = logging.getLogger(__name__)

_CORE_MODEL = None # Lazy load


# ── Regex patterns ────────────────────────────────────────────────────────────
_CONTROL_CHARS   = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_MULTI_NEWLINE   = re.compile(r"\n{3,}")
_MULTI_SPACE     = re.compile(r"[ \t]{2,}")
_BROKEN_HYPHEN   = re.compile(r"(\w)-\n(\w)")   # word- \n word → wordword
_PAGE_NUMBER     = re.compile(r"^\s*\d+\s*$", re.MULTILINE)
_URL_PATTERN     = re.compile(r"https?://\S+|www\.\S+")
_PHONE_PATTERN   = re.compile(r"(?<!\w)(?:\+?\d[\d\-\s().]{7,}\d)(?!\w)")
_ONLY_PUNCT      = re.compile(r"^[\W_]+$")
_REPEAT_CHAR     = re.compile(r"(.)\1{6,}")
_SPACE_AROUND_PUNCT = re.compile(r"\s+([?.!,;:])")
# Only insert a space after punctuation when followed by 2+ uppercase-starting word
# (avoids breaking abbreviations like U.S., e.g., Fig.2, etc.)
_MISSING_SPACE_AFTER_PUNCT = re.compile(r"([?,!;:])([A-Z][a-z])")
_NAV_LINE = re.compile(
    r"(?i)^\s*(back to|return to|go to|skip to|home|about|contact|privacy|terms|cookies|sitemap|search|menu|login|sign in|register|subscribe)\b.*$"
)
_FOOTER_LINE = re.compile(
    r"(?i)^\s*(copyright|all rights reserved|powered by|designed by|follow us|share|newsletter)\b.*$"
)
_TEAM_META_LINE = re.compile(
    r"(?i)^\s*(team members?|members?|contributors?|author[s]?|prepared by|supervisor|mentor|faculty|department|institution|university|college)\b.*$"
)
_BULLET_LIST_META = re.compile(r"(?i)^\s*[-*•]\s*(name|roll|id|reg(istration)?|batch|class|section)\b")
_CODE_FENCE = re.compile(r"^\s*(```|~~~)")
_CODEISH = re.compile(r"[{}[\]<>]|(::)|(#include\b)|(\b(def|class|import|from|return|public|private|static|void|int|float|double|printf|cout|System\.out)\b)")
_BROKEN_FORMULA = re.compile(r"[=+\-*/^]{2,}|[∑∫√∞≈≠≤≥]")

# Common OCR ligature fixes
_LIGATURES = {
    "\ufb01": "fi",
    "\ufb02": "fl",
    "\ufb00": "ff",
    "\ufb03": "ffi",
    "\ufb04": "ffl",
}

# ── camelCase compound masking ─────────────────────────────────────────────────────
# Well-known compound tech words that must NOT be split by the camelCase rule.
_KNOWN_COMPOUNDS = re.compile(
    r'\b(JavaScript|TypeScript|GitHub|GitLab|LinkedIn|YouTube|MySQL|MongoDB|'
    r'PostgreSQL|WordPress|OpenAI|ChatGPT|NumPy|PyTorch|TensorFlow|ReactJS|'
    r'NextJS|NodeJS|VueJS|AngularJS|FastAPI|GraphQL|RestAPI|DevOps|FullStack|'
    r'BackEnd|FrontEnd|SaaS|PaaS|IaaS|HTML5|CSS3)\b',
    re.IGNORECASE,
)
_CAMELCASE_SPLIT = re.compile(r'([a-z]{4,})([A-Z][a-z]+)')

class CompoundMasker:
    """Helper to mask and unmask known tech compounds during cleaning."""
    def __init__(self):
        self.table: dict[str, str] = {}
        self.counter = 0

    def mask(self, text: str) -> str:
        def _repl(m: re.Match) -> str:
            key = f"__CC{self.counter}__"
            self.table[key] = m.group(0)
            self.counter += 1
            return key
        return _KNOWN_COMPOUNDS.sub(_repl, text)

    def unmask(self, text: str) -> str:
        for key, val in self.table.items():
            text = text.replace(key, val)
        return text

# ── Digit / alpha merge split ───────────────────────────────────────────────────
_ALPHA_DIGIT = re.compile(r'([A-Za-z]{3,})(\d{2,})')
_DIGIT_ALPHA = re.compile(r'(\d{2,})([A-Za-z]{3,})')

# ── OCR line-joining — universal linguistic signals ───────────────────────────
# These patterns use ZERO content-specific knowledge.
# They fire on structural/grammatical cues that are valid across any input.

# Previous line has no sentence-terminal punctuation at its end
_NO_TERMINAL = re.compile(r'[^.!?:;]\s*$')
# Previous line ends with exactly one uppercase letter + period (abbreviation half)
# (?<=[A-Z]) lookbehind avoids the \b issue with "the U." (space before U)
_ENDS_ABBREV = re.compile(r'(?<=[A-Z])\.\s*$')
# Current line STARTS with a single uppercase letter + period (abbreviation second half)
# e.g. "S. between Jan..." or just "S." alone
_IS_ABBREV_FRAG = re.compile(r'^\s*[A-Z]\.')
# Words that can legitimately START a new sentence even when short
_SENTENCE_STARTERS = re.compile(
    r'^(I|He|She|They|We|It|This|That|These|Those|There|Here|'
    r'My|His|Her|Our|Their|Its|The|A|An)\b',
    re.IGNORECASE,
)
# Words that CANNOT start a standalone sentence — always a mid-sentence join
_CONTINUATION_START = re.compile(
    r'^(and|or|but|nor|yet|so|between|among|from|to|with|without|'
    r'including|such as|as well as|along with|in addition|'
    r'whereas|although|however|therefore|thus|hence|consequently|'
    r'furthermore|moreover|nevertheless|meanwhile|otherwise|'
    r'since|until|unless|because|if|though|even though|'
    r'as well|as a result|in contrast|on the other hand)\b',
    re.IGNORECASE,
)


def _join_ocr_lines(text: str) -> str:
    """
    General-purpose OCR line-joining using 6 universal linguistic signals.
    No content-specific patterns. Works for resumes, papers, slides, PDFs.

    Signal 1 — No terminal punctuation on previous line
        Previous line did not end with . ! ? : ;
        The sentence is unfinished; next line continues it.

    Signal 2 — Lowercase start on current line
        Current line begins with a lowercase letter.
        Cannot be the start of a new English sentence.

    Signal 3 — Continuation-word start on current line
        Starts with: and, or, between, including, however, therefore…
        These words mid-sentence must stay attached.

    Signal 4 — Abbreviation split (both halves detected)
        Prev ends "U."  AND  current starts "S." → rejoin as "U.S."
        Catches Ph.D., U.K., U.S.A., etc. automatically.

    Signal 5 — Orphan abbreviation fragment
        Current line IS entirely "S." (single letter + period alone).
        Always a fragment — attach regardless of previous line.

    Signal 6 — Short dangling line after non-terminal previous line
        Current is ≤ 3 words, previous has no terminal punctuation,
        and current is not a pronoun/article-started new sentence.
    """
    lines = text.splitlines()
    merged: list[str] = []

    for line in lines:
        stripped = line.strip()

        if not merged:
            merged.append(stripped)
            continue

        prev = merged[-1]

        if not prev or not stripped:
            merged.append(stripped)
            continue

        # ── Compute signals ───────────────────────────────────────────────────
        prev_no_terminal  = bool(_NO_TERMINAL.search(prev))
        prev_ends_abbrev  = bool(_ENDS_ABBREV.search(prev))
        curr_lower_start  = stripped[0].islower()
        curr_continuation = bool(_CONTINUATION_START.match(stripped))
        curr_abbrev_frag  = bool(_IS_ABBREV_FRAG.match(stripped))
        curr_words        = stripped.split()
        curr_short_orphan = (
            len(curr_words) <= 3
            and prev_no_terminal
            and not _SENTENCE_STARTERS.match(stripped)
        )

        should_merge = (
            (prev_no_terminal and curr_lower_start)      # Signals 1+2
            or (prev_no_terminal and curr_continuation)  # Signals 1+3
            or (prev_ends_abbrev and curr_abbrev_frag)   # Signal 4
            or curr_abbrev_frag                          # Signal 5
            or curr_short_orphan                         # Signal 6
        )

        if should_merge:
            # Abbreviation halves join without a space; everything else with one
            sep = "" if (prev_ends_abbrev and curr_abbrev_frag) else " "
            merged[-1] = prev.rstrip() + sep + stripped
        else:
            merged.append(stripped)

    return "\n".join(line for line in merged if line)


_NOISE_PHRASES = [
    # hotline / support boilerplate
    r"\b(hotline|helpline|help line|support line|crisis line)\b",
    r"\b(if you are in crisis|if you need help|call\s+911|contact\s+support)\b",
    r"\b(suicide prevention|samaritans|mental health)\b",
    r"\b(for confidential (support|help|assistance|advice|guidance))\b",
    # legal / disclaimer boilerplate
    r"\b(disclaimer|terms\s+of\s+use|terms\s+and\s+conditions|privacy\s+policy)\b",
    r"\b(all rights reserved|copyright\s*\u00a9|do not distribute|confidential)\b",
    # marketing / footer boilerplate
    r"\b(unsubscribe|manage preferences|view in browser)\b",
    # navigation boilerplate
    r"\b(back to (home|top)|return to page|next page|previous page|page \d+ of \d+)\b",
    # slide footer patterns
    r"\b(for more (information|details)|visit (http|www)|see (http|www))\b",
    r"\b(developer guide|docs|documentation|reference|manual)\s*[|·]\b",
]
_NOISE_LINE_PATTERN = re.compile("|".join(_NOISE_PHRASES), re.IGNORECASE)


def _non_alnum_ratio(s: str) -> float:
    if not s:
        return 1.0
    non = sum(1 for ch in s if not ch.isalnum() and not ch.isspace())
    return non / max(1, len(s))


def _is_obvious_ocr_garbage(line: str) -> bool:
    s = line.strip()
    if not s:
        return True
    if _ONLY_PUNCT.match(s):
        return True
    if _REPEAT_CHAR.search(s):
        return True
    # Detect table of contents / page number lines
    if re.match(r'^\d+(\.\d+)*\s+[A-Z]', s):  # "2.1 Women's Workforce"
        return True
    if re.match(r'^(Page|p\.|pp\.)\s*\d+', s, re.IGNORECASE):  # "Page 2"
        return True
    if _non_alnum_ratio(s) > 0.35 and len(s) > 20:
        return True
    tokens = s.split()
    if len(tokens) >= 8:
        short_tokens = sum(1 for t in tokens if len(t) <= 2)
        if short_tokens / len(tokens) > 0.65:
            return True
    return False


def _looks_like_code_fragment(line: str) -> bool:
    s = line.strip()
    if not s:
        return False
    if _CODE_FENCE.match(s):
        return True
    # short, dense punctuation / braces / keywords = likely code
    if _CODEISH.search(s) and _non_alnum_ratio(s) > 0.20:
        return True
    # broken formulas without any natural language context
    if _BROKEN_FORMULA.search(s) and len(re.findall(r"\b[a-zA-Z]{3,}\b", s)) < 3:
        return True
    return False


def _line_is_url_or_phone_heavy(line: str) -> bool:
    s = line.strip()
    if not s:
        return False
    # Detect lines that are primarily URLs, references, or navigation
    url_hits = len(re.findall(r"https?://\S+|www\.\S+", s))
    phone_hits = 1 if _PHONE_PATTERN.search(s) else 0
    
    # Hard reject: Lines starting with "For more information/details" followed by URL/visit
    if re.match(r"^For more (information|details|info)", s, re.IGNORECASE):
        return True
    
    # Slide footer patterns: "For more info, visit...", "See: http..."
    if re.search(r"\b(for (more|the)|visit|see|check out|refer to|available at)\b.*\b(http|www|guide|docs?|manual)", s, re.IGNORECASE):
        return True
    if url_hits >= 1:  # Changed from 2 to 1 - reject any line with URLs
        return True
    if phone_hits and len(s) < 140:
        return True
    if (url_hits or phone_hits) and _non_alnum_ratio(s) > 0.20:
        return True
    return False


def _filter_noise_lines(text: str) -> tuple[str, dict]:
    stats = {
        "removed_lines_total": 0,
        "removed_by_rule": {
            "page_number": 0,
            "website_footer_noise": 0,
            "metadata_lines": 0,
            "url_or_phone_heavy": 0,
            "malformed_ocr_or_code": 0,
        },
        "rules_added": [
            "noise_phrase (hotline/support/disclaimer/footer patterns)",
            "website_footer_noise (navigation/footer UI lines)",
            "metadata_lines (team/faculty/institution boilerplate)",
            "url_or_phone_heavy (repeated URLs, phone/contact lines)",
            "malformed_ocr_or_code (symbol-heavy OCR garbage / broken code/formula fragments)",
        ],
    }

    lines = text.splitlines()
    kept = []
    for line in lines:
        if not line.strip():
            kept.append(line)
            continue

        if _PAGE_NUMBER.match(line.strip()):
            stats["removed_lines_total"] += 1
            stats["removed_by_rule"]["page_number"] += 1
            continue

        # navigation / footer UI boilerplate
        if _NAV_LINE.match(line) or _FOOTER_LINE.match(line) or _NOISE_LINE_PATTERN.search(line):
            stats["removed_lines_total"] += 1
            stats["removed_by_rule"]["website_footer_noise"] += 1
            continue

        # team/member/institution metadata (usually not central to technical content)
        if _TEAM_META_LINE.match(line) or _BULLET_LIST_META.match(line):
            stats["removed_lines_total"] += 1
            stats["removed_by_rule"]["metadata_lines"] += 1
            continue

        if _line_is_url_or_phone_heavy(line):
            stats["removed_lines_total"] += 1
            stats["removed_by_rule"]["url_or_phone_heavy"] += 1
            continue

        # broken OCR / code-like fragments
        if _looks_like_code_fragment(line) or _is_obvious_ocr_garbage(line):
            stats["removed_lines_total"] += 1
            stats["removed_by_rule"]["malformed_ocr_or_code"] += 1
            continue

        kept.append(line)

    return "\n".join(kept), stats


def clean_text(text: str, return_stats: bool = False, resolve_coref: bool = False):
    """Full cleaning pipeline. Returns cleaned string (and optional stats)."""
    if not text:
        return ("", {"removed_lines_total": 0, "removed_by_rule": {}, "rules_added": []}) if return_stats else ""

    # ── 0. Coreference Resolution (Optional, high overhead) ───────────────
    if resolve_coref and HAS_FASTCOREF:
        global _CORE_MODEL
        try:
            if _CORE_MODEL is None:
                logger.info("Loading FastCoref model...")
                _CORE_MODEL = FCoref()
            logger.info("Resolving coreferences in document...")
            preds = _CORE_MODEL.predict(texts=[text])
            # TODO: Implement actual coreference replacement if needed
        except Exception as e:
            logger.warning("Coreference resolution failed: %s", e)

    # Additional normalization: fix unicode dashes/quotes, remove BOM, normalize ellipsis
    text = text.replace('\ufeff', '')  # Remove BOM
    text = text.replace('\u2013', '-')  # En dash
    text = text.replace('\u2014', '-')  # Em dash
    text = text.replace('\u2018', "'").replace('\u2019', "'")  # Single quotes
    text = text.replace('\u201c', '"').replace('\u201d', '"')  # Double quotes
    text = text.replace('…', '...')  # Ellipsis

    # Remove bracketed references (e.g., [1], (see Fig. 2))
    text = re.sub(r'\[\d+\]', '', text)
    text = re.sub(r'\(see [^)]+\)', '', text, flags=re.IGNORECASE)

    # Remove repeated punctuation (e.g., "!!!", "??")
    text = re.sub(r'([!?.,])\1{2,}', r'\1', text)

    # Remove stray non-ASCII characters
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)

    # Remove extra spaces before punctuation
    text = re.sub(r'\s+([?.!,;:])', r'\1', text)

    # Remove leading/trailing whitespace on each line
    text = '\n'.join([line.strip() for line in text.splitlines()])

    # Unicode normalisation (NFC)
    text = unicodedata.normalize("NFC", text)

    # Fix ligatures
    for lig, replacement in _LIGATURES.items():
        text = text.replace(lig, replacement)

    # Remove control characters
    text = _CONTROL_CHARS.sub(" ", text)

    # Fix broken hyphenated words split across lines
    text = _BROKEN_HYPHEN.sub(r"\1\2", text)

    # ── OCR line-joining: MUST run before noise filter and segmenter ──────────
    # Tesseract splits mid-sentence and mid-abbreviation at column/line boundaries.
    # This merges fragments like "...the U." + "S. between..." → "...the U.S. between..."
    text = _join_ocr_lines(text)
    logger.debug("OCR line joining complete.")

    # Pre-ranking noise filter (line-based) before we destruct URLs/phones
    text, noise_stats = _filter_noise_lines(text)

    # Remove URLs (inline remnants)
    text = _URL_PATTERN.sub("", text)

    # Remove phone numbers (inline remnants)
    text = _PHONE_PATTERN.sub("", text)

# ── OCR Formatting Fixes
    # 1. Fix common OCR spacing errors
    text = re.sub(r'\b(wo) (men)\b', r'\1\2', text)  # wo men -> women
    text = re.sub(r'\b(une) (mployed)\b', r'\1\2', text)  # une mployed -> unemployed
    text = re.sub(r'\b(peopl) (e)\b', r'\1\2', text)  # peopl e -> people
    text = re.sub(r'\b(immi) (gration)\b', r'\1\2', text)  # immi gration -> immigration
    text = re.sub(r'\b(speak) (ing)\b', r'\1\2', text)  # speak ing -> speaking
    text = re.sub(r'\b(Tertia) (ry)\b', r'\1\2', text)  # Tertia ry -> Tertiary
    text = re.sub(r'\b(De) (mographic)\b', r'\1\2', text)  # De mographic -> Demographic
    text = re.sub(r'\b(Wo) (rkforce)\b', r'\1\2', text)  # Wo rkforce -> Workforce
    text = re.sub(r'\b(Forc) (e)\b', r'\1\2', text)  # Forc e -> Force
    text = re.sub(r'\b(Austral) (ian)\b', r'\1\2', text)  # Austral ian -> Australian
    
    # 2. Separate heading-like prefixes into their own line
    text = re.sub(
        r'(?m)^(?P<h>[A-Za-z][A-Za-z0-9 /&()\-\u2013\u2014]{2,60}):\s*(?P<b>\S)',
        r'\g<h>:\n\g<b>',
        text,
    )
    
    # 2. Repair camelCase merged words from bad block OCR (e.g. "ThisIs" -> "This Is")
    # Mask known compound tech words so they are never split, then restore after.
    masker = CompoundMasker()
    text = masker.mask(text)
    text = _CAMELCASE_SPLIT.sub(r'\1 \2', text)
    text = masker.unmask(text)

    # 3. Split digit/alpha merges only when both sides have >=3 chars
    # (avoids splitting "U.S.", "v2", "4G", "Apr22", dates, versions)
    text = _ALPHA_DIGIT.sub(r'\1 \2', text)
    text = _DIGIT_ALPHA.sub(r'\1 \2', text)

    # 4. Fix missing spacing around punctuation
    text = _SPACE_AROUND_PUNCT.sub(r"\1", text)
    text = _MISSING_SPACE_AFTER_PUNCT.sub(r"\1 \2", text)

    # Collapse whitespace
    text = _MULTI_NEWLINE.sub("\n\n", text)
    text = _MULTI_SPACE.sub(" ", text)

    cleaned = text.strip()
    logger.debug("Text cleaned: %d → %d characters.", len(text), len(cleaned))
    if return_stats:
        return cleaned, noise_stats
    return cleaned