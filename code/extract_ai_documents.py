"""
Extraction pipeline: turn raw transcripts into AI-term "documents" for NLP.

Logic (per transcript, per section):
1. Load the transcript JSON. Process `prepared_remarks` and `qa` as SEPARATE
   texts -- this is not a convenience split, it's the theoretically motivated
   distinction from the dissertation outline (section 5.3): prepared remarks
   are scripted/management-controlled, Q&A is live/adversarial, so they are
   treated as generating distinct measures rather than pooled.
2. Match every AI term from ai_terms_dictionary.AI_TERMS in the section text,
   using a combined regex per case-sensitivity bucket (longest term first, so
   e.g. "AI agent" consumes its span before the bare "AI" pattern can double-count it).
3. Drop matches whose span is fully contained inside a longer match already kept.
3b. `infrastructure_ai` terms (data center, hybrid cloud, GPU, ...) are also
   used constantly in non-AI contexts (utility power-grid demand, commercial
   real estate, general chip/CPU business) -- a spot check of 25 sampled
   `infrastructure_ai` rows found ~75% were not actually about AI. To fix
   this without discarding the category (tracking infra->generative mentions
   is a named analytical goal), an `infrastructure_ai` match is only kept if
   an AI-qualifying term from another category (general_ai, predictive_
   analytical_ai, generative_ai, autonomous_ai) also appears within its
   sentence window. Matches failing this co-occurrence check are dropped.
3c. Some individual dictionary terms are ambiguous proper nouns or generic
   phrases that collide with unrelated meanings -- e.g. "Bard" matches
   Google's chatbot but ALSO C.R. Bard (a medical-device company BD acquired
   in 2017) and "Bard Glad Pedersen" (an Equinor IR host's first name);
   "Watson"/"Claude" collide with real people's surnames/first names;
   "fine-tuning" collides overwhelmingly with its generic English sense
   ("fine-tuning our assortment/proposition"), not ML model fine-tuning
   (~80% false-positive rate, found via the dictionary validation pass, see
   Dictionary Validation.docx); "NLP" collides with unrelated acronym uses.
   Fixed the same way as 3b: `AMBIGUOUS_TERM_QUALIFIERS` maps such terms to
   a regex of terms that must ALSO appear in the sentence window (e.g.
   "Google", "model", "language") for the match to be kept.
4. For each surviving match, extract a SENTENCE-boundary window around it
   (default: the sentence containing the match, plus 1 full sentence each
   side) -- this window is the "document" used for downstream NLP (sentiment,
   topic modeling). A fixed word-count window was tried first and rejected:
   it regularly truncated mid-sentence, clipping the very clause needed to
   judge whether a mention was substantive or aspirational. Sentence
   boundaries are detected with a lightweight abbreviation-aware regex
   splitter (see `split_sentences`) rather than a fixed word count, so
   snippet length varies but every snippet is grammatically complete.
   A `--max-chars` cap (default 1000) guards against run-on sentences
   producing an unreasonably long snippet.
5. Best-effort tag the speaker of the line containing the match (found via the
   last "Speaker Name: " marker preceding the match in the section text).
   Note: the sentence window is not clipped at speaker-turn boundaries, so a
   snippet can occasionally span the end of one speaker's turn and the start
   of the next (e.g. an operator's introduction bleeding into an analyst's
   first sentence). This is a known simplification.
6. Write one row per match to a CSV, carrying the taxonomy category and
   specificity tag from the dictionary so they're available as columns
   for the substantiveness / H3 analysis without re-deriving them later.

Run:
    python extract_ai_documents.py --data-dir "../Data/fmp_transcripts" \
        --output "../Data/ai_documents.csv" --sentences 1 --max-chars 1000
"""

import argparse
import csv
import json
import re
from pathlib import Path

from ai_terms_dictionary import AI_TERMS, CASE_SENSITIVE_TERMS, INFRASTRUCTURE_AI

SPEAKER_LINE_RE = re.compile(r"(?m)^([A-Z][A-Za-z.\'\-]+(?: [A-Za-z.\'\-]+){0,4}):\s")

SENTENCE_END_RE = re.compile(r"[.!?]+(?=\s|$)")
TRAILING_QUOTE_RE = re.compile(r"^[\"'”\)\]]+")

ABBREVIATIONS = {
    "mr", "mrs", "ms", "dr", "prof", "st", "vs", "etc", "inc", "co", "corp",
    "ltd", "jr", "sr", "no", "fig", "u.s", "u.k", "e.g", "i.e", "approx",
    "govt", "dept", "gen", "rep", "sen", "gov", "vol", "op", "cf", "al",
}

AMBIGUOUS_TERM_QUALIFIERS = {
    "Bard": re.compile(
        r"\b(Google|Alphabet|Gemini|PaLM|Sundar|Bing|ChatGPT|chatbot|"
        r"generative AI|conversational AI|large language model|LLM)\b",
        re.IGNORECASE,
    ),
    "Watson": re.compile(
        r"\bIBM\b|Watson Health|cognitive comput\w*|AI capabilit\w*|"
        r"artificial intelligence capabilit\w*",
        re.IGNORECASE,
    ),
    "Claude": re.compile(
        r"\bAnthropic\b|large language model|\bLLM\b|chatbot|generative AI|"
        r"AI model|AI assistant",
        re.IGNORECASE,
    ),
    "fine-tuning": re.compile(
        r"\bmodel\b|\bmodels\b|\bLLM\b|neural network|machine learning|"
        r"language model|\balgorithm|training data|generative AI|AI model",
        re.IGNORECASE,
    ),
    "NLP": re.compile(
        r"\blanguage\b|natural language|\btext\b|\bmodel\b|machine learning|"
        r"\bAI\b",
        re.IGNORECASE,
    ),
}

_AUTONOMOUS_AI_GENERIC_QUALIFIER = re.compile(
    r"\bAI\b|artificial intelligence|machine learning|large language "
    r"model|\bLLM\b|generative AI|neural network|\bagentic\b|"
    r"AI[- ]powered|AI model|AI agent",
    re.IGNORECASE,
)
for _term in (
    "autonomous system", "autonomous systems", "autonomous agent",
    "autonomous agents", "robotic process automation",
    "intelligent automation", "cognitive automation",
):
    AMBIGUOUS_TERM_QUALIFIERS[_term] = _AUTONOMOUS_AI_GENERIC_QUALIFIER


def _normalize_key(text: str, case_sensitive: bool) -> str:
    key = text.replace("-", " ")
    key = re.sub(r"\s+", " ", key).strip()
    return key if case_sensitive else key.lower()


def _escape_hyphen_variants(term: str) -> str:
    return re.escape(term).replace(r"\-", "[- ]")


def _compile_bucket(terms):
    """terms: list of (term, category, specificity). Returns (ci_pattern,
    ci_lookup, cs_pattern, cs_lookup) split by case-sensitivity, longest term
    first within each bucket."""
    ci_terms = [(t, c, s) for t, c, s in terms if t not in CASE_SENSITIVE_TERMS]
    cs_terms = [(t, c, s) for t, c, s in terms if t in CASE_SENSITIVE_TERMS]

    ci_terms.sort(key=lambda row: len(row[0]), reverse=True)
    cs_terms.sort(key=lambda row: len(row[0]), reverse=True)

    ci_pattern = None
    cs_pattern = None
    if ci_terms:
        alt = "|".join(_escape_hyphen_variants(t) for t, _, _ in ci_terms)
        ci_pattern = re.compile(rf"\b(?:{alt})\b", re.IGNORECASE)
    if cs_terms:
        alt = "|".join(_escape_hyphen_variants(t) for t, _, _ in cs_terms)
        cs_pattern = re.compile(rf"\b(?:{alt})\b")

    ci_lookup = {_normalize_key(t, False): (t, c, s) for t, c, s in ci_terms}
    cs_lookup = {_normalize_key(t, True): (t, c, s) for t, c, s in cs_terms}

    return ci_pattern, ci_lookup, cs_pattern, cs_lookup


def build_matchers():
    """Compile one alternation regex per case-sensitivity bucket, plus a
    normalized-key -> (term, category, specificity) lookup per bucket, over
    the full AI_TERMS dictionary."""
    return _compile_bucket(AI_TERMS)


def build_qualifier_matcher():
    """Same as build_matchers, but restricted to terms OUTSIDE
    infrastructure_ai -- used to check whether an infrastructure_ai match is
    accompanied by a genuine AI-qualifying term in its sentence window."""
    qualifying_terms = [(t, c, s) for t, c, s in AI_TERMS if c != INFRASTRUCTURE_AI]
    ci_pattern, _, cs_pattern, _ = _compile_bucket(qualifying_terms)
    return ci_pattern, cs_pattern


def has_ai_qualifier(text: str, qualifier_ci_pattern, qualifier_cs_pattern) -> bool:
    if qualifier_ci_pattern is not None and qualifier_ci_pattern.search(text):
        return True
    if qualifier_cs_pattern is not None and qualifier_cs_pattern.search(text):
        return True
    return False


def find_matches(text: str, ci_pattern, ci_lookup, cs_pattern, cs_lookup):
    """Return a de-duplicated list of (start, end, term, category, specificity),
    sorted by position. Matches fully contained inside a longer match are dropped."""
    raw = []
    if ci_pattern is not None:
        for m in ci_pattern.finditer(text):
            key = _normalize_key(m.group(0), False)
            term, category, specificity = ci_lookup[key]
            raw.append((m.start(), m.end(), term, category, specificity))
    if cs_pattern is not None:
        for m in cs_pattern.finditer(text):
            key = _normalize_key(m.group(0), True)
            term, category, specificity = cs_lookup[key]
            raw.append((m.start(), m.end(), term, category, specificity))

    raw.sort(key=lambda row: (row[0], -(row[1] - row[0])))

    kept = []
    for start, end, term, category, specificity in raw:
        if kept and start < kept[-1][1]:
            continue
        kept.append((start, end, term, category, specificity))
    return kept


def build_speaker_index(text: str):
    """List of (char_position, speaker_name) for each 'Speaker: ' line marker,
    in order -- used to look up who was speaking at a given match position."""
    return [(m.start(), m.group(1)) for m in SPEAKER_LINE_RE.finditer(text)]


def speaker_at(position: int, speaker_index):
    speaker = ""
    for pos, name in speaker_index:
        if pos > position:
            break
        speaker = name
    return speaker


def split_sentences(text: str):
    """Lightweight abbreviation-aware sentence splitter. Returns a list of
    (start, end) character spans covering the whole text (no gaps)."""
    spans = []
    start = 0
    for m in SENTENCE_END_RE.finditer(text):
        punct_start, punct_end = m.start(), m.end()

        preceding_word = re.findall(r"[A-Za-z]+$", text[start:punct_start])
        last_word = preceding_word[0].lower() if preceding_word else ""
        if last_word in ABBREVIATIONS:
            continue

        next_pos = punct_end
        trail = TRAILING_QUOTE_RE.match(text[next_pos:])
        if trail:
            next_pos += trail.end()

        rest = text[next_pos:].lstrip()
        if rest and not (rest[0].isupper() or rest[0] in "\"'“"):
            continue

        spans.append((start, next_pos))
        start = next_pos

    if start < len(text):
        spans.append((start, len(text)))

    return [(s, e) for s, e in spans if text[s:e].strip()]


def sentence_window(text: str, sentence_spans, start: int, n_sentences: int, max_chars: int):
    idx = 0
    for i, (s, e) in enumerate(sentence_spans):
        if s <= start:
            idx = i
        else:
            break

    lo = max(0, idx - n_sentences)
    hi = min(len(sentence_spans) - 1, idx + n_sentences)
    char_start = sentence_spans[lo][0]
    char_end = sentence_spans[hi][1]
    snippet = text[char_start:char_end].strip()

    if len(snippet) > max_chars:
        center = min(max(start - char_start, 0), len(snippet))
        half = max_chars // 2
        lo_c = max(0, center - half)
        hi_c = min(len(snippet), center + half)
        snippet = snippet[lo_c:hi_c].strip()

    return snippet


def filter_infra_noise(matches, text, sentence_spans, n_sentences, max_chars,
                        qualifier_ci_pattern, qualifier_cs_pattern):
    """Drop infrastructure_ai matches lacking a nearby AI-qualifying term (see
    module docstring, point 3b). Shared by extract_documents and the volume
    panel builder so both apply exactly the same filtering rule."""
    kept = []
    for start, end, term, category, specificity in matches:
        if category == INFRASTRUCTURE_AI:
            snippet = sentence_window(text, sentence_spans, start, n_sentences, max_chars)
            if not has_ai_qualifier(snippet, qualifier_ci_pattern, qualifier_cs_pattern):
                continue
        kept.append((start, end, term, category, specificity))
    return kept


def filter_ambiguous_terms(matches, text, sentence_spans, n_sentences, max_chars):
    """Drop matches on terms in AMBIGUOUS_TERM_QUALIFIERS (e.g. "Bard") unless
    their term-specific qualifier regex also appears in the sentence window.
    See module docstring point 3c."""
    kept = []
    for start, end, term, category, specificity in matches:
        qualifier_re = AMBIGUOUS_TERM_QUALIFIERS.get(term)
        if qualifier_re is not None:
            snippet = sentence_window(text, sentence_spans, start, n_sentences, max_chars)
            if not qualifier_re.search(snippet):
                continue
        kept.append((start, end, term, category, specificity))
    return kept


DATE_CORRECTIONS_PATH = Path(__file__).parent / "../Data/date_corrections.csv"


def load_date_corrections(path=DATE_CORRECTIONS_PATH):
    """(ticker, year, period) -> corrected date string, for the 410 transcripts
    whose raw `date` field is corrupted (see fix_transcript_dates.py). year/
    period are reliable even when date isn't, so these are looked up by
    (ticker, year, period), not by the bad date itself."""
    corrections = {}
    if not Path(path).exists():
        return corrections
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = (row["ticker"], int(row["year"]), row["period"])
            corrections[key] = row["corrected_date"]
    return corrections


def apply_date_correction(doc: dict, corrections: dict) -> dict:
    symbol, year, period = doc.get("symbol"), doc.get("year"), doc.get("period")
    try:
        year = int(year)
    except (TypeError, ValueError):
        return doc
    fixed = corrections.get((symbol, year, period))
    if fixed is not None:
        doc = dict(doc)
        doc["date"] = fixed
    return doc


def iter_transcripts(data_dir: Path):
    corrections = load_date_corrections()
    for fp in sorted(data_dir.glob("*/*.json")):
        try:
            with open(fp, encoding="utf-8") as f:
                doc = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        yield fp, apply_date_correction(doc, corrections)


def extract_documents(data_dir: Path, n_sentences: int, max_chars: int):
    ci_pattern, ci_lookup, cs_pattern, cs_lookup = build_matchers()
    qualifier_ci_pattern, qualifier_cs_pattern = build_qualifier_matcher()

    for fp, doc in iter_transcripts(data_dir):
        symbol = doc.get("symbol", "")
        year = doc.get("year", "")
        period = doc.get("period", "")
        date = doc.get("date", "")

        for section in ("prepared_remarks", "qa"):
            text = doc.get(section) or ""
            if not text:
                continue

            matches = find_matches(text, ci_pattern, ci_lookup, cs_pattern, cs_lookup)
            if not matches:
                continue

            speaker_index = build_speaker_index(text)
            sentence_spans = split_sentences(text)
            matches = filter_infra_noise(
                matches, text, sentence_spans, n_sentences, max_chars,
                qualifier_ci_pattern, qualifier_cs_pattern,
            )
            matches = filter_ambiguous_terms(matches, text, sentence_spans, n_sentences, max_chars)

            for i, (start, end, term, category, specificity) in enumerate(matches):
                snippet = sentence_window(text, sentence_spans, start, n_sentences, max_chars)
                speaker = speaker_at(start, speaker_index)
                yield {
                    "symbol": symbol,
                    "year": year,
                    "period": period,
                    "date": date,
                    "section": section,
                    "speaker": speaker,
                    "matched_term": term,
                    "category": category,
                    "specificity": specificity,
                    "match_start": start,
                    "match_end": end,
                    "snippet_id": f"{symbol}_{year}_{period}_{section}_{i}",
                    "snippet_text": snippet,
                }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="../Data/fmp_transcripts")
    ap.add_argument("--output", default="../Data/ai_documents.csv")
    ap.add_argument("--sentences", type=int, default=1, help="full sentences each side of a hit")
    ap.add_argument("--max-chars", type=int, default=1000, help="hard cap on snippet length")
    args = ap.parse_args()

    fieldnames = [
        "symbol", "year", "period", "date", "section", "speaker",
        "matched_term", "category", "specificity",
        "match_start", "match_end", "snippet_id", "snippet_text",
    ]

    data_dir = Path(args.data_dir)
    output_path = Path(args.output)

    n_rows = 0
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in extract_documents(data_dir, args.sentences, args.max_chars):
            writer.writerow(row)
            n_rows += 1

    print(f"Wrote {n_rows} AI-term documents to {output_path}")


if __name__ == "__main__":
    main()
