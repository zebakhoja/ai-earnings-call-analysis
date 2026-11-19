"""
Empirical pass to expand/validate the seed AI-terms dictionary.

Logic:
1. Load a sample of transcripts (default: all of them, or pass --sample N to
   randomly subsample for speed).
2. Find every occurrence of a seed term from ai_terms_dictionary.AI_TERMS.
3. Pull the words immediately surrounding each hit (a small window, default 4
   words each side).
4. Count the most frequent unigrams/bigrams appearing near AI-term hits that
   are NOT already in the dictionary. These are candidates for terms the seed
   list missed (e.g. "foundation model", "AI copilot", specific vendor names).
5. Print a ranked candidate list for manual review -- this is a human-in-the-loop
   step, not something to auto-add to the dictionary.

Run:
    python explore_ai_term_candidates.py --data-dir "../Data/fmp_transcripts" --sample 300
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from ai_terms_dictionary import AI_TERMS, CASE_SENSITIVE_TERMS

STOPWORDS = {
    "the", "a", "an", "of", "to", "in", "and", "or", "is", "are", "was", "were",
    "for", "on", "with", "as", "we", "our", "us", "that", "this", "it", "be",
    "have", "has", "had", "will", "would", "can", "could", "at", "by", "from",
    "about", "into", "not", "so", "but", "you", "your", "i", "they", "their",
    "these", "those", "if", "which", "what", "how", "more", "some", "than",
}


def build_pattern(terms, case_sensitive_terms):
    """Compile one regex alternation per case-sensitivity bucket."""
    ci_terms = [t for t in terms if t not in case_sensitive_terms]
    cs_terms = [t for t in terms if t in case_sensitive_terms]

    def escape_hyphen_variants(t):
        return re.escape(t).replace(r"\-", "[- ]")

    patterns = []
    if ci_terms:
        alt = "|".join(escape_hyphen_variants(t) for t in sorted(ci_terms, key=len, reverse=True))
        patterns.append(re.compile(rf"\b(?:{alt})\b", re.IGNORECASE))
    if cs_terms:
        alt = "|".join(escape_hyphen_variants(t) for t in sorted(cs_terms, key=len, reverse=True))
        patterns.append(re.compile(rf"\b(?:{alt})\b"))
    return patterns


def all_seed_terms():
    return {term for term, _category, _specificity in AI_TERMS}


def iter_transcripts(data_dir: Path, sample: int = None):
    files = sorted(data_dir.glob("*/*.json"))
    if sample:
        import random
        random.seed(0)
        files = random.sample(files, min(sample, len(files)))
    for fp in files:
        try:
            with open(fp, encoding="utf-8") as f:
                yield fp, json.load(f)
        except (json.JSONDecodeError, OSError):
            continue


def tokenize(text: str):
    return re.findall(r"[A-Za-z][A-Za-z\-']*", text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="../Data/fmp_transcripts")
    ap.add_argument("--sample", type=int, default=None, help="randomly subsample N transcripts")
    ap.add_argument("--window", type=int, default=4, help="words each side of a hit to inspect")
    ap.add_argument("--top", type=int, default=40, help="how many candidate terms to print")
    args = ap.parse_args()

    seed_terms = all_seed_terms()
    patterns = build_pattern(seed_terms, CASE_SENSITIVE_TERMS)

    unigram_counts = Counter()
    bigram_counts = Counter()
    n_hits = 0
    n_transcripts_with_hit = 0

    data_dir = Path(args.data_dir)
    for fp, doc in iter_transcripts(data_dir, args.sample):
        text = doc.get("content", "")
        if not text:
            continue
        found_here = False
        for pattern in patterns:
            for m in pattern.finditer(text):
                found_here = True
                n_hits += 1
                start = max(0, m.start() - 200)
                end = min(len(text), m.end() + 200)
                window_text = text[start:end]
                tokens = [t.lower() for t in tokenize(window_text) if t.lower() not in STOPWORDS and len(t) > 2]
                unigram_counts.update(tokens)
                bigram_counts.update(" ".join(bg) for bg in zip(tokens, tokens[1:]))
        if found_here:
            n_transcripts_with_hit += 1

    print(f"Transcripts scanned: {n_hits} hits across {n_transcripts_with_hit} transcripts\n")

    seed_lower = {t.lower() for t in seed_terms}

    print(f"--- Top {args.top} candidate UNIGRAMS not already in dictionary ---")
    shown = 0
    for word, count in unigram_counts.most_common():
        if word in seed_lower:
            continue
        print(f"{count:6d}  {word}")
        shown += 1
        if shown >= args.top:
            break

    print(f"\n--- Top {args.top} candidate BIGRAMS not already in dictionary ---")
    shown = 0
    for phrase, count in bigram_counts.most_common():
        if phrase in seed_lower:
            continue
        print(f"{count:6d}  {phrase}")
        shown += 1
        if shown >= args.top:
            break


if __name__ == "__main__":
    main()
