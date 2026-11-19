"""
Dictionary validation: draw the two samples needed to report precision and
recall against a manually (LLM-assisted, human-spot-checked) annotated
sample, per the outline's methods commitment.

Precision sample: 150 rows from ai_documents.csv (already-flagged snippets),
stratified across the 5 taxonomy categories so the smallest category
(autonomous_ai) isn't drowned out by general_ai.

Recall sample: 150 random INDIVIDUAL SENTENCES drawn uniformly from the
full, unfiltered transcript corpus (every section of every transcript,
regardless of whether the dictionary fired) via single-pass reservoir
sampling -- avoids a first pass just to count total sentences. For each
sampled sentence, the RAW dictionary matcher (before any co-occurrence
filters) is run directly on that sentence alone, recording whether it
fired -- this checks vocabulary coverage specifically, deliberately
separate from the infrastructure_ai/Bard/Watson/Claude co-occurrence
filters, which have their own already-documented precision/recall
tradeoff (see AI Terms Dictionary and Extraction Methodology.docx, 5b/5c).

Run:
    python 01_sample_for_validation.py
"""

import random
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, "..")
from ai_terms_dictionary import AI_TERMS
from extract_ai_documents import build_matchers, find_matches, iter_transcripts, split_sentences

N_PRECISION = 150
N_RECALL = 150
RANDOM_SEED = 42

DATA_DIR = Path("../../Data/fmp_transcripts")
OUT_PRECISION = Path("../../Data/dv_precision_sample.csv")
OUT_RECALL = Path("../../Data/dv_recall_sample.csv")


def build_precision_sample():
    df = pd.read_csv("../../Data/ai_documents.csv")
    categories = df["category"].unique()
    per_cat = max(1, N_PRECISION // len(categories))

    rng = random.Random(RANDOM_SEED)
    parts = []
    for cat in categories:
        sub = df[df["category"] == cat]
        n = min(len(sub), per_cat)
        parts.append(sub.sample(n=n, random_state=RANDOM_SEED))

    sample = pd.concat(parts)
    if len(sample) < N_PRECISION:
        remaining = df.drop(sample.index)
        top_up = remaining.sample(n=min(len(remaining), N_PRECISION - len(sample)), random_state=RANDOM_SEED)
        sample = pd.concat([sample, top_up])

    sample = sample.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    out = sample[["snippet_id", "symbol", "year", "period", "section", "category",
                  "specificity", "matched_term", "snippet_text"]].copy()
    out["is_genuinely_ai"] = ""
    out["notes"] = ""
    out.to_csv(OUT_PRECISION, index=False)
    print(f"Wrote {len(out)} precision-sample rows to {OUT_PRECISION}")
    print(out["category"].value_counts())


def build_recall_sample():
    ci_pattern, ci_lookup, cs_pattern, cs_lookup = build_matchers()
    rng = random.Random(RANDOM_SEED + 1)

    reservoir = []
    n_seen = 0

    for fp, doc in iter_transcripts(DATA_DIR):
        symbol = doc.get("symbol", "")
        year = doc.get("year", "")
        period = doc.get("period", "")

        for section in ("prepared_remarks", "qa"):
            text = doc.get(section) or ""
            if not text:
                continue
            spans = split_sentences(text)
            for start, end in spans:
                sentence = text[start:end].strip()
                if len(sentence) < 15:
                    continue
                n_seen += 1
                item = (symbol, year, period, section, sentence)
                if len(reservoir) < N_RECALL:
                    reservoir.append(item)
                else:
                    j = rng.randint(0, n_seen - 1)
                    if j < N_RECALL:
                        reservoir[j] = item

    print(f"Reservoir-sampled {len(reservoir)} sentences from {n_seen} total sentences seen")

    rows = []
    for symbol, year, period, section, sentence in reservoir:
        matches = find_matches(sentence, ci_pattern, ci_lookup, cs_pattern, cs_lookup)
        dictionary_fired = len(matches) > 0
        matched_terms = ", ".join(sorted({m[2] for m in matches})) if matches else ""
        rows.append({
            "symbol": symbol, "year": year, "period": period, "section": section,
            "sentence": sentence, "dictionary_fired": dictionary_fired,
            "matched_terms": matched_terms,
            "is_genuinely_ai": "", "notes": "",
        })

    out = pd.DataFrame(rows).sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    out.to_csv(OUT_RECALL, index=False)
    print(f"Wrote {len(out)} recall-sample rows to {OUT_RECALL}")
    print(f"Dictionary fired on {out['dictionary_fired'].sum()}/{len(out)} sampled sentences "
          f"(base rate check, not yet the recall estimate)")


if __name__ == "__main__":
    build_precision_sample()
    build_recall_sample()
