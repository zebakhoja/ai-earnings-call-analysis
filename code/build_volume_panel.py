"""
Build a call-level AI-discourse volume/density panel.

`ai_documents.csv` (from extract_ai_documents.py) is mention-level and only
contains rows for transcript-sections that had at least one AI-term match --
it has no row at all for a transcript-section with zero mentions. Volume and
density analysis needs a denominator (total sentence count) for EVERY
transcript-section, including the zero-mention ones, so this script
re-iterates the full transcript corpus and produces one row per
(symbol, year, period, section) regardless of hit count.

Reuses the exact matching + infrastructure_ai co-occurrence filter from
extract_ai_documents.py (build_matchers, build_qualifier_matcher,
find_matches, filter_infra_noise, split_sentences), so mention counts here
are guaranteed consistent with ai_documents.csv.

Run:
    python build_volume_panel.py --data-dir "../Data/fmp_transcripts" \
        --output "../Data/ai_volume_panel.csv"
"""

import argparse
import csv
from pathlib import Path

from ai_terms_dictionary import (
    GENERAL_AI, INFRASTRUCTURE_AI, PREDICTIVE_ANALYTICAL_AI,
    GENERATIVE_AI, AUTONOMOUS_AI, GENERIC, SPECIFIC,
)
from extract_ai_documents import (
    build_matchers, build_qualifier_matcher, find_matches, filter_infra_noise,
    filter_ambiguous_terms, split_sentences, iter_transcripts,
)

CATEGORIES = [GENERAL_AI, INFRASTRUCTURE_AI, PREDICTIVE_ANALYTICAL_AI, GENERATIVE_AI, AUTONOMOUS_AI]

FIELDNAMES = (
    ["symbol", "year", "period", "date", "section", "n_sentences", "n_ai_mentions_total"]
    + [f"n_ai_mentions_{c}" for c in CATEGORIES]
    + ["n_ai_mentions_generic", "n_ai_mentions_specific", "ai_density"]
)


def build_panel_rows(data_dir: Path, n_sentences: int, max_chars: int):
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

            sentence_spans = split_sentences(text)
            n_sent = len(sentence_spans)
            if n_sent == 0:
                continue

            matches = find_matches(text, ci_pattern, ci_lookup, cs_pattern, cs_lookup)
            matches = filter_infra_noise(
                matches, text, sentence_spans, n_sentences, max_chars,
                qualifier_ci_pattern, qualifier_cs_pattern,
            )
            matches = filter_ambiguous_terms(matches, text, sentence_spans, n_sentences, max_chars)

            cat_counts = {c: 0 for c in CATEGORIES}
            spec_counts = {GENERIC: 0, SPECIFIC: 0}
            for _start, _end, _term, category, specificity in matches:
                cat_counts[category] += 1
                spec_counts[specificity] += 1

            n_total = len(matches)
            row = {
                "symbol": symbol,
                "year": year,
                "period": period,
                "date": date,
                "section": section,
                "n_sentences": n_sent,
                "n_ai_mentions_total": n_total,
                "n_ai_mentions_generic": spec_counts[GENERIC],
                "n_ai_mentions_specific": spec_counts[SPECIFIC],
                "ai_density": n_total / n_sent,
            }
            for c in CATEGORIES:
                row[f"n_ai_mentions_{c}"] = cat_counts[c]

            yield row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="../Data/fmp_transcripts")
    ap.add_argument("--output", default="../Data/ai_volume_panel.csv")
    ap.add_argument("--sentences", type=int, default=1,
                     help="must match the --sentences used for ai_documents.csv, "
                          "since it feeds the infra co-occurrence filter's snippet window")
    ap.add_argument("--max-chars", type=int, default=1000)
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    output_path = Path(args.output)

    n_rows = 0
    total_mentions = 0
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in build_panel_rows(data_dir, args.sentences, args.max_chars):
            writer.writerow(row)
            n_rows += 1
            total_mentions += row["n_ai_mentions_total"]

    print(f"Wrote {n_rows} transcript-section rows to {output_path}")
    print(f"Total AI mentions across panel: {total_mentions}")


if __name__ == "__main__":
    main()
