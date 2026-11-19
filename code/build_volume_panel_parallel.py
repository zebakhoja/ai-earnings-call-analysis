"""
Parallelized wrapper around build_volume_panel.py's per-file logic, same
rationale as extract_ai_documents_parallel.py: each transcript file is
processed independently, so fan the work out across worker processes.

Run:
    python build_volume_panel_parallel.py --data-dir "../Data/fmp_transcripts" \
        --output "../Data/ai_volume_panel.csv" --workers 8
"""

import argparse
import csv
import time
import json
from multiprocessing import Pool
from pathlib import Path

from ai_terms_dictionary import (
    GENERAL_AI, INFRASTRUCTURE_AI, PREDICTIVE_ANALYTICAL_AI,
    GENERATIVE_AI, AUTONOMOUS_AI, GENERIC, SPECIFIC,
)
from extract_ai_documents import (
    build_matchers, build_qualifier_matcher, find_matches, filter_infra_noise,
    filter_ambiguous_terms, split_sentences, load_date_corrections, apply_date_correction,
)

CATEGORIES = [GENERAL_AI, INFRASTRUCTURE_AI, PREDICTIVE_ANALYTICAL_AI, GENERATIVE_AI, AUTONOMOUS_AI]
FIELDNAMES = (
    ["symbol", "year", "period", "date", "section", "n_sentences", "n_ai_mentions_total"]
    + [f"n_ai_mentions_{c}" for c in CATEGORIES]
    + ["n_ai_mentions_generic", "n_ai_mentions_specific", "ai_density"]
)

PROGRESS_EVERY = 1000
_STATE = {}


def _init_worker():
    ci_pattern, ci_lookup, cs_pattern, cs_lookup = build_matchers()
    qualifier_ci_pattern, qualifier_cs_pattern = build_qualifier_matcher()
    _STATE.update(
        ci_pattern=ci_pattern, ci_lookup=ci_lookup,
        cs_pattern=cs_pattern, cs_lookup=cs_lookup,
        qualifier_ci_pattern=qualifier_ci_pattern,
        qualifier_cs_pattern=qualifier_cs_pattern,
        date_corrections=load_date_corrections(),
    )


def _process_file(args):
    fp, n_sentences, max_chars = args
    try:
        with open(fp, encoding="utf-8") as f:
            doc = json.load(f)
    except (json.JSONDecodeError, OSError):
        return []
    doc = apply_date_correction(doc, _STATE["date_corrections"])

    symbol = doc.get("symbol", "")
    year = doc.get("year", "")
    period = doc.get("period", "")
    date = doc.get("date", "")

    rows = []
    for section in ("prepared_remarks", "qa"):
        text = doc.get(section) or ""
        if not text:
            continue

        sentence_spans = split_sentences(text)
        n_sent = len(sentence_spans)
        if n_sent == 0:
            continue

        matches = find_matches(
            text, _STATE["ci_pattern"], _STATE["ci_lookup"],
            _STATE["cs_pattern"], _STATE["cs_lookup"],
        )
        matches = filter_infra_noise(
            matches, text, sentence_spans, n_sentences, max_chars,
            _STATE["qualifier_ci_pattern"], _STATE["qualifier_cs_pattern"],
        )
        matches = filter_ambiguous_terms(matches, text, sentence_spans, n_sentences, max_chars)

        cat_counts = {c: 0 for c in CATEGORIES}
        spec_counts = {GENERIC: 0, SPECIFIC: 0}
        for _start, _end, _term, category, specificity in matches:
            cat_counts[category] += 1
            spec_counts[specificity] += 1

        n_total = len(matches)
        row = {
            "symbol": symbol, "year": year, "period": period, "date": date,
            "section": section, "n_sentences": n_sent, "n_ai_mentions_total": n_total,
            "n_ai_mentions_generic": spec_counts[GENERIC],
            "n_ai_mentions_specific": spec_counts[SPECIFIC],
            "ai_density": n_total / n_sent,
        }
        for c in CATEGORIES:
            row[f"n_ai_mentions_{c}"] = cat_counts[c]
        rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="../Data/fmp_transcripts")
    ap.add_argument("--output", default="../Data/ai_volume_panel.csv")
    ap.add_argument("--sentences", type=int, default=1)
    ap.add_argument("--max-chars", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    files = sorted(Path(args.data_dir).glob("*/*.json"))
    print(f"{len(files)} transcript files to process across {args.workers} workers")

    t0 = time.time()
    n_rows = 0
    n_files_done = 0
    tasks = [(fp, args.sentences, args.max_chars) for fp in files]

    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()

        with Pool(processes=args.workers, initializer=_init_worker) as pool:
            for rows in pool.imap_unordered(_process_file, tasks, chunksize=25):
                for row in rows:
                    writer.writerow(row)
                    n_rows += 1
                n_files_done += 1
                if n_files_done % PROGRESS_EVERY == 0:
                    elapsed = time.time() - t0
                    rate = n_files_done / elapsed
                    eta = (len(files) - n_files_done) / rate if rate > 0 else float("nan")
                    print(f"  {n_files_done}/{len(files)} files, {n_rows} rows, "
                          f"{elapsed:.0f}s elapsed, ETA {eta:.0f}s", flush=True)

    elapsed = time.time() - t0
    print(f"Wrote {n_rows} panel rows to {args.output} in {elapsed:.0f}s")


if __name__ == "__main__":
    main()
