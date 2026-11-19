"""
Parallelized wrapper around extract_ai_documents.py's per-file extraction
logic. The single-threaded script (extract_ai_documents.py) is unchanged and
remains the canonical, documented reference implementation -- this script
just fans the same per-file logic out across worker processes, since each
transcript file is processed independently (no cross-file state), and the
machine has idle cores sitting unused during a single-threaded run.

Prints progress every PROGRESS_EVERY files, unlike the single-threaded
script, specifically so a long run is visibly alive rather than silent
until completion.

Run:
    python extract_ai_documents_parallel.py --data-dir "../Data/fmp_transcripts" \
        --output "../Data/ai_documents.csv" --sentences 1 --max-chars 1000 --workers 8
"""

import argparse
import csv
import json
import time
from multiprocessing import Pool
from pathlib import Path

from extract_ai_documents import (
    build_matchers, build_qualifier_matcher, find_matches, build_speaker_index,
    split_sentences, filter_infra_noise, filter_ambiguous_terms, sentence_window,
    load_date_corrections, apply_date_correction,
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

        matches = find_matches(
            text, _STATE["ci_pattern"], _STATE["ci_lookup"],
            _STATE["cs_pattern"], _STATE["cs_lookup"],
        )
        if not matches:
            continue

        speaker_index = build_speaker_index(text)
        sentence_spans = split_sentences(text)
        matches = filter_infra_noise(
            matches, text, sentence_spans, n_sentences, max_chars,
            _STATE["qualifier_ci_pattern"], _STATE["qualifier_cs_pattern"],
        )
        matches = filter_ambiguous_terms(matches, text, sentence_spans, n_sentences, max_chars)

        for i, (start, end, term, category, specificity) in enumerate(matches):
            snippet = sentence_window(text, sentence_spans, start, n_sentences, max_chars)
            speaker = None
            for pos, name in speaker_index:
                if pos > start:
                    break
                speaker = name
            rows.append({
                "symbol": symbol, "year": year, "period": period, "date": date,
                "section": section, "speaker": speaker or "",
                "matched_term": term, "category": category, "specificity": specificity,
                "match_start": start, "match_end": end,
                "snippet_id": f"{symbol}_{year}_{period}_{section}_{i}",
                "snippet_text": snippet,
            })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="../Data/fmp_transcripts")
    ap.add_argument("--output", default="../Data/ai_documents.csv")
    ap.add_argument("--sentences", type=int, default=1)
    ap.add_argument("--max-chars", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    fieldnames = [
        "symbol", "year", "period", "date", "section", "speaker",
        "matched_term", "category", "specificity",
        "match_start", "match_end", "snippet_id", "snippet_text",
    ]

    files = sorted(Path(args.data_dir).glob("*/*.json"))
    print(f"{len(files)} transcript files to process across {args.workers} workers")

    t0 = time.time()
    n_rows = 0
    n_files_done = 0
    tasks = [(fp, args.sentences, args.max_chars) for fp in files]

    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
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
    print(f"Wrote {n_rows} AI-term documents to {args.output} in {elapsed:.0f}s")


if __name__ == "__main__":
    main()
