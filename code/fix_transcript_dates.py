"""
Builds a correction table for the corrupted `date` field found in 410
transcript files (see conversation notes / methodology doc): `year` and
`period` are correct, but `date` is sometimes populated with an unrelated,
often decades-old date -- a bug in the original transcript pull, not a
labeling error. Since most downstream analysis buckets by calendar quarter
derived from `date`, a bad date silently drops that firm-quarter from the
2018-2026 study window entirely (369/410 cases) or misfiles it into the
wrong quarter.

A file's date is flagged "bad" if its year differs from the labeled `year`
field by more than 1 (allows for the normal ~1-quarter fiscal-vs-calendar
offset already present in good rows).

Fix: per ticker, correct a bad row's date by offsetting the nearest GOOD
row for the SAME ticker by (quarter distance x 3 months) -- using each
firm's own reporting cadence rather than a single fixed rule, since fiscal
calendars vary by company. This is far more accurate than a universal
month-mapping rule.

Output: Data/date_corrections.csv (ticker, year, period, corrected_date),
consumed by extract_ai_documents.py / build_volume_panel*.py at read time.

Run:
    python fix_transcript_dates.py
"""

import csv
import json
from pathlib import Path

import pandas as pd

DATA_DIR = Path("../Data/fmp_transcripts")
OUTPUT = Path("../Data/date_corrections.csv")

PERIOD_TO_Q = {"Q1": 1, "Q2": 2, "Q3": 3, "Q4": 4}


def quarter_index(year: int, period: str) -> int:
    return year * 4 + PERIOD_TO_Q[period]


def main():
    by_ticker = {}
    for fp in sorted(DATA_DIR.glob("*/*.json")):
        try:
            with open(fp, encoding="utf-8") as f:
                d = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        symbol = d.get("symbol") or fp.parent.name
        year, period, date = d.get("year"), d.get("period"), d.get("date")
        if not year or period not in PERIOD_TO_Q or not date:
            continue
        try:
            year = int(year)
            date_ts = pd.Timestamp(date)
        except (TypeError, ValueError):
            continue
        by_ticker.setdefault(symbol, []).append(
            {"year": year, "period": period, "date": date_ts,
             "qidx": quarter_index(year, period),
             "good": abs(date_ts.year - year) <= 1}
        )

    corrections = []
    n_bad, n_fixed, n_unfixable = 0, 0, 0
    for symbol, rows in by_ticker.items():
        good = [r for r in rows if r["good"]]
        bad = [r for r in rows if not r["good"]]
        n_bad += len(bad)
        if not good:
            n_unfixable += len(bad)
            continue
        for r in bad:
            nearest = min(good, key=lambda g: abs(g["qidx"] - r["qidx"]))
            offset_quarters = r["qidx"] - nearest["qidx"]
            corrected = nearest["date"] + pd.DateOffset(months=3 * offset_quarters)
            corrections.append({
                "ticker": symbol, "year": r["year"], "period": r["period"],
                "corrected_date": corrected.strftime("%Y-%m-%d"),
                "original_date": r["date"].strftime("%Y-%m-%d"),
                "anchor_year": nearest["year"], "anchor_period": nearest["period"],
            })
            n_fixed += 1

    with open(OUTPUT, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "ticker", "year", "period", "corrected_date",
            "original_date", "anchor_year", "anchor_period",
        ])
        writer.writeheader()
        writer.writerows(corrections)

    print(f"{n_bad} bad (ticker, year, period) rows found")
    print(f"{n_fixed} corrected via same-ticker interpolation")
    print(f"{n_unfixable} could not be fixed (no good quarter for that ticker at all)")
    print(f"Saved {OUTPUT}")


if __name__ == "__main__":
    main()
