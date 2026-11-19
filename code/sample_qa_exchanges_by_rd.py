"""
Pulls a stratified sample of AI-flagged Q&A snippets, by R&D-intensity
tercile, and recovers the full analyst-question / management-answer
exchange each one sits inside (not just the isolated matched snippet),
to qualitatively check the selection-into-questions hypothesis raised
against the H3 register x capacity result: do high-R&D firms get asked
different (less probing / more incidental) AI questions than low-R&D
firms, in a way that could produce the observed wrong-signed interaction
mechanically rather than behaviourally?

Speaker turns are labelled either "Name - Affiliation:" or just "Name:"
depending on transcript vintage -- both are handled. Role (management vs
analyst) is inferred by matching each Q&A speaker's name against the set
of names who delivered prepared remarks on the same call (management, by
construction); anyone in Q&A not in that set is treated as an analyst.
This is a heuristic, not a trained classifier, but is far more reliable
than guessing from affiliation text alone once affiliation isn't present.

Run:
    python sample_qa_exchanges_by_rd.py --n-per-tercile 15
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

TURN_RE = re.compile(r"(?m)^([A-Z][^\n:]{2,80}?)(?: - ([^\n:]{2,80}))?:\s*")


def load_rd_terciles():
    acct = pd.read_csv("../Data/firm_accounting_panel_combined.csv")
    firm = acct.groupby("ticker").agg(mean_rd_intensity=("rd_intensity", "mean")).reset_index()
    cap = firm["mean_rd_intensity"].quantile(0.95)
    firm["mean_rd_intensity"] = firm["mean_rd_intensity"].clip(upper=cap)
    firm["rd_tercile"] = pd.qcut(firm["mean_rd_intensity"], 3, labels=["Low", "Mid", "High"])
    return firm[["ticker", "mean_rd_intensity", "rd_tercile"]]


def split_turns(text: str):
    """Returns list of (speaker, affiliation_or_None, turn_text)."""
    matches = list(TURN_RE.finditer(text))
    turns = []
    for i, m in enumerate(matches):
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        turns.append((m.group(1).strip(), (m.group(2) or "").strip(), text[start:end].strip()))
    return turns


def management_names(prepared_remarks_text: str):
    return {speaker for speaker, _, _ in split_turns(prepared_remarks_text)}


def find_exchange(ticker, year, period, snippet_text):
    path = Path(f"../Data/fmp_transcripts/{ticker}/{ticker}_{year}_{period}.json")
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    qa_text = d.get("qa", "")
    turns = split_turns(qa_text)
    mgmt_names = management_names(d.get("prepared_remarks", ""))

    idx = next((i for i, (_, _, text) in enumerate(turns) if snippet_text[:60] in text), None)
    if idx is None:
        frag = snippet_text[:40]
        idx = next((i for i, (_, _, text) in enumerate(turns) if frag in text), None)
    if idx is None:
        frag = snippet_text[:25]
        idx = next((i for i, (_, _, text) in enumerate(turns) if frag in text), None)
    if idx is None:
        return None

    lo = max(0, idx - 1)
    hi = min(len(turns), idx + 2)
    return turns[lo:hi], idx - lo, mgmt_names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-tercile", type=int, default=None,
                     help="Exact count to draw from each tercile (overrides --total).")
    ap.add_argument("--total", type=int, default=45,
                     help="Total sample size, split as evenly as possible across the 3 terciles.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", default="../Data/qa_exchange_sample.txt")
    args = ap.parse_args()

    snippets = pd.read_csv("../Data/ai_snippets_scored.csv")
    snippets = snippets[snippets["section"] == "qa"].copy()

    rd = load_rd_terciles()
    snippets = snippets.merge(rd, on="ticker", how="inner")
    snippets = snippets.dropna(subset=["rd_tercile"])

    tercile_order = ["Low", "Mid", "High"]
    if args.n_per_tercile is not None:
        per_tercile = {t: args.n_per_tercile for t in tercile_order}
    else:
        base, rem = divmod(args.total, 3)
        per_tercile = {t: base + (1 if i < rem else 0) for i, t in enumerate(tercile_order)}

    def _sample(g):
        n = per_tercile[g.name]
        return g.sample(min(n, len(g)), random_state=args.seed)

    sampled = (
        snippets.groupby("rd_tercile", observed=True, group_keys=False)
        .apply(_sample, include_groups=False)
    )
    sampled = sampled.join(snippets[["rd_tercile"]], how="left")
    print(f"Sampled {len(sampled)} Q&A snippets across terciles: "
          f"{sampled['rd_tercile'].value_counts().to_dict()}")

    lines = []
    n_found = 0
    for _, row in sampled.iterrows():
        result = find_exchange(row["ticker"], row["year"], row["period"], row["text"])
        lines.append(
            f"\n{'='*100}\n"
            f"TICKER={row['ticker']}  {row['year']}_{row['period']}  "
            f"RD_TERCILE={row['rd_tercile']} (mean_rd={row['mean_rd_intensity']:.3f})  "
            f"specificity_score={row['specificity_score']} ({row['specificity_label']})\n"
            f"MATCHED SNIPPET: {row['text']!r}\n"
        )
        if result is None:
            lines.append("  [exchange not recovered -- snippet not matched into a turn]\n")
            continue
        turns, matched_idx, mgmt_names = result
        n_found += 1
        for i, (speaker, affil, text) in enumerate(turns):
            tag = "  >>> MATCHED TURN >>>" if i == matched_idx else ""
            role = "MANAGEMENT" if speaker in mgmt_names else "ANALYST(?)"
            affil_str = f" - {affil}" if affil else ""
            lines.append(f"\n  [{role}] {speaker}{affil_str}:{tag}\n  {text[:600]}")

    print(f"Recovered full exchange for {n_found}/{len(sampled)} snippets")

    with open(args.output, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Wrote sample to {args.output}")


if __name__ == "__main__":
    main()
