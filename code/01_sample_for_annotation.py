"""
01_sample_for_annotation.py

Draws a stratified sample of AI-keyword snippets for manual annotation on the
exploratory <-> operational specificity axis (H1's compositional shift claim).

Stratification variables:
    - period: pre / post Nov 2022 (ChatGPT shock) -- avoids the seed set being
      dominated by one era's discourse style
    - sector: GICS sector -- NOT YET AVAILABLE (WRDS Compustat/GICS pull is a
      separate, still-outstanding task -- see specificity_common.py). Falls
      back to period-only stratification until that data exists.

Output: a CSV ready for labeling in 02_annotate.py. Also carves off a small
held-out sample that is NOT used for training -- this is the independent
validation set for the holdout kappa check.

Usage:
    python 01_sample_for_annotation.py
"""

import pandas as pd
import numpy as np
from specificity_common import load_standardized_corpus

INPUT_PATH = "../../Data/ai_documents.csv"
TRAIN_SAMPLE_PATH = "../../Data/specificity_annotation_train.csv"
HOLDOUT_SAMPLE_PATH = "../../Data/specificity_annotation_holdout.csv"
N_TRAIN = 200
N_HOLDOUT = 80
RANDOM_SEED = 42
CHATGPT_SHOCK_DATE = "2022-11-30"
SECTOR_COL = "gics_sector"
DATE_COL = "call_date"

def stratified_draw(df, n, seed):
    n_strata = df["stratum"].nunique()
    per_stratum = max(1, n // n_strata)
    drawn = (
        df.groupby("stratum", group_keys=False)
        .apply(lambda g: g.sample(n=min(len(g), per_stratum), random_state=seed))
    )
    if len(drawn) < n:
        remaining = df.drop(drawn.index)
        top_up = remaining.sample(n=min(len(remaining), n - len(drawn)), random_state=seed)
        drawn = pd.concat([drawn, top_up])
    return drawn.sample(frac=1, random_state=seed).reset_index(drop=True)


def main():
    df = load_standardized_corpus(INPUT_PATH)
    df["period"] = np.where(df[DATE_COL] < CHATGPT_SHOCK_DATE, "pre_chatgpt", "post_chatgpt")

    if SECTOR_COL in df.columns:
        df[SECTOR_COL] = df[SECTOR_COL].fillna("unknown_sector")
        df["stratum"] = df["period"] + "__" + df[SECTOR_COL].astype(str)
    else:
        print(f"NOTE: '{SECTOR_COL}' not available yet (WRDS/GICS pull outstanding) "
              "-- stratifying by period only.")
        df["stratum"] = df["period"]

    print(f"Corpus size: {len(df)} snippets across {df['stratum'].nunique()} strata")

    train_sample = stratified_draw(df, N_TRAIN, RANDOM_SEED)
    remaining_df = df.drop(train_sample.index)
    holdout_sample = stratified_draw(remaining_df, N_HOLDOUT, RANDOM_SEED + 1)

    out_cols = ["snippet_id", "ticker", "firm_name", "quarter", DATE_COL,
                "section", "speaker_role", "period", SECTOR_COL, "text"]
    out_cols = [c for c in out_cols if c in df.columns]

    for name, sample, path in [
        ("train", train_sample, TRAIN_SAMPLE_PATH),
        ("holdout", holdout_sample, HOLDOUT_SAMPLE_PATH),
    ]:
        out = sample[out_cols].copy()
        out["label"] = ""
        out["annotator_notes"] = ""
        out.to_csv(path, index=False)
        print(f"Wrote {len(out)} {name} snippets to {path}")

    print("\nStratum coverage (train):")
    print(train_sample.groupby(["period"]).size() if SECTOR_COL not in df.columns
          else train_sample.groupby(["period", SECTOR_COL]).size())


if __name__ == "__main__":
    main()
