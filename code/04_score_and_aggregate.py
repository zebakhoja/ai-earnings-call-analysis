"""
04_score_and_aggregate.py

Applies the trained specificity classifier to the full snippet corpus, then
aggregates snippet-level scores up to firm-quarter -- the level H1's temporal
plot and H2's cross-sectional comparisons actually run on.

Produces:
    - ai_snippets_scored.csv: every snippet with a specificity_score (0/1/2)
      and specificity_label (categorical) attached
    - firm_quarter_specificity_panel.csv: one row per firm-quarter with:
        - snippet_count (volume, itself a variable per our earlier discussion)
        - mean_specificity (continuous, 0-2 scale)
        - share_operational (share of snippets scored 2 -- the H1 headline measure)
        - share_exploratory (share scored 0)
        - same three broken out by section (prepared_remarks vs qa) for H3 prep

Usage:
    python 04_score_and_aggregate.py
"""

import pandas as pd
import joblib
from sentence_transformers import SentenceTransformer
from specificity_common import load_standardized_corpus

INPUT_PATH = "../../Data/ai_documents.csv"
MODEL_PATH = "../../Data/specificity_classifier.joblib"
SCORED_SNIPPETS_OUT = "../../Data/ai_snippets_scored.csv"
FIRM_QUARTER_OUT = "../../Data/firm_quarter_specificity_panel.csv"
TEXT_COL = "text"
SECTION_COL = "section"
BATCH_SIZE = 256

LABEL_NAMES = {0: "exploratory", 1: "mixed", 2: "operational"}


def main():
    print("Loading model and corpus...")
    saved = joblib.load(MODEL_PATH)
    clf = saved["classifier"]
    embedder = SentenceTransformer(saved["embed_model_name"])

    df = load_standardized_corpus(INPUT_PATH)
    print(f"Scoring {len(df)} snippets...")

    embeddings = embedder.encode(
        df[TEXT_COL].tolist(), batch_size=BATCH_SIZE, show_progress_bar=True
    )
    df["specificity_score"] = clf.predict(embeddings)
    df["specificity_label"] = df["specificity_score"].map(LABEL_NAMES)

    proba = clf.predict_proba(embeddings)
    for i, cls in enumerate(clf.classes_):
        df[f"prob_{LABEL_NAMES[cls]}"] = proba[:, i]

    df.to_csv(SCORED_SNIPPETS_OUT, index=False)
    print(f"Saved scored snippets to {SCORED_SNIPPETS_OUT}")

    print("\nAggregating to firm-quarter panel...")

    def agg_block(g):
        n = len(g)
        return pd.Series({
            "snippet_count": n,
            "mean_specificity": g["specificity_score"].mean(),
            "share_operational": (g["specificity_score"] == 2).mean(),
            "share_mixed": (g["specificity_score"] == 1).mean(),
            "share_exploratory": (g["specificity_score"] == 0).mean(),
        })

    group_cols = ["ticker", "firm_name", "quarter"]
    group_cols = [c for c in group_cols if c in df.columns]

    panel = df.groupby(group_cols).apply(agg_block).reset_index()

    if SECTION_COL in df.columns:
        section_panel = (
            df.groupby(group_cols + [SECTION_COL])
            .apply(agg_block)
            .reset_index()
            .pivot_table(index=group_cols, columns=SECTION_COL,
                         values=["snippet_count", "mean_specificity", "share_operational"])
        )
        section_panel.columns = [f"{a}__{b}" for a, b in section_panel.columns]
        section_panel = section_panel.reset_index()
        panel = panel.merge(section_panel, on=group_cols, how="left")

        pr_col = "mean_specificity__prepared_remarks"
        qa_col = "mean_specificity__qa"
        if pr_col in panel.columns and qa_col in panel.columns:
            panel["prepared_qa_specificity_gap"] = panel[pr_col] - panel[qa_col]

    panel.to_csv(FIRM_QUARTER_OUT, index=False)
    print(f"Saved firm-quarter panel ({len(panel)} rows) to {FIRM_QUARTER_OUT}")
    print("\nColumns in panel:")
    print(list(panel.columns))


if __name__ == "__main__":
    main()
