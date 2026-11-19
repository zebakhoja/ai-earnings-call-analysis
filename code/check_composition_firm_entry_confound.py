"""
Robustness check for H1's category-composition finding (and, as a secondary
check, the specificity-classifier trend): is the post-2022 rise in generic
`general_ai` language (and, on the classifier, exploratory language) a
genuine within-firm shift, or an artefact of new, less-practised firms
joining the MSCI World sample over time and diluting the average?

Design: restrict the quarterly category-share series to a "continuously
present" subsample of firms -- present in at least MIN_QUARTERS of the 34
calendar quarters in the 2018-2026 study window -- and recompute the same
composition trend used in analyze_temporal_trend.py on this fixed
sub-population only. If the same qualitative pattern (general_ai rising,
generative_ai peaking then falling) survives restricting to firms that were
already established in the sample, new-firm entry cannot be the sole driver.

A secondary, more intuitive cut is also reported: firms present in both 2018
and 2026 (i.e. span the full window, gaps allowed in between).

Run:
    python check_composition_firm_entry_confound.py --panel "../Data/ai_volume_panel.csv"
"""

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

CHATGPT_SHOCK_DATE = pd.Timestamp("2022-11-30")
MIN_QUARTERS = 30

CATEGORIES = [
    "general_ai", "infrastructure_ai", "predictive_analytical_ai",
    "generative_ai", "autonomous_ai",
]


def build_category_share_series(df: pd.DataFrame) -> pd.DataFrame:
    cat_cols = [f"n_ai_mentions_{c}" for c in CATEGORIES]
    agg = df.groupby("calendar_quarter")[cat_cols].sum().reset_index()
    total = agg[cat_cols].sum(axis=1)
    for c, col in zip(CATEGORIES, cat_cols):
        agg[f"share_{c}"] = agg[col] / total.replace(0, pd.NA)
    agg["quarter_start"] = agg["calendar_quarter"].dt.to_timestamp()
    return agg.sort_values("quarter_start").reset_index(drop=True)


def summarize(label: str, agg: pd.DataFrame):
    first, last = agg.iloc[0], agg.iloc[-1]
    print(f"\n--- {label} (n_quarters={len(agg)}) ---")
    for c in CATEGORIES:
        col = f"share_{c}"
        print(f"  {c}: {first[col]*100:.1f}% (start) -> {last[col]*100:.1f}% (end)")
    peak = agg.loc[agg["share_generative_ai"].idxmax()]
    print(f"  generative_ai peak: {peak['share_generative_ai']*100:.1f}% "
          f"at {peak['calendar_quarter']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default="../Data/ai_volume_panel.csv")
    ap.add_argument("--outdir", default="../Data")
    args = ap.parse_args()

    df = pd.read_csv(args.panel, parse_dates=["date"])
    df = df[(df["date"] >= "2018-01-01") & (df["date"] <= "2026-06-30")]
    df["calendar_quarter"] = df["date"].dt.to_period("Q")

    full = build_category_share_series(df)
    summarize("Full sample (all firms, all quarters)", full)

    q_counts = df.groupby("symbol")["calendar_quarter"].nunique()
    continuous_firms = q_counts[q_counts >= MIN_QUARTERS].index
    n_continuous = len(continuous_firms)
    n_total = df["symbol"].nunique()
    df_continuous = df[df["symbol"].isin(continuous_firms)]
    continuous_series = build_category_share_series(df_continuous)
    summarize(
        f"Continuously-present firms only (>= {MIN_QUARTERS}/34 quarters, "
        f"n={n_continuous}/{n_total} firms)",
        continuous_series,
    )

    present_2018 = set(df[df["date"].dt.year == 2018]["symbol"])
    present_2026 = set(df[df["date"].dt.year == 2026]["symbol"])
    spanning_firms = present_2018 & present_2026
    df_spanning = df[df["symbol"].isin(spanning_firms)]
    spanning_series = build_category_share_series(df_spanning)
    summarize(
        f"Firms present in both 2018 and 2026 (n={len(spanning_firms)}/{n_total} firms)",
        spanning_series,
    )

    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.plot(full["quarter_start"], full["share_general_ai"],
            label="general_ai -- full sample", color="tab:blue", linestyle="-")
    ax.plot(continuous_series["quarter_start"], continuous_series["share_general_ai"],
            label=f"general_ai -- continuously present (n={n_continuous})",
            color="tab:blue", linestyle="--")
    ax.plot(full["quarter_start"], full["share_generative_ai"],
            label="generative_ai -- full sample", color="tab:red", linestyle="-")
    ax.plot(continuous_series["quarter_start"], continuous_series["share_generative_ai"],
            label=f"generative_ai -- continuously present (n={n_continuous})",
            color="tab:red", linestyle="--")
    ax.axvline(CHATGPT_SHOCK_DATE, color="black", linestyle=":", label="ChatGPT release")
    ax.set_xlabel("Calendar quarter")
    ax.set_ylabel("Share of total AI mentions")
    ax.set_title("Category composition: full sample vs. continuously-present firms only")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{args.outdir}/h1_composition_firm_entry_check.png", dpi=150)
    print(f"\nSaved {args.outdir}/h1_composition_firm_entry_check.png")


if __name__ == "__main__":
    main()
