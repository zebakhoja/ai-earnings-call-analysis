"""
Discriminates between two competing explanations for H1's rising general_ai
share (Discussion "Growing General AI Talk" section):

  - Signalling/cheap-talk: genericism is a market-wide erosion of "AI" as a
    credible term, so it should rise roughly UNIFORMLY across R&D terciles
    -- the mechanism doesn't depend on a firm's own adoption stage.
  - Timing/adoption-stage: the rise should be CONCENTRATED in low-R&D-
    intensity firms (newer, less mature adopters), with high-R&D firms
    -- already past the exploratory phase -- showing comparatively less
    drift toward genericism.

Re-cuts existing H1/H2 outputs (ai_volume_panel.csv's category counts,
R&D terciles from firm_accounting_panel_combined.csv) rather than requiring
new data collection.

Run:
    python check_general_ai_signalling_vs_timing.py
"""

import numpy as np
import pandas as pd

CATEGORIES = [
    "general_ai", "infrastructure_ai", "predictive_analytical_ai",
    "generative_ai", "autonomous_ai",
]


def load_rd_terciles():
    acct = pd.read_csv("../Data/firm_accounting_panel_rd_imputed.csv")
    firm = acct.groupby("ticker").agg(mean_rd_intensity=("rd_intensity", "mean")).reset_index()
    cap = firm["mean_rd_intensity"].quantile(0.95)
    firm["mean_rd_intensity"] = firm["mean_rd_intensity"].clip(upper=cap)
    firm["rd_tercile"] = pd.qcut(firm["mean_rd_intensity"], 3, labels=["Low", "Mid", "High"])
    return firm[["ticker", "rd_tercile"]]


def main():
    df = pd.read_csv("../Data/ai_volume_panel.csv", parse_dates=["date"])
    df = df[(df["date"] >= "2018-01-01") & (df["date"] <= "2026-06-30")]
    df["calendar_quarter"] = df["date"].dt.to_period("Q")

    rd = load_rd_terciles().rename(columns={"ticker": "symbol"})
    df = df.merge(rd, on="symbol", how="inner")
    df = df.dropna(subset=["rd_tercile"])

    cat_cols = [f"n_ai_mentions_{c}" for c in CATEGORIES]
    agg = df.groupby(["rd_tercile", "calendar_quarter"], observed=True)[cat_cols].sum().reset_index()
    total = agg[cat_cols].sum(axis=1)
    agg["share_general_ai"] = agg["n_ai_mentions_general_ai"] / total.replace(0, pd.NA)

    all_quarters = sorted(agg["calendar_quarter"].unique())
    early_post = all_quarters[:4] if all_quarters[0] >= pd.Period("2022-Q4") else None

    print("General AI share by R&D tercile, start vs end of window:\n")
    results = {}
    for tercile in ["Low", "Mid", "High"]:
        sub = agg[agg["rd_tercile"] == tercile].sort_values("calendar_quarter")
        start_val = sub["share_general_ai"].iloc[:4].mean()
        end_val = sub["share_general_ai"].iloc[-4:].mean()
        delta = end_val - start_val
        results[tercile] = (start_val, end_val, delta)
        print(f"  {tercile:5s} R&D: {start_val*100:.1f}% (start) -> {end_val*100:.1f}% (end), "
              f"delta = {delta*100:+.1f}pp")

    print()
    deltas = {k: v[2] for k, v in results.items()}
    spread = max(deltas.values()) - min(deltas.values())
    print(f"Spread in delta across terciles: {spread*100:.1f}pp")
    if deltas["Low"] > deltas["Mid"] and deltas["Low"] > deltas["High"]:
        print("Low-R&D tercile shows the LARGEST increase -> favours the TIMING explanation.")
    elif spread < 0.05:
        print("Deltas are similar in magnitude across terciles -> favours the SIGNALLING explanation.")
    else:
        print("Pattern is mixed -- does not cleanly favour either explanation.")

    agg.to_csv("../Data/general_ai_share_by_rd_tercile_over_time.csv", index=False)
    print("\nWrote ../Data/general_ai_share_by_rd_tercile_over_time.csv (full quarterly series)")


if __name__ == "__main__":
    main()
