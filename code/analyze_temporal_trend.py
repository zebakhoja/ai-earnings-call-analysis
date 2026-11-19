"""
H1 (temporal evolution / GPT diffusion): does AI discourse rise over
2018-2026, with a structural break around the November 2022 ChatGPT shock?

A single pooled "mentions per sentence" density hides two theoretically
distinct margins, plus the technology-taxonomy composition:

1. EXTENSIVE margin: the % of calls that mention AI at all, per quarter.
   This is the diffusion-across-firms measure (an adoption S-curve, in the
   spirit of Jovanovic & Rousseau 2005) -- closer to what "GPT diffusion"
   means than a pooled density number, since it answers "how many firms have
   joined the conversation," not "how much is said in total."
2. INTENSIVE margin: mean AI mentions per call, CONDITIONAL on the call
   mentioning AI at all, per quarter. This is a depth-of-engagement measure
   among firms already discussing AI -- closer to the J-curve idea of
   deepening complementary investment/discussion once adopted.
3. CATEGORY SHARE: each taxonomy category's (general/infrastructure/
   predictive/generative/autonomous) share of total AI mentions, per
   quarter -- the direct operationalisation of the outline's "technology
   taxonomy" analysis (tracking the infrastructure -> generative transition
   as firms traverse the J-curve).

Sanity check behind this design: total sentence counts per quarter grew
~6x over 2018-2024, but that's almost entirely a coverage effect (unique
companies in the panel grew ~5x over the same period) -- average sentences
per company-quarter barely moved (492 -> 545, ~11%). So a pooled "per
sentence" density isn't distorted by anything odd, but it also doesn't
distinguish "more firms talking" from "existing firms talking more," which
is exactly the extensive/intensive split above is for.

Logic:
1. Load ai_volume_panel.csv (one row per transcript-section, including
   zero-mention rows). Bucket by CALENDAR quarter (not each company's own
   fiscal quarter, which doesn't align across the MSCI World universe).
2. Collapse both sections to one row per company-quarter (sum
   n_ai_mentions_total across prepared_remarks + qa) to compute the
   extensive/intensive margins at the call level.
3. Aggregate category-level mention sums per quarter (across both sections)
   to compute category shares of total volume.
4. Chow test at 2022-11-30 (ChatGPT release) run on the EXTENSIVE margin as
   the primary H1 diffusion measure, plus on pooled density as a secondary
   check -- classic Chow F-test comparing pooled OLS RSS to the sum of
   pre/post-shock RSS.

Run:
    python analyze_temporal_trend.py --panel "../Data/ai_volume_panel.csv" \
        --outdir "../Data"
"""

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
from scipy import stats

CHATGPT_SHOCK_DATE = pd.Timestamp("2022-11-30")

CATEGORIES = [
    "general_ai", "infrastructure_ai", "predictive_analytical_ai",
    "generative_ai", "autonomous_ai",
]


def _load_panel(panel_path: str, min_date: str, max_date: str) -> pd.DataFrame:
    df = pd.read_csv(panel_path, parse_dates=["date"])
    df = df[(df["date"] >= min_date) & (df["date"] <= max_date)]
    df["calendar_quarter"] = df["date"].dt.to_period("Q")
    return df


def build_density_series(df: pd.DataFrame) -> pd.DataFrame:
    agg = df.groupby("calendar_quarter").agg(
        n_ai_mentions_total=("n_ai_mentions_total", "sum"),
        n_sentences=("n_sentences", "sum"),
    ).reset_index()
    agg["ai_density"] = agg["n_ai_mentions_total"] / agg["n_sentences"]
    return _finalize(agg)


def build_margin_series(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse both sections to one row per company-quarter, then compute
    the extensive (% mentioning AI) and intensive (mentions | mentioned)
    margins per calendar quarter."""
    call = df.groupby(["symbol", "year", "period", "calendar_quarter"]).agg(
        n_ai_mentions_total=("n_ai_mentions_total", "sum"),
    ).reset_index()

    def summarize(g):
        mentioned = g["n_ai_mentions_total"] > 0
        extensive = mentioned.mean()
        intensive = g.loc[mentioned, "n_ai_mentions_total"].mean() if mentioned.any() else 0.0
        return pd.Series({"extensive_margin": extensive, "intensive_margin": intensive, "n_calls": len(g)})

    agg = call.groupby("calendar_quarter").apply(summarize, include_groups=False).reset_index()
    return _finalize(agg)


def build_category_share_series(df: pd.DataFrame) -> pd.DataFrame:
    cat_cols = [f"n_ai_mentions_{c}" for c in CATEGORIES]
    agg = df.groupby("calendar_quarter")[cat_cols].sum().reset_index()
    total = agg[cat_cols].sum(axis=1)
    for c, col in zip(CATEGORIES, cat_cols):
        agg[f"share_{c}"] = agg[col] / total.replace(0, pd.NA)
    return _finalize(agg)


def _finalize(agg: pd.DataFrame) -> pd.DataFrame:
    agg["quarter_start"] = agg["calendar_quarter"].dt.to_timestamp()
    agg = agg.sort_values("quarter_start").reset_index(drop=True)
    agg["quarter_index"] = range(len(agg))
    agg["post_shock"] = agg["quarter_start"] >= CHATGPT_SHOCK_DATE
    return agg


def chow_test(agg: pd.DataFrame, value_col: str):
    pre = agg[~agg["post_shock"]]
    post = agg[agg["post_shock"]]

    def fit_ols(sub):
        X = sm.add_constant(sub["quarter_index"])
        return sm.OLS(sub[value_col], X).fit()

    pooled_model = fit_ols(agg)
    pre_model = fit_ols(pre)
    post_model = fit_ols(post)

    n = len(agg)
    k = 2
    numerator = (pooled_model.ssr - (pre_model.ssr + post_model.ssr)) / k
    denominator = (pre_model.ssr + post_model.ssr) / (n - 2 * k)
    f_stat = numerator / denominator
    p_value = 1 - stats.f.cdf(f_stat, k, n - 2 * k)

    return {
        "f_stat": f_stat, "p_value": p_value,
        "n_pre": len(pre), "n_post": len(post),
        "pre_slope": pre_model.params["quarter_index"],
        "post_slope": post_model.params["quarter_index"],
        "pooled_slope": pooled_model.params["quarter_index"],
    }


def plot_series(agg: pd.DataFrame, value_col: str, ylabel: str, title: str, out_path: str):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(agg["quarter_start"], agg[value_col], marker="o")
    ax.axvline(CHATGPT_SHOCK_DATE, color="red", linestyle="--", label="ChatGPT release (Nov 2022)")
    ax.set_xlabel("Calendar quarter")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved plot to {out_path}")


def plot_category_shares(agg: pd.DataFrame, out_path: str):
    fig, ax = plt.subplots(figsize=(11, 5.5))
    share_cols = [f"share_{c}" for c in CATEGORIES]
    ax.stackplot(agg["quarter_start"], *[agg[c] for c in share_cols], labels=CATEGORIES, alpha=0.85)
    ax.axvline(CHATGPT_SHOCK_DATE, color="black", linestyle="--", label="ChatGPT release (Nov 2022)")
    ax.set_xlabel("Calendar quarter")
    ax.set_ylabel("Share of total AI mentions")
    ax.set_title("AI-discourse technology-taxonomy composition, 2018-2026")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved plot to {out_path}")


def print_chow_result(label: str, result: dict):
    print(f"\n--- Chow test at 2022-11-30 ({label}) ---")
    print(f"n_pre={result['n_pre']} quarters, n_post={result['n_post']} quarters")
    print(f"F-statistic = {result['f_stat']:.3f}, p-value = {result['p_value']:.6f}")
    print(f"Pre-shock trend slope:  {result['pre_slope']:.6f}")
    print(f"Post-shock trend slope: {result['post_slope']:.6f}")
    print(f"Pooled (no-break) trend slope: {result['pooled_slope']:.6f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default="../Data/ai_volume_panel.csv")
    ap.add_argument("--outdir", default="../Data")
    ap.add_argument("--min-date", default="2018-01-01")
    ap.add_argument("--max-date", default="2026-06-30")
    args = ap.parse_args()

    df = _load_panel(args.panel, args.min_date, args.max_date)

    density = build_density_series(df)
    margins = build_margin_series(df)
    shares = build_category_share_series(df)

    print("=== Pooled density ===")
    print(density[["calendar_quarter", "n_ai_mentions_total", "n_sentences", "ai_density"]].to_string(index=False))
    print("\n=== Extensive / intensive margins ===")
    print(margins[["calendar_quarter", "n_calls", "extensive_margin", "intensive_margin"]].to_string(index=False))
    print("\n=== Category shares ===")
    print(shares[["calendar_quarter"] + [f"share_{c}" for c in CATEGORIES]].to_string(index=False))

    plot_series(density, "ai_density", "AI mentions per sentence (pooled density)",
                "AI-discourse density in earnings calls, 2018-2026", f"{args.outdir}/h1_temporal_trend.png")
    plot_series(margins, "extensive_margin", "% of calls mentioning AI at all",
                "AI-discourse extensive margin (adoption), 2018-2026", f"{args.outdir}/h1_extensive_margin.png")
    plot_series(margins, "intensive_margin", "Mean AI mentions per call | call mentions AI",
                "AI-discourse intensive margin (depth), 2018-2026", f"{args.outdir}/h1_intensive_margin.png")
    plot_category_shares(shares, f"{args.outdir}/h1_category_shares.png")

    print_chow_result("extensive margin -- primary H1 diffusion measure", chow_test(margins, "extensive_margin"))
    print_chow_result("intensive margin -- secondary depth measure", chow_test(margins, "intensive_margin"))
    print_chow_result("pooled density -- secondary/robustness", chow_test(density, "ai_density"))


if __name__ == "__main__":
    main()
