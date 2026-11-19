"""
H2 (firm heterogeneity / absorptive capacity): does AI discourse intensity
and specificity vary systematically with sector, firm size, and R&D
intensity?

Builds a firm-level panel (one row per ticker, averaged across all its
quarters -- absorptive capacity is treated as a comparatively persistent
firm characteristic here, not something re-measured quarter to quarter;
a firm-quarter panel with time-varying covariates would be a natural
refinement once fiscal-year-aligned merging is worth the added complexity)
from:
    - ai_volume_panel.csv: AI mention density and extensive margin (%
      of calls mentioning AI at all), collapsed across prepared_remarks/qa.
    - firm_quarter_specificity_panel.csv: mean specificity score and
      share operationally specific, conditional on a quarter having any
      AI mentions (this panel has no zero-mention rows by construction).
    - ticker_sector_mapping.csv: GICS sector, country (100% coverage).
    - firm_accounting_panel.csv: total assets (size proxy) and R&D
      intensity, North America only (80.9% of tickers) -- rd_intensity is
      winsorized at the 95th percentile here (43 firm-years exceeded 100%
      R&D-to-revenue pre-winsorization; see H2 Data Preparation.docx) since
      no winsorization threshold had been chosen yet.

Analysis, per the outline's "cross-sectional heterogeneity" design:
    1. AI density and specificity by GICS sector (descriptive + one-way
       ANOVA / Kruskal-Wallis for a formal sector-difference test).
    2. AI density and specificity regressed on log(size) and R&D intensity
       (North America subset only, since that's the only subset with this
       data) -- OLS with sector fixed effects.

Run:
    python analyze_h2_heterogeneity.py
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

RD_WINSOR_PCTILE = 0.95


def build_firm_panel():
    volume = pd.read_csv("../Data/ai_volume_panel.csv")
    call = volume.groupby(["symbol", "year", "period"]).agg(
        n_ai_mentions_total=("n_ai_mentions_total", "sum"),
        n_sentences=("n_sentences", "sum"),
    ).reset_index()
    call["mentioned"] = call["n_ai_mentions_total"] > 0

    firm_volume = call.groupby("symbol").agg(
        total_mentions=("n_ai_mentions_total", "sum"),
        total_sentences=("n_sentences", "sum"),
        n_calls=("mentioned", "size"),
        n_calls_with_mention=("mentioned", "sum"),
    ).reset_index()
    firm_volume["ai_density"] = firm_volume["total_mentions"] / firm_volume["total_sentences"]
    firm_volume["extensive_margin"] = firm_volume["n_calls_with_mention"] / firm_volume["n_calls"]
    firm_volume = firm_volume.rename(columns={"symbol": "ticker"})

    spec = pd.read_csv("../Data/firm_quarter_specificity_panel.csv")
    firm_spec = spec.groupby("ticker").agg(
        mean_specificity=("mean_specificity", "mean"),
        share_operational=("share_operational", "mean"),
        prepared_qa_specificity_gap=("prepared_qa_specificity_gap", "mean"),
    ).reset_index()

    sector = pd.read_csv("../Data/ticker_sector_mapping.csv")[["ticker", "gics_sector", "country"]]

    acct = pd.read_csv("../Data/firm_accounting_panel_combined.csv")
    acct_firm = acct.groupby("ticker").agg(
        mean_at=("at", "mean"),
        mean_rd_intensity=("rd_intensity", "mean"),
        source=("source", "first"),
    ).reset_index()
    cap = acct_firm["mean_rd_intensity"].quantile(RD_WINSOR_PCTILE)
    n_capped = (acct_firm["mean_rd_intensity"] > cap).sum()
    acct_firm["mean_rd_intensity_winsorized"] = acct_firm["mean_rd_intensity"].clip(upper=cap)
    print(f"R&D intensity winsorized at {RD_WINSOR_PCTILE:.0%} percentile = {cap:.3f}; "
          f"{n_capped} firms capped")
    acct_firm["log_size"] = np.log(acct_firm["mean_at"])

    panel = firm_volume.merge(firm_spec, on="ticker", how="left")
    panel = panel.merge(sector, on="ticker", how="left")
    panel = panel.merge(acct_firm, on="ticker", how="left")
    return panel


def sector_comparison(panel):
    print("\n=== AI density by GICS sector ===")
    density_by_sector = panel.groupby("gics_sector")["ai_density"].agg(["mean", "count"]).sort_values("mean", ascending=False)
    print(density_by_sector)

    groups = [g["ai_density"].dropna().values for _, g in panel.groupby("gics_sector")]
    f_stat, p_val = stats.f_oneway(*groups)
    print(f"\nOne-way ANOVA (ai_density ~ sector): F = {f_stat:.3f}, p = {p_val:.6f}")
    h_stat, p_val_kw = stats.kruskal(*groups)
    print(f"Kruskal-Wallis (non-parametric robustness check): H = {h_stat:.3f}, p = {p_val_kw:.6f}")

    print("\n=== Share operationally specific by GICS sector ===")
    spec_by_sector = panel.groupby("gics_sector")["share_operational"].agg(["mean", "count"]).sort_values("mean", ascending=False)
    print(spec_by_sector)

    groups_spec = [g["share_operational"].dropna().values for _, g in panel.groupby("gics_sector")]
    groups_spec = [g for g in groups_spec if len(g) > 1]
    f_stat2, p_val2 = stats.f_oneway(*groups_spec)
    print(f"\nOne-way ANOVA (share_operational ~ sector): F = {f_stat2:.3f}, p = {p_val2:.6f}")


def size_rd_regression(panel):
    with_accounting = panel[panel["source"].notna()].copy()
    print(f"\n=== Size/R&D regressions (North America + global fuzzy-matched, n={len(with_accounting)}) ===")

    for dv in ["ai_density", "extensive_margin", "share_operational"]:
        sub = with_accounting.dropna(subset=[dv, "log_size", "mean_rd_intensity_winsorized"])
        if len(sub) < 20:
            print(f"\n{dv}: insufficient data (n={len(sub)}), skipping")
            continue
        model = smf.ols(f"{dv} ~ log_size + mean_rd_intensity_winsorized + C(gics_sector)", data=sub).fit()
        print(f"\n--- DV = {dv} (n={len(sub)}) ---")
        print(f"log_size coef:  {model.params.get('log_size', float('nan')):.5f}  "
              f"(p={model.pvalues.get('log_size', float('nan')):.4f})")
        print(f"rd_intensity coef: {model.params.get('mean_rd_intensity_winsorized', float('nan')):.5f}  "
              f"(p={model.pvalues.get('mean_rd_intensity_winsorized', float('nan')):.4f})")
        print(f"R-squared: {model.rsquared:.4f}")


def main():
    panel = build_firm_panel()
    panel.to_csv("../Data/firm_level_h2_panel.csv", index=False)
    print(f"Built firm-level panel: {len(panel)} firms")
    print(f"  with specificity data: {panel['share_operational'].notna().sum()}")
    print(f"  with accounting data: {panel['mean_at'].notna().sum()}")

    sector_comparison(panel)
    size_rd_regression(panel)


if __name__ == "__main__":
    main()
