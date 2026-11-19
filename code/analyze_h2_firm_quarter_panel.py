"""
H2, firm-QUARTER version: does within-firm-over-time and cross-firm
variation in AI discourse volume/specificity track TIME-VARYING size and
R&D intensity, rather than each firm's all-time average (the firm-level
version in analyze_h2_heterogeneity.py)?

Two things a firm-level panel cannot do, that this can:
    1. Use each call's actual size/R&D as of that call date (lagged to the
       most recent prior fiscal year, avoiding look-ahead bias), rather
       than one all-time average per firm -- a firm that grew from small
       to large over 2018-2026 gets credited with its size AT EACH CALL,
       not a single blended average.
    2. Include year fixed effects, so the size/R&D coefficients are
       identified from cross-sectional variation WITHIN each year, not
       confounded with the huge aggregate AI-discourse growth over time
       (H1's own volume result) that a pooled-without-time-controls
       regression would otherwise absorb into the size/R&D coefficients.
    3. Cluster standard errors by firm -- repeated quarterly observations
       per firm are not independent, and firm-level analysis's standard
       errors would be overstated (too significant) without this.

Design:
    - Call-level AI volume from ai_volume_panel.csv, collapsed across
      prepared_remarks/qa per (ticker, year, period, date).
    - Specificity from firm_quarter_specificity_panel.csv (only exists for
      quarters with >=1 AI mention, same restriction as the firm-level
      version).
    - Size/R&D: merge_asof (backward) against firm_accounting_panel_combined.csv,
      matching each call's date to the most recent prior fiscal year-end
      for that ticker -- a genuinely lagged, time-varying covariate.
    - rd_intensity winsorized at the 95th percentile computed over the
      firm-YEAR panel (not re-winsorized per quarter), consistent with the
      firm-level version's approach.

Run:
    python analyze_h2_firm_quarter_panel.py
"""

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

RD_WINSOR_PCTILE = 0.95


def build_call_level_volume():
    volume = pd.read_csv("../Data/ai_volume_panel.csv", parse_dates=["date"])
    call = volume.groupby(["symbol", "year", "period", "date"]).agg(
        n_ai_mentions_total=("n_ai_mentions_total", "sum"),
        n_sentences=("n_sentences", "sum"),
    ).reset_index()
    call["ai_density"] = call["n_ai_mentions_total"] / call["n_sentences"]
    call["mentioned"] = (call["n_ai_mentions_total"] > 0).astype(int)
    call = call.rename(columns={"symbol": "ticker"})
    call["quarter"] = call["year"].astype(str) + "_" + call["period"]
    return call


def merge_specificity(call):
    spec = pd.read_csv("../Data/firm_quarter_specificity_panel.csv")
    return call.merge(
        spec[["ticker", "quarter", "mean_specificity", "share_operational"]],
        on=["ticker", "quarter"], how="left",
    )


def merge_lagged_accounting(call):
    acct = pd.read_csv("../Data/firm_accounting_panel_combined.csv", parse_dates=["datadate"])
    acct = acct.dropna(subset=["datadate", "at"]).sort_values("datadate")

    cap = acct["rd_intensity"].quantile(RD_WINSOR_PCTILE)
    n_capped = (acct["rd_intensity"] > cap).sum()
    acct["rd_intensity_winsorized"] = acct["rd_intensity"].clip(upper=cap)
    print(f"R&D intensity winsorized at {RD_WINSOR_PCTILE:.0%} percentile = {cap:.3f}; "
          f"{n_capped} firm-years capped")

    call = call.sort_values("date")
    merged = pd.merge_asof(
        call, acct[["ticker", "datadate", "at", "rd_intensity_winsorized"]],
        left_on="date", right_on="datadate", by="ticker", direction="backward",
    )
    merged["log_size"] = np.log(merged["at"])
    return merged


def merge_sector(panel):
    sector = pd.read_csv("../Data/ticker_sector_mapping.csv")[["ticker", "gics_sector"]]
    return panel.merge(sector, on="ticker", how="left")


def run_panel_regression(panel, dv):
    sub = panel.dropna(subset=[dv, "log_size", "rd_intensity_winsorized", "gics_sector", "year"]).copy()
    if len(sub) < 50:
        print(f"\n{dv}: insufficient data (n={len(sub)}), skipping")
        return
    model = smf.ols(
        f"{dv} ~ log_size + rd_intensity_winsorized + C(gics_sector) + C(year)",
        data=sub,
    ).fit(cov_type="cluster", cov_kwds={"groups": sub["ticker"]})
    print(f"\n--- DV = {dv} (n={len(sub)} firm-quarters, {sub['ticker'].nunique()} firms, "
          f"firm-clustered SE) ---")
    print(f"log_size coef:  {model.params.get('log_size', float('nan')):.5f}  "
          f"(p={model.pvalues.get('log_size', float('nan')):.4f})")
    print(f"rd_intensity coef: {model.params.get('rd_intensity_winsorized', float('nan')):.5f}  "
          f"(p={model.pvalues.get('rd_intensity_winsorized', float('nan')):.4f})")
    print(f"R-squared: {model.rsquared:.4f}")


def main():
    call = build_call_level_volume()
    panel = merge_specificity(call)
    panel = merge_lagged_accounting(panel)
    panel = merge_sector(panel)

    panel.to_csv("../Data/firm_quarter_h2_panel.csv", index=False)
    print(f"\nBuilt firm-quarter panel: {len(panel)} call-level rows, "
          f"{panel['ticker'].nunique()} firms")
    print(f"  with lagged accounting data: {panel['at'].notna().sum()} rows "
          f"({panel.loc[panel['at'].notna(), 'ticker'].nunique()} firms)")
    print(f"  with specificity data: {panel['share_operational'].notna().sum()} rows")

    for dv in ["ai_density", "mentioned", "share_operational"]:
        run_panel_regression(panel, dv)


if __name__ == "__main__":
    main()
