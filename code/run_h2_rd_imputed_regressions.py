"""
Reruns Table 4.3's six regressions (3 DVs x 2 specifications) using the
R&D-imputed accounting data (build_rd_imputed_accounting.py's sector x
country mean fill, including Financials/Utilities) instead of dropping
firms with missing R&D intensity.

Run:
    python run_h2_rd_imputed_regressions.py
"""

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

RD_WINSOR_PCTILE = 0.95
ACCT_PATH = "../Data/firm_accounting_panel_rd_imputed.csv"


def fmt_stars(p):
    if p < 0.01:
        return "***"
    if p < 0.05:
        return "**"
    if p < 0.10:
        return "*"
    return ""


def report(model, dv, label, n, n_firms=None, cluster=True):
    print(f"\n--- {label}: DV={dv} (n={n}" + (f", {n_firms} firms" if n_firms else "") + ") ---")
    for var, pretty in [("log_size", "log_size"), (var_rd := [c for c in model.params.index if "rd_intensity" in c][0], "rd_intensity")]:
        coef = model.params[var]
        se = model.bse[var]
        p = model.pvalues[var]
        print(f"  {pretty}: {coef:+.5f}{fmt_stars(p)}  (SE={se:.5f}, p={p:.4f})")
    print(f"  R-squared: {model.rsquared:.4f}")


def build_cross_sectional_panel():
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
        share_operational=("share_operational", "mean"),
    ).reset_index()

    sector = pd.read_csv("../Data/ticker_sector_mapping.csv")[["ticker", "gics_sector", "country"]]

    acct = pd.read_csv(ACCT_PATH)
    acct_firm = acct.groupby("ticker").agg(
        mean_at=("at", "mean"),
        mean_rd_intensity=("rd_intensity", "mean"),
    ).reset_index()
    cap = acct_firm["mean_rd_intensity"].quantile(RD_WINSOR_PCTILE)
    acct_firm["mean_rd_intensity_winsorized"] = acct_firm["mean_rd_intensity"].clip(upper=cap)
    acct_firm["log_size"] = np.log(acct_firm["mean_at"])

    panel = firm_volume.merge(firm_spec, on="ticker", how="left")
    panel = panel.merge(sector, on="ticker", how="left")
    panel = panel.merge(acct_firm, on="ticker", how="left")
    return panel


def run_cross_sectional():
    panel = build_cross_sectional_panel()
    for dv in ["ai_density", "extensive_margin", "share_operational"]:
        sub = panel.dropna(subset=[dv, "log_size", "mean_rd_intensity_winsorized"])
        model = smf.ols(
            f"{dv} ~ log_size + mean_rd_intensity_winsorized + C(gics_sector)", data=sub
        ).fit()
        report(model, dv, "Cross-sectional (R&D-imputed)", len(sub))


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


def run_panel():
    call = build_call_level_volume()
    spec = pd.read_csv("../Data/firm_quarter_specificity_panel.csv")
    panel = call.merge(
        spec[["ticker", "quarter", "share_operational"]], on=["ticker", "quarter"], how="left",
    )

    acct = pd.read_csv(ACCT_PATH, parse_dates=["datadate"])
    acct = acct.dropna(subset=["datadate", "at"]).sort_values("datadate")
    cap = acct["rd_intensity"].quantile(RD_WINSOR_PCTILE)
    acct["rd_intensity_winsorized"] = acct["rd_intensity"].clip(upper=cap)

    panel = panel.sort_values("date")
    panel = pd.merge_asof(
        panel, acct[["ticker", "datadate", "at", "rd_intensity_winsorized"]],
        left_on="date", right_on="datadate", by="ticker", direction="backward",
    )
    panel["log_size"] = np.log(panel["at"])

    sector = pd.read_csv("../Data/ticker_sector_mapping.csv")[["ticker", "gics_sector"]]
    panel = panel.merge(sector, on="ticker", how="left")

    for dv in ["ai_density", "mentioned", "share_operational"]:
        sub = panel.dropna(subset=[dv, "log_size", "rd_intensity_winsorized", "gics_sector", "year"]).copy()
        model = smf.ols(
            f"{dv} ~ log_size + rd_intensity_winsorized + C(gics_sector) + C(year)", data=sub,
        ).fit(cov_type="cluster", cov_kwds={"groups": sub["ticker"]})
        report(model, dv, "Firm-quarter panel (R&D-imputed)", len(sub), sub["ticker"].nunique())


if __name__ == "__main__":
    print("=" * 20 + " CROSS-SECTIONAL " + "=" * 20)
    run_cross_sectional()
    print("\n" + "=" * 20 + " FIRM-QUARTER PANEL " + "=" * 20)
    run_panel()
