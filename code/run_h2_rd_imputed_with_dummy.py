"""
Robustness check on Table 4.3's R&D-imputed regressions: adds a
was_rd_imputed indicator alongside the imputed R&D-intensity value, per
Koh and Reeb's (2015, "Missing R&D") finding that including a dummy for
missing-R&D firms improves on using an imputed value alone -- it lets the
regression separately absorb any systematic difference between firms that
genuinely reported R&D and firms whose R&D was filled in, rather than
treating both as identical.

Cross-sectional spec: R&D intensity is a firm-level average across years,
so the dummy here is `firm_never_reported` = 1 if the firm has ZERO
reporting years in firm_accounting_panel_combined.csv (i.e. its firm-level
R&D figure is entirely synthetic), 0 if it reported R&D in at least one
year (even if other years needed filling).

Firm-quarter panel spec: `rd_intensity_imputed` is already tracked
per-firm-year by build_rd_imputed_accounting.py and carried through the
same merge_asof lagged-match used for the R&D value itself, so each call
gets the imputed flag from its own matched fiscal year directly.

Run:
    python run_h2_rd_imputed_with_dummy.py
"""

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from run_h2_rd_imputed_regressions import (
    build_call_level_volume, fmt_stars, RD_WINSOR_PCTILE, ACCT_PATH,
)

RAW_ACCT_PATH = "../Data/firm_accounting_panel_combined.csv"


def report_with_dummy(model, dv, label, n, n_firms=None):
    print(f"\n--- {label}: DV={dv} (n={n}" + (f", {n_firms} firms" if n_firms else "") + ") ---")
    rd_var = [c for c in model.params.index if "rd_intensity" in c][0]
    dummy_var = [c for c in model.params.index if "imputed" in c or "never_reported" in c][0]
    for var, pretty in [("log_size", "log_size"), (rd_var, "rd_intensity"), (dummy_var, "was_rd_imputed dummy")]:
        coef = model.params[var]
        se = model.bse[var]
        p = model.pvalues[var]
        print(f"  {pretty}: {coef:+.5f}{fmt_stars(p)}  (SE={se:.5f}, p={p:.4f})")
    print(f"  R-squared: {model.rsquared:.4f}")


def build_cross_sectional_panel_with_dummy():
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
    firm_spec = spec.groupby("ticker").agg(share_operational=("share_operational", "mean")).reset_index()

    sector = pd.read_csv("../Data/ticker_sector_mapping.csv")[["ticker", "gics_sector", "country"]]

    acct = pd.read_csv(ACCT_PATH)
    acct_firm = acct.groupby("ticker").agg(
        mean_at=("at", "mean"),
        mean_rd_intensity=("rd_intensity", "mean"),
    ).reset_index()
    cap = acct_firm["mean_rd_intensity"].quantile(RD_WINSOR_PCTILE)
    acct_firm["mean_rd_intensity_winsorized"] = acct_firm["mean_rd_intensity"].clip(upper=cap)
    acct_firm["log_size"] = np.log(acct_firm["mean_at"])

    raw = pd.read_csv(RAW_ACCT_PATH)
    ever_reported = raw.groupby("ticker")["rd_intensity"].apply(lambda s: s.notna().any())
    acct_firm["firm_never_reported"] = acct_firm["ticker"].map(ever_reported).eq(False).astype(int)

    panel = firm_volume.merge(firm_spec, on="ticker", how="left")
    panel = panel.merge(sector, on="ticker", how="left")
    panel = panel.merge(acct_firm, on="ticker", how="left")
    return panel


def run_cross_sectional_with_dummy():
    panel = build_cross_sectional_panel_with_dummy()
    print(f"firm_never_reported: {panel['firm_never_reported'].sum()} of {len(panel)} firms "
          f"never reported R&D in any year")
    for dv in ["ai_density", "extensive_margin", "share_operational"]:
        sub = panel.dropna(subset=[dv, "log_size", "mean_rd_intensity_winsorized", "firm_never_reported"])
        model = smf.ols(
            f"{dv} ~ log_size + mean_rd_intensity_winsorized + firm_never_reported + C(gics_sector)",
            data=sub,
        ).fit()
        report_with_dummy(model, dv, "Cross-sectional (R&D-imputed + dummy)", len(sub))


def run_panel_with_dummy():
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
        panel, acct[["ticker", "datadate", "at", "rd_intensity_winsorized", "rd_intensity_imputed"]],
        left_on="date", right_on="datadate", by="ticker", direction="backward",
    )
    panel["log_size"] = np.log(panel["at"])
    panel["rd_intensity_imputed"] = panel["rd_intensity_imputed"].astype("boolean").astype("Int64")

    sector = pd.read_csv("../Data/ticker_sector_mapping.csv")[["ticker", "gics_sector"]]
    panel = panel.merge(sector, on="ticker", how="left")

    for dv in ["ai_density", "mentioned", "share_operational"]:
        sub = panel.dropna(
            subset=[dv, "log_size", "rd_intensity_winsorized", "rd_intensity_imputed", "gics_sector", "year"]
        ).copy()
        model = smf.ols(
            f"{dv} ~ log_size + rd_intensity_winsorized + rd_intensity_imputed + C(gics_sector) + C(year)",
            data=sub,
        ).fit(cov_type="cluster", cov_kwds={"groups": sub["ticker"]})
        report_with_dummy(model, dv, "Firm-quarter panel (R&D-imputed + dummy)", len(sub), sub["ticker"].nunique())


if __name__ == "__main__":
    print("=" * 20 + " CROSS-SECTIONAL (+ was_rd_imputed dummy) " + "=" * 20)
    run_cross_sectional_with_dummy()
    print("\n" + "=" * 20 + " FIRM-QUARTER PANEL (+ was_rd_imputed dummy) " + "=" * 20)
    run_panel_with_dummy()
