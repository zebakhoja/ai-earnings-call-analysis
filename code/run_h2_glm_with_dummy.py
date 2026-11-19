"""
GLM (distribution-appropriate) version of run_h2_rd_imputed_with_dummy.py's
was_rd_imputed-dummy comparison. That script's before/after coefficient
comparison (currently reported in Results/Discussion as "the R&D-intensity
coefficient gets stronger once the dummy is added") was built entirely on
OLS. Since Table 4.3 has since consolidated to report the
distribution-appropriate model (Negative Binomial for ai_density, grouped
Binomial GLM for extensive_margin/mentioned and share_operational) as the
primary result, the with-dummy comparison needs re-running on that same
scale for the before/after numbers to be internally consistent with the
rest of the document -- the OLS-based percentages currently cited are
stale relative to that methodological shift.

Reports, for both cross-sectional and firm-quarter panel: the
rd_intensity coefficient WITHOUT the was_rd_imputed dummy (matching
Table 4.3's GLM rows) and WITH it, on the appropriate model's own scale
(log-count semi-elasticity for ai_density, log-odds for
extensive_margin/mentioned and share_operational) -- NOT comparable in
magnitude to the OLS percentages previously reported, since these are
different scales entirely (see Table 4.3's own note on this).

Run:
    python run_h2_glm_with_dummy.py
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

from run_h2_rd_imputed_regressions import ACCT_PATH, RD_WINSOR_PCTILE
from run_h2_rd_imputed_with_dummy import (
    build_cross_sectional_panel_with_dummy, RAW_ACCT_PATH,
)


def fmt_stars(p):
    if p < 0.01:
        return "***"
    if p < 0.05:
        return "**"
    if p < 0.10:
        return "*"
    return ""


def report(coef_no_dummy, p_no_dummy, coef_with_dummy, p_with_dummy, label):
    print(f"\n--- {label} ---")
    print(f"  without dummy: {coef_no_dummy:+.5f}{fmt_stars(p_no_dummy)}  (p={p_no_dummy:.4f})")
    print(f"  with dummy:    {coef_with_dummy:+.5f}{fmt_stars(p_with_dummy)}  (p={p_with_dummy:.4f})")
    pct_change = (coef_with_dummy - coef_no_dummy) / abs(coef_no_dummy) * 100
    print(f"  change: {pct_change:+.1f}%")


def run_cross_sectional():
    panel = build_cross_sectional_panel_with_dummy()

    sub = panel.dropna(subset=["total_mentions", "total_sentences", "log_size",
                                "mean_rd_intensity_winsorized", "firm_never_reported",
                                "gics_sector"]).copy()
    sub = sub[sub["total_sentences"] > 0]
    nb_no_dummy = smf.glm(
        "total_mentions ~ log_size + mean_rd_intensity_winsorized + C(gics_sector)",
        data=sub, family=sm.families.NegativeBinomial(), offset=np.log(sub["total_sentences"]),
    ).fit()
    nb_with_dummy = smf.glm(
        "total_mentions ~ log_size + mean_rd_intensity_winsorized + firm_never_reported + C(gics_sector)",
        data=sub, family=sm.families.NegativeBinomial(), offset=np.log(sub["total_sentences"]),
    ).fit()
    report(nb_no_dummy.params["mean_rd_intensity_winsorized"], nb_no_dummy.pvalues["mean_rd_intensity_winsorized"],
           nb_with_dummy.params["mean_rd_intensity_winsorized"], nb_with_dummy.pvalues["mean_rd_intensity_winsorized"],
           "ai_density -- Negative Binomial, cross-sectional")

    sub2 = panel.dropna(subset=["n_calls_with_mention", "n_calls", "log_size",
                                 "mean_rd_intensity_winsorized", "firm_never_reported",
                                 "gics_sector"]).copy()

    def fit_binom(df, with_dummy):
        endog = np.column_stack([df["n_calls_with_mention"], df["n_calls"] - df["n_calls_with_mention"]])
        cols = ["log_size", "mean_rd_intensity_winsorized"] + (["firm_never_reported"] if with_dummy else [])
        exog = pd.get_dummies(df[cols + ["gics_sector"]], columns=["gics_sector"], drop_first=True)
        exog = sm.add_constant(exog.astype(float))
        return sm.GLM(endog, exog, family=sm.families.Binomial()).fit()

    m_no = fit_binom(sub2, False)
    m_yes = fit_binom(sub2, True)
    report(m_no.params["mean_rd_intensity_winsorized"], m_no.pvalues["mean_rd_intensity_winsorized"],
           m_yes.params["mean_rd_intensity_winsorized"], m_yes.pvalues["mean_rd_intensity_winsorized"],
           "extensive_margin -- grouped Binomial GLM, cross-sectional")

    spec = pd.read_csv("../Data/firm_quarter_specificity_panel.csv")
    spec["n_operational_q"] = (spec["share_operational"] * spec["snippet_count"]).round()
    firm_counts = spec.groupby("ticker").agg(
        n_operational=("n_operational_q", "sum"), snippet_count=("snippet_count", "sum"),
    ).reset_index()
    sub3 = panel.drop(columns=["share_operational"], errors="ignore").merge(firm_counts, on="ticker", how="left")
    sub3 = sub3.dropna(subset=["n_operational", "snippet_count", "log_size",
                                "mean_rd_intensity_winsorized", "firm_never_reported", "gics_sector"])
    sub3 = sub3[sub3["snippet_count"] > 0]

    def fit_binom3(df, with_dummy):
        endog = np.column_stack([df["n_operational"], df["snippet_count"] - df["n_operational"]])
        cols = ["log_size", "mean_rd_intensity_winsorized"] + (["firm_never_reported"] if with_dummy else [])
        exog = pd.get_dummies(df[cols + ["gics_sector"]], columns=["gics_sector"], drop_first=True)
        exog = sm.add_constant(exog.astype(float))
        return sm.GLM(endog, exog, family=sm.families.Binomial()).fit()

    m3_no = fit_binom3(sub3, False)
    m3_yes = fit_binom3(sub3, True)
    report(m3_no.params["mean_rd_intensity_winsorized"], m3_no.pvalues["mean_rd_intensity_winsorized"],
           m3_yes.params["mean_rd_intensity_winsorized"], m3_yes.pvalues["mean_rd_intensity_winsorized"],
           "share_operational -- grouped Binomial GLM, cross-sectional")


def run_panel():
    volume = pd.read_csv("../Data/ai_volume_panel.csv", parse_dates=["date"])
    call = volume.groupby(["symbol", "year", "period", "date"]).agg(
        n_ai_mentions_total=("n_ai_mentions_total", "sum"),
        n_sentences=("n_sentences", "sum"),
    ).reset_index()
    call["mentioned"] = (call["n_ai_mentions_total"] > 0).astype(int)
    call = call.rename(columns={"symbol": "ticker"})
    call["quarter"] = call["year"].astype(str) + "_" + call["period"]

    acct = pd.read_csv(ACCT_PATH, parse_dates=["datadate"])
    acct = acct.dropna(subset=["datadate", "at"]).sort_values("datadate")
    cap = acct["rd_intensity"].quantile(RD_WINSOR_PCTILE)
    acct["rd_intensity_winsorized"] = acct["rd_intensity"].clip(upper=cap)

    call = call.sort_values("date")
    call = pd.merge_asof(
        call, acct[["ticker", "datadate", "at", "rd_intensity_winsorized", "rd_intensity_imputed"]],
        left_on="date", right_on="datadate", by="ticker", direction="backward",
    )
    call["log_size"] = np.log(call["at"])
    call["rd_intensity_imputed"] = call["rd_intensity_imputed"].astype("boolean").astype("Int64").astype(float)
    sector = pd.read_csv("../Data/ticker_sector_mapping.csv")[["ticker", "gics_sector"]]
    call = call.merge(sector, on="ticker", how="left")
    call["year_s"] = call["year"].astype(str)

    sub = call.dropna(subset=["n_ai_mentions_total", "n_sentences", "log_size",
                               "rd_intensity_winsorized", "rd_intensity_imputed",
                               "gics_sector", "year_s"]).copy()
    sub = sub[sub["n_sentences"] > 0]
    nb_no = smf.glm(
        "n_ai_mentions_total ~ log_size + rd_intensity_winsorized + C(gics_sector) + C(year_s)",
        data=sub, family=sm.families.NegativeBinomial(), offset=np.log(sub["n_sentences"]),
    ).fit(cov_type="cluster", cov_kwds={"groups": sub["ticker"]})
    nb_yes = smf.glm(
        "n_ai_mentions_total ~ log_size + rd_intensity_winsorized + rd_intensity_imputed + C(gics_sector) + C(year_s)",
        data=sub, family=sm.families.NegativeBinomial(), offset=np.log(sub["n_sentences"]),
    ).fit(cov_type="cluster", cov_kwds={"groups": sub["ticker"]})
    report(nb_no.params["rd_intensity_winsorized"], nb_no.pvalues["rd_intensity_winsorized"],
           nb_yes.params["rd_intensity_winsorized"], nb_yes.pvalues["rd_intensity_winsorized"],
           "ai_density -- Negative Binomial, firm-quarter panel")

    sub2 = call.dropna(subset=["mentioned", "log_size", "rd_intensity_winsorized",
                                "rd_intensity_imputed", "gics_sector", "year_s"]).copy()
    lg_no = smf.glm(
        "mentioned ~ log_size + rd_intensity_winsorized + C(gics_sector) + C(year_s)",
        data=sub2, family=sm.families.Binomial(),
    ).fit(cov_type="cluster", cov_kwds={"groups": sub2["ticker"]})
    lg_yes = smf.glm(
        "mentioned ~ log_size + rd_intensity_winsorized + rd_intensity_imputed + C(gics_sector) + C(year_s)",
        data=sub2, family=sm.families.Binomial(),
    ).fit(cov_type="cluster", cov_kwds={"groups": sub2["ticker"]})
    report(lg_no.params["rd_intensity_winsorized"], lg_no.pvalues["rd_intensity_winsorized"],
           lg_yes.params["rd_intensity_winsorized"], lg_yes.pvalues["rd_intensity_winsorized"],
           "mentioned -- binary logistic, firm-quarter panel")

    spec = pd.read_csv("../Data/firm_quarter_specificity_panel.csv")
    spec["n_operational"] = (spec["share_operational"] * spec["snippet_count"]).round()
    dates = call[["ticker", "quarter", "date"]].drop_duplicates()
    spec = spec.merge(dates, on=["ticker", "quarter"], how="left").dropna(subset=["date"])
    spec = spec.sort_values("date")
    spec = pd.merge_asof(
        spec, acct[["ticker", "datadate", "at", "rd_intensity_winsorized", "rd_intensity_imputed"]],
        left_on="date", right_on="datadate", by="ticker", direction="backward",
    )
    spec["log_size"] = np.log(spec["at"])
    spec["rd_intensity_imputed"] = spec["rd_intensity_imputed"].astype("boolean").astype("Int64").astype(float)
    spec = spec.merge(sector, on="ticker", how="left")
    spec["year_s"] = spec["quarter"].str.slice(0, 4)

    sub3 = spec.dropna(subset=["n_operational", "snippet_count", "log_size",
                                "rd_intensity_winsorized", "rd_intensity_imputed",
                                "gics_sector", "year_s"]).copy()
    sub3 = sub3[sub3["snippet_count"] > 0]

    def fit_binom_panel(df, with_dummy):
        endog = np.column_stack([df["n_operational"], df["snippet_count"] - df["n_operational"]])
        cols = ["log_size", "rd_intensity_winsorized"] + (["rd_intensity_imputed"] if with_dummy else [])
        exog = pd.get_dummies(df[cols + ["gics_sector", "year_s"]], columns=["gics_sector", "year_s"], drop_first=True)
        exog = sm.add_constant(exog.astype(float))
        return sm.GLM(endog, exog, family=sm.families.Binomial()).fit(
            cov_type="cluster", cov_kwds={"groups": df["ticker"]})

    b_no = fit_binom_panel(sub3, False)
    b_yes = fit_binom_panel(sub3, True)
    report(b_no.params["rd_intensity_winsorized"], b_no.pvalues["rd_intensity_winsorized"],
           b_yes.params["rd_intensity_winsorized"], b_yes.pvalues["rd_intensity_winsorized"],
           "share_operational -- grouped Binomial GLM, firm-quarter panel")


if __name__ == "__main__":
    print("=" * 15 + " CROSS-SECTIONAL " + "=" * 15)
    run_cross_sectional()
    print("\n" + "=" * 15 + " FIRM-QUARTER PANEL " + "=" * 15)
    run_panel()
