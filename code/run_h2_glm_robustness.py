"""
GLM robustness check on Table 4.3's six OLS regressions
(run_h2_rd_imputed_regressions.py). All three DVs there are bounded
proportions (or, for ai_density, a rate constructed from a count over an
exposure), so OLS can in principle predict outside [0,1], understate
heteroskedasticity, and miss nonlinear marginal effects near the bounds.
This reruns each with the more natural model for its construction,
using the actual success/trial counts behind each proportion rather than
approximating a continuous fractional response wherever those counts are
available:

  - ai_density (mentions / sentences): negative binomial GLM on the raw
    mention COUNT, with log(sentences) as an exposure offset -- the
    standard way to model a rate. Poisson is reported too, with a
    dispersion check, since NB reduces to Poisson-equivalent conclusions
    only if overdispersion is actually present.
  - extensive_margin (calls mentioning AI / total calls) and "mentioned"
    (binary at the call level): grouped/binary binomial GLM (logit link)
    using the real successes/trials counts, not an approximation.
  - share_operational (operationally-specific snippets / total snippets):
    grouped binomial GLM (logit link) reconstructed from
    firm_quarter_specificity_panel.csv's snippet_count and
    share_operational columns (n_operational = round(share_operational *
    snippet_count), n_total = snippet_count) -- a real grouped-binomial
    model, not a fractional-response approximation, since the underlying
    counts are available.

Reports each GLM's log_size / rd_intensity coefficients (log-odds or
log-rate scale) alongside the OLS coefficients already reported by
run_h2_rd_imputed_regressions.py, and flags whether the sign/significance
conclusion changes.

Run:
    python run_h2_glm_robustness.py
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
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


def report_glm(model, label, n, key_vars, n_firms=None, dispersion=None):
    print(f"\n--- {label} (n={n}" + (f", {n_firms} firms" if n_firms else "") + ") ---")
    for var, pretty in key_vars:
        if var not in model.params.index:
            continue
        coef = model.params[var]
        se = model.bse[var]
        p = model.pvalues[var]
        print(f"  {pretty}: {coef:+.5f}{fmt_stars(p)}  (SE={se:.5f}, p={p:.4f})")
    if dispersion is not None:
        print(f"  Pearson dispersion statistic: {dispersion:.3f} "
              f"({'overdispersed -- NB preferred' if dispersion > 1.5 else 'close to 1 -- Poisson adequate'})")


def rd_var_name(model):
    return [c for c in model.params.index if "rd_intensity" in c][0]


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
    firm_volume = firm_volume.rename(columns={"symbol": "ticker"})

    spec = pd.read_csv("../Data/firm_quarter_specificity_panel.csv")
    firm_spec = spec.groupby("ticker").agg(
        n_operational=("share_operational", lambda s: np.nan),
        snippet_count=("snippet_count", "sum"),
    ).reset_index()
    spec["n_operational_q"] = (spec["share_operational"] * spec["snippet_count"]).round()
    n_op = spec.groupby("ticker")["n_operational_q"].sum().reset_index().rename(
        columns={"n_operational_q": "n_operational"})
    firm_spec = firm_spec.drop(columns=["n_operational"]).merge(n_op, on="ticker", how="left")

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


def run_cross_sectional_glm():
    panel = build_cross_sectional_panel()

    sub = panel.dropna(subset=["total_mentions", "total_sentences", "log_size",
                                "mean_rd_intensity_winsorized", "gics_sector"]).copy()
    sub = sub[sub["total_sentences"] > 0]
    poisson = smf.glm(
        "total_mentions ~ log_size + mean_rd_intensity_winsorized + C(gics_sector)",
        data=sub, family=sm.families.Poisson(),
        offset=np.log(sub["total_sentences"]),
    ).fit()
    pearson_chi2 = poisson.pearson_chi2 / poisson.df_resid
    nb = smf.glm(
        "total_mentions ~ log_size + mean_rd_intensity_winsorized + C(gics_sector)",
        data=sub, family=sm.families.NegativeBinomial(),
        offset=np.log(sub["total_sentences"]),
    ).fit()
    report_glm(poisson, "ai_density -- Poisson GLM, count with sentence-count offset (cross-sectional)",
               len(sub), [("log_size", "log_size"), (rd_var_name(poisson), "rd_intensity")],
               dispersion=pearson_chi2)
    report_glm(nb, "ai_density -- Negative Binomial GLM, count with sentence-count offset (cross-sectional)",
               len(sub), [("log_size", "log_size"), (rd_var_name(nb), "rd_intensity")])

    sub2 = panel.dropna(subset=["n_calls_with_mention", "n_calls", "log_size",
                                 "mean_rd_intensity_winsorized", "gics_sector"]).copy()
    endog = np.column_stack([sub2["n_calls_with_mention"], sub2["n_calls"] - sub2["n_calls_with_mention"]])
    exog = pd.get_dummies(sub2[["log_size", "mean_rd_intensity_winsorized", "gics_sector"]],
                           columns=["gics_sector"], drop_first=True)
    exog = sm.add_constant(exog.astype(float))
    binom = sm.GLM(endog, exog, family=sm.families.Binomial()).fit()
    report_glm(binom, "extensive_margin -- grouped Binomial GLM (logit link), successes/trials (cross-sectional)",
               len(sub2), [("log_size", "log_size"), (rd_var_name(binom), "rd_intensity")])

    sub3 = panel.dropna(subset=["n_operational", "snippet_count", "log_size",
                                 "mean_rd_intensity_winsorized", "gics_sector"]).copy()
    sub3 = sub3[sub3["snippet_count"] > 0]
    endog3 = np.column_stack([sub3["n_operational"], sub3["snippet_count"] - sub3["n_operational"]])
    exog3 = pd.get_dummies(sub3[["log_size", "mean_rd_intensity_winsorized", "gics_sector"]],
                            columns=["gics_sector"], drop_first=True)
    exog3 = sm.add_constant(exog3.astype(float))
    binom3 = sm.GLM(endog3, exog3, family=sm.families.Binomial()).fit()
    report_glm(binom3, "share_operational -- grouped Binomial GLM (logit link), successes/trials (cross-sectional)",
               len(sub3), [("log_size", "log_size"), (rd_var_name(binom3), "rd_intensity")])


def build_call_level_volume():
    volume = pd.read_csv("../Data/ai_volume_panel.csv", parse_dates=["date"])
    call = volume.groupby(["symbol", "year", "period", "date"]).agg(
        n_ai_mentions_total=("n_ai_mentions_total", "sum"),
        n_sentences=("n_sentences", "sum"),
    ).reset_index()
    call["mentioned"] = (call["n_ai_mentions_total"] > 0).astype(int)
    call = call.rename(columns={"symbol": "ticker"})
    call["quarter"] = call["year"].astype(str) + "_" + call["period"]
    return call


def _merge_accounting(panel):
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
    return panel


def run_panel_glm():
    call = build_call_level_volume()
    call = _merge_accounting(call)

    sub = call.dropna(subset=["n_ai_mentions_total", "n_sentences", "log_size",
                               "rd_intensity_winsorized", "gics_sector", "year"]).copy()
    sub = sub[sub["n_sentences"] > 0]
    poisson = smf.glm(
        "n_ai_mentions_total ~ log_size + rd_intensity_winsorized + C(gics_sector) + C(year)",
        data=sub, family=sm.families.Poisson(), offset=np.log(sub["n_sentences"]),
    ).fit(cov_type="cluster", cov_kwds={"groups": sub["ticker"]})
    pearson_chi2 = poisson.pearson_chi2 / poisson.df_resid
    nb = smf.glm(
        "n_ai_mentions_total ~ log_size + rd_intensity_winsorized + C(gics_sector) + C(year)",
        data=sub, family=sm.families.NegativeBinomial(), offset=np.log(sub["n_sentences"]),
    ).fit(cov_type="cluster", cov_kwds={"groups": sub["ticker"]})
    report_glm(poisson, "ai_density -- Poisson GLM, count with sentence-count offset (firm-quarter panel, clustered SE)",
               len(sub), [("log_size", "log_size"), (rd_var_name(poisson), "rd_intensity")],
               sub["ticker"].nunique(), dispersion=pearson_chi2)
    report_glm(nb, "ai_density -- Negative Binomial GLM, count with sentence-count offset (firm-quarter panel, clustered SE)",
               len(sub), [("log_size", "log_size"), (rd_var_name(nb), "rd_intensity")], sub["ticker"].nunique())

    sub2 = call.dropna(subset=["mentioned", "log_size", "rd_intensity_winsorized",
                                "gics_sector", "year"]).copy()
    logit = smf.glm(
        "mentioned ~ log_size + rd_intensity_winsorized + C(gics_sector) + C(year)",
        data=sub2, family=sm.families.Binomial(),
    ).fit(cov_type="cluster", cov_kwds={"groups": sub2["ticker"]})
    report_glm(logit, "mentioned -- binary logistic GLM (firm-quarter panel, clustered SE)",
               len(sub2), [("log_size", "log_size"), (rd_var_name(logit), "rd_intensity")], sub2["ticker"].nunique())

    spec = pd.read_csv("../Data/firm_quarter_specificity_panel.csv")
    spec["n_operational"] = (spec["share_operational"] * spec["snippet_count"]).round()
    volume_dates = call[["ticker", "quarter", "date"]].drop_duplicates()
    spec = spec.merge(volume_dates, on=["ticker", "quarter"], how="left")
    spec = _merge_accounting(spec.dropna(subset=["date"]))
    sub3 = spec.dropna(subset=["n_operational", "snippet_count", "log_size",
                                "rd_intensity_winsorized", "gics_sector"]).copy()
    sub3 = sub3[sub3["snippet_count"] > 0]
    sub3["year"] = sub3["quarter"].str.slice(0, 4)
    endog3 = np.column_stack([sub3["n_operational"], sub3["snippet_count"] - sub3["n_operational"]])
    exog3 = pd.get_dummies(sub3[["log_size", "rd_intensity_winsorized", "gics_sector", "year"]],
                            columns=["gics_sector", "year"], drop_first=True)
    exog3 = sm.add_constant(exog3.astype(float))
    binom3 = sm.GLM(endog3, exog3, family=sm.families.Binomial()).fit(
        cov_type="cluster", cov_kwds={"groups": sub3["ticker"]})
    report_glm(binom3, "share_operational -- grouped Binomial GLM (logit link), successes/trials (firm-quarter panel, clustered SE)",
               len(sub3), [("log_size", "log_size"), (rd_var_name(binom3), "rd_intensity")], sub3["ticker"].nunique())


if __name__ == "__main__":
    print("=" * 20 + " CROSS-SECTIONAL GLM ROBUSTNESS " + "=" * 20)
    run_cross_sectional_glm()
    print("\n" + "=" * 20 + " FIRM-QUARTER PANEL GLM ROBUSTNESS " + "=" * 20)
    run_panel_glm()
