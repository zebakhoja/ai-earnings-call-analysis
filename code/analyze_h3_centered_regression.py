"""
Mean-centered version of H3's register x capacity regression (Table 4.5 /
analyze_h3_register_specificity.py's run_regression(), and its Binomial
GLM robustness counterpart in analyze_h3_glm_robustness.py).

Both log_size and rd_intensity are mean-centered before building the
qa interaction terms, so the "qa" main-effect coefficient becomes the
estimated register gap for a firm AT THE SAMPLE MEAN size and R&D
intensity -- a real, interpretable point in the data -- rather than the
uncentered specification's implicit evaluation at log_size=0
(total assets of $1, ~5-10 orders of magnitude outside the actual
4.98-17.55 range) and rd_intensity=0 (a real but extreme low value).
This does not change the qa:rd_intensity or qa:log_size interaction
coefficients (mean-centering a variable involved in an interaction
shifts the main effects, not the slopes), so H3's central finding is
unaffected either way -- this only clarifies what "qa" itself means.

Reports both OLS and the grouped Binomial GLM (logit link, actual
successes/trials counts) on the centered specification, for direct
comparison against the uncentered numbers already reported.

Run:
    python analyze_h3_centered_regression.py
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

from analyze_h3_register_specificity import (
    build_section_panel, merge_firm_level_accounting, merge_lagged_accounting,
)


def fmt_stars(p):
    if p < 0.01:
        return "***"
    if p < 0.05:
        return "**"
    if p < 0.10:
        return "*"
    return ""


def center(df, rd_col):
    df = df.copy()
    df["log_size_c"] = df["log_size"] - df["log_size"].mean()
    df[f"{rd_col}_c"] = df[rd_col] - df[rd_col].mean()
    return df, f"{rd_col}_c"


def run_ols_centered(df, rd_col, extra_fe="", label=""):
    sub = df.dropna(subset=["share_operational", "qa", rd_col, "log_size", "gics_sector"]).copy()
    sub, rd_c = center(sub, rd_col)
    formula = f"share_operational ~ qa * {rd_c} + qa * log_size_c + C(gics_sector)"
    if extra_fe:
        formula += f" + {extra_fe}"
    model = smf.ols(formula, data=sub).fit(
        cov_type="cluster", cov_kwds={"groups": sub["ticker"]}
    )
    print(f"\n--- {label}: OLS, mean-centered (n={len(sub)}, "
          f"{sub['ticker'].nunique()} firms, firm-clustered SE) ---")
    for term in ["qa", rd_c, f"qa:{rd_c}", "log_size_c", "qa:log_size_c"]:
        if term in model.params.index:
            coef, p = model.params[term], model.pvalues[term]
            print(f"  {term:20s} coef={coef:+.5f}{fmt_stars(p)}  p={p:.4f}")
    return model


def run_binomial_glm_centered(df, rd_col, extra_fe="", label=""):
    sub = df.dropna(subset=["n_operational", "snippet_count", "qa", rd_col,
                             "log_size", "gics_sector"]).copy()
    sub = sub[sub["snippet_count"] > 0]
    sub, rd_c = center(sub, rd_col)

    cols = ["gics_sector"] + (["year"] if extra_fe else [])
    exog = sm.add_constant(
        pd.get_dummies(sub[cols], columns=cols, drop_first=True).astype(float)
    )
    exog["qa"] = sub["qa"].astype(float)
    exog[rd_c] = sub[rd_c]
    exog["log_size_c"] = sub["log_size_c"]
    exog[f"qa:{rd_c}"] = sub["qa"] * sub[rd_c]
    exog["qa:log_size_c"] = sub["qa"] * sub["log_size_c"]

    endog = np.column_stack([sub["n_operational"], sub["snippet_count"] - sub["n_operational"]])
    model = sm.GLM(endog, exog, family=sm.families.Binomial()).fit(
        cov_type="cluster", cov_kwds={"groups": sub["ticker"]}
    )
    print(f"\n--- {label}: Binomial GLM, mean-centered (n={len(sub)}, "
          f"{sub['ticker'].nunique()} firms, firm-clustered SE) ---")
    for term in ["qa", rd_c, f"qa:{rd_c}", "log_size_c", "qa:log_size_c"]:
        if term in model.params.index:
            coef, p = model.params[term], model.pvalues[term]
            print(f"  {term:20s} coef={coef:+.5f}{fmt_stars(p)}  p={p:.4f}")
    return model


def main():
    section_panel = build_section_panel()

    cross_panel = merge_firm_level_accounting(section_panel)
    print("=" * 15 + " CROSS-SECTIONAL " + "=" * 15)
    run_ols_centered(cross_panel, "mean_rd_intensity_winsorized", label="Cross-sectional")
    run_binomial_glm_centered(cross_panel, "mean_rd_intensity_winsorized", label="Cross-sectional")

    panel_panel = merge_lagged_accounting(section_panel)
    print("\n" + "=" * 15 + " FIRM-QUARTER PANEL " + "=" * 15)
    run_ols_centered(panel_panel, "rd_intensity_winsorized", extra_fe="C(year)",
                      label="Firm-quarter panel")
    run_binomial_glm_centered(panel_panel, "rd_intensity_winsorized", extra_fe="year",
                               label="Firm-quarter panel")


if __name__ == "__main__":
    main()
