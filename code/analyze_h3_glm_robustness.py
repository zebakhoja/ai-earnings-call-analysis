"""
GLM robustness check on H3's register x capacity regression
(analyze_h3_register_specificity.py's run_regression(), the source of
Table 4.5: share_operational ~ qa * rd_intensity + qa * log_size +
sector [+ year]).

share_operational is a bounded proportion built from a real
successes/trials pair (n_operational snippets out of snippet_count
snippets, per firm-quarter-section cell), so the same concern that
motivated Table 4.4's H2 check applies here: OLS can predict outside
[0,1], understate heteroskedasticity from cells with very different
snippet_count, and miss nonlinear effects near the bounds.

The existing run_logit_regression() in analyze_h3_register_specificity.py
already checks this approximately, via a Haldane-Anscombe-corrected WLS
on the logit-transformed share. This script instead fits a proper grouped
Binomial GLM (logit link) directly on the actual (n_operational,
snippet_count - n_operational) counts -- not an approximation of the
proportion, the real trial-level model -- for both the cross-sectional
and firm-quarter panel specifications, to check whether the qa,
rd_intensity, qa:rd_intensity, log_size, and qa:log_size coefficients
(sign and significance) are robust to this functional-form change.

Run:
    python analyze_h3_glm_robustness.py
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm

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


def run_binomial_glm(df, rd_col, extra_fe="", label=""):
    sub = df.dropna(subset=["n_operational", "snippet_count", "qa", rd_col,
                             "log_size", "gics_sector"]).copy()
    sub = sub[sub["snippet_count"] > 0]
    if len(sub) < 50:
        print(f"\n{label} (Binomial GLM): insufficient data (n={len(sub)}), skipping")
        return None

    formula = f"qa * {rd_col} + qa * log_size + C(gics_sector)"
    if extra_fe:
        formula += f" + {extra_fe}"
    exog = sm.add_constant(
        pd.get_dummies(sub[["qa", rd_col, "log_size"] +
                            (["gics_sector"] if True else []) +
                            (["year"] if extra_fe else [])],
                       columns=["gics_sector"] + (["year"] if extra_fe else []),
                       drop_first=True).astype(float)
    )
    exog[f"qa:{rd_col}"] = sub["qa"] * sub[rd_col]
    exog["qa:log_size"] = sub["qa"] * sub["log_size"]

    endog = np.column_stack([sub["n_operational"], sub["snippet_count"] - sub["n_operational"]])
    model = sm.GLM(endog, exog, family=sm.families.Binomial()).fit(
        cov_type="cluster", cov_kwds={"groups": sub["ticker"]}
    )
    print(f"\n--- {label}: Binomial GLM (logit link), successes/trials "
          f"(n={len(sub)}, {sub['ticker'].nunique()} firms, firm-clustered SE) ---")
    for term in ["qa", rd_col, f"qa:{rd_col}", "log_size", "qa:log_size"]:
        if term in model.params.index:
            coef = model.params[term]
            p = model.pvalues[term]
            print(f"  {term:30s} coef={coef:+.5f}{fmt_stars(p)}  p={p:.4f}")
    return model


def main():
    section_panel = build_section_panel()

    cross_panel = merge_firm_level_accounting(section_panel)
    run_binomial_glm(cross_panel, "mean_rd_intensity_winsorized", label="Cross-sectional")

    panel_panel = merge_lagged_accounting(section_panel)
    run_binomial_glm(panel_panel, "rd_intensity_winsorized", extra_fe="year",
                      label="Firm-quarter panel")


if __name__ == "__main__":
    main()
