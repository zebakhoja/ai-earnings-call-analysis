"""
H3 (Signalling Differences by Speaker): does the prepared-remarks/Q&A
specificity gap shrink for higher-absorptive-capacity firms?

H3 (as currently stated): For firms with lower absorptive capacity, AI
discourse is more specific in prepared remarks than in the Q&A section,
while there is a substantially smaller gap between the two in the case
of higher capacity firms.

This merges three things that already existed separately:
    - ai_snippets_scored.csv: per-snippet specificity_score (0/1/2) and
      section label (prepared_remarks / qa), from the specificity
      classifier pipeline.
    - firm_accounting_panel_combined.csv: R&D intensity and total assets
      (size), the same source used for H2.
    - ticker_sector_mapping.csv: GICS sector.

...into a firm-quarter-SECTION panel (two rows per firm-quarter: one for
prepared_remarks, one for qa) and tests the QA x R&D-intensity
interaction that actually constitutes H3, in a cross-sectional
specification (firm-average R&D/size, mirroring H2's Table 4.3 firm-level
row) and a firm-quarter panel specification with lagged time-varying
R&D/size and year fixed effects (mirroring H2's panel robustness check).

A winsorization sensitivity sweep on R&D intensity is run on the
interaction term specifically, since H2 already found the R&D-to-volume
result (not the R&D-to-specificity result) was sensitive to this choice.

Run:
    python analyze_h3_register_specificity.py
"""

import re

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

MIN_SNIPPETS_PER_CELL = 5
RD_WINSOR_PCTILE = 0.95


TURN_RE = re.compile(r"(?m)^[A-Z][^\n:]{2,80}:")


def build_section_panel(clean_only=False):
    """Firm-quarter-section panel: two rows per firm-quarter (prepared_remarks, qa).

    clean_only=True drops snippets whose wide (+/-1 sentence) extraction window
    crosses into an adjacent speaker turn (a second "Name:" marker embedded in
    the snippet text) -- a window-contamination check on the specificity
    measure, parallel to the sentence-window contamination already found and
    corrected for in the FinBERT sentiment analysis (H3 Sentiment Results.docx).
    This artefact is far more common in Q&A (22% of snippets) than prepared
    remarks (2.7%), since Q&A has many more speaker changes per unit of text --
    a plausible mechanical contributor to the unconditional register gap (beta1).
    """
    df = pd.read_csv("../Data/ai_snippets_scored.csv", parse_dates=["call_date"])
    df = df[df["section"].isin(["prepared_remarks", "qa"])].copy()

    if clean_only:
        crosses_turn = df["text"].str.contains(TURN_RE, regex=True, na=False)
        n_before = len(df)
        df = df[~crosses_turn].copy()
        print(f"clean_only filter: {n_before} -> {len(df)} snippets "
              f"({n_before - len(df)} dropped for embedded turn-boundary markers)")

    panel = (
        df.groupby(["ticker", "quarter", "section"])
        .agg(
            snippet_count=("specificity_score", "size"),
            n_operational=("specificity_score", lambda s: (s == 2).sum()),
            mean_specificity=("specificity_score", "mean"),
            share_operational=("specificity_score", lambda s: (s == 2).mean()),
            call_date=("call_date", "min"),
        )
        .reset_index()
    )

    n_before = len(panel)
    panel = panel[panel["snippet_count"] >= MIN_SNIPPETS_PER_CELL].copy()
    print(f"Firm-quarter-section cells: {n_before} -> {len(panel)} after dropping "
          f"n<{MIN_SNIPPETS_PER_CELL} (H1's quarterly composition trend applied no such "
          f"per-cell floor, only an overall date-range filter -- this is a fresh design "
          f"choice for this cell-level analysis, not a reused threshold)")

    panel["qa"] = (panel["section"] == "qa").astype(int)
    panel["year"] = panel["quarter"].str.split("_").str[0].astype(int)
    return panel


def merge_firm_level_accounting(panel):
    """Cross-sectional covariates: firm-average R&D intensity/size, mirroring
    analyze_h2_heterogeneity.py's build_firm_panel()."""
    acct = pd.read_csv("../Data/firm_accounting_panel_combined.csv")
    acct_firm = acct.groupby("ticker").agg(
        mean_at=("at", "mean"),
        mean_rd_intensity=("rd_intensity", "mean"),
    ).reset_index()
    cap = acct_firm["mean_rd_intensity"].quantile(RD_WINSOR_PCTILE)
    acct_firm["mean_rd_intensity_winsorized"] = acct_firm["mean_rd_intensity"].clip(upper=cap)
    acct_firm["log_size"] = np.log(acct_firm["mean_at"])

    sector = pd.read_csv("../Data/ticker_sector_mapping.csv")[["ticker", "gics_sector"]]

    merged = panel.merge(acct_firm, on="ticker", how="left")
    merged = merged.merge(sector, on="ticker", how="left")
    return merged


def merge_lagged_accounting(panel):
    """Panel covariates: lagged time-varying R&D intensity/size via merge_asof,
    mirroring analyze_h2_firm_quarter_panel.py's merge_lagged_accounting()."""
    acct = pd.read_csv("../Data/firm_accounting_panel_combined.csv", parse_dates=["datadate"])
    acct = acct.dropna(subset=["datadate", "at"]).sort_values("datadate")

    cap = acct["rd_intensity"].quantile(RD_WINSOR_PCTILE)
    acct["rd_intensity_winsorized"] = acct["rd_intensity"].clip(upper=cap)

    panel = panel.sort_values("call_date")
    merged = pd.merge_asof(
        panel, acct[["ticker", "datadate", "at", "rd_intensity_winsorized"]],
        left_on="call_date", right_on="datadate", by="ticker", direction="backward",
    )
    merged["log_size"] = np.log(merged["at"])

    sector = pd.read_csv("../Data/ticker_sector_mapping.csv")[["ticker", "gics_sector"]]
    merged = merged.merge(sector, on="ticker", how="left")
    return merged


def run_regression(df, dv, rd_col, extra_fe="", label=""):
    sub = df.dropna(subset=[dv, "qa", rd_col, "log_size", "gics_sector"]).copy()
    if len(sub) < 50:
        print(f"\n{label} -- {dv}: insufficient data (n={len(sub)}), skipping")
        return None
    formula = f"{dv} ~ qa * {rd_col} + qa * log_size + C(gics_sector)"
    if extra_fe:
        formula += f" + {extra_fe}"
    model = smf.ols(formula, data=sub).fit(
        cov_type="cluster", cov_kwds={"groups": sub["ticker"]}
    )
    print(f"\n--- {label}: DV = {dv} (n={len(sub)}, {sub['ticker'].nunique()} firms, "
          f"firm-clustered SE) ---")
    for term in ["qa", rd_col, f"qa:{rd_col}", "log_size", "qa:log_size"]:
        if term in model.params.index:
            print(f"  {term:30s} coef={model.params[term]:+.5f}  p={model.pvalues[term]:.4f}")
    print(f"  R-squared: {model.rsquared:.4f}")
    return model


def run_logit_regression(df, rd_col, extra_fe="", label=""):
    """Ceiling/floor-effect check: WLS on the logit-transformed operational
    share (Haldane-Anscombe continuity correction: (n_op + 0.5) / (n + 1)),
    weighted by snippet_count so noisier small cells count less. A raw-share
    linear model treats a drop from 0.52->0.27 and a drop from 0.33->0.14 as
    similar in scale; the logit transform instead measures proportional
    movement, which is the more appropriate scale if firms starting from a
    higher prepared-remarks baseline mechanically have more room to fall."""
    sub = df.dropna(subset=["snippet_count", "n_operational", "qa", rd_col, "log_size", "gics_sector"]).copy()
    if len(sub) < 50:
        print(f"\n{label} (logit scale): insufficient data (n={len(sub)}), skipping")
        return None
    p = (sub["n_operational"] + 0.5) / (sub["snippet_count"] + 1)
    sub["logit_share_operational"] = np.log(p / (1 - p))

    formula = f"logit_share_operational ~ qa * {rd_col} + qa * log_size + C(gics_sector)"
    if extra_fe:
        formula += f" + {extra_fe}"
    model = smf.wls(formula, data=sub, weights=sub["snippet_count"]).fit(
        cov_type="cluster", cov_kwds={"groups": sub["ticker"]}
    )
    print(f"\n--- {label}: DV = logit(share_operational), weighted WLS "
          f"(n={len(sub)}, {sub['ticker'].nunique()} firms, firm-clustered SE) ---")
    for term in ["qa", rd_col, f"qa:{rd_col}", "log_size", "qa:log_size"]:
        if term in model.params.index:
            print(f"  {term:30s} coef={model.params[term]:+.5f}  p={model.pvalues[term]:.4f}")
    print(f"  R-squared: {model.rsquared:.4f}")
    return model


def report_proportional_retention(cross_panel):
    """Formalises the by-hand check: what share of prepared-remarks
    specificity survives into Q&A, by R&D tercile."""
    firm_rd = cross_panel.drop_duplicates("ticker")[["ticker", "mean_rd_intensity_winsorized"]].dropna()
    firm_rd["rd_tercile"] = pd.qcut(
        firm_rd["mean_rd_intensity_winsorized"], 3, labels=["Low R&D", "Mid R&D", "High R&D"]
    )
    plot_df = cross_panel.merge(firm_rd[["ticker", "rd_tercile"]], on="ticker", how="left")
    grouped = plot_df.groupby(["rd_tercile", "section"], observed=True)["share_operational"].mean().unstack()
    grouped = grouped[["prepared_remarks", "qa"]]
    grouped["absolute_drop"] = grouped["prepared_remarks"] - grouped["qa"]
    grouped["pct_retained"] = grouped["qa"] / grouped["prepared_remarks"]

    print("\n=== Proportional retention by R&D tercile (prepared remarks -> Q&A) ===")
    print(grouped)
    print("\nH3 needs pct_retained to rise monotonically with R&D tercile (less lost under scrutiny).")


def run_unconditional(df, dv, label=""):
    sub = df.dropna(subset=[dv, "qa", "gics_sector"]).copy()
    model = smf.ols(f"{dv} ~ qa + C(gics_sector)", data=sub).fit(
        cov_type="cluster", cov_kwds={"groups": sub["ticker"]}
    )
    print(f"\n--- {label}: unconditional QA effect on {dv} (n={len(sub)}) ---")
    print(f"  qa coef={model.params['qa']:+.5f}  p={model.pvalues['qa']:.4f}")
    return model


def winsorization_sweep(section_panel):
    """Sweep RD_WINSOR_PCTILE and report the QA x R&D interaction term
    (cross-sectional specification) at each threshold."""
    print("\n=== Winsorization sensitivity sweep (cross-sectional spec, "
          "share_operational DV) ===")
    acct = pd.read_csv("../Data/firm_accounting_panel_combined.csv")
    acct_firm_raw = acct.groupby("ticker").agg(
        mean_at=("at", "mean"), mean_rd_intensity=("rd_intensity", "mean"),
    ).reset_index()
    sector = pd.read_csv("../Data/ticker_sector_mapping.csv")[["ticker", "gics_sector"]]

    thresholds = {"none (no cap)": None, "90th": 0.90, "95th": 0.95,
                  "99th": 0.99, "hard cap at 1.0": "hard1"}
    for name, pct in thresholds.items():
        acct_firm = acct_firm_raw.copy()
        if pct is None:
            acct_firm["rd_w"] = acct_firm["mean_rd_intensity"]
        elif pct == "hard1":
            acct_firm["rd_w"] = acct_firm["mean_rd_intensity"].clip(upper=1.0)
        else:
            cap = acct_firm["mean_rd_intensity"].quantile(pct)
            acct_firm["rd_w"] = acct_firm["mean_rd_intensity"].clip(upper=cap)
        acct_firm["log_size"] = np.log(acct_firm["mean_at"])

        merged = section_panel.drop(
            columns=["mean_rd_intensity_winsorized", "log_size", "gics_sector",
                     "mean_at", "mean_rd_intensity"], errors="ignore",
        ).merge(acct_firm, on="ticker", how="left").merge(sector, on="ticker", how="left")

        sub = merged.dropna(subset=["share_operational", "qa", "rd_w", "log_size", "gics_sector",
                                     "snippet_count", "n_operational"]).copy()
        model = smf.ols(
            "share_operational ~ qa * rd_w + qa * log_size + C(gics_sector)", data=sub
        ).fit(cov_type="cluster", cov_kwds={"groups": sub["ticker"]})
        term = "qa:rd_w"
        coef = model.params.get(term, float("nan"))
        p = model.pvalues.get(term, float("nan"))

        p_hat = (sub["n_operational"] + 0.5) / (sub["snippet_count"] + 1)
        sub["logit_share_operational"] = np.log(p_hat / (1 - p_hat))
        logit_model = smf.wls(
            "logit_share_operational ~ qa * rd_w + qa * log_size + C(gics_sector)",
            data=sub, weights=sub["snippet_count"],
        ).fit(cov_type="cluster", cov_kwds={"groups": sub["ticker"]})
        logit_coef = logit_model.params.get(term, float("nan"))
        logit_p = logit_model.pvalues.get(term, float("nan"))

        print(f"  {name:20s} n={len(sub):5d}  raw coef={coef:+.5f} p={p:.4f}   |   "
              f"logit coef={logit_coef:+.5f} p={logit_p:.4f}")


def make_slope_plot(cross_panel):
    import matplotlib.pyplot as plt

    firm_rd = cross_panel.drop_duplicates("ticker")[["ticker", "mean_rd_intensity_winsorized"]].dropna()
    firm_rd["rd_tercile"] = pd.qcut(
        firm_rd["mean_rd_intensity_winsorized"], 3, labels=["Low R&D", "Mid R&D", "High R&D"]
    )
    plot_df = cross_panel.merge(firm_rd[["ticker", "rd_tercile"]], on="ticker", how="left")

    grouped = plot_df.groupby(["rd_tercile", "section"], observed=True)["share_operational"].mean().unstack()
    grouped = grouped[["prepared_remarks", "qa"]]

    fig, ax = plt.subplots(figsize=(6, 4.5))
    colors = {"Low R&D": "#c0392b", "Mid R&D": "#7f8c8d", "High R&D": "#2980b9"}
    for tercile in ["Low R&D", "Mid R&D", "High R&D"]:
        if tercile in grouped.index:
            ax.plot(
                ["Prepared remarks", "Q&A"], grouped.loc[tercile].values,
                marker="o", linewidth=2.5, markersize=8,
                color=colors[tercile], label=tercile,
            )
    ax.set_ylabel("Share operationally specific")
    ax.set_title("Specificity by register, split by R&D-intensity tercile")
    ax.legend(title="Firm R&D intensity")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig("../Data/h3_register_specificity_by_rd_tercile.png", dpi=150)
    print("\nSaved plot to ../Data/h3_register_specificity_by_rd_tercile.png")
    print(grouped)


def main():
    section_panel = build_section_panel()

    cross_panel = merge_firm_level_accounting(section_panel)
    cross_panel.to_csv("../Data/h3_firm_quarter_section_panel.csv", index=False)

    run_unconditional(cross_panel, "share_operational", label="Cross-sectional")
    run_unconditional(cross_panel, "mean_specificity", label="Cross-sectional")

    run_regression(cross_panel, "share_operational", "mean_rd_intensity_winsorized",
                    label="Cross-sectional")
    run_regression(cross_panel, "mean_specificity", "mean_rd_intensity_winsorized",
                    label="Cross-sectional")

    panel_panel = merge_lagged_accounting(section_panel)
    run_regression(panel_panel, "share_operational", "rd_intensity_winsorized",
                    extra_fe="C(year)", label="Firm-quarter panel")
    run_regression(panel_panel, "mean_specificity", "rd_intensity_winsorized",
                    extra_fe="C(year)", label="Firm-quarter panel")

    run_logit_regression(cross_panel, "mean_rd_intensity_winsorized",
                          label="Cross-sectional")
    run_logit_regression(panel_panel, "rd_intensity_winsorized",
                          extra_fe="C(year)", label="Firm-quarter panel")

    report_proportional_retention(cross_panel)

    winsorization_sweep(section_panel)

    print("\n=== Window-contamination robustness check (clean_only, excludes "
          "snippets whose extraction window crosses a speaker-turn boundary) ===")
    clean_section_panel = build_section_panel(clean_only=True)
    clean_cross_panel = merge_firm_level_accounting(clean_section_panel)
    run_unconditional(clean_cross_panel, "share_operational", label="Clean-window cross-sectional")
    run_regression(clean_cross_panel, "share_operational", "mean_rd_intensity_winsorized",
                    label="Clean-window cross-sectional")
    clean_panel_panel = merge_lagged_accounting(clean_section_panel)
    run_regression(clean_panel_panel, "share_operational", "rd_intensity_winsorized",
                    extra_fe="C(year)", label="Clean-window firm-quarter panel")

    make_slope_plot(cross_panel)


if __name__ == "__main__":
    main()
