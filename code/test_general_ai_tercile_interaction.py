"""
Formal test companion to check_general_ai_signalling_vs_timing.py: regresses
firm-quarter general_ai category share on R&D tercile x post-ChatGPT
interactions, with firm-clustered SEs, to test whether the post-ChatGPT
increase in general_ai share differs significantly across R&D terciles.

  - Signalling/cheap-talk predicts the interaction terms should be small
    and non-significant (uniform increase across terciles).
  - Timing/adoption-stage predicts a significant, negative interaction for
    Mid and especially High (smaller increase than Low, the reference
    category), since higher-R&D firms are further along their adoption
    curve and should show less drift toward genericism.

Run:
    python test_general_ai_tercile_interaction.py
"""

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

CATEGORIES = [
    "general_ai", "infrastructure_ai", "predictive_analytical_ai",
    "generative_ai", "autonomous_ai",
]
CHATGPT_SHOCK_DATE = pd.Timestamp("2022-11-30")


def load_rd_terciles():
    acct = pd.read_csv("../Data/firm_accounting_panel_rd_imputed.csv")
    firm = acct.groupby("ticker").agg(mean_rd_intensity=("rd_intensity", "mean")).reset_index()
    cap = firm["mean_rd_intensity"].quantile(0.95)
    firm["mean_rd_intensity"] = firm["mean_rd_intensity"].clip(upper=cap)
    firm["rd_tercile"] = pd.qcut(firm["mean_rd_intensity"], 3, labels=["Low", "Mid", "High"])
    return firm[["ticker", "rd_tercile"]]


def main():
    df = pd.read_csv("../Data/ai_volume_panel.csv", parse_dates=["date"])
    df = df[(df["date"] >= "2018-01-01") & (df["date"] <= "2026-06-30")]
    df["quarter"] = df["year"].astype(str) + "_" + df["period"]

    cat_cols = [f"n_ai_mentions_{c}" for c in CATEGORIES]
    fq = df.groupby(["symbol", "quarter"]).agg(
        **{c: (col, "sum") for c, col in zip(CATEGORIES, cat_cols)},
        date=("date", "min"),
    ).reset_index()

    fq["total_cat_mentions"] = fq[CATEGORIES].sum(axis=1)
    fq = fq[fq["total_cat_mentions"] > 0].copy()
    fq["general_ai_share"] = fq["general_ai"] / fq["total_cat_mentions"]
    fq["post_chatgpt"] = (fq["date"] >= CHATGPT_SHOCK_DATE).astype(int)

    rd = load_rd_terciles().rename(columns={"ticker": "symbol"})
    fq = fq.merge(rd, on="symbol", how="inner").dropna(subset=["rd_tercile"])
    fq["rd_tercile"] = pd.Categorical(fq["rd_tercile"], categories=["Low", "Mid", "High"], ordered=True)

    print(f"n firm-quarter observations: {len(fq)}, n firms: {fq['symbol'].nunique()}")
    print(fq.groupby("rd_tercile", observed=True)["symbol"].nunique(), "\n")

    model = smf.ols(
        "general_ai_share ~ C(rd_tercile, Treatment(reference='Low')) * post_chatgpt",
        data=fq,
    ).fit(cov_type="cluster", cov_kwds={"groups": fq["symbol"]})

    print(model.summary())

    print("\n--- Key coefficients (interaction terms) ---")
    for name in model.params.index:
        if ":post_chatgpt" in name or name == "post_chatgpt":
            coef = model.params[name]
            se = model.bse[name]
            p = model.pvalues[name]
            stars = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""
            print(f"  {name}: {coef:+.4f}{stars} (SE={se:.4f}, p={p:.4f})")

    interaction_terms = [n for n in model.params.index if ":post_chatgpt" in n]
    hypothesis = " = ".join([f"{t} = 0" for t in interaction_terms]).replace(" = 0 = 0", " = 0, ")
    hyp_str = ", ".join([f"{t} = 0" for t in interaction_terms])
    f_test = model.f_test(hyp_str)
    print(f"\nJoint F-test that Mid/High interaction terms = 0 (uniform-increase null): {f_test}")


if __name__ == "__main__":
    main()
