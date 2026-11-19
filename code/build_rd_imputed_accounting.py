"""
Builds an R&D-imputed version of firm_accounting_panel_combined.csv:
firm-years missing `rd_intensity` (3,272 of 7,736 rows) are filled with the
mean rd_intensity among REPORTING firm-years in the same (gics_sector,
country) cell, falling back to the sector mean, then the overall mean, if
a cell has no reporters. This is the "sector x country mean, including
Financials/Utilities" imputation agreed on in place of dropping missing
R&D firms entirely (Table 4.3's prior approach) or a harsher R&D=0 fill.

Feeds both the firm-level cross-section (analyze_h2_heterogeneity.py) and
the firm-quarter panel (analyze_h2_firm_quarter_panel.py) from the same
imputed source, so the two specifications stay consistent with each other.

Run:
    python build_rd_imputed_accounting.py
"""

import pandas as pd

INPUT = "../Data/firm_accounting_panel_combined.csv"
OUTPUT = "../Data/firm_accounting_panel_rd_imputed.csv"


def main():
    acct = pd.read_csv(INPUT, parse_dates=["datadate"])
    country = pd.read_csv("../Data/ticker_sector_mapping.csv")[["ticker", "country"]]
    acct = acct.merge(country, on="ticker", how="left")

    reporters = acct[acct["rd_intensity"].notna()]
    cell_mean = reporters.groupby(["gics_sector", "country"])["rd_intensity"].mean()
    sector_mean = reporters.groupby("gics_sector")["rd_intensity"].mean()
    overall_mean = reporters["rd_intensity"].mean()

    missing_mask = acct["rd_intensity"].isna()
    n_missing = missing_mask.sum()

    def fill_for_row(row):
        val = cell_mean.get((row["gics_sector"], row["country"]))
        if pd.isna(val):
            val = sector_mean.get(row["gics_sector"])
        if pd.isna(val):
            val = overall_mean
        return val

    fill_values = acct.loc[missing_mask].apply(fill_for_row, axis=1)
    n_cell_hit = acct.loc[missing_mask].apply(
        lambda r: not pd.isna(cell_mean.get((r["gics_sector"], r["country"]))), axis=1
    ).sum()

    acct.loc[missing_mask, "rd_intensity"] = fill_values
    acct["rd_intensity_imputed"] = missing_mask

    print(f"{n_missing} of {len(acct)} firm-year rows had missing rd_intensity")
    print(f"  {n_cell_hit} filled via exact (sector, country) cell mean")
    print(f"  {n_missing - n_cell_hit} fell back to sector-only or overall mean")

    acct.to_csv(OUTPUT, index=False)
    print(f"Saved {OUTPUT}")


if __name__ == "__main__":
    main()
