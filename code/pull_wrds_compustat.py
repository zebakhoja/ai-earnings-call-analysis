"""
Pull Compustat firm-size and R&D-intensity data via WRDS to unblock H2
(absorptive capacity: sector, size, R&D orientation) and the firm-
heterogeneity halves of H1/H3.

GICS sector was already available locally (msci_world_constituents.csv +
the extended/manually-approved mapping CSVs from the transcript-coverage
work) and did not need a WRDS pull -- see ticker_sector_mapping.csv. This
script covers the two things that DO need financial-statement data: firm
size and R&D intensity.

Design:
1. North America firms: comp.funda already carries `tic` directly, so
   query it filtered to our ticker list with the standard "primary,
   standardized, consolidated annual" filter (indfmt='INDL', datafmt='STD',
   popsrc='D', consol='C').
2. Global (non-North America) firms: comp.g_funda has NO `tic` column and
   no market-cap fields -- tickers are only in comp.g_security, joined on
   gvkey. Queried with the standard global filter (indfmt='INDL',
   datafmt='HIST_STD', popsrc='I', consol='C').
3. Size proxy: total assets (`at`), not market cap -- avoids needing a
   separate daily-price pull (comp.g_secd) just for market cap, and total
   assets/revenue are standard, well-precedented size proxies in the
   absorptive-capacity literature (Cohen & Levinthal itself uses firm size
   without committing to a single measure).
4. R&D intensity = xrd / sale (R&D expense / revenue), the standard
   normalisation -- computed per fiscal year, left as NaN where sale <= 0
   or xrd is missing (R&D expense is frequently unreported/blank for
   non-R&D-intensive sectors, not necessarily zero -- do not fill with 0
   without checking which is more defensible for the sectors in this
   sample).

Run:
    python pull_wrds_compustat.py --tickers "../Data/ticker_sector_mapping.csv" \
        --output "../Data/firm_accounting_panel.csv" --wrds-username zebakhoja
"""

import argparse

import pandas as pd
import wrds

MIN_FYEAR = 2016


def pull_north_america(db, tickers: list) -> pd.DataFrame:
    tic_list = ",".join(f"'{t}'" for t in tickers)
    query = f"""
        SELECT gvkey, tic, conm, fyear, datadate, at, sale, xrd, fic
        FROM comp.funda
        WHERE indfmt = 'INDL' AND datafmt = 'STD' AND popsrc = 'D' AND consol = 'C'
          AND fyear >= {MIN_FYEAR}
          AND tic IN ({tic_list})
    """
    df = db.raw_sql(query)
    df["source"] = "north_america"
    return df


def pull_global(db, tickers: list) -> pd.DataFrame:
    tic_list = ",".join(f"'{t}'" for t in tickers)
    sec_query = f"""
        SELECT DISTINCT gvkey, tic
        FROM comp.g_security
        WHERE tic IN ({tic_list})
    """
    sec = db.raw_sql(sec_query)
    if sec.empty:
        return pd.DataFrame()

    gvkey_list = ",".join(f"'{g}'" for g in sec["gvkey"].unique())
    query = f"""
        SELECT gvkey, conm, fyear, datadate, at, sale, xrd, fic, loc
        FROM comp.g_funda
        WHERE indfmt = 'INDL' AND datafmt = 'HIST_STD' AND popsrc = 'I' AND consol = 'C'
          AND fyear >= {MIN_FYEAR}
          AND gvkey IN ({gvkey_list})
    """
    df = db.raw_sql(query)
    df = df.merge(sec, on="gvkey", how="left")
    df["source"] = "global"
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", default="../Data/ticker_sector_mapping.csv")
    ap.add_argument("--output", default="../Data/firm_accounting_panel.csv")
    ap.add_argument("--wrds-username", required=True)
    args = ap.parse_args()

    tickers_df = pd.read_csv(args.tickers)
    tickers = sorted(tickers_df["ticker"].dropna().unique().tolist())
    print(f"Looking up {len(tickers)} tickers in Compustat...")

    db = wrds.Connection(wrds_username=args.wrds_username)
    try:
        na = pull_north_america(db, tickers)
        print(f"North America (comp.funda): {len(na)} firm-year rows, "
              f"{na['tic'].nunique() if not na.empty else 0} unique tickers")

        gl = pull_global(db, tickers)
        print(f"Global (comp.g_funda): {len(gl)} firm-year rows, "
              f"{gl['tic'].nunique() if not gl.empty else 0} unique tickers")
    finally:
        db.close()

    combined = pd.concat([na, gl], ignore_index=True, sort=False)
    combined = combined.rename(columns={"tic": "ticker"})
    combined["rd_intensity"] = combined["xrd"] / combined["sale"]
    combined.loc[combined["sale"] <= 0, "rd_intensity"] = pd.NA

    combined = combined.merge(
        tickers_df[["ticker", "gics_sector", "country"]], on="ticker", how="left"
    )

    combined.to_csv(args.output, index=False)
    matched = combined["ticker"].nunique()
    print(f"\nWrote {len(combined)} firm-year rows to {args.output}")
    print(f"Matched {matched}/{len(tickers)} tickers ({100*matched/len(tickers):.1f}%)")

    unmatched = set(tickers) - set(combined["ticker"])
    if unmatched:
        print(f"Unmatched tickers (sample): {sorted(unmatched)[:20]}")


if __name__ == "__main__":
    main()
