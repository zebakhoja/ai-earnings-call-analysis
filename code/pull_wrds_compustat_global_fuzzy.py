"""
Extend the Compustat pull (pull_wrds_compustat.py) to the 164 firms that
didn't match on ticker -- almost all non-US/Canada, since our tickers for
those firms are FMP's US OTC/pink-sheet symbols, not the companies' home-
exchange tickers Compustat Global's `tic` field is keyed to (see
H2 Data Preparation.docx for the full diagnosis).

Method: fuzzy company-name matching against comp.g_company, restricted by
country (mapped to ISO3 via COUNTRY_TO_ISO3) to narrow candidates and avoid
cross-country false positives -- the same name_similarity() approach
fmp_utils.py already used for the original ticker-resolution phase (that
phase's accepted matches ranged ~0.53-0.85 similarity, informing the
thresholds here).

Three-tier output, mirroring the project's existing resolution convention
(_manually_approved_mapping.csv / _needs_manual_review.csv):
    - similarity >= 0.75: auto-accepted, pulled into the accounting panel
    - 0.55 <= similarity < 0.75: written to a review file, NOT auto-included
    - < 0.55: left unmatched

Run:
    python pull_wrds_compustat_global_fuzzy.py --wrds-username zebakhoja
"""

import argparse
import difflib

import pandas as pd
import wrds

ACCEPT_THRESHOLD = 0.75
REVIEW_THRESHOLD = 0.55
MIN_FYEAR = 2016

COUNTRY_TO_ISO3 = {
    "United Kingdom": "GBR", "Japan": "JPN", "Germany": "DEU",
    "Switzerland": "CHE", "Italy": "ITA", "Australia": "AUS",
    "Sweden": "SWE", "Finland": "FIN", "Singapore": "SGP",
    "Norway": "NOR", "Denmark": "DNK", "Canada": "CAN",
    "Spain": "ESP", "Ireland": "IRL", "France": "FRA",
    "Israel": "ISR", "United States": "USA", "Austria": "AUT",
    "New Zealand": "NZL", "Belgium": "BEL",
}


def name_similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio()


def find_unmatched(accounting_path: str, mapping_path: str) -> pd.DataFrame:
    acct = pd.read_csv(accounting_path)
    mapping = pd.read_csv(mapping_path)
    matched_tickers = set(acct["ticker"].unique())
    return mapping[~mapping["ticker"].isin(matched_tickers)].copy()


def fetch_candidates(db, iso3: str) -> pd.DataFrame:
    query = f"""
        SELECT gvkey, conm, conml, loc
        FROM comp.g_company
        WHERE loc = '{iso3}'
    """
    return db.raw_sql(query)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--accounting", default="../Data/firm_accounting_panel.csv")
    ap.add_argument("--mapping", default="../Data/ticker_sector_mapping.csv")
    ap.add_argument("--output-accepted", default="../Data/firm_accounting_panel_global_fuzzy.csv")
    ap.add_argument("--output-review", default="../Data/global_fuzzy_needs_review.csv")
    ap.add_argument("--wrds-username", required=True)
    args = ap.parse_args()

    unmatched = find_unmatched(args.accounting, args.mapping)
    print(f"{len(unmatched)} unmatched firms to attempt fuzzy matching")

    db = wrds.Connection(wrds_username=args.wrds_username)

    candidate_cache = {}
    accepted_rows = []
    review_rows = []
    no_match = []

    try:
        for _, row in unmatched.iterrows():
            country = row["country"]
            iso3 = COUNTRY_TO_ISO3.get(country)
            if iso3 is None:
                no_match.append({**row, "reason": f"no ISO3 mapping for '{country}'"})
                continue

            if iso3 not in candidate_cache:
                candidate_cache[iso3] = fetch_candidates(db, iso3)
            candidates = candidate_cache[iso3]
            if candidates.empty:
                no_match.append({**row, "reason": f"no comp.g_company rows for {iso3}"})
                continue

            candidates = candidates.copy()
            candidates["sim_conm"] = candidates["conm"].apply(lambda c: name_similarity(row["company_name"], c))
            candidates["sim_conml"] = candidates["conml"].apply(lambda c: name_similarity(row["company_name"], c))
            candidates["sim"] = candidates[["sim_conm", "sim_conml"]].max(axis=1)
            best = candidates.loc[candidates["sim"].idxmax()]

            record = {
                "ticker": row["ticker"], "company_name": row["company_name"],
                "country": country, "matched_gvkey": best["gvkey"],
                "matched_conm": best["conm"], "similarity": round(best["sim"], 4),
            }
            if best["sim"] >= ACCEPT_THRESHOLD:
                accepted_rows.append(record)
            elif best["sim"] >= REVIEW_THRESHOLD:
                review_rows.append(record)
            else:
                no_match.append({**row, "reason": f"best similarity {best['sim']:.3f} below review threshold"})

        print(f"Accepted (>= {ACCEPT_THRESHOLD}): {len(accepted_rows)}")
        print(f"Needs review ({REVIEW_THRESHOLD}-{ACCEPT_THRESHOLD}): {len(review_rows)}")
        print(f"No match: {len(no_match)}")

        if review_rows:
            pd.DataFrame(review_rows).to_csv(args.output_review, index=False)
            print(f"Wrote review candidates to {args.output_review}")

        if not accepted_rows:
            print("No accepted matches -- nothing to pull financials for.")
            return

        accepted_df = pd.DataFrame(accepted_rows)
        gvkey_list = ",".join(f"'{g}'" for g in accepted_df["matched_gvkey"].unique())
        fin_query = f"""
            SELECT gvkey, conm, fyear, datadate, at, sale, xrd, fic, loc
            FROM comp.g_funda
            WHERE indfmt = 'INDL' AND datafmt = 'HIST_STD' AND popsrc = 'I' AND consol = 'C'
              AND fyear >= {MIN_FYEAR}
              AND gvkey IN ({gvkey_list})
        """
        fin = db.raw_sql(fin_query)
    finally:
        db.close()

    fin = fin.rename(columns={"gvkey": "matched_gvkey"})
    combined = accepted_df.merge(fin, on="matched_gvkey", how="left")
    combined = combined.rename(columns={"matched_gvkey": "gvkey"})
    combined["rd_intensity"] = combined["xrd"] / combined["sale"]
    combined.loc[combined["sale"] <= 0, "rd_intensity"] = pd.NA
    combined["source"] = "global_fuzzy"

    mapping = pd.read_csv(args.mapping)[["ticker", "gics_sector"]]
    combined = combined.merge(mapping, on="ticker", how="left")

    combined.to_csv(args.output_accepted, index=False)
    print(f"\nWrote {len(combined)} firm-year rows for {combined['ticker'].nunique()} "
          f"newly-matched tickers to {args.output_accepted}")


if __name__ == "__main__":
    main()
