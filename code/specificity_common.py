"""
Shared column-mapping helper for the specificity-classifier pipeline
(00-04, originally designed in a separate session against a placeholder
schema: ticker/firm_name/quarter/call_date/section/speaker_role/text/
gics_sector).

Our actual corpus is Data/ai_documents.csv (from extract_ai_documents.py),
with columns: symbol, year, period, date, section, speaker, matched_term,
category, specificity, match_start, match_end, snippet_id, snippet_text.

This maps our real schema onto the pipeline's expected names so 02/03/04
work unchanged, and documents what's missing:
    - firm_name: not present (only ticker/symbol). Harmless -- downstream
      scripts already do `col in df.columns` checks before using it.
    - speaker_role: not present -- `speaker` (a name, not a role label) is
      kept as-is but not renamed, since nothing downstream currently uses it.
    - gics_sector: NOT YET AVAILABLE. The WRDS Compustat/GICS pull is a
      separate, still-outstanding task. Sector-stratified sampling in
      01_sample_for_annotation.py is degraded to period-only stratification
      until that data exists -- see the fallback there. Re-run 01 with
      proper sector stratification once GICS codes are merged in, if a
      stratification-driven sample composition materially matters.

Note: our own `specificity` column (values "generic"/"specific", the
AI-terms dictionary tag) is UNRELATED to this pipeline's 0/1/2
`specificity_score` classifier output -- both are kept, under different
names, no collision.
"""

import pandas as pd


def load_standardized_corpus(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.rename(columns={"symbol": "ticker", "date": "call_date", "snippet_text": "text"})
    df["quarter"] = df["year"].astype(str) + "_" + df["period"]
    return df
