"""
Shared logic for the FMP earnings-call-transcript extraction pipeline.

Why this file exists
--------------------
All the reusable functions live here instead of in notebook cells. When logic
needs changing, edit THIS file and just re-run the notebook's reload cell:

    import importlib, fmp_utils
    importlib.reload(fmp_utils)

That re-imports the updated code in a couple of seconds -- no need to re-run
the whole notebook every time. The notebook keeps only config + the data
variables produced by each step.

API-hitting logic is wrapped in FMPClient so the call counter / budget /
api key are instance state rather than module globals (globals are what made
resetting the counter and reloading fiddly before).
"""

import csv
import difflib
import json
import re
import time
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

import requests


BASE_URL = "https://financialmodelingprep.com/stable"

YEAR_START = 2018
YEAR_END = date.today().year

US_EXCHANGE_HINTS = ["nyse", "nasdaq", "amex", "otc", "cboe", "nyse american"]

RESOLVED_FIELDNAMES = [
    "ticker", "expected_name", "expected_country", "resolved_symbol", "resolved_name",
    "resolved_exchange", "similarity", "on_us_exchange", "method", "n_candidates",
]

TIER1_MARKERS = [
    r"(?:our|the|next) first question",
    r"first questions?,? please",
    r"we'?ll take our first question",
]

TIER2_MARKERS = [
    r"(?:we would like|we'?ll|we will|let'?s)\s*(?:now\s*)?(?:take|open\s*(?:up\s*)?(?:the\s*(?:call|floor)\s*)?(?:for|to))\s*(?:any\s*|your\s*)?questions",
    r"turn\s*(?:it\s*|the call\s*)?(?:over\s*)?to the operator for questions",
    r"queue up the q\s*&\s*a",
]

_NOT_A_QA_SESSION = r"(?<!no )(?<!not any )(?<!without a )"

TIER3_MARKERS = [
    _NOT_A_QA_SESSION + r"question-and-answer session",
    _NOT_A_QA_SESSION + r"questions?\s*and\s*answers?\s*session",
    _NOT_A_QA_SESSION + r"q\s*&\s*a session",
]

_CONCLUDE_NEAR_RE = re.compile(r"(?:conclud\w*|wraps? up|that'?s all the time)", re.IGNORECASE)

QA_SECTION_MARKERS = TIER1_MARKERS + TIER2_MARKERS + TIER3_MARKERS


def is_ambiguous(row: dict) -> bool:
    """True if this ticker row was flagged as a collision. CSV round-trips
    booleans as the strings 'True'/'False' (and 'False' is truthy!), so accept
    both the real bool and the string."""
    val = row.get("fmp_ambiguous_symbol", False)
    return val is True or str(val).strip().lower() == "true"


def name_similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio()


def is_us_exchange(exch: str) -> bool:
    exch = (exch or "").lower()
    return any(hint in exch for hint in US_EXCHANGE_HINTS)


def split_prepared_remarks_qa(content: str):
    """Returns (prepared_remarks, qa, method). See TIER1/2/3_MARKERS comment
    above for why this is tiered rather than a single flat priority list."""
    if not content:
        return "", "", "empty"

    for tier_name, tier in (("tier1", TIER1_MARKERS), ("tier2", TIER2_MARKERS)):
        best = None
        for pattern in tier:
            m = re.search(pattern, content, re.IGNORECASE)
            if m and (best is None or m.start() < best):
                best = m.start()
        if best is not None:
            return content[:best], content[best:], f"heuristic_marker_{tier_name}"

    n = len(content)
    candidates = []
    for pattern in TIER3_MARKERS:
        for m in re.finditer(pattern, content, re.IGNORECASE):
            if m.start() < 0.1 * n:
                continue
            preceding = content[max(0, m.start() - 60):m.start()]
            if _CONCLUDE_NEAR_RE.search(preceding):
                continue
            candidates.append(m.start())

    if candidates:
        best = min(candidates)
        return content[:best], content[best:], "heuristic_marker_tier3"

    return content, "", "marker_not_found"


def _normalize_resolved_row(row: dict) -> dict:
    """Coerce a resolved row back to real types whether it came straight from
    FMPClient.verify_symbol() (already typed) or was reloaded from CSV
    (everything is a string)."""
    def to_bool(v):
        return v is True or str(v).strip().lower() == "true"

    def to_float(v):
        try:
            return float(v)
        except (TypeError, ValueError):
            return 0.0

    def to_int(v):
        try:
            return int(v)
        except (TypeError, ValueError):
            return 0

    def none_if_blank(v):
        return v if v not in (None, "", "None") else None

    return {
        "ticker": row["ticker"],
        "expected_name": row.get("expected_name"),
        "expected_country": row.get("expected_country"),
        "resolved_symbol": none_if_blank(row.get("resolved_symbol")),
        "resolved_name": none_if_blank(row.get("resolved_name")),
        "resolved_exchange": none_if_blank(row.get("resolved_exchange")),
        "similarity": to_float(row.get("similarity")),
        "on_us_exchange": to_bool(row.get("on_us_exchange")),
        "method": row.get("method"),
        "n_candidates": to_int(row.get("n_candidates")),
    }


def build_ticker_list_from_ishares(ishares_src: Path, ticker_list_path: Path,
                                   collisions_path: Path) -> dict:
    """Parse the iShares Core MSCI World holdings export into a clean ticker
    list, flagging bare-ticker collisions. Writes ticker_list_path and
    collisions_path; returns a small summary dict."""
    ishares_src = Path(ishares_src)
    with open(ishares_src, encoding="utf-8-sig") as f:
        lines = f.readlines()

    reader = csv.reader([lines[2]] + lines[3:])
    rows = list(reader)
    header = rows[0]
    data = [r for r in rows[1:] if len(r) == len(header)]

    col = {name: header.index(name)
           for name in ["Ticker", "Name", "Sector", "Asset Class", "Location", "Exchange"]}
    equity = [r for r in data if r[col["Asset Class"]] == "Equity"]

    by_ticker = defaultdict(list)
    for r in equity:
        by_ticker[r[col["Ticker"]]].append(r)

    def make_row(r, ambiguous):
        return {
            "ticker": r[col["Ticker"]], "company_name": r[col["Name"]],
            "gics_sector": r[col["Sector"]], "country": r[col["Location"]],
            "exchange": r[col["Exchange"]], "fmp_ambiguous_symbol": ambiguous,
        }

    clean_rows, collision_rows = [], []
    for ticker, group in by_ticker.items():
        if len({r[col["Name"]] for r in group}) == 1:
            clean_rows.append(make_row(group[0], False))
            continue

        us_rows = [r for r in group if r[col["Location"]] == "United States"]
        non_us_rows = [r for r in group if r[col["Location"]] != "United States"]
        if us_rows:
            clean_rows.append(make_row(us_rows[0], False))
            flagged = non_us_rows
        else:
            flagged = group

        for r in flagged:
            row = make_row(r, True)
            clean_rows.append(row)
            collision_rows.append(row)

    clean_rows.sort(key=lambda r: r["ticker"])
    fieldnames = ["ticker", "company_name", "gics_sector", "country", "exchange", "fmp_ambiguous_symbol"]

    for path, rows_out in [(ticker_list_path, clean_rows), (collisions_path, collision_rows)]:
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows_out)

    return {
        "equity_holdings": len(equity),
        "ticker_rows": len(clean_rows),
        "ambiguous": sum(r["fmp_ambiguous_symbol"] for r in clean_rows),
        "collisions": len(collision_rows),
        "countries": len({r["country"] for r in clean_rows}),
    }


def load_ticker_list(path: Path) -> list:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"No ticker list found at {path}.\n"
            "Run build_ticker_list_from_ishares() first (Step 0)."
        )
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    if not rows or "ticker" not in rows[0]:
        found = list(rows[0].keys()) if rows else "none"
        raise ValueError(f"{path} must have a 'ticker' column. Found columns: {found}")
    for r in rows:
        r["ticker"] = r["ticker"].strip().upper()
    return rows


def split_universe(msci_tickers: list, covered_symbols: set):
    """Split the ticker list into (covered, uncovered, ambiguous) given the
    set of symbols FMP has transcripts for. If covered_symbols is empty (the
    master list couldn't be fetched), everything non-ambiguous is routed to
    'uncovered' -- unknown is treated as not-covered, never as covered."""
    ambiguous = [r for r in msci_tickers if is_ambiguous(r)]
    safe = [r for r in msci_tickers if not is_ambiguous(r)]
    if covered_symbols:
        covered = [r for r in safe if r["ticker"] in covered_symbols]
        uncovered = [r for r in safe if r["ticker"] not in covered_symbols]
    else:
        covered, uncovered = [], safe
    return covered, uncovered, ambiguous


def write_coverage_report(msci_tickers: list, covered_symbols: set,
                          master_list_available: bool, out_path: Path):
    fieldnames = sorted({k for r in msci_tickers for k in r.keys()} | {"fmp_covered"})
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in msci_tickers:
            row = dict(r)
            if is_ambiguous(r):
                row["fmp_covered"] = "excluded_ambiguous"
            else:
                row["fmp_covered"] = (r["ticker"] in covered_symbols) if master_list_available else "unknown"
            writer.writerow(row)


def load_coverage_subset(coverage_report_path: Path, fmp_covered_value: str) -> list:
    """Reconstruct a subset of tickers from a saved coverage report, so a
    later step can run standalone without re-running the whole upstream chain.
    e.g. load_coverage_subset(path, 'False') -> the uncovered rows,
         load_coverage_subset(path, 'True')  -> the covered rows."""
    coverage_report_path = Path(coverage_report_path)
    target = str(fmp_covered_value).strip().lower()
    with open(coverage_report_path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    return [r for r in rows if str(r.get("fmp_covered")).strip().lower() == target]


def classify_uncovered(uncovered_resolved: list, threshold: float = 0.5):
    """Split resolved rows into (likely_mismatch, true_gap, gap_countries)."""
    likely_mismatch = [
        r for r in uncovered_resolved
        if r["resolved_symbol"] and r["on_us_exchange"] and r["similarity"] >= threshold
    ]
    mismatch_tickers = {r["ticker"] for r in likely_mismatch}
    true_gap = [r for r in uncovered_resolved if r["ticker"] not in mismatch_tickers]
    gap_countries = Counter(r["expected_country"] for r in true_gap)
    return likely_mismatch, true_gap, gap_countries


def load_resolved(path: Path) -> list:
    """Load a previously-saved resolved-symbol CSV (written by
    FMPClient.run_symbol_resolution) with proper types restored."""
    path = Path(path)
    with open(path, newline="", encoding="utf-8-sig") as f:
        return [_normalize_resolved_row(r) for r in csv.DictReader(f)]


def write_rows_csv(rows: list, path: Path, fieldnames: list = None):
    fieldnames = fieldnames or (list(rows[0].keys()) if rows else [])
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def build_country_coverage_table(msci_tickers: list, covered: list, ambiguous: list,
                                 likely_mismatch: list, true_gap: list) -> list:
    """Per-country breakdown: how many of that country's MSCI World
    constituents are covered / a recoverable mismatch / a true gap /
    excluded as ambiguous, both as counts and as a rate of that country's
    total. Sorted by total constituent count, largest first."""
    total_by_country = Counter(r["country"] for r in msci_tickers)
    covered_by_country = Counter(r["country"] for r in covered)
    ambiguous_by_country = Counter(r["country"] for r in ambiguous)
    mismatch_by_country = Counter(r["expected_country"] for r in likely_mismatch)
    gap_by_country = Counter(r["expected_country"] for r in true_gap)

    rows = []
    for country, total in sorted(total_by_country.items(), key=lambda kv: -kv[1]):
        cov, amb = covered_by_country.get(country, 0), ambiguous_by_country.get(country, 0)
        mis, gap = mismatch_by_country.get(country, 0), gap_by_country.get(country, 0)
        rows.append({
            "country": country, "total": total,
            "covered": cov, "covered_rate": round(cov / total, 4) if total else 0,
            "likely_mismatch": mis, "mismatch_rate": round(mis / total, 4) if total else 0,
            "true_gap": gap, "true_gap_rate": round(gap / total, 4) if total else 0,
            "ambiguous": amb, "ambiguous_rate": round(amb / total, 4) if total else 0,
        })
    return rows


def build_extended_coverage(uncovered_resolved: list, msci_tickers: list,
                            auto_threshold: float = 0.85, review_threshold: float = 0.5):
    """From Step 3's resolved rows, split recoverable mismatches into:
    - auto_include: high-confidence (>= auto_threshold) resolved US symbol,
      safe to bulk-fetch under that symbol instead of the original ticker.
    - needs_review: moderate-confidence (review_threshold <= sim < auto_threshold),
      resolved but NOT auto-included -- needs a manual look before fetching.
    Each returned row has 'ticker' set to the RESOLVED symbol (what to fetch),
    the original iShares fields (company_name/country/gics_sector) for
    reference, and FMP's own fmp_resolved_name/fmp_resolved_exchange side by
    side -- needed to actually judge the match, not just the similarity
    score."""
    by_original_ticker = {r["ticker"]: r for r in msci_tickers}

    auto_include, needs_review = [], []
    for r in uncovered_resolved:
        if not (r["resolved_symbol"] and r["on_us_exchange"] and r["similarity"] >= review_threshold):
            continue
        orig = by_original_ticker.get(r["ticker"], {})
        row = {
            "ticker": r["resolved_symbol"],
            "original_ticker": r["ticker"],
            "company_name": r["expected_name"],
            "country": r["expected_country"],
            "gics_sector": orig.get("gics_sector", ""),
            "fmp_resolved_name": r["resolved_name"],
            "fmp_resolved_exchange": r["resolved_exchange"],
            "fmp_method": r["method"],
            "fmp_n_candidates": r["n_candidates"],
            "resolution_similarity": r["similarity"],
        }
        (auto_include if r["similarity"] >= auto_threshold else needs_review).append(row)

    return auto_include, needs_review


def dedupe_by_ticker(existing_rows: list, new_rows: list) -> list:
    """new_rows entries whose ticker isn't already present in existing_rows --
    avoids double-fetching a resolved symbol that happens to already be
    covered under its own bare ticker."""
    existing = {r["ticker"] for r in existing_rows}
    return [r for r in new_rows if r["ticker"] not in existing]


def resplit_saved_transcripts(manifest_path: Path, only_status: str = "marker_not_found") -> dict:
    """Re-run split_prepared_remarks_qa() over already-downloaded transcript
    JSON files -- no API calls, just re-processes files already on disk. Use
    this whenever QA_SECTION_MARKERS changes, instead of re-fetching.

    only_status: re-split only rows whose current qa_split_method equals this
    (default 'marker_not_found', i.e. only fix what was previously unsplit).
    Pass None to re-split every saved transcript regardless of current status.

    Rewrites both the transcript JSON (prepared_remarks/qa/qa_split_method)
    and the manifest CSV's qa_split_method column, so the two stay
    consistent. Returns counts of {checked, changed, still_unresolved}."""
    manifest_path = Path(manifest_path)
    with open(manifest_path, newline="") as f:
        rows = list(csv.DictReader(f))
        fieldnames = list(rows[0].keys()) if rows else []

    checked = changed = still_unresolved = 0
    for row in rows:
        if row.get("status") != "ok":
            continue
        if only_status is not None and row.get("qa_split_method") != only_status:
            continue

        filepath = Path(row["filepath"])
        if not filepath.exists():
            continue

        checked += 1
        data = json.loads(filepath.read_text(encoding="utf-8"))
        prepared, qa, method = split_prepared_remarks_qa(data.get("content", ""))

        if method != row.get("qa_split_method"):
            changed += 1
        if method == "marker_not_found":
            still_unresolved += 1

        data["prepared_remarks"], data["qa"], data["qa_split_method"] = prepared, qa, method
        filepath.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        row["qa_split_method"] = method
        row["chars"] = len(data.get("content", ""))

    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return {"checked": checked, "changed": changed, "still_unresolved": still_unresolved}


class FMPClient:
    def __init__(self, api_key: str, out_dir: Path, request_delay: float = 1.0,
                 max_calls_per_run=None):
        self.api_key = api_key
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.request_delay = request_delay
        self.max_calls_per_run = max_calls_per_run
        self.api_call_count = 0

    def reset_budget(self, max_calls_per_run=None, request_delay=None):
        """Call at the start of each day's run to reset the counter (and
        optionally re-tune pacing/budget)."""
        self.api_call_count = 0
        if max_calls_per_run is not None:
            self.max_calls_per_run = max_calls_per_run
        if request_delay is not None:
            self.request_delay = request_delay

    def _get(self, endpoint: str, params: dict, max_retries: int = 4):
        if self.max_calls_per_run is not None and self.api_call_count >= self.max_calls_per_run:
            raise RuntimeError(
                f"Hit max_calls_per_run ({self.max_calls_per_run}) for this run. "
                "Call client.reset_budget(...) then re-run the bulk cell -- "
                "already-saved transcripts are skipped."
            )
        params = {**params, "apikey": self.api_key}
        url = f"{BASE_URL}/{endpoint}"

        for attempt in range(max_retries):
            try:
                resp = requests.get(url, params=params, timeout=30)
            except requests.exceptions.RequestException as e:
                wait = 5 * (attempt + 1)
                print(f"  [network error] {type(e).__name__}, retrying in {wait}s "
                      f"(attempt {attempt + 1}/{max_retries})...")
                time.sleep(wait)
                continue

            self.api_call_count += 1
            if resp.status_code == 429:
                print("  [rate limit] sleeping 30s before retry...")
                time.sleep(30)
                continue

            try:
                payload = resp.json()
            except ValueError:
                payload = resp.text
            return resp.status_code, payload

        print(f"  [network error] giving up on {endpoint} after {max_retries} attempts.")
        return None, {"_error": "request_failed_after_retries"}


    def get_transcript_dates(self, symbol: str):
        """List of {quarter, year, date} for which a transcript exists."""
        status, payload = self._get("earning-call-transcript-dates", {"symbol": symbol})
        if status != 200 or not isinstance(payload, list):
            print(f"  [{symbol}] could not list transcript dates (HTTP {status}): {payload}")
            return []
        return payload

    def get_transcript(self, symbol: str, quarter: int, year: int):
        status, payload = self._get(
            "earning-call-transcript",
            {"symbol": symbol, "quarter": quarter, "year": year},
        )
        if status != 200:
            print(f"  [{symbol} Q{quarter} {year}] request failed (HTTP {status}): {payload}")
            return None
        if not payload:
            print(f"  [{symbol} Q{quarter} {year}] empty response")
            return None
        return payload[0] if isinstance(payload, list) else payload

    def save_transcript(self, symbol: str, quarter: int, year: int, data: dict) -> Path:
        symbol_dir = self.out_dir / symbol
        symbol_dir.mkdir(parents=True, exist_ok=True)
        out_path = symbol_dir / f"{symbol}_{year}_Q{quarter}.json"
        out_path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
        return out_path

    def fetch_covered_symbols(self) -> set:
        """Symbols FMP has at least one transcript for. Tries both endpoint
        names since FMP's docs list this under two slugs."""
        for endpoint in ["available-transcript-symbols", "earnings-transcript-list"]:
            status, payload = self._get(endpoint, {})
            if status == 200 and isinstance(payload, list) and payload:
                symbols = set()
                for item in payload:
                    if isinstance(item, str):
                        symbols.add(item.upper())
                    elif isinstance(item, dict):
                        sym = item.get("symbol") or item.get("ticker")
                        if sym:
                            symbols.add(sym.upper())
                print(f"  covered-symbols endpoint used: {endpoint} ({len(symbols)} symbols)")
                return symbols
            print(f"  {endpoint} -> HTTP {status}, trying next candidate...")
        print("  could not retrieve a covered-symbol master list.")
        return set()


    def verify_symbol(self, row: dict) -> dict:
        """Look up what a ticker/company actually resolves to on FMP, scoring
        each candidate by name similarity plus a US-exchange bonus."""
        ticker = row["ticker"]
        expected_name = row["company_name"]
        result = {
            "ticker": ticker, "expected_name": expected_name,
            "expected_country": row.get("country"),
            "resolved_symbol": None, "resolved_name": None, "resolved_exchange": None,
            "similarity": 0.0, "on_us_exchange": False, "method": None, "n_candidates": 0,
        }

        status, payload = self._get("search-exchange-variants", {"symbol": ticker})
        candidates = payload if status == 200 and isinstance(payload, list) else []
        method = "search-exchange-variants"
        time.sleep(self.request_delay)

        if not candidates:
            status, payload = self._get("search-name", {"query": expected_name})
            candidates = payload if status == 200 and isinstance(payload, list) else []
            method = "search-name"
            time.sleep(self.request_delay)

        result["method"] = method
        result["n_candidates"] = len(candidates)

        best, best_score = None, -1.0
        for c in candidates:
            if not isinstance(c, dict):
                continue
            name = c.get("name") or c.get("companyName") or ""
            sym = c.get("symbol")
            exch = c.get("exchangeShortName") or c.get("exchange") or ""
            sim = name_similarity(name, expected_name)
            score = sim + (0.15 if is_us_exchange(exch) else 0.0)
            if score > best_score:
                best_score = score
                best = (sym, name, exch, sim)

        if best:
            result["resolved_symbol"], result["resolved_name"], \
                result["resolved_exchange"], result["similarity"] = best
            result["on_us_exchange"] = is_us_exchange(result["resolved_exchange"])

        return result

    def run_symbol_resolution(self, rows: list, out_path: Path) -> list:
        """Resumable: writes each result immediately and skips tickers already
        present in out_path from a previous (possibly crashed) run."""
        out_path = Path(out_path)
        already = {}
        if out_path.exists():
            with open(out_path, newline="") as f:
                already = {r["ticker"]: _normalize_resolved_row(r) for r in csv.DictReader(f)}
            print(f"Resuming {out_path.name}: {len(already)} tickers already resolved.")

        write_header = not out_path.exists()
        out_file = open(out_path, "a", newline="")
        writer = csv.DictWriter(out_file, fieldnames=RESOLVED_FIELDNAMES)
        if write_header:
            writer.writeheader()

        results = list(already.values())
        try:
            for row in rows:
                if row["ticker"] in already:
                    continue
                result = self.verify_symbol(row)
                writer.writerow(result)
                out_file.flush()
                results.append(result)
        finally:
            out_file.close()
        return results


    def run_bulk_extraction(self, covered: list, manifest_path: Path,
                            year_start: int = YEAR_START, year_end: int = YEAR_END) -> dict:
        """Fetch transcripts for every (ticker, year, quarter) in range.
        Resumable (skips already-saved JSON), budget-aware (stops cleanly when
        max_calls_per_run is hit), and logs every attempt to manifest_path."""
        manifest_path = Path(manifest_path)
        manifest_is_new = not manifest_path.exists()
        manifest_file = open(manifest_path, "a", newline="")
        manifest_writer = csv.writer(manifest_file)
        if manifest_is_new:
            manifest_writer.writerow(
                ["ticker", "year", "quarter", "date", "status", "chars", "qa_split_method", "filepath"]
            )

        budget_hit = False
        try:
            for row in covered:
                symbol = row["ticker"]
                if budget_hit:
                    break
                try:
                    dates = self.get_transcript_dates(symbol)
                except RuntimeError as e:
                    print(e)
                    budget_hit = True
                    break
                time.sleep(self.request_delay)

                for d in dates:
                    quarter = d.get("quarter", d.get("period"))
                    year = d.get("year", d.get("fiscalYear", d.get("calendarYear")))
                    if isinstance(quarter, str) and quarter.upper().startswith("Q"):
                        quarter = int(quarter[1:])
                    if quarter is None or year is None or not (year_start <= int(year) <= year_end):
                        continue

                    out_path = self.out_dir / symbol / f"{symbol}_{year}_Q{quarter}.json"
                    if out_path.exists():
                        continue

                    try:
                        transcript = self.get_transcript(symbol, quarter, year)
                    except RuntimeError as e:
                        print(e)
                        budget_hit = True
                        break
                    time.sleep(self.request_delay)

                    if transcript is None:
                        manifest_writer.writerow([symbol, year, quarter, d.get("date"), "no_content", 0, "", ""])
                        continue

                    content = transcript.get("content", "")
                    prepared, qa, method = split_prepared_remarks_qa(content)
                    transcript["prepared_remarks"] = prepared
                    transcript["qa"] = qa
                    transcript["qa_split_method"] = method

                    saved_path = self.save_transcript(symbol, quarter, year, transcript)
                    manifest_writer.writerow(
                        [symbol, year, quarter, d.get("date"), "ok", len(content), method, str(saved_path)]
                    )
                    manifest_file.flush()
        finally:
            manifest_file.close()

        return {"api_calls": self.api_call_count, "budget_hit": budget_hit,
                "budget": self.max_calls_per_run}
