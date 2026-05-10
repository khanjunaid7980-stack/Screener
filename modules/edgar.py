"""SEC EDGAR data retrieval.

Uses the public SEC data APIs (no key required). The SEC requires a
descriptive User-Agent on every request.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

import pandas as pd
import requests

SEC_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Accept-Encoding": "gzip, deflate",
    "Host": "data.sec.gov",
}

TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"


# ---------- Ticker lookup ----------

@lru_cache(maxsize=1)
def _ticker_map() -> dict[str, dict[str, Any]]:
    """Ticker -> {cik, title}. Cached in-process."""
    try:
        headers = {**SEC_HEADERS, "Host": "www.sec.gov"}
        r = requests.get(TICKERS_URL, headers=headers, timeout=30)
        r.raise_for_status()
        data = r.json()
        out: dict[str, dict[str, Any]] = {}
        for row in data.values():
            ticker = str(row["ticker"]).upper()
            out[ticker] = {
                "cik": str(row["cik_str"]).zfill(10),
                "title": row["title"],
            }
        return out
    except Exception:
        return _fallback_ticker_map()


def _fallback_ticker_map() -> dict[str, dict[str, Any]]:
    """Fallback ticker map for common US stocks when SEC API is unavailable."""
    return {
        "AAPL": {"cik": "0000320193", "title": "Apple Inc."},
        "MSFT": {"cik": "0000789019", "title": "Microsoft Corporation"},
        "GOOGL": {"cik": "0001652044", "title": "Alphabet Inc."},
        "GOOG": {"cik": "0001652044", "title": "Alphabet Inc."},
        "AMZN": {"cik": "0001018724", "title": "Amazon.com Inc."},
        "NVDA": {"cik": "0001045810", "title": "NVIDIA Corporation"},
        "META": {"cik": "0001326801", "title": "Meta Platforms Inc."},
        "TSLA": {"cik": "0001318605", "title": "Tesla Inc."},
        "JPM": {"cik": "0000019617", "title": "JPMorgan Chase & Co."},
        "V": {"cik": "0001403161", "title": "Visa Inc."},
        "MA": {"cik": "0001141391", "title": "Mastercard Inc."},
        "WMT": {"cik": "0000104169", "title": "Walmart Inc."},
        "KO": {"cik": "0000021344", "title": "The Coca-Cola Company"},
        "MCD": {"cik": "0000063908", "title": "McDonald's Corporation"},
        "INTC": {"cik": "0000050104", "title": "Intel Corporation"},
        "AMD": {"cik": "0000002488", "title": "Advanced Micro Devices Inc."},
        "CSCO": {"cik": "0000858877", "title": "Cisco Systems Inc."},
        "ORCL": {"cik": "0001652735", "title": "Oracle Corporation"},
        "SAP": {"cik": "0000709519", "title": "SAP SE"},
        "IBM": {"cik": "0000051143", "title": "International Business Machines"},
        "BA": {"cik": "0000012927", "title": "The Boeing Company"},
        "CAT": {"cik": "0000018230", "title": "Caterpillar Inc."},
        "MMM": {"cik": "0000066740", "title": "3M Company"},
        "HON": {"cik": "0000773840", "title": "Honeywell International Inc."},
        "LMT": {"cik": "0000060086", "title": "Lockheed Martin Corporation"},
        "GD": {"cik": "0000040533", "title": "General Dynamics Corporation"},
        "TXN": {"cik": "0000097476", "title": "Texas Instruments Incorporated"},
        "QCOM": {"cik": "0000804842", "title": "Qualcomm Inc."},
        "SONY": {"cik": "0001037409", "title": "Sony Corporation"},
        "NFLX": {"cik": "0001564590", "title": "Netflix Inc."},
        "DIS": {"cik": "0000018442", "title": "The Walt Disney Company"},
    }


def resolve_ticker(ticker: str) -> dict[str, Any] | None:
    return _ticker_map().get(ticker.upper())


# ---------- Company facts ----------

# Keys mapped from US-GAAP concepts to our canonical line-items.
# We try several tags because issuers tag differently across years.
CONCEPT_MAP: dict[str, list[str]] = {
    "Revenue": [
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "RevenueFromContractWithCustomerIncludingAssessedTax",
        "Revenues",
        "SalesRevenueNet",
        "SalesRevenueGoodsNet",
    ],
    "CostOfRevenue": [
        "CostOfRevenue",
        "CostOfGoodsAndServicesSold",
        "CostOfGoodsSold",
    ],
    "GrossProfit": ["GrossProfit"],
    "OperatingIncome": [
        "OperatingIncomeLoss",
    ],
    "NetIncome": [
        "NetIncomeLoss",
        "ProfitLoss",
    ],
    "EPS": [
        "EarningsPerShareDiluted",
        "EarningsPerShareBasic",
    ],
    "SharesOutstanding": [
        "CommonStockSharesOutstanding",
        "EntityCommonStockSharesOutstanding",
        "WeightedAverageNumberOfDilutedSharesOutstanding",
        "WeightedAverageNumberOfSharesOutstandingBasic",
    ],
    "TotalAssets": ["Assets"],
    "TotalLiabilities": ["Liabilities"],
    "TotalEquity": [
        "StockholdersEquity",
        "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
    ],
    "CashAndEquivalents": [
        "CashAndCashEquivalentsAtCarryingValue",
        "Cash",
    ],
    "ShortTermDebt": [
        "ShortTermBorrowings",
        "LongTermDebtCurrent",
        "DebtCurrent",
    ],
    "LongTermDebt": [
        "LongTermDebtNoncurrent",
        "LongTermDebt",
    ],
    "CapEx": [
        "PaymentsToAcquirePropertyPlantAndEquipment",
        "PaymentsForCapitalImprovements",
    ],
    "DepreciationAmortization": [
        "DepreciationDepletionAndAmortization",
        "DepreciationAndAmortization",
        "Depreciation",
    ],
    "CashFromOps": [
        "NetCashProvidedByUsedInOperatingActivities",
    ],
    "InterestExpense": ["InterestExpense"],
    "IncomeTaxExpense": ["IncomeTaxExpenseBenefit"],
    "PreTaxIncome": ["IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest"],
    # --- Income Statement additions ---
    "ResearchAndDevelopment": ["ResearchAndDevelopmentExpense"],
    "SellingGeneralAdmin": [
        "SellingGeneralAndAdministrativeExpense",
        "GeneralAndAdministrativeExpense",
    ],
    "OperatingExpenses": ["OperatingExpenses"],
    "InterestIncome": ["InterestIncomeOperating", "InvestmentIncomeInterest"],
    # --- Balance Sheet additions ---
    "AccountsReceivable": [
        "AccountsReceivableNetCurrent",
        "ReceivablesNetCurrent",
    ],
    "Inventory": ["InventoryNet"],
    "CurrentAssets": ["AssetsCurrent"],
    "PropertyPlantEquipment": ["PropertyPlantAndEquipmentNet"],
    "Goodwill": ["Goodwill"],
    "IntangibleAssets": ["IntangibleAssetsNetExcludingGoodwill"],
    "AccountsPayable": ["AccountsPayableCurrent"],
    "CurrentLiabilities": ["LiabilitiesCurrent"],
    "RetainedEarnings": ["RetainedEarningsAccumulatedDeficit"],
    # --- Cash Flow additions ---
    "CashFromInvesting": ["NetCashProvidedByUsedInInvestingActivities"],
    "CashFromFinancing": ["NetCashProvidedByUsedInFinancingActivities"],
    "Dividends": [
        "PaymentsOfDividends",
        "PaymentsOfDividendsCommonStock",
    ],
    "StockBuybacks": [
        "PaymentsForRepurchaseOfCommonStock",
        "PaymentsForRepurchaseOfEquity",
    ],
    "DebtIssued": [
        "ProceedsFromIssuanceOfLongTermDebt",
        "ProceedsFromRepaymentsOfLongTermDebt",
    ],
    "DebtRepaid": ["RepaymentsOfLongTermDebt"],
    "Acquisitions": [
        "PaymentsToAcquireBusinessesNetOfCashAcquired",
        "PaymentsToAcquireBusinessesGross",
    ],
    "StockBasedComp": ["ShareBasedCompensation"],
    "ChangesInWorkingCapital": ["IncreaseDecreaseInOperatingCapital"],
}


@dataclass
class CompanyFacts:
    cik: str
    name: str
    raw: dict[str, Any] = field(default_factory=dict)

    def annual_series(self, concept_tags: list[str]) -> pd.Series:
        """Return annual (FY) values merged across all candidate tags.

        Root-cause of the "wrong segment revenue" bug:
        The EDGAR company-facts API includes BOTH consolidated (non-dimensional)
        AND per-segment (dimensional) values for the same tag/fy/filed combo.
        The SEC distinguishes them via the ``frame`` field:
          - Consolidated facts → have a ``frame`` like "CY2024Q3I" or "CY2024"
          - Dimensional/segment facts → ``frame`` is absent or empty

        Selection priority for each fiscal year:
          1. Frame-tagged  > non-frame-tagged  (consolidated beats segment)
          2. Most recently filed wins (handles restatements/amendments)
          3. Largest absolute value wins (tie-breaker for same date+frame-status)
        """
        us_gaap = self.raw.get("facts", {}).get("us-gaap", {})
        allowed_forms = ("10-K", "10-K/A", "20-F", "20-F/A", "40-F", "40-F/A")

        # (val, filed_date, has_frame) keyed by fiscal year int
        merged: dict[int, tuple[float, str, bool]] = {}

        for tag in concept_tags:
            node = us_gaap.get(tag)
            if not node:
                continue
            units = node.get("units", {})
            for preferred in ("USD", "shares", "USD/shares"):
                if preferred in units:
                    unit_key = preferred
                    break
            else:
                if not units:
                    continue
                unit_key = next(iter(units))

            for row in units[unit_key]:
                if row.get("fp") != "FY" or row.get("form") not in allowed_forms:
                    continue
                fy = row.get("fy")
                val = row.get("val")
                if fy is None or val is None:
                    continue
                fy = int(fy)
                val = float(val)
                filed = str(row.get("filed", ""))
                has_frame = bool(row.get("frame", ""))

                prev = merged.get(fy)
                if prev is None:
                    merged[fy] = (val, filed, has_frame)
                    continue

                prev_val, prev_filed, prev_frame = prev
                # Rule 1: consolidated (framed) beats segment (unframed)
                if has_frame and not prev_frame:
                    merged[fy] = (val, filed, has_frame)
                elif not has_frame and prev_frame:
                    pass  # keep existing consolidated value
                # Rule 2: newer filing wins (same frame status)
                elif filed > prev_filed:
                    merged[fy] = (val, filed, has_frame)
                # Rule 3: same date + same frame → keep largest absolute value
                elif filed == prev_filed and abs(val) > abs(prev_val):
                    merged[fy] = (val, filed, has_frame)

        if not merged:
            return pd.Series(dtype="float64")
        return pd.Series(
            {k: v[0] for k, v in sorted(merged.items())},
            dtype="float64",
        )

    def build_financials(self, years: int) -> pd.DataFrame:
        """Wide DataFrame: rows = canonical line items, columns = fiscal years."""
        rows: dict[str, pd.Series] = {}
        for canonical, tags in CONCEPT_MAP.items():
            rows[canonical] = self.annual_series(tags)
        df = pd.DataFrame(rows).T  # rows = items, cols = years
        if df.shape[1] == 0:
            return df
        df = df.reindex(sorted(df.columns), axis=1)
        # Trim to last N years that are actually populated (any row non-null).
        populated = df.columns[df.notna().any(axis=0)]
        df = df[populated]
        if years and df.shape[1] > years:
            df = df.iloc[:, -years:]
        return df


def fetch_company_facts(cik: str, retries: int = 3, sleep: float = 0.5) -> CompanyFacts:
    url = FACTS_URL.format(cik=cik)
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            r = requests.get(url, headers=SEC_HEADERS, timeout=30)
            if r.status_code == 429:
                time.sleep(1.0 + attempt)
                continue
            r.raise_for_status()
            data = r.json()
            return CompanyFacts(
                cik=cik,
                name=data.get("entityName", ""),
                raw=data,
            )
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            time.sleep(sleep * (attempt + 1))
    raise RuntimeError(f"Failed to fetch company facts for CIK {cik}: {last_err}")


def fetch_recent_filings(cik: str, forms: tuple[str, ...] = ("10-K", "10-Q", "8-K")) -> pd.DataFrame:
    """List recent filings; returns form, filing date, accession, primary doc URL."""
    r = requests.get(SUBMISSIONS_URL.format(cik=cik), headers=SEC_HEADERS, timeout=30)
    r.raise_for_status()
    recent = r.json().get("filings", {}).get("recent", {})
    if not recent:
        return pd.DataFrame()
    df = pd.DataFrame(recent)
    df = df[df["form"].isin(forms)].copy()
    if df.empty:
        return df
    df["url"] = df.apply(
        lambda row: (
            "https://www.sec.gov/Archives/edgar/data/"
            f"{int(cik)}/{str(row['accessionNumber']).replace('-', '')}/"
            f"{row['primaryDocument']}"
        ),
        axis=1,
    )
    return df[["form", "filingDate", "accessionNumber", "primaryDocument", "url"]].reset_index(drop=True)


def fetch_transcripts_hint(ticker: str) -> str:
    """Conference-call transcripts are not hosted by SEC directly. Return a
    free-source search URL so users can retrieve them manually without any
    paid API dependency."""
    q = f"{ticker} earnings call transcript"
    return f"https://www.google.com/search?q={q.replace(' ', '+')}"


# Groupings for the traditional three financial statements.
INCOME_STATEMENT_ROWS = [
    "Revenue", "CostOfRevenue", "GrossProfit",
    "ResearchAndDevelopment", "SellingGeneralAdmin", "OperatingExpenses",
    "OperatingIncome", "InterestExpense", "InterestIncome",
    "PreTaxIncome", "IncomeTaxExpense", "NetIncome", "EPS",
]

BALANCE_SHEET_ROWS = [
    "CashAndEquivalents", "AccountsReceivable", "Inventory", "CurrentAssets",
    "PropertyPlantEquipment", "Goodwill", "IntangibleAssets", "TotalAssets",
    "AccountsPayable", "ShortTermDebt", "CurrentLiabilities",
    "LongTermDebt", "TotalLiabilities", "RetainedEarnings", "TotalEquity",
]

CASH_FLOW_ROWS = [
    "NetIncome", "DepreciationAmortization", "StockBasedComp",
    "ChangesInWorkingCapital", "CashFromOps",
    "CapEx", "Acquisitions", "CashFromInvesting",
    "DebtIssued", "DebtRepaid", "Dividends", "StockBuybacks",
    "CashFromFinancing",
]


def extract_statement(facts: "CompanyFacts", rows: list[str], years: int) -> pd.DataFrame:
    """Build a traditional statement (rows = canonical items) from CompanyFacts."""
    data: dict[str, pd.Series] = {}
    for row in rows:
        tags = CONCEPT_MAP.get(row, [row])
        data[row] = facts.annual_series(tags)
    df = pd.DataFrame(data).T
    if df.shape[1] == 0:
        return df
    df = df.reindex(sorted(df.columns), axis=1)
    populated_cols = df.columns[df.notna().any(axis=0)]
    df = df[populated_cols]
    if years and df.shape[1] > years:
        df = df.iloc[:, -years:]
    return df


__all__ = [
    "resolve_ticker",
    "fetch_company_facts",
    "fetch_recent_filings",
    "fetch_transcripts_hint",
    "CompanyFacts",
    "CONCEPT_MAP",
    "INCOME_STATEMENT_ROWS",
    "BALANCE_SHEET_ROWS",
    "CASH_FLOW_ROWS",
    "extract_statement",
]
