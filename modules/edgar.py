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
    "User-Agent": "FinancialScreener research-tool contact@example.com",
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
}


@dataclass
class CompanyFacts:
    cik: str
    name: str
    raw: dict[str, Any] = field(default_factory=dict)

    def annual_series(self, concept_tags: list[str]) -> pd.Series:
        """Return annual (FY) values indexed by fiscal year, **merged** across
        every candidate tag. When multiple tags report a value for the same
        fiscal year, keep the most recently filed one — which naturally prefers
        the post-ASC-606 tag (e.g. RevenueFromContractWithCustomer...) over
        legacy tags (e.g. Revenues) that issuers stopped using."""
        us_gaap = self.raw.get("facts", {}).get("us-gaap", {})
        allowed_forms = ("10-K", "10-K/A", "20-F", "20-F/A", "40-F", "40-F/A")
        merged: dict[int, tuple[float, str]] = {}
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
                filed = str(row.get("filed", ""))
                prev = merged.get(int(fy))
                if prev is None or filed > prev[1]:
                    merged[int(fy)] = (float(val), filed)
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


__all__ = [
    "resolve_ticker",
    "fetch_company_facts",
    "fetch_recent_filings",
    "fetch_transcripts_hint",
    "CompanyFacts",
    "CONCEPT_MAP",
]
