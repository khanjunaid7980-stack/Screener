"""Market data via yfinance (free, no API key).

Used for the bits SEC doesn't serve: live price, market cap, beta, shares
outstanding today, and a few industry/sector tags.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import pandas as pd

try:
    import yfinance as yf
except Exception:  # noqa: BLE001
    yf = None  # type: ignore[assignment]


@dataclass
class MarketSnapshot:
    ticker: str
    price: float | None = None
    market_cap: float | None = None
    shares_outstanding: float | None = None
    beta: float | None = None
    sector: str | None = None
    industry: str | None = None
    currency: str | None = None
    long_name: str | None = None

    @property
    def is_valid(self) -> bool:
        return self.price is not None or self.market_cap is not None


def _get(info: dict[str, Any], *keys: str) -> Any:
    for k in keys:
        v = info.get(k)
        if v not in (None, "", float("nan")):
            return v
    return None


@lru_cache(maxsize=128)
def fetch_market_snapshot(ticker: str) -> MarketSnapshot:
    snap = MarketSnapshot(ticker=ticker.upper())
    if yf is None:
        return snap
    try:
        tk = yf.Ticker(ticker)
        info: dict[str, Any] = {}
        try:
            info = tk.info or {}
        except Exception:  # noqa: BLE001
            info = {}
        snap.price = _get(info, "currentPrice", "regularMarketPrice", "previousClose")
        snap.market_cap = _get(info, "marketCap")
        snap.shares_outstanding = _get(info, "sharesOutstanding", "impliedSharesOutstanding")
        snap.beta = _get(info, "beta", "beta3Year")
        snap.sector = _get(info, "sector")
        snap.industry = _get(info, "industry")
        snap.currency = _get(info, "currency", "financialCurrency")
        snap.long_name = _get(info, "longName", "shortName")
        if snap.price is None:
            hist = tk.history(period="5d")
            if not hist.empty:
                snap.price = float(hist["Close"].iloc[-1])
        if snap.market_cap is None and snap.price and snap.shares_outstanding:
            snap.market_cap = float(snap.price) * float(snap.shares_outstanding)
    except Exception:  # noqa: BLE001
        pass
    return snap


def fetch_price_history(ticker: str, period: str = "5y") -> pd.DataFrame:
    if yf is None:
        return pd.DataFrame()
    try:
        tk = yf.Ticker(ticker)
        hist = tk.history(period=period, auto_adjust=True)
        if hist.empty:
            return pd.DataFrame()
        hist = hist.reset_index()
        return hist[["Date", "Close", "Volume"]]
    except Exception:  # noqa: BLE001
        return pd.DataFrame()


def fetch_risk_free_rate() -> float:
    """10-year US Treasury yield as a proxy for the risk-free rate.

    Falls back to 4.25% if offline. Returns a decimal (e.g. 0.0425).
    """
    if yf is None:
        return 0.0425
    try:
        tnx = yf.Ticker("^TNX").history(period="5d")
        if not tnx.empty:
            # ^TNX is quoted in basis points / 10 (e.g. 42.5 -> 4.25%).
            return float(tnx["Close"].iloc[-1]) / 100.0
    except Exception:  # noqa: BLE001
        pass
    return 0.0425


__all__ = ["MarketSnapshot", "fetch_market_snapshot", "fetch_price_history", "fetch_risk_free_rate"]
