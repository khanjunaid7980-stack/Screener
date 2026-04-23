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

        # 1) Try .info (rich, but often flaky / rate-limited).
        info: dict[str, Any] = {}
        try:
            info = tk.info or {}
        except Exception:  # noqa: BLE001
            info = {}

        # 2) .fast_info is a separate, lighter endpoint — use as a complement.
        fast: dict[str, Any] = {}
        try:
            fi = tk.fast_info
            for k in (
                "last_price", "previous_close", "market_cap", "shares",
                "currency", "quote_type",
            ):
                try:
                    v = getattr(fi, k, None)
                    if v is not None:
                        fast[k] = v
                except Exception:  # noqa: BLE001
                    continue
        except Exception:  # noqa: BLE001
            fast = {}

        snap.price = (
            _get(info, "currentPrice", "regularMarketPrice", "previousClose")
            or fast.get("last_price")
            or fast.get("previous_close")
        )
        snap.market_cap = _get(info, "marketCap") or fast.get("market_cap")
        snap.shares_outstanding = (
            _get(info, "sharesOutstanding", "impliedSharesOutstanding")
            or fast.get("shares")
        )
        snap.beta = _get(info, "beta", "beta3Year")
        snap.sector = _get(info, "sector")
        snap.industry = _get(info, "industry")
        snap.currency = _get(info, "currency", "financialCurrency") or fast.get("currency")
        snap.long_name = _get(info, "longName", "shortName")

        # 3) Last-resort price: 5-day history.
        if snap.price is None:
            try:
                hist = tk.history(period="5d", auto_adjust=False)
                if not hist.empty:
                    snap.price = float(hist["Close"].iloc[-1])
            except Exception:  # noqa: BLE001
                pass

        # 4) Derive market cap if we have the pieces.
        if snap.market_cap is None and snap.price and snap.shares_outstanding:
            snap.market_cap = float(snap.price) * float(snap.shares_outstanding)

        # 5) Derive beta from regression vs. S&P 500 if yfinance didn't give us one.
        if snap.beta is None:
            snap.beta = _estimate_beta(ticker)
    except Exception:  # noqa: BLE001
        pass
    return snap


def _estimate_beta(ticker: str) -> float | None:
    """5-year monthly beta vs. ^GSPC. Falls back to None on any failure."""
    if yf is None:
        return None
    try:
        import numpy as np
        data = yf.download(
            [ticker, "^GSPC"], period="5y", interval="1mo",
            auto_adjust=True, progress=False,
        )
        if data is None or data.empty:
            return None
        closes = data["Close"] if "Close" in data.columns.get_level_values(0) else data
        closes = closes.dropna()
        if len(closes) < 24:
            return None
        rets = closes.pct_change().dropna()
        if ticker not in rets.columns or "^GSPC" not in rets.columns:
            return None
        cov = np.cov(rets[ticker].values, rets["^GSPC"].values, ddof=1)
        if cov[1, 1] == 0:
            return None
        return float(cov[0, 1] / cov[1, 1])
    except Exception:  # noqa: BLE001
        return None


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
