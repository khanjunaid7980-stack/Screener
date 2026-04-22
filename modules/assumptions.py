"""Smart default assumptions.

Combines three inputs to pre-fill every DCF assumption so the user rarely
needs to touch a number:

1. Sector/industry benchmarks (Damodaran-style, from published academic tables
   and widely cited averages; hard-coded here because the source data is free
   and public but not available as a stable free API).
2. Company historicals (fit a linear-regression trend on revenue/margins from
   EDGAR data, with gentle mean-reversion toward sector averages).
3. Live market data (10Y Treasury for Rf, yfinance beta).

Everything returned is editable in the UI. Nothing is opaque.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


# Sector benchmarks (approx. long-run averages). Values are decimals.
# Sources: Damodaran online data, NYU Stern (public/free).
SECTOR_BENCHMARKS: dict[str, dict[str, float]] = {
    "Technology": {"rev_growth": 0.12, "op_margin": 0.22, "tax": 0.18, "beta": 1.15, "capex_pct_rev": 0.06, "da_pct_rev": 0.05, "nwc_pct_rev": 0.02},
    "Communication Services": {"rev_growth": 0.07, "op_margin": 0.18, "tax": 0.21, "beta": 1.05, "capex_pct_rev": 0.10, "da_pct_rev": 0.09, "nwc_pct_rev": 0.02},
    "Consumer Cyclical": {"rev_growth": 0.06, "op_margin": 0.10, "tax": 0.22, "beta": 1.10, "capex_pct_rev": 0.05, "da_pct_rev": 0.04, "nwc_pct_rev": 0.05},
    "Consumer Defensive": {"rev_growth": 0.04, "op_margin": 0.09, "tax": 0.22, "beta": 0.70, "capex_pct_rev": 0.04, "da_pct_rev": 0.03, "nwc_pct_rev": 0.04},
    "Healthcare": {"rev_growth": 0.08, "op_margin": 0.14, "tax": 0.18, "beta": 0.85, "capex_pct_rev": 0.05, "da_pct_rev": 0.04, "nwc_pct_rev": 0.06},
    "Financial Services": {"rev_growth": 0.05, "op_margin": 0.28, "tax": 0.22, "beta": 1.00, "capex_pct_rev": 0.02, "da_pct_rev": 0.02, "nwc_pct_rev": 0.00},
    "Industrials": {"rev_growth": 0.05, "op_margin": 0.11, "tax": 0.22, "beta": 1.05, "capex_pct_rev": 0.05, "da_pct_rev": 0.04, "nwc_pct_rev": 0.08},
    "Energy": {"rev_growth": 0.03, "op_margin": 0.12, "tax": 0.24, "beta": 1.20, "capex_pct_rev": 0.12, "da_pct_rev": 0.10, "nwc_pct_rev": 0.02},
    "Utilities": {"rev_growth": 0.03, "op_margin": 0.18, "tax": 0.20, "beta": 0.55, "capex_pct_rev": 0.18, "da_pct_rev": 0.10, "nwc_pct_rev": 0.01},
    "Real Estate": {"rev_growth": 0.04, "op_margin": 0.25, "tax": 0.18, "beta": 0.90, "capex_pct_rev": 0.10, "da_pct_rev": 0.15, "nwc_pct_rev": 0.00},
    "Basic Materials": {"rev_growth": 0.04, "op_margin": 0.12, "tax": 0.22, "beta": 1.10, "capex_pct_rev": 0.08, "da_pct_rev": 0.07, "nwc_pct_rev": 0.06},
}

DEFAULT_SECTOR = {"rev_growth": 0.06, "op_margin": 0.12, "tax": 0.21, "beta": 1.00, "capex_pct_rev": 0.06, "da_pct_rev": 0.05, "nwc_pct_rev": 0.03}


@dataclass
class Assumptions:
    # Growth / margin
    revenue_growth_rate: float = 0.06        # stage-1 explicit-period CAGR
    terminal_growth_rate: float = 0.025      # <= long-run GDP / risk-free
    operating_margin: float = 0.12           # EBIT / Revenue
    tax_rate: float = 0.21
    capex_pct_revenue: float = 0.06
    da_pct_revenue: float = 0.05
    nwc_pct_revenue: float = 0.03

    # WACC inputs
    risk_free_rate: float = 0.0425
    equity_risk_premium: float = 0.055       # long-run US ERP
    beta: float = 1.00
    cost_of_debt_pretax: float = 0.055
    debt_weight: float = 0.20                # D / (D+E)

    # Projection
    projection_years: int = 5

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _fit_trend(series: pd.Series) -> float | None:
    """Return implied annual CAGR from a linear fit on log values."""
    s = series.dropna().astype(float)
    s = s[s > 0]
    if len(s) < 3:
        return None
    X = np.arange(len(s)).reshape(-1, 1)
    y = np.log(s.values)
    try:
        model = LinearRegression().fit(X, y)
        return float(np.exp(model.coef_[0]) - 1.0)
    except Exception:  # noqa: BLE001
        return None


def _blend(company: float | None, sector: float, weight_company: float = 0.6) -> float:
    if company is None or not np.isfinite(company):
        return sector
    return weight_company * company + (1 - weight_company) * sector


def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def derive_assumptions(
    financials: pd.DataFrame,
    sector: str | None,
    risk_free_rate: float | None = None,
    beta: float | None = None,
) -> Assumptions:
    """Build an Assumptions block by blending historicals with sector norms."""
    bench = SECTOR_BENCHMARKS.get(sector or "", DEFAULT_SECTOR)

    # Historical CAGR for revenue.
    rev_cagr = None
    op_margin_avg = None
    capex_pct = None
    da_pct = None
    tax_rate_hist = None

    if financials is not None and not financials.empty:
        if "Revenue" in financials.index:
            rev_cagr = _fit_trend(financials.loc["Revenue"])
        if "OperatingIncome" in financials.index and "Revenue" in financials.index:
            rev = financials.loc["Revenue"].astype(float)
            op = financials.loc["OperatingIncome"].astype(float)
            ratios = (op / rev).replace([np.inf, -np.inf], np.nan).dropna()
            if not ratios.empty:
                op_margin_avg = float(ratios.tail(3).mean())
        if "CapEx" in financials.index and "Revenue" in financials.index:
            cx = financials.loc["CapEx"].astype(float).abs()
            rev = financials.loc["Revenue"].astype(float)
            ratios = (cx / rev).replace([np.inf, -np.inf], np.nan).dropna()
            if not ratios.empty:
                capex_pct = float(ratios.tail(3).mean())
        if "DepreciationAmortization" in financials.index and "Revenue" in financials.index:
            da = financials.loc["DepreciationAmortization"].astype(float)
            rev = financials.loc["Revenue"].astype(float)
            ratios = (da / rev).replace([np.inf, -np.inf], np.nan).dropna()
            if not ratios.empty:
                da_pct = float(ratios.tail(3).mean())
        if "IncomeTaxExpense" in financials.index and "PreTaxIncome" in financials.index:
            tx = financials.loc["IncomeTaxExpense"].astype(float)
            pt = financials.loc["PreTaxIncome"].astype(float)
            ratios = (tx / pt).replace([np.inf, -np.inf], np.nan).dropna()
            ratios = ratios[(ratios > 0) & (ratios < 0.5)]
            if not ratios.empty:
                tax_rate_hist = float(ratios.tail(3).mean())

    growth = _clip(_blend(rev_cagr, bench["rev_growth"]), -0.05, 0.35)
    op_margin = _clip(_blend(op_margin_avg, bench["op_margin"]), -0.10, 0.60)
    capex = _clip(_blend(capex_pct, bench["capex_pct_rev"]), 0.0, 0.35)
    da = _clip(_blend(da_pct, bench["da_pct_rev"]), 0.0, 0.25)
    tax = _clip(_blend(tax_rate_hist, bench["tax"]), 0.0, 0.35)

    beta_final = beta if (beta and np.isfinite(beta)) else bench["beta"]
    rf = risk_free_rate if (risk_free_rate and np.isfinite(risk_free_rate)) else 0.0425

    # Terminal growth cannot exceed risk-free rate (no perpetual real-growth above the economy).
    terminal = min(0.025, rf * 0.8)

    return Assumptions(
        revenue_growth_rate=round(growth, 4),
        terminal_growth_rate=round(terminal, 4),
        operating_margin=round(op_margin, 4),
        tax_rate=round(tax, 4),
        capex_pct_revenue=round(capex, 4),
        da_pct_revenue=round(da, 4),
        nwc_pct_revenue=round(bench["nwc_pct_rev"], 4),
        risk_free_rate=round(rf, 4),
        equity_risk_premium=0.055,
        beta=round(float(beta_final), 3),
        cost_of_debt_pretax=round(rf + 0.015, 4),
        debt_weight=0.20,
        projection_years=5,
    )


ASSUMPTION_HELP: dict[str, str] = {
    "revenue_growth_rate": "Annual revenue growth during the explicit forecast window. Pre-filled from a log-linear fit on historical revenue blended with sector norms.",
    "terminal_growth_rate": "Perpetual growth after the forecast window. Should be <= long-run GDP / risk-free rate (typically 2-3%).",
    "operating_margin": "EBIT / Revenue. Pre-filled from trailing 3-year average blended with sector norm.",
    "tax_rate": "Effective tax rate used on EBIT to compute NOPAT. Pre-filled from historical effective tax blended with jurisdictional norm.",
    "capex_pct_revenue": "Capital expenditures as % of revenue. Drives reinvestment in the FCF bridge.",
    "da_pct_revenue": "Depreciation & amortization as % of revenue. Added back to EBIT in the FCF bridge.",
    "nwc_pct_revenue": "Change in net working capital as % of revenue growth. Higher for working-capital-heavy businesses.",
    "risk_free_rate": "Live 10-year US Treasury yield (fetched from the market). Anchor for the cost of equity.",
    "equity_risk_premium": "Long-run US equity risk premium. Academic consensus ~5-6%; 5.5% is a common prior.",
    "beta": "Systematic risk relative to the market. Pre-filled from yfinance; sector average used if unavailable.",
    "cost_of_debt_pretax": "Pre-tax cost of debt. Rf + credit spread is a simple, defensible proxy.",
    "debt_weight": "Debt / (Debt + Equity). Target capital structure, typically computed from the balance sheet.",
    "projection_years": "Length of the explicit-forecast period before the terminal-value stage.",
}


__all__ = ["Assumptions", "derive_assumptions", "ASSUMPTION_HELP", "SECTOR_BENCHMARKS"]
