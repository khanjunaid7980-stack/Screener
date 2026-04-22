"""Discounted Cash Flow valuation.

Produces per-year projections and a terminal value, discounts each at WACC,
and returns an equity value and per-share price.

FCFF bridge (per year):
    Revenue_t       = Revenue_{t-1} * (1 + g)
    EBIT_t          = Revenue_t * operating_margin
    NOPAT_t         = EBIT_t * (1 - tax)
    +  D&A_t        = Revenue_t * da_pct_revenue
    -  CapEx_t      = Revenue_t * capex_pct_revenue
    -  ΔNWC_t       = (Revenue_t - Revenue_{t-1}) * nwc_pct_revenue
    = FCFF_t
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .assumptions import Assumptions


@dataclass
class DCFResult:
    projections: pd.DataFrame
    terminal_value: float
    enterprise_value: float
    equity_value: float
    fair_value_per_share: float
    current_price: float | None
    upside_pct: float | None
    wacc_used: float

    def sensitivity(self, wacc_bps_range: int = 200, g_bps_range: int = 100, steps: int = 5) -> pd.DataFrame:
        """Fair-value-per-share grid over WACC x terminal growth."""
        from .dcf import run_dcf  # self-ref ok
        # built in caller via run_dcf_with_overrides
        raise NotImplementedError  # placeholder; caller uses run_dcf_with_overrides


def run_dcf(
    base_revenue: float,
    a: Assumptions,
    wacc: float,
    shares_outstanding: float,
    net_debt: float,
    current_price: float | None = None,
) -> DCFResult:
    years = max(1, int(a.projection_years))
    yrs = list(range(1, years + 1))

    rev_prev = float(base_revenue)
    rows: list[dict[str, float]] = []
    for t in yrs:
        rev_t = rev_prev * (1.0 + a.revenue_growth_rate)
        ebit_t = rev_t * a.operating_margin
        nopat_t = ebit_t * (1.0 - a.tax_rate)
        da_t = rev_t * a.da_pct_revenue
        capex_t = rev_t * a.capex_pct_revenue
        dnwc_t = (rev_t - rev_prev) * a.nwc_pct_revenue
        fcff_t = nopat_t + da_t - capex_t - dnwc_t
        rows.append({
            "Year": t,
            "Revenue": rev_t,
            "EBIT": ebit_t,
            "NOPAT": nopat_t,
            "D&A": da_t,
            "CapEx": capex_t,
            "ChangeInNWC": dnwc_t,
            "FCFF": fcff_t,
        })
        rev_prev = rev_t

    proj = pd.DataFrame(rows).set_index("Year")

    # Discount each FCFF.
    discount_factors = np.array([(1.0 + wacc) ** t for t in yrs], dtype=float)
    pv_fcff = proj["FCFF"].values / discount_factors
    proj["DiscountFactor"] = discount_factors
    proj["PV_FCFF"] = pv_fcff

    # Terminal value via Gordon growth on FCFF_{N+1}.
    g = min(a.terminal_growth_rate, wacc - 0.005)  # ensure WACC > g
    fcff_terminal_next = float(proj["FCFF"].iloc[-1]) * (1.0 + g)
    tv = fcff_terminal_next / (wacc - g) if (wacc - g) > 0 else float("nan")
    pv_tv = tv / ((1.0 + wacc) ** years)

    ev = float(np.nansum(pv_fcff) + pv_tv)
    equity_value = ev - float(net_debt or 0.0)
    fair_ps = equity_value / shares_outstanding if shares_outstanding else float("nan")
    upside = None
    if current_price and fair_ps and np.isfinite(fair_ps) and current_price > 0:
        upside = fair_ps / current_price - 1.0

    return DCFResult(
        projections=proj,
        terminal_value=tv,
        enterprise_value=ev,
        equity_value=equity_value,
        fair_value_per_share=fair_ps,
        current_price=current_price,
        upside_pct=upside,
        wacc_used=wacc,
    )


def sensitivity_grid(
    base_revenue: float,
    a: Assumptions,
    wacc_base: float,
    shares_outstanding: float,
    net_debt: float,
    wacc_steps: tuple[float, ...] = (-0.02, -0.01, 0.0, 0.01, 0.02),
    g_steps: tuple[float, ...] = (-0.01, -0.005, 0.0, 0.005, 0.01),
) -> pd.DataFrame:
    """Fair-value-per-share grid: rows=WACC, cols=terminal g."""
    rows: dict[str, dict[str, float]] = {}
    for dw in wacc_steps:
        w = wacc_base + dw
        row: dict[str, float] = {}
        for dg in g_steps:
            a2 = Assumptions(**a.to_dict())
            a2.terminal_growth_rate = a.terminal_growth_rate + dg
            if w - a2.terminal_growth_rate <= 0.005:
                row[f"{a.terminal_growth_rate + dg:.2%}"] = float("nan")
                continue
            res = run_dcf(base_revenue, a2, w, shares_outstanding, net_debt)
            row[f"{a.terminal_growth_rate + dg:.2%}"] = res.fair_value_per_share
        rows[f"{w:.2%}"] = row
    df = pd.DataFrame(rows).T
    df.index.name = "WACC"
    df.columns.name = "Terminal g"
    return df


__all__ = ["run_dcf", "DCFResult", "sensitivity_grid"]
