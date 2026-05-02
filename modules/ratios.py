"""Comprehensive financial ratios.

For every ratio we expose:
  - name:    short label (e.g. "Current Ratio")
  - formula: readable LaTeX-ish formula
  - meaning: 1-2 sentence explanation
  - good:    interpretation hint (high / low / threshold)
  - series:  pd.Series indexed by fiscal year (most-recent-last)
  - latest:  the latest-year value (None if unavailable)

Categories: liquidity, solvency, profitability, efficiency, valuation,
growth, cash_flow_quality.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class Ratio:
    name: str
    category: str
    formula: str
    meaning: str
    good: str
    series: pd.Series
    latest: Optional[float]
    unit: str = "x"  # "x" (multiplier), "%", "days", "$", or ""

    def fmt(self, val: Optional[float] = None) -> str:
        v = self.latest if val is None else val
        if v is None or (isinstance(v, float) and not np.isfinite(v)):
            return "—"
        if self.unit == "%":
            return f"{v*100:.2f}%"
        if self.unit == "days":
            return f"{v:.1f} days"
        if self.unit == "$":
            return f"${v:,.2f}"
        return f"{v:.2f}x"


def _row(fin: pd.DataFrame, key: str) -> pd.Series:
    if key not in fin.index:
        return pd.Series(dtype="float64")
    return fin.loc[key].astype(float)


def _safe_div(num: pd.Series, den: pd.Series) -> pd.Series:
    if num.empty or den.empty:
        return pd.Series(dtype="float64")
    out = num / den
    return out.replace([np.inf, -np.inf], np.nan)


def _last(s: pd.Series) -> Optional[float]:
    if s is None or s.empty:
        return None
    s = s.dropna()
    if s.empty:
        return None
    v = float(s.iloc[-1])
    return v if np.isfinite(v) else None


def _cagr(s: pd.Series, years: int) -> Optional[float]:
    s = s.dropna()
    if len(s) < 2:
        return None
    end = float(s.iloc[-1])
    n = min(years, len(s) - 1)
    start = float(s.iloc[-n - 1])
    if start <= 0 or end <= 0:
        return None
    return (end / start) ** (1.0 / n) - 1.0


def _build_ebitda(fin: pd.DataFrame) -> pd.Series:
    ebit = _row(fin, "OperatingIncome")
    da = _row(fin, "DepreciationAmortization")
    if ebit.empty:
        return pd.Series(dtype="float64")
    if da.empty:
        return ebit
    # Use index union, fillna(0) for da when missing
    return ebit.add(da.reindex(ebit.index).fillna(0), fill_value=0)


def _effective_tax(fin: pd.DataFrame) -> float:
    tx = _row(fin, "IncomeTaxExpense")
    pt = _row(fin, "PreTaxIncome")
    if tx.empty or pt.empty:
        return 0.21
    r = (tx / pt).replace([np.inf, -np.inf], np.nan).dropna()
    r = r[(r > 0) & (r < 0.5)]
    if r.empty:
        return 0.21
    return float(r.tail(3).mean())


def compute_all(
    fin: pd.DataFrame,
    market_cap: Optional[float] = None,
    price: Optional[float] = None,
) -> list[Ratio]:
    """Compute every ratio that's derivable from `fin`. Missing inputs → empty series."""
    out: list[Ratio] = []

    # Pull line items once
    revenue   = _row(fin, "Revenue")
    cogs      = _row(fin, "CostOfRevenue")
    gp        = _row(fin, "GrossProfit")
    ebit      = _row(fin, "OperatingIncome")
    ni        = _row(fin, "NetIncome")
    eps       = _row(fin, "EPS")
    interest  = _row(fin, "InterestExpense")
    da        = _row(fin, "DepreciationAmortization")
    cfo       = _row(fin, "CashFromOps")
    capex     = _row(fin, "CapEx").abs() if not _row(fin, "CapEx").empty else _row(fin, "CapEx")

    cash      = _row(fin, "CashAndEquivalents")
    ar        = _row(fin, "AccountsReceivable")
    inventory = _row(fin, "Inventory")
    cur_assets = _row(fin, "CurrentAssets")
    total_assets = _row(fin, "TotalAssets")
    ap        = _row(fin, "AccountsPayable")
    cur_liab  = _row(fin, "CurrentLiabilities")
    st_debt   = _row(fin, "ShortTermDebt").fillna(0) if not _row(fin, "ShortTermDebt").empty else pd.Series(dtype="float64")
    lt_debt   = _row(fin, "LongTermDebt").fillna(0) if not _row(fin, "LongTermDebt").empty else pd.Series(dtype="float64")
    total_debt = pd.Series(dtype="float64")
    if not lt_debt.empty or not st_debt.empty:
        total_debt = (lt_debt.add(st_debt, fill_value=0))
    total_liab = _row(fin, "TotalLiabilities")
    equity    = _row(fin, "TotalEquity")
    ebitda    = _build_ebitda(fin)
    fcf       = pd.Series(dtype="float64")
    if not cfo.empty and not capex.empty:
        fcf = (cfo - capex.reindex(cfo.index).fillna(0))

    tax_rate = _effective_tax(fin)
    nopat = ebit * (1 - tax_rate) if not ebit.empty else pd.Series(dtype="float64")
    invested_capital = pd.Series(dtype="float64")
    if not equity.empty:
        invested_capital = equity.fillna(0)
        if not total_debt.empty:
            invested_capital = invested_capital.add(total_debt.reindex(equity.index).fillna(0), fill_value=0)
        if not cash.empty:
            invested_capital = invested_capital.sub(cash.reindex(equity.index).fillna(0), fill_value=0)
        invested_capital = invested_capital.replace(0, np.nan)

    # ─── LIQUIDITY ────────────────────────────────────────────────────────
    cur_ratio = _safe_div(cur_assets, cur_liab)
    out.append(Ratio(
        name="Current Ratio", category="Liquidity", unit="x",
        formula="Current Assets / Current Liabilities",
        meaning="Short-term solvency: how many times the company can cover its current liabilities with current assets.",
        good="Healthy ≈ 1.5–3.0. Below 1 indicates short-term stress.",
        series=cur_ratio, latest=_last(cur_ratio),
    ))

    quick_num = pd.Series(dtype="float64")
    if not cur_assets.empty:
        inv_aligned = inventory.reindex(cur_assets.index).fillna(0) if not inventory.empty else pd.Series(0.0, index=cur_assets.index)
        quick_num = cur_assets - inv_aligned
    quick = _safe_div(quick_num, cur_liab)
    out.append(Ratio(
        name="Quick Ratio (Acid-Test)", category="Liquidity", unit="x",
        formula="(Current Assets − Inventory) / Current Liabilities",
        meaning="Stricter liquidity test that excludes inventory (which may take time to convert to cash).",
        good="Healthy ≥ 1.0.",
        series=quick, latest=_last(quick),
    ))

    cash_ratio = _safe_div(cash, cur_liab)
    out.append(Ratio(
        name="Cash Ratio", category="Liquidity", unit="x",
        formula="Cash & Equivalents / Current Liabilities",
        meaning="The most conservative liquidity test — only counts truly liquid assets.",
        good="Healthy ≥ 0.5; very high values may suggest idle capital.",
        series=cash_ratio, latest=_last(cash_ratio),
    ))

    ocf_ratio = _safe_div(cfo, cur_liab)
    out.append(Ratio(
        name="Operating Cash Flow Ratio", category="Liquidity", unit="x",
        formula="Cash from Operations / Current Liabilities",
        meaning="Tests whether ongoing operations generate enough cash to cover near-term obligations.",
        good="Healthy ≥ 1.0.",
        series=ocf_ratio, latest=_last(ocf_ratio),
    ))

    # ─── SOLVENCY / LEVERAGE ──────────────────────────────────────────────
    de = _safe_div(total_debt, equity)
    out.append(Ratio(
        name="Debt-to-Equity", category="Solvency", unit="x",
        formula="Total Debt / Total Equity",
        meaning="Capital structure: how much debt the company carries per dollar of equity.",
        good="Industry-dependent. <1.0 conservative, >2.0 aggressive.",
        series=de, latest=_last(de),
    ))

    da_ratio = _safe_div(total_debt, total_assets)
    out.append(Ratio(
        name="Debt-to-Assets", category="Solvency", unit="x",
        formula="Total Debt / Total Assets",
        meaning="Share of assets financed by debt.",
        good="<0.5 generally healthy.",
        series=da_ratio, latest=_last(da_ratio),
    ))

    debt_ebitda = _safe_div(total_debt, ebitda)
    out.append(Ratio(
        name="Debt / EBITDA", category="Solvency", unit="x",
        formula="Total Debt / EBITDA",
        meaning="Years of EBITDA needed to pay off debt — the credit-rating workhorse.",
        good="<3x healthy, >5x risky.",
        series=debt_ebitda, latest=_last(debt_ebitda),
    ))

    net_debt = pd.Series(dtype="float64")
    if not total_debt.empty:
        net_debt = total_debt.sub(cash.reindex(total_debt.index).fillna(0), fill_value=0) if not cash.empty else total_debt
    nd_ebitda = _safe_div(net_debt, ebitda)
    out.append(Ratio(
        name="Net Debt / EBITDA", category="Solvency", unit="x",
        formula="(Total Debt − Cash) / EBITDA",
        meaning="Net leverage — most common metric in credit covenants.",
        good="<3x healthy, negative = net cash position.",
        series=nd_ebitda, latest=_last(nd_ebitda),
    ))

    # Interest expense is reported as a positive number; coverage = EBIT / |Interest|
    interest_cov = _safe_div(ebit, interest.abs() if not interest.empty else interest)
    out.append(Ratio(
        name="Interest Coverage", category="Solvency", unit="x",
        formula="EBIT / Interest Expense",
        meaning="Operating earnings cushion over interest payments.",
        good=">8x strong, <2x distressed.",
        series=interest_cov, latest=_last(interest_cov),
    ))

    eq_mult = _safe_div(total_assets, equity)
    out.append(Ratio(
        name="Equity Multiplier", category="Solvency", unit="x",
        formula="Total Assets / Total Equity",
        meaning="Total leverage, including operating liabilities. Used in DuPont decomposition.",
        good="<3x typical for non-financials.",
        series=eq_mult, latest=_last(eq_mult),
    ))

    # ─── PROFITABILITY ────────────────────────────────────────────────────
    gm = _safe_div(gp, revenue)
    out.append(Ratio(
        name="Gross Margin", category="Profitability", unit="%",
        formula="Gross Profit / Revenue",
        meaning="Pricing power and unit economics — what's left after direct cost of producing the good/service.",
        good="Higher = stronger pricing power; software ~70%+, retail ~30%.",
        series=gm, latest=_last(gm),
    ))

    om = _safe_div(ebit, revenue)
    out.append(Ratio(
        name="Operating Margin", category="Profitability", unit="%",
        formula="EBIT / Revenue",
        meaning="Profit margin from core operations after operating costs but before financing & tax.",
        good="Higher = more efficient operations.",
        series=om, latest=_last(om),
    ))

    nm = _safe_div(ni, revenue)
    out.append(Ratio(
        name="Net Margin", category="Profitability", unit="%",
        formula="Net Income / Revenue",
        meaning="Bottom-line profit margin after everything (interest, tax, one-offs).",
        good="Higher = stronger overall profitability.",
        series=nm, latest=_last(nm),
    ))

    em = _safe_div(ebitda, revenue)
    out.append(Ratio(
        name="EBITDA Margin", category="Profitability", unit="%",
        formula="EBITDA / Revenue",
        meaning="Cash-operating margin — strips out non-cash D&A. Useful cross-industry comparison.",
        good="Industry-dependent.",
        series=em, latest=_last(em),
    ))

    roa = _safe_div(ni, total_assets)
    out.append(Ratio(
        name="Return on Assets (ROA)", category="Profitability", unit="%",
        formula="Net Income / Total Assets",
        meaning="How much profit the company generates per dollar of assets.",
        good=">5% solid for non-financials.",
        series=roa, latest=_last(roa),
    ))

    roe = _safe_div(ni, equity)
    out.append(Ratio(
        name="Return on Equity (ROE)", category="Profitability", unit="%",
        formula="Net Income / Total Equity",
        meaning="Profit per dollar of shareholder equity. Buffett's favorite single metric.",
        good=">15% strong, >20% excellent.",
        series=roe, latest=_last(roe),
    ))

    roic = _safe_div(nopat, invested_capital)
    out.append(Ratio(
        name="Return on Invested Capital (ROIC)", category="Profitability", unit="%",
        formula="EBIT × (1 − tax) / (Equity + Debt − Cash)",
        meaning="Cleanest measure of capital efficiency. Compared against WACC: ROIC > WACC = value creation.",
        good="Should exceed WACC by a meaningful spread.",
        series=roic, latest=_last(roic),
    ))

    # ─── EFFICIENCY ───────────────────────────────────────────────────────
    asset_turn = _safe_div(revenue, total_assets)
    out.append(Ratio(
        name="Asset Turnover", category="Efficiency", unit="x",
        formula="Revenue / Total Assets",
        meaning="How efficiently the company generates sales from its asset base.",
        good="Higher = more sales per asset dollar.",
        series=asset_turn, latest=_last(asset_turn),
    ))

    inv_turn = _safe_div(cogs, inventory)
    out.append(Ratio(
        name="Inventory Turnover", category="Efficiency", unit="x",
        formula="COGS / Inventory",
        meaning="How many times inventory is sold and replaced per year.",
        good="Higher = leaner inventory; industry-dependent.",
        series=inv_turn, latest=_last(inv_turn),
    ))

    dio = (365.0 / inv_turn).replace([np.inf, -np.inf], np.nan) if not inv_turn.empty else pd.Series(dtype="float64")
    out.append(Ratio(
        name="Days Inventory Outstanding (DIO)", category="Efficiency", unit="days",
        formula="365 / Inventory Turnover",
        meaning="Average days inventory sits before being sold.",
        good="Lower = faster inventory cycle.",
        series=dio, latest=_last(dio),
    ))

    ar_turn = _safe_div(revenue, ar)
    out.append(Ratio(
        name="Receivables Turnover", category="Efficiency", unit="x",
        formula="Revenue / Accounts Receivable",
        meaning="How quickly the company collects cash from credit sales.",
        good="Higher = faster collection.",
        series=ar_turn, latest=_last(ar_turn),
    ))

    dso = (365.0 / ar_turn).replace([np.inf, -np.inf], np.nan) if not ar_turn.empty else pd.Series(dtype="float64")
    out.append(Ratio(
        name="Days Sales Outstanding (DSO)", category="Efficiency", unit="days",
        formula="365 / Receivables Turnover",
        meaning="Average days to collect cash after a sale.",
        good="Lower is better; <60 days typical.",
        series=dso, latest=_last(dso),
    ))

    ap_turn = _safe_div(cogs, ap)
    out.append(Ratio(
        name="Payables Turnover", category="Efficiency", unit="x",
        formula="COGS / Accounts Payable",
        meaning="How quickly the company pays its suppliers.",
        good="Stable preferred; very high values may strain supplier relationships.",
        series=ap_turn, latest=_last(ap_turn),
    ))

    dpo = (365.0 / ap_turn).replace([np.inf, -np.inf], np.nan) if not ap_turn.empty else pd.Series(dtype="float64")
    out.append(Ratio(
        name="Days Payable Outstanding (DPO)", category="Efficiency", unit="days",
        formula="365 / Payables Turnover",
        meaning="Average days the company takes to pay suppliers.",
        good="Higher = better working capital — but don't strain suppliers.",
        series=dpo, latest=_last(dpo),
    ))

    # Cash conversion cycle = DIO + DSO − DPO
    if not dio.empty and not dso.empty and not dpo.empty:
        ccc = dio.add(dso, fill_value=0).sub(dpo, fill_value=0)
    else:
        ccc = pd.Series(dtype="float64")
    out.append(Ratio(
        name="Cash Conversion Cycle", category="Efficiency", unit="days",
        formula="DIO + DSO − DPO",
        meaning="Days between paying suppliers and collecting from customers. Negative = customers pay you before suppliers do (Amazon, Apple).",
        good="Lower (or negative) = better.",
        series=ccc, latest=_last(ccc),
    ))

    # ─── VALUATION (latest only — needs market cap) ────────────────────────
    if market_cap and price and not eps.empty and _last(eps) and _last(eps) > 0:
        pe_val = price / _last(eps)
        out.append(Ratio(
            name="P/E (Trailing)", category="Valuation", unit="x",
            formula="Price / Diluted EPS",
            meaning="Years of current earnings to recoup the share price. Mean-reverts; compare to peers and history.",
            good="Industry-dependent; lower can mean cheap or distressed.",
            series=pd.Series({fin.columns[-1]: pe_val}), latest=pe_val,
        ))

    if market_cap and not equity.empty and _last(equity):
        pb_val = market_cap / _last(equity)
        out.append(Ratio(
            name="P/B (Price / Book)", category="Valuation", unit="x",
            formula="Market Cap / Total Equity",
            meaning="Market value relative to book value of equity. <1 may be value or trouble.",
            good="Industry-dependent; banks ~1, tech often >5.",
            series=pd.Series({fin.columns[-1]: pb_val}), latest=pb_val,
        ))

    if market_cap and not revenue.empty and _last(revenue):
        ps_val = market_cap / _last(revenue)
        out.append(Ratio(
            name="P/S (Price / Sales)", category="Valuation", unit="x",
            formula="Market Cap / Revenue",
            meaning="Useful when earnings are negative or volatile. Comparable cross-margin businesses.",
            good="Lower = cheaper. Industry-dependent.",
            series=pd.Series({fin.columns[-1]: ps_val}), latest=ps_val,
        ))

    if market_cap and _last(net_debt) is not None and not ebitda.empty and _last(ebitda):
        ev = market_cap + _last(net_debt)
        ev_ebitda = ev / _last(ebitda)
        out.append(Ratio(
            name="EV / EBITDA", category="Valuation", unit="x",
            formula="(Market Cap + Net Debt) / EBITDA",
            meaning="Capital-structure-neutral valuation multiple. The standard for M&A pricing.",
            good="<10x cheap, >20x expensive (industry-dependent).",
            series=pd.Series({fin.columns[-1]: ev_ebitda}), latest=ev_ebitda,
        ))
        if not revenue.empty and _last(revenue):
            ev_sales = ev / _last(revenue)
            out.append(Ratio(
                name="EV / Sales", category="Valuation", unit="x",
                formula="(Market Cap + Net Debt) / Revenue",
                meaning="Capital-structure-neutral revenue multiple.",
                good="Lower = cheaper. Industry-dependent.",
                series=pd.Series({fin.columns[-1]: ev_sales}), latest=ev_sales,
            ))

    if market_cap and not fcf.empty and _last(fcf):
        fcfy = _last(fcf) / market_cap
        out.append(Ratio(
            name="FCF Yield", category="Valuation", unit="%",
            formula="Free Cash Flow / Market Cap",
            meaning="Cash returned to investors as a yield. Inverse of P/FCF.",
            good=">5% attractive; >8% deep value.",
            series=pd.Series({fin.columns[-1]: fcfy}), latest=fcfy,
        ))

    if market_cap and not ni.empty and _last(ni):
        ey = _last(ni) / market_cap
        out.append(Ratio(
            name="Earnings Yield", category="Valuation", unit="%",
            formula="Net Income / Market Cap",
            meaning="The 'E/P' — inverse of P/E. Compare against the 10Y Treasury for an equity-vs-bond view.",
            good=">5% attractive vs. risk-free rate.",
            series=pd.Series({fin.columns[-1]: ey}), latest=ey,
        ))

    # ─── GROWTH ───────────────────────────────────────────────────────────
    out.append(Ratio(
        name="Revenue CAGR (3-yr)", category="Growth", unit="%",
        formula="(Rev_t / Rev_{t-3})^(1/3) − 1",
        meaning="Compound annual revenue growth over the last 3 reported years.",
        good=">10% strong; negative = shrinking.",
        series=pd.Series(dtype="float64"),
        latest=_cagr(revenue, 3),
    ))
    out.append(Ratio(
        name="Revenue CAGR (5-yr)", category="Growth", unit="%",
        formula="(Rev_t / Rev_{t-5})^(1/5) − 1",
        meaning="Compound annual revenue growth over the last 5 reported years.",
        good=">10% strong; negative = shrinking.",
        series=pd.Series(dtype="float64"),
        latest=_cagr(revenue, 5),
    ))
    out.append(Ratio(
        name="EPS CAGR (3-yr)", category="Growth", unit="%",
        formula="(EPS_t / EPS_{t-3})^(1/3) − 1",
        meaning="Compound annual diluted EPS growth.",
        good=">10% strong.",
        series=pd.Series(dtype="float64"),
        latest=_cagr(eps, 3),
    ))
    out.append(Ratio(
        name="FCF CAGR (3-yr)", category="Growth", unit="%",
        formula="(FCF_t / FCF_{t-3})^(1/3) − 1",
        meaning="Compound annual free-cash-flow growth.",
        good=">10% strong; consistent positive FCF growth = compounding machine.",
        series=pd.Series(dtype="float64"),
        latest=_cagr(fcf, 3),
    ))

    # ─── CASH FLOW QUALITY ────────────────────────────────────────────────
    cash_conv = _safe_div(cfo, ni)
    out.append(Ratio(
        name="Cash Conversion (CFO / NI)", category="Cash Flow Quality", unit="x",
        formula="Cash from Operations / Net Income",
        meaning="How much of reported earnings actually shows up as cash. Persistent <1 = accruals risk.",
        good=">1 healthy; >1.2 excellent.",
        series=cash_conv, latest=_last(cash_conv),
    ))

    fcf_ni = _safe_div(fcf, ni)
    out.append(Ratio(
        name="FCF / Net Income", category="Cash Flow Quality", unit="x",
        formula="Free Cash Flow / Net Income",
        meaning="Stricter cash-quality test, after CapEx.",
        good=">0.8 healthy.",
        series=fcf_ni, latest=_last(fcf_ni),
    ))

    capex_rev = _safe_div(capex, revenue) if not capex.empty else pd.Series(dtype="float64")
    out.append(Ratio(
        name="CapEx / Revenue", category="Cash Flow Quality", unit="%",
        formula="|CapEx| / Revenue",
        meaning="Capital intensity. Software ~3-5%, telecom/utilities ~15-20%.",
        good="Lower = capital-light; depends on industry.",
        series=capex_rev, latest=_last(capex_rev),
    ))

    capex_da = _safe_div(capex.abs() if not capex.empty else capex, da)
    out.append(Ratio(
        name="CapEx / D&A", category="Cash Flow Quality", unit="x",
        formula="|CapEx| / Depreciation & Amortization",
        meaning="≈ 1.0 = maintenance-only investment. >1.0 = growth investment, <1.0 = under-investing.",
        good="Industry-dependent.",
        series=capex_da, latest=_last(capex_da),
    ))

    return out


def by_category(ratios: list[Ratio]) -> dict[str, list[Ratio]]:
    out: dict[str, list[Ratio]] = {}
    for r in ratios:
        out.setdefault(r.category, []).append(r)
    return out


__all__ = ["Ratio", "compute_all", "by_category"]
