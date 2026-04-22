"""Interactive Plotly charts for the Visualization tab."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def _safe_ratio(num: pd.Series, den: pd.Series) -> pd.Series:
    return (num / den).replace([np.inf, -np.inf], np.nan)


def revenue_and_margins(fin: pd.DataFrame) -> go.Figure:
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    if "Revenue" in fin.index:
        rev = fin.loc["Revenue"].astype(float)
        fig.add_trace(go.Bar(x=rev.index.astype(str), y=rev.values / 1e9, name="Revenue ($B)", marker_color="#1f6feb"), secondary_y=False)
    if "GrossProfit" in fin.index and "Revenue" in fin.index:
        gm = _safe_ratio(fin.loc["GrossProfit"].astype(float), fin.loc["Revenue"].astype(float)) * 100
        fig.add_trace(go.Scatter(x=gm.index.astype(str), y=gm.values, name="Gross margin (%)", mode="lines+markers", line=dict(color="#2da44e")), secondary_y=True)
    if "OperatingIncome" in fin.index and "Revenue" in fin.index:
        om = _safe_ratio(fin.loc["OperatingIncome"].astype(float), fin.loc["Revenue"].astype(float)) * 100
        fig.add_trace(go.Scatter(x=om.index.astype(str), y=om.values, name="Operating margin (%)", mode="lines+markers", line=dict(color="#bf8700")), secondary_y=True)
    if "NetIncome" in fin.index and "Revenue" in fin.index:
        nm = _safe_ratio(fin.loc["NetIncome"].astype(float), fin.loc["Revenue"].astype(float)) * 100
        fig.add_trace(go.Scatter(x=nm.index.astype(str), y=nm.values, name="Net margin (%)", mode="lines+markers", line=dict(color="#cf222e")), secondary_y=True)
    fig.update_layout(title="Revenue & Profit Margins", template="simple_white", hovermode="x unified", legend=dict(orientation="h", y=-0.2))
    fig.update_yaxes(title_text="Revenue ($B)", secondary_y=False)
    fig.update_yaxes(title_text="Margin (%)", secondary_y=True)
    return fig


def roic_vs_wacc(fin: pd.DataFrame, wacc: float) -> go.Figure:
    fig = go.Figure()
    if all(k in fin.index for k in ("OperatingIncome", "TotalAssets", "TotalLiabilities", "CashAndEquivalents")):
        ebit = fin.loc["OperatingIncome"].astype(float)
        # Invested capital ~= Total assets - current liabilities - excess cash.
        # Approximation: Total assets - Total liabilities + interest-bearing debt.
        debt = (
            fin.loc["LongTermDebt"].astype(float).fillna(0)
            if "LongTermDebt" in fin.index else pd.Series(0.0, index=ebit.index)
        ) + (
            fin.loc["ShortTermDebt"].astype(float).fillna(0)
            if "ShortTermDebt" in fin.index else pd.Series(0.0, index=ebit.index)
        )
        equity = fin.loc["TotalEquity"].astype(float) if "TotalEquity" in fin.index else pd.Series(0.0, index=ebit.index)
        cash = fin.loc["CashAndEquivalents"].astype(float).fillna(0) if "CashAndEquivalents" in fin.index else pd.Series(0.0, index=ebit.index)
        invested = (equity.fillna(0) + debt - cash).replace(0, np.nan)
        tax_rate = 0.21
        if "IncomeTaxExpense" in fin.index and "PreTaxIncome" in fin.index:
            tx = fin.loc["IncomeTaxExpense"].astype(float)
            pt = fin.loc["PreTaxIncome"].astype(float)
            ratios = (tx / pt).replace([np.inf, -np.inf], np.nan).dropna()
            ratios = ratios[(ratios > 0) & (ratios < 0.5)]
            if not ratios.empty:
                tax_rate = float(ratios.tail(3).mean())
        roic = ((ebit * (1 - tax_rate)) / invested) * 100
        fig.add_trace(go.Bar(x=roic.index.astype(str), y=roic.values, name="ROIC (%)", marker_color="#1f6feb"))
        fig.add_hline(y=wacc * 100, line_dash="dash", line_color="#cf222e", annotation_text=f"WACC = {wacc:.2%}", annotation_position="top left")
    fig.update_layout(title="ROIC vs. WACC", template="simple_white", yaxis_title="%", hovermode="x unified")
    return fig


def ebit_to_market_cap(fin: pd.DataFrame, market_cap: float | None) -> go.Figure:
    fig = go.Figure()
    if "OperatingIncome" in fin.index and market_cap:
        ebit = fin.loc["OperatingIncome"].astype(float)
        ratio = (ebit / market_cap) * 100
        fig.add_trace(go.Scatter(x=ratio.index.astype(str), y=ratio.values, name="EBIT / Market Cap (%)", mode="lines+markers", line=dict(color="#8250df")))
    fig.update_layout(title="EBIT / Market Cap — Earnings Yield Proxy", template="simple_white", yaxis_title="%", hovermode="x unified")
    return fig


def cash_flow_breakdown(fin: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if "CashFromOps" in fin.index:
        cfo = fin.loc["CashFromOps"].astype(float) / 1e9
        fig.add_trace(go.Bar(x=cfo.index.astype(str), y=cfo.values, name="Cash from Ops ($B)", marker_color="#2da44e"))
    if "CapEx" in fin.index:
        cx = fin.loc["CapEx"].astype(float).abs() / 1e9
        fig.add_trace(go.Bar(x=cx.index.astype(str), y=-cx.values, name="CapEx ($B)", marker_color="#cf222e"))
    if "CashFromOps" in fin.index and "CapEx" in fin.index:
        fcf = (fin.loc["CashFromOps"].astype(float) - fin.loc["CapEx"].astype(float).abs()) / 1e9
        fig.add_trace(go.Scatter(x=fcf.index.astype(str), y=fcf.values, name="Free Cash Flow ($B)", mode="lines+markers", line=dict(color="#1f6feb", width=3)))
    fig.update_layout(title="Cash Flow Breakdown", template="simple_white", yaxis_title="$B", barmode="relative", hovermode="x unified")
    return fig


def leverage_and_liquidity(fin: pd.DataFrame) -> go.Figure:
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    debt = None
    if "LongTermDebt" in fin.index or "ShortTermDebt" in fin.index:
        lt = fin.loc["LongTermDebt"].astype(float).fillna(0) if "LongTermDebt" in fin.index else 0
        st = fin.loc["ShortTermDebt"].astype(float).fillna(0) if "ShortTermDebt" in fin.index else 0
        debt = (lt + st) / 1e9
        fig.add_trace(go.Bar(x=debt.index.astype(str), y=debt.values, name="Total Debt ($B)", marker_color="#cf222e"), secondary_y=False)
    if "CashAndEquivalents" in fin.index:
        cash = fin.loc["CashAndEquivalents"].astype(float) / 1e9
        fig.add_trace(go.Bar(x=cash.index.astype(str), y=cash.values, name="Cash ($B)", marker_color="#2da44e"), secondary_y=False)
    if debt is not None and "OperatingIncome" in fin.index:
        ebit = fin.loc["OperatingIncome"].astype(float) / 1e9
        dte = (debt / ebit).replace([np.inf, -np.inf], np.nan)
        fig.add_trace(go.Scatter(x=dte.index.astype(str), y=dte.values, name="Debt / EBIT (x)", mode="lines+markers", line=dict(color="#8250df")), secondary_y=True)
    fig.update_layout(title="Leverage & Liquidity", template="simple_white", barmode="group", hovermode="x unified")
    fig.update_yaxes(title_text="$B", secondary_y=False)
    fig.update_yaxes(title_text="x", secondary_y=True)
    return fig


def dcf_projection_chart(projections: pd.DataFrame) -> go.Figure:
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=projections.index.astype(str), y=projections["FCFF"].values / 1e9, name="FCFF ($B)", marker_color="#1f6feb"), secondary_y=False)
    fig.add_trace(go.Bar(x=projections.index.astype(str), y=projections["PV_FCFF"].values / 1e9, name="PV of FCFF ($B)", marker_color="#8250df"), secondary_y=False)
    fig.add_trace(go.Scatter(x=projections.index.astype(str), y=projections["Revenue"].values / 1e9, name="Revenue ($B)", mode="lines+markers", line=dict(color="#2da44e", width=3)), secondary_y=True)
    fig.update_layout(title="DCF Projections", template="simple_white", barmode="group", hovermode="x unified", xaxis_title="Projection Year")
    fig.update_yaxes(title_text="Cash Flow ($B)", secondary_y=False)
    fig.update_yaxes(title_text="Revenue ($B)", secondary_y=True)
    return fig


def price_history_chart(hist: pd.DataFrame, fair_value: float | None = None) -> go.Figure:
    fig = go.Figure()
    if not hist.empty:
        fig.add_trace(go.Scatter(x=hist["Date"], y=hist["Close"], name="Close", mode="lines", line=dict(color="#1f6feb")))
    if fair_value and np.isfinite(fair_value):
        fig.add_hline(y=fair_value, line_dash="dash", line_color="#2da44e", annotation_text=f"DCF Fair Value ${fair_value:,.2f}", annotation_position="top left")
    fig.update_layout(title="Price History vs. DCF Fair Value", template="simple_white", yaxis_title="Price", hovermode="x unified")
    return fig


def sensitivity_heatmap(grid: pd.DataFrame, current_price: float | None = None) -> go.Figure:
    z = grid.values
    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=grid.columns.astype(str),
        y=grid.index.astype(str),
        colorscale="RdYlGn",
        colorbar=dict(title="Fair Value / share"),
        text=np.round(z, 2),
        texttemplate="%{text}",
    ))
    title = "Sensitivity: Fair Value per Share (WACC × Terminal g)"
    if current_price:
        title += f" — current price ${current_price:,.2f}"
    fig.update_layout(title=title, template="simple_white", xaxis_title="Terminal g", yaxis_title="WACC")
    return fig


__all__ = [
    "revenue_and_margins",
    "roic_vs_wacc",
    "ebit_to_market_cap",
    "cash_flow_breakdown",
    "leverage_and_liquidity",
    "dcf_projection_chart",
    "price_history_chart",
    "sensitivity_heatmap",
]
