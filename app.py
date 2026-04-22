"""Financial Screening Tool — Streamlit app.

Run locally:  streamlit run app.py
Deploy free:  push to GitHub, then connect the repo at https://streamlit.io/cloud
"""

from __future__ import annotations

import io
import json
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st

from modules import edgar, market_data, visualizations as viz
from modules.assumptions import ASSUMPTION_HELP, Assumptions, derive_assumptions
from modules.dcf import run_dcf, sensitivity_grid
from modules.wacc import compute_wacc, infer_debt_weight_from_balance_sheet

st.set_page_config(
    page_title="Financial Screening Tool",
    page_icon=":bar_chart:",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------- Sidebar: company + years ----------------

st.sidebar.title("Financial Screener")
st.sidebar.caption("Free · SEC EDGAR + Yahoo Finance · No API keys required")

ticker = st.sidebar.text_input("Ticker symbol", value="AAPL").strip().upper()
years = st.sidebar.slider("Years of historical data", 3, 10, 5, 1,
                          help="How many annual 10-K periods to pull from EDGAR.")

fetch = st.sidebar.button("Fetch data", type="primary", use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**How to use**\n"
    "1. Enter a ticker and click *Fetch data*\n"
    "2. Review / edit assumptions (pre-filled from ML + sector benchmarks)\n"
    "3. Check WACC and DCF outputs\n"
    "4. Open *Visualizations* for interactive charts"
)


# ---------------- Cached data fetchers ----------------

@st.cache_data(ttl=60 * 60, show_spinner=False)
def load_company_data(ticker_symbol: str, n_years: int) -> dict:
    info = edgar.resolve_ticker(ticker_symbol)
    if not info:
        raise ValueError(f"Ticker {ticker_symbol} not found in SEC's public ticker file.")
    facts = edgar.fetch_company_facts(info["cik"])
    fin = facts.build_financials(n_years)
    filings = edgar.fetch_recent_filings(info["cik"])
    transcripts_hint = edgar.fetch_transcripts_hint(ticker_symbol)
    snap = market_data.fetch_market_snapshot(ticker_symbol)
    price_hist = market_data.fetch_price_history(ticker_symbol, period=f"{min(n_years, 10)}y")
    rf = market_data.fetch_risk_free_rate()
    return {
        "cik": info["cik"],
        "name": facts.name or info["title"],
        "financials": fin,
        "filings": filings,
        "transcripts_hint": transcripts_hint,
        "snapshot": snap,
        "price_history": price_hist,
        "risk_free_rate": rf,
    }


def _init_session_assumptions(data: dict) -> None:
    if "assumptions" in st.session_state and st.session_state.get("assumptions_ticker") == data["cik"]:
        return
    snap: market_data.MarketSnapshot = data["snapshot"]
    a = derive_assumptions(
        data["financials"],
        sector=snap.sector,
        risk_free_rate=data["risk_free_rate"],
        beta=snap.beta,
    )
    # Try to anchor debt weight to balance sheet * market cap if we have both.
    fin = data["financials"]
    if not fin.empty and snap.market_cap:
        total_debt = 0.0
        if "LongTermDebt" in fin.index:
            v = fin.loc["LongTermDebt"].dropna()
            if not v.empty:
                total_debt += float(v.iloc[-1])
        if "ShortTermDebt" in fin.index:
            v = fin.loc["ShortTermDebt"].dropna()
            if not v.empty:
                total_debt += float(v.iloc[-1])
        dw = infer_debt_weight_from_balance_sheet(total_debt, snap.market_cap)
        if dw is not None:
            a.debt_weight = round(dw, 4)
    st.session_state["assumptions"] = a
    st.session_state["assumptions_ticker"] = data["cik"]


# ---------------- Fetch on demand ----------------

if fetch or "data" not in st.session_state:
    if fetch:
        st.session_state.pop("data", None)
        st.session_state.pop("assumptions", None)
        st.session_state.pop("assumptions_ticker", None)
    try:
        with st.spinner(f"Fetching {ticker} from SEC EDGAR + Yahoo Finance..."):
            st.session_state["data"] = load_company_data(ticker, years)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not load data for **{ticker}**: {exc}")
        st.stop()

data = st.session_state.get("data")
if not data:
    st.info("Enter a US-listed ticker in the sidebar and click **Fetch data** to begin.")
    st.stop()

_init_session_assumptions(data)
assumptions: Assumptions = st.session_state["assumptions"]
fin: pd.DataFrame = data["financials"]
snap: market_data.MarketSnapshot = data["snapshot"]


# ---------------- Header / KPIs ----------------

st.title(f":bar_chart: {data['name']} ({ticker})")
st.caption(f"CIK {data['cik']} · {snap.sector or '—'} · {snap.industry or '—'} · Currency {snap.currency or 'USD'}")

kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
kpi1.metric("Price", f"${snap.price:,.2f}" if snap.price else "—")
kpi2.metric("Market Cap", f"${snap.market_cap/1e9:,.1f}B" if snap.market_cap else "—")
kpi3.metric("Beta", f"{snap.beta:.2f}" if snap.beta else "—")
kpi4.metric("10Y Treasury (Rf)", f"{data['risk_free_rate']:.2%}")
latest_rev = None
if "Revenue" in fin.index and not fin.loc["Revenue"].dropna().empty:
    latest_rev = float(fin.loc["Revenue"].dropna().iloc[-1])
kpi5.metric("Latest Revenue (FY)", f"${latest_rev/1e9:,.1f}B" if latest_rev else "—")

st.markdown("---")


# ---------------- Tabs ----------------

tab_data, tab_assume, tab_wacc, tab_dcf, tab_viz, tab_docs = st.tabs(
    ["Data", "Assumptions", "WACC", "DCF Valuation", "Visualizations", "Guide"]
)


# ---------- Tab: Data ----------
with tab_data:
    st.subheader("Historical financials (from SEC EDGAR company-facts API)")
    if fin.empty:
        st.warning("No annual US-GAAP facts returned. Try a larger US issuer or a different ticker.")
    else:
        display = fin.copy()
        display.columns = [str(c) for c in display.columns]
        st.dataframe(
            display.style.format("{:,.0f}", na_rep="—"),
            use_container_width=True,
            height=min(600, 40 + 32 * len(display.index)),
        )
        csv = display.to_csv().encode()
        st.download_button("Download financials as CSV", csv, f"{ticker}_financials.csv", "text/csv")

    st.markdown("#### Recent SEC filings")
    filings = data["filings"]
    if isinstance(filings, pd.DataFrame) and not filings.empty:
        show = filings.head(25).copy()
        show["link"] = show["url"].apply(lambda u: f"[open]({u})")
        st.dataframe(show[["form", "filingDate", "accessionNumber", "link"]], use_container_width=True, hide_index=True)
    else:
        st.info("No recent filings found.")

    st.markdown("#### Earnings call transcripts")
    st.markdown(
        f"SEC does not host transcripts directly. Free transcripts are typically available via IR sites and "
        f"third-party aggregators — [search for the latest {ticker} transcript]({data['transcripts_hint']})."
    )


# ---------- Tab: Assumptions ----------
with tab_assume:
    st.subheader("Assumptions (pre-filled; editable)")
    st.caption("Blended from your company's trend + sector benchmark + live market. Hover the **?** for context.")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**Growth & margins**")
        assumptions.revenue_growth_rate = st.number_input("Revenue growth (yearly)", -0.20, 0.60, float(assumptions.revenue_growth_rate), 0.005, format="%.4f", help=ASSUMPTION_HELP["revenue_growth_rate"])
        assumptions.terminal_growth_rate = st.number_input("Terminal growth", 0.0, 0.05, float(assumptions.terminal_growth_rate), 0.0025, format="%.4f", help=ASSUMPTION_HELP["terminal_growth_rate"])
        assumptions.operating_margin = st.number_input("Operating margin (EBIT)", -0.20, 0.70, float(assumptions.operating_margin), 0.005, format="%.4f", help=ASSUMPTION_HELP["operating_margin"])
        assumptions.tax_rate = st.number_input("Effective tax rate", 0.0, 0.45, float(assumptions.tax_rate), 0.005, format="%.4f", help=ASSUMPTION_HELP["tax_rate"])

    with col2:
        st.markdown("**Reinvestment**")
        assumptions.capex_pct_revenue = st.number_input("CapEx / Revenue", 0.0, 0.40, float(assumptions.capex_pct_revenue), 0.005, format="%.4f", help=ASSUMPTION_HELP["capex_pct_revenue"])
        assumptions.da_pct_revenue = st.number_input("D&A / Revenue", 0.0, 0.30, float(assumptions.da_pct_revenue), 0.005, format="%.4f", help=ASSUMPTION_HELP["da_pct_revenue"])
        assumptions.nwc_pct_revenue = st.number_input("ΔNWC / ΔRevenue", 0.0, 0.30, float(assumptions.nwc_pct_revenue), 0.005, format="%.4f", help=ASSUMPTION_HELP["nwc_pct_revenue"])
        assumptions.projection_years = int(st.number_input("Projection years", 3, 15, int(assumptions.projection_years), 1, help=ASSUMPTION_HELP["projection_years"]))

    with col3:
        st.markdown("**Capital costs**")
        assumptions.risk_free_rate = st.number_input("Risk-free rate (Rf)", 0.0, 0.15, float(assumptions.risk_free_rate), 0.0025, format="%.4f", help=ASSUMPTION_HELP["risk_free_rate"])
        assumptions.equity_risk_premium = st.number_input("Equity risk premium", 0.02, 0.10, float(assumptions.equity_risk_premium), 0.0025, format="%.4f", help=ASSUMPTION_HELP["equity_risk_premium"])
        assumptions.beta = st.number_input("Beta", 0.0, 3.0, float(assumptions.beta), 0.05, format="%.3f", help=ASSUMPTION_HELP["beta"])
        assumptions.cost_of_debt_pretax = st.number_input("Cost of debt (pre-tax)", 0.0, 0.20, float(assumptions.cost_of_debt_pretax), 0.0025, format="%.4f", help=ASSUMPTION_HELP["cost_of_debt_pretax"])
        assumptions.debt_weight = st.number_input("Debt weight D/(D+E)", 0.0, 0.90, float(assumptions.debt_weight), 0.01, format="%.4f", help=ASSUMPTION_HELP["debt_weight"])

    st.session_state["assumptions"] = assumptions

    # Reset / export
    reset_col, export_col, import_col = st.columns([1, 1, 2])
    if reset_col.button("Reset to ML defaults"):
        a = derive_assumptions(fin, sector=snap.sector, risk_free_rate=data["risk_free_rate"], beta=snap.beta)
        st.session_state["assumptions"] = a
        st.rerun()
    export_col.download_button(
        "Export assumptions (JSON)",
        data=json.dumps(assumptions.to_dict(), indent=2).encode(),
        file_name=f"{ticker}_assumptions.json",
        mime="application/json",
    )
    uploaded = import_col.file_uploader("Import assumptions JSON", type=["json"], label_visibility="collapsed")
    if uploaded is not None:
        try:
            blob = json.load(uploaded)
            st.session_state["assumptions"] = Assumptions(**{**assumptions.to_dict(), **blob})
            st.success("Assumptions imported.")
            st.rerun()
        except Exception as exc:  # noqa: BLE001
            st.error(f"Invalid JSON: {exc}")


# ---------- Tab: WACC ----------
with tab_wacc:
    st.subheader("Weighted Average Cost of Capital")
    wacc_result = compute_wacc(assumptions)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Cost of Equity (Ke)", f"{wacc_result.cost_of_equity:.2%}")
    c2.metric("After-tax Kd", f"{wacc_result.after_tax_cost_of_debt:.2%}")
    c3.metric("Equity weight", f"{wacc_result.equity_weight:.1%}")
    c4.metric("Debt weight", f"{wacc_result.debt_weight:.1%}")
    c5.metric("WACC", f"{wacc_result.wacc:.2%}")

    st.markdown("**Formula**")
    st.latex(r"Ke = R_f + \beta \cdot ERP")
    st.latex(r"WACC = W_e \cdot K_e + W_d \cdot K_d \cdot (1 - t)")

    with st.expander("Walk-through with your current inputs"):
        st.markdown(
            f"- Rf = **{assumptions.risk_free_rate:.2%}**, β = **{assumptions.beta:.2f}**, "
            f"ERP = **{assumptions.equity_risk_premium:.2%}**  →  Ke = Rf + β·ERP = **{wacc_result.cost_of_equity:.2%}**\n"
            f"- Kd (pre-tax) = **{assumptions.cost_of_debt_pretax:.2%}**, tax = **{assumptions.tax_rate:.2%}**  →  "
            f"Kd·(1−t) = **{wacc_result.after_tax_cost_of_debt:.2%}**\n"
            f"- Weights: We = **{wacc_result.equity_weight:.1%}**, Wd = **{wacc_result.debt_weight:.1%}**\n"
            f"- WACC = We·Ke + Wd·Kd·(1−t) = **{wacc_result.wacc:.2%}**"
        )

    st.session_state["wacc"] = wacc_result.wacc


# ---------- Tab: DCF ----------
with tab_dcf:
    st.subheader("Discounted Cash Flow valuation")

    if "Revenue" not in fin.index or fin.loc["Revenue"].dropna().empty:
        st.warning("Cannot run DCF — no revenue data available.")
        st.stop()

    base_rev = float(fin.loc["Revenue"].dropna().iloc[-1])

    # Shares outstanding: prefer market snapshot, fall back to last reported.
    shares = snap.shares_outstanding
    if not shares and "SharesOutstanding" in fin.index:
        s = fin.loc["SharesOutstanding"].dropna()
        if not s.empty:
            shares = float(s.iloc[-1])
    if not shares and snap.market_cap and snap.price:
        shares = snap.market_cap / snap.price
    shares = float(shares) if shares else 0.0

    # Net debt: last-reported total debt minus cash.
    total_debt = 0.0
    for k in ("LongTermDebt", "ShortTermDebt"):
        if k in fin.index:
            v = fin.loc[k].dropna()
            if not v.empty:
                total_debt += float(v.iloc[-1])
    cash = 0.0
    if "CashAndEquivalents" in fin.index:
        v = fin.loc["CashAndEquivalents"].dropna()
        if not v.empty:
            cash = float(v.iloc[-1])
    net_debt = total_debt - cash

    wacc_used = float(st.session_state.get("wacc", compute_wacc(assumptions).wacc))

    # Allow user override of shares / net debt for edge cases.
    o1, o2, o3 = st.columns(3)
    shares = o1.number_input("Shares outstanding", min_value=0.0, value=float(shares), step=1e6, format="%.0f")
    net_debt = o2.number_input("Net debt ($)", value=float(net_debt), step=1e8, format="%.0f")
    wacc_used = o3.number_input("WACC used (from WACC tab)", min_value=0.005, max_value=0.40, value=float(wacc_used), step=0.0025, format="%.4f")

    result = run_dcf(
        base_revenue=base_rev,
        a=assumptions,
        wacc=wacc_used,
        shares_outstanding=shares,
        net_debt=net_debt,
        current_price=snap.price,
    )

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Enterprise Value", f"${result.enterprise_value/1e9:,.2f}B")
    k2.metric("Equity Value", f"${result.equity_value/1e9:,.2f}B")
    k3.metric("Fair Value / Share", f"${result.fair_value_per_share:,.2f}" if np.isfinite(result.fair_value_per_share) else "—")
    k4.metric(
        "Upside vs. market",
        f"{result.upside_pct:.1%}" if result.upside_pct is not None else "—",
        delta=f"${(result.fair_value_per_share - (snap.price or 0)):,.2f}" if snap.price else None,
    )

    st.markdown("**Explicit-period projections**")
    proj_disp = result.projections.copy()
    for col in proj_disp.columns:
        if col != "DiscountFactor":
            proj_disp[col] = proj_disp[col] / 1e9
    proj_disp.columns = [f"{c} ($B)" if c != "DiscountFactor" else c for c in proj_disp.columns]
    st.dataframe(proj_disp.style.format({c: "{:,.2f}" for c in proj_disp.columns}), use_container_width=True)

    st.markdown("**Terminal value**")
    st.info(
        f"FCFF in year {assumptions.projection_years} × (1 + g) / (WACC − g) = "
        f"**${result.terminal_value/1e9:,.2f}B** → PV **${(result.terminal_value / (1+wacc_used)**assumptions.projection_years)/1e9:,.2f}B**"
    )

    st.markdown("**Sensitivity (Fair Value / Share)**")
    grid = sensitivity_grid(
        base_revenue=base_rev,
        a=assumptions,
        wacc_base=wacc_used,
        shares_outstanding=shares,
        net_debt=net_debt,
    )
    st.plotly_chart(viz.sensitivity_heatmap(grid, snap.price), use_container_width=True)

    # Persist for visualization tab
    st.session_state["dcf_result"] = result


# ---------- Tab: Visualizations ----------
with tab_viz:
    st.subheader("Interactive visualizations")
    if fin.empty:
        st.warning("No data to visualize yet.")
    else:
        left, right = st.columns(2)
        with left:
            st.plotly_chart(viz.revenue_and_margins(fin), use_container_width=True)
            st.plotly_chart(viz.roic_vs_wacc(fin, st.session_state.get("wacc", 0.09)), use_container_width=True)
            st.plotly_chart(viz.leverage_and_liquidity(fin), use_container_width=True)
        with right:
            st.plotly_chart(viz.ebit_to_market_cap(fin, snap.market_cap), use_container_width=True)
            st.plotly_chart(viz.cash_flow_breakdown(fin), use_container_width=True)
            dcf_res = st.session_state.get("dcf_result")
            fair = dcf_res.fair_value_per_share if dcf_res else None
            st.plotly_chart(viz.price_history_chart(data["price_history"], fair), use_container_width=True)

        if dcf_res := st.session_state.get("dcf_result"):
            st.plotly_chart(viz.dcf_projection_chart(dcf_res.projections), use_container_width=True)


# ---------- Tab: Guide ----------
with tab_docs:
    st.subheader("User guide")
    st.markdown(
        """
### How the tool works

1. **Data retrieval.** Annual financials come from the SEC EDGAR *company-facts* API
   (XBRL US-GAAP tags). Price, beta, sector, and the risk-free rate come from
   Yahoo Finance via `yfinance`. No API keys or paid services are required.
2. **Historical window.** Use the sidebar slider to choose 3–10 years.
3. **Pre-filled assumptions.** Each assumption is blended from
   *(a)* a log-linear fit on your company's own history,
   *(b)* the sector benchmark for its Yahoo-tagged sector, and
   *(c)* live market data (10-year Treasury, yfinance beta).
   Every field is editable, and *Reset to ML defaults* rebuilds them.
4. **WACC.** Standard CAPM → WACC formula. All inputs are visible in the
   *Assumptions* tab; numbers recompute instantly.
5. **DCF.** 5-year explicit FCFF projection + Gordon-growth terminal value,
   discounted at WACC. Sensitivity grid shows fair-value/share across WACC ×
   terminal-g.
6. **Visualizations.** Plotly charts for revenue & margins, ROIC vs WACC,
   EBIT/Market-Cap, cash flow breakdown, leverage & liquidity, DCF projections,
   price history vs fair value, and a WACC × g heatmap.

### Free, reliable hosting

- **Streamlit Community Cloud** (free) — push this repo to GitHub, sign in at
  [streamlit.io/cloud](https://streamlit.io/cloud), select the repo, and it's live.
- **Local** — `pip install -r requirements.txt && streamlit run app.py`.

### Limitations & accuracy

- SEC XBRL tagging varies across issuers; some line items may be missing for
  smaller or non-US filers. The tool is configured for US-listed 10-K filers.
- Earnings-call transcripts are not distributed by the SEC in a machine-
  readable way. The *Data* tab deep-links to a free-search page for the
  requested ticker.
- DCFs are only as good as their assumptions — this tool makes every
  assumption explicit and editable, but it does not absolve you from thinking
  critically about each input.
"""
    )
    st.caption(f"Generated {datetime.utcnow().isoformat(timespec='seconds')}Z")
