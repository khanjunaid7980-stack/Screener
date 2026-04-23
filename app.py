"""Financial Screening Tool — main Streamlit app."""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
import streamlit as st

from modules import edgar, market_data, visualizations as viz
from modules.assumptions import ASSUMPTION_HELP, Assumptions, derive_assumptions
from modules.dcf import run_dcf, sensitivity_grid
from modules.wacc import compute_wacc, infer_debt_weight_from_balance_sheet

st.set_page_config(
    page_title="Financial Screener",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Base */
html, body, [class*="css"] { font-family: 'Inter', 'Segoe UI', sans-serif; }

/* Metric cards */
[data-testid="metric-container"] {
    background: #141a2a;
    border: 1px solid #1f2d45;
    border-radius: 12px;
    padding: 16px 20px;
}
[data-testid="metric-container"] label { color: #8b9ab5 !important; font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.06em; }
[data-testid="metric-container"] [data-testid="stMetricValue"] { color: #e6edf3 !important; font-size: 1.55rem; font-weight: 700; }
[data-testid="metric-container"] [data-testid="stMetricDelta"] { font-size: 0.82rem; }

/* Tabs */
[data-baseweb="tab-list"] { border-bottom: 1px solid #1f2d45 !important; gap: 4px; }
[data-baseweb="tab"] { border-radius: 8px 8px 0 0 !important; padding: 10px 20px !important; color: #8b9ab5 !important; font-weight: 500; }
[data-baseweb="tab"][aria-selected="true"] { color: #4f8cff !important; border-bottom: 2px solid #4f8cff !important; background: transparent !important; }
[data-baseweb="tab"]:hover { color: #e6edf3 !important; background: #1a2236 !important; }

/* Sidebar */
[data-testid="stSidebar"] { background: #0d1117 !important; border-right: 1px solid #1f2d45; }
[data-testid="stSidebar"] .stMarkdown { color: #8b9ab5; }

/* Buttons */
[data-testid="baseButton-primary"] { background: linear-gradient(135deg, #4f8cff, #3b72e0) !important; border: none !important; border-radius: 8px !important; font-weight: 600 !important; letter-spacing: 0.02em; }
[data-testid="baseButton-primary"]:hover { background: linear-gradient(135deg, #6a9eff, #4f8cff) !important; }
[data-testid="baseButton-secondary"] { border: 1px solid #1f2d45 !important; border-radius: 8px !important; color: #8b9ab5 !important; }

/* DataFrame */
[data-testid="stDataFrame"] { border: 1px solid #1f2d45; border-radius: 10px; overflow: hidden; }

/* Expander */
[data-testid="stExpander"] { border: 1px solid #1f2d45 !important; border-radius: 10px !important; background: #141a2a !important; }

/* Number inputs */
[data-testid="stNumberInput"] input { background: #0b0f19 !important; border: 1px solid #1f2d45 !important; border-radius: 6px !important; color: #e6edf3 !important; }

/* Info / warning / success boxes */
[data-testid="stAlert"] { border-radius: 10px !important; border: none !important; }

/* Divider */
hr { border-color: #1f2d45 !important; margin: 8px 0 !important; }

/* Section headers */
.section-title {
    font-size: 1.1rem; font-weight: 700; color: #e6edf3;
    border-left: 3px solid #4f8cff; padding-left: 10px;
    margin: 16px 0 8px 0;
}

/* KPI banner */
.kpi-banner {
    background: linear-gradient(135deg, #0d1a2d 0%, #0b1929 100%);
    border: 1px solid #1f2d45;
    border-radius: 14px;
    padding: 18px 24px;
    margin-bottom: 16px;
}

/* Company header */
.company-header { margin-bottom: 4px; }
.company-name { font-size: 2rem; font-weight: 800; color: #e6edf3; }
.company-sub { font-size: 0.82rem; color: #5c7099; margin-top: 2px; }
.badge {
    display: inline-block;
    background: #1a2d4a; color: #4f8cff;
    border: 1px solid #1f4080; border-radius: 20px;
    padding: 2px 10px; font-size: 0.75rem; font-weight: 600;
    margin-right: 4px; margin-top: 4px;
}
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.markdown("## 📊 Financial Screener")
st.sidebar.caption("Free · SEC EDGAR + Yahoo Finance · No API key required")
st.sidebar.markdown("---")

ticker = st.sidebar.text_input("🔍 Ticker symbol", value="AAPL").strip().upper()
years  = st.sidebar.slider("Years of historical data", 3, 10, 5, 1,
    help="Number of annual 10-K periods pulled from SEC EDGAR.")
fetch  = st.sidebar.button("⚡ Fetch data", type="primary", use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.markdown("""
**How to use**
1. Enter a US-listed ticker
2. Click **Fetch data**
3. Review pre-filled **Assumptions**
4. See **WACC** and **DCF** results
5. Explore the **Visualizations** tab
""")
st.sidebar.markdown("---")
st.sidebar.markdown("<small>Data: SEC EDGAR XBRL API · Yahoo Finance<br>No paid APIs · 100% open-source</small>", unsafe_allow_html=True)

# ── Data loading ──────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def load_company_data(sym: str, n: int) -> dict:
    info = edgar.resolve_ticker(sym)
    if not info:
        raise ValueError(f"'{sym}' not found in SEC ticker file. Check the symbol.")
    facts    = edgar.fetch_company_facts(info["cik"])
    fin      = facts.build_financials(n)
    filings  = edgar.fetch_recent_filings(info["cik"])
    snap     = market_data.fetch_market_snapshot(sym)
    ph       = market_data.fetch_price_history(sym, period=f"{min(n,10)}y")
    rf       = market_data.fetch_risk_free_rate()
    return dict(
        cik=info["cik"], name=facts.name or info["title"],
        financials=fin, filings=filings,
        transcripts_hint=edgar.fetch_transcripts_hint(sym),
        snapshot=snap, price_history=ph, risk_free_rate=rf,
    )

def _init_assumptions(data: dict) -> None:
    if "assumptions" in st.session_state and st.session_state.get("_ticker_cik") == data["cik"]:
        return
    snap: market_data.MarketSnapshot = data["snapshot"]
    fin = data["financials"]
    a = derive_assumptions(fin, sector=snap.sector, risk_free_rate=data["risk_free_rate"], beta=snap.beta)
    # Anchor debt weight to live balance sheet + market cap where possible.
    if not fin.empty and snap.market_cap:
        td = 0.0
        for k in ("LongTermDebt", "ShortTermDebt"):
            if k in fin.index:
                v = fin.loc[k].dropna()
                if not v.empty:
                    td += float(v.iloc[-1])
        dw = infer_debt_weight_from_balance_sheet(td, snap.market_cap)
        if dw is not None:
            a.debt_weight = round(dw, 4)
    st.session_state["assumptions"]  = a
    st.session_state["_ticker_cik"]  = data["cik"]

if fetch:
    st.session_state.pop("data", None)
    st.session_state.pop("assumptions", None)
    st.session_state.pop("_ticker_cik", None)

if "data" not in st.session_state:
    try:
        with st.spinner(f"Fetching **{ticker}** from SEC EDGAR + Yahoo Finance…"):
            st.session_state["data"] = load_company_data(ticker, years)
    except Exception as exc:
        st.error(f"**Could not load {ticker}:** {exc}")
        st.stop()

data: dict = st.session_state["data"]
_init_assumptions(data)
assumptions: Assumptions = st.session_state["assumptions"]
fin: pd.DataFrame         = data["financials"]
snap: market_data.MarketSnapshot = data["snapshot"]

# ── Company header ────────────────────────────────────────────────────────────
def _fmt(val, prefix="$", suffix="", div=1, decimals=2, na="—"):
    if val is None or (isinstance(val, float) and not np.isfinite(val)):
        return na
    return f"{prefix}{val/div:,.{decimals}f}{suffix}"

badges = ""
for b in filter(None, [snap.sector, snap.industry]):
    badges += f'<span class="badge">{b}</span>'
st.markdown(f"""
<div class="company-header">
  <div class="company-name">📊 {data['name']} <span style="color:#4f8cff;font-size:1.3rem">({ticker})</span></div>
  <div class="company-sub">CIK {data['cik']} · Currency {snap.currency or 'USD'}</div>
  <div style="margin-top:6px">{badges}</div>
</div>
""", unsafe_allow_html=True)

latest_rev = None
if "Revenue" in fin.index:
    rv = fin.loc["Revenue"].dropna()
    if not rv.empty:
        latest_rev = float(rv.iloc[-1])

k1,k2,k3,k4,k5 = st.columns(5)
k1.metric("💵 Price",          _fmt(snap.price,           prefix="$", div=1,    decimals=2))
k2.metric("🏦 Market Cap",     _fmt(snap.market_cap,      prefix="$", suffix="B", div=1e9, decimals=1))
k3.metric("📐 Beta",           f"{snap.beta:.2f}" if snap.beta else "—")
k4.metric("🏛️ 10Y Treasury",   f"{data['risk_free_rate']:.2%}")
k5.metric("📈 Latest Revenue", _fmt(latest_rev,           prefix="$", suffix="B", div=1e9, decimals=1))

st.markdown("---")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_data, tab_assume, tab_wacc, tab_dcf, tab_viz, tab_guide = st.tabs([
    "📂 Data", "🎯 Assumptions", "⚖️ WACC", "💰 DCF Valuation", "📊 Visualizations", "📖 Guide"
])

# ────────────────────────── DATA TAB ─────────────────────────────────────────
with tab_data:
    st.markdown('<div class="section-title">Historical financials — SEC EDGAR (XBRL / US-GAAP)</div>', unsafe_allow_html=True)

    if fin.empty:
        st.warning("⚠️ No annual US-GAAP data returned for this ticker. Only large US 10-K filers are fully supported.")
    else:
        # Format in billions where values > 1M, raw otherwise
        disp = fin.copy()
        disp.columns = [str(c) for c in disp.columns]
        # Rows whose max absolute value > 1M → show in $B
        scale_rows = disp.index[disp.apply(lambda r: r.dropna().abs().max() if not r.dropna().empty else 0, axis=1) > 1e6]
        disp_fmt = disp.copy().astype(object)
        for idx in disp.index:
            for col in disp.columns:
                v = disp.loc[idx, col]
                if pd.isna(v):
                    disp_fmt.loc[idx, col] = "—"
                elif idx in scale_rows:
                    disp_fmt.loc[idx, col] = f"${v/1e9:,.2f}B"
                else:
                    disp_fmt.loc[idx, col] = f"{v:,.0f}"
        st.dataframe(disp_fmt, use_container_width=True, height=min(620, 44 + 34*len(disp.index)))
        csv_bytes = fin.to_csv().encode()
        st.download_button("⬇️ Download CSV", csv_bytes, f"{ticker}_financials.csv", "text/csv")

    st.markdown('<div class="section-title">Recent SEC filings</div>', unsafe_allow_html=True)
    filings_df = data["filings"]
    if isinstance(filings_df, pd.DataFrame) and not filings_df.empty:
        shown = filings_df.head(20).copy()
        shown["Filing"] = shown.apply(lambda r: f'<a href="{r["url"]}" target="_blank">{r["form"]} {r["filingDate"]}</a>', axis=1)
        st.write(shown[["form","filingDate","accessionNumber","Filing"]].to_html(escape=False, index=False), unsafe_allow_html=True)
    else:
        st.info("No recent filings found.")

    st.markdown('<div class="section-title">Earnings call transcripts</div>', unsafe_allow_html=True)
    st.markdown(
        f"SEC doesn't host transcripts. Free transcripts are available via IR pages and news aggregators — "
        f"[🔍 Search for {ticker} earnings transcript]({data['transcripts_hint']})"
    )

# ────────────────────────── ASSUMPTIONS TAB ──────────────────────────────────
with tab_assume:
    st.markdown('<div class="section-title">Assumptions — pre-filled by ML, fully editable</div>', unsafe_allow_html=True)
    st.caption("Blended from your company's historical trend + sector benchmark (Damodaran-style) + live market data. Every field is editable.")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**📈 Growth & Margins**")
        assumptions.revenue_growth_rate = st.number_input("Revenue growth (annual)", -0.20, 0.60, float(assumptions.revenue_growth_rate), 0.005, format="%.4f", help=ASSUMPTION_HELP["revenue_growth_rate"])
        assumptions.terminal_growth_rate = st.number_input("Terminal growth rate",   0.0,  0.05, float(assumptions.terminal_growth_rate), 0.0025, format="%.4f", help=ASSUMPTION_HELP["terminal_growth_rate"])
        assumptions.operating_margin     = st.number_input("Operating margin (EBIT/Rev)", -0.20, 0.70, float(assumptions.operating_margin), 0.005, format="%.4f", help=ASSUMPTION_HELP["operating_margin"])
        assumptions.tax_rate             = st.number_input("Effective tax rate",     0.0,  0.45, float(assumptions.tax_rate), 0.005, format="%.4f", help=ASSUMPTION_HELP["tax_rate"])
    with col2:
        st.markdown("**🏗️ Reinvestment**")
        assumptions.capex_pct_revenue    = st.number_input("CapEx / Revenue",        0.0, 0.40, float(assumptions.capex_pct_revenue),    0.005, format="%.4f", help=ASSUMPTION_HELP["capex_pct_revenue"])
        assumptions.da_pct_revenue       = st.number_input("D&A / Revenue",          0.0, 0.30, float(assumptions.da_pct_revenue),        0.005, format="%.4f", help=ASSUMPTION_HELP["da_pct_revenue"])
        assumptions.nwc_pct_revenue      = st.number_input("ΔNWC / ΔRevenue",        0.0, 0.30, float(assumptions.nwc_pct_revenue),       0.005, format="%.4f", help=ASSUMPTION_HELP["nwc_pct_revenue"])
        assumptions.projection_years     = int(st.number_input("Projection years",    3, 15, int(assumptions.projection_years), 1, help=ASSUMPTION_HELP["projection_years"]))
    with col3:
        st.markdown("**💹 Capital Costs**")
        assumptions.risk_free_rate       = st.number_input("Risk-free rate (Rf)",    0.0,  0.15, float(assumptions.risk_free_rate),        0.0025, format="%.4f", help=ASSUMPTION_HELP["risk_free_rate"])
        assumptions.equity_risk_premium  = st.number_input("Equity risk premium",    0.02, 0.10, float(assumptions.equity_risk_premium),   0.0025, format="%.4f", help=ASSUMPTION_HELP["equity_risk_premium"])
        assumptions.beta                 = st.number_input("Beta",                   0.0,  3.0,  float(assumptions.beta),                 0.05,   format="%.3f",  help=ASSUMPTION_HELP["beta"])
        assumptions.cost_of_debt_pretax  = st.number_input("Cost of debt (pre-tax)", 0.0,  0.20, float(assumptions.cost_of_debt_pretax),   0.0025, format="%.4f", help=ASSUMPTION_HELP["cost_of_debt_pretax"])
        assumptions.debt_weight          = st.number_input("Debt weight D/(D+E)",    0.0,  0.90, float(assumptions.debt_weight),           0.01,   format="%.4f", help=ASSUMPTION_HELP["debt_weight"])

    st.session_state["assumptions"] = assumptions

    st.markdown("---")
    rc, ec, ic = st.columns([1,1,2])
    if rc.button("🔄 Reset to ML defaults"):
        st.session_state["assumptions"] = derive_assumptions(fin, sector=snap.sector, risk_free_rate=data["risk_free_rate"], beta=snap.beta)
        st.rerun()
    ec.download_button("⬇️ Export JSON", json.dumps(assumptions.to_dict(), indent=2).encode(), f"{ticker}_assumptions.json", "application/json")
    up = ic.file_uploader("⬆️ Import assumptions JSON", type=["json"], label_visibility="collapsed")
    if up:
        try:
            blob = json.load(up)
            st.session_state["assumptions"] = Assumptions(**{**assumptions.to_dict(), **blob})
            st.success("✅ Assumptions imported.")
            st.rerun()
        except Exception as exc:
            st.error(f"Invalid JSON: {exc}")

# ────────────────────────── WACC TAB ─────────────────────────────────────────
with tab_wacc:
    st.markdown('<div class="section-title">Weighted Average Cost of Capital</div>', unsafe_allow_html=True)
    wacc_result = compute_wacc(assumptions)
    st.session_state["wacc"] = wacc_result.wacc

    w1,w2,w3,w4,w5 = st.columns(5)
    w1.metric("Cost of Equity Ke",       f"{wacc_result.cost_of_equity:.2%}")
    w2.metric("After-tax Cost of Debt",  f"{wacc_result.after_tax_cost_of_debt:.2%}")
    w3.metric("Equity Weight We",        f"{wacc_result.equity_weight:.1%}")
    w4.metric("Debt Weight Wd",          f"{wacc_result.debt_weight:.1%}")
    w5.metric("**WACC**",               f"{wacc_result.wacc:.2%}")

    with st.expander("📐 Formula walk-through"):
        st.latex(r"K_e = R_f + \beta \times ERP")
        st.latex(r"WACC = W_e \cdot K_e + W_d \cdot K_d \cdot (1 - t)")
        st.markdown(f"""
| Input | Value |
|---|---|
| Risk-free rate (Rf) | {assumptions.risk_free_rate:.2%} |
| Beta (β) | {assumptions.beta:.3f} |
| Equity Risk Premium (ERP) | {assumptions.equity_risk_premium:.2%} |
| **→ Cost of Equity Ke** | **{wacc_result.cost_of_equity:.2%}** |
| Cost of Debt pre-tax (Kd) | {assumptions.cost_of_debt_pretax:.2%} |
| Tax rate (t) | {assumptions.tax_rate:.2%} |
| **→ After-tax Kd** | **{wacc_result.after_tax_cost_of_debt:.2%}** |
| Equity weight (We) | {wacc_result.equity_weight:.1%} |
| Debt weight (Wd) | {wacc_result.debt_weight:.1%} |
| **→ WACC** | **{wacc_result.wacc:.2%}** |
""")

# ────────────────────────── DCF TAB ──────────────────────────────────────────
with tab_dcf:
    st.markdown('<div class="section-title">Discounted Cash Flow Valuation</div>', unsafe_allow_html=True)

    if "Revenue" not in fin.index or fin.loc["Revenue"].dropna().empty:
        st.error("❌ No Revenue data found for this ticker. DCF requires revenue. Check the Data tab.")
    else:
        base_rev = float(fin.loc["Revenue"].dropna().iloc[-1])

        # Shares outstanding — cascade through several sources.
        shares = snap.shares_outstanding
        if not shares and "SharesOutstanding" in fin.index:
            sv = fin.loc["SharesOutstanding"].dropna()
            if not sv.empty:
                shares = float(sv.iloc[-1])
        if not shares and snap.market_cap and snap.price:
            shares = snap.market_cap / snap.price
        shares = float(shares) if shares else 0.0

        # Net debt from latest balance sheet.
        total_debt = 0.0
        for k in ("LongTermDebt", "ShortTermDebt"):
            if k in fin.index:
                v = fin.loc[k].dropna()
                if not v.empty:
                    total_debt += float(v.iloc[-1])
        cash = 0.0
        if "CashAndEquivalents" in fin.index:
            cv = fin.loc["CashAndEquivalents"].dropna()
            if not cv.empty:
                cash = float(cv.iloc[-1])
        net_debt = total_debt - cash

        wacc_used = float(st.session_state.get("wacc", compute_wacc(assumptions).wacc))

        ov1, ov2, ov3 = st.columns(3)
        shares   = ov1.number_input("Shares outstanding",  0.0, value=shares,   step=1e6,  format="%.0f", help="From yfinance or EDGAR; override if needed.")
        net_debt = ov2.number_input("Net debt ($)",        value=net_debt, step=1e8, format="%.0f", help="Total interest-bearing debt minus cash.")
        wacc_used = ov3.number_input("WACC",               0.005, 0.40, wacc_used, 0.0025,  format="%.4f", help="Pre-filled from WACC tab; override freely.")

        result = run_dcf(base_rev, assumptions, wacc_used, shares, net_debt, current_price=snap.price)

        d1,d2,d3,d4 = st.columns(4)
        d1.metric("Enterprise Value",   _fmt(result.enterprise_value,   prefix="$", suffix="B", div=1e9, decimals=2))
        d2.metric("Equity Value",       _fmt(result.equity_value,       prefix="$", suffix="B", div=1e9, decimals=2))
        d3.metric("Fair Value / Share", f"${result.fair_value_per_share:,.2f}" if np.isfinite(result.fair_value_per_share) else "—")
        upside_str = f"{result.upside_pct:.1%}" if result.upside_pct is not None else "—"
        delta_str  = f"${result.fair_value_per_share - snap.price:,.2f}" if (snap.price and np.isfinite(result.fair_value_per_share)) else None
        d4.metric("Upside vs market", upside_str, delta=delta_str)

        st.markdown('<div class="section-title">Explicit period projections</div>', unsafe_allow_html=True)
        p = result.projections.copy()
        p_disp = pd.DataFrame(index=p.index)
        for c in ["Revenue","EBIT","NOPAT","D&A","CapEx","FCFF","PV_FCFF"]:
            if c in p.columns:
                p_disp[f"{c} ($B)"] = (p[c]/1e9).map("{:,.2f}".format)
        if "DiscountFactor" in p.columns:
            p_disp["Discount Factor"] = p["DiscountFactor"].map("{:.4f}".format)
        st.dataframe(p_disp, use_container_width=True)

        pv_tv = result.terminal_value / (1 + wacc_used) ** assumptions.projection_years
        st.info(f"**Terminal Value:** ${result.terminal_value/1e9:,.2f}B  →  PV of TV: **${pv_tv/1e9:,.2f}B** ({pv_tv/result.enterprise_value:.0%} of Enterprise Value)")

        st.markdown('<div class="section-title">Sensitivity — Fair Value / Share</div>', unsafe_allow_html=True)
        grid = sensitivity_grid(base_rev, assumptions, wacc_used, shares, net_debt)
        st.plotly_chart(viz.sensitivity_heatmap(grid, snap.price), use_container_width=True)

        st.session_state["dcf_result"] = result

# ────────────────────────── VISUALIZATIONS TAB ───────────────────────────────
with tab_viz:
    st.markdown('<div class="section-title">Interactive charts</div>', unsafe_allow_html=True)
    if fin.empty:
        st.warning("No data to chart yet — fetch a ticker first.")
    else:
        wacc_val = st.session_state.get("wacc", 0.09)
        dcf_res  = st.session_state.get("dcf_result")
        fair_val = dcf_res.fair_value_per_share if dcf_res else None

        l, r = st.columns(2)
        with l:
            st.plotly_chart(viz.revenue_and_margins(fin),                      use_container_width=True)
            st.plotly_chart(viz.roic_vs_wacc(fin, wacc_val),                   use_container_width=True)
            st.plotly_chart(viz.leverage_and_liquidity(fin),                   use_container_width=True)
        with r:
            st.plotly_chart(viz.ebit_to_market_cap(fin, snap.market_cap),      use_container_width=True)
            st.plotly_chart(viz.cash_flow_breakdown(fin),                      use_container_width=True)
            st.plotly_chart(viz.price_history_chart(data["price_history"], fair_val), use_container_width=True)

        if dcf_res:
            st.plotly_chart(viz.dcf_projection_chart(dcf_res.projections),     use_container_width=True)

# ────────────────────────── GUIDE TAB ────────────────────────────────────────
with tab_guide:
    st.markdown('<div class="section-title">User guide</div>', unsafe_allow_html=True)
    st.markdown("""
### How it works

| Step | What happens |
|---|---|
| **Fetch data** | Annual financials pulled from SEC EDGAR XBRL company-facts API. Price, beta, sector from Yahoo Finance. No API keys. |
| **Assumptions** | Every input is pre-filled: historical log-linear trend + sector benchmark + live market data. Fully editable, exportable as JSON. |
| **WACC** | CAPM Ke + after-tax Kd, weighted by D/(D+E) inferred from balance sheet. |
| **DCF** | 5-year explicit FCFF projection + Gordon-growth terminal value, discounted at WACC. |
| **Sensitivity** | Heatmap of fair-value/share across ±200bps WACC × ±100bps terminal growth. |
| **Charts** | Revenue/margins, ROIC vs WACC, EBIT/MCap, cash-flow bridge, leverage, DCF projections, price vs fair value. |

### Free hosting (Streamlit Community Cloud)
1. Push this repo to GitHub (already done).
2. Sign in at [streamlit.io/cloud](https://streamlit.io/cloud) with GitHub.
3. **New app** → pick this repo → main file = `app.py` → **Deploy**.
4. You get a permanent free URL accessible from any device — no server, no credit card.

### Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

### Caveats
- Revenue tags vary across issuers. The tool merges all candidate XBRL tags per fiscal year, preferring the most recently filed, to handle ASC 606 tag transitions (e.g. Apple).
- Non-US or very small filers may have incomplete XBRL tagging.
- A DCF is only as reliable as its inputs — treat output as a directional estimate, not a verdict.
""")
