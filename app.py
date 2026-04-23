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
from modules.rating import compute_rating

st.set_page_config(
    page_title="Financial Screener",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
html, body, [class*="css"] { font-family: 'Inter','Segoe UI',sans-serif; }

[data-testid="metric-container"] {
    background: #141a2a; border: 1px solid #1f2d45;
    border-radius: 12px; padding: 16px 20px;
}
[data-testid="metric-container"] label { color:#8b9ab5 !important; font-size:0.78rem; text-transform:uppercase; letter-spacing:0.06em; }
[data-testid="metric-container"] [data-testid="stMetricValue"] { color:#e6edf3 !important; font-size:1.55rem; font-weight:700; }
[data-testid="metric-container"] [data-testid="stMetricDelta"] { font-size:0.82rem; }

[data-baseweb="tab-list"] { border-bottom:1px solid #1f2d45 !important; gap:4px; }
[data-baseweb="tab"] { border-radius:8px 8px 0 0 !important; padding:10px 20px !important; color:#8b9ab5 !important; font-weight:500; }
[data-baseweb="tab"][aria-selected="true"] { color:#4f8cff !important; border-bottom:2px solid #4f8cff !important; background:transparent !important; }
[data-baseweb="tab"]:hover { color:#e6edf3 !important; background:#1a2236 !important; }

[data-testid="stSidebar"] { background:#0d1117 !important; border-right:1px solid #1f2d45; }

[data-testid="baseButton-primary"] { background:linear-gradient(135deg,#4f8cff,#3b72e0) !important; border:none !important; border-radius:8px !important; font-weight:600 !important; }
[data-testid="baseButton-primary"]:hover { background:linear-gradient(135deg,#6a9eff,#4f8cff) !important; }

[data-testid="stDataFrame"] { border:1px solid #1f2d45; border-radius:10px; overflow:hidden; }
[data-testid="stExpander"] { border:1px solid #1f2d45 !important; border-radius:10px !important; background:#141a2a !important; }
[data-testid="stNumberInput"] input { background:#0b0f19 !important; border:1px solid #1f2d45 !important; border-radius:6px !important; color:#e6edf3 !important; }
[data-testid="stAlert"] { border-radius:10px !important; border:none !important; }
hr { border-color:#1f2d45 !important; margin:8px 0 !important; }

.section-title { font-size:1.1rem;font-weight:700;color:#e6edf3;border-left:3px solid #4f8cff;padding-left:10px;margin:16px 0 8px 0; }

.company-header {
    display:flex; align-items:center; justify-content:space-between;
    background:linear-gradient(135deg,#0d1a2d 0%,#0b1929 100%);
    border:1px solid #1f2d45; border-radius:14px;
    padding:18px 24px; margin-bottom:16px; flex-wrap:wrap; gap:20px;
}
.company-name { font-size:1.8rem;font-weight:800;color:#e6edf3;margin:0; }
.company-sub { font-size:0.78rem;color:#5c7099;margin-top:2px; }

.fair-value-block {
    display:flex; gap:28px; align-items:center;
}
.fair-value-card {
    background:#0b1220; border:1px solid #1f2d45; border-radius:10px;
    padding:10px 16px; min-width:140px;
}
.fv-label { color:#8b9ab5; font-size:0.7rem; text-transform:uppercase; letter-spacing:0.08em; }
.fv-value { color:#e6edf3; font-size:1.4rem; font-weight:700; line-height:1.1; margin-top:2px; }
.fv-delta-pos { color:#2da44e; font-size:0.82rem; font-weight:600; margin-top:2px; }
.fv-delta-neg { color:#cf222e; font-size:0.82rem; font-weight:600; margin-top:2px; }
.fv-delta-neu { color:#8b9ab5; font-size:0.82rem; font-weight:600; margin-top:2px; }

.badge { display:inline-block; background:#1a2d4a; color:#4f8cff;
    border:1px solid #1f4080; border-radius:20px;
    padding:2px 10px; font-size:0.72rem; font-weight:600; margin-right:4px; margin-top:4px; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.markdown("## 📊 Financial Screener")
st.sidebar.caption("Free · SEC EDGAR + Yahoo Finance · No API key")
st.sidebar.markdown("---")
ticker = st.sidebar.text_input("🔍 Ticker symbol", value="AAPL").strip().upper()
years = st.sidebar.slider("Years of historical data", 3, 10, 5, 1,
    help="Annual 10-K periods to pull from SEC EDGAR.")
fetch = st.sidebar.button("⚡ Fetch data", type="primary", use_container_width=True)
if st.sidebar.button("🗑️ Clear cache + refresh", help="Forces a fresh pull from SEC EDGAR and Yahoo Finance, discarding cached data."):
    st.cache_data.clear()
    for k in list(st.session_state.keys()):
        del st.session_state[k]
    st.rerun()
st.sidebar.markdown("---")
st.sidebar.markdown("""
**How to use**
1. Enter a US-listed ticker
2. Click **Fetch data**
3. Review pre-filled **Assumptions** + set **Moat**
4. Check the **Rating** in the banner
5. Open **Visualizations** for charts
""")
st.sidebar.markdown("<small>Data: SEC EDGAR XBRL API · Yahoo Finance<br>100% free · no paid APIs</small>", unsafe_allow_html=True)

# ── Helpers ──────────────────────────────────────────────────────────────────
def _fmt(val, prefix="$", suffix="", div=1, decimals=2, na="—"):
    if val is None or (isinstance(val, float) and not np.isfinite(val)):
        return na
    return f"{prefix}{val/div:,.{decimals}f}{suffix}"

def _fmt_pct(val, decimals=2, na="—"):
    if val is None or (isinstance(val, float) and not np.isfinite(val)):
        return na
    return f"{val*100:.{decimals}f}%"

def pct_input(label, value_dec, lo_dec=0.0, hi_dec=1.0, step_pct=0.25, help=None, key=None):
    """number_input that displays as percent (34.85) but returns decimal (0.3485)."""
    v_pct = st.number_input(
        label, float(lo_dec*100), float(hi_dec*100), float(value_dec*100),
        step=float(step_pct), format="%.2f", help=help, key=key,
    )
    return v_pct / 100.0

# ── Data loading ──────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def load_company_data(sym: str, n: int) -> dict:
    info = edgar.resolve_ticker(sym)
    if not info:
        raise ValueError(f"'{sym}' not found in SEC ticker file. Check the symbol.")
    facts = edgar.fetch_company_facts(info["cik"])
    fin = facts.build_financials(n)
    income_stmt = edgar.extract_statement(facts, edgar.INCOME_STATEMENT_ROWS, n)
    balance_sheet = edgar.extract_statement(facts, edgar.BALANCE_SHEET_ROWS, n)
    cash_flow = edgar.extract_statement(facts, edgar.CASH_FLOW_ROWS, n)
    filings = edgar.fetch_recent_filings(info["cik"])
    snap = market_data.fetch_market_snapshot(sym)
    ph = market_data.fetch_price_history(sym, period=f"{min(n,10)}y")
    rf = market_data.fetch_risk_free_rate()
    return dict(
        cik=info["cik"], name=facts.name or info["title"],
        financials=fin, income_statement=income_stmt,
        balance_sheet=balance_sheet, cash_flow=cash_flow,
        filings=filings, transcripts_hint=edgar.fetch_transcripts_hint(sym),
        snapshot=snap, price_history=ph, risk_free_rate=rf,
    )

def _init_assumptions(data: dict) -> None:
    if "assumptions" in st.session_state and st.session_state.get("_ticker_cik") == data["cik"]:
        return
    snap = data["snapshot"]
    fin = data["financials"]
    a = derive_assumptions(fin, sector=snap.sector, risk_free_rate=data["risk_free_rate"], beta=snap.beta)
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
    st.session_state["assumptions"] = a
    st.session_state["_ticker_cik"] = data["cik"]

if fetch:
    for k in ("data", "assumptions", "_ticker_cik", "dcf_result", "wacc"):
        st.session_state.pop(k, None)

if "data" not in st.session_state:
    try:
        with st.spinner(f"Fetching **{ticker}** from SEC EDGAR + Yahoo Finance…"):
            st.session_state["data"] = load_company_data(ticker, years)
    except Exception as exc:
        st.error(f"**Could not load {ticker}:** {exc}")
        st.stop()

data = st.session_state["data"]
_init_assumptions(data)
assumptions: Assumptions = st.session_state["assumptions"]
fin: pd.DataFrame = data["financials"]
snap = data["snapshot"]

# ── Compute WACC + DCF eagerly so the banner always reflects live numbers ──
def _derive_shares_and_debt(fin, snap):
    shares = snap.shares_outstanding
    if not shares and "SharesOutstanding" in fin.index:
        sv = fin.loc["SharesOutstanding"].dropna()
        if not sv.empty:
            shares = float(sv.iloc[-1])
    if not shares and snap.market_cap and snap.price:
        shares = snap.market_cap / snap.price
    shares = float(shares) if shares else 0.0

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
    return shares, total_debt - cash

wacc_result = compute_wacc(assumptions)
st.session_state["wacc"] = wacc_result.wacc

dcf_result = None
if "Revenue" in fin.index and not fin.loc["Revenue"].dropna().empty:
    base_rev = float(fin.loc["Revenue"].dropna().iloc[-1])
    shares_auto, net_debt_auto = _derive_shares_and_debt(fin, snap)
    # Persist so DCF tab can override
    st.session_state.setdefault("shares_override", shares_auto)
    st.session_state.setdefault("net_debt_override", net_debt_auto)
    try:
        dcf_result = run_dcf(
            base_revenue=base_rev, a=assumptions, wacc=wacc_result.wacc,
            shares_outstanding=st.session_state["shares_override"],
            net_debt=st.session_state["net_debt_override"],
            current_price=snap.price,
        )
        st.session_state["dcf_result"] = dcf_result
    except Exception:
        dcf_result = None

# ── Rating based on upside + moat ─────────────────────────────────────────────
rating = compute_rating(
    upside_pct=dcf_result.upside_pct if dcf_result else None,
    moat_score=int(assumptions.moat_score),
)

# ── Persistent Company Header (Fair Value + Stars visible on every tab) ──────
badges_html = ""
for b in filter(None, [snap.sector, snap.industry]):
    badges_html += f'<span class="badge">{b}</span>'

fair_value_str = "—"
delta_html = ""
if dcf_result and np.isfinite(dcf_result.fair_value_per_share):
    fair_value_str = f"${dcf_result.fair_value_per_share:,.2f}"
    if dcf_result.upside_pct is not None:
        up_pct = dcf_result.upside_pct
        cls = "fv-delta-pos" if up_pct > 0.02 else ("fv-delta-neg" if up_pct < -0.02 else "fv-delta-neu")
        arrow = "▲" if up_pct > 0 else ("▼" if up_pct < 0 else "▶")
        delta_html = f'<div class="{cls}">{arrow} {up_pct*100:+.2f}% vs market</div>'

price_str = f"${snap.price:,.2f}" if snap.price else "—"

st.markdown(f"""
<div class="company-header">
  <div>
    <div class="company-name">📊 {data['name']} <span style="color:#4f8cff;font-size:1.25rem">({ticker})</span></div>
    <div class="company-sub">CIK {data['cik']} · Currency {snap.currency or 'USD'}</div>
    <div style="margin-top:6px">{badges_html}</div>
  </div>
  <div class="fair-value-block">
    <div class="fair-value-card">
      <div class="fv-label">Market Price</div>
      <div class="fv-value">{price_str}</div>
    </div>
    <div class="fair-value-card">
      <div class="fv-label">DCF Fair Value</div>
      <div class="fv-value">{fair_value_str}</div>
      {delta_html}
    </div>
    <div class="fair-value-card" style="min-width:260px">
      <div class="fv-label">Rating</div>
      {rating['html']}
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# Secondary KPI strip (compact, 2-decimal percentages)
latest_rev = None
if "Revenue" in fin.index:
    rv = fin.loc["Revenue"].dropna()
    if not rv.empty:
        latest_rev = float(rv.iloc[-1])

k1,k2,k3,k4,k5 = st.columns(5)
k1.metric("🏦 Market Cap",    _fmt(snap.market_cap, suffix="B", div=1e9, decimals=2))
k2.metric("📐 Beta",          f"{snap.beta:.2f}" if snap.beta else "—")
k3.metric("🏛️ 10Y Treasury",  _fmt_pct(data['risk_free_rate']))
k4.metric("⚖️ WACC",          _fmt_pct(wacc_result.wacc))
k5.metric("📈 Latest Revenue", _fmt(latest_rev, suffix="B", div=1e9, decimals=2))

st.markdown("---")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_data, tab_statements, tab_assume, tab_wacc, tab_dcf, tab_viz, tab_guide = st.tabs([
    "📂 Data", "📑 Financial Statements", "🎯 Assumptions", "⚖️ WACC",
    "💰 DCF", "📊 Visualizations", "📖 Guide"
])

# ───── helpers for statement tables (2-decimal $B display) ───────────────────
def _format_statement(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = pd.DataFrame(index=df.index, columns=[str(c) for c in df.columns], dtype=object)
    for idx in df.index:
        for col in df.columns:
            v = df.loc[idx, col]
            if pd.isna(v):
                out.loc[idx, str(col)] = "—"
            elif idx == "EPS":
                out.loc[idx, str(col)] = f"${v:,.2f}"
            elif abs(v) >= 1e6:
                out.loc[idx, str(col)] = f"${v/1e9:,.2f}B"
            else:
                out.loc[idx, str(col)] = f"{v:,.2f}"
    return out

# ────────────────────────── DATA TAB ─────────────────────────────────────────
with tab_data:
    st.markdown('<div class="section-title">Historical financials — SEC EDGAR (XBRL / US-GAAP)</div>', unsafe_allow_html=True)
    if fin.empty:
        st.warning("⚠️ No annual US-GAAP data returned. Only large US 10-K filers are fully supported.")
    else:
        st.dataframe(_format_statement(fin), use_container_width=True, height=min(620, 44 + 34*len(fin.index)))
        st.download_button("⬇️ Download CSV", fin.to_csv().encode(), f"{ticker}_financials.csv", "text/csv")

    st.markdown('<div class="section-title">Recent SEC filings</div>', unsafe_allow_html=True)
    filings_df = data["filings"]
    if isinstance(filings_df, pd.DataFrame) and not filings_df.empty:
        shown = filings_df.head(20).copy()
        shown["Filing"] = shown.apply(lambda r: f'<a href="{r["url"]}" target="_blank">{r["form"]} {r["filingDate"]}</a>', axis=1)
        st.write(shown[["form","filingDate","accessionNumber","Filing"]].to_html(escape=False, index=False), unsafe_allow_html=True)
    else:
        st.info("No recent filings found.")

    st.markdown('<div class="section-title">Earnings call transcripts</div>', unsafe_allow_html=True)
    st.markdown(f"SEC doesn't host transcripts. Free aggregators are your friend — [🔍 Search for {ticker} transcript]({data['transcripts_hint']})")

# ────────────────────────── FINANCIAL STATEMENTS TAB ─────────────────────────
with tab_statements:
    st.markdown('<div class="section-title">As-reported financial statements</div>', unsafe_allow_html=True)
    st.caption("Reconstructed from SEC XBRL (`us-gaap`) tags. Missing rows mean the issuer didn't tag that concept — not that it doesn't exist in the filing.")

    inc = data["income_statement"]
    bal = data["balance_sheet"]
    cfs = data["cash_flow"]

    # Income Statement
    st.markdown("### 📈 Income Statement")
    if inc.empty:
        st.info("No income-statement data available for this ticker.")
    else:
        st.dataframe(_format_statement(inc), use_container_width=True, height=min(620, 44 + 34*len(inc.index)))
        st.download_button("⬇️ Income Statement CSV", inc.to_csv().encode(), f"{ticker}_income_statement.csv", "text/csv", key="dl_is")

    # Balance Sheet
    st.markdown("### 🏦 Balance Sheet")
    if bal.empty:
        st.info("No balance-sheet data available for this ticker.")
    else:
        st.dataframe(_format_statement(bal), use_container_width=True, height=min(620, 44 + 34*len(bal.index)))
        st.download_button("⬇️ Balance Sheet CSV", bal.to_csv().encode(), f"{ticker}_balance_sheet.csv", "text/csv", key="dl_bs")

    # Cash Flow
    st.markdown("### 💵 Cash Flow Statement")
    if cfs.empty:
        st.info("No cash-flow data available for this ticker.")
    else:
        st.dataframe(_format_statement(cfs), use_container_width=True, height=min(620, 44 + 34*len(cfs.index)))
        st.download_button("⬇️ Cash Flow CSV", cfs.to_csv().encode(), f"{ticker}_cash_flow.csv", "text/csv", key="dl_cf")

    st.caption("💡 Values ≥ $1M are shown in billions (2 decimals). EPS shown per share.")

# ────────────────────────── ASSUMPTIONS TAB ──────────────────────────────────
with tab_assume:
    st.markdown('<div class="section-title">Assumptions — pre-filled by ML, fully editable</div>', unsafe_allow_html=True)
    st.caption("Percentages shown as percent values (e.g. 34.85 for 34.85%). Blended from historical trend + sector benchmark + live market.")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**📈 Growth & Margins**")
        assumptions.revenue_growth_rate  = pct_input("Revenue growth (yearly)",  assumptions.revenue_growth_rate, -0.20, 0.60, 0.50, help=ASSUMPTION_HELP["revenue_growth_rate"], key="rev_g")
        assumptions.terminal_growth_rate = pct_input("Terminal growth rate",     assumptions.terminal_growth_rate, 0.0,  0.05, 0.10, help=ASSUMPTION_HELP["terminal_growth_rate"], key="term_g")
        assumptions.operating_margin     = pct_input("Operating margin (EBIT/Rev)", assumptions.operating_margin, -0.20, 0.70, 0.50, help=ASSUMPTION_HELP["operating_margin"], key="op_m")
        assumptions.tax_rate             = pct_input("Effective tax rate",       assumptions.tax_rate,             0.0,  0.45, 0.50, help=ASSUMPTION_HELP["tax_rate"], key="tax_r")
    with col2:
        st.markdown("**🏗️ Reinvestment**")
        assumptions.capex_pct_revenue    = pct_input("CapEx / Revenue",  assumptions.capex_pct_revenue, 0.0, 0.40, 0.25, help=ASSUMPTION_HELP["capex_pct_revenue"], key="capex_r")
        assumptions.da_pct_revenue       = pct_input("D&A / Revenue",    assumptions.da_pct_revenue,    0.0, 0.30, 0.25, help=ASSUMPTION_HELP["da_pct_revenue"], key="da_r")
        assumptions.nwc_pct_revenue      = pct_input("ΔNWC / ΔRevenue",  assumptions.nwc_pct_revenue,   0.0, 0.30, 0.25, help=ASSUMPTION_HELP["nwc_pct_revenue"], key="nwc_r")
        assumptions.projection_years     = int(st.number_input("Projection years", 3, 15, int(assumptions.projection_years), 1, help=ASSUMPTION_HELP["projection_years"], key="proj_y"))
    with col3:
        st.markdown("**💹 Capital Costs**")
        assumptions.risk_free_rate       = pct_input("Risk-free rate (Rf)",    assumptions.risk_free_rate,       0.0,  0.15, 0.10, help=ASSUMPTION_HELP["risk_free_rate"], key="rf")
        assumptions.equity_risk_premium  = pct_input("Equity risk premium",    assumptions.equity_risk_premium,  0.02, 0.10, 0.10, help=ASSUMPTION_HELP["equity_risk_premium"], key="erp")
        assumptions.beta                 = st.number_input("Beta", 0.0, 3.0, float(assumptions.beta), 0.05, format="%.2f", help=ASSUMPTION_HELP["beta"], key="beta")
        assumptions.cost_of_debt_pretax  = pct_input("Cost of debt (pre-tax)", assumptions.cost_of_debt_pretax,  0.0, 0.20, 0.10, help=ASSUMPTION_HELP["cost_of_debt_pretax"], key="kd")
        assumptions.debt_weight          = pct_input("Debt weight D/(D+E)",    assumptions.debt_weight,          0.0, 0.90, 0.50, help=ASSUMPTION_HELP["debt_weight"], key="dw")

    st.markdown("---")
    st.markdown("**🏰 Qualitative Moat Score**")
    mc1, mc2 = st.columns([3,1])
    with mc1:
        assumptions.moat_score = int(st.slider(
            "Moat score (0 = no moat · 10 = Mag-7 / dominant franchise)",
            0, 10, int(assumptions.moat_score), 1,
            help=ASSUMPTION_HELP["moat_score"], key="moat",
        ))
    with mc2:
        st.markdown(f"<div style='text-align:center;padding-top:28px;font-size:1.6rem;font-weight:700;color:#4f8cff'>{assumptions.moat_score}/10</div>", unsafe_allow_html=True)

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

    w1,w2,w3,w4,w5 = st.columns(5)
    w1.metric("Cost of Equity Ke",      _fmt_pct(wacc_result.cost_of_equity))
    w2.metric("After-tax Cost of Debt", _fmt_pct(wacc_result.after_tax_cost_of_debt))
    w3.metric("Equity Weight We",       _fmt_pct(wacc_result.equity_weight))
    w4.metric("Debt Weight Wd",         _fmt_pct(wacc_result.debt_weight))
    w5.metric("**WACC**",              _fmt_pct(wacc_result.wacc))

    with st.expander("📐 Formula walk-through"):
        st.latex(r"K_e = R_f + \beta \times ERP")
        st.latex(r"WACC = W_e \cdot K_e + W_d \cdot K_d \cdot (1-t)")
        st.markdown(f"""
| Input | Value |
|---|---|
| Risk-free rate (Rf) | {_fmt_pct(assumptions.risk_free_rate)} |
| Beta (β) | {assumptions.beta:.2f} |
| Equity Risk Premium (ERP) | {_fmt_pct(assumptions.equity_risk_premium)} |
| **→ Cost of Equity Ke** | **{_fmt_pct(wacc_result.cost_of_equity)}** |
| Cost of Debt pre-tax (Kd) | {_fmt_pct(assumptions.cost_of_debt_pretax)} |
| Tax rate (t) | {_fmt_pct(assumptions.tax_rate)} |
| **→ After-tax Kd** | **{_fmt_pct(wacc_result.after_tax_cost_of_debt)}** |
| Equity weight (We) | {_fmt_pct(wacc_result.equity_weight)} |
| Debt weight (Wd) | {_fmt_pct(wacc_result.debt_weight)} |
| **→ WACC** | **{_fmt_pct(wacc_result.wacc)}** |
""")

# ────────────────────────── DCF TAB ──────────────────────────────────────────
with tab_dcf:
    st.markdown('<div class="section-title">Discounted Cash Flow Valuation</div>', unsafe_allow_html=True)

    if "Revenue" not in fin.index or fin.loc["Revenue"].dropna().empty:
        st.error("❌ No Revenue data found. DCF requires revenue. Check the Data tab.")
    else:
        base_rev = float(fin.loc["Revenue"].dropna().iloc[-1])
        shares_auto, net_debt_auto = _derive_shares_and_debt(fin, snap)

        ov1, ov2, ov3 = st.columns(3)
        shares = ov1.number_input("Shares outstanding", 0.0, value=float(st.session_state.get("shares_override", shares_auto)),
                                   step=1e6, format="%.0f", help="From yfinance / EDGAR — override if needed.", key="sh_in")
        net_debt = ov2.number_input("Net debt ($)", value=float(st.session_state.get("net_debt_override", net_debt_auto)),
                                     step=1e8, format="%.0f", help="Total debt – cash.", key="nd_in")
        wacc_used = pct_input("WACC override", wacc_result.wacc, 0.005, 0.40, 0.25, help="Pre-filled from WACC tab.", key="wacc_ov")

        st.session_state["shares_override"] = shares
        st.session_state["net_debt_override"] = net_debt

        result = run_dcf(base_rev, assumptions, wacc_used, shares, net_debt, current_price=snap.price)
        st.session_state["dcf_result"] = result

        d1,d2,d3,d4 = st.columns(4)
        d1.metric("Enterprise Value",   _fmt(result.enterprise_value, suffix="B", div=1e9, decimals=2))
        d2.metric("Equity Value",       _fmt(result.equity_value,     suffix="B", div=1e9, decimals=2))
        d3.metric("Fair Value / Share", f"${result.fair_value_per_share:,.2f}" if np.isfinite(result.fair_value_per_share) else "—")
        upside_str = _fmt_pct(result.upside_pct) if result.upside_pct is not None else "—"
        delta_str = f"${result.fair_value_per_share - snap.price:,.2f}" if (snap.price and np.isfinite(result.fair_value_per_share)) else None
        d4.metric("Upside vs market", upside_str, delta=delta_str)

        st.markdown('<div class="section-title">Explicit period projections</div>', unsafe_allow_html=True)
        p = result.projections.copy()
        pdisp = pd.DataFrame(index=p.index)
        for c in ["Revenue","EBIT","NOPAT","D&A","CapEx","FCFF","PV_FCFF"]:
            if c in p.columns:
                pdisp[f"{c} ($B)"] = (p[c]/1e9).map("{:,.2f}".format)
        if "DiscountFactor" in p.columns:
            pdisp["Discount Factor"] = p["DiscountFactor"].map("{:.4f}".format)
        st.dataframe(pdisp, use_container_width=True)

        pv_tv = result.terminal_value / (1 + wacc_used) ** assumptions.projection_years
        tv_pct = pv_tv / result.enterprise_value if result.enterprise_value else 0
        st.info(f"**Terminal Value:** ${result.terminal_value/1e9:,.2f}B → PV of TV: **${pv_tv/1e9:,.2f}B** ({tv_pct*100:.2f}% of Enterprise Value)")

        st.markdown('<div class="section-title">Sensitivity — Fair Value / Share</div>', unsafe_allow_html=True)
        grid = sensitivity_grid(base_rev, assumptions, wacc_used, shares, net_debt)
        st.plotly_chart(viz.sensitivity_heatmap(grid, snap.price), use_container_width=True)

# ────────────────────────── VISUALIZATIONS TAB ───────────────────────────────
with tab_viz:
    st.markdown('<div class="section-title">Interactive charts</div>', unsafe_allow_html=True)
    if fin.empty:
        st.warning("No data to chart — fetch a ticker first.")
    else:
        wacc_val = wacc_result.wacc
        dcf_res = st.session_state.get("dcf_result")
        fair_val = dcf_res.fair_value_per_share if dcf_res else None

        l, r = st.columns(2)
        with l:
            st.plotly_chart(viz.revenue_and_margins(fin), use_container_width=True)
            st.plotly_chart(viz.roic_vs_wacc(fin, wacc_val), use_container_width=True)
            st.plotly_chart(viz.leverage_and_liquidity(fin), use_container_width=True)
        with r:
            st.plotly_chart(viz.ebit_to_market_cap(fin, snap.market_cap), use_container_width=True)
            st.plotly_chart(viz.cash_flow_breakdown(fin), use_container_width=True)
            st.plotly_chart(viz.price_history_chart(data["price_history"], fair_val), use_container_width=True)
        if dcf_res:
            st.plotly_chart(viz.dcf_projection_chart(dcf_res.projections), use_container_width=True)

# ────────────────────────── GUIDE TAB ────────────────────────────────────────
with tab_guide:
    st.markdown('<div class="section-title">User guide</div>', unsafe_allow_html=True)
    st.markdown("""
### Rating system (header banner)
The star rating in the banner combines **DCF upside** with your **moat score** (0–10).

- **+3 to +5 (green)** — undervalued by the DCF → buy signal
- **−2 to +2 (yellow)** — signal close to neutral → wait / reassess
- **−3 to −5 (red)** — overvalued by the DCF → short signal

**Moat adjustment:** a high moat (≥8) pulls a "short" rating toward zero (the
DCF alone can underestimate premium franchises); a low moat (≤3) pulls a
"buy" rating toward zero (upside is less reliable without a durable edge).

### Percentage display
All ratios display as percentages with 2-decimal precision (e.g. `34.85%`).
Inputs accept the percent form too — type `34.85` to mean 34.85%.

### Financial Statements tab
Reconstructs Income Statement, Balance Sheet, and Cash Flow Statement from
SEC XBRL us-gaap tags as reported by the issuer. Values ≥ $1M are shown in
billions; EPS is shown per-share.

### Free hosting (Streamlit Community Cloud)
1. Push this repo to GitHub.
2. Sign in at [streamlit.io/cloud](https://streamlit.io/cloud) with GitHub.
3. **New app** → pick repo → main file = `app.py` → **Deploy**.
""")
