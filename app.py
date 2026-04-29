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
@import url('https://rsms.me/inter/inter.css');
html, body, [class*="css"] { font-family: 'Inter','SF Pro Display','Segoe UI',sans-serif; }
* { -webkit-font-smoothing: antialiased; }

/* App background gradient for depth */
[data-testid="stAppViewContainer"] {
    background: radial-gradient(ellipse at top, #0e1525 0%, #0b0f19 70%) !important;
}
.block-container { padding-top: 2rem !important; padding-bottom: 4rem !important; max-width: 1400px; }

/* Metric cards */
[data-testid="metric-container"] {
    background: linear-gradient(180deg, #161e30 0%, #131a2a 100%);
    border: 1px solid #1f2d45;
    border-radius: 14px; padding: 18px 20px;
    box-shadow: 0 1px 0 rgba(255,255,255,0.03) inset, 0 8px 24px rgba(0,0,0,0.25);
    transition: transform .15s ease, border-color .15s ease;
}
[data-testid="metric-container"]:hover { border-color:#2a3a5a; transform: translateY(-1px); }
[data-testid="metric-container"] label { color:#8b9ab5 !important; font-size:0.74rem !important; text-transform:uppercase; letter-spacing:0.08em; font-weight:600 !important; }
[data-testid="metric-container"] [data-testid="stMetricValue"] { color:#f1f5f9 !important; font-size:1.65rem !important; font-weight:700 !important; letter-spacing:-0.01em; }
[data-testid="metric-container"] [data-testid="stMetricDelta"] { font-size:0.8rem !important; font-weight:600 !important; }

/* Tabs */
[data-baseweb="tab-list"] { border-bottom:1px solid #1f2d45 !important; gap:2px; background:transparent !important; }
[data-baseweb="tab"] { border-radius:8px 8px 0 0 !important; padding:11px 18px !important; color:#7a8aa6 !important; font-weight:500 !important; font-size:0.92rem !important; transition: all .15s ease; }
[data-baseweb="tab"][aria-selected="true"] { color:#4f8cff !important; border-bottom:2px solid #4f8cff !important; background:linear-gradient(180deg,transparent,#4f8cff10) !important; font-weight:600 !important; }
[data-baseweb="tab"]:hover { color:#e6edf3 !important; background:#1a2236 !important; }

/* Sidebar */
[data-testid="stSidebar"] { background: linear-gradient(180deg,#0a0e17 0%,#0d1117 100%) !important; border-right:1px solid #1f2d45; }
[data-testid="stSidebar"] h2 { color:#f1f5f9 !important; font-weight:700 !important; }

/* Buttons */
[data-testid="baseButton-primary"] {
    background: linear-gradient(135deg,#4f8cff 0%,#3b72e0 100%) !important;
    border: none !important; border-radius: 10px !important; font-weight: 600 !important;
    box-shadow: 0 4px 14px rgba(79,140,255,0.35) !important;
    transition: all .15s ease !important;
}
[data-testid="baseButton-primary"]:hover { transform: translateY(-1px); box-shadow: 0 6px 20px rgba(79,140,255,0.5) !important; }
[data-testid="baseButton-secondary"] { background:#161e30 !important; border:1px solid #2a3a5a !important; border-radius:10px !important; color:#cbd5e1 !important; }
[data-testid="baseButton-secondary"]:hover { background:#1a2236 !important; border-color:#4f8cff !important; }

/* DataFrame */
[data-testid="stDataFrame"] { border:1px solid #1f2d45; border-radius:12px; overflow:hidden; box-shadow: 0 4px 16px rgba(0,0,0,0.2); }

/* Expander */
[data-testid="stExpander"] { border:1px solid #1f2d45 !important; border-radius:12px !important; background:#131a2a !important; }
[data-testid="stExpander"] summary { padding: 12px 16px !important; font-weight:600 !important; color:#cbd5e1 !important; }

/* Inputs */
[data-testid="stNumberInput"] input,
[data-testid="stTextInput"] input {
    background:#0a0f1c !important; border:1px solid #1f2d45 !important;
    border-radius:8px !important; color:#f1f5f9 !important; font-weight:500 !important;
}
[data-testid="stNumberInput"] input:focus,
[data-testid="stTextInput"] input:focus { border-color:#4f8cff !important; box-shadow: 0 0 0 2px rgba(79,140,255,0.18) !important; }
[data-testid="stNumberInput"] label,
[data-testid="stTextInput"] label,
[data-testid="stSlider"] label,
[data-testid="stFileUploader"] label { color:#cbd5e1 !important; font-weight:500 !important; font-size:0.85rem !important; }

/* Slider */
[data-testid="stSlider"] [role="slider"] { background:#4f8cff !important; }

/* Alerts */
[data-testid="stAlert"] { border-radius:12px !important; border:1px solid !important; }
[data-testid="stAlert"][kind="error"] { background:#3a0d12 !important; border-color:#ef444455 !important; }
[data-testid="stAlert"][kind="warning"] { background:#3a2a0d !important; border-color:#eab30855 !important; }
[data-testid="stAlert"][kind="info"] { background:#0d2a3a !important; border-color:#4f8cff55 !important; }
[data-testid="stAlert"][kind="success"] { background:#0d3a1a !important; border-color:#22c55e55 !important; }

hr { border-color:#1f2d45 !important; margin:14px 0 !important; }

.section-title {
    font-size:0.95rem; font-weight:700; color:#cbd5e1;
    border-left:3px solid #4f8cff; padding-left:12px;
    margin:24px 0 10px 0; text-transform:uppercase; letter-spacing:0.06em;
}

/* Company header banner */
.company-header {
    display:flex; align-items:center; justify-content:space-between;
    background: linear-gradient(135deg, #0e1a2e 0%, #0b1426 50%, #0d1928 100%);
    border:1px solid #1f2d45; border-radius:16px;
    padding:22px 28px; margin-bottom:18px; flex-wrap:wrap; gap:24px;
    box-shadow: 0 8px 32px rgba(0,0,0,0.3), 0 1px 0 rgba(255,255,255,0.04) inset;
}
.company-name { font-size:1.85rem; font-weight:800; color:#f1f5f9; margin:0; letter-spacing:-0.02em; }
.company-sub { font-size:0.78rem; color:#5c7099; margin-top:4px; font-weight:500; }
.ticker-badge {
    color:#4f8cff; font-size:1.15rem; font-weight:700; background:#4f8cff15;
    padding:2px 10px; border-radius:8px; border:1px solid #4f8cff33; margin-left:8px;
}

.fair-value-block { display:flex; gap:14px; align-items:stretch; flex-wrap:wrap; }
.fair-value-card {
    background: linear-gradient(180deg, #0c1322 0%, #0a1020 100%);
    border:1px solid #1f2d45; border-radius:12px;
    padding:12px 18px; min-width:150px;
    box-shadow: 0 1px 0 rgba(255,255,255,0.03) inset;
}
.fv-label { color:#7a8aa6; font-size:0.68rem; text-transform:uppercase; letter-spacing:0.1em; font-weight:600; }
.fv-value { color:#f1f5f9; font-size:1.5rem; font-weight:700; line-height:1.15; margin-top:4px; letter-spacing:-0.01em; }
.fv-delta-pos { color:#22c55e; font-size:0.82rem; font-weight:700; margin-top:4px; }
.fv-delta-neg { color:#ef4444; font-size:0.82rem; font-weight:700; margin-top:4px; }
.fv-delta-neu { color:#8b9ab5; font-size:0.82rem; font-weight:700; margin-top:4px; }

.rating-card {
    background: linear-gradient(180deg, #0c1322 0%, #0a1020 100%);
    border:1px solid #1f2d45; border-radius:12px;
    padding:12px 18px; min-width:300px;
    box-shadow: 0 1px 0 rgba(255,255,255,0.03) inset;
}

.badge {
    display:inline-block; background:#1a2d4a; color:#7eb0ff;
    border:1px solid #2a4a80; border-radius:20px;
    padding:3px 12px; font-size:0.72rem; font-weight:600; margin-right:5px; margin-top:6px;
    letter-spacing:0.02em;
}

/* Markdown body text */
.stMarkdown p { color:#cbd5e1; }
.stMarkdown a { color:#4f8cff; text-decoration:none; }
.stMarkdown a:hover { color:#7eb0ff; text-decoration:underline; }
.stMarkdown code { background:#1a2236; color:#7eb0ff; padding:2px 6px; border-radius:4px; font-size:0.85em; }
.stMarkdown table { border-collapse:collapse; }
.stMarkdown table th { background:#161e30; color:#cbd5e1; border:1px solid #1f2d45; padding:8px 12px; }
.stMarkdown table td { border:1px solid #1f2d45; padding:8px 12px; color:#e6edf3; }
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

def _fmt_int(val, na="—"):
    """Format a large integer with thousands separators (e.g. 15,408,272,000)."""
    if val is None or (isinstance(val, float) and not np.isfinite(val)):
        return na
    return f"{int(round(float(val))):,}"

def _fmt_pct(val, decimals=2, na="—"):
    if val is None or (isinstance(val, float) and not np.isfinite(val)):
        return na
    return f"{val*100:.{decimals}f}%"

def _fmt_pct_signed(val, decimals=2, na="—"):
    """Signed percent with explicit + or − for unambiguous direction."""
    if val is None or (isinstance(val, float) and not np.isfinite(val)):
        return na
    return f"{val*100:+.{decimals}f}%"

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
        up = dcf_result.upside_pct
        if up > 0.02:
            cls, arrow = "fv-delta-pos", "▲"
        elif up < -0.02:
            cls, arrow = "fv-delta-neg", "▼"
        else:
            cls, arrow = "fv-delta-neu", "•"
        delta_html = f'<div class="{cls}">{arrow} {up*100:+.2f}% vs market</div>'

price_str = f"${snap.price:,.2f}" if snap.price else "—"

st.markdown(f"""
<div class="company-header">
  <div style="flex:1;min-width:260px;">
    <div class="company-name">{data['name']} <span class="ticker-badge">{ticker}</span></div>
    <div class="company-sub">CIK {data['cik']} · {snap.currency or 'USD'} · 10-K Annual Filings</div>
    <div>{badges_html}</div>
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
    <div class="rating-card">
      <div class="fv-label">Investment Rating</div>
      <div style="margin-top:6px;">{rating['html']}</div>
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

upside_pct = dcf_result.upside_pct if dcf_result and dcf_result.upside_pct is not None else None
upside_delta = None
if upside_pct is not None:
    upside_delta = f"{upside_pct*100:+.2f}%"

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Market Cap",    _fmt(snap.market_cap, suffix="B", div=1e9, decimals=2))
k2.metric("Beta",          f"{snap.beta:.2f}" if snap.beta else "—")
k3.metric("Risk-free (10Y)", _fmt_pct(data['risk_free_rate']))
k4.metric("WACC",          _fmt_pct(wacc_result.wacc))
k5.metric("Upside vs market", _fmt_pct_signed(upside_pct), delta=upside_delta)


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
        shown = shown.rename(columns={
            "form": "Form", "filingDate": "Filed", "accessionNumber": "Accession",
        })
        st.dataframe(
            shown[["Form", "Filed", "Accession", "url"]],
            use_container_width=True, hide_index=True,
            column_config={
                "url": st.column_config.LinkColumn("Document", display_text="Open ↗"),
                "Form": st.column_config.TextColumn(width="small"),
                "Filed": st.column_config.TextColumn(width="small"),
            },
        )
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
        assumptions.projection_years     = int(st.slider("Projection years", 3, 10, int(min(10, max(3, assumptions.projection_years))), 1, help=ASSUMPTION_HELP["projection_years"], key="proj_y"))
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
            "Moat strength (0 = none · 10 = dominant Mag-7 franchise)",
            0, 10, int(assumptions.moat_score), 1,
            help=ASSUMPTION_HELP["moat_score"], key="moat",
        ))
    with mc2:
        # Color-graded moat chip
        m = assumptions.moat_score
        if m >= 8:    moat_color, moat_label = "#22c55e", "Wide"
        elif m >= 6:  moat_color, moat_label = "#4f8cff", "Strong"
        elif m >= 4:  moat_color, moat_label = "#eab308", "Modest"
        else:         moat_color, moat_label = "#ef4444", "Narrow"
        st.markdown(
            f"<div style='text-align:center;margin-top:24px;padding:12px;"
            f"background:{moat_color}1a;border:1px solid {moat_color}55;"
            f"border-radius:10px;'>"
            f"<div style='font-size:0.65rem;color:#8b9ab5;text-transform:uppercase;letter-spacing:0.08em;font-weight:600;'>Moat</div>"
            f"<div style='font-size:1.4rem;font-weight:700;color:{moat_color};line-height:1.2;'>{m} / 10</div>"
            f"<div style='font-size:0.7rem;color:{moat_color};font-weight:600;'>{moat_label}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

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
        with ov1:
            shares = st.number_input(
                "Shares outstanding", min_value=0.0,
                value=float(st.session_state.get("shares_override", shares_auto)),
                step=1e6, format="%.0f",
                help="From yfinance / EDGAR — override if needed.", key="sh_in",
            )
            st.caption(f"≈ **{_fmt_int(shares)}** shares")
        with ov2:
            net_debt = st.number_input(
                "Net debt", value=float(st.session_state.get("net_debt_override", net_debt_auto)),
                step=1e8, format="%.0f",
                help="Total debt – cash. Negative = net cash.", key="nd_in",
            )
            st.caption(f"≈ **${_fmt_int(net_debt)}**" + ("  (net cash)" if net_debt < 0 else ""))
        with ov3:
            wacc_used = pct_input("WACC override", wacc_result.wacc, 0.005, 0.40, 0.25, help="Pre-filled from WACC tab.", key="wacc_ov")
            st.caption(f"≈ **{_fmt_pct(wacc_used)}**")

        st.session_state["shares_override"] = shares
        st.session_state["net_debt_override"] = net_debt

        result = run_dcf(base_rev, assumptions, wacc_used, shares, net_debt, current_price=snap.price)
        st.session_state["dcf_result"] = result

        d1,d2,d3,d4 = st.columns(4)
        d1.metric("Enterprise Value",   _fmt(result.enterprise_value, suffix="B", div=1e9, decimals=2))
        d2.metric("Equity Value",       _fmt(result.equity_value,     suffix="B", div=1e9, decimals=2))
        d3.metric("Fair Value / Share", f"${result.fair_value_per_share:,.2f}" if np.isfinite(result.fair_value_per_share) else "—")
        if result.upside_pct is None:
            d4.metric("Upside vs market", "—")
        else:
            up_p = result.upside_pct
            up_label = _fmt_pct_signed(up_p)
            delta_str = f"${result.fair_value_per_share - (snap.price or 0):+,.2f}" if (snap.price and np.isfinite(result.fair_value_per_share)) else None
            # delta_color="inverse" makes Streamlit render red for positive deltas → we want
            # green for positive upside (long), red for negative (short). "normal" does that.
            d4.metric("Upside vs market", up_label, delta=delta_str, delta_color="normal")

        st.markdown('<div class="section-title">Explicit-period projections</div>', unsafe_allow_html=True)
        p = result.projections.copy()
        # Display in transposed form (years across columns) so it reads like a real model.
        rows_to_show = [r for r in ("Revenue","EBIT","NOPAT","D&A","CapEx","ChangeInNWC","FCFF","DiscountFactor","PV_FCFF") if r in p.columns]
        proj_t = p[rows_to_show].T
        proj_t.columns = [f"Year {y}" for y in proj_t.columns]
        # Format each row appropriately
        formatted = pd.DataFrame(index=proj_t.index, columns=proj_t.columns, dtype=object)
        for r in proj_t.index:
            for c in proj_t.columns:
                v = proj_t.loc[r, c]
                if pd.isna(v):
                    formatted.loc[r, c] = "—"
                elif r == "DiscountFactor":
                    formatted.loc[r, c] = f"{v:.4f}"
                else:
                    formatted.loc[r, c] = f"${v/1e9:,.2f}B"
        st.dataframe(formatted, use_container_width=True, height=44 + 36*len(formatted.index))

        if wacc_used <= assumptions.terminal_growth_rate:
            st.error(
                f"⚠️ WACC ({_fmt_pct(wacc_used)}) ≤ terminal growth ({_fmt_pct(assumptions.terminal_growth_rate)}). "
                "Terminal value is mathematically undefined. Lower the terminal growth or raise WACC."
            )

        pv_tv = result.terminal_value / (1 + wacc_used) ** assumptions.projection_years
        tv_pct = (pv_tv / result.enterprise_value) if result.enterprise_value else 0
        explicit_pct = 1 - tv_pct
        st.markdown(f"""
<div style="background:linear-gradient(135deg,#0d1a2d,#0b1425);border:1px solid #1f2d45;
            border-radius:12px;padding:16px 20px;margin-top:12px;">
  <div style="display:flex;gap:32px;flex-wrap:wrap;">
    <div>
      <div style="font-size:0.7rem;color:#7a8aa6;text-transform:uppercase;letter-spacing:0.08em;font-weight:600;">Terminal Value</div>
      <div style="font-size:1.3rem;font-weight:700;color:#f1f5f9;margin-top:2px;">${result.terminal_value/1e9:,.2f}B</div>
    </div>
    <div>
      <div style="font-size:0.7rem;color:#7a8aa6;text-transform:uppercase;letter-spacing:0.08em;font-weight:600;">PV of Terminal</div>
      <div style="font-size:1.3rem;font-weight:700;color:#f1f5f9;margin-top:2px;">${pv_tv/1e9:,.2f}B</div>
      <div style="font-size:0.78rem;color:#7a8aa6;">{tv_pct*100:.1f}% of EV</div>
    </div>
    <div>
      <div style="font-size:0.7rem;color:#7a8aa6;text-transform:uppercase;letter-spacing:0.08em;font-weight:600;">Explicit Period</div>
      <div style="font-size:1.3rem;font-weight:700;color:#f1f5f9;margin-top:2px;">${(result.enterprise_value - pv_tv)/1e9:,.2f}B</div>
      <div style="font-size:0.78rem;color:#7a8aa6;">{explicit_pct*100:.1f}% of EV</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

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

The investment rating combines **DCF upside** with your **moat score** (0–10):

| Stars | Color | Label | Meaning |
|---|---|---|---|
| ★★★★★ +5 | Green | Strong Buy | DCF says >+45% upside |
| ★★★★ +4 / ★★★ +3 | Green | Buy | Material upside |
| ★★ +2 / ★ +1 | Yellow | Weak Buy | Marginal upside |
| 0 | Gray | Hold | Roughly fairly priced |
| ★ −1 / ★★ −2 | Yellow | Weak Short | Marginal overvaluation |
| ★★★ −3 / ★★★★ −4 | Red | Short | Material overvaluation |
| ★★★★★ −5 | Red | Strong Short | DCF says <−45% downside |

**Moat adjustment:** a high moat (≥8) pulls a "Short" rating toward zero — a
DCF alone can underestimate premium franchises with pricing power. A low moat
(≤3) pulls a "Buy" rating toward zero — apparent upside is less reliable
without a durable edge.

### Number formatting

- All ratios display as percentages with 2-decimal precision (`34.85%`).
- Percent inputs accept the percent form: type `34.85` to mean 34.85%.
- Large numbers (shares, debt) display with thousand separators.
- Negative upside is shown in red; positive upside in green.

### Financial Statements tab
Reconstructs Income Statement, Balance Sheet, and Cash Flow Statement from
SEC XBRL us-gaap tags as reported by the issuer. Values ≥ $1M are shown in
billions; EPS is shown per-share.

### Free hosting (Streamlit Community Cloud)
1. Push this repo to GitHub.
2. Sign in at [streamlit.io/cloud](https://streamlit.io/cloud) with GitHub.
3. **New app** → pick repo → main file = `app.py` → **Deploy**.
""")
