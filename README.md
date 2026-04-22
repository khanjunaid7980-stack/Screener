# Financial Screening Tool

A free, interactive, standalone financial screener with DCF valuation. All
data comes from public sources (SEC EDGAR + Yahoo Finance) — **no API keys,
no paid services**.

## Features

- **Automatic data retrieval** from the SEC EDGAR `company-facts` XBRL API
  (10-K annual values) plus Yahoo Finance for live price, beta, and the
  10-year Treasury yield.
- **Historical window is user-controlled** (3–10 years).
- **Every assumption pre-filled** from a blend of:
  - a log-linear trend fit on the company's own history,
  - sector benchmarks (Damodaran-style averages), and
  - live market data (Rf, beta).

  Every input is editable, exportable/importable as JSON, and one click away
  from being reset to ML defaults.
- **WACC** computed via CAPM + after-tax cost of debt, with a live walk-through.
- **DCF valuation** — 5-year explicit FCFF projection + Gordon-growth terminal
  value, discounted at WACC. Output: enterprise value, equity value, fair
  value per share, upside vs. market.
- **Sensitivity heatmap** over WACC × terminal growth.
- **Dedicated Visualizations tab** with Plotly charts:
  - Revenue & profit margins
  - ROIC vs. WACC
  - EBIT / Market-Cap
  - Cash flow breakdown (CFO, CapEx, FCF)
  - Leverage & liquidity
  - DCF projections
  - Price history vs. DCF fair value
  - Sensitivity heatmap

## Quick start — local

```bash
git clone https://github.com/khanjunaid7980-stack/screener.git
cd screener
pip install -r requirements.txt
streamlit run app.py
```

Open http://localhost:8501 in your browser.

## Free hosting — Streamlit Community Cloud (recommended)

1. Push this repository to GitHub (already done on branch
   `claude/financial-screening-tool-KMcj0`; merge to `main` for deployment).
2. Go to [streamlit.io/cloud](https://streamlit.io/cloud) and sign in with
   GitHub.
3. Click **New app** → pick this repo → set the main file to `app.py` →
   **Deploy**.
4. You get a permanent free URL like
   `https://<you>-screener.streamlit.app` accessible from anywhere.

No credit card, no key management, no server ops.

## How the pre-filled assumptions work

Each assumption is derived by `modules/assumptions.derive_assumptions`:

| Input | Source |
| --- | --- |
| Revenue growth | Log-linear regression on historical revenue, blended 60/40 with sector norm, clipped to `[-5%, 35%]`. |
| Operating margin | Trailing 3-year average EBIT / Revenue, blended with sector norm. |
| Tax rate | Trailing 3-year effective tax (filtered to 0–50%), blended with jurisdictional norm. |
| CapEx % revenue | Trailing 3-year average, blended with sector norm. |
| D&A % revenue | Trailing 3-year average, blended with sector norm. |
| ΔNWC % revenue | Sector benchmark. |
| Risk-free rate | Live 10Y US Treasury (^TNX) from Yahoo Finance. |
| Equity risk premium | 5.5% (long-run US consensus). |
| Beta | Yahoo Finance; falls back to sector beta. |
| Cost of debt (pre-tax) | Rf + 150 bps spread. |
| Debt weight | Balance sheet debt / (debt + market cap); sector default if unavailable. |
| Terminal growth | min(2.5%, 0.8 × Rf). Never exceeds Rf. |

Sector benchmarks live in `modules/assumptions.SECTOR_BENCHMARKS` and can be
tuned for your own work.

## Project layout

```
app.py                     Main Streamlit app (6 tabs)
modules/
  edgar.py                 SEC EDGAR company-facts client
  market_data.py           yfinance wrapper (price, beta, Rf)
  assumptions.py           ML/heuristic pre-filled assumptions
  wacc.py                  CAPM + WACC
  dcf.py                   FCFF DCF + sensitivity grid
  visualizations.py        Plotly charts
requirements.txt
.streamlit/config.toml
```

## Caveats

- SEC XBRL tagging varies across issuers; smaller or non-US filers may be
  missing line items. The tool targets US-listed 10-K filers.
- Earnings-call transcripts are not redistributable by the SEC; the *Data*
  tab links to a free Google search for the latest transcript.
- A DCF is only as reliable as its assumptions. This tool makes every
  assumption explicit and editable — use it as a starting point, not a verdict.
