"""Auto-derive moat strength from financial data.

The qualitative-moat slider is meant to capture *durable competitive advantage*
that a one-shot DCF cannot. In the absence of expert judgement, a reasonable
data-driven proxy is built from five components, each scored 0–2:

  1. ROIC excess over WACC (5y average vs 9% baseline) — real value creation
  2. Gross-margin level + stability (CV of GM)                — pricing power
  3. Operating-margin level                                   — scale economies
  4. Revenue growth consistency (mean / std of YoY growth)    — durability
  5. Cash conversion (CFO / NI)                              — earnings quality

Sum → 0–10. The score is a *suggestion* — the slider remains editable.
The scoring tries to be conservative: a company has to clear several bars
to land in the 8–10 "wide moat" zone.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class MoatBreakdown:
    score: int                       # final integer 0..10
    components: dict[str, dict]      # per-component {value, points, why}
    rationale: str                   # one-line summary

    def to_html(self) -> str:
        rows = []
        for k, v in self.components.items():
            color = (
                "#22c55e" if v["points"] >= 1.5 else
                "#eab308" if v["points"] >= 1.0 else
                "#ef4444"
            )
            rows.append(
                f'<tr>'
                f'<td style="color:#cbd5e1;padding:4px 10px;font-size:0.82rem;">{k}</td>'
                f'<td style="color:{color};padding:4px 10px;font-weight:700;text-align:right;">{v["points"]:.1f}/2</td>'
                f'<td style="color:#7a8aa6;padding:4px 10px;font-size:0.78rem;">{v["why"]}</td>'
                f'</tr>'
            )
        return (
            '<table style="border-collapse:collapse;width:100%;">'
            '<thead><tr>'
            '<th style="text-align:left;color:#7a8aa6;padding:6px 10px;font-size:0.72rem;text-transform:uppercase;letter-spacing:0.06em;">Component</th>'
            '<th style="text-align:right;color:#7a8aa6;padding:6px 10px;font-size:0.72rem;text-transform:uppercase;letter-spacing:0.06em;">Score</th>'
            '<th style="text-align:left;color:#7a8aa6;padding:6px 10px;font-size:0.72rem;text-transform:uppercase;letter-spacing:0.06em;">Why</th>'
            '</tr></thead><tbody>'
            + "".join(rows)
            + '</tbody></table>'
        )


def _row(fin: pd.DataFrame, key: str) -> pd.Series:
    if key not in fin.index:
        return pd.Series(dtype="float64")
    return fin.loc[key].astype(float)


def _safe_pct(num: pd.Series, den: pd.Series) -> pd.Series:
    return (num / den).replace([np.inf, -np.inf], np.nan).dropna()


def _component_roic(fin: pd.DataFrame, wacc_baseline: float = 0.09) -> dict:
    ebit = _row(fin, "OperatingIncome")
    if ebit.empty or len(ebit.dropna()) < 2:
        return {"value": None, "points": 1.0, "why": "Insufficient data, neutral score"}

    eq = _row(fin, "TotalEquity").fillna(0) if not _row(fin, "TotalEquity").empty else pd.Series(0.0, index=ebit.index)
    debt_lt = _row(fin, "LongTermDebt").fillna(0) if not _row(fin, "LongTermDebt").empty else pd.Series(0.0, index=ebit.index)
    debt_st = _row(fin, "ShortTermDebt").fillna(0) if not _row(fin, "ShortTermDebt").empty else pd.Series(0.0, index=ebit.index)
    cash = _row(fin, "CashAndEquivalents").fillna(0) if not _row(fin, "CashAndEquivalents").empty else pd.Series(0.0, index=ebit.index)
    invested = (eq + debt_lt + debt_st - cash).replace(0, np.nan)
    if invested.dropna().empty:
        return {"value": None, "points": 1.0, "why": "No invested-capital data"}

    # Effective tax
    tx = _row(fin, "IncomeTaxExpense"); pt = _row(fin, "PreTaxIncome")
    tax = 0.21
    if not tx.empty and not pt.empty:
        r = (tx/pt).replace([np.inf,-np.inf], np.nan).dropna()
        r = r[(r > 0) & (r < 0.5)]
        if not r.empty:
            tax = float(r.tail(3).mean())

    roic_series = (ebit * (1 - tax) / invested).replace([np.inf,-np.inf], np.nan).dropna()
    if roic_series.empty:
        return {"value": None, "points": 1.0, "why": "ROIC undefined"}
    avg = float(roic_series.tail(5).mean())
    excess = avg - wacc_baseline

    if excess >= 0.15:
        pts, why = 2.0, f"ROIC {avg*100:.1f}% — exceptional (>{wacc_baseline*100:.0f}% WACC by 15%+)"
    elif excess >= 0.08:
        pts, why = 1.6, f"ROIC {avg*100:.1f}% — strong value creation"
    elif excess >= 0.03:
        pts, why = 1.2, f"ROIC {avg*100:.1f}% — moderate value creation"
    elif excess >= -0.02:
        pts, why = 0.6, f"ROIC {avg*100:.1f}% — roughly at cost of capital"
    else:
        pts, why = 0.0, f"ROIC {avg*100:.1f}% — destroys capital"

    return {"value": avg, "points": pts, "why": why}


def _component_gross_margin(fin: pd.DataFrame) -> dict:
    rev = _row(fin, "Revenue"); gp = _row(fin, "GrossProfit")
    if rev.empty or gp.empty:
        return {"value": None, "points": 1.0, "why": "Gross profit not reported"}
    gm = _safe_pct(gp, rev)
    if gm.empty or len(gm) < 2:
        return {"value": None, "points": 1.0, "why": "Insufficient GM history"}
    mean_gm = float(gm.tail(5).mean())
    cv = float(gm.tail(5).std() / abs(mean_gm)) if mean_gm else 1.0

    # Level component (0-1) + stability component (0-1)
    if mean_gm >= 0.55:
        lvl = 1.0
    elif mean_gm >= 0.40:
        lvl = 0.7
    elif mean_gm >= 0.25:
        lvl = 0.4
    else:
        lvl = 0.1
    stb = max(0.0, min(1.0, 1.0 - cv * 4))  # CV of 0% → 1.0, CV ≥25% → 0
    pts = lvl + stb
    why = f"Avg GM {mean_gm*100:.1f}% (level {lvl:.1f}) · CV {cv*100:.1f}% (stability {stb:.1f})"
    return {"value": mean_gm, "points": pts, "why": why}


def _component_op_margin(fin: pd.DataFrame) -> dict:
    rev = _row(fin, "Revenue"); ebit = _row(fin, "OperatingIncome")
    if rev.empty or ebit.empty:
        return {"value": None, "points": 1.0, "why": "Operating margin unavailable"}
    om = _safe_pct(ebit, rev)
    if om.empty:
        return {"value": None, "points": 1.0, "why": "OM undefined"}
    avg = float(om.tail(5).mean())
    if avg >= 0.30:
        pts, why = 2.0, f"Avg OM {avg*100:.1f}% — exceptional scale economics"
    elif avg >= 0.20:
        pts, why = 1.6, f"Avg OM {avg*100:.1f}% — strong"
    elif avg >= 0.12:
        pts, why = 1.2, f"Avg OM {avg*100:.1f}% — solid"
    elif avg >= 0.06:
        pts, why = 0.6, f"Avg OM {avg*100:.1f}% — modest"
    else:
        pts, why = 0.0, f"Avg OM {avg*100:.1f}% — weak"
    return {"value": avg, "points": pts, "why": why}


def _component_growth(fin: pd.DataFrame) -> dict:
    rev = _row(fin, "Revenue").dropna()
    if len(rev) < 3:
        return {"value": None, "points": 1.0, "why": "<3 yrs revenue history"}
    yoy = rev.pct_change().dropna()
    if yoy.empty:
        return {"value": None, "points": 1.0, "why": "YoY growth undefined"}
    mean_g = float(yoy.tail(5).mean())
    std_g = float(yoy.tail(5).std()) if len(yoy.tail(5)) > 1 else 0.05

    # Reward consistent positive growth; penalise volatility/decline
    if mean_g >= 0.10 and std_g <= 0.10:
        pts, why = 2.0, f"Mean growth {mean_g*100:.1f}% with low volatility (σ {std_g*100:.1f}%)"
    elif mean_g >= 0.05:
        pts, why = 1.4, f"Mean growth {mean_g*100:.1f}% — steady"
    elif mean_g >= 0.0:
        pts, why = 0.8, f"Mean growth {mean_g*100:.1f}% — slow"
    else:
        pts, why = 0.0, f"Mean growth {mean_g*100:.1f}% — declining"
    if std_g > 0.20 and pts > 0.4:
        pts -= 0.4
        why += f" (penalised: σ {std_g*100:.1f}% volatile)"
    return {"value": mean_g, "points": max(0.0, pts), "why": why}


def _component_cash_conversion(fin: pd.DataFrame) -> dict:
    cfo = _row(fin, "CashFromOps"); ni = _row(fin, "NetIncome")
    if cfo.empty or ni.empty:
        return {"value": None, "points": 1.0, "why": "Cash conversion unavailable"}
    cc = (cfo / ni).replace([np.inf, -np.inf], np.nan).dropna()
    cc = cc[(cc > 0) & (cc < 5)]
    if cc.empty:
        return {"value": None, "points": 1.0, "why": "CFO/NI undefined"}
    avg = float(cc.tail(5).mean())
    if avg >= 1.20:
        pts, why = 2.0, f"CFO/NI {avg:.2f} — earnings backed by cash plus more"
    elif avg >= 1.00:
        pts, why = 1.5, f"CFO/NI {avg:.2f} — earnings turn into cash"
    elif avg >= 0.80:
        pts, why = 0.8, f"CFO/NI {avg:.2f} — modest cash conversion"
    else:
        pts, why = 0.0, f"CFO/NI {avg:.2f} — accruals risk"
    return {"value": avg, "points": pts, "why": why}


def derive_moat(fin: pd.DataFrame, wacc_baseline: float = 0.09) -> MoatBreakdown:
    if fin is None or fin.empty:
        return MoatBreakdown(score=5, components={}, rationale="No data — neutral default 5/10")

    components = {
        "ROIC vs WACC":           _component_roic(fin, wacc_baseline),
        "Gross-margin power":     _component_gross_margin(fin),
        "Operating margin":       _component_op_margin(fin),
        "Growth consistency":     _component_growth(fin),
        "Cash conversion":        _component_cash_conversion(fin),
    }
    total = sum(c["points"] for c in components.values())
    score = int(round(max(0.0, min(10.0, total))))
    if score >= 8:
        verdict = "Wide moat — compounding machine"
    elif score >= 6:
        verdict = "Strong moat — durable advantages"
    elif score >= 4:
        verdict = "Modest moat — average competitive position"
    elif score >= 2:
        verdict = "Narrow moat — fragile economics"
    else:
        verdict = "No moat — commodity-like business"
    return MoatBreakdown(
        score=score,
        components=components,
        rationale=f"{score}/10 · {verdict}",
    )


__all__ = ["derive_moat", "MoatBreakdown"]
