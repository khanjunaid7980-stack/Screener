"""WACC calculation.

Cost of Equity via CAPM: Ke = Rf + beta * ERP
After-tax cost of debt: Kd * (1 - t)
WACC = We * Ke + Wd * Kd * (1 - t)
"""

from __future__ import annotations

from dataclasses import dataclass

from .assumptions import Assumptions


@dataclass
class WACCResult:
    cost_of_equity: float
    after_tax_cost_of_debt: float
    equity_weight: float
    debt_weight: float
    wacc: float

    def summary(self) -> dict[str, float]:
        return {
            "Cost of Equity (Ke)": self.cost_of_equity,
            "After-tax Cost of Debt (Kd*(1-t))": self.after_tax_cost_of_debt,
            "Equity Weight (We)": self.equity_weight,
            "Debt Weight (Wd)": self.debt_weight,
            "WACC": self.wacc,
        }


def compute_wacc(a: Assumptions) -> WACCResult:
    ke = a.risk_free_rate + a.beta * a.equity_risk_premium
    kd_after = a.cost_of_debt_pretax * (1.0 - a.tax_rate)
    wd = max(0.0, min(1.0, a.debt_weight))
    we = 1.0 - wd
    wacc = we * ke + wd * kd_after
    return WACCResult(
        cost_of_equity=ke,
        after_tax_cost_of_debt=kd_after,
        equity_weight=we,
        debt_weight=wd,
        wacc=wacc,
    )


def infer_debt_weight_from_balance_sheet(
    total_debt: float | None,
    market_cap: float | None,
) -> float | None:
    """Target cap structure: D / (D + Market-Cap equity)."""
    if not total_debt or not market_cap or market_cap <= 0:
        return None
    total = float(total_debt) + float(market_cap)
    if total <= 0:
        return None
    return float(total_debt) / total


__all__ = ["WACCResult", "compute_wacc", "infer_debt_weight_from_balance_sheet"]
