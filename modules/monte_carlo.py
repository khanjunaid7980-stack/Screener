"""Monte Carlo simulation for DCF fair value per share.

Samples key DCF inputs from normal distributions to produce a probability
distribution of outcomes rather than a single point estimate.

Inputs varied (all via truncated-normal to avoid nonsensical values):
  - revenue_growth_rate   (σ default = 30% of mean)
  - operating_margin      (σ default = 15% of mean, or 3 pp absolute)
  - terminal_growth_rate  (σ default = 0.5 pp absolute)
  - wacc                  (σ default = 1.0 pp absolute)
  - capex_pct_revenue     (σ default = 15% of mean)
  - da_pct_revenue        (σ default = 10% of mean)

Each simulation independently samples all six parameters, runs a full DCF,
and records the fair value per share.  Results that are numerically undefined
(NaN / ±Inf) or outside the [-10×current_price, +50×current_price] window
are dropped as non-converged.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .assumptions import Assumptions
from .dcf import run_dcf


@dataclass
class MCResult:
    simulations: np.ndarray          # 1-D array of fair-value-per-share, post-filtering
    n_total: int                      # raw simulations attempted
    n_valid: int                      # simulations that produced finite values
    mean: float
    median: float
    std: float
    p5: float
    p10: float
    p25: float
    p75: float
    p90: float
    p95: float
    current_price: float | None
    prob_undervalued: float | None    # P(fair_value > current_price)
    prob_upside_20: float | None      # P(fair_value > 1.2 * current_price)
    prob_downside_20: float | None    # P(fair_value < 0.8 * current_price)

    def percentile_table(self) -> pd.DataFrame:
        rows = [
            ("5th  (Bear case)",   self.p5),
            ("10th",               self.p10),
            ("25th",               self.p25),
            ("50th (Base case)",   self.median),
            ("75th",               self.p75),
            ("90th",               self.p90),
            ("95th (Bull case)",   self.p95),
        ]
        df = pd.DataFrame(rows, columns=["Percentile", "Fair Value / Share"])
        df["Fair Value / Share"] = df["Fair Value / Share"].apply(lambda x: f"${x:,.2f}")
        if self.current_price:
            df["vs Current Price"] = [
                f"{(v/self.current_price - 1)*100:+.1f}%"
                if isinstance(v, float) else "—"
                for _, v in rows
            ]
        return df.set_index("Percentile")


def _truncnorm_sample(
    rng: np.random.Generator,
    mean: float,
    std: float,
    low: float,
    high: float,
    n: int,
) -> np.ndarray:
    """Sample from a normal clamped to [low, high]."""
    raw = rng.normal(mean, std, size=n * 3)  # oversample then filter
    raw = raw[(raw >= low) & (raw <= high)]
    if len(raw) >= n:
        return raw[:n]
    # fall back: clamp remainder
    shortfall = n - len(raw)
    extra = np.clip(rng.normal(mean, std, size=shortfall), low, high)
    return np.concatenate([raw, extra])


def run_monte_carlo(
    base_revenue: float,
    assumptions: Assumptions,
    wacc_base: float,
    shares_outstanding: float,
    net_debt: float,
    current_price: float | None = None,
    n_sims: int = 2000,
    # Standard deviations for each sampled input
    # Pass None to use a default relative or absolute σ
    growth_std: float | None = None,
    margin_std: float | None = None,
    terminal_g_std: float | None = None,
    wacc_std: float | None = None,
    capex_std: float | None = None,
    da_std: float | None = None,
    seed: int | None = None,
) -> MCResult:
    rng = np.random.default_rng(seed)

    # ── resolve default σ ────────────────────────────────────────────────────
    g_mean   = float(assumptions.revenue_growth_rate)
    om_mean  = float(assumptions.operating_margin)
    tg_mean  = float(assumptions.terminal_growth_rate)
    w_mean   = float(wacc_base)
    cx_mean  = float(assumptions.capex_pct_revenue)
    da_mean  = float(assumptions.da_pct_revenue)

    g_std   = growth_std    if growth_std    is not None else max(abs(g_mean) * 0.30, 0.02)
    om_std  = margin_std    if margin_std    is not None else max(abs(om_mean) * 0.20, 0.02)
    tg_std  = terminal_g_std if terminal_g_std is not None else 0.005
    w_std   = wacc_std      if wacc_std      is not None else 0.010
    cx_std  = capex_std     if capex_std     is not None else max(abs(cx_mean) * 0.20, 0.005)
    da_std  = da_std        if da_std        is not None else max(abs(da_mean) * 0.15, 0.003)

    # ── draw samples ─────────────────────────────────────────────────────────
    g_s   = _truncnorm_sample(rng, g_mean,  g_std,  -0.50, 1.0,   n_sims)
    om_s  = _truncnorm_sample(rng, om_mean, om_std, -0.50, 0.80,  n_sims)
    tg_s  = _truncnorm_sample(rng, tg_mean, tg_std, -0.05, 0.10,  n_sims)
    w_s   = _truncnorm_sample(rng, w_mean,  w_std,  0.01,  0.50,  n_sims)
    cx_s  = _truncnorm_sample(rng, cx_mean, cx_std, 0.0,   0.30,  n_sims)
    da_s  = _truncnorm_sample(rng, da_mean, da_std, 0.0,   0.20,  n_sims)

    # Enforce wacc > terminal_g + 0.5 pp in each sim
    tg_s = np.minimum(tg_s, w_s - 0.005)

    fair_values: list[float] = []
    n_total = n_sims

    for i in range(n_sims):
        a_sim = Assumptions(**assumptions.to_dict())
        a_sim.revenue_growth_rate = float(g_s[i])
        a_sim.operating_margin    = float(om_s[i])
        a_sim.terminal_growth_rate = float(tg_s[i])
        a_sim.capex_pct_revenue   = float(cx_s[i])
        a_sim.da_pct_revenue      = float(da_s[i])

        try:
            res = run_dcf(
                base_revenue=base_revenue,
                a=a_sim,
                wacc=float(w_s[i]),
                shares_outstanding=shares_outstanding,
                net_debt=net_debt,
                current_price=None,  # skip upside calc for speed
            )
            fv = res.fair_value_per_share
            if np.isfinite(fv):
                fair_values.append(fv)
        except Exception:
            pass

    arr = np.array(fair_values, dtype=float)

    # Drop extreme outliers: keep within ±50× current price (or ±500% of median)
    if len(arr) > 10:
        med = float(np.median(arr))
        hi = med * 20 if med > 0 else abs(med) * 20
        lo = med / 20 if med > 0 else -abs(med) * 20
        arr = arr[(arr >= lo) & (arr <= hi)]

    n_valid = len(arr)
    if n_valid == 0:
        # Return a degenerate result
        return MCResult(
            simulations=arr, n_total=n_total, n_valid=0,
            mean=float("nan"), median=float("nan"), std=float("nan"),
            p5=float("nan"), p10=float("nan"), p25=float("nan"),
            p75=float("nan"), p90=float("nan"), p95=float("nan"),
            current_price=current_price,
            prob_undervalued=None, prob_upside_20=None, prob_downside_20=None,
        )

    p5, p10, p25, p50, p75, p90, p95 = np.percentile(arr, [5, 10, 25, 50, 75, 90, 95])
    prob_uv = float(np.mean(arr > current_price)) if current_price else None
    prob_u20 = float(np.mean(arr > current_price * 1.20)) if current_price else None
    prob_d20 = float(np.mean(arr < current_price * 0.80)) if current_price else None

    return MCResult(
        simulations=arr,
        n_total=n_total,
        n_valid=n_valid,
        mean=float(np.mean(arr)),
        median=float(p50),
        std=float(np.std(arr)),
        p5=float(p5),
        p10=float(p10),
        p25=float(p25),
        p75=float(p75),
        p90=float(p90),
        p95=float(p95),
        current_price=current_price,
        prob_undervalued=prob_uv,
        prob_upside_20=prob_u20,
        prob_downside_20=prob_d20,
    )


__all__ = ["run_monte_carlo", "MCResult"]
