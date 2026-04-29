"""Star rating for DCF-based investment signal.

Converts (DCF upside, moat score) into a signed integer rating in [-5, +5]
plus colour + label + HTML for the header banner.

Rules:
- Pure upside mapping first (ignoring moat):
    > +45% → +5,  +30..+45% → +4,  +15..+30% → +3,
    +5..+15% → +2,  0..+5% → +1,  -5..0% → 0,
    -5..-15% → -1, -15..-25% → -2, -25..-35% → -3,
    -35..-45% → -4, <-45% → -5

- Moat adjustment (0..10, 5 = neutral):
    A strong moat (>=8) partly justifies premium pricing — so it pulls the
    rating back toward neutral when the DCF says "overvalued". A weak moat
    (<=3) reduces confidence in apparent upside.
    adjustment = round((moat - 5) / 2.5)   → in {-2,-1,0,+1,+2}
    Applied only when it narrows the gap to zero (never flips the sign).
"""

from __future__ import annotations

UPSIDE_BUCKETS: list[tuple[float, int]] = [
    (0.45,  5),
    (0.30,  4),
    (0.15,  3),
    (0.05,  2),
    (0.0,   1),
    (-0.05, 0),
    (-0.15, -1),
    (-0.25, -2),
    (-0.35, -3),
    (-0.45, -4),
]


def _upside_to_stars(upside: float) -> int:
    for lo, stars in UPSIDE_BUCKETS:
        if upside >= lo:
            return stars
    return -5


def _moat_adjustment(moat: int, base_stars: int) -> int:
    """Nudge stars toward zero when moat disagrees with signal."""
    adj = round((moat - 5) / 2.5)  # in [-2, +2]
    if base_stars < 0 and adj > 0:
        return min(0, base_stars + adj) - base_stars
    if base_stars > 0 and adj < 0:
        return max(0, base_stars + adj) - base_stars
    return 0


def compute_rating(upside_pct: float | None, moat_score: int = 5) -> dict:
    """Return dict with stars (-5..+5), label, colour, and html."""
    if upside_pct is None:
        return {
            "stars": 0, "label": "No Signal", "color": "#8b9ab5",
            "html": _render_html(0, "No Signal", "#8b9ab5", note="DCF unavailable"),
            "note": "DCF unavailable",
        }
    base = _upside_to_stars(upside_pct)
    moat_clamped = max(0, min(10, int(moat_score)))
    final = max(-5, min(5, base + _moat_adjustment(moat_clamped, base)))

    if final >= 3:
        color = "#22c55e"
        label = {5: "Strong Buy", 4: "Buy", 3: "Buy"}[final]
    elif final <= -3:
        color = "#ef4444"
        label = {-5: "Strong Short", -4: "Short", -3: "Short"}[final]
    elif final == 0:
        color = "#94a3b8"
        label = "Hold"
    else:
        color = "#eab308"
        label = "Weak Buy" if final > 0 else "Weak Short"

    note = f"Upside {upside_pct*100:+.2f}% · Moat {moat_clamped}/10"
    return {
        "stars": final,
        "label": label,
        "color": color,
        "html": _render_html(final, label, color, note=note),
        "note": note,
    }


def _render_html(stars: int, label: str, color: str, note: str = "") -> str:
    """Render 5 stars (filled count = |stars|). Sign indicates direction.

    Direction symbol:
        ↑ for buy/long, ↓ for short, • for hold/no-signal
    """
    filled = abs(stars)
    if stars > 0:
        direction = "↑"
        sign_text = "Long"
    elif stars < 0:
        direction = "↓"
        sign_text = "Short"
    else:
        direction = "•"
        sign_text = "Neutral"

    star_html_parts = []
    for i in range(1, 6):
        if i <= filled:
            star_html_parts.append(
                f'<span style="color:{color};text-shadow:0 0 6px {color}66;'
                f'font-size:1.45rem;line-height:1;">★</span>'
            )
        else:
            star_html_parts.append(
                '<span style="color:#2a3550;font-size:1.45rem;line-height:1;">★</span>'
            )
    star_html = "".join(star_html_parts)

    return (
        f'<div style="display:flex;flex-direction:column;gap:6px;">'
        f'<div style="display:flex;align-items:center;gap:10px;">'
        f'<div style="display:flex;gap:3px;">{star_html}</div>'
        f'<div style="background:{color}22;color:{color};border:1px solid {color}55;'
        f'padding:2px 10px;border-radius:14px;font-size:0.72rem;font-weight:700;'
        f'letter-spacing:0.05em;text-transform:uppercase;white-space:nowrap;">'
        f'{direction} {label}</div>'
        f'</div>'
        f'<div style="font-size:0.72rem;color:#8b9ab5;font-weight:500;">'
        f'{sign_text} signal · {note}</div>'
        f'</div>'
    )


__all__ = ["compute_rating"]
