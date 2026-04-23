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
        # High moat on an "overvalued" signal — bring toward zero (don't overshoot)
        return min(0, base_stars + adj) - base_stars
    if base_stars > 0 and adj < 0:
        # Low moat on an "undervalued" signal — bring toward zero
        return max(0, base_stars + adj) - base_stars
    return 0


def compute_rating(upside_pct: float | None, moat_score: int = 5) -> dict:
    """Return dict with stars (-5..+5), label, colour, and html."""
    if upside_pct is None:
        return {
            "stars": 0, "label": "No signal", "color": "#8b9ab5",
            "html": _render_html(0, "No signal", "#8b9ab5", note="DCF not computed"),
            "note": "DCF not computed",
        }
    base = _upside_to_stars(upside_pct)
    moat_clamped = max(0, min(10, int(moat_score)))
    final = max(-5, min(5, base + _moat_adjustment(moat_clamped, base)))

    if final >= 3:
        color, label = "#2da44e", _label_for_positive(final)
    elif final <= -3:
        color, label = "#cf222e", _label_for_negative(final)
    elif final == 0:
        color, label = "#8b9ab5", "Hold"
    else:
        color, label = "#d4a72c", _label_for_neutral(final)

    note = f"DCF upside {upside_pct:.2%} · moat {moat_clamped}/10"
    return {
        "stars": final,
        "label": label,
        "color": color,
        "html": _render_html(final, label, color, note=note),
        "note": note,
    }


def _label_for_positive(stars: int) -> str:
    return {5: "Strong Buy", 4: "Buy", 3: "Buy"}.get(stars, "Buy")


def _label_for_negative(stars: int) -> str:
    return {-5: "Strong Short", -4: "Short", -3: "Short"}.get(stars, "Short")


def _label_for_neutral(stars: int) -> str:
    if stars > 0:
        return "Weak Buy"
    if stars < 0:
        return "Weak Short"
    return "Hold"


def _render_html(stars: int, label: str, color: str, note: str = "") -> str:
    """Render 5 stars. Filled count = |stars|. Sign controls colour direction."""
    filled = abs(stars)
    sign = "+" if stars > 0 else ("−" if stars < 0 else "±")

    star_html = ""
    for i in range(1, 6):
        if i <= filled:
            star_html += f'<span style="color:{color};text-shadow:0 0 8px {color}80;">★</span>'
        else:
            star_html += '<span style="color:#2a3550;">★</span>'

    return f"""
    <div style="display:flex;align-items:center;gap:14px;">
      <div style="font-size:1.6rem;letter-spacing:2px;line-height:1;">{star_html}</div>
      <div style="display:flex;flex-direction:column;gap:2px;">
        <div style="font-weight:700;font-size:0.95rem;color:{color};letter-spacing:0.04em;text-transform:uppercase;">
          {label} <span style="color:#8b9ab5;font-weight:500;">({sign}{filled}/5)</span>
        </div>
        <div style="font-size:0.72rem;color:#5c7099;">{note}</div>
      </div>
    </div>
    """


__all__ = ["compute_rating"]
