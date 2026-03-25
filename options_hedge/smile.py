"""
Smile Engine — reads and interpolates the implied-volatility smile.

Supports interpolation by moneyness (K/S) or by absolute strike.
All functions are pure: no hidden state, no I/O.
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

from .models import MarketState, OptionSpec, SmileSurface


# ---------------------------------------------------------------------------
# Interpolation helpers
# ---------------------------------------------------------------------------

def _linear_interpolate(x: float, xs: List[float], ys: List[float]) -> float:
    """
    Piecewise-linear interpolation with flat extrapolation at the boundaries.
    """
    if x <= xs[0]:
        return ys[0]
    if x >= xs[-1]:
        return ys[-1]
    for i in range(len(xs) - 1):
        if xs[i] <= x <= xs[i + 1]:
            t = (x - xs[i]) / (xs[i + 1] - xs[i])
            return ys[i] + t * (ys[i + 1] - ys[i])
    return ys[-1]


def _cubic_spline_natural(xs: List[float], ys: List[float], x: float) -> float:
    """
    Natural cubic-spline interpolation (no external dependencies).
    Falls back to linear if fewer than 3 points.
    """
    n = len(xs)
    if n < 3:
        return _linear_interpolate(x, xs, ys)

    # Build tridiagonal system for second derivatives
    h = [xs[i + 1] - xs[i] for i in range(n - 1)]
    alpha = [
        3.0 * ((ys[i + 1] - ys[i]) / h[i] - (ys[i] - ys[i - 1]) / h[i - 1])
        for i in range(1, n - 1)
    ]

    l = [1.0] + [0.0] * (n - 1)
    mu = [0.0] * n
    z = [0.0] * n

    for i in range(1, n - 1):
        l[i] = 2.0 * (xs[i + 1] - xs[i - 1]) - h[i - 1] * mu[i - 1]
        mu[i] = h[i] / l[i]
        z[i] = (alpha[i - 1] - h[i - 1] * z[i - 1]) / l[i]

    l[n - 1] = 1.0
    c = [0.0] * n
    b = [0.0] * (n - 1)
    d = [0.0] * (n - 1)

    for j in range(n - 2, -1, -1):
        c[j] = z[j] - mu[j] * c[j + 1]
        b[j] = (ys[j + 1] - ys[j]) / h[j] - h[j] * (c[j + 1] + 2.0 * c[j]) / 3.0
        d[j] = (c[j + 1] - c[j]) / (3.0 * h[j])

    # Evaluate
    if x <= xs[0]:
        dx = x - xs[0]
        return ys[0] + b[0] * dx + c[0] * dx ** 2 + d[0] * dx ** 3
    if x >= xs[-1]:
        i = n - 2
        dx = x - xs[i]
        return ys[i] + b[i] * dx + c[i] * dx ** 2 + d[i] * dx ** 3
    for i in range(n - 1):
        if xs[i] <= x <= xs[i + 1]:
            dx = x - xs[i]
            return ys[i] + b[i] * dx + c[i] * dx ** 2 + d[i] * dx ** 3
    return ys[-1]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def interpolate_iv_from_smile(
    smile: SmileSurface,
    moneyness: float,
    *,
    method: str = "cubic",
) -> Dict[str, float]:
    """
    Interpolate the implied volatility from a SmileSurface at a given moneyness.

    Parameters
    ----------
    smile:
        SmileSurface mapping moneyness -> implied vol.
    moneyness:
        Target K/S ratio.
    method:
        ``"cubic"`` (natural cubic spline, default) or ``"linear"``.

    Returns
    -------
    dict with keys:
        iv              — interpolated implied volatility
        moneyness       — the requested moneyness
        extrapolated    — True if moneyness is outside the smile range
    """
    sorted_points: List[Tuple[float, float]] = sorted(smile.points.items())
    xs = [p[0] for p in sorted_points]
    ys = [p[1] for p in sorted_points]

    extrapolated = moneyness < xs[0] or moneyness > xs[-1]

    if method == "linear":
        iv = _linear_interpolate(moneyness, xs, ys)
    else:
        iv = _cubic_spline_natural(xs, ys, moneyness)

    iv = max(iv, 0.001)  # floor at 0.1 % to avoid non-positive vols

    return {
        "iv": iv,
        "moneyness": moneyness,
        "extrapolated": extrapolated,
    }


def price_option_from_moneyness(
    option: OptionSpec,
    market: MarketState,
    smile: SmileSurface,
    *,
    method: str = "cubic",
) -> Dict[str, float]:
    """
    Re-price an option using the smile surface instead of a flat vol.

    The smile IV at the option's moneyness (strike / spot) is used as the
    input volatility to the Black-Scholes pricer.

    Parameters
    ----------
    option:
        The option to price.
    market:
        Current market state.
    smile:
        SmileSurface to read IV from.
    method:
        Interpolation method passed to ``interpolate_iv_from_smile``.

    Returns
    -------
    Merged dict of Black-Scholes output fields plus:
        smile_iv        — IV taken from the smile
        smile_moneyness — moneyness used for the smile lookup
        extrapolated    — True if the smile was extrapolated
    """
    from .pricing import black_scholes_price  # local import to avoid circular

    moneyness = option.strike / market.spot
    smile_result = interpolate_iv_from_smile(smile, moneyness, method=method)
    smile_iv = smile_result["iv"]

    bs_result = black_scholes_price(option, market, override_vol=smile_iv)

    return {
        **bs_result,
        "smile_iv": smile_iv,
        "smile_moneyness": moneyness,
        "extrapolated": smile_result["extrapolated"],
    }


def build_smile_from_atm_skew(
    atm_vol: float,
    skew_per_delta: float = 0.02,
    moneyness_grid: List[float] | None = None,
) -> SmileSurface:
    """
    Convenience constructor: build a parametric smile from ATM vol and a linear skew.

    The skew is applied symmetrically: OTM puts get higher vol, OTM calls lower.
    ``skew_per_delta`` is the vol increment per unit of |1 - moneyness|.

    Parameters
    ----------
    atm_vol:
        At-the-money implied vol (annualised).
    skew_per_delta:
        Additional vol per unit of distance from ATM.
    moneyness_grid:
        List of moneyness points to generate; defaults to
        [0.80, 0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15, 1.20].

    Returns
    -------
    SmileSurface
    """
    if moneyness_grid is None:
        moneyness_grid = [0.80, 0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15, 1.20]

    points: Dict[float, float] = {}
    for m in moneyness_grid:
        if m <= 1.0:
            # OTM put territory: vol increases as moneyness falls
            iv = atm_vol + skew_per_delta * (1.0 - m)
        else:
            # OTM call territory: slight smile
            iv = atm_vol + 0.5 * skew_per_delta * (m - 1.0)
        points[m] = max(iv, 0.001)

    return SmileSurface(points=points)
