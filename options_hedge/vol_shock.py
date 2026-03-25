"""
Vol Shock Engine — applies vol shocks and re-prices options.

All functions are pure: no hidden state, no I/O.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from .models import MarketState, OptionSpec, SmileSurface
from .pricing import black_scholes_price
from .smile import interpolate_iv_from_smile, price_option_from_moneyness


def apply_vol_shock(
    market: MarketState,
    vol_shock_abs: float,
) -> MarketState:
    """
    Return a new MarketState with a flat vol shift applied.

    Parameters
    ----------
    market:
        Baseline market state.
    vol_shock_abs:
        Absolute change in implied vol (e.g. +0.10 adds 10 pp).

    Returns
    -------
    New MarketState with modified implied_vol.
    """
    new_vol = max(0.001, market.implied_vol + vol_shock_abs)
    return MarketState(
        spot=market.spot,
        risk_free_rate=market.risk_free_rate,
        dividend_yield=market.dividend_yield,
        implied_vol=new_vol,
    )


def shift_smile(
    smile: SmileSurface,
    vol_shock_abs: float,
) -> SmileSurface:
    """
    Apply a parallel shift to an entire SmileSurface.

    Parameters
    ----------
    smile:
        Original SmileSurface.
    vol_shock_abs:
        Absolute vol shift to add to every point.

    Returns
    -------
    New SmileSurface with shifted vols (floored at 0.001).
    """
    shifted_points = {
        m: max(0.001, iv + vol_shock_abs)
        for m, iv in smile.points.items()
    }
    return SmileSurface(points=shifted_points)


def price_with_vol_shock(
    option: OptionSpec,
    market: MarketState,
    vol_shock_abs: float,
    smile: Optional[SmileSurface] = None,
    *,
    interp_method: str = "cubic",
) -> Dict[str, float]:
    """
    Re-price an option after applying a vol shock.

    If a ``smile`` is provided the shock is applied to the smile first,
    then the smile-adjusted vol is used for pricing.  Otherwise a flat
    Black-Scholes reprice is performed with the shocked flat vol.

    Parameters
    ----------
    option:
        The option to reprice.
    market:
        Current (pre-shock) market state.
    vol_shock_abs:
        Absolute vol shift.
    smile:
        Optional SmileSurface.  When supplied the reprice uses the smile.
    interp_method:
        Interpolation method for the smile.

    Returns
    -------
    Merged dict from the underlying pricer plus:
        shocked_vol     — the vol actually used for pricing
        vol_shock_abs   — the shock that was applied
    """
    if smile is not None:
        shocked_smile = shift_smile(smile, vol_shock_abs)
        result = price_option_from_moneyness(
            option, market, shocked_smile, method=interp_method
        )
        shocked_vol = result["smile_iv"]
    else:
        shocked_market = apply_vol_shock(market, vol_shock_abs)
        result = black_scholes_price(option, shocked_market)
        shocked_vol = shocked_market.implied_vol

    return {**result, "shocked_vol": shocked_vol, "vol_shock_abs": vol_shock_abs}


def vol_shock_pnl_matrix(
    option: OptionSpec,
    market: MarketState,
    spot_shocks: List[float],
    vol_shocks: List[float],
    premium_paid: float,
    smile: Optional[SmileSurface] = None,
    *,
    interp_method: str = "cubic",
) -> List[Dict[str, float]]:
    """
    Build a P&L matrix across a grid of spot × vol shocks.

    Parameters
    ----------
    option:
        Option to evaluate.
    market:
        Baseline market state.
    spot_shocks:
        Fractional spot shocks, e.g. [-0.20, -0.10, 0.0, 0.10].
    vol_shocks:
        Absolute vol shocks, e.g. [-0.05, 0.0, 0.10, 0.20].
    premium_paid:
        Original premium per unit.
    smile:
        Optional SmileSurface for vol lookup.
    interp_method:
        Interpolation method for the smile.

    Returns
    -------
    List of dicts with keys:
        spot_shock_pct, vol_shock_abs, shocked_spot, shocked_vol,
        option_price, net_pnl, return_on_premium
    """
    rows = []
    for s_shock in spot_shocks:
        shocked_spot = market.spot * (1.0 + s_shock)
        shocked_market = MarketState(
            spot=shocked_spot,
            risk_free_rate=market.risk_free_rate,
            dividend_yield=market.dividend_yield,
            implied_vol=market.implied_vol,
        )
        for v_shock in vol_shocks:
            result = price_with_vol_shock(
                option,
                shocked_market,
                v_shock,
                smile=smile,
                interp_method=interp_method,
            )
            price = result["price"]
            net_pnl = (price - premium_paid) * option.quantity
            total_premium = premium_paid * option.quantity
            rop = (net_pnl / total_premium) if total_premium != 0 else None
            rows.append(
                {
                    "spot_shock_pct": s_shock,
                    "vol_shock_abs": v_shock,
                    "shocked_spot": shocked_spot,
                    "shocked_vol": result["shocked_vol"],
                    "option_price": price,
                    "net_pnl": net_pnl,
                    "return_on_premium": rop,
                }
            )
    return rows
