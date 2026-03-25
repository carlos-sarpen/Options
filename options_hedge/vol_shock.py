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


def price_with_moneyness_shock(
    option: OptionSpec,
    market: MarketState,
    smile: SmileSurface,
    moneyness_shock_pct: float,
    premium_paid: float | None = None,
    *,
    interp_method: str = "hyperbolic",
) -> Dict[str, float | None]:
    """
    Re-price an option after shocking its smile moneyness.

    The current smile moneyness is ``strike / spot``. A fractional shock is
    applied to that ratio, the new smile IV is interpolated, and the option is
    re-priced with the shocked spot implied by the new moneyness.

    Parameters
    ----------
    option:
        Option to reprice.
    market:
        Baseline market state.
    smile:
        SmileSurface used to interpolate the shocked IV.
    moneyness_shock_pct:
        Fractional moneyness shock, e.g. ``0.05`` for +5%.
    premium_paid:
        Original premium per unit. When omitted, the baseline Black-Scholes
        premium is used as the invested amount.
    interp_method:
        Smile interpolation method. Defaults to ``"hyperbolic"``.

    Returns
    -------
    dict with keys:
        initial_price       — price paid initially (or BS fallback)
        shocked_price       — repriced option premium after the shock
        shocked_vol         — IV interpolated at the shocked moneyness
        initial_moneyness   — baseline strike / spot
        shocked_moneyness   — shocked strike / spot
        shocked_spot        — spot implied by the shocked moneyness
        absolute_gain       — (shocked_price - initial_price) * quantity
        return_pct          — absolute_gain / invested_value, or None if zero
        extrapolated        — whether smile lookup extrapolated
    """
    initial_moneyness = option.strike / market.spot
    shocked_moneyness = initial_moneyness * (1.0 + moneyness_shock_pct)
    if shocked_moneyness <= 0:
        raise ValueError("moneyness_shock_pct results in non-positive moneyness")

    baseline_price = (
        premium_paid
        if premium_paid is not None
        else black_scholes_price(option, market)["price"]
    )

    smile_result = interpolate_iv_from_smile(
        smile,
        shocked_moneyness,
        method=interp_method,
    )
    shocked_spot = option.strike / shocked_moneyness
    shocked_market = MarketState(
        spot=shocked_spot,
        risk_free_rate=market.risk_free_rate,
        dividend_yield=market.dividend_yield,
        implied_vol=market.implied_vol,
    )
    repriced = black_scholes_price(option, shocked_market, override_vol=smile_result["iv"])

    absolute_gain = (repriced["price"] - baseline_price) * option.quantity
    invested_value = baseline_price * option.quantity
    return_pct = (absolute_gain / invested_value) if invested_value != 0 else None

    return {
        "initial_price": baseline_price,
        "shocked_price": repriced["price"],
        "shocked_vol": smile_result["iv"],
        "initial_moneyness": initial_moneyness,
        "shocked_moneyness": shocked_moneyness,
        "shocked_spot": shocked_spot,
        "absolute_gain": absolute_gain,
        "return_pct": return_pct,
        "extrapolated": smile_result["extrapolated"],
    }


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
