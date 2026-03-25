"""
Theta Engine — time decay and option re-pricing as time elapses.

All functions are pure: no hidden state, no I/O.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional

from .models import MarketState, OptionSpec, SmileSurface
from .pricing import black_scholes_price
from .smile import price_option_from_moneyness


def apply_time_decay(
    option: OptionSpec,
    elapsed_days: int,
) -> OptionSpec:
    """
    Return a new OptionSpec with the time-to-expiry reduced by ``elapsed_days``.

    Parameters
    ----------
    option:
        Original option spec.
    elapsed_days:
        Number of calendar days that have passed.

    Returns
    -------
    New OptionSpec with ``expiry_days = max(0, option.expiry_days - elapsed_days)``.
    """
    new_expiry = max(0, option.expiry_days - elapsed_days)
    return OptionSpec(
        option_type=option.option_type,
        strike=option.strike,
        expiry_days=new_expiry,
        quantity=option.quantity,
        moneyness=option.moneyness,
    )


def price_over_time(
    option: OptionSpec,
    market: MarketState,
    time_grid: List[int],
    smile: Optional[SmileSurface] = None,
    *,
    interp_method: str = "cubic",
) -> List[Dict[str, float]]:
    """
    Price an option at multiple points in time along a time grid.

    Parameters
    ----------
    option:
        Option at inception (full expiry_days).
    market:
        Market state (assumed static — spot and vol do not move).
    time_grid:
        List of elapsed days since inception to evaluate at.
    smile:
        If provided, the smile is used instead of market.implied_vol.
    interp_method:
        Interpolation method when a smile is used.

    Returns
    -------
    List of dicts with keys:
        elapsed_days, remaining_days, option_price, theta_daily,
        time_value, intrinsic_value
    """
    from .pricing import option_intrinsic_value  # local import

    rows = []
    for elapsed in time_grid:
        aged_option = apply_time_decay(option, elapsed)
        if smile is not None:
            bs = price_option_from_moneyness(aged_option, market, smile, method=interp_method)
        else:
            bs = black_scholes_price(aged_option, market)

        intrinsic = option_intrinsic_value(aged_option, market)["intrinsic_value"]
        time_val = max(0.0, bs["price"] - intrinsic)

        rows.append(
            {
                "elapsed_days": elapsed,
                "remaining_days": aged_option.expiry_days,
                "option_price": bs["price"],
                "theta_daily": bs["theta"],
                "time_value": time_val,
                "intrinsic_value": intrinsic,
            }
        )
    return rows


def compute_theta_profile(
    option: OptionSpec,
    market: MarketState,
    total_days: int,
    steps: int = 30,
    smile: Optional[SmileSurface] = None,
) -> List[Dict[str, float]]:
    """
    Compute the daily theta (time decay) at every step from inception to expiry.

    Parameters
    ----------
    option:
        Option at inception.
    market:
        Market state (static).
    total_days:
        Total calendar days to expiry.
    steps:
        Number of evaluation points.
    smile:
        Optional SmileSurface.

    Returns
    -------
    List of dicts with keys:
        elapsed_days, remaining_days, option_price, theta_daily
    """
    from .scenarios import generate_time_grid

    grid = generate_time_grid(total_days, steps)
    elapsed_list = [g["elapsed_days"] for g in grid]
    profile = price_over_time(option, market, elapsed_list, smile)
    return [
        {
            "elapsed_days": row["elapsed_days"],
            "remaining_days": row["remaining_days"],
            "option_price": row["option_price"],
            "theta_daily": row["theta_daily"],
        }
        for row in profile
    ]
