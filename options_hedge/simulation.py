"""
Simulation Master — orchestrates full price × vol × time × strikes grids.

Combines all engines into a single simulation entry point.
All functions are pure: no hidden state, no I/O.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from .models import (
    HedgeStrategy,
    MarketState,
    OptionSpec,
    ScenarioSpec,
    SimulationResult,
    SmileSurface,
)
from .pricing import black_scholes_price, net_option_pnl, option_intrinsic_value
from .scenarios import apply_scenario, build_scenario_grid
from .smile import price_option_from_moneyness
from .theta import apply_time_decay


# ---------------------------------------------------------------------------
# Single-option simulation
# ---------------------------------------------------------------------------

def simulate_single_hedge(
    option: OptionSpec,
    market: MarketState,
    scenario: ScenarioSpec,
    premium_paid: float,
    smile: Optional[SmileSurface] = None,
    *,
    interp_method: str = "cubic",
) -> SimulationResult:
    """
    Simulate the value and P&L of a single option under a given scenario.

    Parameters
    ----------
    option:
        Original option at inception.
    market:
        Baseline market state at inception.
    scenario:
        Stress scenario to apply (spot shock, vol shock, elapsed days).
    premium_paid:
        Price paid per unit at inception.
    smile:
        Optional SmileSurface for vol interpolation in the shocked state.
    interp_method:
        Interpolation method when a smile is used.

    Returns
    -------
    SimulationResult populated with price, payoff, PnL and greeks.
    """
    # Apply time decay to the option
    aged_option = apply_time_decay(option, scenario.elapsed_days)

    # Apply scenario to the market
    shocked_market = apply_scenario(market, scenario)

    # Re-price
    if smile is not None:
        bs = price_option_from_moneyness(aged_option, shocked_market, smile, method=interp_method)
    else:
        bs = black_scholes_price(aged_option, shocked_market)

    intrinsic_val = option_intrinsic_value(aged_option, shocked_market)["intrinsic_value"]
    current_price = bs["price"]
    qty = option.quantity

    net_pnl = (current_price - premium_paid) * qty
    total_premium = premium_paid * qty
    rop = (net_pnl / total_premium) if total_premium != 0 else None

    return SimulationResult(
        option_price=current_price,
        intrinsic_value=intrinsic_val,
        payoff=intrinsic_val * qty,
        net_pnl=net_pnl,
        return_on_premium=rop,
        delta=bs.get("delta"),
        gamma=bs.get("gamma"),
        theta=bs.get("theta"),
        vega=bs.get("vega"),
        scenario=scenario,
        metadata={
            "shocked_spot": shocked_market.spot,
            "shocked_vol": shocked_market.implied_vol,
            "remaining_days": aged_option.expiry_days,
            "premium_paid": premium_paid,
        },
    )


# ---------------------------------------------------------------------------
# Strategy-level simulation
# ---------------------------------------------------------------------------

def simulate_strategy(
    strategy: HedgeStrategy,
    market: MarketState,
    scenario: ScenarioSpec,
    leg_premiums: List[float],
    smile: Optional[SmileSurface] = None,
    *,
    interp_method: str = "cubic",
) -> Dict[str, object]:
    """
    Simulate a full multi-leg hedge strategy under a given scenario.

    Parameters
    ----------
    strategy:
        HedgeStrategy with one or more option legs.
    market:
        Baseline market state at inception.
    scenario:
        Stress scenario.
    leg_premiums:
        Premiums paid per unit, one per leg (same order as strategy.legs).
    smile:
        Optional SmileSurface.
    interp_method:
        Interpolation method.

    Returns
    -------
    dict with keys:
        total_option_value  — sum of repriced option values
        total_net_pnl       — total_option_value - strategy.total_premium
        return_on_premium   — total_net_pnl / strategy.total_premium
        leg_results         — list of SimulationResult per leg
        scenario            — the applied ScenarioSpec
        shocked_spot        — post-scenario spot
        shocked_vol         — post-scenario vol (from first leg's shocked market)
    """
    if len(leg_premiums) != len(strategy.legs):
        raise ValueError("leg_premiums must have same length as strategy.legs")

    leg_results = []
    total_option_value = 0.0

    for option, premium in zip(strategy.legs, leg_premiums):
        result = simulate_single_hedge(
            option,
            market,
            scenario,
            premium,
            smile,
            interp_method=interp_method,
        )
        leg_results.append(result)
        total_option_value += result.option_price * option.quantity

    total_net_pnl = total_option_value - strategy.total_premium
    rop = (total_net_pnl / strategy.total_premium) if strategy.total_premium != 0 else None

    shocked_market = apply_scenario(market, scenario)

    return {
        "total_option_value": total_option_value,
        "total_net_pnl": total_net_pnl,
        "return_on_premium": rop,
        "leg_results": leg_results,
        "scenario": scenario,
        "shocked_spot": shocked_market.spot,
        "shocked_vol": shocked_market.implied_vol,
    }


# ---------------------------------------------------------------------------
# Full grid simulation
# ---------------------------------------------------------------------------

def simulate_strategy_grid(
    strategy: HedgeStrategy,
    market: MarketState,
    spot_shocks: List[float],
    vol_shocks: List[float],
    time_steps: List[int],
    leg_premiums: List[float],
    smile: Optional[SmileSurface] = None,
    *,
    interp_method: str = "cubic",
) -> List[Dict[str, object]]:
    """
    Run the strategy simulation over the full cartesian product of
    spot × vol × time scenarios.

    Parameters
    ----------
    strategy:
        HedgeStrategy to evaluate.
    market:
        Baseline market state.
    spot_shocks:
        Fractional spot shocks, e.g. [-0.40, -0.20, 0.0, 0.20].
    vol_shocks:
        Absolute vol shocks, e.g. [-0.05, 0.0, 0.10, 0.20].
    time_steps:
        Elapsed days, e.g. [0, 30, 60, 90].
    leg_premiums:
        Premiums paid per unit, one per leg.
    smile:
        Optional SmileSurface.
    interp_method:
        Interpolation method.

    Returns
    -------
    List of simulation result dicts, one per scenario.
    """
    scenarios = build_scenario_grid(spot_shocks, vol_shocks, time_steps)
    return [
        simulate_strategy(
            strategy, market, sc, leg_premiums, smile, interp_method=interp_method
        )
        for sc in scenarios
    ]
