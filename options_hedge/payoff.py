"""
Payoff Engine — gross payoff, premium cost, net PnL and return on premium.

Handles single legs and multi-leg strategies (HedgeStrategy).
All functions are pure: no hidden state, no I/O.
"""

from __future__ import annotations

from typing import Dict, List

from .models import HedgeStrategy, MarketState, OptionSpec, SimulationResult


def compute_leg_payoff(
    option: OptionSpec,
    market: MarketState,
    premium_paid: float,
) -> Dict[str, float]:
    """
    Compute the payoff of a single option leg at expiry (intrinsic only).

    Parameters
    ----------
    option:
        The option specification.
    market:
        MarketState at expiry (spot at expiry).
    premium_paid:
        Price paid per unit at inception.

    Returns
    -------
    dict with keys:
        gross_payoff        — max(0, S-K) or max(0, K-S) scaled by quantity
        net_pnl             — gross_payoff - premium_paid * quantity
        return_on_premium   — net_pnl / (premium_paid * quantity) or None
        quantity            — number of contracts
        strike              — option strike
        spot_at_expiry      — market.spot
    """
    S = market.spot
    K = option.strike
    qty = option.quantity

    if option.option_type == "call":
        gross = max(0.0, S - K) * qty
    else:
        gross = max(0.0, K - S) * qty

    net_pnl = gross - premium_paid * qty
    total_premium = premium_paid * qty
    rop = (net_pnl / total_premium) if total_premium != 0 else None

    return {
        "gross_payoff": gross,
        "net_pnl": net_pnl,
        "return_on_premium": rop,
        "quantity": qty,
        "strike": K,
        "spot_at_expiry": S,
    }


def compute_strategy_payoff(
    strategy: HedgeStrategy,
    market: MarketState,
    leg_premiums: List[float],
) -> Dict[str, object]:
    """
    Compute the aggregate payoff of a multi-leg hedge strategy at expiry.

    Parameters
    ----------
    strategy:
        HedgeStrategy with one or more option legs.
    market:
        MarketState at expiry.
    leg_premiums:
        List of premiums paid per unit, one per leg (same order as strategy.legs).

    Returns
    -------
    dict with keys:
        total_gross_payoff  — sum of gross payoffs across legs
        total_net_pnl       — total_gross_payoff - strategy.total_premium
        return_on_premium   — total_net_pnl / strategy.total_premium or None
        leg_results         — list of per-leg dicts from compute_leg_payoff
        spot_at_expiry      — market.spot
    """
    if len(leg_premiums) != len(strategy.legs):
        raise ValueError("leg_premiums must have same length as strategy.legs")

    leg_results = []
    total_gross = 0.0

    for option, premium in zip(strategy.legs, leg_premiums):
        result = compute_leg_payoff(option, market, premium)
        leg_results.append(result)
        total_gross += result["gross_payoff"]

    total_premium = strategy.total_premium
    total_net_pnl = total_gross - total_premium
    rop = (total_net_pnl / total_premium) if total_premium != 0 else None

    return {
        "total_gross_payoff": total_gross,
        "total_net_pnl": total_net_pnl,
        "return_on_premium": rop,
        "leg_results": leg_results,
        "spot_at_expiry": market.spot,
    }


def build_payoff_profile(
    option: OptionSpec,
    premium_paid: float,
    spot_range: List[float],
) -> List[Dict[str, float]]:
    """
    Build the full payoff profile of a single option across a range of spot prices.

    Parameters
    ----------
    option:
        The option to profile.
    premium_paid:
        Premium paid per unit at inception.
    spot_range:
        List of spot prices to evaluate at expiry.

    Returns
    -------
    List of dicts, each containing:
        spot, gross_payoff, net_pnl, return_on_premium
    """
    profile = []
    for S in spot_range:
        dummy_market = MarketState(
            spot=S,
            risk_free_rate=0.0,
            dividend_yield=0.0,
            implied_vol=0.0,
        )
        row = compute_leg_payoff(option, dummy_market, premium_paid)
        profile.append({
            "spot": S,
            "gross_payoff": row["gross_payoff"],
            "net_pnl": row["net_pnl"],
            "return_on_premium": row["return_on_premium"],
        })
    return profile


def build_strategy_payoff_profile(
    strategy: HedgeStrategy,
    leg_premiums: List[float],
    spot_range: List[float],
) -> List[Dict[str, object]]:
    """
    Build the payoff profile of a multi-leg strategy across a range of spot prices.

    Parameters
    ----------
    strategy:
        HedgeStrategy with one or more option legs.
    leg_premiums:
        Premiums paid per unit, one per leg.
    spot_range:
        List of spot prices to evaluate at expiry.

    Returns
    -------
    List of dicts, each containing:
        spot, total_gross_payoff, total_net_pnl, return_on_premium
    """
    profile = []
    for S in spot_range:
        dummy_market = MarketState(
            spot=S,
            risk_free_rate=0.0,
            dividend_yield=0.0,
            implied_vol=0.0,
        )
        row = compute_strategy_payoff(strategy, dummy_market, leg_premiums)
        profile.append({
            "spot": S,
            "total_gross_payoff": row["total_gross_payoff"],
            "total_net_pnl": row["total_net_pnl"],
            "return_on_premium": row["return_on_premium"],
        })
    return profile
