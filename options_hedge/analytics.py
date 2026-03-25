"""
Analytics Layer — break-even, convexity metrics, payoff tables and summaries.

All functions are pure: no hidden state, no I/O.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

from .models import HedgeStrategy, MarketState, OptionSpec, SmileSurface
from .payoff import build_payoff_profile, build_strategy_payoff_profile


# ---------------------------------------------------------------------------
# Break-even
# ---------------------------------------------------------------------------

def compute_breakeven(
    option: OptionSpec,
    premium_paid: float,
) -> Dict[str, float]:
    """
    Compute the break-even spot price at expiry for a long option.

    Parameters
    ----------
    option:
        The option specification.
    premium_paid:
        Premium paid per unit at inception.

    Returns
    -------
    dict with keys:
        breakeven_spot  — spot level where net PnL = 0 at expiry
        premium_paid    — as provided
        breakeven_pct   — (breakeven_spot / option.strike - 1) * 100
    """
    K = option.strike
    if option.option_type == "call":
        breakeven_spot = K + premium_paid
    else:
        breakeven_spot = K - premium_paid

    breakeven_pct = (breakeven_spot / K - 1.0) * 100.0

    return {
        "breakeven_spot": breakeven_spot,
        "premium_paid": premium_paid,
        "breakeven_pct": breakeven_pct,
    }


def compute_strategy_breakeven(
    strategy: HedgeStrategy,
    leg_premiums: List[float],
    spot_range: List[float],
) -> Dict[str, object]:
    """
    Find break-even spot(s) for a multi-leg strategy by scanning the payoff profile.

    A break-even is any spot price at which net PnL changes sign.

    Parameters
    ----------
    strategy:
        HedgeStrategy.
    leg_premiums:
        Premiums per unit per leg.
    spot_range:
        Sorted list of spot prices to scan.

    Returns
    -------
    dict with keys:
        breakeven_spots     — list of approximate break-even spot levels
        payoff_profile      — full payoff profile (list of dicts)
    """
    profile = build_strategy_payoff_profile(strategy, leg_premiums, spot_range)
    breakevens: List[float] = []

    for i in range(1, len(profile)):
        pnl_prev = profile[i - 1]["total_net_pnl"]
        pnl_curr = profile[i]["total_net_pnl"]
        # Sign change → break-even between these two points
        if pnl_prev * pnl_curr <= 0 and pnl_curr != pnl_prev:
            t = pnl_prev / (pnl_prev - pnl_curr)
            be = profile[i - 1]["spot"] + t * (profile[i]["spot"] - profile[i - 1]["spot"])
            breakevens.append(round(be, 4))

    return {
        "breakeven_spots": breakevens,
        "payoff_profile": profile,
    }


# ---------------------------------------------------------------------------
# Convexity metrics
# ---------------------------------------------------------------------------

def compute_convexity_metrics(
    option: OptionSpec,
    market: MarketState,
    spot_range: List[float],
    premium_paid: float,
) -> Dict[str, float]:
    """
    Measure the convexity of a long option's payoff profile.

    Convexity here is defined as the second derivative of net PnL w.r.t. spot,
    approximated numerically at the current spot.

    Parameters
    ----------
    option:
        The option to analyse.
    market:
        Current market state (used only for the spot reference).
    spot_range:
        Sorted list of spot prices for the payoff scan.
    premium_paid:
        Premium paid at inception.

    Returns
    -------
    dict with keys:
        max_payoff          — maximum gross payoff across spot_range
        max_net_pnl         — maximum net PnL across spot_range
        max_return_pct      — maximum return-on-premium * 100 across spot_range
        convexity           — numerical second derivative at current spot
        spot_at_max_pnl     — spot level that gives the best net PnL
    """
    profile = build_payoff_profile(option, premium_paid, spot_range)

    max_gross = max(row["gross_payoff"] for row in profile)
    max_net = max(row["net_pnl"] for row in profile)
    max_rop = max(
        (row["return_on_premium"] for row in profile if row["return_on_premium"] is not None),
        default=None,
    )
    spot_at_max = max(profile, key=lambda r: r["net_pnl"])["spot"]

    # Numerical second derivative around current spot
    S = market.spot
    h = S * 0.01  # 1 % step

    def pnl_at(s: float) -> float:
        for row in profile:
            if abs(row["spot"] - s) < h * 0.5:
                return row["net_pnl"]
        # Extrapolate if needed
        K = option.strike
        if option.option_type == "call":
            gross = max(0.0, s - K) * option.quantity
        else:
            gross = max(0.0, K - s) * option.quantity
        return gross - premium_paid * option.quantity

    convexity = (pnl_at(S + h) - 2.0 * pnl_at(S) + pnl_at(S - h)) / (h ** 2)

    return {
        "max_payoff": max_gross,
        "max_net_pnl": max_net,
        "max_return_pct": max_rop * 100.0 if max_rop is not None else None,
        "convexity": convexity,
        "spot_at_max_pnl": spot_at_max,
    }


# ---------------------------------------------------------------------------
# Payoff table
# ---------------------------------------------------------------------------

def build_payoff_table(
    option: OptionSpec,
    premium_paid: float,
    spot_range: List[float],
    *,
    include_return: bool = True,
) -> List[Dict[str, object]]:
    """
    Build a formatted payoff table for a single option.

    Parameters
    ----------
    option:
        The option to tabulate.
    premium_paid:
        Premium paid per unit at inception.
    spot_range:
        Sorted list of spot prices.
    include_return:
        Whether to include the return_on_premium column.

    Returns
    -------
    List of row dicts with keys:
        spot, moneyness_pct, gross_payoff, net_pnl[, return_pct]
    """
    profile = build_payoff_profile(option, premium_paid, spot_range)
    table = []
    for row in profile:
        entry: Dict[str, object] = {
            "spot": row["spot"],
            "moneyness_pct": round((row["spot"] / option.strike - 1.0) * 100.0, 2),
            "gross_payoff": round(row["gross_payoff"], 4),
            "net_pnl": round(row["net_pnl"], 4),
        }
        if include_return:
            rop = row["return_on_premium"]
            entry["return_pct"] = round(rop * 100.0, 2) if rop is not None else None
        table.append(entry)
    return table


def build_strategy_payoff_table(
    strategy: HedgeStrategy,
    leg_premiums: List[float],
    spot_range: List[float],
    *,
    include_return: bool = True,
) -> List[Dict[str, object]]:
    """
    Build a formatted payoff table for a multi-leg strategy.

    Parameters
    ----------
    strategy:
        HedgeStrategy.
    leg_premiums:
        Premiums per unit per leg.
    spot_range:
        Sorted list of spot prices.
    include_return:
        Whether to include the return_on_premium column.

    Returns
    -------
    List of row dicts with keys:
        spot, moneyness_pct (vs first leg), total_gross_payoff, total_net_pnl[, return_pct]
    """
    profile = build_strategy_payoff_profile(strategy, leg_premiums, spot_range)
    first_strike = strategy.legs[0].strike
    table = []
    for row in profile:
        entry: Dict[str, object] = {
            "spot": row["spot"],
            "moneyness_pct": round((row["spot"] / first_strike - 1.0) * 100.0, 2),
            "total_gross_payoff": round(row["total_gross_payoff"], 4),
            "total_net_pnl": round(row["total_net_pnl"], 4),
        }
        if include_return:
            rop = row["return_on_premium"]
            entry["return_pct"] = round(rop * 100.0, 2) if rop is not None else None
        table.append(entry)
    return table


# ---------------------------------------------------------------------------
# Scenario summary
# ---------------------------------------------------------------------------

def summarise_grid_results(
    grid_results: List[Dict[str, object]],
) -> Dict[str, object]:
    """
    Summarise a list of scenario simulation results from ``simulate_strategy_grid``.

    Parameters
    ----------
    grid_results:
        Output of simulate_strategy_grid.

    Returns
    -------
    dict with keys:
        total_scenarios     — number of scenarios evaluated
        profitable_count    — scenarios where total_net_pnl > 0
        loss_count          — scenarios where total_net_pnl <= 0
        best_scenario       — result dict with highest total_net_pnl
        worst_scenario      — result dict with lowest total_net_pnl
        avg_return_pct      — mean return-on-premium (in %) across all scenarios
        avg_net_pnl         — mean net PnL
    """
    if not grid_results:
        return {}

    pnls = [float(r["total_net_pnl"]) for r in grid_results]
    rops = [
        float(r["return_on_premium"]) * 100.0
        for r in grid_results
        if r["return_on_premium"] is not None
    ]

    profitable = sum(1 for p in pnls if p > 0)

    return {
        "total_scenarios": len(grid_results),
        "profitable_count": profitable,
        "loss_count": len(grid_results) - profitable,
        "best_scenario": grid_results[pnls.index(max(pnls))],
        "worst_scenario": grid_results[pnls.index(min(pnls))],
        "avg_return_pct": sum(rops) / len(rops) if rops else None,
        "avg_net_pnl": sum(pnls) / len(pnls),
    }
