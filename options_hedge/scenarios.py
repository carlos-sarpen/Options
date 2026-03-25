"""
Scenario Generator — price, vol and time-grid scenarios.

Generates ScenarioSpec objects for use by the simulation master.
All functions are pure: no hidden state, no I/O.
"""

from __future__ import annotations

from typing import Dict, List

from .models import MarketState, ScenarioSpec


def generate_price_scenarios(
    base_spot: float,
    shock_pcts: List[float],
) -> List[Dict[str, float]]:
    """
    Generate a list of spot-price scenarios from a list of shock percentages.

    Parameters
    ----------
    base_spot:
        Baseline spot price (e.g. Ibovespa index level).
    shock_pcts:
        List of fractional shocks, e.g. [-0.30, -0.20, -0.10, 0.0, 0.10, 0.20].

    Returns
    -------
    List of dicts with keys:
        shock_pct       — the applied shock
        shocked_spot    — base_spot * (1 + shock_pct)
    """
    return [
        {"shock_pct": pct, "shocked_spot": base_spot * (1.0 + pct)}
        for pct in shock_pcts
    ]


def generate_vol_scenarios(
    base_vol: float,
    shock_abs: List[float],
) -> List[Dict[str, float]]:
    """
    Generate a list of vol scenarios from a list of absolute vol shocks.

    Parameters
    ----------
    base_vol:
        Baseline implied volatility (annualised).
    shock_abs:
        List of absolute vol additions, e.g. [-0.05, 0.0, 0.10, 0.20].

    Returns
    -------
    List of dicts with keys:
        shock_abs       — the applied absolute shock
        shocked_vol     — max(0.001, base_vol + shock_abs)
    """
    return [
        {"shock_abs": shock, "shocked_vol": max(0.001, base_vol + shock)}
        for shock in shock_abs
    ]


def generate_time_grid(
    total_days: int,
    steps: int,
) -> List[Dict[str, int]]:
    """
    Generate a uniform time grid from inception to expiry.

    Parameters
    ----------
    total_days:
        Total calendar days to maturity.
    steps:
        Number of equally-spaced time steps.

    Returns
    -------
    List of dicts with keys:
        elapsed_days    — days since inception
        remaining_days  — total_days - elapsed_days
    """
    if steps <= 0:
        raise ValueError("steps must be positive")
    interval = total_days / steps
    result = []
    for i in range(steps + 1):
        elapsed = round(i * interval)
        result.append({
            "elapsed_days": elapsed,
            "remaining_days": max(0, total_days - elapsed),
        })
    return result


def build_scenario_grid(
    spot_shocks: List[float],
    vol_shocks: List[float],
    time_steps: List[int],
) -> List[ScenarioSpec]:
    """
    Cartesian product of spot shocks × vol shocks × time steps.

    Parameters
    ----------
    spot_shocks:
        Fractional spot shocks, e.g. [-0.30, -0.20, …, 0.20].
    vol_shocks:
        Absolute vol shocks, e.g. [-0.05, 0.0, 0.10].
    time_steps:
        Elapsed days, e.g. [0, 30, 60, 90].

    Returns
    -------
    List of ScenarioSpec, one per combination.
    """
    scenarios: List[ScenarioSpec] = []
    for s in spot_shocks:
        for v in vol_shocks:
            for t in time_steps:
                scenarios.append(
                    ScenarioSpec(
                        spot_shock_pct=s,
                        vol_shock_abs=v,
                        elapsed_days=t,
                    )
                )
    return scenarios


def apply_scenario(market: MarketState, scenario: ScenarioSpec) -> MarketState:
    """
    Apply a ScenarioSpec to a base MarketState, returning a new MarketState.

    The ``expiry_days`` of options must be adjusted separately (see
    ``theta.apply_time_decay``).

    Parameters
    ----------
    market:
        Baseline market state.
    scenario:
        The scenario to apply.

    Returns
    -------
    New MarketState with shocked spot and vol.
    """
    new_spot = market.spot * (1.0 + scenario.spot_shock_pct)
    new_vol = max(0.001, market.implied_vol + scenario.vol_shock_abs)
    return MarketState(
        spot=new_spot,
        risk_free_rate=market.risk_free_rate,
        dividend_yield=market.dividend_yield,
        implied_vol=new_vol,
    )


def named_stress_scenarios() -> List[Dict[str, object]]:
    """
    Return a catalogue of well-known historical stress scenarios for Brazil/Ibovespa.

    Each entry is a dict with keys:
        label           — human-readable name
        spot_shock_pct  — approximate index move
        vol_shock_abs   — approximate vol spike (annualised)
        elapsed_days    — 0 (instantaneous scenario)

    Scenarios covered: 2008 Lehman, 2020 Covid, Dilma re-election 2014,
    2018 truckers' strike, Lula election 2022.
    """
    return [
        {
            "label": "2008 Lehman (peak-to-trough ~-60 %)",
            "spot_shock_pct": -0.60,
            "vol_shock_abs": 0.40,
            "elapsed_days": 0,
        },
        {
            "label": "2020 Covid (peak-to-trough ~-45 %)",
            "spot_shock_pct": -0.45,
            "vol_shock_abs": 0.35,
            "elapsed_days": 0,
        },
        {
            "label": "2014 Dilma re-election shock (~-10 %)",
            "spot_shock_pct": -0.10,
            "vol_shock_abs": 0.10,
            "elapsed_days": 0,
        },
        {
            "label": "2018 Truckers' strike (~-5 %)",
            "spot_shock_pct": -0.05,
            "vol_shock_abs": 0.05,
            "elapsed_days": 0,
        },
        {
            "label": "2022 Lula election (~-5 %)",
            "spot_shock_pct": -0.05,
            "vol_shock_abs": 0.05,
            "elapsed_days": 0,
        },
        {
            "label": "Generic 5 % daily drop",
            "spot_shock_pct": -0.05,
            "vol_shock_abs": 0.05,
            "elapsed_days": 0,
        },
    ]
