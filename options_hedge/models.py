"""
Data models for the options hedge simulation engine.

All objects are pure dataclasses — no hidden state, no side effects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional


OptionType = Literal["call", "put"]


@dataclass
class OptionSpec:
    """Specification of a single option contract."""

    option_type: OptionType
    strike: float
    expiry_days: int
    quantity: float = 1.0
    moneyness: Optional[float] = None  # K / S; computed externally if needed

    def __post_init__(self) -> None:
        if self.strike <= 0:
            raise ValueError("strike must be positive")
        if self.expiry_days < 0:
            raise ValueError("expiry_days must be non-negative")
        if self.option_type not in ("call", "put"):
            raise ValueError("option_type must be 'call' or 'put'")


@dataclass
class MarketState:
    """Current market state used as input to every pricing function."""

    spot: float
    risk_free_rate: float
    dividend_yield: float = 0.0
    implied_vol: float = 0.20

    def __post_init__(self) -> None:
        if self.spot <= 0:
            raise ValueError("spot must be positive")
        if self.implied_vol < 0:
            raise ValueError("implied_vol must be non-negative")


@dataclass
class SmileSurface:
    """
    Implied-volatility smile indexed by moneyness (K / S).

    Keys are moneyness values (float), values are annualised implied vols.
    Example: {0.90: 0.25, 0.95: 0.22, 1.00: 0.20, 1.05: 0.21, 1.10: 0.23}
    """

    points: Dict[float, float]

    def __post_init__(self) -> None:
        if len(self.points) < 2:
            raise ValueError("SmileSurface requires at least 2 data points")
        for m, iv in self.points.items():
            if iv < 0:
                raise ValueError(f"Negative IV at moneyness {m}")


@dataclass
class ScenarioSpec:
    """Definition of a single stress scenario."""

    spot_shock_pct: float = 0.0      # e.g. -0.10 means spot drops 10 %
    vol_shock_abs: float = 0.0       # e.g. +0.05 means IV rises 5 pp
    elapsed_days: int = 0            # calendar days elapsed since inception


@dataclass
class HedgeStrategy:
    """
    A collection of option legs forming one hedge strategy.

    total_premium is the net cash paid (positive = debit) at inception.
    """

    legs: List[OptionSpec]
    total_premium: float = 0.0
    label: str = ""

    def __post_init__(self) -> None:
        if not self.legs:
            raise ValueError("HedgeStrategy must have at least one leg")


@dataclass
class SimulationResult:
    """Output of a single option or strategy simulation."""

    option_price: float
    intrinsic_value: float
    payoff: float
    net_pnl: float
    return_on_premium: Optional[float]
    delta: Optional[float] = None
    gamma: Optional[float] = None
    theta: Optional[float] = None
    vega: Optional[float] = None
    scenario: Optional[ScenarioSpec] = None
    metadata: Dict[str, object] = field(default_factory=dict)
