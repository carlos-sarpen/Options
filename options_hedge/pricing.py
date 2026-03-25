"""
Pricing Engine — Black-Scholes pricing and auxiliary functions.

All functions are pure: no hidden state, no I/O.
"""

from __future__ import annotations

import math
from typing import Dict

from .models import MarketState, OptionSpec


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _d1(spot: float, strike: float, T: float, r: float, q: float, sigma: float) -> float:
    """Compute d1 from the Black-Scholes formula."""
    if T <= 0:
        return math.inf if spot >= strike else -math.inf
    return (math.log(spot / strike) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))


def _d2(d1_val: float, sigma: float, T: float) -> float:
    """Compute d2 = d1 - sigma * sqrt(T)."""
    if T <= 0:
        return d1_val
    return d1_val - sigma * math.sqrt(T)


def _norm_cdf(x: float) -> float:
    """Standard normal cumulative distribution function."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_pdf(x: float) -> float:
    """Standard normal probability density function."""
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def black_scholes_price(
    option: OptionSpec,
    market: MarketState,
    *,
    override_vol: float | None = None,
) -> Dict[str, float]:
    """
    Price a European option using the Black-Scholes-Merton model.

    Parameters
    ----------
    option:
        OptionSpec with strike, expiry_days and option_type.
    market:
        MarketState with spot, risk_free_rate, dividend_yield and implied_vol.
    override_vol:
        When provided, uses this vol instead of market.implied_vol.

    Returns
    -------
    dict with keys:
        price       — option fair value
        d1, d2      — intermediate values
        delta       — first derivative w.r.t. spot
        gamma       — second derivative w.r.t. spot
        theta       — daily time decay (in price units per calendar day)
        vega        — sensitivity to 1-pp change in vol
        rho         — sensitivity to 1-pp change in rate
    """
    S = market.spot
    K = option.strike
    r = market.risk_free_rate
    q = market.dividend_yield
    sigma = override_vol if override_vol is not None else market.implied_vol
    T = option.expiry_days / 365.0

    if sigma <= 0 or T <= 0:
        price = option_intrinsic_value(option, market)["intrinsic_value"]
        return {
            "price": price,
            "d1": float("nan"),
            "d2": float("nan"),
            "delta": 1.0 if option.option_type == "call" and S > K else (
                -1.0 if option.option_type == "put" and S < K else 0.0
            ),
            "gamma": 0.0,
            "theta": 0.0,
            "vega": 0.0,
            "rho": 0.0,
        }

    d1 = _d1(S, K, T, r, q, sigma)
    d2 = _d2(d1, sigma, T)

    discount_r = math.exp(-r * T)
    discount_q = math.exp(-q * T)

    if option.option_type == "call":
        price = S * discount_q * _norm_cdf(d1) - K * discount_r * _norm_cdf(d2)
        delta = discount_q * _norm_cdf(d1)
        rho = K * T * discount_r * _norm_cdf(d2) / 100.0
    else:
        price = K * discount_r * _norm_cdf(-d2) - S * discount_q * _norm_cdf(-d1)
        delta = -discount_q * _norm_cdf(-d1)
        rho = -K * T * discount_r * _norm_cdf(-d2) / 100.0

    gamma = (discount_q * _norm_pdf(d1)) / (S * sigma * math.sqrt(T))

    # Theta expressed as daily decay (divided by 365)
    theta_annual = (
        -(S * discount_q * _norm_pdf(d1) * sigma) / (2.0 * math.sqrt(T))
        + (
            -r * K * discount_r * _norm_cdf(d2)
            + q * S * discount_q * _norm_cdf(d1)
            if option.option_type == "call"
            else r * K * discount_r * _norm_cdf(-d2)
            - q * S * discount_q * _norm_cdf(-d1)
        )
    )
    theta_daily = theta_annual / 365.0

    # Vega expressed per 1-pp move in vol
    vega = S * discount_q * _norm_pdf(d1) * math.sqrt(T) / 100.0

    return {
        "price": max(price, 0.0),
        "d1": d1,
        "d2": d2,
        "delta": delta,
        "gamma": gamma,
        "theta": theta_daily,
        "vega": vega,
        "rho": rho,
    }


def option_intrinsic_value(option: OptionSpec, market: MarketState) -> Dict[str, float]:
    """
    Compute the intrinsic (exercise) value of an option.

    Returns
    -------
    dict with keys:
        intrinsic_value — max(0, S - K) for call, max(0, K - S) for put
        moneyness       — spot / strike ratio
        itm             — True if option is in the money
    """
    S = market.spot
    K = option.strike
    if option.option_type == "call":
        intrinsic = max(0.0, S - K)
    else:
        intrinsic = max(0.0, K - S)

    moneyness = S / K
    itm = intrinsic > 0.0

    return {
        "intrinsic_value": intrinsic,
        "moneyness": moneyness,
        "itm": itm,
    }


def net_option_pnl(
    option: OptionSpec,
    market: MarketState,
    premium_paid: float,
    *,
    override_vol: float | None = None,
) -> Dict[str, float]:
    """
    Compute the net P&L for a long option position.

    Parameters
    ----------
    option:
        The option specification.
    market:
        Current market state (post-scenario).
    premium_paid:
        Price paid per unit at inception (positive for a debit).
    override_vol:
        When provided, uses this vol for re-pricing.

    Returns
    -------
    dict with keys:
        current_price       — current Black-Scholes price
        premium_paid        — original premium
        gross_payoff        — intrinsic value at current spot
        net_pnl             — current_price * quantity - premium_paid * quantity
        return_on_premium   — net_pnl / (premium_paid * quantity) or None if premium == 0
        quantity            — option.quantity
    """
    bs = black_scholes_price(option, market, override_vol=override_vol)
    current_price = bs["price"]
    intrinsic = option_intrinsic_value(option, market)["intrinsic_value"]
    quantity = option.quantity

    net_pnl = (current_price - premium_paid) * quantity
    return_on_premium = (net_pnl / (premium_paid * quantity)) if premium_paid != 0 else None

    return {
        "current_price": current_price,
        "premium_paid": premium_paid,
        "gross_payoff": intrinsic * quantity,
        "net_pnl": net_pnl,
        "return_on_premium": return_on_premium,
        "quantity": quantity,
    }
