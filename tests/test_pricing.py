"""Tests for the Black-Scholes Pricing Engine."""

import math
import pytest

from options_hedge import (
    MarketState,
    OptionSpec,
    black_scholes_price,
    net_option_pnl,
    option_intrinsic_value,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def atm_call():
    return OptionSpec(option_type="call", strike=100.0, expiry_days=365)


@pytest.fixture
def atm_put():
    return OptionSpec(option_type="put", strike=100.0, expiry_days=365)


@pytest.fixture
def market():
    return MarketState(spot=100.0, risk_free_rate=0.05, dividend_yield=0.0, implied_vol=0.20)


# ---------------------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------------------

def test_invalid_strike():
    with pytest.raises(ValueError):
        OptionSpec(option_type="call", strike=-1.0, expiry_days=90)


def test_invalid_expiry():
    with pytest.raises(ValueError):
        OptionSpec(option_type="call", strike=100.0, expiry_days=-1)


def test_invalid_option_type():
    with pytest.raises(ValueError):
        OptionSpec(option_type="straddle", strike=100.0, expiry_days=90)


def test_invalid_spot():
    with pytest.raises(ValueError):
        MarketState(spot=0.0, risk_free_rate=0.05)


def test_invalid_vol():
    with pytest.raises(ValueError):
        MarketState(spot=100.0, risk_free_rate=0.05, implied_vol=-0.10)


# ---------------------------------------------------------------------------
# Black-Scholes correctness
# ---------------------------------------------------------------------------

def test_call_price_known_value(atm_call, market):
    """ATM call: BS price must be close to 10.45 (classic result for S=K=100, r=5%, sigma=20%, T=1y)."""
    result = black_scholes_price(atm_call, market)
    assert abs(result["price"] - 10.45) < 0.20, f"ATM call price unexpected: {result['price']}"


def test_put_price_put_call_parity(atm_call, atm_put, market):
    """Put-call parity: C - P = S * e^{-qT} - K * e^{-rT}."""
    call_result = black_scholes_price(atm_call, market)
    put_result = black_scholes_price(atm_put, market)

    S = market.spot
    K = atm_call.strike
    r = market.risk_free_rate
    q = market.dividend_yield
    T = atm_call.expiry_days / 365.0

    parity_lhs = call_result["price"] - put_result["price"]
    parity_rhs = S * math.exp(-q * T) - K * math.exp(-r * T)
    assert abs(parity_lhs - parity_rhs) < 1e-6, f"Put-call parity violated: {parity_lhs} != {parity_rhs}"


def test_call_delta_range(atm_call, market):
    result = black_scholes_price(atm_call, market)
    assert 0.0 <= result["delta"] <= 1.0


def test_put_delta_range(atm_put, market):
    result = black_scholes_price(atm_put, market)
    assert -1.0 <= result["delta"] <= 0.0


def test_gamma_positive(atm_call, market):
    result = black_scholes_price(atm_call, market)
    assert result["gamma"] >= 0.0


def test_vega_positive(atm_call, market):
    result = black_scholes_price(atm_call, market)
    assert result["vega"] >= 0.0


def test_theta_negative_for_atm_call(atm_call, market):
    """Theta should be negative for a long option position."""
    result = black_scholes_price(atm_call, market)
    assert result["theta"] < 0.0


def test_deep_itm_call_price(market):
    """Deep ITM call should be close to intrinsic value."""
    deep_itm_call = OptionSpec(option_type="call", strike=50.0, expiry_days=1)
    result = black_scholes_price(deep_itm_call, market)
    # S=100, K=50 → intrinsic = 50
    assert result["price"] >= 49.0


def test_deep_otm_call_near_zero(market):
    """Deep OTM call with 1 day to expiry should be nearly worthless."""
    deep_otm_call = OptionSpec(option_type="call", strike=300.0, expiry_days=1)
    result = black_scholes_price(deep_otm_call, market)
    assert result["price"] < 0.01


def test_zero_expiry_returns_intrinsic(market):
    """At expiry the option price equals its intrinsic value."""
    call = OptionSpec(option_type="call", strike=90.0, expiry_days=0)
    result = black_scholes_price(call, market)
    assert abs(result["price"] - 10.0) < 1e-6  # max(100 - 90, 0)


def test_override_vol(atm_call, market):
    """override_vol should change the output price."""
    base = black_scholes_price(atm_call, market)
    high_vol = black_scholes_price(atm_call, market, override_vol=0.40)
    assert high_vol["price"] > base["price"]


# ---------------------------------------------------------------------------
# Intrinsic value tests
# ---------------------------------------------------------------------------

def test_intrinsic_itm_call(market):
    call = OptionSpec(option_type="call", strike=90.0, expiry_days=30)
    result = option_intrinsic_value(call, market)
    assert result["intrinsic_value"] == pytest.approx(10.0)
    assert result["itm"] is True


def test_intrinsic_otm_put(market):
    put = OptionSpec(option_type="put", strike=90.0, expiry_days=30)
    result = option_intrinsic_value(put, market)
    assert result["intrinsic_value"] == pytest.approx(0.0)
    assert result["itm"] is False


def test_moneyness(market):
    call = OptionSpec(option_type="call", strike=110.0, expiry_days=30)
    result = option_intrinsic_value(call, market)
    assert result["moneyness"] == pytest.approx(100.0 / 110.0)


# ---------------------------------------------------------------------------
# Net PnL tests
# ---------------------------------------------------------------------------

def test_net_pnl_atm_call(atm_call, market):
    """Long ATM call: if market moves up 20 %, net PnL should be positive."""
    premium = black_scholes_price(atm_call, market)["price"]
    new_market = MarketState(spot=120.0, risk_free_rate=0.05, implied_vol=0.20)
    result = net_option_pnl(atm_call, new_market, premium)
    assert result["net_pnl"] > 0


def test_net_pnl_loss_when_unchanged(atm_call, market):
    """After time decay with spot unchanged the option loses value (theta)."""
    premium = black_scholes_price(atm_call, market)["price"]
    # Option almost at expiry
    near_expiry_call = OptionSpec(option_type="call", strike=100.0, expiry_days=1)
    result = net_option_pnl(near_expiry_call, market, premium)
    assert result["net_pnl"] < 0


def test_return_on_premium_is_none_for_zero_premium(atm_call, market):
    result = net_option_pnl(atm_call, market, premium_paid=0.0)
    assert result["return_on_premium"] is None
