"""Tests for the Payoff Engine."""

import pytest

from options_hedge import (
    HedgeStrategy,
    MarketState,
    OptionSpec,
    build_payoff_profile,
    build_strategy_payoff_profile,
    compute_leg_payoff,
    compute_strategy_payoff,
)


@pytest.fixture
def call_100():
    return OptionSpec(option_type="call", strike=100.0, expiry_days=90, quantity=1.0)


@pytest.fixture
def put_90():
    return OptionSpec(option_type="put", strike=90.0, expiry_days=90, quantity=1.0)


# ---------------------------------------------------------------------------
# compute_leg_payoff
# ---------------------------------------------------------------------------

def test_call_itm_payoff():
    call = OptionSpec(option_type="call", strike=100.0, expiry_days=0)
    market = MarketState(spot=120.0, risk_free_rate=0.0)
    result = compute_leg_payoff(call, market, premium_paid=5.0)
    assert result["gross_payoff"] == pytest.approx(20.0)
    assert result["net_pnl"] == pytest.approx(15.0)
    assert result["return_on_premium"] == pytest.approx(3.0)


def test_call_otm_payoff():
    call = OptionSpec(option_type="call", strike=100.0, expiry_days=0)
    market = MarketState(spot=90.0, risk_free_rate=0.0)
    result = compute_leg_payoff(call, market, premium_paid=5.0)
    assert result["gross_payoff"] == pytest.approx(0.0)
    assert result["net_pnl"] == pytest.approx(-5.0)
    assert result["return_on_premium"] == pytest.approx(-1.0)


def test_put_itm_payoff():
    put = OptionSpec(option_type="put", strike=90.0, expiry_days=0)
    market = MarketState(spot=70.0, risk_free_rate=0.0)
    result = compute_leg_payoff(put, market, premium_paid=3.0)
    assert result["gross_payoff"] == pytest.approx(20.0)
    assert result["net_pnl"] == pytest.approx(17.0)


def test_put_otm_payoff():
    put = OptionSpec(option_type="put", strike=90.0, expiry_days=0)
    market = MarketState(spot=100.0, risk_free_rate=0.0)
    result = compute_leg_payoff(put, market, premium_paid=3.0)
    assert result["gross_payoff"] == pytest.approx(0.0)
    assert result["net_pnl"] == pytest.approx(-3.0)


def test_quantity_scaling():
    call = OptionSpec(option_type="call", strike=100.0, expiry_days=0, quantity=5.0)
    market = MarketState(spot=110.0, risk_free_rate=0.0)
    result = compute_leg_payoff(call, market, premium_paid=2.0)
    assert result["gross_payoff"] == pytest.approx(50.0)
    assert result["net_pnl"] == pytest.approx(40.0)


def test_zero_premium_rop_is_none():
    call = OptionSpec(option_type="call", strike=100.0, expiry_days=0)
    market = MarketState(spot=110.0, risk_free_rate=0.0)
    result = compute_leg_payoff(call, market, premium_paid=0.0)
    assert result["return_on_premium"] is None


# ---------------------------------------------------------------------------
# compute_strategy_payoff
# ---------------------------------------------------------------------------

def test_strategy_mismatched_premiums_raises():
    strategy = HedgeStrategy(
        legs=[OptionSpec(option_type="call", strike=100.0, expiry_days=0)],
        total_premium=5.0,
    )
    market = MarketState(spot=110.0, risk_free_rate=0.0)
    with pytest.raises(ValueError):
        compute_strategy_payoff(strategy, market, leg_premiums=[5.0, 3.0])


def test_strategy_two_legs(call_100, put_90):
    """Long call + long put (strangle) aggregation."""
    strategy = HedgeStrategy(
        legs=[call_100, put_90],
        total_premium=8.0,
    )
    market = MarketState(spot=120.0, risk_free_rate=0.0)
    result = compute_strategy_payoff(strategy, market, leg_premiums=[5.0, 3.0])
    # call pays 20, put pays 0
    assert result["total_gross_payoff"] == pytest.approx(20.0)
    assert result["total_net_pnl"] == pytest.approx(12.0)  # 20 - 8
    assert result["return_on_premium"] == pytest.approx(1.5)


# ---------------------------------------------------------------------------
# build_payoff_profile
# ---------------------------------------------------------------------------

def test_payoff_profile_length():
    call = OptionSpec(option_type="call", strike=100.0, expiry_days=0)
    spots = [80.0, 90.0, 100.0, 110.0, 120.0]
    profile = build_payoff_profile(call, premium_paid=5.0, spot_range=spots)
    assert len(profile) == len(spots)


def test_payoff_profile_values():
    call = OptionSpec(option_type="call", strike=100.0, expiry_days=0)
    spots = [90.0, 100.0, 110.0]
    profile = build_payoff_profile(call, premium_paid=5.0, spot_range=spots)
    assert profile[0]["net_pnl"] == pytest.approx(-5.0)
    assert profile[1]["net_pnl"] == pytest.approx(-5.0)
    assert profile[2]["net_pnl"] == pytest.approx(5.0)


def test_strategy_payoff_profile_consistency():
    """Strategy profile with one leg should match single leg profile."""
    call = OptionSpec(option_type="call", strike=100.0, expiry_days=0)
    strategy = HedgeStrategy(legs=[call], total_premium=5.0)
    spots = [85.0, 95.0, 105.0, 115.0]
    single = build_payoff_profile(call, premium_paid=5.0, spot_range=spots)
    multi = build_strategy_payoff_profile(strategy, leg_premiums=[5.0], spot_range=spots)
    for s, m in zip(single, multi):
        assert s["net_pnl"] == pytest.approx(m["total_net_pnl"])
