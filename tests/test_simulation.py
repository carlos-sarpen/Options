"""Tests for the Simulation Master and Analytics Layer."""

import pytest

from options_hedge import (
    HedgeStrategy,
    MarketState,
    OptionSpec,
    ScenarioSpec,
    SimulationResult,
    SmileSurface,
    build_payoff_table,
    build_strategy_payoff_table,
    build_smile_from_atm_skew,
    compute_breakeven,
    compute_convexity_metrics,
    compute_strategy_breakeven,
    compute_theta_profile,
    price_over_time,
    simulate_single_hedge,
    simulate_strategy,
    simulate_strategy_grid,
    summarise_grid_results,
)


@pytest.fixture
def market():
    return MarketState(spot=100_000.0, risk_free_rate=0.13, dividend_yield=0.0, implied_vol=0.25)


@pytest.fixture
def atm_put(market):
    return OptionSpec(option_type="put", strike=market.spot, expiry_days=90, quantity=1.0)


@pytest.fixture
def atm_call(market):
    return OptionSpec(option_type="call", strike=market.spot, expiry_days=90, quantity=1.0)


@pytest.fixture
def smile():
    return build_smile_from_atm_skew(atm_vol=0.25, skew_per_delta=0.03)


# ---------------------------------------------------------------------------
# simulate_single_hedge
# ---------------------------------------------------------------------------

def test_simulate_single_hedge_returns_simulation_result(market, atm_put):
    scenario = ScenarioSpec(spot_shock_pct=-0.10, vol_shock_abs=0.10, elapsed_days=30)
    result = simulate_single_hedge(atm_put, market, scenario, premium_paid=3000.0)
    assert isinstance(result, SimulationResult)


def test_simulate_put_hedge_crash_pnl(market, atm_put):
    """A long put should make money in a -20 % crash with a vol spike."""
    scenario = ScenarioSpec(spot_shock_pct=-0.20, vol_shock_abs=0.20, elapsed_days=0)
    result = simulate_single_hedge(atm_put, market, scenario, premium_paid=3000.0)
    assert result.net_pnl > 0


def test_simulate_put_hedge_rally_loss(market, atm_put):
    """A long put in a +20 % rally with vol dropping should lose value."""
    premium = 3000.0
    scenario = ScenarioSpec(spot_shock_pct=0.20, vol_shock_abs=-0.05, elapsed_days=0)
    result = simulate_single_hedge(atm_put, market, scenario, premium_paid=premium)
    assert result.net_pnl < 0


def test_simulate_with_smile(market, atm_put, smile):
    scenario = ScenarioSpec(spot_shock_pct=-0.10, vol_shock_abs=0.05, elapsed_days=0)
    result = simulate_single_hedge(atm_put, market, scenario, premium_paid=3000.0, smile=smile)
    assert result.option_price > 0


# ---------------------------------------------------------------------------
# simulate_strategy
# ---------------------------------------------------------------------------

def test_simulate_strategy_two_legs(market, atm_call, atm_put):
    strategy = HedgeStrategy(
        legs=[atm_call, atm_put],
        total_premium=6000.0,
        label="Straddle",
    )
    scenario = ScenarioSpec(spot_shock_pct=-0.20, vol_shock_abs=0.15, elapsed_days=0)
    result = simulate_strategy(strategy, market, scenario, leg_premiums=[3000.0, 3000.0])
    assert "total_net_pnl" in result
    assert len(result["leg_results"]) == 2


def test_simulate_strategy_mismatched_premiums_raises(market, atm_call):
    strategy = HedgeStrategy(legs=[atm_call], total_premium=3000.0)
    scenario = ScenarioSpec()
    with pytest.raises(ValueError):
        simulate_strategy(strategy, market, scenario, leg_premiums=[3000.0, 1000.0])


# ---------------------------------------------------------------------------
# simulate_strategy_grid
# ---------------------------------------------------------------------------

def test_simulate_strategy_grid_length(market, atm_put):
    strategy = HedgeStrategy(legs=[atm_put], total_premium=3000.0)
    spot_shocks = [-0.20, 0.0, 0.20]
    vol_shocks = [0.0, 0.10]
    time_steps = [0, 30]
    results = simulate_strategy_grid(
        strategy, market, spot_shocks, vol_shocks, time_steps, leg_premiums=[3000.0]
    )
    assert len(results) == 3 * 2 * 2


# ---------------------------------------------------------------------------
# Theta Engine
# ---------------------------------------------------------------------------

def test_price_over_time_decreases_for_atm_call(market, atm_call):
    """An ATM call with static spot should lose value over time."""
    elapsed = [0, 30, 60, 89]
    rows = price_over_time(atm_call, market, elapsed)
    prices = [r["option_price"] for r in rows]
    # Price at t=0 should be higher than near expiry
    assert prices[0] > prices[-1]


def test_compute_theta_profile_length(market, atm_put):
    profile = compute_theta_profile(atm_put, market, total_days=90, steps=9)
    assert len(profile) == 10  # 0..9 inclusive


def test_theta_profile_values_are_dicts(market, atm_put):
    profile = compute_theta_profile(atm_put, market, total_days=90, steps=3)
    for row in profile:
        assert "elapsed_days" in row
        assert "option_price" in row
        assert "theta_daily" in row


# ---------------------------------------------------------------------------
# Analytics: break-even
# ---------------------------------------------------------------------------

def test_breakeven_call():
    call = OptionSpec(option_type="call", strike=100.0, expiry_days=0)
    result = compute_breakeven(call, premium_paid=5.0)
    assert result["breakeven_spot"] == pytest.approx(105.0)


def test_breakeven_put():
    put = OptionSpec(option_type="put", strike=100.0, expiry_days=0)
    result = compute_breakeven(put, premium_paid=5.0)
    assert result["breakeven_spot"] == pytest.approx(95.0)


def test_strategy_breakeven_found():
    call = OptionSpec(option_type="call", strike=100.0, expiry_days=0)
    strategy = HedgeStrategy(legs=[call], total_premium=5.0)
    spots = [float(s) for s in range(80, 130)]
    result = compute_strategy_breakeven(strategy, leg_premiums=[5.0], spot_range=spots)
    assert len(result["breakeven_spots"]) >= 1
    assert abs(result["breakeven_spots"][0] - 105.0) < 2.0


# ---------------------------------------------------------------------------
# Analytics: convexity metrics
# ---------------------------------------------------------------------------

def test_convexity_metrics_keys(market, atm_call):
    spots = [float(s) for s in range(70000, 130001, 1000)]
    metrics = compute_convexity_metrics(atm_call, market, spots, premium_paid=3000.0)
    assert "max_net_pnl" in metrics
    assert "convexity" in metrics
    assert "spot_at_max_pnl" in metrics


def test_convexity_positive_for_long_call(market, atm_call):
    """Long option (convex payoff) should have positive gamma/convexity."""
    spots = [float(s) for s in range(70000, 130001, 1000)]
    metrics = compute_convexity_metrics(atm_call, market, spots, premium_paid=3000.0)
    assert metrics["convexity"] >= 0


# ---------------------------------------------------------------------------
# Analytics: payoff tables
# ---------------------------------------------------------------------------

def test_build_payoff_table_row_structure():
    call = OptionSpec(option_type="call", strike=100.0, expiry_days=0)
    table = build_payoff_table(call, premium_paid=5.0, spot_range=[90.0, 100.0, 110.0])
    for row in table:
        assert "spot" in row
        assert "gross_payoff" in row
        assert "net_pnl" in row
        assert "return_pct" in row


def test_build_strategy_payoff_table():
    call = OptionSpec(option_type="call", strike=100.0, expiry_days=0)
    strategy = HedgeStrategy(legs=[call], total_premium=5.0)
    table = build_strategy_payoff_table(strategy, leg_premiums=[5.0], spot_range=[90.0, 105.0, 115.0])
    assert len(table) == 3
    for row in table:
        assert "total_net_pnl" in row


# ---------------------------------------------------------------------------
# Analytics: summarise_grid_results
# ---------------------------------------------------------------------------

def test_summarise_grid_results_keys(market, atm_put):
    strategy = HedgeStrategy(legs=[atm_put], total_premium=3000.0)
    results = simulate_strategy_grid(
        strategy, market,
        spot_shocks=[-0.20, 0.0, 0.20],
        vol_shocks=[0.0, 0.10],
        time_steps=[0],
        leg_premiums=[3000.0],
    )
    summary = summarise_grid_results(results)
    assert "total_scenarios" in summary
    assert summary["total_scenarios"] == 6
    assert "profitable_count" in summary
    assert "best_scenario" in summary
    assert "worst_scenario" in summary


def test_summarise_empty_returns_empty():
    assert summarise_grid_results([]) == {}
