"""Tests for the Scenario Generator and Vol Shock Engine."""

import pytest

from options_hedge import (
    MarketState,
    OptionSpec,
    ScenarioSpec,
    SmileSurface,
    apply_scenario,
    apply_vol_shock,
    build_scenario_grid,
    generate_price_scenarios,
    generate_time_grid,
    generate_vol_scenarios,
    named_stress_scenarios,
    price_with_vol_shock,
    shift_smile,
    vol_shock_pnl_matrix,
)


@pytest.fixture
def market():
    return MarketState(spot=100_000.0, risk_free_rate=0.13, dividend_yield=0.0, implied_vol=0.25)


# ---------------------------------------------------------------------------
# generate_price_scenarios
# ---------------------------------------------------------------------------

def test_price_scenarios_count():
    shocks = [-0.30, -0.20, -0.10, 0.0, 0.10]
    result = generate_price_scenarios(100.0, shocks)
    assert len(result) == 5


def test_price_scenarios_values():
    result = generate_price_scenarios(100.0, [-0.10, 0.0, 0.20])
    assert result[0]["shocked_spot"] == pytest.approx(90.0)
    assert result[1]["shocked_spot"] == pytest.approx(100.0)
    assert result[2]["shocked_spot"] == pytest.approx(120.0)


# ---------------------------------------------------------------------------
# generate_vol_scenarios
# ---------------------------------------------------------------------------

def test_vol_scenarios_count():
    result = generate_vol_scenarios(0.20, [-0.05, 0.0, 0.10])
    assert len(result) == 3


def test_vol_scenarios_floor():
    """Vol must never go below 0.001."""
    result = generate_vol_scenarios(0.01, [-0.50])
    assert result[0]["shocked_vol"] >= 0.001


# ---------------------------------------------------------------------------
# generate_time_grid
# ---------------------------------------------------------------------------

def test_time_grid_boundaries():
    grid = generate_time_grid(90, 3)
    assert grid[0]["elapsed_days"] == 0
    assert grid[-1]["remaining_days"] == 0


def test_time_grid_step_count():
    grid = generate_time_grid(60, 4)
    assert len(grid) == 5  # 0, 15, 30, 45, 60


def test_time_grid_zero_steps_raises():
    with pytest.raises(ValueError):
        generate_time_grid(90, 0)


# ---------------------------------------------------------------------------
# build_scenario_grid
# ---------------------------------------------------------------------------

def test_scenario_grid_cartesian_product():
    spot_shocks = [-0.20, 0.0, 0.20]
    vol_shocks = [0.0, 0.10]
    time_steps = [0, 30]
    grid = build_scenario_grid(spot_shocks, vol_shocks, time_steps)
    assert len(grid) == 3 * 2 * 2


def test_scenario_grid_type():
    grid = build_scenario_grid([-0.10], [0.05], [0])
    assert isinstance(grid[0], ScenarioSpec)


# ---------------------------------------------------------------------------
# apply_scenario
# ---------------------------------------------------------------------------

def test_apply_scenario_spot(market):
    scenario = ScenarioSpec(spot_shock_pct=-0.10, vol_shock_abs=0.0, elapsed_days=0)
    new_market = apply_scenario(market, scenario)
    assert new_market.spot == pytest.approx(90_000.0)
    assert new_market.implied_vol == pytest.approx(0.25)


def test_apply_scenario_vol(market):
    scenario = ScenarioSpec(spot_shock_pct=0.0, vol_shock_abs=0.10, elapsed_days=0)
    new_market = apply_scenario(market, scenario)
    assert new_market.implied_vol == pytest.approx(0.35)


def test_apply_scenario_vol_floor(market):
    scenario = ScenarioSpec(spot_shock_pct=0.0, vol_shock_abs=-0.99, elapsed_days=0)
    new_market = apply_scenario(market, scenario)
    assert new_market.implied_vol >= 0.001


# ---------------------------------------------------------------------------
# named_stress_scenarios
# ---------------------------------------------------------------------------

def test_named_stress_scenarios_non_empty():
    scenarios = named_stress_scenarios()
    assert len(scenarios) > 0


def test_named_stress_scenarios_keys():
    scenarios = named_stress_scenarios()
    for s in scenarios:
        assert "label" in s
        assert "spot_shock_pct" in s
        assert "vol_shock_abs" in s


# ---------------------------------------------------------------------------
# Vol Shock Engine
# ---------------------------------------------------------------------------

def test_apply_vol_shock_increase(market):
    new_market = apply_vol_shock(market, 0.10)
    assert new_market.implied_vol == pytest.approx(0.35)
    assert new_market.spot == market.spot


def test_apply_vol_shock_floor(market):
    new_market = apply_vol_shock(market, -10.0)
    assert new_market.implied_vol >= 0.001


def test_shift_smile():
    smile = SmileSurface(points={0.90: 0.25, 1.00: 0.20, 1.10: 0.22})
    shifted = shift_smile(smile, 0.05)
    assert shifted.points[1.00] == pytest.approx(0.25)
    assert shifted.points[0.90] == pytest.approx(0.30)


def test_price_with_vol_shock_increases_price(market):
    call = OptionSpec(option_type="call", strike=100_000.0, expiry_days=90)
    base = price_with_vol_shock(call, market, vol_shock_abs=0.0)
    shocked = price_with_vol_shock(call, market, vol_shock_abs=0.10)
    assert shocked["price"] > base["price"]


def test_vol_shock_pnl_matrix_shape(market):
    call = OptionSpec(option_type="call", strike=100_000.0, expiry_days=90)
    spot_shocks = [-0.10, 0.0, 0.10]
    vol_shocks = [0.0, 0.10]
    matrix = vol_shock_pnl_matrix(call, market, spot_shocks, vol_shocks, premium_paid=3000.0)
    assert len(matrix) == 6  # 3 spot × 2 vol
