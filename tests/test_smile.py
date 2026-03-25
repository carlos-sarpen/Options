"""Tests for the Smile Engine."""

import pytest

from options_hedge import (
    MarketState,
    OptionSpec,
    SmileSurface,
    build_smile_from_atm_skew,
    interpolate_iv_from_smile,
    price_option_from_moneyness,
)


@pytest.fixture
def sample_smile():
    return SmileSurface(
        points={0.90: 0.25, 0.95: 0.22, 1.00: 0.20, 1.05: 0.21, 1.10: 0.23}
    )


@pytest.fixture
def market():
    return MarketState(spot=100.0, risk_free_rate=0.05, dividend_yield=0.0, implied_vol=0.20)


# ---------------------------------------------------------------------------
# SmileSurface validation
# ---------------------------------------------------------------------------

def test_smile_requires_at_least_2_points():
    with pytest.raises(ValueError):
        SmileSurface(points={1.00: 0.20})


def test_smile_negative_iv_raises():
    with pytest.raises(ValueError):
        SmileSurface(points={0.90: -0.10, 1.00: 0.20})


# ---------------------------------------------------------------------------
# Interpolation correctness
# ---------------------------------------------------------------------------

def test_atm_interpolation_linear(sample_smile):
    result = interpolate_iv_from_smile(sample_smile, 1.00, method="linear")
    assert result["iv"] == pytest.approx(0.20, abs=1e-6)
    assert result["extrapolated"] is False


def test_atm_interpolation_cubic(sample_smile):
    result = interpolate_iv_from_smile(sample_smile, 1.00, method="cubic")
    assert result["iv"] == pytest.approx(0.20, abs=0.001)


def test_midpoint_interpolation(sample_smile):
    """IV at moneyness=0.925 should be between 0.22 and 0.25."""
    result = interpolate_iv_from_smile(sample_smile, 0.925, method="linear")
    assert 0.22 <= result["iv"] <= 0.25


def test_extrapolation_left(sample_smile):
    result = interpolate_iv_from_smile(sample_smile, 0.80, method="linear")
    assert result["extrapolated"] is True
    assert result["iv"] == pytest.approx(0.25, abs=1e-6)  # flat extrapolation at boundary


def test_extrapolation_right(sample_smile):
    result = interpolate_iv_from_smile(sample_smile, 1.20, method="linear")
    assert result["extrapolated"] is True
    assert result["iv"] == pytest.approx(0.23, abs=1e-6)


def test_iv_floor_is_positive(sample_smile):
    """IV must never be zero or negative."""
    for m in [0.70, 0.85, 1.00, 1.15, 1.30]:
        result = interpolate_iv_from_smile(sample_smile, m)
        assert result["iv"] > 0.0


# ---------------------------------------------------------------------------
# price_option_from_moneyness
# ---------------------------------------------------------------------------

def test_smile_price_exceeds_intrinsic(sample_smile, market):
    call = OptionSpec(option_type="call", strike=100.0, expiry_days=90)
    result = price_option_from_moneyness(call, market, sample_smile)
    assert result["price"] > 0.0
    assert "smile_iv" in result
    assert result["smile_iv"] == pytest.approx(0.20, abs=0.005)


def test_smile_price_otm_put(sample_smile, market):
    """OTM put should be priced with higher vol (smile skew) than ATM."""
    otm_put = OptionSpec(option_type="put", strike=90.0, expiry_days=90)
    atm_put = OptionSpec(option_type="put", strike=100.0, expiry_days=90)

    otm_result = price_option_from_moneyness(otm_put, market, sample_smile)
    atm_result = price_option_from_moneyness(atm_put, market, sample_smile)

    # OTM put vol > ATM vol due to skew
    assert otm_result["smile_iv"] > atm_result["smile_iv"]


# ---------------------------------------------------------------------------
# build_smile_from_atm_skew
# ---------------------------------------------------------------------------

def test_build_smile_from_atm_skew_structure():
    smile = build_smile_from_atm_skew(atm_vol=0.20, skew_per_delta=0.02)
    assert isinstance(smile, SmileSurface)
    assert 1.00 in smile.points
    assert smile.points[1.00] == pytest.approx(0.20)


def test_build_smile_skew_applied():
    """Moneyness 0.90 should have higher vol than ATM due to skew."""
    smile = build_smile_from_atm_skew(atm_vol=0.20, skew_per_delta=0.05)
    assert smile.points[0.90] > smile.points[1.00]


def test_build_smile_custom_grid():
    grid = [0.95, 1.00, 1.05]
    smile = build_smile_from_atm_skew(atm_vol=0.18, moneyness_grid=grid)
    assert set(smile.points.keys()) == set(grid)
