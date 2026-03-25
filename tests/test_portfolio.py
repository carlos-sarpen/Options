"""Tests for the Portfolio Impact module."""

import pytest

from options_hedge import compute_portfolio_impact


# ---------------------------------------------------------------------------
# Basic computation
# ---------------------------------------------------------------------------

def test_basic_portfolio_impact():
    """Equity impact = beta_ibov * ibov_move + beta_spread * spread_move."""
    result = compute_portfolio_impact(
        ibov_move_pct=-0.05,
        smal_ibov_spread_move=0.02,
        beta_ibov=1.0,
        beta_spread=0.5,
        options_return_pct=0.01,
    )
    assert result["ibov_impact"] == pytest.approx(-0.05)
    assert result["spread_impact"] == pytest.approx(0.01)
    assert result["equity_impact"] == pytest.approx(-0.04)
    assert result["options_return_pct"] == pytest.approx(0.01)
    assert result["total_impact"] == pytest.approx(-0.03)


def test_zero_moves_zero_impact():
    result = compute_portfolio_impact(
        ibov_move_pct=0.0,
        smal_ibov_spread_move=0.0,
        beta_ibov=1.2,
        beta_spread=0.8,
        options_return_pct=0.0,
    )
    assert result["total_impact"] == pytest.approx(0.0)
    assert result["equity_impact"] == pytest.approx(0.0)


def test_default_options_return_is_zero():
    result = compute_portfolio_impact(
        ibov_move_pct=-0.10,
        smal_ibov_spread_move=0.0,
        beta_ibov=1.0,
        beta_spread=1.0,
    )
    assert result["options_return_pct"] == pytest.approx(0.0)
    assert result["total_impact"] == pytest.approx(-0.10)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_negative_beta_ibov():
    """Negative beta means inverse exposure to IBOV."""
    result = compute_portfolio_impact(
        ibov_move_pct=-0.10,
        smal_ibov_spread_move=0.0,
        beta_ibov=-0.5,
        beta_spread=0.0,
    )
    assert result["ibov_impact"] == pytest.approx(0.05)
    assert result["total_impact"] == pytest.approx(0.05)


def test_zero_betas():
    """Zero betas means no equity exposure, only options matter."""
    result = compute_portfolio_impact(
        ibov_move_pct=-0.20,
        smal_ibov_spread_move=0.05,
        beta_ibov=0.0,
        beta_spread=0.0,
        options_return_pct=0.03,
    )
    assert result["equity_impact"] == pytest.approx(0.0)
    assert result["total_impact"] == pytest.approx(0.03)


def test_large_crash_with_hedge():
    """Simulate a large crash where options hedge provides protection."""
    result = compute_portfolio_impact(
        ibov_move_pct=-0.30,
        smal_ibov_spread_move=-0.05,
        beta_ibov=1.0,
        beta_spread=0.5,
        options_return_pct=0.15,
    )
    assert result["ibov_impact"] == pytest.approx(-0.30)
    assert result["spread_impact"] == pytest.approx(-0.025)
    assert result["equity_impact"] == pytest.approx(-0.325)
    assert result["total_impact"] == pytest.approx(-0.175)


# ---------------------------------------------------------------------------
# Return dict structure
# ---------------------------------------------------------------------------

def test_return_keys():
    result = compute_portfolio_impact(
        ibov_move_pct=0.0,
        smal_ibov_spread_move=0.0,
        beta_ibov=1.0,
        beta_spread=1.0,
    )
    expected_keys = {
        "ibov_impact",
        "spread_impact",
        "equity_impact",
        "options_return_pct",
        "total_impact",
    }
    assert set(result.keys()) == expected_keys


# ---------------------------------------------------------------------------
# Additivity: total = equity + options
# ---------------------------------------------------------------------------

def test_total_is_sum_of_equity_and_options():
    result = compute_portfolio_impact(
        ibov_move_pct=-0.08,
        smal_ibov_spread_move=0.03,
        beta_ibov=0.9,
        beta_spread=1.1,
        options_return_pct=0.04,
    )
    assert result["total_impact"] == pytest.approx(
        result["equity_impact"] + result["options_return_pct"]
    )


def test_equity_is_sum_of_ibov_and_spread():
    result = compute_portfolio_impact(
        ibov_move_pct=0.05,
        smal_ibov_spread_move=-0.02,
        beta_ibov=1.3,
        beta_spread=0.7,
    )
    assert result["equity_impact"] == pytest.approx(
        result["ibov_impact"] + result["spread_impact"]
    )
