# Options Hedge Simulation Engine

A pure-Python library for simulating options strategies used as hedges.  
No external dependencies beyond the standard library — pricing, smile interpolation, scenario generation, vol shocks, theta decay and analytics all in one place.

---

## Architecture

| Module | Responsibility |
|---|---|
| `models.py` | Data classes: `OptionSpec`, `MarketState`, `SmileSurface`, `ScenarioSpec`, `HedgeStrategy`, `SimulationResult` |
| `pricing.py` | Black-Scholes pricing + Greeks (`black_scholes_price`, `option_intrinsic_value`, `net_option_pnl`) |
| `smile.py` | IV smile interpolation by moneyness (`interpolate_iv_from_smile`, `price_option_from_moneyness`, `build_smile_from_atm_skew`) |
| `payoff.py` | Expiry payoffs for single legs and multi-leg strategies |
| `scenarios.py` | Scenario generators: price grids, vol grids, time grids, Cartesian products, named stress events |
| `vol_shock.py` | Vol shock engine: flat shift, smile shift, P&L matrix |
| `theta.py` | Theta engine: time decay and option re-pricing over a time grid |
| `simulation.py` | Simulation master: `simulate_single_hedge`, `simulate_strategy`, `simulate_strategy_grid` |
| `analytics.py` | Analytics: break-even, convexity, payoff tables, grid summaries |

---

## Design Principles

- **Pure functions** — no hidden state, no side effects.
- **Input/output via dataclasses or dicts** — easy to inspect and serialise.
- **Model, scenario and strategy are strictly separated** — never mixed.
- **Calculation and plotting are separate** — the library never draws charts.
- **Both strike and moneyness are accepted** — `SmileSurface` is indexed by `K/S`.

---

## Quick Start

```python
from options_hedge import (
    OptionSpec, MarketState, ScenarioSpec,
    black_scholes_price,
    simulate_single_hedge,
    build_smile_from_atm_skew,
    build_payoff_table,
    compute_breakeven,
)

# 1. Define a put option on Ibovespa
market = MarketState(spot=130_000, risk_free_rate=0.135, dividend_yield=0.0, implied_vol=0.25)
hedge = OptionSpec(option_type="put", strike=130_000, expiry_days=63, quantity=1)

# 2. Price it
bs = black_scholes_price(hedge, market)
print(f"Put price: {bs['price']:.0f}  delta: {bs['delta']:.4f}")

# 3. Stress-test: spot -20 %, vol +20 pp, 30 days elapsed
scenario = ScenarioSpec(spot_shock_pct=-0.20, vol_shock_abs=0.20, elapsed_days=30)
smile = build_smile_from_atm_skew(atm_vol=0.25, skew_per_delta=0.03)
result = simulate_single_hedge(hedge, market, scenario, premium_paid=bs["price"], smile=smile)
print(f"Net PnL: {result.net_pnl:.0f}  Return: {result.return_on_premium:.1%}")

# 4. Break-even analysis
be = compute_breakeven(hedge, premium_paid=bs["price"])
print(f"Break-even spot: {be['breakeven_spot']:.0f}")

# 5. Payoff table at expiry
spots = [float(s) for s in range(90_000, 140_001, 5_000)]
table = build_payoff_table(hedge, premium_paid=bs["price"], spot_range=spots)
for row in table:
    print(f"Spot {row['spot']:>10.0f}  net PnL {row['net_pnl']:>10.0f}  return {row['return_pct']:>7.1f}%")
```

---

## Running Tests

```bash
pip install -e ".[dev]"
pytest
```
