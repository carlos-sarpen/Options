"""
Options Hedge Simulation Engine

Public API — import everything from this package:

    from options_hedge import (
        OptionSpec, MarketState, SmileSurface, ScenarioSpec,
        HedgeStrategy, SimulationResult,
        black_scholes_price, option_intrinsic_value, net_option_pnl,
        interpolate_iv_from_smile, price_option_from_moneyness, build_smile_from_atm_skew,
        compute_leg_payoff, compute_strategy_payoff, build_payoff_profile,
        generate_price_scenarios, generate_vol_scenarios, generate_time_grid,
        build_scenario_grid, apply_scenario, named_stress_scenarios,
        apply_vol_shock, shift_smile, price_with_vol_shock, vol_shock_pnl_matrix,
        apply_time_decay, price_over_time, compute_theta_profile,
        simulate_single_hedge, simulate_strategy, simulate_strategy_grid,
        compute_breakeven, compute_strategy_breakeven,
        compute_convexity_metrics, build_payoff_table,
        build_strategy_payoff_table, summarise_grid_results,
        compute_portfolio_impact,
    )
"""

from .models import (
    HedgeStrategy,
    MarketState,
    OptionSpec,
    ScenarioSpec,
    SimulationResult,
    SmileSurface,
)
from .pricing import (
    black_scholes_price,
    net_option_pnl,
    option_intrinsic_value,
)
from .smile import (
    build_smile_from_atm_skew,
    interpolate_iv_from_smile,
    price_option_from_moneyness,
)
from .payoff import (
    build_payoff_profile,
    build_strategy_payoff_profile,
    compute_leg_payoff,
    compute_strategy_payoff,
)
from .scenarios import (
    apply_scenario,
    build_scenario_grid,
    generate_price_scenarios,
    generate_time_grid,
    generate_vol_scenarios,
    named_stress_scenarios,
)
from .vol_shock import (
    apply_vol_shock,
    price_with_vol_shock,
    shift_smile,
    vol_shock_pnl_matrix,
)
from .theta import (
    apply_time_decay,
    compute_theta_profile,
    price_over_time,
)
from .simulation import (
    simulate_single_hedge,
    simulate_strategy,
    simulate_strategy_grid,
)
from .analytics import (
    build_payoff_table,
    build_strategy_payoff_table,
    compute_breakeven,
    compute_convexity_metrics,
    compute_strategy_breakeven,
    summarise_grid_results,
)
from .portfolio import (
    compute_portfolio_impact,
)

__all__ = [
    # models
    "OptionSpec",
    "MarketState",
    "SmileSurface",
    "ScenarioSpec",
    "HedgeStrategy",
    "SimulationResult",
    # pricing
    "black_scholes_price",
    "option_intrinsic_value",
    "net_option_pnl",
    # smile
    "interpolate_iv_from_smile",
    "price_option_from_moneyness",
    "build_smile_from_atm_skew",
    # payoff
    "compute_leg_payoff",
    "compute_strategy_payoff",
    "build_payoff_profile",
    "build_strategy_payoff_profile",
    # scenarios
    "generate_price_scenarios",
    "generate_vol_scenarios",
    "generate_time_grid",
    "build_scenario_grid",
    "apply_scenario",
    "named_stress_scenarios",
    # vol shock
    "apply_vol_shock",
    "shift_smile",
    "price_with_vol_shock",
    "vol_shock_pnl_matrix",
    # theta
    "apply_time_decay",
    "price_over_time",
    "compute_theta_profile",
    # simulation
    "simulate_single_hedge",
    "simulate_strategy",
    "simulate_strategy_grid",
    # analytics
    "compute_breakeven",
    "compute_strategy_breakeven",
    "compute_convexity_metrics",
    "build_payoff_table",
    "build_strategy_payoff_table",
    "summarise_grid_results",
    # portfolio
    "compute_portfolio_impact",
]
