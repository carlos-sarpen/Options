"""
demo.py
=======
Script central de demonstração que mostra como usar cada módulo do sistema,
montando uma sequência completa de:

  1. Acesso a dados — gera uma planilha Excel de exemplo
  2. Extração do sorriso de volatilidade — lê a planilha e constrói as superfícies
  3. Simulação do preço com base em um choque determinado — aplica choques de spot
     e de volatilidade e reprecia as opções usando o sorriso

Uso::

    python demo.py

O script não requer argumentos e não persiste nenhum arquivo (a planilha de
exemplo é criada em um diretório temporário e removida ao final).
"""

from __future__ import annotations

import os
import sys
import tempfile
from typing import Dict

import pandas as pd

# ---------------------------------------------------------------------------
# Utilitário de formatação
# ---------------------------------------------------------------------------

_SEP = "=" * 65


def _header(title: str) -> None:
    print(f"\n{_SEP}")
    print(f"  {title}")
    print(_SEP)


def _sub(title: str) -> None:
    print(f"\n--- {title} ---")


# ---------------------------------------------------------------------------
# SEÇÃO 1 — Geração de dados de amostra
# ---------------------------------------------------------------------------

def secao_1_gerar_dados(tmp_dir: str) -> str:
    """
    Demonstra o uso de :mod:`create_sample_data`.

    Gera uma planilha Excel com 8 abas (call/put × BOVA/SMAL × 17d/35d)
    e retorna o caminho do arquivo criado.
    """
    _header("SEÇÃO 1 — Geração de dados de amostra (create_sample_data)")

    from create_sample_data import create_sample_workbook

    filepath = os.path.join(tmp_dir, "options_sample.xlsx")
    create_sample_workbook(output_path=filepath)

    # Inspeciona as abas criadas
    xl = pd.ExcelFile(filepath)
    print(f"\nAbas geradas ({len(xl.sheet_names)} no total):")
    for name in xl.sheet_names:
        print(f"  • {name}")

    return filepath


# ---------------------------------------------------------------------------
# SEÇÃO 2 — Extração do sorriso de volatilidade
# ---------------------------------------------------------------------------

def secao_2_extrair_sorriso(filepath: str) -> Dict[str, pd.DataFrame]:
    """
    Demonstra o uso de :func:`volatility_smile.build_smiles` e
    :func:`volatility_smile.smiles_to_dataframe`.

    Lê a planilha gerada na Seção 1 e constrói um dicionário de DataFrames de
    sorriso, um por combinação (tipo × ativo × prazo).
    """
    _header("SEÇÃO 2 — Extração do sorriso de volatilidade (volatility_smile)")

    from volatility_smile import build_smiles, smiles_to_dataframe

    _sub("2.1  build_smiles — lê todas as abas automaticamente")
    smiles = build_smiles(filepath)

    print(f"\nSorrisos extraídos ({len(smiles)} no total): {sorted(smiles.keys())}")

    _sub("2.2  Inspecionando um sorriso individual: call_B_17")
    df = smiles["call_B_17"]
    print(df.to_string(index=False))

    _sub("2.3  smiles_to_dataframe — combina todos em um único DataFrame")
    tidy = smiles_to_dataframe(smiles)
    print(f"\nShape: {tidy.shape}  |  Colunas: {list(tidy.columns)}")
    print(tidy.groupby("smile_id").size().rename("n_pontos").to_string())

    return smiles


# ---------------------------------------------------------------------------
# SEÇÃO 3 — Construção da SmileSurface a partir do DataFrame
# ---------------------------------------------------------------------------

def secao_3_construir_smile_surface(
    smiles: Dict[str, pd.DataFrame],
) -> "SmileSurface":
    """
    Demonstra como converter um DataFrame de sorriso extraído da planilha em
    uma :class:`options_hedge.SmileSurface`, que é usada pelo motor de
    precificação.
    """
    _header("SEÇÃO 3 — Construção da SmileSurface (options_hedge.SmileSurface)")

    from options_hedge import SmileSurface, interpolate_iv_from_smile

    # Usamos o sorriso de call BOVA 17 dias
    df = smiles["call_B_17"]
    smile = SmileSurface(
        points=dict(zip(df["moneyness"].tolist(), df["implied_vol"].tolist()))
    )

    print(f"\nSmileSurface criada com {len(smile.points)} pontos.")
    print("Pontos (moneyness → IV):")
    for m, iv in sorted(smile.points.items()):
        print(f"  {m:.4f}  →  {iv*100:.2f}%")

    _sub("3.1  Interpolação por moneyness (método cúbico)")
    for m_test in [0.90, 0.95, 1.00, 1.05, 1.10]:
        result = interpolate_iv_from_smile(smile, m_test, method="cubic")
        tag = " [extrapolado]" if result["extrapolated"] else ""
        print(f"  IV @ moneyness={m_test:.2f}: {result['iv']*100:.2f}%{tag}")

    _sub("3.2  Construção paramétrica via build_smile_from_atm_skew")
    from options_hedge import build_smile_from_atm_skew
    smile_param = build_smile_from_atm_skew(atm_vol=0.22, skew_per_delta=0.03)
    print(f"  SmileSurface paramétrica com {len(smile_param.points)} pontos")
    for m, iv in sorted(smile_param.points.items()):
        print(f"    {m:.2f}  →  {iv*100:.2f}%")

    return smile


# ---------------------------------------------------------------------------
# SEÇÃO 4 — Precificação Black-Scholes com vol plana
# ---------------------------------------------------------------------------

def secao_4_precificacao_bs() -> tuple:
    """
    Demonstra o uso de :func:`options_hedge.black_scholes_price` e
    :func:`options_hedge.option_intrinsic_value`.
    """
    _header("SEÇÃO 4 — Precificação Black-Scholes (options_hedge.pricing)")

    from options_hedge import (
        MarketState,
        OptionSpec,
        black_scholes_price,
        option_intrinsic_value,
    )

    # Opção de exemplo: CALL BOVA, strike 130 000, spot 128 000
    spot = 128_000.0
    strike = 130_000.0
    market = MarketState(spot=spot, risk_free_rate=0.1075, dividend_yield=0.03,
                         implied_vol=0.22)
    option = OptionSpec(option_type="call", strike=strike, expiry_days=35,
                        quantity=1.0)

    _sub("4.1  Definição do contrato e do mercado")
    print(f"  Opção  : {option.option_type.upper()} | Strike={strike:,.0f} | "
          f"Vencimento={option.expiry_days}d | Qty={option.quantity}")
    print(f"  Mercado: Spot={spot:,.0f} | r={market.risk_free_rate*100:.2f}% | "
          f"q={market.dividend_yield*100:.2f}% | IV={market.implied_vol*100:.2f}%")

    _sub("4.2  Preço e gregas Black-Scholes")
    bs = black_scholes_price(option, market)
    print(f"  Preço     : {bs['price']:>10.4f}")
    print(f"  Delta     : {bs['delta']:>10.4f}")
    print(f"  Gamma     : {bs['gamma']:>10.6f}")
    print(f"  Theta/dia : {bs['theta']:>10.4f}")
    print(f"  Vega/1pp  : {bs['vega']:>10.4f}")

    _sub("4.3  Valor intrínseco")
    intrinsic = option_intrinsic_value(option, market)
    print(f"  Intrínseco : {intrinsic['intrinsic_value']:.4f}")
    print(f"  Moneyness  : {intrinsic['moneyness']:.4f}  "
          f"({'ITM' if intrinsic['itm'] else 'OTM'})")

    premium = bs["price"]
    return option, market, premium


# ---------------------------------------------------------------------------
# SEÇÃO 5 — Choque de volatilidade e repreco com sorriso
# ---------------------------------------------------------------------------

def secao_5_choque_vol(
    option: "OptionSpec",
    market: "MarketState",
    smile: "SmileSurface",
    premium: float,
) -> None:
    """
    Demonstra :func:`options_hedge.price_with_vol_shock`,
    :func:`options_hedge.shift_smile` e
    :func:`options_hedge.vol_shock_pnl_matrix`.
    """
    _header("SEÇÃO 5 — Choque de volatilidade (options_hedge.vol_shock)")

    from options_hedge import (
        price_with_vol_shock,
        shift_smile,
        vol_shock_pnl_matrix,
    )

    _sub("5.1  Repreco após choque de vol plana (+10 pp)")
    shocked_flat = price_with_vol_shock(option, market, vol_shock_abs=0.10)
    print(f"  Preço pós-choque (vol plana) : {shocked_flat['price']:.4f}")
    print(f"  Vol usada                    : {shocked_flat['shocked_vol']*100:.2f}%")

    _sub("5.2  Repreco após choque de vol com sorriso (+10 pp no sorriso)")
    shocked_smile_result = price_with_vol_shock(
        option, market, vol_shock_abs=0.10, smile=smile, interp_method="cubic"
    )
    print(f"  Preço pós-choque (sorriso)   : {shocked_smile_result['price']:.4f}")
    print(f"  Vol do sorriso usada         : {shocked_smile_result['shocked_vol']*100:.2f}%")

    _sub("5.3  Sorriso deslocado (+10 pp em todos os pontos)")
    shifted = shift_smile(smile, 0.10)
    # Ponto mais próximo de ATM (moneyness = 1.0) — original e deslocado
    atm_original = sorted(smile.points.items(), key=lambda kv: abs(kv[0] - 1.0))[0][1]
    atm_shifted  = sorted(shifted.points.items(), key=lambda kv: abs(kv[0] - 1.0))[0][1]
    print(f"  IV @ ATM original : {atm_original*100:.2f}%")
    print(f"  IV @ ATM deslocada: {atm_shifted*100:.2f}%")

    _sub("5.4  Matriz de PnL: spot × vol shocks")
    spot_shocks = [-0.10, -0.05, 0.0, 0.05, 0.10]
    vol_shocks  = [-0.05, 0.0, 0.10, 0.20]
    matrix = vol_shock_pnl_matrix(
        option, market, spot_shocks, vol_shocks, premium_paid=premium, smile=smile
    )

    df_matrix = pd.DataFrame(matrix)
    df_pivot = df_matrix.pivot_table(
        index="spot_shock_pct", columns="vol_shock_abs",
        values="net_pnl", aggfunc="first"
    )
    df_pivot.index = [f"{v*100:+.0f}%" for v in df_pivot.index]
    df_pivot.columns = [f"Δvol={v*100:+.0f}pp" for v in df_pivot.columns]
    print("\n  Net PnL por (choque spot, choque vol):")
    print(df_pivot.to_string())


# ---------------------------------------------------------------------------
# SEÇÃO 6 — Simulação com choque de spot (simulate_single_hedge)
# ---------------------------------------------------------------------------

def secao_6_simular_choque(
    option: "OptionSpec",
    market: "MarketState",
    smile: "SmileSurface",
    premium: float,
) -> None:
    """
    Demonstra :func:`options_hedge.simulate_single_hedge` com um choque
    determinado de spot e de volatilidade.
    """
    _header("SEÇÃO 6 — Simulação com choque determinado (options_hedge.simulation)")

    from options_hedge import ScenarioSpec, simulate_single_hedge

    choques = [
        ScenarioSpec(spot_shock_pct=-0.10, vol_shock_abs=0.10, elapsed_days=0),
        ScenarioSpec(spot_shock_pct=-0.20, vol_shock_abs=0.20, elapsed_days=0),
        ScenarioSpec(spot_shock_pct=-0.30, vol_shock_abs=0.30, elapsed_days=10),
    ]
    labels = [
        "Choque leve   (spot -10%, vol +10pp)",
        "Choque médio  (spot -20%, vol +20pp)",
        "Choque severo (spot -30%, vol +30pp, 10d passados)",
    ]

    print(f"\n{'Cenário':<42} {'Spot chocado':>13} {'Preço opt':>10} "
          f"{'Net PnL':>10} {'Retorno':>9}")
    print("-" * 90)

    for label, sc in zip(labels, choques):
        result = simulate_single_hedge(
            option, market, sc, premium_paid=premium, smile=smile
        )
        spot_s = result.metadata["shocked_spot"]
        rop = f"{result.return_on_premium*100:+.1f}%" if result.return_on_premium else "N/A"
        print(f"  {label:<40} {spot_s:>13,.0f} {result.option_price:>10.4f} "
              f"{result.net_pnl:>10.4f} {rop:>9}")


# ---------------------------------------------------------------------------
# SEÇÃO 7 — Grid de cenários e sumário
# ---------------------------------------------------------------------------

def secao_7_grid_cenarios(
    option: "OptionSpec",
    market: "MarketState",
    smile: "SmileSurface",
    premium: float,
) -> None:
    """
    Demonstra :func:`options_hedge.simulate_strategy_grid` e
    :func:`options_hedge.summarise_grid_results` para uma estratégia de uma
    perna.
    """
    _header("SEÇÃO 7 — Grid de cenários (options_hedge.simulation + analytics)")

    from options_hedge import (
        HedgeStrategy,
        summarise_grid_results,
        simulate_strategy_grid,
    )

    strategy = HedgeStrategy(legs=[option], total_premium=premium, label="CALL BOVA 35d")

    spot_shocks = [-0.40, -0.20, -0.10, 0.0, 0.10, 0.20]
    vol_shocks  = [-0.05, 0.0, 0.10, 0.20, 0.40]
    time_steps  = [0, 10, 20]

    results = simulate_strategy_grid(
        strategy,
        market,
        spot_shocks,
        vol_shocks,
        time_steps,
        leg_premiums=[premium],
        smile=smile,
    )

    print(f"\n  Cenários simulados : {len(results)}")

    sumario = summarise_grid_results(results)
    print(f"  Cenários lucrativos: {sumario['profitable_count']}")
    print(f"  Cenários com perda : {sumario['loss_count']}")
    avg_r = sumario["avg_return_pct"]
    print(f"  Retorno médio      : {avg_r:.2f}%" if avg_r is not None else "  Retorno médio: N/A")
    print(f"  PnL médio          : {sumario['avg_net_pnl']:.4f}")

    best = sumario["best_scenario"]
    worst = sumario["worst_scenario"]
    print(f"\n  Melhor cenário  → spot chocado={best['shocked_spot']:,.0f} | "
          f"PnL={best['total_net_pnl']:.4f}")
    print(f"  Pior cenário    → spot chocado={worst['shocked_spot']:,.0f} | "
          f"PnL={worst['total_net_pnl']:.4f}")

    _sub("7.1  Cenários históricos nomeados (named_stress_scenarios)")
    from options_hedge import ScenarioSpec, named_stress_scenarios, simulate_single_hedge
    # Nota: choques históricos extremos (ex.: Lehman, -60 % no spot) levam o
    # moneyness para regiões muito fora do alcance do sorriso (moneyness > 2),
    # o que tornaria a extrapolação cúbica instável.  Por isso, esses cenários
    # são simulados com volatilidade plana (sem sorriso), o que é mais adequado
    # para análise de stress histórico.
    print(f"\n  (simulação com vol plana — choques extremos excedem o alcance do sorriso)")
    print(f"\n  {'Cenário':<42} {'Net PnL':>10} {'Retorno':>9}")
    print("  " + "-" * 64)
    for s in named_stress_scenarios():
        sc = ScenarioSpec(
            spot_shock_pct=s["spot_shock_pct"],
            vol_shock_abs=s["vol_shock_abs"],
            elapsed_days=s["elapsed_days"],
        )
        res = simulate_single_hedge(option, market, sc, premium_paid=premium)
        rop = f"{res.return_on_premium*100:+.1f}%" if res.return_on_premium else "N/A"
        print(f"  {s['label']:<42} {res.net_pnl:>10.4f} {rop:>9}")


# ---------------------------------------------------------------------------
# SEÇÃO 8 — Break-even e tabela de payoff
# ---------------------------------------------------------------------------

def secao_8_breakeven_payoff(
    option: "OptionSpec",
    market: "MarketState",
    premium: float,
) -> None:
    """
    Demonstra :func:`options_hedge.compute_breakeven` e
    :func:`options_hedge.build_payoff_table`.
    """
    _header("SEÇÃO 8 — Break-even e tabela de payoff (options_hedge.analytics)")

    from options_hedge import build_payoff_table, compute_breakeven

    _sub("8.1  Break-even")
    be = compute_breakeven(option, premium)
    print(f"  Strike       : {option.strike:,.0f}")
    print(f"  Prêmio pago  : {premium:.4f}")
    print(f"  Break-even   : {be['breakeven_spot']:,.2f} "
          f"({be['breakeven_pct']:+.2f}% acima do strike)")

    _sub("8.2  Tabela de payoff (expiry)")
    spot_range = [s * 1000 for s in range(100, 160, 5)]
    table = build_payoff_table(option, premium, spot_range)

    print(f"\n  {'Spot':>10} {'Moneyness%':>12} {'Gross Payoff':>14} "
          f"{'Net PnL':>12} {'Retorno%':>10}")
    print("  " + "-" * 62)
    for row in table:
        rop = f"{row['return_pct']:+.1f}%" if row["return_pct"] is not None else "N/A"
        print(f"  {row['spot']:>10,.0f} {row['moneyness_pct']:>11.2f}% "
              f"{row['gross_payoff']:>14.4f} {row['net_pnl']:>12.4f} {rop:>10}")


# ---------------------------------------------------------------------------
# SEÇÃO 9 — Impacto no portfólio
# ---------------------------------------------------------------------------

def secao_9_portfolio_impact(premium: float) -> None:
    """
    Demonstra :func:`options_hedge.compute_portfolio_impact`.
    """
    _header("SEÇÃO 9 — Impacto no portfólio (options_hedge.portfolio)")

    from options_hedge import compute_portfolio_impact

    # Suponha retorno das opções de 50 % sobre o prêmio investido
    # em uma carteira de R$ 10M com alocação de 1 % em opções
    portfolio_size = 10_000_000.0
    options_alloc_pct = 0.01
    options_return_raw = 0.50  # 50 % sobre o prêmio

    options_return_portfolio = options_alloc_pct * options_return_raw

    cenarios = [
        {"label": "Queda severa (-20 % IBOV)", "ibov": -0.20, "spread": -0.02},
        {"label": "Queda moderada (-10 %)",    "ibov": -0.10, "spread": -0.01},
        {"label": "Mercado neutro (0 %)",       "ibov":  0.00, "spread":  0.00},
        {"label": "Alta moderada (+10 %)",      "ibov":  0.10, "spread":  0.01},
    ]

    beta_ibov   = 0.90
    beta_spread = 0.20

    print(f"\n  Carteira: R$ {portfolio_size:,.0f} | "
          f"Alocação em opções: {options_alloc_pct*100:.1f}% | "
          f"β_IBOV={beta_ibov} | β_spread={beta_spread}")
    print(f"\n  {'Cenário':<32} {'Impacto equity':>15} "
          f"{'Impacto opções':>16} {'Impacto total':>14}")
    print("  " + "-" * 80)

    for c in cenarios:
        impact = compute_portfolio_impact(
            ibov_move_pct=c["ibov"],
            smal_ibov_spread_move=c["spread"],
            beta_ibov=beta_ibov,
            beta_spread=beta_spread,
            options_return_pct=options_return_portfolio,
        )
        eq   = impact["equity_impact"] * 100
        opts = impact["options_return_pct"] * 100
        tot  = impact["total_impact"] * 100
        print(f"  {c['label']:<32} {eq:>14.2f}%  {opts:>15.2f}%  {tot:>13.2f}%")


# ---------------------------------------------------------------------------
# SEÇÃO 10 — Decaimento temporal (theta)
# ---------------------------------------------------------------------------

def secao_10_theta(
    option: "OptionSpec",
    market: "MarketState",
    smile: "SmileSurface",
) -> None:
    """
    Demonstra :func:`options_hedge.compute_theta_profile` e
    :func:`options_hedge.price_over_time`.
    """
    _header("SEÇÃO 10 — Decaimento temporal / Theta (options_hedge.theta)")

    from options_hedge import compute_theta_profile, price_over_time

    _sub("10.1  Preço ao longo do tempo (spot e vol estáticos)")
    time_grid = list(range(0, option.expiry_days + 1, 5))
    rows = price_over_time(option, market, time_grid, smile=smile)

    print(f"\n  {'Dias decorridos':>17} {'Dias restantes':>15} "
          f"{'Preço':>10} {'Valor temporal':>15} {'Theta/dia':>11}")
    print("  " + "-" * 72)
    for r in rows:
        print(f"  {r['elapsed_days']:>17}  {r['remaining_days']:>14}  "
              f"{r['option_price']:>10.4f}  {r['time_value']:>14.4f}  "
              f"{r['theta_daily']:>11.4f}")

    _sub("10.2  Perfil de theta completo (steps=7)")
    profile = compute_theta_profile(option, market, total_days=option.expiry_days,
                                    steps=7, smile=smile)
    print(f"\n  {'Dias decorridos':>17} {'Theta/dia':>11}")
    print("  " + "-" * 30)
    for r in profile:
        print(f"  {r['elapsed_days']:>17}  {r['theta_daily']:>11.4f}")


# ---------------------------------------------------------------------------
# Ponto de entrada
# ---------------------------------------------------------------------------

def main() -> None:
    print("\n" + _SEP)
    print("  DEMO — Sistema de Hedge com Opções")
    print("  Sequência: dados → sorriso → simulação com choque")
    print(_SEP)

    with tempfile.TemporaryDirectory() as tmp_dir:
        # 1. Gerar planilha Excel de exemplo
        filepath = secao_1_gerar_dados(tmp_dir)

        # 2. Ler e extrair os sorrisos de volatilidade
        smiles = secao_2_extrair_sorriso(filepath)

        # 3. Converter DataFrame de sorriso em SmileSurface
        smile = secao_3_construir_smile_surface(smiles)

    # 4. Precificar com Black-Scholes (vol plana)
    option, market, premium = secao_4_precificacao_bs()

    # 5. Aplicar choque de vol e reprecificar
    secao_5_choque_vol(option, market, smile, premium)

    # 6. Simular com choque de spot determinado
    secao_6_simular_choque(option, market, smile, premium)

    # 7. Grid completo de cenários + sumário
    secao_7_grid_cenarios(option, market, smile, premium)

    # 8. Break-even e tabela de payoff
    secao_8_breakeven_payoff(option, market, premium)

    # 9. Impacto no portfólio
    secao_9_portfolio_impact(premium)

    # 10. Decaimento temporal
    secao_10_theta(option, market, smile)

    _header("FIM DA DEMONSTRAÇÃO")
    print("  Todos os módulos foram exercitados com sucesso.\n")


if __name__ == "__main__":
    main()
