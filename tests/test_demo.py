"""
tests/testdemo_module.py
==================
Smoke tests for demo.py — ensures the central demo script runs without errors
and that each section function executes and returns the expected types.
"""

from __future__ import annotations

import os
import sys
import tempfile

import pytest

# Make sure the project root is on the path (mirrors how pytest is configured)
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import demo as demo_module


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_smile():
    """Return a SmileSurface built from create_sample_data + build_smiles."""
    from create_sample_data import create_sample_workbook
    from volatility_smile import build_smiles
    from options_hedge import SmileSurface

    with tempfile.TemporaryDirectory() as tmp:
        fp = os.path.join(tmp, "test.xlsx")
        create_sample_workbook(fp)
        smiles = build_smiles(fp)

    df = smiles["call_B_17"]
    return SmileSurface(
        points=dict(zip(df["moneyness"].tolist(), df["implied_vol"].tolist()))
    )


# ---------------------------------------------------------------------------
# Section-level smoke tests
# ---------------------------------------------------------------------------

class TestSections:

    def test_section1_generates_workbook(self, tmp_path):
        fp = demo_module.secao_1_gerar_dados(str(tmp_path))
        assert os.path.exists(fp)

    def test_section2_returns_eight_smiles(self, tmp_path):
        fp = demo_module.secao_1_gerar_dados(str(tmp_path))
        smiles = demo_module.secao_2_extrair_sorriso(fp)
        assert len(smiles) == 8
        expected_keys = {
            "call_B_17", "put_B_17", "call_B_35", "put_B_35",
            "call_S_17", "put_S_17", "call_S_35", "put_S_35",
        }
        assert set(smiles.keys()) == expected_keys

    def test_section3_returns_smile_surface(self, tmp_path):
        fp = demo_module.secao_1_gerar_dados(str(tmp_path))
        smiles = demo_module.secao_2_extrair_sorriso(fp)
        from options_hedge import SmileSurface
        smile = demo_module.secao_3_construir_smile_surface(smiles)
        assert isinstance(smile, SmileSurface)
        assert len(smile.points) == 11

    def test_section4_returns_option_market_premium(self):
        option, market, premium = demo_module.secao_4_precificacao_bs()
        assert option.option_type == "call"
        assert market.spot == 128_000.0
        assert premium > 0

    def test_section5_runs_without_error(self):
        option, market, premium = demo_module.secao_4_precificacao_bs()
        smile = _build_smile()
        demo_module.secao_5_choque_vol(option, market, smile, premium)

    def test_section6_runs_without_error(self):
        option, market, premium = demo_module.secao_4_precificacao_bs()
        smile = _build_smile()
        demo_module.secao_6_simular_choque(option, market, smile, premium)

    def test_section7_runs_without_error(self):
        option, market, premium = demo_module.secao_4_precificacao_bs()
        smile = _build_smile()
        demo_module.secao_7_grid_cenarios(option, market, smile, premium)

    def test_section8_runs_without_error(self):
        option, market, premium = demo_module.secao_4_precificacao_bs()
        demo_module.secao_8_breakeven_payoff(option, market, premium)

    def test_section9_runs_without_error(self):
        option, market, premium = demo_module.secao_4_precificacao_bs()
        demo_module.secao_9_portfolio_impact(premium)

    def test_section10_runs_without_error(self):
        option, market, premium = demo_module.secao_4_precificacao_bs()
        smile = _build_smile()
        demo_module.secao_10_theta(option, market, smile)


# ---------------------------------------------------------------------------
# End-to-end: the main() entry-point completes without raising
# ---------------------------------------------------------------------------

class TestMainEntrypoint:

    def test_main_runs_without_error(self):
        demo_module.main()
