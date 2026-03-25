"""
create_sample_data.py
=====================
Generates a sample Excel workbook (``options_sample.xlsx``) that demonstrates
the expected input format for :mod:`volatility_smile`.

Each sheet is named to encode option type, underlying and duration so that
:func:`volatility_smile.build_smiles` can parse it automatically, e.g.:

  "BOVA Call 17d"   → call_B_17
  "BOVA Put 17d"    → put_B_17
  "BOVA Call 35d"   → call_B_35
  "BOVA Put 35d"    → put_B_35
  "SMAL Call 17d"   → call_S_17
  "SMAL Put 17d"    → put_S_17
  "SMAL Call 35d"   → call_S_35
  "SMAL Put 35d"    → put_S_35

The columns use intentionally varied naming to show that auto-detection works
regardless of how the columns are labelled in your spreadsheet.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _make_smile_data(
    n_strikes: int = 11,
    atm_iv: float = 0.25,
    skew: float = -0.05,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Return a realistic-looking options table with extra columns (strike, delta,
    gamma, vega, theta, price) in addition to moneyness and implied vol.
    """
    rng = np.random.default_rng(seed)
    moneyness = np.linspace(0.80, 1.20, n_strikes)  # K/S from 0.80 to 1.20

    # Simple quadratic smile + linear skew
    iv = atm_iv + 0.10 * (moneyness - 1.0) ** 2 + skew * (moneyness - 1.0)
    iv = np.clip(iv + rng.normal(0, 0.003, n_strikes), 0.05, 1.0)

    spot = 100.0
    strikes = moneyness * spot
    prices = np.maximum(spot - strikes, 0) + rng.uniform(0.5, 3.0, n_strikes)

    delta = -0.5 + 0.4 * (1 - moneyness) + rng.normal(0, 0.01, n_strikes)
    gamma = 0.02 + rng.uniform(0, 0.005, n_strikes)
    vega = 0.15 + rng.uniform(0, 0.02, n_strikes)
    theta = -0.05 - rng.uniform(0, 0.01, n_strikes)

    return pd.DataFrame(
        {
            "Strike": strikes.round(2),
            "Moneyness": moneyness.round(4),
            "Type": "call",
            "Delta": delta.round(4),
            "Gamma": gamma.round(4),
            "Vega": vega.round(4),
            "Theta": theta.round(4),
            "IV": iv.round(4),          # ← implied vol column (intentional short name)
            "Price": prices.round(2),
        }
    )


# Mapping: sheet name → (atm_iv, skew, seed)
_SHEETS: dict[str, tuple[float, float, int]] = {
    "BOVA Call 17d": (0.22, -0.04, 1),
    "BOVA Put 17d":  (0.24, -0.06, 2),
    "BOVA Call 35d": (0.20, -0.03, 3),
    "BOVA Put 35d":  (0.23, -0.05, 4),
    "SMAL Call 17d": (0.28, -0.07, 5),
    "SMAL Put 17d":  (0.30, -0.08, 6),
    "SMAL Call 35d": (0.26, -0.06, 7),
    "SMAL Put 35d":  (0.29, -0.07, 8),
}


def create_sample_workbook(output_path: str = "options_sample.xlsx") -> None:
    """Write the sample workbook to *output_path*."""
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        for sheet_name, (atm_iv, skew, seed) in _SHEETS.items():
            df = _make_smile_data(atm_iv=atm_iv, skew=skew, seed=seed)
            df.to_excel(writer, sheet_name=sheet_name, index=False)
    print(f"Sample workbook written to: {output_path}")


if __name__ == "__main__":
    create_sample_workbook()
