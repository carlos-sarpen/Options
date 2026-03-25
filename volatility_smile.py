"""
volatility_smile.py
===================
Reads an Excel workbook with multiple sheets containing options data and
builds "volatility smile" structures – i.e. (moneyness, implied-volatility)
pairs – from every sheet.

Output naming
-------------
Each smile is stored under a key that encodes three attributes, separated by
underscores:

  <option_type>_<underlying>_<duration>

  option_type : "call" or "put"
  underlying  : "B" (BOVA) or "S" (SMAL)
  duration    : "17" or "35"  (days to expiry)

Examples: "call_B_17", "put_S_35"

The values are taken from the sheet name.  Recognised keywords (case-
insensitive):

  option_type : "call" / "put"
  underlying  : "bova" → "B",  "smal" → "S"
  duration    : "17" → "17",   "35"  → "35"

Column auto-detection
---------------------
Moneyness columns are identified by any column whose name contains one of:
  moneyness, money, ln_strike, log_moneyness, m/s, k/s, strike_spot

Implied-volatility columns are identified by any column whose name contains
one of:
  vol_impl, implied_vol, impliedvol, iv, sigma, volatilidade, vol_imp

Both searches are case-insensitive.  If a sheet has more than one match the
first match wins; if no match is found the sheet is skipped with a warning.

Usage
-----
>>> from volatility_smile import build_smiles
>>> smiles = build_smiles("options_data.xlsx")
>>> smiles["call_B_17"]
   moneyness  implied_vol
0      0.90       0.2543
1      0.95       0.2201
...
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path
from typing import Dict, Optional

import pandas as pd


# ---------------------------------------------------------------------------
# Column-detection keyword lists
# ---------------------------------------------------------------------------

_MONEYNESS_KEYWORDS: list[str] = [
    "moneyness",
    "money",
    "ln_strike",
    "log_moneyness",
    "m/s",
    "k/s",
    "strike_spot",
    "k_s",
]

_IV_KEYWORDS: list[str] = [
    "vol_impl",
    "implied_vol",
    "impliedvol",
    "iv",
    "sigma",
    "volatilidade",
    "vol_imp",
    "impvol",
    "imp_vol",
]

# ---------------------------------------------------------------------------
# Sheet-name parsing patterns
# ---------------------------------------------------------------------------

_OPTION_TYPE_RE = re.compile(r"(?<![a-zA-Z])(call|put)(?![a-zA-Z])", re.IGNORECASE)
_UNDERLYING_RE = re.compile(r"(?<![a-zA-Z])(bova|smal)(?![a-zA-Z])", re.IGNORECASE)
_DURATION_RE = re.compile(r"(?<!\d)(17|35)(?!\d)")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _detect_column(columns: pd.Index, keywords: list[str]) -> Optional[str]:
    """Return the first column name that contains any of *keywords*."""
    columns_lower = [c.lower().strip() for c in columns]
    for kw in keywords:
        for col_lower, col_orig in zip(columns_lower, columns):
            if kw in col_lower:
                return col_orig
    return None


def _parse_sheet_name(sheet_name: str) -> Optional[dict[str, str]]:
    """
    Extract option_type, underlying and duration from *sheet_name*.

    Returns a dict like ``{"option_type": "call", "underlying": "B",
    "duration": "17"}`` or ``None`` if any attribute is missing.
    """
    m_type = _OPTION_TYPE_RE.search(sheet_name)
    m_under = _UNDERLYING_RE.search(sheet_name)
    m_dur = _DURATION_RE.search(sheet_name)

    if not (m_type and m_under and m_dur):
        return None

    option_type = m_type.group(1).lower()
    underlying_raw = m_under.group(1).lower()
    underlying = "B" if underlying_raw == "bova" else "S"
    duration = m_dur.group(1)

    return {"option_type": option_type, "underlying": underlying, "duration": duration}


def _smile_key(attrs: dict[str, str]) -> str:
    """Build the output key from parsed sheet attributes."""
    return f"{attrs['option_type']}_{attrs['underlying']}_{attrs['duration']}"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_smiles(
    filepath: str | Path,
    moneyness_col: Optional[str] = None,
    iv_col: Optional[str] = None,
    *,
    sort_by_moneyness: bool = True,
    drop_na: bool = True,
) -> Dict[str, pd.DataFrame]:
    """
    Read *filepath* (Excel .xlsx/.xls) and return a dictionary of volatility
    smile DataFrames, one per recognised sheet.

    Parameters
    ----------
    filepath:
        Path to the Excel workbook.
    moneyness_col:
        Override auto-detection – use exactly this column name for moneyness.
    iv_col:
        Override auto-detection – use exactly this column name for implied vol.
    sort_by_moneyness:
        If True (default), each smile DataFrame is sorted by moneyness.
    drop_na:
        If True (default), rows where moneyness or IV is NaN are dropped.

    Returns
    -------
    dict[str, pd.DataFrame]
        Keys are strings like ``"call_B_17"``, ``"put_S_35"`` etc.
        Each DataFrame has exactly two columns: ``"moneyness"`` and
        ``"implied_vol"``.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    xl = pd.ExcelFile(filepath)
    smiles: Dict[str, pd.DataFrame] = {}

    for sheet in xl.sheet_names:
        attrs = _parse_sheet_name(sheet)
        if attrs is None:
            warnings.warn(
                f"Sheet '{sheet}' skipped – could not determine option type, "
                "underlying or duration from its name.",
                stacklevel=2,
            )
            continue

        df = xl.parse(sheet)

        # --- locate moneyness column ---
        mon_col = moneyness_col or _detect_column(df.columns, _MONEYNESS_KEYWORDS)
        if mon_col is None:
            warnings.warn(
                f"Sheet '{sheet}' skipped – no moneyness column found. "
                f"Columns: {list(df.columns)}",
                stacklevel=2,
            )
            continue

        # --- locate implied-vol column ---
        iv_c = iv_col or _detect_column(df.columns, _IV_KEYWORDS)
        if iv_c is None:
            warnings.warn(
                f"Sheet '{sheet}' skipped – no implied-volatility column found. "
                f"Columns: {list(df.columns)}",
                stacklevel=2,
            )
            continue

        smile = df[[mon_col, iv_c]].copy()
        smile.columns = pd.Index(["moneyness", "implied_vol"])
        smile["moneyness"] = pd.to_numeric(smile["moneyness"], errors="coerce")
        smile["implied_vol"] = pd.to_numeric(smile["implied_vol"], errors="coerce")

        if drop_na:
            smile = smile.dropna(subset=["moneyness", "implied_vol"])

        if sort_by_moneyness:
            smile = smile.sort_values("moneyness").reset_index(drop=True)
        else:
            smile = smile.reset_index(drop=True)

        key = _smile_key(attrs)
        smiles[key] = smile

    return smiles


def smiles_to_dataframe(smiles: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Combine all smile DataFrames into a single tidy DataFrame with an extra
    column ``"smile_id"`` that carries the key (e.g. ``"call_B_17"``).

    Parameters
    ----------
    smiles:
        Dictionary returned by :func:`build_smiles`.

    Returns
    -------
    pd.DataFrame
        Always has exactly three columns: ``["smile_id", "moneyness",
        "implied_vol"]``.  Empty (zero rows) when *smiles* is empty.
    """
    frames = []
    for key, df in smiles.items():
        tmp = df.copy()
        tmp.insert(0, "smile_id", key)
        frames.append(tmp)
    if not frames:
        return pd.DataFrame(columns=["smile_id", "moneyness", "implied_vol"])
    return pd.concat(frames, ignore_index=True)
