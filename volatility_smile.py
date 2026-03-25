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
  duration    : "17" or "35"  (business days to expiry)

Examples: "call_B_17", "put_S_35"

Sheet name formats supported
-----------------------------
Two formats are recognised (case-insensitive):

  1. All-in-name  – option type, underlying **and** duration are all part of
     the sheet name, e.g. ``"BOVA Call 17d"``, ``"call_bova_17"``.

  2. Ticker-duration  – only the underlying and duration are in the sheet
     name (e.g. ``"BOVA-17du"``, ``"SMAL-35du"``).  In this case the option
     type (call/put) is read from a dedicated column inside the sheet itself
     (auto-detected via keywords: "tipo", "type", etc.).

Column auto-detection
---------------------
Column names are matched case-insensitively as substrings.

  Moneyness : moneyness, money, ln_strike, log_moneyness, m/s, k/s,
              strike_spot, k_s, dist
  Implied vol: vol_impl, implied_vol, impliedvol, iv, sigma, volatilidade,
              vol_imp, impvol, imp_vol, impl
  Option type: tipo, type, opt_type, option_type

If a sheet has more than one match for a given axis the first match wins;
if no match is found the sheet is skipped with a warning.

Title rows
----------
Some workbooks have one or more title/metadata rows before the real header.
Use the ``header_row`` parameter (0-based index of the header row) to skip
them.  For example, a workbook whose first two rows are titles and whose
third row contains column names needs ``header_row=2``.

Usage
-----
>>> from volatility_smile import build_smiles
>>> # workbook where header is the 3rd row (index 2) and option type is in
>>> # a "Tipo" column inside each sheet
>>> smiles = build_smiles("options_data.xlsx", header_row=2)
>>> smiles["call_B_17"]
   moneyness  implied_vol
0      -2.68       26.40
1       0.00       25.00
...
>>> # return full dataframe instead of just the smile pair
>>> smiles_full = build_smiles("options_data.xlsx", header_row=2, full_dataframe=True)
>>> smiles_full["call_B_17"].columns
Index(['Ticker', 'Tipo', ..., 'moneyness', ..., 'implied_vol', ...])
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
    "dist",
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
    "impl",
]

# Keywords used to locate the option-type column when it is not encoded in
# the sheet name (e.g. a "Tipo" column with values "CALL" / "PUT").
_TIPO_KEYWORDS: list[str] = [
    "tipo",
    "type",
    "opt_type",
    "option_type",
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
    "duration": "17"}`` or ``None`` if underlying or duration cannot be
    determined.

    When the option type is **not** encoded in the sheet name (e.g.
    ``"BOVA-17du"``), ``option_type`` is ``None`` – the caller must infer it
    from a dedicated column inside the sheet (see :data:`_TIPO_KEYWORDS`).
    """
    m_type = _OPTION_TYPE_RE.search(sheet_name)
    m_under = _UNDERLYING_RE.search(sheet_name)
    m_dur = _DURATION_RE.search(sheet_name)

    if not (m_under and m_dur):
        return None

    underlying_raw = m_under.group(1).lower()
    underlying = "B" if underlying_raw == "bova" else "S"
    duration = m_dur.group(1)
    option_type = m_type.group(1).lower() if m_type else None

    return {"option_type": option_type, "underlying": underlying, "duration": duration}


def _smile_key(attrs: dict[str, str]) -> str:
    """Build the output key from parsed sheet attributes."""
    return f"{attrs['option_type']}_{attrs['underlying']}_{attrs['duration']}"


def _build_single_smile(
    sheet: str,
    df: pd.DataFrame,
    attrs: dict[str, str],
    moneyness_col: Optional[str],
    iv_col: Optional[str],
    *,
    sort_by_moneyness: bool,
    drop_na: bool,
    full_dataframe: bool,
) -> Optional[pd.DataFrame]:
    """
    Extract the smile DataFrame from *df*.

    Returns a DataFrame (smile pair or full) or ``None`` if required columns
    are missing.
    """
    mon_col = moneyness_col or _detect_column(df.columns, _MONEYNESS_KEYWORDS)
    if mon_col is None:
        warnings.warn(
            f"Sheet '{sheet}' skipped – no moneyness column found. "
            f"Columns: {list(df.columns)}",
            stacklevel=3,
        )
        return None

    iv_c = iv_col or _detect_column(df.columns, _IV_KEYWORDS)
    if iv_c is None:
        warnings.warn(
            f"Sheet '{sheet}' skipped – no implied-volatility column found. "
            f"Columns: {list(df.columns)}",
            stacklevel=3,
        )
        return None

    if full_dataframe:
        result = df.copy()
        # Standardise the two key columns while keeping all others.
        result = result.rename(
            columns={mon_col: "moneyness", iv_c: "implied_vol"},
            errors="ignore",
        )
    else:
        result = df[[mon_col, iv_c]].copy()
        result.columns = pd.Index(["moneyness", "implied_vol"])

    result["moneyness"] = pd.to_numeric(result["moneyness"], errors="coerce")
    result["implied_vol"] = pd.to_numeric(result["implied_vol"], errors="coerce")

    if drop_na:
        result = result.dropna(subset=["moneyness", "implied_vol"])

    if sort_by_moneyness:
        result = result.sort_values("moneyness").reset_index(drop=True)
    else:
        result = result.reset_index(drop=True)

    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_smiles(
    filepath: str | Path,
    header_row: int = 0,
    moneyness_col: Optional[str] = None,
    iv_col: Optional[str] = None,
    *,
    sort_by_moneyness: bool = True,
    drop_na: bool = True,
    full_dataframe: bool = False,
) -> Dict[str, pd.DataFrame]:
    """
    Read *filepath* (Excel .xlsx/.xls) and return a dictionary of volatility
    smile DataFrames, one per recognised sheet / option-type combination.

    Parameters
    ----------
    filepath:
        Path to the Excel workbook.
    header_row:
        Zero-based row index that contains the column headers.  Use ``2``
        when the first two rows of every sheet are titles that should be
        discarded and the real header is on the third row.  Defaults to
        ``0`` (first row is the header).
    moneyness_col:
        Override auto-detection – use exactly this column name for moneyness.
    iv_col:
        Override auto-detection – use exactly this column name for implied vol.
    sort_by_moneyness:
        If ``True`` (default), each smile DataFrame is sorted by moneyness.
    drop_na:
        If ``True`` (default), rows where moneyness or IV is NaN are dropped.
    full_dataframe:
        If ``False`` (default), each returned DataFrame contains exactly two
        columns: ``"moneyness"`` and ``"implied_vol"``.
        If ``True``, the full sheet DataFrame is returned with all original
        columns; the moneyness and implied-vol columns are renamed to
        ``"moneyness"`` and ``"implied_vol"`` for consistency.

    Returns
    -------
    dict[str, pd.DataFrame]
        Keys are strings like ``"call_B_17"``, ``"put_S_35"`` etc.
        When *full_dataframe* is ``False``, each DataFrame has exactly two
        columns: ``"moneyness"`` and ``"implied_vol"``.
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
                f"Sheet '{sheet}' skipped – could not determine underlying "
                "or duration from its name.",
                stacklevel=2,
            )
            continue

        df = xl.parse(sheet, header=header_row)

        if attrs["option_type"] is None:
            # Sheet name carries only ticker + duration (e.g. "BOVA-17du").
            # Split the sheet by the option-type column (e.g. "Tipo").
            tipo_col = _detect_column(df.columns, _TIPO_KEYWORDS)
            if tipo_col is None:
                warnings.warn(
                    f"Sheet '{sheet}' skipped – option type not in sheet name "
                    "and no option-type column found. "
                    f"Columns: {list(df.columns)}",
                    stacklevel=2,
                )
                continue

            for raw_type in ("CALL", "PUT"):
                mask = df[tipo_col].astype(str).str.upper().str.strip() == raw_type
                subset = df[mask].copy()
                if subset.empty:
                    continue
                opt_attrs = {**attrs, "option_type": raw_type.lower()}
                smile = _build_single_smile(
                    sheet,
                    subset,
                    opt_attrs,
                    moneyness_col,
                    iv_col,
                    sort_by_moneyness=sort_by_moneyness,
                    drop_na=drop_na,
                    full_dataframe=full_dataframe,
                )
                if smile is not None:
                    smiles[_smile_key(opt_attrs)] = smile
        else:
            smile = _build_single_smile(
                sheet,
                df,
                attrs,
                moneyness_col,
                iv_col,
                sort_by_moneyness=sort_by_moneyness,
                drop_na=drop_na,
                full_dataframe=full_dataframe,
            )
            if smile is not None:
                smiles[_smile_key(attrs)] = smile

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
