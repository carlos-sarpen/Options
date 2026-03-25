"""
test_volatility_smile.py
========================
Unit tests for :mod:`volatility_smile`.
"""

from __future__ import annotations

import warnings
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from volatility_smile import (
    _detect_column,
    _parse_sheet_name,
    _smile_key,
    build_smiles,
    smiles_to_dataframe,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_sheet(
    moneyness_col: str = "Moneyness",
    iv_col: str = "IV",
    n: int = 5,
    extra_cols: dict | None = None,
) -> pd.DataFrame:
    """Return a minimal options DataFrame."""
    rng = np.random.default_rng(0)
    data = {
        moneyness_col: np.linspace(0.90, 1.10, n),
        iv_col: 0.20 + 0.05 * (np.linspace(0.90, 1.10, n) - 1.0) ** 2,
    }
    if extra_cols:
        data.update(extra_cols)
    return pd.DataFrame(data)


def _write_workbook(sheets: dict[str, pd.DataFrame]) -> BytesIO:
    """Write *sheets* to an in-memory Excel workbook and return it."""
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        for name, df in sheets.items():
            df.to_excel(writer, sheet_name=name, index=False)
    buf.seek(0)
    return buf


# ---------------------------------------------------------------------------
# _detect_column
# ---------------------------------------------------------------------------

class TestDetectColumn:
    def test_exact_keyword_match(self):
        cols = pd.Index(["Strike", "Moneyness", "IV", "Price"])
        assert _detect_column(cols, ["moneyness"]) == "Moneyness"

    def test_partial_keyword_match(self):
        cols = pd.Index(["Strike", "Log_Moneyness", "Impl_Vol"])
        assert _detect_column(cols, ["log_moneyness"]) == "Log_Moneyness"

    def test_iv_keyword_variants(self):
        for col_name, keyword in [
            ("ImpliedVol", "impliedvol"),
            ("vol_imp", "vol_imp"),
            ("IV", "iv"),
            ("Sigma", "sigma"),
            ("volatilidade", "volatilidade"),
        ]:
            cols = pd.Index([col_name, "Strike"])
            result = _detect_column(cols, [keyword])
            assert result == col_name, f"Expected '{col_name}', got '{result}'"

    def test_no_match_returns_none(self):
        cols = pd.Index(["Strike", "Price", "Delta"])
        assert _detect_column(cols, ["moneyness", "money"]) is None

    def test_first_keyword_wins(self):
        cols = pd.Index(["moneyness", "money_ratio"])
        result = _detect_column(cols, ["moneyness", "money"])
        assert result == "moneyness"


# ---------------------------------------------------------------------------
# _parse_sheet_name
# ---------------------------------------------------------------------------

class TestParseSheetName:
    @pytest.mark.parametrize("sheet,expected", [
        ("BOVA Call 17d",   {"option_type": "call", "underlying": "B", "duration": "17"}),
        ("BOVA Put 17d",    {"option_type": "put",  "underlying": "B", "duration": "17"}),
        ("BOVA Call 35d",   {"option_type": "call", "underlying": "B", "duration": "35"}),
        ("BOVA Put 35d",    {"option_type": "put",  "underlying": "B", "duration": "35"}),
        ("SMAL Call 17d",   {"option_type": "call", "underlying": "S", "duration": "17"}),
        ("SMAL Put 17d",    {"option_type": "put",  "underlying": "S", "duration": "17"}),
        ("SMAL Call 35d",   {"option_type": "call", "underlying": "S", "duration": "35"}),
        ("SMAL Put 35d",    {"option_type": "put",  "underlying": "S", "duration": "35"}),
        ("call_bova_17",    {"option_type": "call", "underlying": "B", "duration": "17"}),
        ("put-SMAL-35",     {"option_type": "put",  "underlying": "S", "duration": "35"}),
    ])
    def test_known_formats(self, sheet, expected):
        assert _parse_sheet_name(sheet) == expected

    def test_missing_type_returns_none(self):
        assert _parse_sheet_name("BOVA 17d") is None

    def test_missing_underlying_returns_none(self):
        assert _parse_sheet_name("Call 35d") is None

    def test_missing_duration_returns_none(self):
        assert _parse_sheet_name("BOVA Call") is None

    def test_case_insensitive(self):
        result = _parse_sheet_name("BOVA CALL 17")
        assert result == {"option_type": "call", "underlying": "B", "duration": "17"}


# ---------------------------------------------------------------------------
# _smile_key
# ---------------------------------------------------------------------------

class TestSmileKey:
    def test_key_format(self):
        assert _smile_key({"option_type": "call", "underlying": "B", "duration": "17"}) == "call_B_17"
        assert _smile_key({"option_type": "put",  "underlying": "S", "duration": "35"}) == "put_S_35"


# ---------------------------------------------------------------------------
# build_smiles
# ---------------------------------------------------------------------------

class TestBuildSmiles:
    def test_basic_happy_path(self, tmp_path):
        sheets = {
            "BOVA Call 17d": _make_sheet(),
            "BOVA Put 35d":  _make_sheet(iv_col="ImpliedVol"),
        }
        buf = _write_workbook(sheets)
        path = tmp_path / "test.xlsx"
        path.write_bytes(buf.read())

        smiles = build_smiles(path)
        assert set(smiles.keys()) == {"call_B_17", "put_B_35"}

    def test_output_columns(self, tmp_path):
        path = tmp_path / "test.xlsx"
        path.write_bytes(_write_workbook({"BOVA Call 17d": _make_sheet()}).read())

        smiles = build_smiles(path)
        df = smiles["call_B_17"]
        assert list(df.columns) == ["moneyness", "implied_vol"]

    def test_sorted_by_moneyness(self, tmp_path):
        df_in = _make_sheet().iloc[::-1]  # reverse order
        path = tmp_path / "test.xlsx"
        path.write_bytes(_write_workbook({"BOVA Call 17d": df_in}).read())

        smiles = build_smiles(path, sort_by_moneyness=True)
        mon = smiles["call_B_17"]["moneyness"]
        assert list(mon) == sorted(mon)

    def test_drop_na(self, tmp_path):
        df_in = _make_sheet()
        df_in.loc[2, "Moneyness"] = float("nan")
        path = tmp_path / "test.xlsx"
        path.write_bytes(_write_workbook({"BOVA Call 17d": df_in}).read())

        smiles = build_smiles(path, drop_na=True)
        assert smiles["call_B_17"]["moneyness"].isna().sum() == 0

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            build_smiles("/nonexistent/path/file.xlsx")

    def test_unknown_sheet_skipped_with_warning(self, tmp_path):
        sheets = {
            "BOVA Call 17d": _make_sheet(),
            "Metadata":       pd.DataFrame({"info": ["some text"]}),
        }
        path = tmp_path / "test.xlsx"
        path.write_bytes(_write_workbook(sheets).read())

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            smiles = build_smiles(path)

        assert "call_B_17" in smiles
        assert any("Metadata" in str(warning.message) for warning in w)

    def test_missing_moneyness_col_skipped_with_warning(self, tmp_path):
        df_no_mon = pd.DataFrame({"Strike": [95, 100, 105], "IV": [0.25, 0.20, 0.22]})
        path = tmp_path / "test.xlsx"
        path.write_bytes(_write_workbook({"BOVA Call 17d": df_no_mon}).read())

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            smiles = build_smiles(path)

        assert smiles == {}
        assert any("moneyness" in str(warning.message).lower() for warning in w)

    def test_missing_iv_col_skipped_with_warning(self, tmp_path):
        df_no_iv = pd.DataFrame({"Moneyness": [0.95, 1.0, 1.05], "Price": [5.0, 3.0, 2.0]})
        path = tmp_path / "test.xlsx"
        path.write_bytes(_write_workbook({"BOVA Call 17d": df_no_iv}).read())

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            smiles = build_smiles(path)

        assert smiles == {}
        assert any("implied" in str(warning.message).lower() or
                   "volatility" in str(warning.message).lower()
                   for warning in w)

    def test_column_override(self, tmp_path):
        df = _make_sheet(moneyness_col="K_S", iv_col="Vol_Impl")
        path = tmp_path / "test.xlsx"
        path.write_bytes(_write_workbook({"BOVA Call 17d": df}).read())

        smiles = build_smiles(path, moneyness_col="K_S", iv_col="Vol_Impl")
        assert "call_B_17" in smiles

    def test_all_eight_sheets(self, tmp_path):
        """All 8 combinations (call/put × BOVA/SMAL × 17/35) are parsed."""
        sheet_names = [
            "BOVA Call 17d", "BOVA Put 17d",
            "BOVA Call 35d", "BOVA Put 35d",
            "SMAL Call 17d", "SMAL Put 17d",
            "SMAL Call 35d", "SMAL Put 35d",
        ]
        sheets = {n: _make_sheet() for n in sheet_names}
        path = tmp_path / "test.xlsx"
        path.write_bytes(_write_workbook(sheets).read())

        smiles = build_smiles(path)
        expected_keys = {
            "call_B_17", "put_B_17", "call_B_35", "put_B_35",
            "call_S_17", "put_S_17", "call_S_35", "put_S_35",
        }
        assert set(smiles.keys()) == expected_keys


# ---------------------------------------------------------------------------
# smiles_to_dataframe
# ---------------------------------------------------------------------------

class TestSmilesToDataframe:
    def test_combined_shape(self):
        smiles = {
            "call_B_17": pd.DataFrame({"moneyness": [0.9, 1.0], "implied_vol": [0.22, 0.20]}),
            "put_S_35":  pd.DataFrame({"moneyness": [0.9, 1.0], "implied_vol": [0.25, 0.23]}),
        }
        result = smiles_to_dataframe(smiles)
        assert list(result.columns) == ["smile_id", "moneyness", "implied_vol"]
        assert len(result) == 4

    def test_smile_id_values(self):
        smiles = {
            "call_B_17": pd.DataFrame({"moneyness": [1.0], "implied_vol": [0.20]}),
        }
        result = smiles_to_dataframe(smiles)
        assert result["smile_id"].iloc[0] == "call_B_17"

    def test_empty_input(self):
        result = smiles_to_dataframe({})
        assert list(result.columns) == ["smile_id", "moneyness", "implied_vol"]
        assert len(result) == 0
