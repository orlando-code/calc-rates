from unittest import mock

import numpy as np
import pandas as pd
import pytest

from calcification.processing.cleaning import (
    calculate_calcification_sd,
    convert_irr_to_par,
    convert_ph_to_hplus,
    convert_types,
    fill_metadata,
    filter_selection,
    normalize_columns,
    preprocess_df,
    process_raw_data,
    remove_unnamed_columns,
    standardise_calcification_rates,
)


# Mock units and utils dependencies for isolated testing
def dummy_ph_to_hplus(p):
    return 42 if pd.notna(p) else None


def dummy_irradiance_conversion(ipar, mode):
    return ipar * 2 if pd.notna(ipar) else np.nan


def dummy_rate_conversion(calc, calc_sd, unit):
    return (calc, calc_sd, unit)


def dummy_calc_sd_from_se(se, n):
    return se * n if pd.notna(se) and pd.notna(n) else np.nan


def dummy_map_units(df):
    return df


@pytest.fixture(autouse=True)
def patch_dependencies(monkeypatch):
    monkeypatch.setattr("calcification.processing.units.ph_to_hplus", dummy_ph_to_hplus)
    monkeypatch.setattr(
        "calcification.processing.units.irradiance_conversion",
        dummy_irradiance_conversion,
    )
    monkeypatch.setattr(
        "calcification.processing.units.rate_conversion", dummy_rate_conversion
    )
    monkeypatch.setattr("calcification.processing.units.map_units", dummy_map_units)
    monkeypatch.setattr(
        "calcification.utils.utils.calc_sd_from_se", dummy_calc_sd_from_se
    )
    yield


def test_normalize_columns():
    df = pd.DataFrame({"A μmol": [1, 2], "B (test)": ["a", "b"]})
    with mock.patch(
        "calcification.utils.file_ops.read_yaml", return_value={"sheet_column_map": {}}
    ):
        out = normalize_columns(df.copy())
    assert "a_umol" in out.columns or "a_μmol" in out.columns or "a μmol" in out.columns
    assert "b_test" in out.columns or "b (test)" in out.columns


def test_fill_metadata():
    df = pd.DataFrame(
        {
            "doi": [None, "d2"],
            "year": [2020, None],
            "authors": [None, "author2"],
            "location": [None, "loc2"],
            "species_types": [None, "sp2"],
            "taxa": [None, "tax2"],
        }
    )
    out = fill_metadata(df.copy())
    assert out["doi"].isna().sum() == 0
    assert out["year"].isna().sum() == 0


def test_filter_selection():
    df = pd.DataFrame({"include": ["yes", "no", "yes"], "val": [1, 2, 3]})
    out = filter_selection(df, {"include": "yes"})
    assert (out["include"] == "yes").all()
    out2 = filter_selection(df, {"val": [1, 3]})
    assert set(out2["val"]) == {1, 3}


def test_convert_types():
    df = pd.DataFrame(
        {"year": ["2020", "2021"], "n": ["2", "M"], "irr": ["1.0", "bad"]}
    )
    out = convert_types(df.copy())
    assert pd.api.types.is_datetime64_any_dtype(out["year"])
    assert "M" not in out["n"].astype(str).values
    assert np.isnan(out["irr"].iloc[1])


def test_remove_unnamed_columns():
    df = pd.DataFrame({"A": [1], "Unnamed: 0": [2]})
    out = remove_unnamed_columns(df)
    assert "Unnamed: 0" not in out.columns


def test_convert_ph_to_hplus():
    df = pd.DataFrame({"phtot": [7.5, np.nan]})
    out = convert_ph_to_hplus(df.copy())
    assert out["hplus"].iloc[0] == 42
    assert pd.isna(out["hplus"].iloc[1])


def test_convert_irr_to_par():
    df = pd.DataFrame({"irr": [1, 2], "ipar": [10, np.nan]})
    out = convert_irr_to_par(df.copy())
    assert out["irr"].iloc[0] == 20
    assert np.isnan(out["irr"].iloc[1])


def test_calculate_calcification_sd():
    df = pd.DataFrame({"calcification_se": [2, np.nan], "n": [3, 4]})
    out = calculate_calcification_sd(df.copy())
    assert out["calcification_sd"].iloc[0] == 6
    assert np.isnan(out["calcification_sd"].iloc[1])


def test_standardise_calcification_rates():
    df = pd.DataFrame(
        {
            "calcification": [1, 2],
            "calcification_sd": [0.1, 0.2],
            "st_calcification_unit": ["u1", "u2"],
        }
    )
    out = standardise_calcification_rates(df.copy())
    assert (out["st_calcification"] == out["calcification"]).all()
    assert (out["st_calcification_sd"] == out["calcification_sd"]).all()
    assert (out["st_calcification_unit"] == out["st_calcification_unit"]).all()


def test_preprocess_df():
    df = pd.DataFrame(
        {
            "A μmol": [1],
            "doi": ["d1"],
            "year": ["2020"],
            "authors": ["a"],
            "location": ["l"],
            "species_types": ["s"],
            "taxa": ["t"],
            "include": ["yes"],
        }
    )
    with mock.patch(
        "calcification.utils.file_ops.read_yaml", return_value={"sheet_column_map": {}}
    ):
        out = preprocess_df(df.copy(), {"include": "yes"})
    assert "a_umol" in out.columns or "a_μmol" in out.columns or "a μmol" in out.columns
    assert out.shape[0] == 1


def test_process_raw_data(monkeypatch):
    # Patch locations and taxonomy functions to be identity
    monkeypatch.setattr(
        "calcification.processing.locations.uniquify_multilocation_study_dois",
        lambda df: df,
    )
    monkeypatch.setattr(
        "calcification.processing.locations.assign_coordinates", lambda df: df
    )
    monkeypatch.setattr(
        "calcification.processing.locations.save_locations_information", lambda df: None
    )
    monkeypatch.setattr(
        "calcification.processing.locations.assign_ecoregions", lambda df: df
    )
    monkeypatch.setattr(
        "calcification.processing.taxonomy.assign_taxonomical_info", lambda df: df
    )
    df = pd.DataFrame(
        {
            "A μmol": [1],
            "doi": ["d1"],
            "year": ["2020"],
            "authors": ["a"],
            "location": ["l"],
            "species_types": ["s"],
            "taxa": ["t"],
            "include": ["yes"],
            "n": [1],
            "calcification": [1],
            "calcification_sd": [0.1],
            "st_calcification_unit": ["u"],
        }
    )
    with mock.patch(
        "calcification.utils.file_ops.read_yaml", return_value={"sheet_column_map": {}}
    ):
        out = process_raw_data(
            df.copy(),
            require_results=True,
            ph_conversion=True,
            selection_dict={"include": "yes"},
        )
    assert out.shape[0] == 1
    assert "st_calcification" in out.columns
