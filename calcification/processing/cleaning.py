import logging
import unicodedata
from typing import Optional

import numpy as np
import pandas as pd

from calcification.processing import locations, taxonomy, units
from calcification.utils import config, file_ops, utils

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names and clean strings."""
    df.columns = df.columns.str.normalize("NFKC").str.replace("μ", "u")
    df = df.map(
        lambda x: unicodedata.normalize("NFKD", str(x)).replace("\xa0", " ")
        if isinstance(x, str)
        else x
    )
    try:
        mapping = file_ops.read_yaml(config.resources_dir / "mapping.yaml")[
            "sheet_column_map"
        ]
        df.rename(columns=mapping, inplace=True)
    except Exception as e:
        logger.warning(f"Could not apply column mapping: {e}")
    df.columns = (
        df.columns.str.lower().str.replace(" ", "_").str.replace("[()]", "", regex=True)
    )
    return df


def fill_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """Fill metadata columns forward."""
    meta_cols = ["doi", "year", "authors", "location", "species_types", "taxa"]
    missing = [col for col in meta_cols if col not in df.columns]
    if missing:
        logger.warning(f"Missing metadata columns: {missing}")
    present_cols = [col for col in meta_cols if col in df.columns]
    if present_cols:
        df[present_cols] = df[present_cols].infer_objects(copy=False).ffill()
    if "coords" in df.columns and "cleaned_coords" in df.columns:
        df[["coords", "cleaned_coords"]] = df.groupby("doi")[
            ["coords", "cleaned_coords"]
        ].ffill()
    return df


def filter_selection(
    df: pd.DataFrame, selection_dict: Optional[dict] = None
) -> pd.DataFrame:
    """Filter dataframe based on selection dictionary."""
    if selection_dict is None:
        selection_dict = {"include": "yes"}
    for key, value in selection_dict.items():
        if key not in df.columns:
            logger.warning(f"Selection key '{key}' not found in DataFrame columns.")
            continue
        df = (
            df[df[key].isin(value)] if isinstance(value, list) else df[df[key] == value]
        )
    return df


def convert_types(df: pd.DataFrame) -> pd.DataFrame:
    """Convert types of columns."""
    if "year" in df.columns:
        df["year"] = pd.to_datetime(df["year"], format="%Y", errors="coerce")
    if "n" in df.columns and df["n"].dtype == "object":
        df = df[~df["n"].str.contains("~", na=False)]
        df = df[df.n != "M"]
    for col in df.columns:
        if col != "year":
            df[col] = utils.safe_to_numeric(df[col])
    for col in ["irr", "ipar", "sal"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def replace_empty_cells_with_nan(df: pd.DataFrame) -> pd.DataFrame:
    """Replace empty cells with NaN."""
    return df.replace(" ", np.nan)


def remove_unnamed_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove unnamed columns."""
    return df.loc[:, ~df.columns.str.contains("^unnamed")]


def convert_ph_to_hplus(df: pd.DataFrame) -> pd.DataFrame:
    """Convert pH to hplus."""
    try:
        df["hplus"] = df["phtot"].apply(
            lambda p: units.ph_to_hplus(p) if pd.notna(p) else None
        )
    except Exception as e:
        logger.error(f"Error during pH to hplus conversion: {e}")
        raise
    return df


def convert_irr_to_par(df: pd.DataFrame) -> pd.DataFrame:
    """Convert irradiance to PAR."""
    try:
        df["irr"] = df.apply(
            lambda row: units.irradiance_conversion(row["ipar"], "PAR")
            if pd.notna(row["ipar"])
            else row["irr"],
            axis=1,
        )
    except Exception as e:
        logger.error(f"Error during irradiance to PAR conversion: {e}")
        raise
    return df


def calculate_calcification_sd(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate calcification standard deviation."""
    try:
        df["calcification_sd"] = df.apply(
            lambda row: utils.calc_sd_from_se(row["calcification_se"], row["n"])
            if pd.notna(row["calcification_se"]) and pd.notna(row["n"])
            else row.get("calcification_sd"),
            axis=1,
        )
    except Exception as e:
        logger.error(f"Error during calcification standard deviation calculation: {e}")
        raise
    return df


def standardise_calcification_rates(df: pd.DataFrame) -> pd.DataFrame:
    """Standardise calcification rates."""
    try:
        df[["st_calcification", "st_calcification_sd", "st_calcification_unit"]] = (
            df.apply(
                lambda x: pd.Series(
                    units.rate_conversion(
                        x.get("calcification"),
                        x.get("calcification_sd"),
                        x.get("st_calcification_unit"),
                    )
                )
                if pd.notna(x.get("calcification"))
                and pd.notna(x.get("st_calcification_unit"))
                else pd.Series(["", "", ""]),
                axis=1,
            )
        )
    except Exception as e:
        logger.error(f"Error during calcification rate standardisation: {e}")
        raise
    return df


def preprocess_df(
    df: pd.DataFrame, selection_dict: Optional[dict] = None
) -> pd.DataFrame:
    """Clean dataframe fields and standardise for future processing."""
    try:
        df = normalize_columns(df)
        df = fill_metadata(df)
        df = filter_selection(df, selection_dict)
        df = convert_types(df)
        df = remove_unnamed_columns(df)
        df = replace_empty_cells_with_nan(df)
        return df
    except Exception as e:
        logger.error(f"Error during preprocessing: {e}")
        raise


def process_raw_data(
    df: pd.DataFrame,
    require_results: bool = True,
    ph_conversion: bool = True,
    selection_dict: Optional[dict] = None,
) -> pd.DataFrame:
    """Process raw data from the spreadsheet to prepare for analysis."""
    try:
        df = preprocess_df(df, selection_dict)
        df = locations.uniquify_multilocation_study_dois(df)
        df = locations.assign_coordinates(df)
        locations.save_locations_information(df)
        df = locations.assign_ecoregions(df)
        df = taxonomy.assign_taxonomical_info(df)
        df = convert_irr_to_par(df)
        if ph_conversion:
            df = convert_ph_to_hplus(df)
        if require_results:
            df = df.dropna(subset=["n", "calcification"])
        df = units.map_units(df)
        df = standardise_calcification_rates(df)
        df = calculate_calcification_sd(df)
        return df
    except Exception as e:
        logger.error(f"Error during raw data processing: {e}")
        raise
