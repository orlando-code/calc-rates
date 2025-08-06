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
    """Standardise calcification rates with proper error handling."""
    df_copy = df.copy()

    # Track problematic rows
    problematic_rows = []
    successful_conversions = 0

    for idx, row in df_copy.iterrows():
        try:
            # Check if we have the required data for conversion
            if pd.notna(row.get("calcification")) and pd.notna(
                row.get("st_calcification_unit")
            ):
                # Perform the conversion
                converted_rate, converted_sd, converted_unit = units.rate_conversion(
                    row.get("calcification"),
                    row.get("calcification_sd"),
                    row.get("st_calcification_unit"),
                )

                # Check if conversion was successful (not just copying original values)
                if (
                    converted_unit
                    and converted_unit != ""
                    and not converted_unit.startswith("Error")
                ):
                    df_copy.loc[idx, "st_calcification"] = converted_rate
                    df_copy.loc[idx, "st_calcification_sd"] = converted_sd
                    df_copy.loc[idx, "st_calcification_unit"] = converted_unit
                    successful_conversions += 1
                else:
                    # Conversion failed or returned error
                    problematic_rows.append(
                        {
                            "index": idx,
                            "doi": row.get("doi", "Unknown"),
                            "calcification": row.get("calcification"),
                            "calcification_unit": row.get("calcification_unit"),
                            "st_calcification_unit": row.get("st_calcification_unit"),
                            "error": f"Conversion failed: {converted_unit}"
                            if converted_unit
                            else "No unit mapping",
                        }
                    )
            else:
                # Missing required data
                missing_data = []
                if pd.isna(row.get("calcification")):
                    missing_data.append("calcification")
                if pd.isna(row.get("st_calcification_unit")):
                    missing_data.append("st_calcification_unit")

                problematic_rows.append(
                    {
                        "index": idx,
                        "doi": row.get("doi", "Unknown"),
                        "calcification": row.get("calcification"),
                        "calcification_unit": row.get("calcification_unit"),
                        "st_calcification_unit": row.get("st_calcification_unit"),
                        "error": f"Missing required data: {', '.join(missing_data)}",
                    }
                )

        except Exception as e:
            # Individual row conversion failed
            problematic_rows.append(
                {
                    "index": idx,
                    "doi": row.get("doi", "Unknown"),
                    "calcification": row.get("calcification"),
                    "calcification_unit": row.get("calcification_unit"),
                    "st_calcification_unit": row.get("st_calcification_unit"),
                    "error": f"Exception: {str(e)}",
                }
            )

    # Log results
    total_rows = len(df_copy)
    logger.info(
        f"Unit standardization complete: {successful_conversions}/{total_rows} rows converted successfully"
    )

    if problematic_rows:
        logger.warning(f"Found {len(problematic_rows)} problematic rows:")
        for problem in problematic_rows[:10]:  # Show first 10 problems
            logger.warning(
                f"  Row {problem['index']} (DOI: {problem['doi']}): {problem['error']}"
            )
        if len(problematic_rows) > 10:
            logger.warning(
                f"  ... and {len(problematic_rows) - 10} more problematic rows"
            )

    return df_copy


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
        df = locations.assign_coordinates(df)
        df = locations.uniquify_multilocation_study_dois(df)
        locations.save_locations_information(df)
        df = locations.assign_ecoregions(df)
        df = taxonomy.assign_taxonomical_info(df)
        df = convert_irr_to_par(df)
        if ph_conversion:
            df = convert_ph_to_hplus(df)
        if require_results:
            df = df.dropna(subset=["n", "calcification"])
        df = units.map_units(df)
        df = calculate_calcification_sd(df)
        df = standardise_calcification_rates(df)
        return df
    except Exception as e:
        logger.error(f"Error during raw data processing: {e}")
        raise
