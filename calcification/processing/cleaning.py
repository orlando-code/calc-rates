# general
import unicodedata

import numpy as np
import pandas as pd

from calcification.processing import locations, taxonomy, units
from calcification.utils import config, file_ops, utils


def preprocess_df(
    df: pd.DataFrame, selection_dict: dict = {"include": "yes"}
) -> pd.DataFrame:
    """Clean dataframe fields and standardise for future processing"""
    ### basic cleaning
    df.columns = df.columns.str.normalize("NFKC").str.replace(
        "μ", "u"
    )  # replace any unicode versions of 'μ' with 'u'
    df = df.map(
        lambda x: unicodedata.normalize("NFKD", str(x)).replace("\xa0", " ")
        if isinstance(x, str)
        else x
    )  # clean non-breaking spaces from string cells
    # general processing
    df.rename(
        columns=file_ops.read_yaml(config.resources_dir / "mapping.yaml")[
            "sheet_column_map"
        ],
        inplace=True,
    )  # rename columns to agree with cbsyst output
    df.columns = (
        df.columns.str.lower()
    )  # columns lower case headers for less confusing access later on
    df.columns = df.columns.str.replace(
        " ", "_"
    )  # process columns to replace whitespace with underscore
    df.columns = df.columns.str.replace(
        "[()]", "", regex=True
    )  # remove '(' and ')' from column names
    df["year"] = pd.to_datetime(
        df["year"], format="%Y"
    )  # datetime format for later plotting

    ### deal with duplicate dois: flag up duplicate dois which also have 'include' as 'yes'
    inclusion_df = df[df["include"] == "yes"]
    duplicate_dois = inclusion_df[inclusion_df.duplicated(subset="doi", keep=False)]
    if not duplicate_dois.empty and not all(pd.isna(duplicate_dois["doi"])):
        print("\nDuplicate DOIs found, treat with caution:")
        print([doi for doi in duplicate_dois.doi.unique() if doi is not np.nan])

    ### formating: fill down necessary repeated metadata values
    df[["doi", "year", "authors", "location", "species_types", "taxa"]] = (
        df[["doi", "year", "authors", "location", "species_types", "taxa"]]
        .infer_objects(copy=False)
        .ffill()
    )
    df[["coords", "cleaned_coords"]] = df.groupby("doi")[
        ["coords", "cleaned_coords"]
    ].ffill()  # fill only as far as the next DOI

    if selection_dict:  # filter for selected values
        for key, value in selection_dict.items():
            if isinstance(value, list):
                df = df[df[key].isin(value)]
            else:
                df = df[df[key] == value]

    ### deal with missing n
    if (
        df["n"].dtype == "object"
    ):  # Only perform string operations if column contains strings
        df = df[
            ~df["n"].str.contains("~", na=False)
        ]  # remove any rows in which 'n' has '~' in the string
        df = df[df.n != "M"]  # remove any rows in which 'n' is 'M'

    ### infer data types
    df.loc[:, df.columns != "year"] = df.loc[:, df.columns != "year"].apply(
        utils.safe_to_numeric
    )
    problem_cols = [
        "irr",
        "ipar",
        "sal",
    ]  # some columns have rogue strings when they should all contain numbers: in this case, convert unconvertable values to NaN
    for col in problem_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    ### remove any columns with 'unnamed' in the header: these are an artefact from messing around outside the spreadsheets necessary columns
    df = df.loc[:, ~df.columns.str.contains("^unnamed")]
    return df


def process_raw_data(
    df: pd.DataFrame,
    require_results: bool = True,
    selection_dict: dict = {"include": "yes"},
    ph_conversion: bool = True,
) -> pd.DataFrame:
    """Process raw data from the spreadsheet to prepare for analysis
    Args:
        df (pd.DataFrame): DataFrame containing raw data
        require_results (bool): Whether to require results for processing
        selection_dict (dict): Dictionary of selections to filter the DataFrame

    Returns:
        pd.DataFrame: Processed DataFrame
    """
    df = preprocess_df(df, selection_dict=selection_dict)  # general processing
    ### location processsing
    df = locations.uniquify_multilocation_study_dois(
        df
    )  # for dois with multiple locations
    df = locations.assign_coordinates(df)  # assign coordinates to locations
    locations.save_locations_information(df)  # save locations information
    df = locations.assign_ecoregions(df)  # assign ecoregions to locations

    ### taxonomy
    df = taxonomy.assign_taxonomical_info(
        df
    )  # create family, genus, species, and functional group columns from species binomials

    ### units
    df["irr"] = df.apply(
        lambda row: units.irradiance_conversion(row["ipar"], "PAR")
        if pd.notna(row["ipar"])
        else row["irr"],
        axis=1,
    )  # convert integrated irradiance to irradiance

    if ph_conversion:
        df["hplus"] = df.apply(
            lambda row: units.ph_to_hplus(row["phtot"])
            if pd.notna(row["phtot"])
            else None,
            axis=1,
        )  # convert pH to H+ concentration in μmol/kg seawater

    if require_results:  # keep only rows with all the necessary data
        df = df.dropna(subset=["n", "calcification"])

    # calculate calcification standard deviation
    df["calcification_sd"] = df.apply(
        lambda row: utils.calc_sd_from_se(row["calcification_se"], row["n"])
        if pd.notna(row["calcification_se"]) and pd.notna(row["n"])
        else row["calcification_sd"],
        axis=1,
    )

    # calculate standarised calcification rates and relevant units
    df = units.map_units(df)  # map units to standardised units
    df[["st_calcification", "st_calcification_sd", "st_calcification_unit"]] = df.apply(
        lambda x: pd.Series(
            units.rate_conversion(
                x["calcification"], x["calcification_sd"], x["st_calcification_unit"]
            )
        )
        if pd.notna(x["calcification"]) and pd.notna(x["st_calcification_unit"])
        else pd.Series(["", "", ""]),
        axis=1,
    )

    return df
