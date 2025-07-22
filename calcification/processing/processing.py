# general
import unicodedata

import cbsyst as cb
import cbsyst.helpers as cbh
import numpy as np
import pandas as pd
from scipy import interpolate
from tqdm.auto import tqdm

from calcification.processing import locations, taxonomy, units
from calcification.utils import config, file_ops, utils


### helper functions
def aggregate_df(df: pd.DataFrame, method: str = "mean") -> pd.DataFrame:
    """Aggregate DataFrame by specified method (mean, median, etc.)"""
    aggregation_funcs = {
        col: method if pd.api.types.is_numeric_dtype(df[col]) else lambda x: x.iloc[0]
        for col in df.columns
    }  # define aggregation functions for each column
    return df.agg(aggregation_funcs)  # aggregate DataFrame


def assign_delta_t_category(delta_t: float | int) -> str:
    """
    Assign a category based on the value of delta_t.
    """
    bins = [-float("inf"), 0.5, 1.5, 2.5, float("inf")]
    labels = ["No change", "Low", "Medium", "High"]
    return pd.cut([delta_t], bins=bins, labels=labels)[0]


def group_irradiance(
    df: pd.DataFrame, irr_col: float = "irr", atol: float = 30
) -> pd.DataFrame:
    """
    Assigns an 'irr_group' to values that are within absolute tolerance.

    Args:
        df (pd.DataFrame): Input dataframe with an 'irr' column.
        irr_col (str): Column name for irradiance values.
        atol (float): Absolute tolerance for grouping.

    Returns:
        pd.DataFrame: Dataframe with new 'irr_group' column.
    """
    df = df.copy()  # avoid overwriting
    irr_values = df[irr_col].to_numpy(dtype=float)  # to apply mask
    irr_groups = np.full(len(df), -1)  # set default group (for nan values)

    # get indices of non-NaN values and sort them by irradiance value
    valid_idx = np.where(~np.isnan(irr_values))[0]
    sorted_idx = valid_idx[np.argsort(irr_values[valid_idx])]

    # assign groups to valid values
    if len(sorted_idx) > 0:  # continue if there are valid values
        group_id = 0
        prev_irr = irr_values[sorted_idx[0]]
        irr_groups[sorted_idx[0]] = group_id

        for i in sorted_idx[1:]:
            if np.abs(irr_values[i] - prev_irr) > atol:
                group_id += 1
            irr_groups[i] = group_id
            prev_irr = irr_values[i]

    df["irr_group"] = irr_groups
    return df


def cluster_values(values: list, tolerance: float) -> list:
    """
    Cluster values based on their proximity.

    Args:
        values (array-like): Values to cluster.
        tolerance (float): Tolerance for clustering.

    Returns:
        list: List of clusters, where each cluster is a list of values.
    """
    if len(values) == 0:  # return empty list if no values
        return []

    # sort values
    sorted_values = np.sort(values)

    # initialize first cluster
    clusters = [[sorted_values[0]]]

    # cluster remaining values
    for val in sorted_values[1:]:
        # check if value is sufficiently close to the last value in the current cluster
        if np.abs(val - np.mean(clusters[-1])) < tolerance:
            # add to current (most recent) cluster
            clusters[-1].append(val)
        else:  # if not close enough
            # start new cluster
            clusters.append([val])

    return clusters


def aggregate_treatments_with_individual_samples(df: pd.DataFrame) -> pd.DataFrame:
    """For treatments with only one sample (most often those for which raw, sample-level data was extracted), aggregate the data to get means and standard deviations of the treatment groups."""
    aggregated_df = (
        df.groupby(
            [
                "doi",
                "species_types",
                "treatment_level_ph",
                "treatment_level_t",
                "calcification_unit",
            ]
        )
        .filter(lambda group: (group["n"] == 1).all())
        .groupby(
            [
                "doi",
                "species_types",
                "treatment_level_ph",
                "treatment_level_t",
                "calcification_unit",
            ]
        )
        .agg(
            ecoregion=("ecoregion", "first"),  # metadata
            lat_zone=("lat_zone", "first"),
            latitude=("latitude", "first"),
            longitude=("longitude", "first"),
            location=("location", "first"),
            taxa=("taxa", "first"),
            genus=("genus", "first"),
            species=("species", "first"),
            family=("family", "first"),
            core_grouping=("core_grouping", "first"),
            authors=("authors", "first"),
            year=("year", "first"),
            treatment_group=("treatment_group", "first"),
            treatment=("treatment", "first"),
            dic=("dic", "mean"),  # carbonate chemistry
            dic_sd=("dic", "std"),
            pco2=("pco2", "mean"),
            pco2_sd=("pco2", "std"),
            phtot=("phtot", "mean"),
            phtot_sd=("phtot", "std"),
            temp=("temp", "mean"),
            temp_sd=("temp", "std"),
            sal=("sal", "mean"),
            sal_sd=("sal", "std"),
            irr=("irr", "mean"),  # irradiance
            irr_sd=("irr", "std"),
            calcification=("calcification", "mean"),  # calcification
            calcification_sd=("calcification", "std"),
            st_calcification=("st_calcification", "mean"),
            st_calcification_sd=("st_calcification", "std"),
            st_calcification_unit=("st_calcification_unit", "first"),
            n=("n", "count"),
        )
        .reset_index()
    )
    # remove rows with n=1
    df_no_ones = df[df["n"] != 1]
    # Append the aggregated data to the DataFrame
    df_no_ones = pd.concat([df_no_ones, aggregated_df], ignore_index=True)
    return df_no_ones


def calc_sd_from_se(se: float, n: int) -> float:
    """Calculate standard deviation from standard error and sample size

    Args:
        se (float): standard error
        n (int): number of samples

    Returns:
        float: standard deviation
    """
    return se * np.sqrt(n)


### raw file wrangling
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
        lambda row: calc_sd_from_se(row["calcification_se"], row["n"])
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


### carbonate chemistry
def populate_carbonate_chemistry(
    fp: str, sheet_name: str = "all_data", selection_dict: dict = {"include": "yes"}
) -> pd.DataFrame:
    df = process_raw_data(
        pd.read_excel(fp, sheet_name=sheet_name),
        require_results=False,
        selection_dict=selection_dict,
    )
    ### load measured values
    print("Loading measured values...")
    measured_df = file_ops.get_highlighted(
        fp, sheet_name=sheet_name
    )  # keeping all cols
    # measured_df = process_raw_data(measured_df, require_results=False, selection_dict=selection_dict)
    measured_df = preprocess_df(measured_df, selection_dict=selection_dict)

    ### convert nbs values to total scale using cbsyst     # TODO: implement uncertainty propagation
    # if one of ph is provided, ensure total ph is calculated
    # Only convert pHnbs to pHtot if pHtot is NaN and pHnbs is available
    mask_missing_phtot_with_nbs = (
        measured_df["phtot"].isna()
        & measured_df["phnbs"].notna()
        & measured_df["temp"].notna()
    )
    measured_df.loc[mask_missing_phtot_with_nbs, "phtot"] = measured_df[
        mask_missing_phtot_with_nbs
    ].apply(
        lambda row: cbh.pH_scale_converter(
            pH=row["phnbs"],
            scale="NBS",
            Temp=row["temp"],
            Sal=row["sal"] if pd.notna(row["sal"]) else 35,
        ).get("pHtot", None),
        axis=1,
    )

    # Only calculate pHtot from DIC and TA if pHtot is still NaN and required parameters are available
    mask_missing_phtot_with_carb = (
        measured_df["phtot"].isna()
        & measured_df["dic"].notna()
        & measured_df["ta"].notna()
        & measured_df["temp"].notna()
    )
    if mask_missing_phtot_with_carb.any():
        measured_df.loc[mask_missing_phtot_with_carb, "phtot"] = measured_df[
            mask_missing_phtot_with_carb
        ].apply(
            lambda row: cb.Csys(
                TA=row["ta"],
                DIC=row["dic"],
                T_in=row["temp"],
                S_in=row["sal"] if pd.notna(row["sal"]) else 35,
            ).pHtot[0],
            axis=1,
        )
    # if phtot is still NaN, calculate from other parameters.
    mask_missing_phtot_with_alt_carb = (
        measured_df["phtot"].isna()
        & measured_df["pco2"].notna()
        & measured_df["ta"].notna()
        & measured_df["temp"].notna()
    )
    if mask_missing_phtot_with_alt_carb.any():
        measured_df.loc[mask_missing_phtot_with_alt_carb, "phtot"] = measured_df[
            mask_missing_phtot_with_alt_carb
        ].apply(
            lambda row: cb.Csys(
                TA=row["ta"],
                pCO2=row["pco2"],
                T_in=row["temp"],
                S_in=row["sal"] if pd.notna(row["sal"]) else 35,
            ).pHtot[0],
            axis=1,
        )

    ### calculate carbonate chemistry
    carb_metadata = file_ops.read_yaml(config.resources_dir / "mapping.yaml")
    carb_chem_cols = carb_metadata["carbonate_chemistry_cols"]
    out_values = carb_metadata["carbonate_chemistry_params"]
    carb_df = measured_df[carb_chem_cols].copy()

    # apply carbonate chemistry calculation row-wise
    tqdm.pandas(desc="Calculating carbonate chemistry")
    carb_df.loc[:, out_values] = carb_df.progress_apply(
        lambda row: pd.Series(calculate_carb_chem(row, out_values)), axis=1
    )
    return df.combine_first(carb_df)


def calculate_carb_chem(row: pd.Series, out_values: list) -> dict:
    """(Re)calculate carbonate chemistry parameters from the dataframe row and return a dictionary."""
    try:
        out_dict = cb.Csys(
            pHtot=row["phtot"],
            TA=row["ta"],
            T_in=row["temp"],
            S_in=35 if pd.isna(row["sal"]) else row["sal"],
        )
        out_dict = {
            key.lower(): value for key, value in out_dict.items()
        }  # lower the keys of the dictionary to ensure case-insensitivity
        return {
            key: (
                out_dict.get(key.lower(), None)[0]
                if isinstance(out_dict.get(key.lower()), (list, np.ndarray))
                else out_dict.get(key.lower(), None)
            )
            for key in out_values
        }
    except Exception as e:
        print(f"Error: {e}")


### assigning treatment groups
def determine_control_conditions(df: pd.DataFrame) -> dict:
    """Identify the rows corresponding to min temperature and/or max pH.

    Args:
        df (pd.DataFrame): Input dataframe with columns 'doi', 'temp', 'phtot', etc.

    Returns:
        dict: Dictionary with control conditions for each treatment group.
    """
    grouped = df.groupby("treatment_group")

    control_treatments = {}

    for group, sub_df in grouped:
        group = int(group)  # convert group to integer for semantics
        min_temp = (
            sub_df.loc[sub_df["temp"].idxmin()]["temp"]
            if not any(sub_df["phtot"].isna())
            else None
        )  # Row with minimum temperature
        max_pH = (
            sub_df.loc[sub_df["phtot"].idxmax()]["phtot"]
            if not any(sub_df["phtot"].isna())
            else None
        )  # Row with maximum pH

        control_treatments[group] = {
            "control_t_in": min_temp,
            "control_phtot": max_pH,
        }

    return control_treatments


def assign_treatment_groups(
    df: pd.DataFrame,
    control_T: float,
    control_pH: float,
    t_mapping: dict,
    ph_mapping: dict,
    irr_group: float,
) -> pd.DataFrame:
    """Assign treatment groups based on temperature and pH values.

    Args:
        df (pd.DataFrame): Input dataframe with columns 'doi', 'temp', 'phtot', etc.
        control_T (float): Control temperature value.
        control_pH (float): Control pH value.
        t_mapping (dict): Mapping of temperature values to cluster indices.
        ph_mapping (dict): Mapping of pH values to cluster indices.
        irr_group (float): Irradiance group identifier.

    Returns:
        pd.DataFrame: Dataframe with added treatment group columns.
    """
    # apply classification to each row in this group
    for idx in df.index:
        row = df.loc[idx]

        # get temperature cluster level (0 is control)
        t_level = None
        if not np.isnan(row["temp"]) and control_T is not None:
            t_cluster_idx = t_mapping.get(row["temp"])
            control_cluster_idx = t_mapping.get(control_T)
            if t_cluster_idx is not None and control_cluster_idx is not None:
                t_level = t_cluster_idx - control_cluster_idx

        # get pH cluster level (0 is control)
        ph_level = None
        if not np.isnan(row["phtot"]) and control_pH is not None:
            ph_cluster_idx = ph_mapping.get(row["phtot"])
            control_cluster_idx = ph_mapping.get(control_pH)
            if ph_cluster_idx is not None and control_cluster_idx is not None:
                ph_level = (
                    control_cluster_idx - ph_cluster_idx
                )  # reverse order since higher pH is control

        # determine clusters for cases where there is only one of t or ph
        if t_level is None and ph_level is not None:
            t_level = 0
        if ph_level is None and t_level is not None:
            ph_level = 0

        # determine if values are in control clusters   # TODO: not currently capturing rare case when studies have both T and pH varied from control with no intermediary values
        is_control_T = t_level == 0 if t_level is not None else False
        is_control_pH = ph_level == 0 if ph_level is not None else False

        # classify the treatment
        if is_control_T and is_control_pH:
            treatment = "cTcP"
        elif is_control_T:
            treatment = "cTtP"
        elif is_control_pH:
            treatment = "tTcP"
        elif not (is_control_T or is_control_pH):
            treatment = "tTtP"
        else:
            treatment = "uncertain"

        # Update the treatment info in the result dataframe
        df.loc[idx, "treatment_group"] = treatment
        df.loc[idx, "treatment_level_t"] = t_level if t_level is not None else np.nan
        df.loc[idx, "treatment_level_ph"] = ph_level if ph_level is not None else np.nan
        df.loc[idx, "irr_group"] = irr_group

    return df


def assign_treatment_groups_multilevel(
    df: pd.DataFrame, t_atol: float = 0.5, pH_atol: float = 0.08, irr_atol: float = 30
) -> pd.DataFrame:
    """
    Assign treatment groups to each row based on temperature and pH values,
    recognizing multiple levels of treatments.

    Args:
        df (pd.DataFrame): Input dataframe with columns 'doi', 'temp', 'phtot', etc.
        t_atol (float): Absolute tolerance for temperature comparison.
        pH_atol (float): Absolute tolerance for pH comparison.
        irr_atol (float): Absolute tolerance for irradiance grouping.

    Returns:
        pd.DataFrame: Original dataframe with added 'treatment_group' and 'treatment_level' columns.
    """
    result_df = df.copy()  # avoid modifying original dataframe

    # Pre-initialize columns with correct types
    result_df["treatment_group"] = pd.NA
    result_df["treatment_level_t"] = pd.NA
    result_df["treatment_level_ph"] = pd.NA
    result_df["irr_group"] = pd.NA

    # process all DOIs in one pass for irradiance grouping
    for doi, study_df in result_df.groupby(
        "doi"
    ):  # group irradiance values by study (DOI)
        # Apply irradiance grouping within each study
        grouped_df = group_irradiance(study_df, atol=irr_atol)
        # Update the result dataframe with the grouped irradiance values
        result_df.loc[grouped_df.index, "irr_group"] = grouped_df["irr_group"]

    # Create a list to store processed dataframes
    processed_dfs = []

    # Group by relevant factors to process in chunks
    groupby_cols = ["doi", "irr_group", "species_types"]
    for (study_doi, irr_group, species), group_df in tqdm(
        result_df.groupby(groupby_cols, dropna=False),
        desc="Assigning treatment groups",
        total=result_df.groupby(groupby_cols, dropna=False).ngroups,
    ):
        if len(group_df) <= 1:  # skip if too few samples
            continue

        # find control values (min T, max pH)
        control_T = (
            group_df["temp"].min() if not group_df["temp"].isna().all() else None
        )
        control_pH = (
            group_df["phtot"].max() if not group_df["phtot"].isna().all() else None
        )

        # cluster temperature values
        t_values = group_df["temp"].dropna().unique()
        t_clusters = cluster_values(t_values, t_atol)

        # cluster pH values
        ph_values = group_df["phtot"].dropna().unique()
        ph_clusters = cluster_values(ph_values, pH_atol)

        # map each value to its cluster index
        t_mapping = {
            val: cluster_idx
            for cluster_idx, cluster in enumerate(t_clusters)
            for val in cluster
        }
        ph_mapping = {
            val: cluster_idx
            for cluster_idx, cluster in enumerate(ph_clusters)
            for val in cluster
        }

        # Process this group
        treatments_df = assign_treatment_groups(
            group_df, control_T, control_pH, t_mapping, ph_mapping, irr_group
        )
        processed_dfs.append(treatments_df)

    if processed_dfs:
        combined_df = pd.concat(processed_dfs)  # concatenating on index
        for col in [
            "treatment_group",
            "treatment_level_t",
            "treatment_level_ph",
        ]:  # update original DataFrame using loc indexing (more efficient)
            result_df.loc[combined_df.index, col] = combined_df[col]
    # vectorized updating
    conditions = [
        (result_df["treatment_group"] == "cTcP"),
        (
            result_df["treatment_group"].str.contains("tT", na=False)
            & result_df["treatment_group"].str.contains("tP", na=False)
        ),
        (result_df["treatment_group"].str.contains("tT", na=False)),
        (result_df["treatment_group"].str.contains("tP", na=False)),
    ]
    choices = ["control", "temp_phtot", "temp", "phtot"]
    result_df["treatment"] = np.select(conditions, choices, default="unknown")
    # Replace 'unknown' with np.nan after the select operation
    result_df.loc[result_df["treatment"] == "unknown", "treatment"] = np.nan

    return result_df


### climatology
def process_climatology_csv(fp: str, index_col: str = "doi") -> pd.DataFrame:
    df = pd.read_csv(fp).drop(columns=["data_ID", "Unnamed: 0"])
    # rename columns to be less wordy
    df = (
        (df.copy())
        .replace({"2021_2040": 2030, "2041_2060": 2050, "2081_2100": 2090})
        .infer_objects(copy=False)
    )

    return df.set_index(index_col) if index_col else df


def convert_climatology_csv_to_multiindex(
    fp: str, locations_yaml_fp: str
) -> pd.DataFrame:
    """
    Convert the climatology CSV file to a multi-index DataFrame.
    """
    df = process_climatology_csv(fp, index_col="doi")  # load the CSV file

    var = "ph" if "ph" in str(fp.name) else "sst" if "sst" in str(fp.name) else None
    if not var:
        raise ValueError(
            "File path must contain 'ph' or 'sst' to determine variable type."
        )
    df = pd.concat(
        [
            df.iloc[:, :4],
            df.iloc[:, 4:].rename(
                columns=lambda col: col if var in col else f"{var}_{col}"
            ),
        ],
        axis=1,
    )

    # load locations yaml as dataframe
    locations_df = pd.DataFrame(file_ops.read_yaml(locations_yaml_fp)).T
    # reorder columns to be latitude, longitude, location
    locations_df = locations_df[["latitude", "longitude", "location"]]

    # merge locations with sst_df
    df = df.merge(
        locations_df,
        left_index=True,
        right_index=True,
        how="left",
        suffixes=("", "_right"),
    )
    df = df.loc[:, ~df.columns.str.endswith("_right")]
    df.reset_index(inplace=True, names="doi")

    return df


def generate_location_specific_anomalies(df: pd.DataFrame, scenario_var: str = "sst"):
    df = (
        df.sort_index()
    )  # Sort the index to avoid PerformanceWarning about lexsort depth
    locations = df.index.unique()
    anomaly_rows = []  # to hold newmods inputs
    metadata_rows = []  # to track what each row corresponds to

    for location in tqdm(
        locations, desc=f"Generating batched anomalies for {scenario_var}"
    ):
        location_df = df.loc[location]
        scenarios = location_df["scenario"].unique()

        for scenario in scenarios:
            scenario_df = location_df[location_df["scenario"] == scenario]
            time_frames = [1995] + list(scenario_df.time_frame.unique())

            for time_frame in time_frames:
                if time_frame == 1995:
                    base = scenario_df[
                        f"mean_historical_{scenario_var}_30y_ensemble"
                    ].mean()
                    mean_scenario = base - base
                    p10_scenario = (
                        scenario_df[
                            f"percentile_10_historical_{scenario_var}_30y_ensemble"
                        ].mean()
                        - base
                    )
                    p90_scenario = (
                        scenario_df[
                            f"percentile_90_historical_{scenario_var}_30y_ensemble"
                        ].mean()
                        - base
                    )
                else:
                    time_scenario_df = scenario_df[
                        scenario_df["time_frame"] == time_frame
                    ]
                    mean_scenario = time_scenario_df[
                        f"mean_{scenario_var}_20y_anomaly_ensemble"
                    ].mean()
                    p10_scenario = time_scenario_df[
                        f"{scenario_var}_percentile_10_anomaly_ensemble"
                    ].mean()
                    p90_scenario = time_scenario_df[
                        f"{scenario_var}_percentile_90_anomaly_ensemble"
                    ].mean()
                    # Generate predictions for mean, p10, and p90 scenarios
                for percentile, anomaly in [
                    ("mean", mean_scenario),
                    ("p10", p10_scenario),
                    ("p90", p90_scenario),
                ]:
                    anomaly_rows.append([anomaly])
                    metadata_rows.append(
                        {
                            "doi": location[0],
                            "location": location[1],
                            "longitude": location[2],
                            "latitude": location[3],
                            "scenario_var": scenario_var,
                            "scenario": scenario,
                            "time_frame": time_frame,
                            # 'anomaly_value': anomaly,
                            "percentile": percentile,
                        }
                    )
    return pd.concat(
        [
            pd.DataFrame(metadata_rows),
            pd.DataFrame(anomaly_rows, columns=["anomaly_value"]),
        ],
        axis=1,
    )


def interpolate_and_extrapolate_predictions(df, target_year=2100):
    grouping_cols = ["core_grouping", "scenario", "percentile", "time_frame"]
    value_cols = [col for col in df.columns if col not in grouping_cols]

    # Filter only mean percentile
    df = df[df["percentile"] == "mean"].copy()

    # Make the full year grid (including up to 2100)
    all_years = np.arange(df["time_frame"].min(), target_year + 1)
    unique_groups = df[["core_grouping", "scenario", "percentile"]].drop_duplicates()
    full_grid = unique_groups.merge(
        pd.DataFrame({"time_frame": all_years}), how="cross"
    )

    # Merge full grid with existing predictions
    df_full = pd.merge(
        full_grid,
        df,
        on=["core_grouping", "scenario", "percentile", "time_frame"],
        how="left",
    )

    # Now interpolate/extrapolate for each group
    for (core_grouping, scenario, percentile), group_df in df_full.groupby(
        ["core_grouping", "scenario", "percentile"]
    ):
        mask = (
            (df_full["core_grouping"] == core_grouping)
            & (df_full["scenario"] == scenario)
            & (df_full["percentile"] == percentile)
        )

        available_years = group_df.dropna(subset=value_cols)["time_frame"].values

        if len(available_years) < 2:
            continue  # Not enough points to interpolate

        for value_col in value_cols:
            available_vals = group_df.dropna(subset=[value_col])[value_col].values

            if len(available_vals) < 2:
                continue  # Not enough data

            # Fit spline
            spline = interpolate.make_interp_spline(
                available_years, available_vals, k=min(2, len(available_vals) - 1)
            )

            # Predict for all years
            df_full.loc[mask, value_col] = spline(all_years)

    return df_full


### DEPRECATED
# def assign_treatment_groups_multilevel(df: pd.DataFrame, t_atol: float=0.5, pH_atol: float=0.05, irr_atol: float=30) -> pd.DataFrame:
#     """
#     Assign treatment groups to each row based on temperature and pH values,
#     recognizing multiple levels of treatments.

#     Args:
#         df (pd.DataFrame): Input dataframe with columns 'doi', 'temp', 'phtot', etc.
#         t_atol (float): Absolute tolerance for temperature comparison.
#         pH_atol (float): Absolute tolerance for pH comparison.
#         irr_rtol (float): Relative tolerance for irradiance grouping.

#     Returns:
#         pd.DataFrame: Original dataframe with added 'treatment_group' and 'treatment_level' columns.
#     """
#     result_df = df.copy()   # avoid modifying original dataframe

#     # initialize treatment columns
#     result_df['treatment_group'] = pd.Series(dtype='object')
#     result_df['treatment_level_t'] = pd.Series(dtype='object')
#     result_df['treatment_level_ph'] = pd.Series(dtype='object')
#     result_df['irr_group'] = pd.Series(dtype='object')

#     for study_doi, study_df in tqdm(df.groupby('doi'), desc="Assigning treatment groups", total=len(df['doi'].unique())):
#         study_with_irr_groups = group_irradiance(study_df, atol=irr_atol)   # group irradiance treatments

#         # process each (irradiance group, species) combination separately
#         for (irr_group, species), group_df in study_with_irr_groups.groupby(['irr_group', 'species_types']):
#             if len(group_df) <= 1:  # skip if too few samples
#                 continue

#             # find control values (min T, max pH)
#             control_T = group_df['temp'].min() if not group_df['temp'].isna().all() else None
#             control_pH = group_df['phtot'].max() if not group_df['phtot'].isna().all() else None

#             # cluster temperature values
#             t_values = group_df['temp'].dropna().unique()
#             t_clusters = cluster_values(t_values, t_atol)

#             # cluster pH values
#             ph_values = group_df['phtot'].dropna().unique()
#             ph_clusters = cluster_values(ph_values, pH_atol)

#             # map each value to its cluster index
#             t_mapping = {val: cluster_idx for cluster_idx, cluster in enumerate(t_clusters) for val in cluster}
#             ph_mapping = {val: cluster_idx for cluster_idx, cluster in enumerate(ph_clusters) for val in cluster}

#             treatments_df = assign_treatment_groups(group_df, control_T, control_pH, t_mapping, ph_mapping, irr_group)

#             # fill in result_df values with results from treatments_df
#             result_df = result_df.combine_first(treatments_df)

#     result_df['treatment'] = result_df['treatment_group'].apply(
#         lambda x: 'temp_phtot' if isinstance(x, str) and 'tT' in x and 'tP' in x else
#                  'temp' if isinstance(x, str) and 'tT' in x else
#                  'phtot' if isinstance(x, str) and 'tP' in x else
#                  'control' if isinstance(x, str) and x == 'cTcP' else np.nan
#     )
#     return result_df
