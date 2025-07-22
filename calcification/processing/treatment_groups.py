# general

import numpy as np
import pandas as pd
from tqdm.auto import tqdm


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
