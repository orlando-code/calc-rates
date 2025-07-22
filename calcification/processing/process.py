import logging
from typing import Optional

import pandas as pd

from calcification.analysis import analysis
from calcification.processing import carbonate_chemistry, climatology, treatment_groups
from calcification.utils import config

logging.basicConfig(level=logging.INFO)


def process_calcification_data(
    fp: str, sheet_name: str = "all_data", selection_dict: Optional[dict] = None
) -> pd.DataFrame:
    """
    Full pipeline for processing calcification data from raw Excel to effect sizes.

    Args:
        fp (str): Path to Excel file.
        sheet_name (str): Sheet name in Excel file.
        selection_dict (dict): Optional dict for row selection.

    Returns:
        pd.DataFrame: DataFrame with effect sizes and all processing applied.
    """
    if selection_dict is None:  # exclude selected rows
        selection_dict = {"include": "yes"}

    logging.info("Populating carbonate chemistry...")
    carbonate_df = carbonate_chemistry.populate_carbonate_chemistry(
        fp, sheet_name=sheet_name, selection_dict=selection_dict
    )

    logging.info("Assigning treatment groups...")
    carbonate_df = treatment_groups.assign_treatment_groups_multilevel(carbonate_df)

    logging.info("Aggregating treatments with individual samples...")
    carbonate_df = treatment_groups.aggregate_treatments_with_individual_samples(
        carbonate_df
    )

    logging.info("Calculating effect sizes...")
    carbonate_df = analysis.calculate_effect_for_df(carbonate_df.reset_index(drop=True))

    return carbonate_df


def process_climatology_data(
    df: pd.DataFrame,
    ph_clim_path: Optional[str] = None,
    sst_clim_path: Optional[str] = None,
    locations_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Merge processed data with climatology and compute global average anomalies.

    Args:
        df (pd.DataFrame): Processed calcification DataFrame.
        ph_clim_path (str): Path to pH climatology CSV.
        sst_clim_path (str): Path to SST climatology CSV.
        locations_path (str): Path to locations YAML.

    Returns:
        pd.DataFrame: DataFrame with global average anomalies by scenario and time_frame.
    """
    ph_clim_path = ph_clim_path or (
        config.climatology_data_dir / "ph_scenarios_output_table_site_locations.csv"
    )
    sst_clim_path = sst_clim_path or (
        config.climatology_data_dir / "sst_scenarios_output_table_site_locations.csv"
    )
    locations_path = locations_path or (config.resources_dir / "locations.yaml")

    logging.info("Loading climatology data...")
    ph_climatology = climatology.convert_climatology_csv_to_multiindex(
        ph_clim_path, locations_path
    )
    sst_climatology = climatology.convert_climatology_csv_to_multiindex(
        sst_clim_path, locations_path
    )

    merged_clim_df = pd.merge(sst_climatology, ph_climatology)

    merged_clim_df_mi = merged_clim_df.set_index(
        ["doi", "location", "longitude", "latitude"]
    )
    df_mi = df.set_index(["doi", "location", "longitude", "latitude"])
    clim_df = df_mi.join(merged_clim_df_mi, how="inner")

    logging.info(
        f"Unique locations in climatology: {len(merged_clim_df_mi.index.unique())}, "
        f"locations in working dataframe: {len(df.drop_duplicates('doi', keep='first'))}"
    )

    # exclude aquaria locations # TODO: make this more robust/automated
    clim_df = clim_df[
        ~clim_df.index.get_level_values("location").str.contains(
            "monaco|portugal|uk", case=False, na=False
        )
    ]

    # calculate global average anomalies for dataframe
    result = (
        clim_df.reset_index()
        .groupby(["scenario", "time_frame"])
        .agg(
            mean_sst_anomaly=("mean_sst_20y_anomaly_ensemble", "mean"),
            mean_ph_anomaly=("mean_ph_20y_anomaly_ensemble", "mean"),
        )
        .reset_index()
    )
    return result
