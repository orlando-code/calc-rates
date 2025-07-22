### overall processing pipeline

import pandas as pd

from calcification.analysis import analysis
from calcification.processing import carbonate_chemistry, climatology, treatment_groups
from calcification.utils import config


def process_calcification_data(
    fp: str, sheet_name: str = "all_data", selection_dict: dict = {"include": "yes"}
) -> pd.DataFrame:
    carbonate_df = carbonate_chemistry.populate_carbonate_chemistry(
        fp, sheet_name=sheet_name, selection_dict=selection_dict
    )
    carbonate_df_with_treatment_groups = (
        treatment_groups.assign_treatment_groups_multilevel(carbonate_df)
    )
    carbonate_df_with_multisample_treatment_groups = (
        treatment_groups.aggregate_treatments_with_individual_samples(
            carbonate_df_with_treatment_groups
        )
    )
    carbonate_df_with_multisample_treatment_groups_and_effect_sizes = (
        analysis.calculate_effect_for_df(
            carbonate_df_with_multisample_treatment_groups.reset_index(drop=True)
        )
    )
    return carbonate_df_with_multisample_treatment_groups_and_effect_sizes


def process_climatology_data(df: pd.DataFrame) -> pd.DataFrame:
    ### load relevant climatology
    ph_climatology = climatology.convert_climatology_csv_to_multiindex(
        config.climatology_data_dir / "ph_scenarios_output_table_site_locations.csv",
        config.resources_dir / "locations.yaml",
    )

    sst_climatology = climatology.convert_climatology_csv_to_multiindex(
        config.climatology_data_dir / "sst_scenarios_output_table_site_locations.csv",
        config.resources_dir / "locations.yaml",
    )

    merged_clim_df = pd.merge(
        sst_climatology,
        ph_climatology,
    )  # concatenated sst and ph climatology dataframes

    # prepare dataframes and merge
    merged_clim_df_mi = merged_clim_df.set_index(
        ["doi", "location", "longitude", "latitude"]
    )
    df_mi = df.set_index(["doi", "location", "longitude", "latitude"])
    clim_df = df_mi.join(merged_clim_df_mi, how="inner")

    print(
        f"There are {len(merged_clim_df_mi.index.unique())} unique locations in the climatology data compared with {len(df.drop_duplicates('doi', keep='first'))} location counts in the working dataframe."
    )

    # exclude locations mentioning non-tropical ranges, so retrieved climatology data is not relevant
    clim_df = clim_df[
        ~clim_df.index.get_level_values("location").str.contains(
            "monaco|portugal|uk", case=False, na=False
        )
    ]

    ### calculate the global average anomalies for each scenario and time_frame
    return (
        clim_df.reset_index()
        .groupby(["scenario", "time_frame"])
        .agg(
            mean_sst_anomaly=("mean_sst_20y_anomaly_ensemble", "mean"),
            mean_ph_anomaly=("mean_ph_20y_anomaly_ensemble", "mean"),
        )
        .reset_index()
    )
