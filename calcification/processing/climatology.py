# general

import numpy as np
import pandas as pd
from scipy import interpolate
from tqdm.auto import tqdm

from calcification.utils import file_ops


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


def process_emissions_sheet(sheet_df: pd.DataFrame, scenario_name: str) -> pd.DataFrame:
    # Process the sheet DataFrame
    sheet_df = sheet_df[["Gas", "CO2"]].iloc[3:]  # years labelled 'Gas'
    sheet_df.rename(columns={"Gas": "year", "CO2": scenario_name}, inplace=True)
    sheet_df["year"] = pd.to_numeric(sheet_df["year"], errors="coerce")
    return sheet_df
