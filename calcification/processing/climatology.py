### process climatology datasets

import logging

import numpy as np
import pandas as pd
from scipy import interpolate
from tqdm.auto import tqdm

from calcification.utils import file_ops

logger = logging.getLogger(__name__)


def process_climatology_csv(fp: str, index_col: str = "doi") -> pd.DataFrame:
    """Process climatology csv files provided in BH's format."""
    try:
        df = pd.read_csv(fp).drop(columns=["data_ID", "Unnamed: 0"])
        df = df.replace(
            {"2021_2040": 2030, "2041_2060": 2050, "2081_2100": 2090}
        ).infer_objects(copy=False)
        return df.set_index(index_col) if index_col else df
    except Exception as e:
        logger.error(f"Error processing climatology CSV: {e}")
        raise


def _detect_variable_type(fp) -> str:
    """Detect variable type (ph or sst) from file name."""
    name = str(getattr(fp, "name", fp))
    if "ph" in name:
        return "ph"
    elif "sst" in name:
        return "sst"
    else:
        raise ValueError(
            "File path must contain 'ph' or 'sst' to determine variable type."
        )


def convert_climatology_csv_to_multiindex(
    fp: str, locations_yaml_fp: str
) -> pd.DataFrame:
    """Convert the climatology CSV file to a multi-index DataFrame."""
    df = process_climatology_csv(fp, index_col="doi")
    var = _detect_variable_type(fp)
    df = pd.concat(
        [
            df.iloc[:, :4],
            df.iloc[:, 4:].rename(
                columns=lambda col: col if var in col else f"{var}_{col}"
            ),
        ],
        axis=1,
    )
    locations_df = pd.DataFrame(file_ops.read_yaml(locations_yaml_fp)).T
    locations_df = locations_df[
        ["latitude", "longitude", "location"]
    ]  # select and reorder columns
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


def generate_location_specific_climatology_anomalies(
    df: pd.DataFrame, scenario_var: str = "sst"
) -> pd.DataFrame:
    """Generate location-specific climatologyanomalies for each location in the dataframe

    Args:
        df (pd.DataFrame): Dataframe with multi-index of locations and scenarios
        scenario_var (str): Variable to generate anomalies for (e.g. "sst", "ph")

    Returns:
        pd.DataFrame: Dataframe with location-specific climatology anomalies
    """
    df = df.sort_index()  # sort index to avoid PerformanceWarning about lexsort depth
    locations = df.index.unique()
    anomaly_rows = []
    metadata_rows = []
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


def interpolate_and_extrapolate_predictions(
    df: pd.DataFrame, target_year: int = 2100
) -> pd.DataFrame:
    """Interpolate and extrapolate predictions for each core_grouping, scenario, and percentile

    Args:
        df (pd.DataFrame): Dataframe with multi-index of locations and scenarios
        target_year (int): Year to extrapolate to

    Returns:
        pd.DataFrame: Dataframe with interpolated and extrapolated predictions
    """

    grouping_cols = ["core_grouping", "scenario", "percentile", "time_frame"]
    value_cols = [col for col in df.columns if col not in grouping_cols]

    # filter only mean percentile
    df = df[df["percentile"] == "mean"].copy()

    # make the full year grid (including up to 2100)
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
        ["core_grouping", "scenario", "percentile"], observed=False
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


# --- emissions data ---


def process_emissions_sheet(sheet_df: pd.DataFrame, scenario_name: str) -> pd.DataFrame:
    """Process the emissions sheet DataFrame as provided from supplementary data of https://doi.org/10.5194/gmd-13-3571-2020."""
    try:
        sheet_df = sheet_df[["Gas", "CO2"]].iloc[3:]
        sheet_df.rename(columns={"Gas": "year", "CO2": scenario_name}, inplace=True)
        sheet_df["year"] = pd.to_numeric(sheet_df["year"], errors="coerce")
        return sheet_df
    except Exception as e:
        logger.error(f"Error processing emissions sheet: {e}")
        raise


def get_scenario_emissions_from_file(
    fp: str, scenario_names: list[str], end_year: int = 2150
) -> pd.DataFrame:
    """Process the emissions file as provided from supplementary data of https://doi.org/10.5194/gmd-13-3571-2020."""
    emissions_data = pd.DataFrame()
    for i, scenario_name in enumerate(scenario_names):
        scenario_df = _get_scenario_emissions_from_sheet(
            fp, scenario_name, end_year=end_year
        )
        if i == 0:
            emissions_data = scenario_df.copy()
        else:
            emissions_data = pd.merge(
                emissions_data,
                scenario_df,
                on="year",
                how="outer",
            )
    return emissions_data


def _get_historic_emissions_from_sheet(fp: str, start_year: int = 1950) -> pd.DataFrame:
    """Read the historic emissions sheet as provided from supplementary data of https://doi.org/10.5194/gmd-13-3571-2020."""
    sheet_df = pd.read_excel(
        fp, sheet_name="T2 - History Year 1750 to 2014", skiprows=8
    )
    historic_emissions = process_emissions_sheet(sheet_df, "Historic")

    return historic_emissions.loc[historic_emissions["year"] >= start_year, :]


def _get_relevant_sheet_for_emissions_scenario(
    fp: str, scenario_name: str
) -> pd.DataFrame:
    """Read the scenario emissions sheet as provided from supplementary data of https://doi.org/10.5194/gmd-13-3571-2020."""
    # find relevant sheet name
    sheet_names = pd.ExcelFile(fp).sheet_names
    relevant_sheet_names = [
        sheet_name for sheet_name in sheet_names if scenario_name in sheet_name
    ]
    # drop items containing '-lowNTCF' (alternative scenarios)
    relevant_sheet_names = [
        sheet_name
        for sheet_name in relevant_sheet_names
        if "-lowNTCF" not in sheet_name
    ]
    if len(relevant_sheet_names) != 1:
        raise ValueError(
            f"Expected 1 sheet name for scenario {scenario_name}, but found {len(relevant_sheet_names)}"
        )
    return relevant_sheet_names[0]


def _get_scenario_emissions_from_sheet(
    fp: str, scenario_name: str, end_year: int = 2150
) -> pd.DataFrame:
    """Read the scenario emissions sheet as provided from supplementary data of https://doi.org/10.5194/gmd-13-3571-2020."""
    relevant_sheet_name = _get_relevant_sheet_for_emissions_scenario(fp, scenario_name)
    sheet_df = pd.read_excel(fp, sheet_name=relevant_sheet_name, skiprows=8)
    scenario_emissions = process_emissions_sheet(sheet_df, scenario_name)
    return scenario_emissions.loc[scenario_emissions["year"] <= end_year, :]


def combine_historic_and_scenario_emissions(
    historic_emissions: pd.DataFrame, scenario_emissions: pd.DataFrame
) -> pd.DataFrame:
    """Combine historic and scenario emissions dataframes."""
    # duplicate historic emissions for each scenario
    historic_emissions_repeated = pd.concat(
        [
            historic_emissions["year"],
            pd.concat(
                [historic_emissions["Historic"]]
                * (len(scenario_emissions.columns) - 1),  # number of scenarios
                axis=1,
            ),
        ],
        axis=1,
    )
    historic_emissions_repeated.columns = scenario_emissions.columns
    return pd.concat([historic_emissions_repeated, scenario_emissions], axis=0)


def get_emissions_data_from_file(fp: str, scenario_names: list[str]) -> pd.DataFrame:
    """Get emissions data from file."""
    historic_emissions = _get_historic_emissions_from_sheet(fp)
    scenario_emissions = get_scenario_emissions_from_file(fp, scenario_names)
    return combine_historic_and_scenario_emissions(
        historic_emissions, scenario_emissions
    )
