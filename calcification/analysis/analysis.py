# general
import os
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

# R
import rpy2.robjects as ro
import rpy2.robjects.packages as rpackages

# stats
import statsmodels.api as sm
from rpy2.robjects import pandas2ri
from tqdm.auto import tqdm

from calcification.analysis import analysis_utils

# custom
from calcification.processing import carbonate_processing, groups_processing
from calcification.utils import config, file_ops

metafor = rpackages.importr("metafor")
base = rpackages.importr("base")


### core analysis calculations
def calc_relative_rate(
    mu1: float,
    mu2: float,
    sd1: float = None,
    sd2: float = None,
    n1: int = None,
    n2: int = None,
    epsilon: float = 1e-6,
) -> tuple[float, float]:
    """
    Calculate percent change between two means with error propagation.

    Examples:
    - 10 to 20: +100%
    - 10 to 0: -100%
    - 10 to -10: -200%

    Args:
        mu1, mu2 (float):   Mean values to compare (mu1=reference/baseline, mu2=new value)
        se1, se2 (float, optional):   Standard errors of mu1 and mu2
        epsilon (float, optional):  Small value to stabilize calculations when means are close to zero

    Returns:
        pc (float): Percent change
        se_pc (float or None):  Standard error of the percent change
    """
    se1, se2 = sd1 / np.sqrt(n1), sd2 / np.sqrt(n2)
    if mu1 == 0 and mu2 == 0:  # special case: both means are exactly zero
        pc = 0  # no change between means

        if (
            se1 is not None and se2 is not None
        ):  # if SEs are provided, calculate uncertainty
            # When both means are zero, consider the ratio of SEs to estimate uncertainty
            # This represents how much percentage change we would expect if the values
            # fluctuated by ±1 SE from zero
            if (
                se1 > 0
            ):  # scale based on the potential percentage fluctuations around zero
                se_pc = 100 * se2 / se1
            else:  # if se1 is zero but se2 is not, technically infinite uncertainty
                se_pc = float("inf") if se2 > 0 else 0
            return pc, se_pc
        return pc

    if mu1 == 0:  # special case: baseline is zero
        pc = np.sign(mu2) * 100  # signed 100% change

        if (
            se1 is not None and se2 is not None
        ):  # if SE provided, calculate the uncertainty
            if (
                abs(mu2) > epsilon
            ):  # use the ratio of SEs to treatment for zero treatment
                # N.B. error in baseline causes very large fluctuations in percent change
                se_pc = 100 * se1 / abs(mu2)
                se_pc = np.sqrt(se_pc**2 + (100 * se2 / abs(mu2)) ** 2)
            else:  # if both are essentially zero, high uncertainty
                se_pc = float("inf") if (se1 > 0 or se2 > 0) else 0
            return pc, se_pc
        return pc

    # standard percent change calculation
    pc = ((mu2 - mu1) / abs(mu1)) * 100

    # return only PC if no standard errors provided
    if se1 is None or se2 is None:
        return pc

    # error propagation via partial derivatives
    dpc_dmu1 = (-mu2 / (mu1**2)) * 100
    dpc_dmu2 = (1 / abs(mu1)) * 100
    var_pc = (dpc_dmu1**2 * se1**2) + (dpc_dmu2**2 * se2**2)

    return pc, var_pc


def calc_absolute_rate(
    mu1: float,
    mu2: float,
    sd1: float = None,
    sd2: float = None,
    n1: int = None,
    n2: int = None,
) -> tuple[float, float]:
    """Calculate the simple difference between two means with error propagation.

    Args:
        mu1, mu2 (float):   Mean values to compare (mu1=reference/baseline, mu2=new value)
        sd1, sd2 (float, optional):   Standard deviations of mu1 and mu2
        n1, n2 (int): number of samples in group 1 (control) and group 2 (treatment)

    Returns:
        tuple: absolute difference between means, standard error of the difference
    """
    abs_diff = mu2 - mu1

    # if standard deviations are provided, calculate the uncertainty
    if sd1 is not None and sd2 is not None and n1 is not None and n2 is not None:
        se1 = sd1 / np.sqrt(n1)
        se2 = sd2 / np.sqrt(n2)

        # Error propagation - calculate partial derivatives
        d_abs_diff_dmu1 = -1
        d_abs_diff_dmu2 = 1

        # Calculate standard error using error propagation
        var_abs_diff = (d_abs_diff_dmu1**2 * se1**2) + (d_abs_diff_dmu2**2 * se2**2)

        return abs_diff, var_abs_diff

    return abs_diff, None


def calc_bias_correction(n1: int, n2: int) -> float:
    """Calculate bias correction for Cohen's d metric: https://www.campbellcollaboration.org/calculator/equations

    Args:
        n1, n2 (int): number of samples in group 1 (control) and group 2 (treatment)

    Returns:
        float: bias correction factor
    """
    return 1 - 3 / (4 * (n1 + n2 - 2) - 1)


def calc_cohens_d(
    mu1: float, mu2: float, sd1: float, sd2: float, n1: int, n2: int
) -> tuple[float, float]:
    """Calculate Cohen's d metric: https://www.itl.nist.gov/div898/software/dataplot/refman2/auxillar/hedgeg.htm

    Args:
        mu1, mu2 (float):   Mean values to compare (mu1=reference/baseline, mu2=new value)
        sd1, sd2 (float, optional):   Standard deviations of mu1 and mu2
        n1, n2 (int): number of samples in group 1 and group 2

    Returns:
        tuple[float, float]: Cohen's d and its variance
    """
    sd_pooled = calc_pooled_sd(n1, n2, sd1, sd2)
    d = (mu2 - mu1) / sd_pooled if sd_pooled != 0 else 0
    d_var = (n1 + n2) / (n1 * n2) + d**2 / (2 * (n1 + n2))
    return d, d_var


def calc_pooled_sd(n1: int, n2: int, sd1: float, sd2: float) -> float:
    """Calculate pooled standard deviation for two groups.
    N.B. BH (2021) uses simple average

    Args:
        n1, n2 (int): number of samples in group 1 (control) and group 2 (treatment)
        sd1, sd2 (float, optional):   Standard deviations of mu1 and mu2

    Returns:
        float: pooled standard deviation
    """
    return np.sqrt(((n1 - 1) * sd1**2 + (n2 - 1) * sd2**2) / (n1 + n2 - 2))


def calc_hedges_g(
    mu1: float, mu2: float, sd1: float, sd2: float, n1: int, n2: int
) -> tuple[float, float]:
    """Calculate Hedges G metric: https://www.campbellcollaboration.org/calculator/equations

    Args:
        mu1, mu2 (float):   Mean values to compare (mu1=reference/baseline, mu2=new value)
        sd1, sd2 (float, optional):   Standard deviations of mu1 and mu2
        n1, n2 (int): number of samples in group 1 and group 2

    Returns:
        float: Hedges G metric
    """
    d, d_var = calc_cohens_d(mu1, mu2, sd1, sd2, n1, n2)
    bias_correction = calc_bias_correction(n1, n2)

    hg = d * bias_correction
    hg_var = d_var * bias_correction**2
    return hg, hg_var


### meta-analysis functions
def calc_cooks_distance(data: pd.Series) -> pd.Series:
    """
    Calculate Cook's distance for a given data series.
    """
    # if data is not numeric
    if not pd.api.types.is_numeric_dtype(data):
        # convert data to numeric
        data = pd.to_numeric(data, errors="coerce")

    # fit OLS model
    X = sm.add_constant(np.asarray(data))
    try:
        model = sm.OLS(data, X).fit()
    except ValueError:
        # convert data to numeric if it is not already
        model = sm.OLS(data, X).fit()
    # calculate Cook's distance
    influence = model.get_influence()
    cooks_d = influence.cooks_distance[0]

    return cooks_d


def calc_cooks_threshold(data: pd.Series, nparams: int) -> float:
    """
    Calculate the Cook's distance threshold for outlier detection via 2√((k+1)/(n - k - 1)): a reproducible numerical replacement of eyeballing for outliers in the distance-study graph.
    """
    n = len(data)
    threshold = 2 * np.sqrt((nparams + 1) / (n - nparams - 1))
    return threshold


def remove_cooks_outliers(
    df: pd.DataFrame, effect_type: str = "hedges_g", nparams: int = 3
) -> pd.DataFrame:
    """
    Remove outliers from a DataFrame based on Cook's distance.
    """
    data = df.copy()
    # calculate cooks distance
    cooks_threshold = calc_cooks_threshold(data[effect_type], nparams=nparams)
    # calculate cooks distance
    data["cooks_d"] = calc_cooks_distance(data[effect_type])

    # remove outliers
    data_no_outliers = data[data["cooks_d"] < cooks_threshold]
    outliers = data[data["cooks_d"] >= cooks_threshold]
    print(
        f"\nRemoved {len(outliers)} outlier(s) (from {len(data)} samples) based on Cook's distance threshold of {cooks_threshold:.2f}"
    )
    return data_no_outliers, outliers


### calculating treatment effects
def calculate_effect_for_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Hedges' g for a DataFrame of experimental results.

    Args:
        df: DataFrame containing experimental data

    Returns:
        pandas.DataFrame: DataFrame with calculated effect sizes
    """
    # copy to avoid modifying original
    result_df = df.copy()

    effect_cols = [
        "delta_t",
        "delta_ph",
        "cohens_d",
        "cohens_d_var",
        "hedges_g",
        "hedges_g_var",
        "relative_calcification",
        "relative_calcification_var",
        "absolute_calcification",
        "absolute_calcification_var",
        "st_relative_calcification",
        "st_relative_calcification_var",
        "st_absolute_calcification",
        "st_absolute_calcification_var",
    ]
    for col in effect_cols:
        result_df[col] = np.nan

    # group by relevant factors and apply processing
    grouped_data = []
    doi_bar = tqdm(result_df.doi.unique())
    for doi in doi_bar:
        doi_bar.set_description(f"Processing {doi}")
        study_df = result_df[result_df["doi"] == doi]
        for irr_group, irr_df in study_df.groupby("irr_group"):
            for species, species_df in irr_df.groupby("species_types"):
                df = process_group_multivar(species_df)
                if isinstance(df, pd.Series):
                    df = pd.DataFrame([df].T)
                if df is not None:
                    grouped_data.extend(df)

    if isinstance(grouped_data, list):
        valid_dfs = [
            df
            for df in grouped_data
            if df is not None and not df.empty and not df.isna().all().all()
        ]
        if valid_dfs:
            df = (
                pd.concat(valid_dfs).sort_index().copy()
            )  # sort index to avoid lexsort depth warning and create a copy to avoid fragmentation
        else:
            # Return empty DataFrame with same columns and dtypes as expected output
            df = pd.DataFrame(
                columns=df.columns
                if len(grouped_data) > 0 and grouped_data[0] is not None
                else None
            )
    df = df.sort_values(by="doi").copy().reset_index()
    df["ID"] = df.index

    df.loc[df.phtot.isna(), "delta_ph"] = (
        0  # assign delta_ph = 0 where phtot is NaN (assumes this variable was controlled throughout the experiment)
    )
    df.loc[df.temp.isna(), "delta_t"] = (
        0  # similarly, assign delta_t = 0 where temp is NaN
    )

    # replace any 0 values in "*_var" columns with mean of that column (there's no such thing as no error)
    for col in df.columns:
        if col.endswith("_var"):
            mean_value = df[col].mean()
            df[col] = df[col].replace(0, mean_value).infer_objects(copy=False)

    return df


def calculate_control_series(control_df: pd.DataFrame) -> pd.Series:
    """Calculates the representative control series, averaging numeric columns."""
    if control_df.empty:
        return pd.Series(dtype=object)  # return empty series if no control data
    if len(control_df) > 1:
        numeric_cols = control_df.select_dtypes(include="number").columns
        # create a Series with first values for non-numeric, means for numeric
        control_series = control_df.iloc[0].copy()
        for col in numeric_cols:
            control_series[col] = control_df[col].mean(
                skipna=True
            )  # Use mean for numeric
    else:
        control_series = control_df.iloc[0].copy()
    return control_series


def process_group_multivar(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process a group of species data to calculate effect size.

    Args:
        df (pd.DataFrame): DataFrame containing data for a specific species

    Returns:
        pd.DataFrame: DataFrame with effect size calculations
    """

    def process_group(group, control_level_col):
        control_level = min(group[control_level_col])
        control_df = group[group[control_level_col] == control_level]
        treatment_df = group[group[control_level_col] > control_level]

        if treatment_df.empty:  # skip if there's no treatment data
            return

        control_series = calculate_control_series(control_df)

        # calculate effect size for each row in treatment_df and create a list of results
        effect_rows = []
        for _, row in treatment_df.iterrows():
            effect_row = calc_treatment_effect_for_row(row, control_series)
            effect_rows.append(effect_row)

        # concatenate all rows to create the effect_size DataFrame
        if effect_rows:
            effect_size = pd.concat(effect_rows, axis=1).T.copy()

            # update treatment label
            if control_level_col == "treatment_level_t":
                effect_size["treatment_level_ph"] = group.name
                if group.name >= 1:
                    effect_size["treatment"] = "temp_mv"
            elif control_level_col == "treatment_level_ph":
                effect_size["treatment_level_t"] = group.name
                if group.name >= 1:
                    effect_size["treatment"] = "phtot_mv"

            return effect_size
        return None

    # process each group and append results
    results_ph = df.groupby("treatment_level_ph").apply(
        process_group, control_level_col="treatment_level_t"
    )
    results_t = df.groupby("treatment_level_t").apply(
        process_group, control_level_col="treatment_level_ph"
    )

    # TODO: this doesn't add effects for where BOTH treatment levels change at once i.e. multivariate, the part which isn't in-level comparison
    def process_orthogonal_group(df):
        # identify absolute control
        control_df = df[df["treatment"] == "control"]
        control_series = calculate_control_series(control_df)

        # for each row where treatment_level_t == treatment_level_ph, calculate effect size
        treatment_df = df[
            (df["treatment_level_t"] == df["treatment_level_ph"])
            & (df["treatment"] != "control")
        ]
        if treatment_df.empty:
            return None
        # Calculate effect size for each row in treatment_df
        effect_rows = []

        for _, row in treatment_df.iterrows():
            effect_row = calc_treatment_effect_for_row(row, control_series)
            effect_rows.append(effect_row)
        # Concatenate all rows to create the effect_size DataFrame
        if effect_rows:
            effect_size = pd.concat(effect_rows, axis=1).T.copy()
            # update treatment label
            effect_size["treatment"] = "phtot_temp_mv"
            return effect_size

    # process orthogonal group
    results_orthogonal = process_orthogonal_group(df)

    results = []
    if not results_ph.empty:
        results.append(results_ph.reset_index(drop=True))
    if not results_t.empty:
        results.append(results_t.reset_index(drop=True))
    if results_orthogonal is not None:
        results.append(results_orthogonal)

    return results


def aggregate_by_treatment_group(df: pd.DataFrame) -> pd.Series:
    """
    Aggregate a DataFrame by treatment group. Useful for when samples are individual datapoints, or multiple slightly-different controls are present.

    Args:
        df: DataFrame containing data for a specific treatment group

    Returns:
        pandas.Series: Series containing aggregated data
    """
    aggregation = df.agg({"calcification": ["mean", "std"], "n": "count"})
    control_row = df.iloc[0].copy()
    control_row["calcification"] = aggregation["calcification"]["mean"]
    control_row["calcification_sd"] = aggregation["calcification"]["std"]
    control_row["n"] = aggregation["n"]["count"]
    return control_row


def calc_treatment_effect_for_row(
    treatment_row: pd.Series, control_data: pd.Series
) -> pd.Series:
    """
    Calculate the effect size (Hedges' g or relative calcification) and append additional columns for a treatment row.

    Args:
        treatment_row: Row containing treatment data
        control_data: Dictionary containing control group data

    Returns:
        pandas.Series: Row with calculated effect sizes and additional metadata
    """
    mu_t, sd_t, n_t = (
        treatment_row["calcification"],
        treatment_row["calcification_sd"],
        treatment_row["n"],
    )
    mu_c, sd_c, n_c = (
        control_data["calcification"],
        control_data["calcification_sd"],
        control_data["n"],
    )
    # standardised values
    s_mu_t, s_sd_t, _ = (
        treatment_row["st_calcification"],
        treatment_row["st_calcification_sd"],
        treatment_row["n"],
    )
    s_mu_c, s_sd_c, _ = (
        control_data["st_calcification"],
        control_data["st_calcification_sd"],
        control_data["n"],
    )
    t_in_c, ph_c = control_data["temp"], control_data["phtot"]

    if np.isnan(mu_t) or np.isnan(mu_c) or np.isnan(sd_t) or np.isnan(sd_c):
        print(
            f"Missing data for effect size calculation. mu_t: {mu_t:.3f}, mu_c: {mu_c:.3f}, sd_t: {sd_t:.3f}, sd_c: {sd_c:.3f}, n_t: {n_t:.3f}, n_c: {n_c:.3f} at \n[index {treatment_row.name} DOI {treatment_row['doi']}]"
        )
        print(treatment_row.doi)

    row_copy = treatment_row.copy()  # create a copy to avoid SettingWithCopyWarning

    d_effect, d_var = calc_cohens_d(mu_c, mu_t, sd_c, sd_t, n_c, n_t)  # Cohen's d
    hg_effect, hg_var = calc_hedges_g(mu_c, mu_t, sd_c, sd_t, n_c, n_t)  # Hedges' g

    # handle relative calcification (use raw value if already stated relative to baseline)
    rc_effect, rc_var = (
        (mu_t, sd_t)
        if isinstance(treatment_row["calcification_unit"], str)
        and "delta" in treatment_row["calcification_unit"]
        else calc_relative_rate(mu_c, mu_t, sd_c, sd_t, n_c, n_t)
    )

    abs_effect, abs_var = calc_absolute_rate(
        mu_c, mu_t, sd_c, sd_t, n_c, n_t
    )  # absolute differences

    st_d_effect, st_d_var = calc_cohens_d(
        s_mu_c, s_mu_t, s_sd_c, s_sd_t, n_c, n_t
    )  # standardised cohen's d
    st_hg_effect, st_hg_var = calc_hedges_g(
        s_mu_c, s_mu_t, s_sd_c, s_sd_t, n_c, n_t
    )  # standardised hedges' g

    # absolute differences between standardised calcification
    st_abs_effect, st_abs_var = calc_absolute_rate(
        s_mu_c, s_mu_t, s_sd_c, s_sd_t, n_c, n_t
    )
    # relative differences between standardised calcification
    st_rc_effect, st_rc_var = (
        (s_mu_t, s_sd_t)
        if isinstance(treatment_row["st_calcification_unit"], str)
        and "delta" in treatment_row["st_calcification_unit"]
        else calc_relative_rate(s_mu_c, s_mu_t, s_sd_c, s_sd_t, n_c, n_t)
    )

    # assign effect sizes
    row_copy.update(
        {
            "cohens_d": d_effect,
            "cohens_d_var": d_var,
            "hedges_g": hg_effect,
            "hedges_g_var": hg_var,
            "relative_calcification": rc_effect,
            "relative_calcification_var": rc_var,
            "absolute_calcification": abs_effect,
            "absolute_calcification_var": abs_var,
            "st_relative_calcification": st_rc_effect,
            "st_relative_calcification_var": st_rc_var,
            "st_absolute_calcification": st_abs_effect,
            "st_absolute_calcification_var": st_abs_var,
        }
    )

    # calculate metadata
    row_copy["control_temp"] = control_data["temp"]
    row_copy["treatment_temp"] = treatment_row["temp"]
    row_copy["delta_t"] = row_copy["temp"] - t_in_c
    row_copy["control_phtot"] = control_data["phtot"]
    row_copy["treatment_phtot"] = treatment_row["phtot"]
    row_copy["delta_ph"] = row_copy["phtot"] - ph_c
    row_copy["treatment_val"] = (
        row_copy["temp"] if row_copy["treatment"] == "temp" else row_copy["phtot"]
    )
    row_copy["control_calcification"] = mu_c
    row_copy["control_calcification_sd"] = sd_c
    row_copy["treatment_calcification"] = mu_t
    row_copy["treatment_calcification_sd"] = sd_t
    row_copy["st_control_calcification"] = s_mu_c
    row_copy["st_control_calcification_sd"] = s_sd_c
    row_copy["st_treatment_calcification"] = s_mu_t
    row_copy["st_treatment_calcification_sd"] = s_sd_t
    row_copy["treatment_n"] = n_t
    row_copy["control_n"] = n_c

    return row_copy


def calculate_effect_sizes_end_to_end(
    raw_data_fp: str,
    data_sheet_name: str,
    climatology_data_fp: str = None,
    selection_dict: dict = {"include": "yes"},
) -> pd.DataFrame:
    """
    Calculate effect sizes from raw data and align with climatology data.

    Args:
        raw_data_fp (str or Path): Path to raw data file
        data_sheet_name (str): Name of the sheet containing data
        climatology_data_fp (str or Path): Path to climatology data file
        selection_dict (dict): Dictionary of selection criteria

    Returns:
        pd.DataFrame: DataFrame with calculated effect sizes
    """
    # load and process carbonate chemistry data
    carbonate_df = carbonate_processing.populate_carbonate_chemistry(
        raw_data_fp, data_sheet_name, selection_dict=selection_dict
    )

    # prepare for alignment with climatology by uniquifying DOIs
    print(
        f"\nShape of dataframe with all rows marked for inclusion: {carbonate_df.shape}"
    )

    # save selected columns of carbonate dataframe to file for reference
    carbonate_save_fields = file_ops.read_yaml(config.resources_dir / "mapping.yaml")[
        "carbonate_save_columns"
    ]
    carbonate_df[carbonate_save_fields].to_csv(
        config.tmp_data_dir / "carbonate_chemistry.csv", index=False
    )

    # assign treatment groups
    carbonate_df_tgs = groups_processing.assign_treatment_groups_multilevel(
        carbonate_df
    )

    carbonate_df_tgs_no_ones = (
        groups_processing.aggregate_treatments_rows_with_individual_samples(
            carbonate_df_tgs
        )
    )
    # calculate effect size
    print("\nCalculating effect sizes...")
    effects_df = calculate_effect_for_df(carbonate_df_tgs_no_ones).reset_index(
        drop=True
    )

    # save results
    save_cols = file_ops.read_yaml(config.resources_dir / "mapping.yaml")["save_cols"]
    effects_df["year"] = pd.to_datetime(effects_df["year"]).dt.strftime(
        "%Y"
    )  # cast year from pd.timestamp to integer
    # check for missing columns in save_cols
    missing_columns = [col for col in save_cols if col not in effects_df.columns]
    if missing_columns:
        print(
            f"\nWARNING: The following columns in save_cols are not in effects_df: {missing_columns}"
        )
        # filter save_cols to only include columns that exist in effects_df
        available_save_cols = [col for col in save_cols if col in effects_df.columns]
        effects_df[available_save_cols].to_csv(
            config.tmp_data_dir / "effect_sizes.csv", index=False
        )
    else:
        effects_df[save_cols].to_csv(
            config.tmp_data_dir / "effect_sizes.csv", index=False
        )

    print(f"\nShape of dataframe with effect sizes: {effects_df.shape}")

    return effects_df


### curve fitting
def fit_curve(
    df: pd.DataFrame, variable: str, effect_type: str, order: int
) -> sm.regression.linear_model.RegressionResultsWrapper:
    """
    Fit a polynomial curve to the data.

    Args:
        df (pd.DataFrame): The dataframe containing the data.
        variable (str): The independent variable.
        effect_type (str): The dependent variable.
        order (int): The order of the polynomial to fit.

    Returns:
        model: The fitted regression model.
    """
    # Remove NaNs
    df = df[df[variable].notna() & df[effect_type].notna()]

    # Create polynomial features
    X = np.vander(df[variable], N=order + 1, increasing=True)

    # Fit the model
    model = sm.OLS(df[effect_type], X).fit()
    return model


def predict_curve(
    model, x: np.ndarray, alpha=0.05
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Predict values using the fitted model with confidence intervals.

    Parameters:
    - model: The fitted regression model.
    - x (np.ndarray): The independent variable values.
    - alpha (float): Significance level for confidence intervals (default: 0.05 for 95% CI)

    Returns:
    - tuple: (predicted values, lower confidence bound, upper confidence bound)
    """
    X = np.vander(x, N=model.params.shape[0], increasing=True)
    prediction = model.get_prediction(X)
    # predictions
    predicted = prediction.predicted_mean
    # confidence intervals
    conf_int = prediction.conf_int(alpha=alpha)
    lower = conf_int[:, 0]
    upper = conf_int[:, 1]

    return predicted, lower, upper


# --- Meta-analysis ---
def process_meta_regplot_data(model, model_comps, x_mod, level, point_size, predlim):
    """
    Process data for meta-regression plotting.

    Args:
        model (rpy2.robjects.vectors.ListVector): An R rma.mv or rma model object from metafor package.
        model_comps (tuple): Model components containing predictor and response info.
        x_mod (str): Name of the moderator variable to plot on x-axis.
        level (float): Confidence level for intervals in percent.
        point_size (str or array-like): Point sizes - either "seinv" (inverse of standard error),
            "vinv" (inverse of variance), or an array of custom sizes.
        predlim (tuple[float, float], optional): Limits for predicted x-axis values (min, max).

    Returns:
        tuple: Containing processed data (xi, yi, vi, norm_weights, xs, pred, ci_lb, ci_ub,
                pred_lb, pred_ub, mod_pos)
    """
    pandas2ri.activate()  # enable automatic conversion between R and pandas
    # get index of x_mod in predictors
    mod_pos = model_comps["predictors"].index(x_mod) if isinstance(x_mod, str) else 0
    if model_comps["intercept"]:
        mod_pos += 1  # adjust for intercept if present

    # extract model components
    yi = np.array(model.rx2("yi.f"))
    vi = np.array(model.rx2("vi.f"))
    X = np.array(model.rx2("X.f"))
    xi = X[:, mod_pos]

    mask = (
        ~np.isnan(yi) & ~np.isnan(vi) & ~np.isnan(xi).any(axis=0)
    )  # handle missing values
    if not all(mask):
        yi = yi[mask]
        vi = vi[mask]
        xi = xi[mask]

    # create weight vector for point sizes
    if point_size == "seinv":
        weights = 1 / np.sqrt(vi)
    elif point_size == "vinv":
        weights = 1 / vi
    elif isinstance(point_size, (list, np.ndarray)):
        weights = np.array(point_size)
    else:
        weights = np.ones_like(yi)

    if len(weights) > 0:  # normalize weights for point sizes
        min_w, max_w = min(weights), max(weights)
        if max_w - min_w > np.finfo(float).eps:
            norm_weights = 30 * (weights - min_w) / (max_w - min_w) + 1
        else:
            norm_weights = np.ones_like(weights) * 20
    else:
        norm_weights = np.ones_like(yi) * 20

    range_xi = max(xi) - min(xi)  # create sequence of x values for the regression line
    predlim = (
        (min(xi) - 0.1 * range_xi, max(xi) + 0.1 * range_xi)
        if predlim is None
        else predlim
    )
    xs = np.linspace(predlim[0], predlim[1], 1000)

    r_xs = ro.FloatVector(xs)

    # create prediction data for the regression line
    # this requires creating a new matrix with mean values for all predictors except the moderator of interest
    predict_function = ro.r("""
    function(model, xs, mod_pos, level) {
        # Get mean values for all predictors
        X_means <- colMeans(model$X.f)
        
        # Create new data for predictions
        Xnew <- matrix(rep(X_means, each=length(xs)), nrow=length(xs))
        colnames(Xnew) <- colnames(model$X.f)
        
        # Set the moderator of interest to the sequence of values
        Xnew[,mod_pos] <- xs
        
        # Remove intercept if present in the model
        if (model$int.incl) {
            Xnew <- Xnew[,-1, drop=FALSE]
        }
        
        # Make predictions
        pred <- predict(model, newmods=Xnew, level=(level/100))
        
        # Return results
        return(pred)
    }
    """)

    ### get predictions
    try:
        pred_res = predict_function(model, r_xs, mod_pos + 1, level)  # R is 1-indexed
        pred = np.array(pred_res.rx2("pred"))
        ci_lb = np.array(pred_res.rx2("ci.lb"))
        ci_ub = np.array(pred_res.rx2("ci.ub"))
        pred_lb = np.array(pred_res.rx2("pi.lb"))
        pred_ub = np.array(pred_res.rx2("pi.ub"))
    except Exception as e:
        print(f"Error in prediction: {e}")
        print("Falling back to simplified prediction")
        # simplified fallback to at least get a regression line
        coeffs = np.array(model.rx2("b"))
        if len(coeffs) > 1:  # multiple coefficients
            if model.rx2("int.incl")[0]:  # model includes intercept
                pred = coeffs[0] + coeffs[mod_pos] * xs
            else:
                pred = coeffs[mod_pos - 1] * xs
        else:  # Single coefficient
            pred = coeffs[0] * xs
        ci_lb = pred - 1.96 * 0.5  # rough approximation
        ci_ub = pred + 1.96 * 0.5  # rough approximation
        # default values for prediction intervals if not available
        pred_lb = pred - 1.96 * 1.0  # rough approximation for prediction interval
        pred_ub = pred + 1.96 * 1.0  # rough approximation for prediction interval

    return xi, yi, vi, norm_weights, xs, pred, ci_lb, ci_ub, pred_lb, pred_ub, mod_pos


def process_df_for_r(df: pd.DataFrame) -> pd.DataFrame:
    """
    Processes a pandas DataFrame by converting columns to floats if possible,
    otherwise keeping them as their original type.

    Parameters:
        df (pd.DataFrame): The input DataFrame to process.

    Returns:
        pd.DataFrame: The processed DataFrame with updated column types.
    """
    df_copy = df.copy()
    for col in df_copy.columns:
        # Only convert columns that are predominantly numeric
        if pd.to_numeric(df_copy[col], errors="coerce").notna().sum() > 0.5 * len(
            df_copy
        ):
            df_copy[col] = pd.to_numeric(df_copy[col], errors="coerce")

    return df_copy


def preprocess_df_for_meta_model(
    df: pd.DataFrame,
    effect_type: str = "hedges_g",
    effect_type_var: bool = None,
    treatment: list[str] = None,
    necessary_vars: list[str] = None,
    formula: str = None,
) -> pd.DataFrame:
    # TODO: get necessary variables more dynamically (probably via a mapping including factor)
    data = df.copy()

    effect_type_var = effect_type_var or f"{effect_type}_var"

    ### specify model
    if not formula:
        formula = analysis_utils.generate_metaregression_formula(
            effect_type, treatment, variables=necessary_vars
        )

    formula_comps = analysis_utils.get_formula_components(formula)
    # select only rows relevant to treatment
    if treatment:
        if isinstance(treatment, list):
            data = data[data["treatment"].astype(str).isin(treatment)]
        else:
            data = data[data["treatment"] == treatment]

    n_investigation = len(data)
    # remove nans for subset effect_type
    required_columns = [
        effect_type,
        effect_type_var,
        "original_doi",
        "ID",
    ] + formula_comps["predictors"]
    data = data.dropna(
        subset=[
            required_col for required_col in required_columns if required_col != "1"
        ]
    )

    # data = process_df_for_r(data)
    data = data.convert_dtypes()

    n_nans = n_investigation - len(data)

    ### summarise processing
    print("\n----- PROCESSING SUMMARY -----")
    print("Treatment: ", treatment)
    print("Total samples in input data: ", len(df))
    print("Total samples of relevant investigation: ", n_investigation)
    print("Dropped due to NaN values in required columns:", n_nans)
    print(
        f"Final sample count: {len(data)} ({n_nans + (len(df) - n_investigation)} rows dropped)"
    )

    # remove outliers
    nparams = len(formula.split("+"))
    data, outliers = remove_cooks_outliers(
        data, effect_type=effect_type, nparams=nparams
    )

    return formula, data


def run_metafor_mv(
    df: pd.DataFrame,
    effect_type: str = "hedges_g",
    effect_type_var: str = None,
    treatment: str = None,
    necessary_vars: list[str] = None,
    formula: str = None,
) -> tuple[ro.vectors.DataFrame, ro.vectors.DataFrame, pd.DataFrame]:
    """
    Run the metafor model on the given dataframe.

    Args:
        df (pd.DataFrame): The dataframe to run the model on.
        effect_type (str): The type of effect to use.
        treatment (str): The treatment to use.
        necessary_vars (list[str]): The necessary variables to use.
        formula (str): The formula to use.

    Returns:
        ro.vectors.DataFrame: The results of the metafor model.
    """
    effect_type_var = effect_type_var or f"{effect_type}_var"
    # preprocess the dataframe
    formula, df = preprocess_df_for_meta_model(
        df, effect_type, effect_type_var, treatment, necessary_vars, formula
    )
    print(f"Using formula {formula}")
    # activate R conversion
    ro.pandas2ri.activate()

    formula_comps = analysis_utils.get_formula_components(formula)
    all_necessary_vars = (
        ["original_doi", "ID"]
        + (necessary_vars or [])
        + [effect_type, effect_type_var]
        + formula_comps["predictors"]
    )
    # ensure original_doi is string type to avoid conversion issues
    df = df.copy()
    df["original_doi"] = df["original_doi"].astype(str)

    df_subset = df[
        [necessary_var for necessary_var in all_necessary_vars if necessary_var != "1"]
    ]
    df_r = ro.pandas2ri.py2rpy(df_subset)

    # run the metafor model
    print("\nRunning metafor model...")
    model = metafor.rma_mv(
        yi=ro.FloatVector(df_r.rx2(effect_type)),
        V=ro.FloatVector(df_r.rx2(effect_type_var)),
        data=df_r,
        mods=ro.Formula(formula),
        random=ro.Formula("~ 1 | original_doi/ID"),
    )
    print("Model fitting complete.")
    return model, base.summary(model), formula, df


def run_parallel_dredge(
    df: pd.DataFrame,
    global_formula: str = None,
    effect_type: str = "hedges_g",
    x_var: str = "temp",
    n_cores: int = 16,
) -> pd.DataFrame:
    """TODO: get actually working in parallel
    Runs a parallel dredge analysis using MuMIn in R.

    Parameters:
        df (rpy2.robjects.vectors.DataFrame): The dataframe in R format.
        global_formula (str): The global formula for the model.
        effect_type (str): The effect type (e.g., 'hedges_g').
        x_var (str): The independent variable (e.g., 'delta_t').
        n_cores (int): Number of cores for parallel processing.

    Returns:
        pandas.DataFrame: The dredge result converted to a pandas DataFrame.
    """
    os.environ["LC_ALL"] = "en_US.UTF-8"  # Set locale to UTF-8

    # assign variables to R environment
    ro.r.assign("df_r", ro.pandas2ri.py2rpy(df))  # convert to R dataframe
    df_r = ro.pandas2ri.py2rpy(df)
    ro.r.assign("effect_col", effect_type)
    ro.r.assign("var_col", f"{effect_type}_var")
    ro.r.assign("original_doi", df_r.rx2("original_doi"))
    ro.r.assign("ID", df_r.rx2("ID"))

    # set up formula
    global_formula = (
        f"{effect_type} ~ {x_var} - 1" if global_formula is None else global_formula
    )
    print(global_formula)
    ro.r.assign("global_formula", ro.Formula(global_formula))

    # run the R code for parallel dredge
    ro.r(f"""
    # Set up for MuMIn
    eval(metafor:::.MuMIn)

    global_model <- rma.mv(
        yi = df_r[[effect_col]],
        V = df_r[[var_col]], 
        mods = global_formula,
        random = ~ 1 | original_doi/ID,
        data=df_r,
    )
    # Create cluster
    clu <- parallel::makeCluster({n_cores})
    # Load packages on each worker
    parallel::clusterEvalQ(clu, {{
      library(metafor)
      library(MuMIn)
    }})

    # Export required variables to each worker
    parallel::clusterExport(clu, varlist = c("df_r", "effect_col", "var_col", "global_formula", "original_doi", "ID"))

    dredge_result <- MuMIn::dredge(global_model)

    # Stop the cluster
    parallel::stopCluster(clu)
    """)

    # retrieve the dredge result and convert to pandas DataFrame
    dredge_result = ro.r("dredge_result")
    # convert to pandas DataFrame
    ro.pandas2ri.activate()
    df = ro.pandas2ri.rpy2py(dredge_result)
    # assign any values of '-2147483648' to NaN (R's placeholder for NA in string columns)
    return df.replace(-2147483648, np.nan)


def predict_model(model, newmods: pd.DataFrame) -> pd.DataFrame:
    """
    Provide a model with a dataframe of moderator values (same dimensions as model X) to get model predictions.
    """
    # convert to R matrix (newmods)
    newmods_np = np.array(newmods, dtype=float)
    newmods_r = ro.r.matrix(
        ro.FloatVector(newmods_np.flatten()), nrow=newmods_np.shape[0], byrow=True
    )

    # predict all at once in R
    predictions_r = ro.r("predict")(model, newmods=newmods_r, digits=2)

    r_selected_columns = ro.r("as.data.frame")(predictions_r).rx(
        True, ro.IntVector([1, 2, 3, 4, 5, 6])
    )  # select columns to avoid heterogenous shape
    ro.pandas2ri.activate()

    # convert the selected columns to a pandas dataframe
    with (ro.default_converter + ro.pandas2ri.converter).context():
        return (
            ro.conversion.get_conversion()
            .rpy2py(r_selected_columns)
            .reset_index(drop=True)
        )


def generate_location_specific_predictions(
    model, df: pd.DataFrame, scenario_var: str = "sst", moderator_pos: int = None
) -> list[dict]:
    # TODO: make this more general
    # get constant terms from the model matrix (excluding intercept/first column)
    model_matrix = ro.r("model.matrix")(model)
    if moderator_pos:  # if moderator position provided, use it
        # take mean of all columns except for moderator pos
        const_terms = np.mean(
            np.delete(np.array(model_matrix)[:, 1:], moderator_pos - 1, axis=1), axis=0
        )

    else:
        const_terms = np.array(model_matrix)[:, 1:].mean(axis=0)
    const_terms_list = const_terms.tolist()

    df = (
        df.sort_index()
    )  # sort the index to avoid PerformanceWarning about lexsort depth
    locations = df.index.unique()
    prediction_rows = []  # to hold newmods inputs
    metadata_rows = []  # to track what each row corresponds to

    for location in tqdm(
        locations, desc=f"Generating batched predictions for {scenario_var}"
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
                    mean_scenario = base - base  # think this causes issues?
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
                    # generate predictions for mean, p10, and p90 scenarios
                for percentile, anomaly in [
                    ("mean", mean_scenario),
                    ("p10", p10_scenario),
                    ("p90", p90_scenario),
                ]:
                    prediction_rows.append([anomaly] + const_terms_list)
                    metadata_rows.append(
                        {
                            "doi": location[0],
                            "location": location[1],
                            "longitude": location[2],
                            "latitude": location[3],
                            "scenario_var": scenario_var,
                            "scenario": scenario,
                            "time_frame": time_frame,
                            "anomaly_value": anomaly,
                            "percentile": percentile,
                        }
                    )

    # convert to R matrix (newmods)
    newmods_np = np.array(prediction_rows, dtype=float)
    newmods_r = ro.r.matrix(
        ro.FloatVector(newmods_np.flatten()), nrow=newmods_np.shape[0], byrow=True
    )

    # predict all at once in R
    predictions_r = ro.r("predict")(model, newmods=newmods_r, digits=2)
    predicted_vals = list(predictions_r)

    # combine metadata and predictions
    for i, val in enumerate(
        predicted_vals[0]
    ):  # for now, just taking mean predictions (ignoring ci, pi)
        metadata_rows[i]["predicted_effect_size"] = val

    return metadata_rows


class MetaForModel:
    """
    Encapsulates metafor model handling via rpy2.
    Handles data preparation, model fitting, prediction, and diagnostics.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        effect_type: str = "hedges_g",
        effect_type_var: Optional[str] = None,
        treatment: Optional[str] = None,
        necessary_vars: Optional[List[str]] = None,
        formula: Optional[str] = None,
        random: str = "~ 1 | original_doi/ID",
    ):
        self.df = df.copy()
        self.effect_type = effect_type
        self.effect_type_var = effect_type_var or f"{effect_type}_var"
        self.treatment = treatment
        self.necessary_vars = necessary_vars
        self.formula = formula
        self.random = random
        self.model = None
        self.summary = None
        self.fitted = False
        self._prepare_data()

    def _prepare_data(self):
        """Preprocess and subset the DataFrame for R model fitting."""
        # Use preprocess_df_for_meta_model logic
        from calcification.analysis.analysis import (
            get_formula_components,
            preprocess_df_for_meta_model,
        )

        self.formula, self.df = preprocess_df_for_meta_model(
            self.df,
            self.effect_type,
            self.effect_type_var,
            self.treatment,
            self.necessary_vars,
            self.formula,
        )
        self.formula_comps = get_formula_components(self.formula)
        # Ensure original_doi is string
        self.df["original_doi"] = self.df["original_doi"].astype(str)

    def fit(self):
        """Fit the metafor model using rpy2."""
        with (ro.default_converter + pandas2ri.converter).context():
            df_r = pandas2ri.py2rpy(self.df)
            self.model = metafor.rma_mv(
                yi=ro.FloatVector(df_r.rx2(self.effect_type)),
                V=ro.FloatVector(df_r.rx2(self.effect_type_var)),
                data=df_r,
                mods=ro.Formula(self.formula),
                random=ro.Formula(self.random),
            )
            self.summary = base.summary(self.model)
            self.fitted = True
        return self

    def predict(self, newmods: pd.DataFrame) -> pd.DataFrame:
        """Predict using the fitted model for new moderator values."""
        if not self.fitted:
            raise RuntimeError("Model must be fitted before prediction.")
        from calcification.analysis.analysis import predict_model

        return predict_model(self.model, newmods)

    def get_coefficients(self) -> np.ndarray:
        """Extract coefficients from the fitted R model."""
        if not self.fitted:
            raise RuntimeError("Model must be fitted before extracting coefficients.")
        return np.array(self.model.rx2("b"))

    def get_summary(self) -> Any:
        """Return the R summary object for the fitted model."""
        if not self.fitted:
            raise RuntimeError("Model must be fitted before getting summary.")
        return self.summary

    def get_formula(self) -> str:
        return self.formula

    @staticmethod
    def generate_metaregression_formula(
        effect_type: str,
        treatment: Optional[str] = None,
        variables: Optional[List[str]] = None,
        include_intercept: bool = False,
    ) -> str:
        return analysis_utils.generate_metaregression_formula(
            effect_type, treatment, variables, include_intercept
        )

    @staticmethod
    def get_formula_components(formula: str) -> Dict[str, Any]:
        from calcification.analysis.analysis import get_formula_components

        return get_formula_components(formula)
