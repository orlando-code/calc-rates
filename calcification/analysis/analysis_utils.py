import numpy as np
import pandas as pd
import rpy2.robjects as ro
import statsmodels.api as sm
from scipy.interpolate import make_interp_spline
from scipy.stats import median_abs_deviation
from scipy.stats import norm as scipy_norm

from calcification.analysis import analysis
from calcification.utils import config, file_ops


def preprocess_df_for_meta_model(
    df: pd.DataFrame,
    effect_type: str = "st_relative_calcification",
    treatment: list[str] | str = None,
    formula_components: dict = None,
    drate_dvar_threshold: float = 100,
    verbose: bool = True,
) -> pd.DataFrame:
    data = df.copy()
    df["original_doi"] = df["original_doi"].astype(str)

    # select only rows relevant to treatment
    if treatment:
        if isinstance(treatment, list):
            data = data[data["treatment"].astype(str).isin(treatment)]
        else:
            data = data[data["treatment"] == treatment]

    n_investigation = len(data)
    # remove nans for subset effect_type
    required_columns = _get_required_columns(treatment, effect_type, formula_components)
    data = data.dropna(subset=required_columns)
    data = data.convert_dtypes()

    n_nans = n_investigation - len(data)
    # be more descriptive about where the nans are (print the number of nans for each column)

    # remove outliers
    # nparams = len(formula.split("+"))
    nparams = (
        len(formula_components["predictors"]) + 1
        if formula_components["intercept"]
        else len(formula_components["predictors"])
    )
    data, cooks_outliers = analysis.remove_cooks_outliers(
        data, effect_type=effect_type, nparams=nparams, verbose=False
    )

    if verbose:
        # summarise processing
        print("\n----- PROCESSING SUMMARY -----")
        print("Treatment: ", treatment)
        print("Total samples in input data: ", len(df))
        print("Total samples of relevant investigation: ", n_investigation)
        print("Dropped due to NaN values: ", n_nans)
        nan_counts = df[required_columns].isna().sum()
        for col, count in nan_counts.items():
            if count > 0:
                print(f"\t{col}: {count} NaNs")
        print("Dropped due to Cook's distance: ", len(cooks_outliers))
        print(
            f"Final sample count: {len(data)} ({len(cooks_outliers) + n_nans + (len(df) - n_investigation)} rows dropped)\n"
        )

    return data


def calculate_dvar(data: pd.DataFrame, treatment: str | list[str]) -> pd.DataFrame:
    """Calculate the dvar for the data."""
    # for each treatment, calculate the change in st_calcification wrt the control
    # then calculate the dvar for each treatment
    if isinstance(treatment, str):
        treatment = [treatment]
        for t in treatment:
            d_calcification = (
                data["st_treatment_calcification"] - data["st_control_calcification"]
            )
            data[f"dvar_{t}"] = d_calcification / data[t]

    return data


def generate_metaregression_formula(
    effect_type: str,
    treatment: str = None,
    include_intercept: bool = False,
) -> str:
    treatment_vars = _get_treatment_vars(treatment)
    variable_mapping = file_ops.read_yaml(config.resources_dir / "mapping.yaml")[
        "meta_model_factor_variables"
    ]
    factor_vars = [f"factor({v})" for v in treatment_vars if v in variable_mapping]
    other_vars = [v for v in treatment_vars if v not in variable_mapping]

    # combine into formula string
    formula = f"{effect_type} ~ {' + '.join(other_vars + factor_vars)}"

    # remove intercept if specified
    return formula + " - 1" if not include_intercept else formula


def _get_required_columns(
    treatment, effect_type, formula_components, required_columns=None
):
    # get required columns from formula components
    formula_requirements = formula_components["raw_predictors"]
    if formula_requirements == ["0"]:
        raise ValueError(
            "Metafor formula requires at least one predictor e.g. an intercept"
        )

    effect_type_var = f"{effect_type}_var"
    base_columns = [
        "original_doi",
        "ID",
        "core_grouping",
        "st_calcification_unit",
        effect_type,
        effect_type_var,
    ]
    use_columns = (
        base_columns + formula_requirements
        if formula_requirements != ["1"]
        else base_columns
    )
    return use_columns + required_columns if required_columns else use_columns


def _get_treatment_vars(treatment: str) -> list[str]:
    """Get the treatment variables for the model."""

    def _get_treatment_var_from_single_treatment(treatment: str) -> list[str]:
        if treatment == "phtot":
            return ["delta_ph"]
        elif treatment == "temp":
            return ["delta_t"]
        elif treatment in ["phtot_mv", "temp_mv", "phtot_temp_mv"]:
            return ["delta_ph", "delta_t"]
        else:
            raise ValueError(f"Unknown treatment: {treatment}")

    treatment_vars = []
    if isinstance(treatment, list):
        for t in treatment:
            treatment_vars.extend(_get_treatment_var_from_single_treatment(t))
    else:
        treatment_vars.extend(_get_treatment_var_from_single_treatment(treatment))

    return list(set(treatment_vars))


def get_formula_components(formula: str) -> dict:
    """
    Extracts the response variable, predictors, and intercept flag from a formula string.

    Args:
        formula (str): A formula string, e.g. "y ~ x1 + x2 - 1" or "y ~ factor(x1) + x2*x3".

    Returns:
        dict: {
            "response": str,
            "predictors": list[str],
            "intercept": bool
            "factor_mods": list[str],
            "nonlinear_mods": list[str],
        }
    """
    # split formula into response and predictors
    print(formula)
    response_part, predictor_part = [p.strip() for p in formula.split("~", 1)]

    # handle intercept removal: replace '-1' with a marker
    predictor_part = predictor_part.replace(" ", "")
    predictor_part = predictor_part.replace("-1", "+__NO_INTERCEPT__")

    # split predictors on '+'
    predictor_terms = predictor_part.split("+")

    predictors = []
    # flatten interaction terms (e.g., x1*x2 -> x1, x2)
    interaction_terms = [p for p in predictor_terms if "*" in p]
    for interaction_term in interaction_terms:
        predictors.extend(interaction_term.split("*"))

    for term in predictor_terms:
        if term not in interaction_terms:
            predictors.append(term)

    # determine if intercept is included
    intercept = "__NO_INTERCEPT__" not in predictors
    predictors = [p for p in predictors if p and p != "__NO_INTERCEPT__"]

    # determine factor moderators
    factor_mods = [p for p in predictors if p.startswith("factor(")]
    # remove 'factor()' wrapper if present
    raw_factor_mods = [p.replace("factor(", "").replace(")", "") for p in factor_mods]
    raw_factor_mods = list(set(raw_factor_mods))

    # determine non-linear moderators
    nonlinear_mods = [p for p in predictors if "I(" in p]
    raw_nonlinear_mods = [p.split("^")[0].replace("I(", "") for p in nonlinear_mods]
    raw_nonlinear_mods = [p.replace(")", "") for p in raw_nonlinear_mods]
    raw_nonlinear_mods = list(set(raw_nonlinear_mods))

    # get list of the raw moderators (e.g. if I(x^2) -> x, factor(x) -> x, x1*x2 -> x1, x2)
    raw_predictors = list(
        set(
            raw_nonlinear_mods
            + raw_factor_mods
            + interaction_terms
            + [
                p
                for p in predictors
                if p not in interaction_terms
                and p not in factor_mods
                and p not in nonlinear_mods
            ]
        )
    )

    # return list of moderators
    return {
        "response": response_part,
        "predictors": predictors,
        "raw_predictors": raw_predictors,
        "factor_mods": raw_factor_mods,
        "nonlinear_mods": raw_nonlinear_mods,
        "intercept": intercept,
    }

    # remove any empty strings (could happen if formula is malformed)
    predictors = [p for p in predictors if p]

    # get raw components e.g. if I(delta_t^2) -> delta_t
    raw_predictors = [
        p.split("^")[0].replace("I(", "") if p.startswith("I(") and "^" in p else p
        for p in predictors
    ]
    # drop duplicates
    raw_predictors = list(set(raw_predictors))

    return {
        "response": response_part,
        "predictors": predictors,
        "intercept": intercept,
        "raw_predictors": raw_predictors,
    }


def p_score(prediction: float, se: float, null_value: float = 0) -> float:
    """
    Calculate the p-value for a given prediction and standard error.
    """
    z = (prediction - null_value) / se
    p = 2 * (1 - scipy_norm.cdf(abs(z)))  # two-tailed p-value
    return p


### assign certainty levels
def assign_certainty(p_score: float) -> int:
    """
    Assign certainty levels based on p-value.
    """
    if p_score < 0.01:
        return 4  # very high certainty
    elif p_score < 0.05:
        return 3  # high certainty
    elif p_score < 0.1:
        return 2  # medium certainty
    else:
        return 1  # low certainty


def filter_robust_zscore(series: pd.Series, threshold: float = 20) -> pd.Series:
    """
    Filter out outliers based on robust z-scores.

    Args:
        series (pd.Series): The series to filter.
        threshold (float): The z-score threshold for filtering.

    Returns:
        pd.Series: A boolean series indicating which values are not outliers.
    """
    median = np.median(series)
    mad = median_abs_deviation(
        series, scale="normal"
    )  # scale for approx equivalence to std dev
    robust_z = np.abs((series - median) / mad)
    return robust_z < threshold


def extrapolate_predictions(df, year=2100):
    grouping_cols = ["scenario", "percentile", "core_grouping", "time_frame"]
    value_cols = [col for col in df.columns if col not in grouping_cols]

    new_rows = []

    for (scenario, percentile, core_grouping), group_df in df.groupby(
        ["scenario", "percentile", "core_grouping"]
    ):
        group_df = group_df[group_df["time_frame"] > 1995]

        if group_df.empty:
            continue

        interp_xs = group_df["time_frame"].values

        # Prepare a dictionary for the new row (constant fields first)
        new_row = {
            "scenario": scenario,
            "percentile": percentile,
            "core_grouping": core_grouping,
            "time_frame": year,
        }

        for value_col in value_cols:
            inter_ys = group_df[value_col].values

            # Need at least 2 points to interpolate/extrapolate
            if len(interp_xs) < 2:
                continue

            spline = make_interp_spline(
                interp_xs, inter_ys, k=min(2, len(interp_xs) - 1)
            )
            value_at_year = float(spline(year))  # returns as array

            new_row[value_col] = value_at_year

        new_rows.append(new_row)

    # Add the new rows to the original dataframe
    df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)

    return df


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


def get_moderator_names(model: ro.vectors.ListVector) -> list[str]:
    """Get the names of the moderators from the model."""
    x = ro.r.assign("x", model)  # noqa
    beta_rownames = []
    for rn in ro.r("rownames(x$beta)"):
        beta_rownames.append(str(rn))
    return beta_rownames


def get_moderator_index(
    model: ro.vectors.ListVector, moderator_names: list[str]
) -> int:
    """Get the index of the moderator variable in the predictors list, accounting for intercept."""
    return (
        get_moderator_names(model).index(moderator_names)
        if isinstance(moderator_names, str)
        else [get_moderator_names(model).index(name) for name in moderator_names]
    )
