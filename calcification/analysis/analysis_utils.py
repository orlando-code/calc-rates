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
    effect_type_var: bool = None,
    treatment: list[str] = None,
    formula: str = None,
    verbose: bool = True,
) -> pd.DataFrame:
    # TODO: get necessary variables more dynamically (probably via a mapping including factor)
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
    required_columns = _get_required_columns(treatment, effect_type, effect_type_var)
    data = data.dropna(
        subset=[
            required_col for required_col in required_columns if required_col != "1"
        ]  # TODO: what is this doing?
    )
    data = data.convert_dtypes()

    n_nans = n_investigation - len(data)

    # remove outliers
    nparams = len(formula.split("+"))
    data, n_cooks_outliers = analysis.remove_cooks_outliers(
        data, effect_type=effect_type, nparams=nparams, verbose=verbose
    )

    if verbose:
        # summarise processing
        print("\n----- PROCESSING SUMMARY -----")
        print("Treatment: ", treatment)
        print("Total samples in input data: ", len(df))
        print("Total samples of relevant investigation: ", n_investigation)
        print("Dropped due to NaN values in required columns:", n_nans)
        print("Dropped due to Cook's distance:", len(n_cooks_outliers))
        print(
            f"Final sample count: {len(data)} ({n_nans + (len(df) - n_investigation)} rows dropped)"
        )

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
    treatment, effect_type, effect_type_var, required_columns=None
):
    # if required_columns is not None:
    #     return required_columns

    treatment_vars = _get_treatment_vars(treatment)
    effect_type_var = effect_type_var or f"{effect_type}_var"
    base_columns = [
        effect_type,
        effect_type_var,
        "original_doi",
        "ID",
        "core_grouping",
    ] + treatment_vars
    return base_columns + required_columns if required_columns else base_columns


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
        }
    """
    # split formula into response and predictors
    response_part, predictor_part = formula.split("~", 1)
    response = response_part.strip()
    predictors_str = predictor_part.strip()

    # handle intercept removal: replace '-1' with a marker
    predictors_str = predictors_str.replace(" ", "")
    predictors_str = predictors_str.replace("-1", "+__NO_INTERCEPT__")

    # split predictors on '+'
    predictor_terms = predictors_str.split("+")

    # flatten interaction terms (e.g., x1*x2 -> x1, x2)
    predictors = []
    for term in predictor_terms:
        if "*" in term:  # interaction term: split to get individual predictors
            predictors.extend(term.split("*"))
        else:
            predictors.append(term)

    # determine if intercept is included
    intercept = "__NO_INTERCEPT__" not in predictors
    predictors = [p for p in predictors if p and p != "__NO_INTERCEPT__"]

    # remove 'factor()' wrapper if present
    predictors = [
        p.replace("factor(", "").replace(")", "") if p.startswith("factor(") else p
        for p in predictors
    ]

    # remove any empty strings (could happen if formula is malformed)
    predictors = [p for p in predictors if p]

    return {
        "response": response,
        "predictors": predictors,
        "intercept": intercept,
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


def get_moderator_index(model: ro.vectors.ListVector, moderator_name: str) -> int:
    """Get the index of the moderator variable in the predictors list, accounting for intercept."""
    return get_moderator_names(model).index(moderator_name)
