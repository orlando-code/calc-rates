import numpy as np
import pandas as pd
import statsmodels.api as sm
from rpy2.robjects import default_converter, r
from rpy2.robjects.conversion import localconverter
from scipy.interpolate import make_interp_spline
from scipy.stats import median_abs_deviation
from scipy.stats import norm as scipy_norm

from calcification.utils import config, file_ops


def generate_metaregression_formula(
    effect_type: str,
    treatment: str = None,
    variables: list[str] = None,
    include_intercept: bool = False,
) -> str:
    variables = variables or []
    if treatment:

        def add_treatment_vars(t):
            if t == "phtot":
                variables.append("delta_ph")
            elif t == "temp":
                variables.append("delta_t")
            elif t in ["phtot_mv", "temp_mv", "phtot_temp_mv"]:
                variables.append("delta_ph")
                variables.append("delta_t")
            else:
                raise ValueError(f"Unknown treatment: {t}")

        if isinstance(treatment, list):
            for t in treatment:
                add_treatment_vars(t)
        else:
            add_treatment_vars(treatment)

    variable_mapping = file_ops.read_yaml(config.resources_dir / "mapping.yaml")[
        "meta_model_factor_variables"
    ]

    # process variables
    variables = list(set(variables))
    factor_vars = [f"factor({v})" for v in variables if v in variable_mapping]
    other_vars = [v for v in variables if v not in variable_mapping]

    # combine into formula string
    formula = f"{effect_type} ~ {' + '.join(other_vars + factor_vars)}"

    # remove intercept if specified
    return formula + " - 1" if not include_intercept else formula


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


def summarize_metafor_models(model_summaries, model_names=None):
    """
    Extracts model diagnostics from a list of metafor model summaries (as ListVectors via rpy2).

    Parameters:
        model_summaries (list): List of <rpy2.robjects.vectors.ListVector> metafor summaries.

    Returns:
        pd.DataFrame: Summary table of key diagnostics.
    """
    summary_data = []

    for i, (summary, name) in enumerate(zip(model_summaries, model_names), start=1):
        with localconverter(default_converter) as _:
            # keys = [str(k) for k in list(summary.names)]
            topline_summary = list(r.fitstats(summary))
        # Convert R ListVector to Python dictionary
        s = {str(k): summary.rx2(str(k)) for k in summary.names}

        # Extract values safely
        loglik, deviance, AIC, BIC, AICc = [
            topline_summary[i] for i in range(len(topline_summary))
        ]

        QE_stat = s.get("QE", [None])[0]
        # QE_df = s.get("QE", [None])[1] if len(s.get("QE", [])) > 1 else None
        QE_pval = s.get("QEp", [None])[0]

        QM_stat = s.get("QM", [None])[0]
        # QM_df = s.get("QM", [None])[1] if len(s.get("QM", [])) > 1 else None
        QM_pval = s.get("QMp", [None])[0]

        var_comp = s.get("sigma2", [None, None])
        sigma2_study = var_comp[0]
        sigma2_within = var_comp[1] if len(var_comp) > 1 else None

        summary_data.append(
            {
                "Model": name,
                "Log-likelihood": f"{loglik:.0f}" if loglik is not None else None,
                "Deviance": f"{deviance:.0f}" if deviance is not None else None,
                "AIC": f"{AIC:.0f}" if AIC is not None else None,
                "AICc": f"{AICc:.0f}" if AICc is not None else None,
                # "QE (df)": f"{int(QE_df)}" if QE_df else None,
                "QE stat": f"{QE_stat:.0f}" if QE_stat is not None else None,
                "QE p-val": f"{QE_pval:.4f}" if QE_pval is not None else None,
                # "QM (df)": f"{int(QM_df)}" if QM_df else None,
                "QM stat": f"{QM_stat:.0f}" if QM_stat is not None else None,
                "QM p-val": f"{QM_pval:.4f}" if QM_pval is not None else None,
                "σ² (Study)": f"{sigma2_study:.0f}"
                if sigma2_study is not None
                else None,
                "σ² (Within)": f"{sigma2_within:.0f}"
                if sigma2_within is not None
                else None,
            }
        )

    return pd.DataFrame(summary_data).reset_index(drop=True)


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
