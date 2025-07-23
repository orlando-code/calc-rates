from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import rpy2.robjects as ro

# --- metafor ---
import rpy2.robjects.packages as rpackages
from rpy2.robjects.packages import importr
from tqdm.auto import tqdm

from calcification.analysis import analysis, analysis_utils

metafor = importr("metafor")
base = importr("base")
pandas2ri = rpackages.importr("pandas2ri")
os = importr("os")


# --- curve fitting ---
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
    data, _ = (
        analysis.remove_cooks_outliers(data, effect_type=effect_type, nparams=nparams),
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

        self.formula, self.df = preprocess_df_for_meta_model(
            self.df,
            self.effect_type,
            self.effect_type_var,
            self.treatment,
            self.necessary_vars,
            self.formula,
        )
        self.formula_comps = analysis_utils.get_formula_components(self.formula)
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
        return analysis_utils.get_formula_components(formula)
