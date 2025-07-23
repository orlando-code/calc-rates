import logging
import os
from dataclasses import dataclass
from typing import Any, List, Optional

import numpy as np
import pandas as pd
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri

# --- metafor ---
from rpy2.robjects.packages import importr
from tqdm.auto import tqdm

from calcification.analysis import analysis, analysis_utils

metafor = importr("metafor")
base = importr("base")

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class MetaforModel:
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
        self.formula, self.df = preprocess_df_for_meta_model(
            self.df,
            self.effect_type,
            self.effect_type_var,
            self.treatment,
            self.necessary_vars,
            self.formula,
        )
        self.formula_components = analysis_utils.get_formula_components(self.formula)

    def _get_model_components(self):
        """Get the components of the model."""
        return get_metafor_model_components(
            self.effect_type,
            self.effect_type_var,
            self.necessary_vars,
            self.formula,
            self.treatment,
        )

    def _fit_metafor_model(self):
        """Fit the metafor model using rpy2."""
        logging.info("Fitting metafor model...")
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
        logger.info("Model fitting complete.")
        return self

    def predict_on_moderator_values(self, moderator_vals: pd.DataFrame) -> pd.DataFrame:
        """Predict using the fitted model for new moderator values."""
        if not self.fitted:
            raise RuntimeError("Model must be fitted before prediction.")

        return predict_metafor_model_on_moderator_values(self.model, moderator_vals)

    def get_coefficients(self) -> np.ndarray:
        """Extract coefficients from the fitted R model."""
        if not self.fitted:
            raise RuntimeError("Model must be fitted before extracting coefficients.")
        return np.array(self.model.rx2("b"))

    def get_model_metadata(self) -> dict:
        """Get the metadata of the model."""
        return {
            "formula": self.formula,
            "formula_components": self.formula_components,
        }

    def get_summary(self) -> Any:
        """Return the R summary object for the fitted model."""
        if not self.fitted:
            raise RuntimeError("Model must be fitted before getting summary.")
        model_summary = analysis_utils.summarise_metafor_model(self.summary)
        model_summary.update(self.get_model_metadata())
        return model_summary

    def run(self):
        """Run the metafor model."""
        self._get_model_components()
        self._prepare_data()
        self._fit_metafor_model()
        return self


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
def _get_mod_pos(model_comps, x_mod):
    """Get the index of the moderator variable in the predictors list, accounting for intercept."""
    mod_pos = model_comps["predictors"].index(x_mod) if isinstance(x_mod, str) else 0
    if model_comps["intercept"]:
        mod_pos += 1  # adjust for intercept if present


def _extract_model_components(model, mod_pos):
    """Extract yi, vi, X, and xi from the model, and mask out missing values."""
    yi = np.array(model.rx2("yi.f"))
    vi = np.array(model.rx2("vi.f"))
    X = np.array(model.rx2("X.f"))
    xi = X[:, mod_pos]
    # Mask for missing values
    mask = ~np.isnan(yi) & ~np.isnan(vi) & ~np.isnan(xi).any(axis=0)
    if not all(mask):
        yi = yi[mask]
        vi = vi[mask]
        xi = xi[mask]
    return yi, vi, xi


def _compute_weights(vi, point_size, yi):
    """Compute and normalize weights for point sizes."""
    if point_size == "seinv":
        weights = 1 / np.sqrt(vi)
    elif point_size == "vinv":
        weights = 1 / vi
    elif isinstance(point_size, (list, np.ndarray)):
        weights = np.array(point_size)
    else:
        weights = np.ones_like(yi)

    if len(weights) > 0:
        min_w, max_w = min(weights), max(weights)
        if max_w - min_w > np.finfo(float).eps:
            norm_weights = 30 * (weights - min_w) / (max_w - min_w) + 1
        else:
            norm_weights = np.ones_like(weights) * 20
    else:
        norm_weights = np.ones_like(yi) * 20
    return norm_weights


def _get_xs_and_prediction_limits(xi, predlim):
    """Compute xs (x values for regression line) and predlim if not provided."""
    range_xi = max(xi) - min(xi)
    if predlim is None:
        predlim = (min(xi) - 0.1 * range_xi, max(xi) + 0.1 * range_xi)
    xs = np.linspace(predlim[0], predlim[1], 1000)
    return xs, predlim


def _get_predict_function():
    """Return the R function for prediction."""
    return ro.r("""
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


def _metafor_prediction_from_model(model, predict_function, xs, mod_pos, level):
    """Try to predict using the R function"""
    r_xs = ro.FloatVector(xs)
    try:
        pred_res = predict_function(model, r_xs, mod_pos + 1, level)  # R is 1-indexed
        pred = np.array(pred_res.rx2("pred"))
        ci_lb = np.array(pred_res.rx2("ci.lb"))
        ci_ub = np.array(pred_res.rx2("ci.ub"))
        pred_lb = np.array(pred_res.rx2("pi.lb"))
        pred_ub = np.array(pred_res.rx2("pi.ub"))
        return pred, ci_lb, ci_ub, pred_lb, pred_ub
    except Exception as e:
        print(f"Error in prediction: {e}")
        return None, None, None, None, None


def predict_metafor_model_on_moderator_values(
    model, moderator_vals: pd.DataFrame
) -> pd.DataFrame:
    """
    Provide a model with a dataframe (same dimensions as model X) of moderator values to get model predictions.
    """
    # convert to R matrix (moderator_vals)
    moderator_vals_np = np.array(moderator_vals, dtype=float)
    moderator_vals_r = ro.r.matrix(
        ro.FloatVector(moderator_vals_np.flatten()),
        nrow=moderator_vals_np.shape[0],
        byrow=True,
    )

    # predict all at once in R
    predictions_r = ro.r("predict")(model, newmods=moderator_vals_r, digits=2)

    r_selected_columns = ro.r(
        "as.data.frame"
    )(
        predictions_r
    ).rx(
        True, ro.IntVector([1, 2, 3, 4, 5, 6])
    )  # select columns to avoid heterogenous shape. TODO: surely this should be dynamic?
    ro.pandas2ri.activate()

    # convert the selected columns to a pandas dataframe
    with (ro.default_converter + ro.pandas2ri.converter).context():
        return (
            ro.conversion.get_conversion()
            .rpy2py(r_selected_columns)
            .reset_index(drop=True)
        )


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
    pandas2ri.activate()
    mod_pos = _get_mod_pos(model_comps, x_mod)
    yi, vi, xi = _extract_model_components(model, mod_pos)
    norm_weights = _compute_weights(vi, point_size, yi)
    xs, predlim = _get_xs_and_prediction_limits(xi, predlim)
    predict_function = _get_predict_function()
    pred, ci_lb, ci_ub, pred_lb, pred_ub = _metafor_prediction_from_model(
        model, predict_function, xs, mod_pos, level
    )
    return xi, yi, vi, norm_weights, xs, pred, ci_lb, ci_ub, pred_lb, pred_ub, mod_pos


def preprocess_df_for_meta_model(
    df: pd.DataFrame,
    effect_type: str = "hedges_g",
    effect_type_var: bool = None,
    treatment: list[str] = None,
    formula: str = None,
    formula_components: dict = None,
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
    required_columns = [
        effect_type,
        effect_type_var,
        "original_doi",
        "ID",
    ] + formula_components["predictors"]

    data = data.dropna(
        subset=[
            required_col for required_col in required_columns if required_col != "1"
        ]
    )
    data = data.convert_dtypes()

    n_nans = n_investigation - len(data)

    # remove outliers
    nparams = len(formula.split("+"))
    data, n_cooks_outliers = (
        analysis.remove_cooks_outliers(data, effect_type=effect_type, nparams=nparams),
    )

    # summarise processing
    print("\n----- PROCESSING SUMMARY -----")
    print("Treatment: ", treatment)
    print("Total samples in input data: ", len(df))
    print("Total samples of relevant investigation: ", n_investigation)
    print("Dropped due to NaN values in required columns:", n_nans)
    print("Dropped due to Cook's distance:", n_cooks_outliers)
    print(
        f"Final sample count: {len(data)} ({n_nans + (len(df) - n_investigation)} rows dropped)"
    )

    return formula, data


def get_metafor_model_components(
    effect_type, effect_type_var, necessary_vars, formula, treatment
) -> tuple[str, dict]:
    """Get the components of the model."""
    effect_type_var = effect_type_var or f"{effect_type}_var"

    # specify model
    if not formula:
        formula = analysis_utils.generate_metaregression_formula(
            effect_type, treatment, variables=necessary_vars
        )
    formula_components = analysis_utils.get_formula_components(formula)

    return formula, formula_components


@dataclass
class DredgeConfig:
    """Configuration for dredge analysis."""

    effect_type: str = "hedges_g"
    x_var: str = "temp"
    n_cores: int = 16
    global_formula: Optional[str] = None
    random_effects: str = "~ 1 | original_doi/ID"


class DredgeAnalysis:
    """
    Class for running MuMIn dredge analysis on meta-analysis data.

    Encapsulates parallel and simple dredge analysis functionality with
    robust error handling and automatic fallback.
    """

    def __init__(self, df: pd.DataFrame, config: Optional[DredgeConfig] = None):
        """
        Initialize DredgeAnalysis with data and configuration.

        Args:
            df: Input DataFrame with effect size data
            config: Configuration object for dredge analysis
        """
        self.df = df.copy()
        self.config = config or DredgeConfig()
        self.results = None
        self._validate_data()

    def _validate_data(self) -> None:
        """Validate that required columns exist in the DataFrame."""
        required_cols = [
            self.config.effect_type,
            f"{self.config.effect_type}_var",
            "original_doi",
        ]
        missing_cols = [col for col in required_cols if col not in self.df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

    def _setup_r_environment(self, formula: str) -> None:
        """Set up R environment variables for dredge analysis."""
        os.environ["LC_ALL"] = "en_US.UTF-8"

        # Convert DataFrame to R and assign variables
        df_r = pandas2ri.py2rpy(self.df)
        ro.r.assign("df_r", df_r)
        ro.r.assign("effect_col", self.config.effect_type)
        ro.r.assign("var_col", f"{self.config.effect_type}_var")
        ro.r.assign("original_doi", df_r.rx2("original_doi"))
        ro.r.assign("ID", df_r.rx2("ID"))
        ro.r.assign("global_formula", ro.Formula(formula))
        ro.r.assign("random_formula", ro.Formula(self.config.random_effects))

    def _create_global_model(self) -> None:
        """Create the global metafor model in R."""
        ro.r("""
        # Set up for MuMIn
        eval(metafor:::.MuMIn)

        global_model <- rma.mv(
            yi = df_r[[effect_col]],
            V = df_r[[var_col]], 
            mods = global_formula,
            random = random_formula,
            data = df_r
        )
        """)

    def _run_parallel_dredge_analysis(self) -> None:
        """Run parallel dredge analysis with cluster management."""
        ro.r(f"""
        # Load required libraries
        library(parallel)
        library(MuMIn)
        
        # Create cluster
        clu <- makeCluster({self.config.n_cores})
        
        # Load packages on each worker
        clusterEvalQ(clu, {{
            library(metafor)
            library(MuMIn)
        }})

        # Export required variables to each worker
        clusterExport(clu, varlist = c("df_r", "effect_col", "var_col", "global_formula", "random_formula", "original_doi", "ID"))

        # Run dredge analysis
        dredge_result <- dredge(global_model, cluster = clu)

        # Stop the cluster
        stopCluster(clu)
        """)

    def _run_simple_dredge_analysis(self) -> None:
        """Run simple dredge analysis without parallel processing."""
        ro.r("""
        # Load required libraries
        library(MuMIn)
        
        # Run dredge analysis
        dredge_result <- dredge(global_model)
        """)

    def _convert_r_results_to_pandas(self) -> pd.DataFrame:
        """Convert R dredge results to pandas DataFrame."""
        pandas2ri.activate()
        dredge_result = ro.r("dredge_result")
        df_result = pandas2ri.rpy2py(dredge_result)
        # Replace R's NA placeholder with NaN
        return df_result.replace(-2147483648, np.nan)

    def _generate_formula(self) -> str:
        """Generate formula for dredge analysis."""
        if self.config.global_formula is None:
            return f"{self.config.effect_type} ~ {self.config.x_var} - 1"
        return self.config.global_formula

    def run_parallel(self) -> pd.DataFrame:
        """
        Run parallel dredge analysis with automatic fallback to simple analysis.

        Returns:
            DataFrame with dredge results and model comparisons
        """
        try:
            formula = self._generate_formula()
            logger.info(f"Running parallel dredge analysis with formula: {formula}")
            logger.info(f"Using {self.config.n_cores} cores for parallel processing")

            self._setup_r_environment(formula)
            self._create_global_model()
            self._run_parallel_dredge_analysis()
            self.results = self._convert_r_results_to_pandas()

            logger.info(
                f"Parallel dredge analysis completed successfully. Found {len(self.results)} model combinations."
            )
            return self.results

        except Exception as e:
            logger.error(f"Error in parallel dredge analysis: {e}")
            logger.warning("Attempting fallback to simple dredge analysis")
            return self.run_simple()

    def run_simple(self) -> pd.DataFrame:
        """
        Run simple dredge analysis without parallel processing.

        Returns:
            DataFrame with dredge results
        """
        try:
            formula = self._generate_formula()
            logger.info(f"Running simple dredge analysis with formula: {formula}")

            self._setup_r_environment(formula)
            self._create_global_model()
            self._run_simple_dredge_analysis()
            self.results = self._convert_r_results_to_pandas()

            logger.info(
                f"Simple dredge analysis completed successfully. Found {len(self.results)} model combinations."
            )
            return self.results

        except Exception as e:
            logger.error(f"Error in simple dredge analysis: {e}")
            raise RuntimeError(f"Both parallel and simple dredge analyses failed: {e}")

    def get_best_models(self, n_models: int = 10) -> pd.DataFrame:
        """
        Get the top n best models from dredge results.

        Args:
            n_models: Number of top models to return

        Returns:
            DataFrame with best models sorted by AIC
        """
        if self.results is None:
            raise RuntimeError("No dredge results available. Run analysis first.")

        # Sort by AIC (lower is better) and return top n models
        return (
            self.results.nsmallest(n_models, "AICc")
            if "AICc" in self.results.columns
            else self.results.head(n_models)
        )

    def summarize_results(self) -> dict:
        """
        Get summary statistics of the dredge results.

        Returns:
            Dictionary with summary information
        """
        if self.results is None:
            raise RuntimeError("No dredge results available. Run analysis first.")

        return {
            "total_models": len(self.results),
            "best_aic": self.results["AICc"].min()
            if "AICc" in self.results.columns
            else None,
            "worst_aic": self.results["AICc"].max()
            if "AICc" in self.results.columns
            else None,
            "delta_aic_range": self.results["delta"].max()
            if "delta" in self.results.columns
            else None,
            "models_within_2_aic": len(self.results[self.results["delta"] <= 2])
            if "delta" in self.results.columns
            else None,
        }


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


# --- DEPRECATED ---


# def process_df_for_r(df: pd.DataFrame) -> pd.DataFrame:
#     """
#     Processes a pandas DataFrame by converting columns to floats if possible,
#     otherwise keeping them as their original type.

#     Parameters:
#         df (pd.DataFrame): The input DataFrame to process.

#     Returns:
#         pd.DataFrame: The processed DataFrame with updated column types.
#     """
#     df_copy = df.copy()
#     for col in df_copy.columns:
#         # Only convert columns that are predominantly numeric
#         if pd.to_numeric(df_copy[col], errors="coerce").notna().sum() > 0.5 * len(
#             df_copy
#         ):
#             df_copy[col] = pd.to_numeric(df_copy[col], errors="coerce")

#     return df_copy


# def run_metafor_mv(
#     df: pd.DataFrame,
#     effect_type: str = "hedges_g",
#     effect_type_var: str = None,
#     treatment: str = None,
#     necessary_vars: list[str] = None,
#     formula: str = None,
# ) -> tuple[ro.vectors.DataFrame, ro.vectors.DataFrame, pd.DataFrame]:
#     """
#     Run the metafor model on the given dataframe.

#     Args:
#         df (pd.DataFrame): The dataframe to run the model on.
#         effect_type (str): The type of effect to use.
#         treatment (str): The treatment to use.
#         necessary_vars (list[str]): The necessary variables to use.
#         formula (str): The formula to use.

#     Returns:
#         ro.vectors.DataFrame: The results of the metafor model.
#     """
#     effect_type_var = effect_type_var or f"{effect_type}_var"
#     # preprocess the dataframe
#     formula, df = preprocess_df_for_meta_model(
#         df, effect_type, effect_type_var, treatment, necessary_vars, formula
#     )
#     print(f"Using formula {formula}")
#     # activate R conversion
#     ro.pandas2ri.activate()

#     formula_components = analysis_utils.get_formula_components(formula)
#     all_necessary_vars = (
#         ["original_doi", "ID"]
#         + (necessary_vars or [])
#         + [effect_type, effect_type_var]
#         + formula_components["predictors"]
#     )
#     # ensure original_doi is string type to avoid conversion issues
#     df = df.copy()
#     df["original_doi"] = df["original_doi"].astype(str)

#     df_subset = df[
#         [necessary_var for necessary_var in all_necessary_vars if necessary_var != "1"]
#     ]
#     df_r = ro.pandas2ri.py2rpy(df_subset)

#     # run the metafor model
#     print("\nRunning metafor model...")
#     model = metafor.rma_mv(
#         yi=ro.FloatVector(df_r.rx2(effect_type)),
#         V=ro.FloatVector(df_r.rx2(effect_type_var)),
#         data=df_r,
#         mods=ro.Formula(formula),
#         random=ro.Formula("~ 1 | original_doi/ID"),
#     )
#     print("Model fitting complete.")
#     return model, base.summary(model), formula, df


# # Backwards compatibility functions using the new class
# def run_parallel_dredge(
#     df: pd.DataFrame,
#     global_formula: Optional[str] = None,
#     effect_type: str = "hedges_g",
#     x_var: str = "temp",
#     n_cores: int = 16,
# ) -> pd.DataFrame:
#     """
#     Run a parallel dredge analysis using MuMIn in R.

#     This is a wrapper around DredgeAnalysis for backwards compatibility.
#     """
#     config = DredgeConfig(
#         effect_type=effect_type,
#         x_var=x_var,
#         n_cores=n_cores,
#         global_formula=global_formula,
#     )
#     dredge = DredgeAnalysis(df, config)
#     return dredge.run_parallel()


# def run_simple_dredge(
#     df: pd.DataFrame,
#     global_formula: Optional[str] = None,
#     effect_type: str = "hedges_g",
#     x_var: str = "temp",
# ) -> pd.DataFrame:
#     """
#     Run a simple dredge analysis (fallback without parallel processing).

#     This is a wrapper around DredgeAnalysis for backwards compatibility.
#     """
#     config = DredgeConfig(
#         effect_type=effect_type, x_var=x_var, global_formula=global_formula
#     )
#     dredge = DredgeAnalysis(df, config)
#     return dredge.run_simple()


# # Remove the old helper functions since they're now part of the class
# def _setup_r_environment(df: pd.DataFrame, effect_type: str, formula: str) -> None:
#     """Deprecated: Use DredgeAnalysis class instead."""
#     logger.warning(
#         "_setup_r_environment is deprecated. Use DredgeAnalysis class instead."
#     )


# def _create_global_model() -> None:
#     """Deprecated: Use DredgeAnalysis class instead."""
#     logger.warning(
#         "_create_global_model is deprecated. Use DredgeAnalysis class instead."
#     )


# def _run_parallel_dredge_analysis(n_cores: int) -> None:
#     """Deprecated: Use DredgeAnalysis class instead."""
#     logger.warning(
#         "_run_parallel_dredge_analysis is deprecated. Use DredgeAnalysis class instead."
#     )


# def _run_simple_dredge_analysis() -> None:
#     """Deprecated: Use DredgeAnalysis class instead."""
#     logger.warning(
#         "_run_simple_dredge_analysis is deprecated. Use DredgeAnalysis class instead."
#     )


# def _convert_r_results_to_pandas() -> pd.DataFrame:
#     """Deprecated: Use DredgeAnalysis class instead."""
#     logger.warning(
#         "_convert_r_results_to_pandas is deprecated. Use DredgeAnalysis class instead."
#     )
#     return pd.DataFrame()


# def _generate_dredge_formula(
#     effect_type: str, x_var: str, global_formula: Optional[str]
# ) -> str:
#     """Deprecated: Use DredgeAnalysis class instead."""
#     logger.warning(
#         "_generate_dredge_formula is deprecated. Use DredgeAnalysis class instead."
#     )
#     if global_formula is None:
#         return f"{effect_type} ~ {x_var} - 1"
#     return global_formula
