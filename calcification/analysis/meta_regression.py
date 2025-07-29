import logging
import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri

# --- metafor ---
from rpy2.robjects.packages import importr
from tqdm.auto import tqdm

from calcification.analysis import analysis_utils
from calcification.utils import config

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
        formula: Optional[str] = None,
        random: str = "~ 1 | original_doi/ID",
        required_columns: Optional[list[str]] = None,
        save_summary: bool = False,
        verbose: bool = True,
    ):
        self.df = df.copy()
        self.effect_type = effect_type
        self.effect_type_var = effect_type_var or f"{effect_type}_var"
        self.treatment = treatment
        self.random = random
        self.verbose = verbose
        self.model = None
        self.summary = None
        self.fitted = False
        self.required_columns = analysis_utils._get_required_columns(
            self.treatment, self.effect_type, self.effect_type_var, required_columns
        )
        self.formula = self._get_model_formula() if formula is None else formula
        self.formula_components = self._get_formula_components()
        self._prepare_data()
        self._get_r_df()
        self.save_summary = save_summary

    def _get_model_formula(self):
        """Get the formula for the model."""
        return analysis_utils.generate_metaregression_formula(
            self.effect_type, self.treatment, include_intercept=False
        )

    def _prepare_data(self):
        """Preprocess and subset the DataFrame for R model fitting."""
        self.df = analysis_utils.preprocess_df_for_meta_model(
            self.df,
            self.effect_type,
            self.effect_type_var,
            self.treatment,
            self.formula,
            self.verbose,
        )

    def _get_formula_components(self) -> dict:
        return analysis_utils.get_formula_components(self.formula)

    def predict_on_moderator_values(
        self, moderator_names: list[str], moderator_vals: pd.DataFrame
    ) -> pd.DataFrame:
        """Predict using the fitted model for new moderator values."""
        # TODO: this currently not used anywhere
        if not self.fitted:
            raise RuntimeError("Model must be fitted before prediction.")

        return metafor_predict_from_model(self.model, moderator_names, moderator_vals)

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

    def get_model_summary(self) -> None:
        """Get the summary of the model."""
        print(self.summary)

    def _save_summary(self) -> None:
        """Save the summary to a file."""
        # summary_fp =
        # TODO: specify summary file path by model type
        summary_fp = config.results_dir / "metafor_summary.txt"
        with open("summary.txt", "w") as f:
            f.write(str(self.summary))
        logger.info(f"Summary saved to {summary_fp}")

    def _get_r_df(self) -> ro.vectors.DataFrame:
        """Get the R dataframe for the model."""
        df_subset = self.df[self.required_columns]
        with (ro.default_converter + pandas2ri.converter).context():
            df_r = pandas2ri.py2rpy(df_subset)
        self.df_r = df_r
        return df_r

    def fit_model(self):
        """Fit the metafor model using rpy2."""
        logging.info(f"Fitting metafor model with formula: {self.formula}")
        self.model = metafor.rma_mv(
            yi=ro.FloatVector(self.df_r.rx2(self.effect_type)),
            V=ro.FloatVector(self.df_r.rx2(self.effect_type_var)),
            data=self.df_r,
            mods=ro.Formula(self.formula),
            random=ro.Formula(self.random),
        )
        self.summary = base.summary(self.model)
        self.fitted = True
        self._save_summary() if self.save_summary else None
        logger.info("Model fitting complete.")
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
def _extract_model_components(
    model: ro.vectors.ListVector, moderator_name: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract xi (moderator values), yi (effect sizes), vi (variance) from the model, and mask out missing values."""
    moderator_index = analysis_utils.get_moderator_index(model, moderator_name)
    yi = np.array(model.rx2("yi.f"))
    vi = np.array(model.rx2("vi.f"))
    xi = np.array(model.rx2("X.f"))[:, moderator_index]
    # mask for missing values
    mask = ~np.isnan(yi) & ~np.isnan(vi) & ~np.isnan(xi).any(axis=0)
    if not all(mask):
        yi = yi[mask]
        vi = vi[mask]
        xi = xi[mask]
    return xi, yi, vi


def _compute_point_weights(
    vi: np.ndarray, weight_pt_by: str | list | np.ndarray
) -> np.ndarray:
    """Compute and normalize weights for point sizes."""
    if weight_pt_by == "seinv":
        weights = 1 / np.sqrt(vi)
    elif weight_pt_by == "vinv":
        weights = 1 / vi
    elif isinstance(weight_pt_by, (list, np.ndarray)):
        weights = np.array(weight_pt_by)
    else:
        weights = np.ones_like(vi)

    if len(weights) > 0:
        min_w, max_w = min(weights), max(weights)
        if max_w - min_w > np.finfo(float).eps:
            norm_weights = 30 * (weights - min_w) / (max_w - min_w) + 1
        else:
            norm_weights = np.ones_like(weights) * 20
    else:
        norm_weights = np.ones_like(vi) * 20
    return norm_weights


def _get_xs_and_prediction_limits(
    xi: np.ndarray,
    prediction_limits: tuple[float, float] = None,
    num_prediction_points: int = 1000,
) -> tuple[np.ndarray, tuple[float, float]]:
    """Compute xs (x values for regression line) and prediction_limits (min and max x values) if not provided."""
    range_xi = max(xi) - min(xi)
    if prediction_limits is None:
        prediction_limits = (min(xi) - 0.1 * range_xi, max(xi) + 0.1 * range_xi)
    xs = np.linspace(prediction_limits[0], prediction_limits[1], num_prediction_points)
    return xs, prediction_limits


def _build_newmods_matrix(
    model: ro.vectors.ListVector,
    moderator_names: list[str],
    xs: np.ndarray,
    npoints: int = 1000,
) -> np.ndarray:
    """
    Build a new moderator matrix for prediction.

    Args:
        model (ro.vectors.ListVector): The fitted metafor model.
        moderator_names (list[str]): list of moderator names (strings) to change
        xs (np.ndarray): np.ndarray of shape (n_points, n_moderators to change)
        npoints (int): number of points to predict on

    Returns:
    """

    all_mods = analysis_utils.get_moderator_names(model)
    X_means = np.array(ro.r("colMeans")(model.rx2("X.f")))
    Xnew = np.tile(X_means, (npoints, 1))
    for i, mod in enumerate(moderator_names):
        mod_idx = all_mods.index(mod)
        Xnew[:, mod_idx] = xs[i, :]
    # handle interaction effects (e.g., "delta_ph:delta_t")
    interaction_mods = [mod for mod in all_mods if ":" in mod]
    for interaction_mod in interaction_mods:
        idx = all_mods.index(interaction_mod)
        Xnew[:, idx] = _generate_interactive_moderator_value(
            all_mods, Xnew, interaction_mod
        )
    return Xnew


def metafor_predict_from_model(
    model: ro.vectors.ListVector,
    moderator_names: list[str],
    xs: np.ndarray,
    confidence_level: int = 95,
    npoints: int = 1000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generalized prediction for one or more moderators.

    Args:
        model (ro.vectors.ListVector): The fitted metafor model.
        moderator_names (list[str]): list of moderator names (strings)
        xs (np.ndarray): np.ndarray of shape (n_points, n_moderators)
        confidence_level (int): confidence level for the prediction intervals e.g. 95 for 95% CI

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]: predicted values, lower confidence bound, upper confidence bound, lower prediction interval, upper prediction interval
    """
    if isinstance(moderator_names, str):
        moderator_names = [moderator_names]
    if len(xs.shape) != 1:
        if xs.shape[0] != len(moderator_names):
            raise ValueError("xs columns must match number of moderator_names")
    xs = np.atleast_2d(xs)

    Xnew = _build_newmods_matrix(model, moderator_names, xs, npoints=npoints)
    # convert to R matrix
    Xnew_r = ro.r.matrix(ro.FloatVector(Xnew.flatten()), nrow=Xnew.shape[0], byrow=True)
    # predict
    predict = ro.r("predict")
    pred_res = predict(model, newmods=Xnew_r, level=(confidence_level / 100))
    pred = np.array(pred_res.rx2("pred"))
    ci_lb = np.array(pred_res.rx2("ci.lb"))
    ci_ub = np.array(pred_res.rx2("ci.ub"))
    pred_lb = np.array(pred_res.rx2("pi.lb"))
    pred_ub = np.array(pred_res.rx2("pi.ub"))
    return pred, ci_lb, ci_ub, pred_lb, pred_ub


def prediction_df_from_model(
    model: ro.vectors.ListVector,
    moderator_names: list[str],
    xs: np.ndarray,
    confidence_level: int = 95,
    npoints: int = 1000,
) -> pd.DataFrame:
    pred, ci_lb, ci_ub, pred_lb, pred_ub = metafor_predict_from_model(
        model, moderator_names, xs, confidence_level, npoints
    )
    return pd.DataFrame(
        {
            "pred": pred,
            "ci.lb": ci_lb,
            "ci.ub": ci_ub,
            "pred.lb": pred_lb,
            "pred.ub": pred_ub,
        }
    )


def predict_nd_surface_from_model(
    model: ro.vectors.ListVector,
    moderator_names: list[str],
    moderator_values: list[np.ndarray],
) -> tuple[np.ndarray, list[np.ndarray]]:
    """
    Generate an n-dimensional prediction surface from a model and moderator values.

    Args:
        model (ro.vectors.ListVector): R model object with .rx2("beta") for coefficients.
        moderator_names (list[str]): list of names (str) for the moderators to vary.
        moderator_values (list[np.ndarray]): list of 1D arrays, each for a moderator.

    Returns:
        pred_surface (np.ndarray): n-dimensional numpy array of predictions.
        meshgrids (list[np.ndarray]): list of meshgrid arrays for each moderator (for plotting).
    """
    # get all moderator names and indices for those to vary
    all_mods = analysis_utils.get_moderator_names(model)
    coefs = np.array(model.rx2("beta"))
    moderator_indices = [all_mods.index(mod) for mod in moderator_names]

    # create meshgrid for moderator values and flatten for vectorized computation
    meshgrids = np.meshgrid(*moderator_values, indexing="ij")
    grid_points = [mg.ravel() for mg in meshgrids]
    n_points = grid_points[0].size
    n_coefs = len(coefs)

    # build the design matrix for prediction
    X = np.zeros((n_points, n_coefs))

    # set mean values for moderators not being varied
    for mod_idx, mod in enumerate(all_mods):
        if mod not in moderator_names:
            X[:, mod_idx] = np.broadcast_to(coefs[mod_idx], n_points)

    # update values for moderators being varied
    for i, mod_idx in enumerate(moderator_indices):
        X[:, mod_idx] = grid_points[i]

    # handle interaction effects (e.g., "delta_ph:delta_t")
    interaction_mods = [mod for mod in all_mods if ":" in mod]
    for interaction_mod in interaction_mods:
        idx = all_mods.index(interaction_mod)
        X[:, idx] = _generate_interactive_moderator_value(all_mods, X, interaction_mod)

    # compute predictions and reshape to n-dimensional grid
    pred = X @ coefs
    pred_surface = pred.reshape(meshgrids[0].shape)
    return pred_surface, meshgrids


def _generate_interactive_moderator_value(
    all_mods: list[str], mod_matrix: np.ndarray, moderator_name: str
) -> np.ndarray:
    """
    Given an interactive moderator name (e.g., "mod1:mod2"), generate the required moderator value
    by multiplying the relevant columns in the moderator matrix.

    Args:
        all_mods (list[str]): List of all moderator names (including interaction terms).
        mod_matrix (np.ndarray): 2D array where each column corresponds to a moderator in all_mods.
        moderator_name (str): The interaction moderator name, e.g., "mod1:mod2".

    Returns:
        np.ndarray: 1D array of the interaction moderator values.

    Raises:
        ValueError: If moderator_name is not a valid two-way interaction.

    N.B. limited to only two moderators in interaction term. Requires moderator matrix columns to correspond to all_mods in order.
    """
    if ":" not in moderator_name:
        raise ValueError(
            f"Moderator name '{moderator_name}' is not an interaction term."
        )
    mod_names = moderator_name.split(":")
    if len(mod_names) != 2:
        raise ValueError(
            f"Interaction term '{moderator_name}' must have exactly two moderators."
        )
    try:
        mod1_idx = all_mods.index(mod_names[0])
        mod2_idx = all_mods.index(mod_names[1])
    except ValueError as e:
        raise ValueError(
            f"One or both moderators in '{moderator_name}' not found in all_mods."
        ) from e
    return mod_matrix[:, mod1_idx] * mod_matrix[:, mod2_idx]


@dataclass
class DredgeConfig:
    """Configuration for dredge analysis."""

    effect_type: str = "st_relative_calcification"
    treatment: list[str] = None
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

    def _prepare_data(self) -> None:
        """Prepare the data for the dredge analysis."""
        self.df = analysis_utils.preprocess_df_for_meta_model(
            self.df,
            self.config.effect_type,
            effect_type_var=None,
            treatment=self.config.treatment,
            formula=self.config.global_formula,
        )

    def _setup_r_environment(self, formula: str) -> None:
        """Set up R environment variables for dredge analysis."""
        os.environ["LC_ALL"] = "en_US.UTF-8"

        # Convert DataFrame to R and assign variables
        from rpy2.robjects import pandas2ri

        with (ro.default_converter + pandas2ri.converter).context():
            ro.r.assign("df_r", pandas2ri.py2rpy(self.df))
        # df_r = pandas2ri.py2rpy(self.df)
        # df_r = ro.pandas2ri.py2rpy(self.df)
        # ro.r.assign("df_r", df_r)
        ro.r.assign("effect_col", self.config.effect_type)
        ro.r.assign("var_col", f"{self.config.effect_type}_var")
        ro.r.assign("original_doi", "original_doi")
        ro.r.assign("ID", "ID")
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
        # ro.r(f"""
        # # Load required libraries
        # library(parallel)
        # library(MuMIn)

        # # Create cluster
        # clu <- makeCluster({self.config.n_cores})

        # # Load packages on each worker
        # clusterEvalQ(clu, {{
        #     library(metafor)
        #     library(MuMIn)
        # }})

        # # Export required variables to each worker
        # clusterExport(clu, varlist = c("df_r", "effect_col", "var_col", "global_formula", "random_formula", "original_doi", "ID"))

        # # Run dredge analysis
        # dredge_result <- dredge(global_model, cluster = clu)

        # # Stop the cluster
        # stopCluster(clu)
        # """)
        # TODO: this wasn't working, so I'm using old code: don't think it does things in parallel

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
        clu <- parallel::makeCluster({self.config.n_cores})
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

            self._prepare_data()
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
            self._prepare_data()
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


# def predict_metafor_model_on_moderator_values(
#     model, moderator_vals: pd.DataFrame
# ) -> pd.DataFrame:
#     """
#     Provide a model with a dataframe (same dimensions as model X) of moderator values to get model predictions.
#     """
#     # convert to R matrix (moderator_vals)
#     moderator_vals_np = np.array(moderator_vals, dtype=float)
#     moderator_vals_r = ro.r.matrix(
#         ro.FloatVector(moderator_vals_np.flatten()),
#         nrow=moderator_vals_np.shape[0],
#         byrow=True,
#     )

#     # predict all at once in R
#     predictions_r = ro.r("predict")(model, newmods=moderator_vals_r, digits=2)

#     r_selected_columns = ro.r(
#         "as.data.frame"
#     )(
#         predictions_r
#     ).rx(
#         True, ro.IntVector([1, 2, 3, 4, 5, 6])
#     )  # select columns to avoid heterogenous shape. TODO: surely this should be dynamic?
#     ro.pandas2ri.activate()

#     # convert the selected columns to a pandas dataframe
#     with (ro.default_converter + ro.pandas2ri.converter).context():
#         return (
#             ro.conversion.get_conversion()
#             .rpy2py(r_selected_columns)
#             .reset_index(drop=True)
#         )


# def process_meta_regplot_data(model, model_comps, x_mod, level, point_size, predlim):
#     """
#     Process data for meta-regression plotting.

#     Args:
#         model (rpy2.robjects.vectors.ListVector): An R rma.mv or rma model object from metafor package.
#         model_comps (tuple): Model components containing predictor and response info.
#         x_mod (str): Name of the moderator variable to plot on x-axis.
#         level (float): Confidence level for intervals in percent.
#         point_size (str or array-like): Point sizes - either "seinv" (inverse of standard error),
#             "vinv" (inverse of variance), or an array of custom sizes.
#         predlim (tuple[float, float], optional): Limits for predicted x-axis values (min, max).

#     Returns:
#         tuple: Containing processed data (xi, yi, vi, norm_weights, xs, pred, ci_lb, ci_ub,
#                 pred_lb, pred_ub, mod_pos)
#     """
#     pandas2ri.activate()
#     mod_pos = get_moderator_index(model_comps, x_mod)
#     yi, vi, xi = _extract_model_components(model, mod_pos)
#     norm_weights = _compute_point_weights(vi, point_size, yi)
#     xs, predlim = _get_xs_and_prediction_limits(xi, predlim)
#     predict_function = _get_predict_function()
#     pred, ci_lb, ci_ub, pred_lb, pred_ub = _metafor_prediction_from_model(
#         model, predict_function, xs, mod_pos, level
#     )
#     return xi, yi, vi, norm_weights, xs, pred, ci_lb, ci_ub, pred_lb, pred_ub, mod_pos


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
#     formula, df = analysis_utils.preprocess_df_for_meta_model(
#         df, effect_type, effect_type_var, treatment, formula, necessary_vars
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


# def _metafor_predict_function() -> ro.r.function:
#     # def _metafor_predict_function(model, r_xs, moderator_index, level) -> ro.r.function:
#     """Return the R function for prediction.

#     Args:
#         model: The fitted metafor model.
#         xs: The values of the moderator to predict on.
#         moderator_index: The index of the moderator in the model.
#         level: The confidence level for the prediction intervals e.g. 95 for 95% CI.

#     Returns:
#         The R function for prediction.
#     """
#     return ro.r("""
#     function(model, r_xs, moderator_index, level) {
#         # Get mean values for all predictors
#         X_means <- colMeans(model$X.f)
#         # Create new data for predictions
#         Xnew <- matrix(rep(X_means, each=length(r_xs)), nrow=length(r_xs))
#         colnames(Xnew) <- colnames(model$X.f)
#         # Set the moderator of interest to the sequence of values
#         Xnew[,moderator_index] <- r_xs
#         # Remove intercept if present in the model
#         if (model$int.incl) {
#             Xnew <- Xnew[,-1, drop=FALSE]
#         }
#         # Make predictions
#         pred <- predict(model, newmods=Xnew, level=(level/100))
#         # Return results
#         return(pred)
#     }
#     """)


# def _metafor_prediction_from_model(
#     model, moderator_name, xs: np.ndarray = None, confidence_level: float = 95
# ):
#     """DEPRECATED: Use metafor_predict_from_model instead."""
#     import warnings

#     warnings.warn(
#         "_metafor_prediction_from_model is deprecated. Use metafor_predict_from_model instead."
#     )
#     # Backward compatibility: only works for a single moderator
#     if xs is None:
#         xs, _, _ = plot_analysis.MetaRegressionResults(
#             model, moderator_name
#         )._get_basic_model_data_for_moderator()
#     moderator_index = plot_analysis.MetaRegressionResults(
#         model, moderator_name
#     ).moderator_index
#     r_xs = ro.FloatVector(xs)
#     predict_function = _metafor_predict_function()
#     try:
#         pred_res = predict_function(
#             model, r_xs, moderator_index + 1, confidence_level
#         )  # R is 1-indexed
#         pred = np.array(pred_res.rx2("pred"))
#         ci_lb = np.array(pred_res.rx2("ci.lb"))
#         ci_ub = np.array(pred_res.rx2("ci.ub"))
#         pred_lb = np.array(pred_res.rx2("pi.lb"))
#         pred_ub = np.array(pred_res.rx2("pi.ub"))
#         return pred, ci_lb, ci_ub, pred_lb, pred_ub
#     except Exception as e:
#         print(f"Error in prediction: {e}")
#         return None, None, None, None, None
