#!/usr/bin/env python3
"""
Unified Metafor Model Implementation

This module provides a streamlit-ready, unified implementation of the metafor
model that combines the best features from both meta_regression.py and
hybrid_metafor_adapter.py.

Key Features:
- Safe rpy2 context management for Streamlit compatibility
- Python-friendly model component extraction
- Comprehensive error handling
- Support for both simple and advanced model operations
- Unified API for prediction and analysis
"""

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import rpy2.robjects as ro

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from calcification.analysis import analysis_utils, meta_regression, analysis  # noqa
from calcification.utils import config  # noqa
from app.infrastructure import RContextManager  # noqa

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class MetaforModel:
    """
    Metafor Model with Streamlit compatibility and comprehensive functionality.

    This class combines the best features from both the original MetaforModel and
    the StreamlitMetaforAdapter, providing a single interface for metafor operations
    with proper context management and error handling.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        effect_type: str = "st_relative_calcification",
        effect_type_var: Optional[str] = None,
        treatment: Optional[str] = None,
        formula: Optional[str] = None,
        random: str = "~ 1 | doi/ID",
        required_columns: Optional[List[str]] = None,
        save_summary: bool = False,
        dvar_threshold: float = None,
        var_threshold: float = None,
        process_data: bool = True,
        verbose: bool = True,
        metafor_model_kwargs: Optional[dict] = None,
        cooks_distance: bool = False,
    ):
        """
        Initialize the unified metafor model.

        Args:
            df: Input DataFrame containing effect size data
            effect_type: Column name for effect sizes
            effect_type_var: Column name for effect size variances (auto-generated if None)
            treatment: Treatment column for filtering
            formula: Model formula (auto-generated if None)
            random: Random effects formula
            required_columns: Additional required columns
            save_summary: Whether to save model summary to file
            dvar_threshold: Threshold for delta variable filtering
            process_data: Whether to preprocess data during initialization
            verbose: Whether to print verbose output
        """
        self.df = df.copy()
        self.original_df = df.copy()
        self.effect_type = effect_type
        self.effect_type_var = effect_type_var or f"{effect_type}_var"
        self.treatment = treatment
        self.random = random
        self.verbose = verbose
        self.save_summary = save_summary
        self.dvar_threshold = dvar_threshold
        self.var_threshold = var_threshold
        self.fitted = False
        self.metafor_model_kwargs = metafor_model_kwargs
        self.cooks_distance = cooks_distance
        # Model storage
        self.r_model = None  # R model object
        self.model_dict = {}  # Python-friendly model data
        self.summary = None

        # Initialize formula and components
        try:
            self.formula = self._get_model_formula() if formula is None else formula
            self.formula_components = self._get_formula_components()
            self.n_params = analysis_utils.get_number_of_params(self.formula_components)

            # Get required columns
            self.required_columns = analysis_utils._get_required_columns(
                self.effect_type,
                self.formula_components,
                self.effect_type_var,
                required_columns,
            )

            # Prepare data if requested
            if process_data:
                self.processed_df = self._prepare_data()
            else:
                self.processed_df = self.df

            # Set up R DataFrame using safe context
            self._setup_r_dataframe()

        except Exception as e:
            logger.error(f"Error during initialization: {e}")
            raise

        if verbose:
            logger.info("Metafor initialized successfully")
            logger.info(f"Formula: {self.formula}")
            logger.info(f"Treatment: {self.treatment}")
            logger.info(f"Data shape: {self.processed_df.shape}")

    def _get_model_formula(self) -> str:
        """Generate model formula using analysis utilities."""
        return analysis_utils.generate_metaregression_formula(
            self.effect_type, self.treatment, include_intercept=False
        )

    def _get_formula_components(self) -> Dict[str, Any]:
        """Parse formula components using analysis utilities."""
        formula_components = analysis_utils.get_formula_components(self.formula)
        self.intercept = formula_components["intercept"]
        return formula_components

    def _prepare_data(self) -> pd.DataFrame:
        """Preprocess DataFrame for model fitting."""
        return analysis_utils.preprocess_df_for_meta_model(
            self.df,
            self.effect_type,
            self.effect_type_var,
            self.treatment,
            self.formula_components,
            self.dvar_threshold,
            self.var_threshold,
            self.verbose,
        )

    def _setup_r_dataframe(self) -> None:
        """Set up R DataFrame using safe context management."""
        with RContextManager() as r_ctx:
            pandas2ri = r_ctx["pandas2ri"]

            # Subset data to required columns
            df_processed = self.processed_df[self.required_columns]

            # Validate factor moderators
            for mod in self.formula_components["factor_terms"]:
                if self.processed_df[mod].nunique() < 2:
                    raise ValueError(
                        f"Factor moderator {mod} has {self.processed_df[mod].nunique()} "
                        f"levels, which is less than 2. This may cause model issues."
                    )

            self.df_processed = df_processed
            self.df_r = pandas2ri.py2rpy(df_processed)

            if self.verbose:
                logger.info(f"R DataFrame created with {len(df_processed)} rows")

    def fit_model(self, formula: Optional[str] = None) -> "MetaforModel":
        """
        Fit the metafor model using safe R context.

        Args:
            formula: Override formula for fitting (uses instance formula if None)

        Returns:
            Self for method chaining
        """

        try:
            with RContextManager() as r_ctx:
                ro = r_ctx["ro"]
                lc = r_ctx["localconverter"]
                p2ri = r_ctx["pandas2ri"]

                metafor = ro.packages.importr("metafor")

                fit_formula = formula or self.formula

                if self.verbose:
                    logger.info(f"Fitting model with formula: {fit_formula}")

                # remove points exceeding the cooks distance threshold

                # Fit the model
                self.r_model = metafor.rma_mv(
                    yi=ro.FloatVector(self.df_r.rx2(self.effect_type)),
                    V=ro.FloatVector(self.df_r.rx2(self.effect_type_var)),
                    data=self.df_r,
                    mods=ro.Formula(fit_formula),
                    random=ro.Formula(self.random),
                    method="REML",
                    # **self.metafor_model_kwargs,
                )
                with lc(ro.default_converter + p2ri.converter):
                    r_df = ro.conversion.py2rpy(self.df_processed)

                if self.cooks_distance:
                    self.cooks_distances = self.calculate_cooks_distance()
                    # remove points exceeding the cooks distance threshold
                    cooks_threshold = analysis.calc_cooks_threshold(
                        self.cooks_distances, nparams=self.n_params
                    )
                    self.cooks_threshold = cooks_threshold
                    self.df_processed = self.df_processed[
                        self.cooks_distances < self.cooks_threshold
                    ]

                    # re-fit model
                    self.r_model = metafor.rma_mv(
                        yi=ro.FloatVector(self.df_r.rx2(self.effect_type)),
                        V=ro.FloatVector(self.df_r.rx2(self.effect_type_var)),
                        data=self.df_r,
                        mods=ro.Formula(fit_formula),
                        random=ro.Formula(self.random),
                        method="REML",
                        # **self.metafor_model_kwargs,
                    )

                # Step 2: Set up global environment with data and call
                ro.globalenv["d"] = r_df
                ro.globalenv["cl"] = self.r_model["call"]
                with lc(ro.default_converter):  # safe generation of summary
                    base = ro.packages.importr("base")
                    r_model_native = ro.r("local({ cl$data <- d; eval(cl) })")
                    self._r_summary = base.summary(r_model_native)

                self.fitted = True

                # extract model components for Python use
                self.extract_model_coefficient_info()
                self._extract_model_components()

                if self.verbose:
                    logger.info("Model fitted successfully")
                    logger.info(f"Components extracted: {list(self.model_dict.keys())}")

                return self

        except Exception as e:
            logger.error(f"Model fitting failed: {e}")
            raise RuntimeError(f"Model fitting failed: {e}")

    def _get_native_r_model(self, r_ctx):
        """Helper method to rebuild the native R model from stored call."""
        ro = r_ctx["ro"]
        lc = r_ctx["localconverter"]
        p2ri = r_ctx["pandas2ri"]

        with lc(ro.default_converter + p2ri.converter):
            r_df = ro.conversion.py2rpy(self.df_processed)

        ro.globalenv["d"] = r_df
        ro.globalenv["cl"] = self.r_model["call"]

        with lc(ro.default_converter):
            return ro.r("local({ cl$data <- d; eval(cl) })")

    def calculate_cooks_distance(self, progbar=True, parallel="multicore", ncpus=24):
        """Calculate Cook's distance for the fitted model using metafor's cooks.distance function."""
        with RContextManager() as r_ctx:
            ro = r_ctx["ro"]
            metafor = ro.packages.importr("metafor")

            # Get native R model
            r_model_native = self._get_native_r_model(r_ctx)

            # Calculate Cook's distance with the native R model
            cooks = metafor.cooks_distance_rma_mv(
                r_model_native, progbar=progbar, parallel=parallel, ncpus=ncpus
            )

            # Convert to numpy array for easier handling in Python
            cooks_array = np.array(cooks)
            return cooks_array

    def cooks_distance_exclusion(self, r_model, df_processed):
        """Remove points exceeding the cooks distance threshold."""
        # use native metafor cooks distance function
        with RContextManager() as r_ctx:
            ro = r_ctx["ro"]
            lc = r_ctx["localconverter"]
            p2ri = r_ctx["pandas2ri"]
            metafor = ro.packages.importr("metafor")

            with lc(ro.default_converter):
                r_df = ro.conversion.py2rpy(df_processed)
                # This would need to be implemented based on your filtering logic
                return metafor.cooks_distance_filter(r_model, df_processed)

    def _extract_model_components(self) -> None:
        """Extract model components into Python-friendly format."""
        try:
            self.model_dict = {}
            # extract coefficient values and their names
            self.extract_model_coefficient_info()

            # Extract key statistics
            for key in [
                "method",
                "k",
                "QE",
                "QEp",
                "QM",
                "QMp",
                "pval",
                "beta",
                "se",
                "zval",
                "ci.lb",
                "ci.ub",
                "fit.stats",
                "sigma2",
            ]:
                if key in self.r_model:
                    try:
                        value = self.r_model[key]
                        if isinstance(value, pd.DataFrame):
                            # Rename common rows for clarity
                            row_names = {"ll": "LogLik", "dev": "Deviance"}
                            value.index = [
                                row_names.get(idx, idx) for idx in value.index
                            ]
                            self.model_dict[key] = value.to_dict()
                        elif hasattr(value, "__iter__") and not isinstance(value, str):
                            if key == "sigma2":
                                for v_i, v in enumerate(value):
                                    self.model_dict[f"{key}.{v_i + 1}"] = v
                            if len(value) == 1:
                                self.model_dict[key] = value[0]
                            else:
                                self.model_dict[key] = list(value)
                        else:
                            self.model_dict[key] = value
                    except Exception as e:
                        logger.warning(f"Error extracting model component {key}: {e}")

            if self.verbose:
                logger.info(
                    f"Extracted model components: {list(self.model_dict.keys())}"
                )

        except Exception as e:
            logger.warning(f"Failed to extract some model components: {e}")

    def extract_model_coefficient_info(self):
        """Extract model coefficients from the OrdDict R model object for Streamlit use."""
        try:
            coefficients = np.array(list(self.r_model["beta"]))
            coeff_names = extract_coefficient_names_from_model(self.r_model)
            result_coeffs = np.full(len(coeff_names), np.nan)
            # determine which columns in the dataframe have all zeros
            zero_cols = set(self.df_processed.columns[self.df_processed.eq(0).all()])
            # this assumes (correctly) that the order matches for non-zero columns
            coeff_idx = 0
            for i, name in enumerate(
                coeff_names
            ):  # check if name is a substring of any of the zero_cols
                if any(zero_col in name for zero_col in zero_cols):
                    result_coeffs[i] = 0
                else:
                    if coeff_idx < len(coefficients):
                        result_coeffs[i] = coefficients[coeff_idx]
                        coeff_idx += 1

            self.coefficients = result_coeffs
            self.dropped_coefficients = list(zero_cols)
            self.coefficient_names = coeff_names

        except Exception as e:
            logger.error(f"Failed to extract model coefficients: {e}")

    def get_coefficients_dataframe(self) -> pd.DataFrame:
        """Get coefficients as a pandas DataFrame for Streamlit display."""
        if not self.fitted:
            raise RuntimeError("Model must be fitted before extracting coefficients.")

        try:
            # Get coefficient values
            coef_values = (
                self.coefficients.flatten()
                if len(self.coefficients.shape) > 1
                else self.coefficients
            )

            for val_type in ["se", "zval", "pval", "ci.lb", "ci.ub"]:
                val_list = self.model_dict.get(val_type, [np.nan] * len(coef_values))
                if isinstance(val_list, (list, np.ndarray)) and len(val_list) != len(
                    coef_values
                ):
                    val_list = [np.nan] * len(coef_values)
                elif not isinstance(val_list, (list, np.ndarray)):
                    val_list = [val_list] * len(coef_values)
                self.model_dict[val_type] = val_list

            # Use actual coefficient names if available
            if hasattr(self, "coefficient_names") and len(
                self.coefficient_names
            ) == len(coef_values):
                row_names = self.coefficient_names
            else:
                row_names = [f"Coef {i + 1}" for i in range(len(coef_values))]

            # Add significance indicators
            p_vals = self.model_dict["pval"]
            significance_stars = [
                "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
                for p in p_vals
            ]

            # Create DataFrame with proper index
            return pd.DataFrame(
                {
                    "Estimate": coef_values,
                    "SE": self.model_dict["se"],
                    "Z-value": self.model_dict["zval"],
                    "P-value": self.model_dict["pval"],
                    "CI Lower": self.model_dict["ci.lb"],
                    "CI Upper": self.model_dict["ci.ub"],
                    "Significance symbols": significance_stars,
                    "Significant": [
                        p < 0.05 if not np.isnan(p) else False
                        for p in self.model_dict["pval"]
                    ],
                },
                index=row_names,
            )

        except Exception as e:
            print(f"⚠️ Error creating coefficients DataFrame: {e}")
            return pd.DataFrame({"Error": [str(e)]})

    def get_heterogeneity_dataframe(self) -> pd.DataFrame:
        """Get dataframe description of heterogeneity"""
        # get all keys in dict with 'sigma' in the key

        sigma_dict = {k: v for k, v in self.model_dict.items() if "sigma2." in k}
        data = {
            "QE": self.model_dict["QE"],
            "QEp": self.model_dict["QEp"],
            "QM": self.model_dict["QM"],
            "QMp": self.model_dict["QMp"],
        }
        data.update(sigma_dict)
        return pd.DataFrame([data])

    def get_model_summary_text(self) -> str:
        """Generate a summary text for Streamlit display."""
        stat_desc_strs = {
            "k": "Number of samples",
            "QE": "Residual heterogeneity",
            "QM": "Model test statistic",
            "LogLik": "Log-likelihood",
            "AIC": "AIC (Akaike Information Criterion, smaller is better)",
            "AICc": "AICc (AIC corrected for small sample size)",
            "BIC": "BIC (Bayesian Information Criterion, smaller is better)",
            "method": "Model fitting method",
        }
        if not self.fitted:
            return "Model not fitted yet."

        try:
            # format model summary
            lines = []
            lines.append("Metafor Model Summary")
            lines.append("=" * (len(self.formula) + len("Formula: ")))
            lines.append(f"Formula: {self.formula}")
            lines.append(f"Random Effects: {self.random}")

            # get fit type
            fit_type = self.model_dict.get("method", "REML")[0]
            # add basic statistics
            for key in stat_desc_strs:
                val = self._summarize_model_data(key, fit_type=fit_type)
                if isinstance(val, str):
                    pass
                elif isinstance(val, (int, float)):
                    if key == "QE" or key == "QM":
                        val = f"{val:.3f} (p: {self.model_dict.get(f'{key}p', [np.nan])[0]:.1e})"
                    elif val is not None:
                        val = f"{val:.3f}"
                lines.append(f"{stat_desc_strs[key]}: {val}")

            # Add coefficients table
            # if "beta" in self.model_dict:
            lines.append("\nCoefficients:")
            coef_table = self._format_coefficients_table()
            lines.append(coef_table)

            return "\n".join(lines)

        except Exception as e:
            return f"Error generating summary: {e}"

    def _format_coefficients_table(self) -> str:
        """Format coefficients as a nice table for text display."""
        try:
            coef_values = self.coefficients.flatten()
            n_coef = len(coef_values)

            # Use actual coefficient names if available
            if (
                hasattr(self, "coefficient_names")
                and len(self.coefficient_names) == n_coef
            ):
                row_names = self.coefficient_names
                # drop any coefficient names that are in the dropped_coefficients list
                row_names = [
                    name for name in row_names if name not in self.dropped_coefficients
                ]
            else:
                row_names = [f"Coef {i + 1}" for i in range(n_coef)]

            # Extract all statistics
            stats = {}
            # stats["Estimate"] = [f"{float(coef):.4f}" for coef in coef_values]

            for key, label in [
                ("beta", "Estimate"),
                ("se", "SE"),
                ("zval", "Z-val"),
                ("pval", "P-value"),
                ("ci.lb", "CI Lower"),
                ("ci.ub", "CI Upper"),
            ]:
                values = self.model_dict.get(key, [np.nan] * n_coef)
                if key == "pval":
                    # Format p-values in scientific notation
                    stats[label] = [
                        f"{float(val):.2e}" if not np.isnan(val) else "N/A"
                        for val in values[:n_coef]
                    ]
                else:
                    # Format other values to 4 decimal places
                    stats[label] = [
                        f"{float(val):.4f}" if not np.isnan(val) else "N/A"
                        for val in values[:n_coef]
                    ]

            # Add significance indicators
            p_vals = self.model_dict.get("pval", [np.nan] * n_coef)
            stats["Sig"] = [
                "***"
                if p < 0.001
                else "**"
                if p < 0.01
                else "*"
                if p < 0.05
                else "."
                if p < 0.1
                else ""
                for p in p_vals[:n_coef]
            ]

            # Create the table
            display_df = pd.DataFrame(stats, index=row_names)

            # Format as string with proper spacing
            table_str = display_df.to_string(
                float_format=lambda x: f"{x:.4f}"
                if isinstance(x, (int, float))
                else str(x),
                justify="right",
            )

            # Add significance legend
            legend = "\nSignificance: *** p<0.001, ** p<0.01, * p<0.05, . p<0.1"
            # specify any dropped coefficients (value zero)
            if self.dropped_coefficients:
                legend += (
                    "\n\nThe following coefficients since all values were zero: "
                    + ", ".join(self.dropped_coefficients)
                )

            return table_str + legend

        except Exception as e:
            return f"Error formatting coefficients table: {e}"

    def _summarize_model_data(self, key: str, fit_type: str = "REML") -> dict:
        if key in ["LogLik", "Deviance", "AIC", "AICc", "BIC"]:
            val = self.model_dict.get("fit.stats", {})[fit_type].get(key, None)
        else:
            val = self.model_dict.get(key, None)

        if isinstance(val, (list, np.ndarray)):
            if isinstance(val[0], (int, float)):
                val = float(val[0]) if len(val) > 0 else "N/A"
            else:
                val = str(val[0]) if len(val) > 0 else "N/A"
        return val

    def _generate_interaction_values(
        self, all_mods: List[str], mod_matrix: np.ndarray, interaction_mod: str
    ) -> np.ndarray:
        """Generate interaction term values."""
        if ":" not in interaction_mod:
            raise ValueError(f"'{interaction_mod}' is not an interaction term")

        mod_names = interaction_mod.split(":")
        if len(mod_names) != 2:
            raise ValueError(f"Only two-way interactions supported: {interaction_mod}")

        try:
            mod1_idx = all_mods.index(mod_names[0])
            mod2_idx = all_mods.index(mod_names[1])
            return mod_matrix[:, mod1_idx] * mod_matrix[:, mod2_idx]
        except ValueError as e:
            raise ValueError(f"Moderators in '{interaction_mod}' not found") from e

    def get_model_metadata(self) -> Dict[str, Any]:
        """Get comprehensive model metadata."""
        return {
            "formula": self.formula,
            "formula_components": self.formula_components,
            "treatment": self.treatment,
            "effect_type": self.effect_type,
            "random_effects": self.random,
            "fitted": self.fitted,
            "n_observations": len(self.processed_df)
            if hasattr(self, "processed_df")
            else 0,
            "n_coefficients": len(self.coefficients) if self.fitted else 0,
            "coefficient_names": self.coefficient_names if self.fitted else [],
            "model_components": list(self.model_dict.keys()) if self.fitted else [],
        }

    def _save_summary(self, summary_fp: Optional[str] = None) -> None:
        """Save model summary to file."""
        summary_fp = summary_fp or (config.results_dir / "unified_metafor_summary.txt")

        try:
            with open(summary_fp, "w") as f:
                f.write(self.get_model_summary_text())
            logger.info(f"Summary saved to {summary_fp}")
        except Exception as e:
            logger.error(f"Failed to save summary: {e}")

    def __repr__(self) -> str:
        """String representation of the model."""
        status = "fitted" if self.fitted else "not fitted"
        return (
            f"MetaforModel(effect_type='{self.effect_type}', "
            f"formula='{self.formula}', status='{status}')"
        )

    def view_model_summary(self) -> None:
        """Get model summary text."""
        # Use the native R summary attribute if available
        if hasattr(self, "_r_summary") and self._r_summary is not None:
            print(self._r_summary)
        else:
            print("No summary available.")

    # --- prediction ---
    def predict_on_moderator(
        self,
        moderator_names: str | list[str],
        confidence_level: float = 95,
        n_points: int = 100,
    ) -> pd.DataFrame:
        """Predict using native metafor object.

        moderator_names: Name(s) of moderator variables
        xs: Range of values over which to predict. N.B. if the moderator has multiple values (e.g. for delta_t, I(delta_t^2)) of shape (n_points, n_moderators)
        confidence_level: Confidence level for intervals

        Returns:
            DataFrame with predictions and confidence intervals
        """
        if not self.fitted:
            raise RuntimeError("Model must be fitted before prediction.")

        if isinstance(moderator_names, str):
            moderator_names = [moderator_names]

        xs = self.get_2d_xs_for_prediction(moderator_names[0], n_points=n_points)

        return predict_with_metafor(self, xs, confidence_level=confidence_level)

    def get_2d_xs_for_prediction(
        self, moderator_name: str, n_points: int = 100
    ) -> np.ndarray:
        """Generate matrix of predictor values with correct values for multiple terms with the moderator, mean values for the rest.

        Args:
            moderator_name: Name of the moderator variable
            n_points: Number of prediction points (the number of points for the moderator)

        Returns:
            np.ndarray of shape (n_points, n_moderators)
        """
        if moderator_name not in self.coefficient_names:
            raise ValueError(f"Moderator '{moderator_name}' not found in model")

        xs = np.linspace(
            np.min(self.df_processed[moderator_name].values),
            np.max(self.df_processed[moderator_name].values),
            n_points,
        )
        # create matrix of mean values for all moderators (n_samples, n_moderators)
        X_means = np.zeros((n_points, len(self.coefficient_names)))
        # Assign mean values from X.f to non-zero coefficients, zero to zero coefficients, skipping intercept
        X_f_means = np.mean(np.array(self.r_model["X.f"]), axis=0)
        non_zero_coeffs_indices = np.where(self.coefficients != 0)[0]
        for i in non_zero_coeffs_indices:
            X_means[:, i] = X_f_means[i]
        Xnew = X_means
        # select moderator of interest (in all its forms: linear, interaction, nonlinear)
        mod_idx = self.coefficient_names.index(moderator_name)
        Xnew[:, mod_idx] = xs
        # interaction
        interaction_mods = [mod for mod in self.coefficient_names if ":" in mod]
        for interaction_mod in interaction_mods:
            idx = self.coefficient_names.index(interaction_mod)
            Xnew[:, idx] = meta_regression.generate_interactive_moderator_value(
                self.coefficient_names, Xnew, interaction_mod
            )
        # nonlinear - FIXED: Handle polynomial terms properly
        nonlinear_mods = [mod for mod in self.coefficient_names if "I(" in mod]
        for nonlinear_mod in nonlinear_mods:
            if moderator_name in nonlinear_mod:
                idx = self.coefficient_names.index(nonlinear_mod)
                # Extract power from expressions like "I(delta_t^2)"
                if "^" in nonlinear_mod:
                    power_str = nonlinear_mod.split("^")[-1].replace(")", "")
                    try:
                        power = float(power_str)
                        Xnew[:, idx] = xs**power
                        if self.verbose:
                            print(
                                f"   🔢 Set polynomial term {nonlinear_mod} = {moderator_name}^{power}"
                            )
                    except ValueError:
                        print(f"⚠️ Could not parse power from {nonlinear_mod}")
                        Xnew[:, idx] = xs  # fallback to linear
                else:
                    # Handle other I() expressions
                    Xnew[:, idx] = xs

        if self.verbose:
            print(
                f"   📊 Final Xnew shape: {Xnew.shape}, coefficients: {len(self.coefficient_names)}"
            )

        return np.atleast_2d(Xnew)

    def predict_nd_surface_from_model(
        self,
        moderator_names: list[str],
        moderator_values: list[np.ndarray],
        confidence_level: float = 0.95,
    ) -> dict[str, np.ndarray]:
        """
        Generate n-dimensional prediction surfaces with uncertainty estimates.

        Args:
            moderator_names (list[str]): list of names (str) for the moderators to vary.
            moderator_values (list[np.ndarray]): list of 1D arrays, each for a moderator.
            include_se (bool): Whether to calculate standard error surface.
            include_ci (bool): Whether to calculate confidence interval surfaces.
            include_pi (bool): Whether to calculate prediction interval surfaces.
            confidence_level (float): Confidence level for intervals (default 0.95).

        Returns:
            dict: Dictionary containing:
                - 'pred': prediction surface
                - 'se': standard error surface (if include_se=True)
                - 'ci_lb', 'ci_ub': confidence interval surfaces (if include_ci=True)
                - 'pi_lb', 'pi_ub': prediction interval surfaces (if include_pi=True)
                - 'meshgrids': list of meshgrid arrays for plotting
        """
        # get all moderator names and indices for those to vary
        all_mods = self.coefficient_names
        coefs = self.coefficients
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
            X[:, idx] = generate_interactive_moderator_value(
                all_mods, X, interaction_mod
            )

        # compute predictions
        pred = X @ coefs
        pred_surface = pred.reshape(meshgrids[0].shape)

        # Initialize results dictionary
        results = {"pred": pred_surface}
        # Calculate standard errors
        se = self._calculate_prediction_se(X)
        se_surface = se.reshape(meshgrids[0].shape)

        results["se"] = se_surface

        # potential t-distribution taking into account degrees of freedom (n-k). Compare with Normal distribution. Pretty much identical.

        # calculate confidence intervals using t-distribution
        alpha = 1 - confidence_level
        # degrees of freedom: n - k (number of observations - number of coefficients)
        n_obs = getattr(self, "n_obs", None)
        if n_obs is None:
            # Try to infer from model or data
            try:
                self.model_dict.get("k", [100])[0]
                # n_obs = self.r_model.rx2("k")[
                #     0
                # ]  # metafor stores k as number of studies
            except Exception:
                n_obs = X.shape[0]
        n_coefs = len(self.coefficients)
        df = max(n_obs - n_coefs, 1)
        from scipy.stats import t as scipy_t

        t_score = scipy_t.ppf(1 - alpha / 2, df)

        ci_lb = pred - t_score * se
        ci_ub = pred + t_score * se

        results["ci_lb"] = ci_lb.reshape(meshgrids[0].shape)
        results["ci_ub"] = ci_ub.reshape(meshgrids[0].shape)

        # # Calculate prediction intervals (still using normal distribution for PI, unless t is desired)
        pi_se = self._calculate_prediction_interval_se(X)
        alpha = 1 - confidence_level
        # For prediction intervals, also use t-distribution for consistency
        pi_t_score = scipy_t.ppf(1 - alpha / 2, df)

        pi_lb = pred - pi_t_score * pi_se
        pi_ub = pred + pi_t_score * pi_se

        results["pi_lb"] = pi_lb.reshape(meshgrids[0].shape)
        results["pi_ub"] = pi_ub.reshape(meshgrids[0].shape)

        # # calculate confidence intervals
        # alpha = 1 - confidence_level
        # z_score = scipy_norm.ppf(1 - alpha / 2)  # two-tailed

        # ci_lb = pred - z_score * se
        # ci_ub = pred + z_score * se

        # results["ci_lb"] = ci_lb.reshape(meshgrids[0].shape)
        # results["ci_ub"] = ci_ub.reshape(meshgrids[0].shape)

        # # # Calculate prediction intervals
        # pi_se = self._calculate_prediction_interval_se(X)
        # alpha = 1 - confidence_level
        # z_score = scipy_norm.ppf(1 - alpha / 2)

        # pi_lb = pred - z_score * pi_se
        # pi_ub = pred + z_score * pi_se

        # results["pi_lb"] = pi_lb.reshape(meshgrids[0].shape)
        # results["pi_ub"] = pi_ub.reshape(meshgrids[0].shape)

        return results, meshgrids

    def _calculate_prediction_se(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate standard errors for predictions using the variance-covariance matrix.

        Standard error for prediction: SE = sqrt(X * V * X^T)
        where V is the variance-covariance matrix of coefficients.
        """
        try:
            # Get variance-covariance matrix from the model
            vb = np.array(self.r_model["vb"])  # metafor stores this as "vb"

            # Calculate standard errors: SE = sqrt(diag(X * V * X^T))
            # For vectorized computation: se_i = sqrt(sum_jk(X_ij * V_jk * X_ik))
            se_squared = np.sum(X * (X @ vb), axis=1)
            se = np.sqrt(se_squared)

            return se

        except Exception as e:
            logger.warning(f"Could not calculate standard errors: {e}")
            # Fallback: use coefficient standard errors as rough approximation
            coef_se = np.array(self.model_dict.get("se", [0] * len(self.coefficients)))
            # Simple approximation: SE ≈ sqrt(sum((X * coef_se)^2))
            se_approx = np.sqrt(np.sum((X * coef_se) ** 2, axis=1))
            return se_approx

    def _calculate_prediction_interval_se(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate standard errors for prediction intervals.

        Prediction intervals account for both coefficient uncertainty AND residual variance.
        PI_SE = sqrt(SE_pred^2 + sigma^2)
        """
        # Get prediction standard error
        pred_se = self._calculate_prediction_se(X)

        try:
            # Get residual variance from the model
            # In metafor, this might be stored as sigma2 or similar
            if "sigma2" in self.r_model:
                sigma2 = float(self.r_model["sigma2"][0])
            else:
                # Fallback: estimate from QE (residual heterogeneity)
                QE = self.model_dict.get("QE", [1.0])[0]
                df_resid = self.model_dict.get("k", [100])[0] - len(self.coefficients)
                sigma2 = QE / max(df_resid, 1)

            # Prediction interval SE includes both sources of uncertainty
            pi_se = np.sqrt(pred_se**2 + sigma2)

            return pi_se

        except Exception as e:
            logger.warning(f"Could not calculate prediction interval SE: {e}")
            # Fallback: inflate prediction SE by factor of 1.5
            return pred_se * 1.5

    def get_model_data_for_plotting(self, moderator_name: str) -> None:
        """Extract data needed for plotting against 'moderator_name' from the model."""
        # merge dataframes by index to necessary columns are present
        combined_df = self.df_processed.merge(self.df)
        if moderator_name in combined_df.columns:
            self.xi = combined_df[moderator_name].values
            self.yi = combined_df[self.effect_type].values
            self.vi = combined_df[self.effect_type_var].values


# --- Helpers ---


def generate_interactive_moderator_value(
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


def extract_coefficient_names_from_model(model: ro.vectors.ListVector) -> list[str]:
    """Extract coefficient names from the model formula and structure."""
    try:
        # Method 1: get from call attribute via context handler
        with RContextManager() as r_ctx:
            ro = r_ctx["ro"]
            ro.globalenv["cl"] = model["call"]
            labels = ro.r(
                "local({"
                "  t <- terms(cl$mods); "
                '  labs <- attr(t, "term.labels"); '
                '  if (isTRUE(attr(t, "intercept") == 1L)) c("(Intercept)", labs) else labs'
                "})"
            )
            return list(labels)
    except Exception as e:
        logger.error(f"Failed to extract coefficient names: {e}")
        return []


def predict_with_metafor(
    model: "MetaforModel",
    xs: np.ndarray,
    confidence_level: float = 95,
) -> pd.DataFrame:
    """Internal prediction method using safe R context."""
    with RContextManager() as r_ctx:
        ro = r_ctx["ro"]
        lc = r_ctx["localconverter"]
        p2ri = r_ctx["pandas2ri"]

        # Step 1: Convert DataFrame using pandas2ri (safe for DataFrames)
        with lc(ro.default_converter + p2ri.converter):
            r_df = ro.conversion.py2rpy(model.df_processed)

        # Step 2: Set up global environment with data and call
        ro.globalenv["d"] = r_df
        ro.globalenv["cl"] = model.r_model["call"]

        # Step 3: Rebuild model using default_converter (preserves ListVector)
        with lc(ro.default_converter):  # no pandas2ri here!
            # Rebuild the R model from the stored call
            r_model_native = ro.r("local({ cl$data <- d; eval(cl) })")

            # Build prediction matrix - FIXED: Handle multi-dimensional xs properly
            if xs.ndim == 1:
                # Single moderator - need to reshape to 2D
                x_values_2d = xs.reshape(-1, 1)
            else:
                # Multi-moderator case (e.g., polynomial: delta_t + I(delta_t^2))
                x_values_2d = xs
            # drop any columns of x_values_2d that are all zeros
            x_values_2d = x_values_2d[:, np.any(x_values_2d != 0, axis=0)]

            Xnew_r = ro.r.matrix(
                ro.FloatVector(x_values_2d.flatten()),
                nrow=x_values_2d.shape[0],
                ncol=x_values_2d.shape[1],
                byrow=True,
            )

            # Use R predict function with native ListVector model
            pred_res = ro.r("predict")(
                r_model_native, newmods=Xnew_r, level=(confidence_level / 100)
            )

            return {
                "pred": np.array(pred_res.rx2("pred")),
                "se": np.array(pred_res.rx2("se")),
                "ci_lb": np.array(pred_res.rx2("ci.lb")),
                "ci_ub": np.array(pred_res.rx2("ci.ub")),
                "pred_lb": np.array(pred_res.rx2("pi.lb")),
                "pred_ub": np.array(pred_res.rx2("pi.ub")),
            }


# --- Deprecated functions ---


# # Convenience functions for backwards compatibility
# def create_metafor_model(
#     df: pd.DataFrame, effect_type: str = "st_relative_calcification", **kwargs
# ) -> MetaforModel:
#     """Create and return a MetaforModel instance."""
#     return MetaforModel(df=df, effect_type=effect_type, **kwargs)


# def fit_metafor_model(
#     df: pd.DataFrame, effect_type: str = "st_relative_calcification", **kwargs
# ) -> MetaforModel:
#     """Create, fit, and return a MetaforModel instance."""
#     model = MetaforModel(df=df, effect_type=effect_type, **kwargs)
#     return model.fit_model()

# def get_prediction_range(
#     self, moderator_name: str, extend_factor: float = 0.1, n_points: int = 100
# ) -> np.ndarray:
#     """
#     Get a suitable range of x values for prediction plotting.

#     Args:
#         moderator_name: Name of the moderator variable
#         extend_factor: Factor to extend range beyond data (0.1 = 10% extension)
#         n_points: Number of prediction points

#     Returns:
#         Array of x values for prediction
#     """
#     if moderator_name not in self.data.columns:
#         raise ValueError(f"Moderator '{moderator_name}' not found in data")

#     values = self.data[moderator_name].dropna()
#     x_min, x_max = values.min(), values.max()
#     x_range = x_max - x_min

#     extended_min = x_min - extend_factor * x_range
#     extended_max = x_max + extend_factor * x_range

#     return np.linspace(extended_min, extended_max, n_points)
