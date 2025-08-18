#!/usr/bin/env python3
"""
Streamlit-compatible meta-regression plotter.
Adapts the existing plotting functionality to work with the hybrid adapter's OrdDict objects.
"""

import sys
from pathlib import Path

import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from app.hybrid_metafor_adapter import streamlit_safe_r_context
from calcification.analysis import meta_regression
from calcification.plotting import plot_config

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


# TODO: investigate ratio of largest to smallest sampling variance extremely large (could just be showing that it works with Hedge's G and results look similar)
# TODO: better outlier excluding (huge st_relative_calcification values)


class StreamlitMetaRegressionPlotter:
    """
    Streamlit-compatible meta-regression plotter that works with the hybrid adapter.

    This adapts the existing plotting functionality to work with OrdDict model objects
    instead of ro.vectors.ListVector objects.
    """

    def __init__(
        self,
        fitted_adapter,
        moderator_name: str,
        colorby: str = None,
        debug: bool = False,
    ):
        """Initialize with a fitted StreamlitMetaforAdapter."""
        self.adapter = fitted_adapter
        self.moderator_name = moderator_name
        self.colorby = colorby
        self.debug = debug

        if not self.adapter.fitted:
            raise ValueError("Adapter must be fitted before plotting")

        # Extract plotting data
        self._extract_plotting_data()

        # Initialize partial residuals as None (calculated on demand)
        self.partial_residuals = None
        self.partial_residuals_x = None
        self.ord_residuals = None
        self.fit_val = None  # Initialize to prevent AttributeError

        if self.debug:
            print(f"🐛 DEBUG: Initialized plotter for moderator '{moderator_name}'")
            print(f"   Available data points: {len(self.xi)}")
            print(f"   Effect type: {self.effect_type}")
            print(f"   Formula: {self.adapter.formula}")

    def _extract_plotting_data(self):
        """Extract data needed for plotting from the adapter."""
        try:
            # Get the subsetted data and model components
            self.df = getattr(self.adapter, "df_subset", self.adapter.processed_df)
            self.original_df = getattr(self.adapter, "df_original", self.adapter.df)
            self.model_dict = self.adapter.model_dict

            # Extract effect sizes and moderator values
            self.effect_type = self.adapter.effect_type
            self.effect_var_type = self.adapter.effect_type_var

            # Get the actual data points
            if self.moderator_name in self.df.columns:
                self.xi = self.df[self.moderator_name].values
                self.yi = self.df[self.effect_type].values
                self.vi = self.df[self.effect_var_type].values
                self.seinv = 1 / np.sqrt(
                    self.vi
                )  # Inverse standard error for point sizing

                # Get values for hover text
                self.dois = self.df.get("original_doi", ["Unknown"] * len(self.xi))
                self.st_control_calcification = self.df.get(
                    "st_control_calcification", ["Unknown"] * len(self.xi)
                )
                self.st_treatment_calcification = self.df.get(
                    "st_treatment_calcification", ["Unknown"] * len(self.xi)
                )

                # Create prediction range
                self.x_min, self.x_max = np.min(self.xi), np.max(self.xi)
                self.x_range = self.x_max - self.x_min
                self.xs = np.linspace(
                    self.x_min - 0.1 * self.x_range,
                    self.x_max + 0.1 * self.x_range,
                    100,
                )

                # Generate predictions (simplified - using the coefficient for now)
                self._generate_predictions()

            else:
                raise ValueError(f"Moderator '{self.moderator_name}' not found in data")

        except Exception as e:
            st.error(f"Error extracting plotting data: {e}")
            raise

    def _generate_predictions(self):
        """Generate predictions using proper metafor prediction methods."""
        try:
            # Use proper metafor prediction if possible
            if hasattr(self.adapter, "r_model") and self.adapter.r_model is not None:
                self._generate_metafor_predictions()
            # else:
            # Fallback to simplified predictions
            # self._generate_simple_predictions()

        except Exception as e:
            print(f"Warning: Prediction generation failed, using fallback: {e}")
        #     self._generate_fallback_predictions()

    def _generate_metafor_predictions(self):
        """Generate predictions using the original metafor prediction functions."""
        try:
            with streamlit_safe_r_context() as r_ctx:
                r_ctx["ro"]

                # Convert the OrdDict r_model back to an R object temporarily
                # This is a bit hacky but necessary for the prediction functions

                # Get the required data for prediction
                xi, yi, vi = self._extract_model_components_for_prediction()
                # TODO: get actual formula for prediction regression

                # Generate prediction x values (similar to original)
                xs, _ = meta_regression._get_xs_and_prediction_limits(
                    xi, prediction_limits=None, num_prediction_points=100
                )

                # For now, try a direct approach using metafor's predict function
                # Create a temporary R model object
                if hasattr(self.adapter.r_model, "rx2"):
                    # It's still an R object
                    r_model = self.adapter.r_model
                else:
                    # It's an OrdDict - we need to work with what we have
                    # Extract predictions using the coefficient-based approach
                    self._generate_coefficient_based_predictions()
                    return

                # Use the metafor prediction function
                pred, se, ci_lb, ci_ub, pred_lb, pred_ub = (
                    meta_regression.metafor_predict_from_model(
                        r_model, [self.moderator_name], xs, confidence_level=95
                    )
                )

                # Store results
                self.xs = xs.flatten() if hasattr(xs, "flatten") else xs
                self.pred = pred.flatten() if hasattr(pred, "flatten") else pred
                self.ci_lb = ci_lb.flatten() if hasattr(ci_lb, "flatten") else ci_lb
                self.ci_ub = ci_ub.flatten() if hasattr(ci_ub, "flatten") else ci_ub
                self.pred_lb = (
                    pred_lb.flatten() if hasattr(pred_lb, "flatten") else pred_lb
                )
                self.pred_ub = (
                    pred_ub.flatten() if hasattr(pred_ub, "flatten") else pred_ub
                )

                print("✅ Generated predictions using metafor predict function")

        except Exception as e:
            print(f"⚠️ Metafor prediction failed: {e}")
            self._generate_coefficient_based_predictions()

    def _extract_model_components_for_prediction(self):
        """Extract xi, yi, vi from the adapter for prediction."""
        try:
            # Get data from the adapter
            if self.moderator_name in self.df.columns:
                xi = self.df[self.moderator_name].values.reshape(-1, 1)
                yi = self.df[self.effect_type].values
                vi = self.df[self.effect_var_type].values
                return xi, yi, vi
            else:
                raise ValueError(f"Moderator {self.moderator_name} not found in data")
        except Exception as e:
            print(f"⚠️ Could not extract model components: {e}")
            # Fallback values
            xi = np.array([[0, 1]]).T
            yi = np.array([0, 1])
            vi = np.array([1, 1])
            return xi, yi, vi

    def _generate_hybrid_predictions(self):
        """Generate predictions using coefficient-based logic to understand formula structure,
        then use metafor_predict_from_model for proper statistical predictions."""

        try:
            # First, use coefficient-based logic to understand the formula structure
            coef_names = getattr(self.adapter, "coefficient_names", [])
            coefficients = self.adapter.coefficients.flatten()

            if len(coef_names) != len(coefficients):
                print(
                    f"⚠️ Coefficient mismatch: {len(coef_names)} names vs {len(coefficients)} values"
                )
                self._generate_coefficient_based_predictions()
                return

            # Parse the formula to understand what terms are present
            # formula_rhs = self.adapter.formula.split("~")[1].strip()

            # Identify all variables in the model (excluding the plotting moderator)
            model_variables = set()
            for coef_name in coef_names:
                if coef_name == "(Intercept)":
                    continue

                # Extract variable name from coefficient name
                var_name = self._extract_variable_name_from_coef(coef_name)
                if var_name:
                    model_variables.add(var_name)

            # Remove the plotting moderator from the list of variables to set to mean values
            model_variables.discard(self.moderator_name)

            print(f"�� Model variables: {model_variables}")
            print(f"�� Plotting moderator: {self.moderator_name}")

            # Now use metafor_predict_from_model with proper matrix construction
            if hasattr(self.adapter, "r_model") and self.adapter.r_model is not None:
                self._generate_metafor_predictions_with_matrix()
            else:
                print(
                    "⚠️ No R model available, falling back to coefficient-based predictions"
                )
                self._generate_coefficient_based_predictions()

        except Exception as e:
            print(f"⚠️ Hybrid prediction failed: {e}")
            import traceback

            traceback.print_exc()
            self._generate_coefficient_based_predictions()

    def _extract_variable_name_from_coef(self, coef_name: str) -> str:
        """Extract the base variable name from a coefficient name."""
        if "I(" in coef_name and "^2" in coef_name:
            # Extract variable name from I(var^2)
            return coef_name.split("(")[1].split("^")[0]
        elif "factor(" in coef_name:
            # Extract variable name from factor(var)level
            return coef_name.split("(")[1].split(")")[0]
        else:
            # Simple variable name
            return coef_name

    def _generate_metafor_predictions_with_matrix(self):
        """Generate predictions using metafor_predict_from_model with proper matrix construction."""

        try:
            with streamlit_safe_r_context() as r_ctx:
                ro = r_ctx["ro"]

                # Get the R model
                r_model = self.adapter.r_model

                # Get all moderator names from the model
                all_mods = self.adapter._extract_coefficient_names_from_model()
                print(f"📊 All model moderators: {all_mods}")

                # Check if our plotting moderator is in the model
                if self.moderator_name not in all_mods:
                    print(
                        f"⚠️ Plotting moderator '{self.moderator_name}' not found in model moderators"
                    )
                    self._generate_coefficient_based_predictions()
                    return

                # Get the column means from the model's X matrix
                X_means = np.mean(np.array(r_model["X.f"]), axis=0)
                print(f"�� X matrix means: {X_means}")

                # Create prediction matrix
                npoints = len(self.xs)
                Xnew = np.tile(X_means, (npoints, 1))

                # Set the plotting moderator to our x values
                mod_idx = all_mods.index(self.moderator_name)
                Xnew[:, mod_idx] = self.xs

                # Handle interaction effects
                interaction_mods = [mod for mod in all_mods if ":" in mod]
                for interaction_mod in interaction_mods:
                    idx = all_mods.index(interaction_mod)
                    Xnew[:, idx] = (
                        meta_regression._generate_interactive_moderator_value(
                            all_mods, Xnew, interaction_mod
                        )
                    )
                nonlinear_mods = [mod for mod in all_mods if "I(" in mod]
                for nonlinear_mod in nonlinear_mods:
                    if self.moderator_name in nonlinear_mod:
                        idx = all_mods.index(nonlinear_mod)
                        power = int(nonlinear_mod.split("^")[-1].replace(")", ""))
                        Xnew[:, idx] = self.xs**power

                print(f"✅ Created prediction matrix: {Xnew.shape}")
                print(f"✅ Prediction matrix: {Xnew[0, :]}")

                # Convert to R matrix
                Xnew_r = ro.r.matrix(
                    ro.FloatVector(Xnew.flatten()), nrow=Xnew.shape[0], byrow=True
                )

                (
                    self.pred,
                    self.se,
                    self.ci_lb,
                    self.ci_ub,
                    self.pred_lb,
                    self.pred_ub,
                ) = self.predict_with_metafor(all_mods, Xnew_r, confidence_level=95)

                print(
                    "✅ Generated predictions using metafor with proper matrix construction"
                )
                print(
                    f"   Prediction range: {self.pred.min():.3f} to {self.pred.max():.3f}"
                )
                print(f"   CI range: {self.ci_lb.min():.3f} to {self.ci_ub.max():.3f}")
                print(
                    f"   PI range: {self.pred_lb.min():.3f} to {self.pred_ub.max():.3f}"
                )

        except Exception as e:
            print(f"⚠️ Metafor prediction with matrix failed: {e}")
            import traceback

            traceback.print_exc()
            self._generate_coefficient_based_predictions()

    def predict_with_metafor(self, moderator_names, xs, confidence_level=95):
        # xs: np.ndarray (n_points, len(moderator_names))
        with streamlit_safe_r_context() as r_ctx:
            ro = r_ctx["ro"]
            lc = r_ctx["localconverter"]
            p2ri = r_ctx["pandas2ri"]

            # 1) Prepare data in R using pandas converter (safe for DataFrame only)
            with lc(ro.default_converter + p2ri.converter):
                r_df = ro.conversion.py2rpy(self.df)
            # 2) Rebuild native R model WITHOUT pandas2ri (avoid OrdDict conversion)
            ro.globalenv["d"] = r_df
            ro.globalenv["cl"] = self.adapter.r_model["call"]
            with lc(ro.default_converter):
                r_model_native = ro.r(
                    "local({ cl$data <- d; eval(cl) })"
                )  # ListVector (native R)

                # Build newmods in R
                xs = np.atleast_2d(xs)
                Xnew_r = ro.r.matrix(
                    ro.FloatVector(xs.flatten()), nrow=xs.shape[0], byrow=True
                )
                print(xs[0, :])

                # Use metafor's predict natively
                pred_res = ro.r("predict")(
                    r_model_native, newmods=Xnew_r, level=(confidence_level / 100)
                )

                pred = np.array(pred_res.rx2("pred"))
                se = np.array(pred_res.rx2("se"))
                ci_lb = np.array(pred_res.rx2("ci.lb"))
                ci_ub = np.array(pred_res.rx2("ci.ub"))
                pi_lb = np.array(pred_res.rx2("pi.lb"))
                pi_ub = np.array(pred_res.rx2("pi.ub"))

            return pred, se, ci_lb, ci_ub, pi_lb, pi_ub

    def _generate_predictions(self):
        """Generate predictions using the hybrid approach."""
        try:
            # Use the new hybrid approach
            self._generate_hybrid_predictions()
        except Exception as e:
            print(f"Warning: Hybrid prediction generation failed, using fallback: {e}")
            self._generate_coefficient_based_predictions()

    def _generate_coefficient_based_predictions(self):
        """Generate predictions using coefficients from the model dict for complex formulas."""
        try:
            # Get coefficients and names
            coef_names = getattr(self.adapter, "coefficient_names", [])
            coefficients = self.adapter.coefficients.flatten()

            if len(coef_names) != len(coefficients):
                print(
                    f"⚠️ Coefficient mismatch: {len(coef_names)} names vs {len(coefficients)} values"
                )
                # self._generate_fallback_predictions()
                # return

            # Initialize prediction array
            self.pred = np.zeros_like(self.xs)

            # # Parse the formula to understand what terms are present
            # formula_rhs = self.adapter.formula.split("~")[1].strip()

            # # Check for intercept
            # has_intercept = "(Intercept)" in coef_names

            # Process each coefficient based on the formula
            for i, (coef_name, coef_value) in enumerate(zip(coef_names, coefficients)):
                coef_value = float(coef_value)

                if coef_name == "(Intercept)":
                    # Add intercept to all predictions
                    self.pred += coef_value

                elif self.moderator_name in coef_name:
                    # This coefficient is for our plotting moderator
                    if "I(" in coef_name and "^2" in coef_name:
                        # Quadratic term: I(moderator^2)
                        self.pred += coef_value * (self.xs**2)
                    elif "factor(" in coef_name:
                        # Factor term - this is more complex, would need factor levels
                        # For now, treat as linear (this is a limitation)
                        print(
                            f"⚠️ Factor term for plotting moderator not fully supported: {coef_name}"
                        )
                        self.pred += coef_value * self.xs
                    else:
                        # Linear term
                        self.pred += coef_value * self.xs

                else:
                    # This coefficient is for a different variable
                    # We need to set it to a fixed value for prediction
                    # Extract variable name from coefficient name
                    var_name = coef_name
                    if "I(" in coef_name and "^2" in coef_name:
                        # Extract variable name from I(var^2)
                        var_name = coef_name.split("(")[1].split("^")[0]
                    elif "factor(" in coef_name:
                        # Extract variable name from factor(var)level
                        var_name = coef_name.split("(")[1].split(")")[0]

                    # Set to mean value if variable exists in data and is numeric
                    if var_name in self.df.columns:
                        # Try to convert to numeric, ignore errors (non-numeric columns become NaN)
                        col_numeric = pd.to_numeric(self.df[var_name], errors="coerce")
                        if col_numeric.notna().any():
                            mean_val = float(col_numeric.mean())
                            if "I(" in coef_name and "^2" in coef_name:
                                # Quadratic term for other variable
                                self.pred += coef_value * (mean_val**2)
                            elif "factor(" in coef_name:
                                # Factor term - use reference level (0)
                                self.pred += coef_value * 0  # Reference level
                            else:
                                # Linear term for other variable
                                self.pred += coef_value * mean_val
                            print(
                                f"ℹ️ Set '{var_name}' to mean value ({mean_val:.3f}) for prediction"
                            )
                        else:
                            print(
                                f"⚠️ Variable '{var_name}' in data is non-numeric, ignoring coefficient"
                            )
                    else:
                        print(
                            f"⚠️ Variable '{var_name}' not found in data, ignoring coefficient"
                        )

            # Generate confidence intervals
            # Use a simple approach based on model fit
            se_values = self.model_dict.get("se", [1.0])
            if len(se_values) > 0:
                # Use average SE as a simple approach
                avg_se = np.mean([float(se) for se in se_values if not np.isnan(se)])
                margin = 1.96 * avg_se
                self.ci_lb = self.pred - margin
                self.ci_ub = self.pred + margin
            else:
                margin = 1.0
                self.ci_lb = self.pred - margin
                self.ci_ub = self.pred + margin

            # Prediction intervals (wider than confidence intervals)
            self.pred_lb = self.ci_lb - 0.5
            self.pred_ub = self.ci_ub + 0.5

            print("✅ Generated coefficient-based predictions for complex formula")
            print(f"   Formula: {self.adapter.formula}")
            print(f"   Coefficients used: {len(coef_names)}")
            print(
                f"   Prediction range: {self.pred.min():.3f} to {self.pred.max():.3f}"
            )

        except Exception as e:
            print(f"⚠️ Coefficient-based prediction failed: {e}")
            import traceback

            traceback.print_exc()
            # self._generate_fallback_predictions()

    def _get_moderator_coefficient(self):
        """Extract the coefficient for the current moderator."""
        try:
            # Get coefficient names and find the one for our moderator
            coef_names = getattr(self.adapter, "coefficient_names", [])
            coefficients = self.adapter.coefficients.flatten()

            # Look for the moderator in coefficient names
            for i, name in enumerate(coef_names):
                if self.moderator_name in name and "Intercept" not in name:
                    return float(coefficients[i])

            # Fallback: if no intercept, first coefficient; otherwise, second
            if "(Intercept)" in coef_names:
                return float(coefficients[1]) if len(coefficients) > 1 else 0.0
            else:
                return float(coefficients[0]) if len(coefficients) > 0 else 0.0

        except Exception as e:
            print(f"⚠️ Could not get moderator coefficient: {e}")
            return 0.0

    def _get_intercept_coefficient(self):
        """Extract the intercept coefficient."""
        try:
            coef_names = getattr(self.adapter, "coefficient_names", [])
            coefficients = self.adapter.coefficients.flatten()

            # Look for intercept
            for i, name in enumerate(coef_names):
                if "Intercept" in name:
                    return float(coefficients[i])

            # If no explicit intercept found and model has intercept
            if "- 1" not in self.adapter.formula and "-1" not in self.adapter.formula:
                return float(coefficients[0]) if len(coefficients) > 0 else 0.0
            else:
                return 0.0  # No intercept model

        except Exception as e:
            print(f"⚠️ Could not get intercept coefficient: {e}")
            return 0.0

    # def _generate_simple_predictions(self):
    #     """Simple linear prediction (legacy method)."""
    #     try:
    #         if len(self.adapter.coefficients) > 0:
    #             coef = float(self.adapter.coefficients.item(0))
    #             self.pred = coef * (self.xs - np.mean(self.xi))

    #             se = (
    #                 self.model_dict.get("se", [1.0])[0]
    #                 if "se" in self.model_dict
    #                 else 1.0
    #             )
    #             margin = 1.96 * float(se)
    #             self.ci_lb = self.pred - margin
    #             self.ci_ub = self.pred + margin
    #             self.pred_lb = self.ci_lb - 0.5
    #             self.pred_ub = self.ci_ub + 0.5
    #         else:
    #             self._generate_fallback_predictions()

    #     except Exception as e:
    #         print(f"⚠️ Simple prediction failed: {e}")
    #         self._generate_fallback_predictions()

    # def _generate_fallback_predictions(self):
    #     """Ultimate fallback: flat line predictions."""
    #     self.pred = np.zeros_like(self.xs)
    #     self.ci_lb = self.pred - 1
    #     self.ci_ub = self.pred + 1
    #     self.pred_lb = self.pred - 2
    #     self.pred_ub = self.pred + 2
    #     print("⚠️ Using fallback flat line predictions")

    def _get_core_grouping_colours(self) -> list[str]:
        """Get colors for core_grouping using standard color scheme."""

        core_grouping_values = self.df.core_grouping
        core_grouping_colours = plot_config.CG_COLOURS

        # Convert seaborn colors to hex strings for Plotly
        colors = []
        for val in core_grouping_values:
            if val in core_grouping_colours:
                # Convert seaborn color (RGB tuple) to hex string
                color = core_grouping_colours[val]
                if isinstance(color, tuple):
                    # Convert RGB tuple to hex
                    hex_color = mcolors.rgb2hex(color)
                    colors.append(hex_color)
                else:
                    # Already a string, use as is
                    colors.append(str(color))
            else:
                # Fallback color for unknown values
                colors.append("#808080")  # Gray

        return colors

    def _calculate_partial_residuals(self):
        """Calculate partial residuals leveraging existing model fitting infrastructure.

        Partial residuals show the relationship between a moderator and the outcome
        after accounting for all other variables in the model.

        Formula: partial_residuals = residuals(reduced_model) + β_moderator × moderator_values
        """
        if self.debug:
            print(
                f"\n🐛 DEBUG: Starting partial residuals calculation for '{self.moderator_name}'"
            )

        try:
            # Use existing coefficient names if available
            coef_names = getattr(self.adapter, "coefficient_names", [])
            coefficients = self.adapter.coefficients.flatten()

            if self.debug:
                print("🐛 DEBUG: Model coefficients:")
                for name, coef in zip(coef_names, coefficients):
                    marker = "👉" if self.moderator_name in name else "  "
                    print(f"   {marker} {name}: {coef:.4f}")

            if len(coef_names) != len(coefficients):
                raise ValueError("Coefficient names and values mismatch")

            # Find ALL coefficients related to the moderator of interest
            moderator_coefficients = []
            moderator_terms = []

            for i, name in enumerate(coef_names):
                # Check if this coefficient involves the moderator
                if self._coefficient_involves_moderator(name, self.moderator_name):
                    moderator_coefficients.append(float(coefficients[i]))
                    moderator_terms.append(name)
                    if self.debug:
                        print(
                            f"   Found moderator term: {name} = {coefficients[i]:.4f}"
                        )

            if not moderator_coefficients:
                raise ValueError(
                    f"Could not find any coefficients for moderator '{self.moderator_name}'"
                )

            if self.debug:
                print(f"   Total moderator terms found: {len(moderator_coefficients)}")
                for term, coef in zip(moderator_terms, moderator_coefficients):
                    print(f"     {term}: {coef:.4f}")

            # Create a reduced formula by removing the moderator term
            reduced_formula = self._create_reduced_formula()

            if self.debug:
                print("🐛 DEBUG: Formula reduction:")
                print(f"   Original: {self.adapter.formula}")
                print(f"   Reduced:  {reduced_formula}")

            # Create a new adapter instance for the reduced model
            reduced_adapter = self._fit_reduced_model(reduced_formula)

            if reduced_adapter is None:
                raise ValueError("Failed to fit reduced model")

            # extract ordinary residuals from the REDUCED model (not the full model!)
            ord_residuals = self._extract_residuals_from_adapter(reduced_adapter)
            self.ord_residuals = ord_residuals
            if ord_residuals is None:
                raise ValueError("Failed to extract residuals from reduced model")

            # Calculate partial residuals with multiple moderator terms
            # For complex formulas like delta_t + I(delta_t^2), we need to sum the contributions
            moderator_contribution = self._calculate_moderator_contribution(
                moderator_terms, moderator_coefficients, self.xi
            )
            # calculate the partial residuals using the CORRECT formula:
            # partial_residuals = residuals_from_reduced_model + beta_moderator * moderator_values
            # This shows what the relationship would look like if ONLY the moderator of interest affected the outcome
            self.partial_residuals = ord_residuals + moderator_contribution
            self.partial_residuals_x = self.xi

            if self.debug:
                print("🐛 DEBUG: Partial residuals calculation:")
                print(f"   Moderator terms: {moderator_terms}")
                print(f"   Moderator coefficients: {moderator_coefficients}")
                print(
                    f"   Moderator contribution range: {np.min(moderator_contribution):.3f} to {np.max(moderator_contribution):.3f}"
                )
                print(
                    f"   Reduced residuals range: {np.min(ord_residuals):.3f} to {np.max(ord_residuals):.3f}"
                )
                print(
                    f"   Moderator values range: {np.min(self.xi):.3f} to {np.max(self.xi):.3f}"
                )
                print(
                    f"   Partial residuals range: {np.min(self.partial_residuals):.3f} to {np.max(self.partial_residuals):.3f}"
                )
                print(
                    f"   Original data range: {np.min(self.yi):.3f} to {np.max(self.yi):.3f}"
                )

            if self.adapter.verbose:
                print(f"✅ Calculated partial residuals for {self.moderator_name}")
                print(f"   Moderator terms: {len(moderator_terms)} terms found")
                for term, coef in zip(moderator_terms, moderator_coefficients):
                    print(f"     {term}: {coef:.4f}")
                print(
                    f"   Partial residuals range: {np.min(self.partial_residuals):.3f} to {np.max(self.partial_residuals):.3f}"
                )
                print(
                    f"   Original data range: {np.min(self.yi):.3f} to {np.max(self.yi):.3f}"
                )

        except Exception as e:
            print(f"⚠️ Failed to calculate partial residuals: {e}")
            # Fallback: extract regular residuals from the current model
            regular_residuals = self._extract_residuals_from_adapter(self.adapter)
            if regular_residuals is not None:
                self.partial_residuals = regular_residuals
                self.partial_residuals_x = self.xi
                print("⚠️ Using regular residuals instead of partial residuals")
            else:
                print("⚠️ Could not calculate any residuals")
                self.partial_residuals = None
                self.partial_residuals_x = None

    def _coefficient_involves_moderator(
        self, coef_name: str, moderator_name: str
    ) -> bool:
        """Check if a coefficient name involves the specified moderator."""
        # Direct match
        if coef_name == moderator_name:
            return True

        # Check for transformations like I(delta_t^2), I(delta_t + delta_ph), etc.
        if coef_name.startswith("I(") and moderator_name in coef_name:
            return True

        # Check for interaction terms like delta_t:delta_ph
        if ":" in coef_name and moderator_name in coef_name.split(":"):
            return True

        # Check for factor transformations like factor(delta_t)level
        if coef_name.startswith(f"factor({moderator_name})"):
            return True

        # check for polynomial terms
        if coef_name.startswith("poly(") and moderator_name in coef_name:
            return True

        # Check for spline terms like bs(delta_t)1, ns(delta_t)2
        if any(
            coef_name.startswith(f"{spline}({moderator_name})")
            for spline in ["bs", "ns", "s", "rcs"]
        ):
            return True

        return False

    def _calculate_moderator_contribution(
        self,
        moderator_terms: list,
        moderator_coefficients: list,
        moderator_values: np.ndarray,
    ) -> np.ndarray:
        """Calculate the total contribution of all moderator terms to the predictions."""
        total_contribution = np.zeros_like(moderator_values)

        for term, coef in zip(moderator_terms, moderator_coefficients):
            # Calculate the transformed values for this term
            transformed_values = self._evaluate_term(term, moderator_values)

            # Add this term's contribution
            contribution = coef * transformed_values
            total_contribution += contribution

            if self.debug:
                print(
                    f"     Term '{term}': coef={coef:.4f}, contribution range={np.min(contribution):.3f} to {np.max(contribution):.3f}"
                )

        self.fit_val = total_contribution
        return total_contribution

    def _evaluate_term(self, term: str, moderator_values: np.ndarray) -> np.ndarray:
        """Evaluate a formula term given moderator values."""
        # Handle different types of transformations

        # Simple linear term
        if term == self.moderator_name:
            return moderator_values

        # Identity transformations like I(delta_t^2), I(delta_t^3), etc.
        if term.startswith("I(") and term.endswith(")"):
            expression = term[2:-1]  # Remove I( and )

            # Handle powers: delta_t^2, delta_t^3, etc.
            if "^" in expression:
                base, power = expression.split("^")
                if base.strip() == self.moderator_name:
                    return moderator_values ** float(power.strip())

        # Factor levels - return indicator variables (would need actual factor levels)
        if term.startswith(f"factor({self.moderator_name})"):
            # This is complex - would need to know the factor levels
            # For now, return the original values (not ideal)
            return moderator_values

        # Default fallback
        return moderator_values

    def _create_reduced_formula(self) -> str:
        """Create a reduced formula by removing the moderator of interest."""
        formula_str = str(self.adapter.formula)
        if "~" not in formula_str:
            raise ValueError("Invalid formula format")

        lhs, rhs = formula_str.split("~", 1)
        terms = [t.strip() for t in rhs.split("+")]

        # Remove terms that contain the plotting moderator
        reduced_terms = []
        for term in terms:
            # Skip terms that are the moderator itself or transformations of it
            if self.moderator_name not in term and not (
                term.startswith("I(") and self.moderator_name in term
            ):
                reduced_terms.append(term)

        if reduced_terms:
            return f"{lhs.strip()} ~ {' + '.join(reduced_terms)}"
        else:
            return f"{lhs.strip()} ~ 1"  # Intercept only

    def _fit_reduced_model(self, reduced_formula: str):
        """Fit a reduced model using the existing adapter infrastructure."""
        try:
            from app.hybrid_metafor_adapter import StreamlitMetaforAdapter

            # Create new adapter with the same configuration but different formula
            reduced_adapter = StreamlitMetaforAdapter(
                df=self.adapter.df.copy(),
                effect_type=self.adapter.effect_type,
                effect_type_var=self.adapter.effect_type_var,
                treatment=self.adapter.treatment,
                formula=reduced_formula,
                random=self.adapter.random,
                required_columns=self.adapter.required_columns,
                process_data=False,  # necessary since otherwise sometimes cooks removes newfound outliers. TODO: further explore via sensitivity analysis
                verbose=False,  # Suppress verbose output for reduced model
            )

            # Fit the reduced model
            reduced_adapter.fit_model()

            return reduced_adapter

        except Exception as e:
            print(f"⚠️ Failed to fit reduced model: {e}")
            return None

    def _extract_residuals_from_adapter(self, adapter):
        """Extract residuals from an adapter using the existing R context handling."""
        try:
            from app.hybrid_metafor_adapter import streamlit_safe_r_context

            with streamlit_safe_r_context() as r_ctx:
                ro = r_ctx["ro"]
                localconverter = r_ctx["localconverter"]

                # Use the existing R model stored in the adapter
                ro.globalenv["cl"] = adapter.r_model["call"]
                ro.globalenv["d"] = adapter.df_r

                with localconverter(ro.default_converter):
                    # Reconstruct model and extract residuals
                    ro.r("local({ cl$data <- d; eval(cl) })")
                    # Use the variable name actually returned by the R code above ("r_model" is only defined in Python, not in R globalenv)
                    # The R code above: r_model <- local({ cl$data <- d; eval(cl) })
                    # So, assign the result to r_model in R globalenv before calling residuals
                    ro.r("r_model <- local({ cl$data <- d; eval(cl) })")
                    residuals = np.array(ro.r("residuals(r_model)"))

                    return residuals

        except Exception as e:
            print(f"⚠️ Failed to extract residuals: {e}")
            return None

    def _get_y_limits(self):
        """Get the y limits of the plot."""
        min_yi, max_yi = np.min(self.yi), np.max(self.yi)

        # if partial residuals available, make sure these are within the y limits
        if self.partial_residuals is not None:
            min_partial, max_partial = (
                np.min(self.partial_residuals),
                np.max(self.partial_residuals),
            )
            min_yi, max_yi = min(min_partial, min_yi), max(max_partial, max_yi)

        # TODO: frame within prediction interval (if available)

        range_y = max_yi - min_yi
        return min_yi - range_y * 0.1, max_yi + range_y * 0.1

    def _get_x_limits(self):
        """Get the x limits of the plot."""
        min_xi, max_xi = np.min(self.xi), np.max(self.xi)
        range_x = max_xi - min_xi
        return min_xi - range_x * 0.1, max_xi + range_x * 0.1

    def _get_colorscale_for_continuous_color_values(self):
        """Get the color values for continuous variables, dependent on the moderator of interest."""
        if self.colorby == "delta_t":
            return "Reds"
        elif self.colorby == "delta_ph":
            return "Reds_r"
        else:
            return "Viridis"

    def create_plotly_figure(
        self,
        title: str = None,
        width: int = 800,
        height: int = 600,
        show_partial_residuals: bool = False,
        custom_y_limits: tuple = None,
        custom_x_limits: tuple = None,
    ) -> go.Figure:
        """Create an interactive Plotly figure with prediction and confidence intervals."""

        fig = go.Figure()

        # Handle coloring based on variable type
        is_numeric_color = False
        if self.colorby == "core_grouping":
            # Categorical variable - create discrete legend
            scatter_points_colours = self._get_core_grouping_colours()
            self._add_discrete_color_legend(fig, scatter_points_colours)
        elif self.colorby:
            # Try to find the color variable in processed df first, then original df
            color_values = None
            if self.colorby in self.df.columns:
                color_values = self.df[self.colorby]
            elif (
                hasattr(self, "original_df")
                and self.colorby in self.original_df.columns
            ):
                # Get values from original df but only for the rows that exist in processed df
                # Match by index to ensure alignment
                color_values = self.original_df.loc[self.df.index, self.colorby]

            if color_values is not None:
                if pd.api.types.is_numeric_dtype(color_values):
                    scatter_points_colours = color_values
                    is_numeric_color = True
                else:
                    # Non-numeric, treat as categorical
                    labels = color_values.astype(str)
                    uniq = list(pd.unique(labels))
                    palette = px.colors.qualitative.Set3
                    n = len(palette)
                    label_to_color = {val: palette[i % n] for i, val in enumerate(uniq)}
                    scatter_points_colours = [label_to_color[val] for val in labels]
                    self._add_discrete_color_legend(fig, labels)
            else:
                scatter_points_colours = "white"
        else:
            scatter_points_colours = "white"

        # Add prediction interval (wider, lighter)
        if (
            hasattr(self, "pred_lb")
            and hasattr(self, "pred_ub")
            and not show_partial_residuals
        ):
            fig.add_trace(
                go.Scatter(
                    x=np.concatenate([self.xs, self.xs[::-1]]),
                    y=np.concatenate([self.pred_ub, self.pred_lb[::-1]]),
                    fill="toself",
                    fillcolor="rgba(173, 216, 230, 0.4)",
                    line=dict(color="rgba(255,255,255,0)"),
                    name="95% Prediction Interval",
                    showlegend=True,
                    hoverinfo="skip",
                )
            )

        # Add confidence interval (narrower, darker)
        fig.add_trace(
            go.Scatter(
                x=np.concatenate([self.xs, self.xs[::-1]]),
                y=np.concatenate([self.ci_ub, self.ci_lb[::-1]]),
                fill="toself",
                fillcolor="rgba(173, 216, 230, 0.4)",
                line=dict(color="rgba(255,255,255,0)"),
                name="95% Confidence Interval",
                showlegend=True,
                hoverinfo="skip",
            )
        ) if not show_partial_residuals else None

        # Add regression line
        fig.add_trace(
            go.Scatter(
                x=self.xs,
                y=self.pred,
                mode="lines",
                line=dict(color="blue", width=3),
                name=f"Meta-regression: {self.effect_type} ~ {self.moderator_name}",
                showlegend=True,
            )
        ) if not show_partial_residuals else None

        # Calculate partial residuals if requested
        if show_partial_residuals and self.partial_residuals is None:
            self._calculate_partial_residuals()

        # Use custom limits if provided, otherwise use automatic calculation

        if custom_y_limits is not None:
            y_min, y_max = custom_y_limits
        else:
            y_min, y_max = self._get_y_limits()

        # Add data points or partial residuals
        if show_partial_residuals and self.partial_residuals is not None:
            # Show partial residuals
            max_seinv = np.max(self.seinv) if np.max(self.seinv) > 0 else 1
            point_sizes = (self.seinv / max_seinv) * 40 + 8

            # TODO: have both delta_t and delta_ph in the hover text (replacing self.moderator_name call)
            # Ensure fit_val is available for hover text
            fit_vals = (
                self.fit_val if self.fit_val is not None else [0.0] * len(self.yi)
            )
            ord_residuals = (
                self.ord_residuals
                if self.ord_residuals is not None
                else [0.0] * len(self.yi)
            )

            hover_text = [
                f"<b>DOI:</b> {doi}<br><b>{self.moderator_name}:</b> {x:.3f}</br><b>Fit value:</b> {fit_val:.3f}</br><b>Ord Residual:</b> {ordr:.3f}<br><b>Partial Residual:</b> {pr:.3f}<br><b>Original {self.effect_type}:</b> {y:.3f}<br><b>Core Grouping:</b> {cg}<br>"
                for doi, x, fit_val, ordr, pr, y, cg in zip(
                    self.dois,
                    self.partial_residuals_x,
                    fit_vals,
                    ord_residuals,
                    self.partial_residuals,
                    self.yi,
                    self.df["core_grouping"],
                )
            ]

            fig.add_trace(
                go.Scatter(
                    x=self.partial_residuals_x,
                    y=self.partial_residuals,
                    mode="markers",
                    marker=dict(
                        size=point_sizes,
                        color=scatter_points_colours,
                        line=dict(color="navy", width=2),
                        opacity=0.8,
                        colorscale=self._get_colorscale_for_continuous_color_values()
                        if is_numeric_color
                        else None,
                        showscale=is_numeric_color,
                        colorbar=dict(
                            title=self._format_axis_label(self.colorby),
                            x=1.15,
                            len=0.8,
                            thickness=20,
                            outlinewidth=1,
                            outlinecolor="black",
                        )
                        if is_numeric_color
                        else None,
                    ),
                    name="Partial Residuals (size ∝ precision)",
                    text=hover_text,
                    hovertemplate="%{text}<extra></extra>",
                    showlegend=True,
                )
            )
        else:
            # Show regular data points
            max_seinv = np.max(self.seinv) if np.max(self.seinv) > 0 else 1
            point_sizes = (
                self.seinv / max_seinv
            ) * 40 + 8  # Scale to 8-48 pixels (larger range)

            # TODO: find out why core_Grouping is unavailable for hover_Text but not coloring; why DOI, temp, phtot, delta_ph fail in colouring
            merged_data = self.df.merge(self.original_df, how="left")

            hover_text = [
                f"<b>DOI:</b> {doi}<br><b>{self.moderator_name}:</b> {xs:.3f}<br><b>{self.effect_type}:</b> {ys:.3f}<br><b>Control Calcification:</b> {st_control_calc:.3f}<br><b>Treatment Calcification:</b> {st_treatment_calc:.3f}<br><b>Core Grouping:</b> {cg}"
                for doi, xs, ys, st_control_calc, st_treatment_calc, cg in zip(
                    self.dois,
                    self.xi,
                    self.yi,
                    merged_data["st_control_calcification"],
                    merged_data["st_treatment_calcification"],
                    merged_data["core_grouping"],
                )
            ]

            fig.add_trace(
                go.Scatter(
                    x=self.xi,
                    y=self.yi,
                    mode="markers",
                    marker=dict(
                        size=point_sizes,
                        color=scatter_points_colours,
                        line=dict(color="navy", width=2),
                        opacity=0.8,
                        colorscale=self._get_colorscale_for_continuous_color_values()
                        if is_numeric_color
                        else None,
                        showscale=is_numeric_color,
                        colorbar=dict(
                            title=self._format_axis_label(self.colorby),
                            x=1.15,
                            len=0.8,
                            thickness=20,
                            outlinewidth=1,
                            outlinecolor="black",
                        )
                        if is_numeric_color
                        else None,
                    ),
                    name="Samples (size ∝ precision)",
                    text=hover_text,
                    hovertemplate="%{text}<extra></extra>",
                    showlegend=True,
                )
            )
        # Add reference line at y=0 (zero effect level)
        fig.add_hline(
            y=0,
            line_dash="dash",
            line_color="gray",
            opacity=0.7,
            annotation_text="Zero effect level"
            if not show_partial_residuals
            else "Zero partial residual",
            annotation_position="top right",
        )

        # Update layout (similar to original plotter style)
        fig.update_layout(
            title=title or f"Meta-regression: {self.adapter.formula}",
            xaxis_title=self._format_axis_label(self.moderator_name),
            yaxis_title=self._format_axis_label(self.effect_type)
            if not show_partial_residuals
            else "Partial Residuals",
            width=width,
            height=height,
            template="plotly_white",
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                bgcolor="rgba(255, 255, 255, 0.8)",
                bordercolor="rgba(0, 0, 0, 0.2)",
                borderwidth=1,
            ),
            font=dict(size=12),
            title_font=dict(size=16),
            yaxis_range=[y_min, y_max],
        )

        return fig

    def _add_discrete_color_legend(self, fig, colors):
        """Add discrete color legend for categorical variables."""

        def _normalize_color(c):
            # Keep valid Plotly color strings as-is; convert tuples/lists to hex
            if isinstance(c, (tuple, list)):
                arr = np.array(c, dtype=float)
                if arr.max() > 1.0:
                    arr = arr / 255.0
                return mcolors.to_hex(arr[:3])
            return str(c)

        values = [v for v in self.df[self.colorby].dropna().unique()]

        if self.colorby == "core_grouping":
            color_map = plot_config.CG_COLOURS
            for value in values:
                if value in color_map:
                    color = _normalize_color(color_map[value])
                    fig.add_trace(
                        go.Scatter(
                            x=[None],
                            y=[None],
                            mode="markers",
                            marker=dict(
                                size=10,
                                color=color,
                                line=dict(color="navy", width=1),
                            ),
                            name=str(value),
                            showlegend=True,
                            legendgroup="color_legend",
                        )
                    )
        else:
            palette = px.colors.qualitative.Set3
            n = len(palette)
            for i, value in enumerate(values):
                color = _normalize_color(palette[i % n])
                fig.add_trace(
                    go.Scatter(
                        x=[None],
                        y=[None],
                        mode="markers",
                        marker=dict(
                            size=10,
                            color=color,
                            line=dict(color="navy", width=1),
                        ),
                        name=str(value),
                        showlegend=True if len(values) < 5 else False,
                        legendgroup="color_legend",
                    )
                )

    def _format_axis_label(self, label: str) -> str:
        """Format axis labels for better display."""
        # Convert from snake_case to Title Case
        formatted = label.replace("_", " ").title()

        # Special formatting for common terms
        replacements = {
            "St Relative Calcification": "% change in calcification rate",
            "Hedges G": "Hedges' g (Effect Size)",
            "Delta T": "ΔT (°C)",
            "Delta Ph": "ΔpH (units)",
            "Delta Omega": "Δ Omega",
        }

        for old, new in replacements.items():
            if old in formatted:
                formatted = new
                break

        return formatted

    def create_summary_stats(self) -> dict:
        """Create summary statistics for the plot."""
        try:
            n_samples = len(self.xi)

            # Extract model statistics
            qe = (
                self.model_dict.get("QE", [None])[0]
                if "QE" in self.model_dict
                else None
            )
            qm = (
                self.model_dict.get("QM", [None])[0]
                if "QM" in self.model_dict
                else None
            )

            # # Calculate R-squared equivalent (tau-squared reduction)
            # tau2_reduction = "Not available"  # Would need more complex calculation

            stats = {
                "Number of samples": n_samples,
                "Moderator": self.moderator_name,
                "Formula": self.adapter.formula,
                "Residual heterogeneity (QE)": f"{float(qe):.3f}"
                if qe is not None
                else "N/A",
                "Model test statistic (QM)": f"{float(qm):.3f}"
                if qm is not None
                else "N/A",
                "Effect size range": f"{np.min(self.yi):.3f} to {np.max(self.yi):.3f}",
                "Moderator range": f"{np.min(self.xi):.3f} to {np.max(self.xi):.3f}",
            }

            return stats

        except Exception as e:
            return {"Error": f"Could not generate statistics: {e}"}


def create_plotting_interface(fitted_adapter):
    """Create the Streamlit plotting interface."""
    # Get available moderators (numeric columns)
    # Use original dataframe to get all available moderators
    priority_moderators = ["delta_t", "delta_ph", "temp", "phtot"]
    available_moderators = []

    # Use original_df if available, otherwise use df
    source_df = getattr(fitted_adapter, "original_df", fitted_adapter.df)

    for col in source_df.columns:
        if source_df[col].dtype in ["float64", "int64", "float32", "int32"]:
            # Skip effect size and variance columns, but allow all other numeric columns
            if col not in [fitted_adapter.effect_type, fitted_adapter.effect_type_var]:
                # Exclude some obviously non-moderator columns
                if col not in ["ID", "latitude", "longitude", "cooks_d"]:
                    available_moderators.append(col)

    # Sort moderators with priority ones first
    def sort_key(col):
        if col in priority_moderators:
            return (0, priority_moderators.index(col))
        else:
            return (1, col)

    available_moderators.sort(key=sort_key)

    if not available_moderators:
        st.warning("No numeric moderators available for plotting.")
        return

    # Set default moderator to delta_t if available, otherwise first available
    default_moderator = (
        "delta_t" if "delta_t" in available_moderators else available_moderators[0]
    )
    try:
        default_index = available_moderators.index(default_moderator)
    except ValueError:
        default_index = 0

    # Moderator selection
    col1, col2 = st.columns([2, 1])

    with col1:
        selected_moderator = st.selectbox(
            "Select moderator for X-axis:",
            available_moderators,
            index=default_index,
            help="Choose a numeric variable to plot against the effect size. Common options: delta_t (temperature), delta_ph (pH)",
        )
        # Get available color variables from both processed and original dataframes
        color_options = ["core_grouping"]  # Always available

        # Add variables from processed dataframe
        processed_columns = set(fitted_adapter.processed_df.columns)

        # Add variables from original dataframe if available
        original_columns = set()
        if hasattr(fitted_adapter, "original_df"):
            original_columns = set(fitted_adapter.original_df.columns)

        # Combine all available columns
        all_columns = processed_columns.union(original_columns)
        # exclude columns from colour options
        excluded_cols = {
            fitted_adapter.effect_type,
            fitted_adapter.effect_type_var,
            "ID",
            "cooks_d",
            "leverage",
            "residuals",
            "fitted",
            "calcification_unit",
            "doi",
            "treatment",
            "dvar_phtot",
            "dvar_temp",
            "st_control_calcification",
            "st_treatment_calcification",
        }

        # Add interesting variables in priority order
        priority_vars = [
            "delta_ph",
            "delta_t",
            "temp",
            "phtot",
            "DOI",
            "treatment",
            "family",
            "original_doi",
        ]
        for var in priority_vars:
            if var in all_columns and var not in excluded_cols:
                color_options.append(var)

        # Add remaining numeric and categorical variables
        for col in sorted(all_columns):
            if col not in color_options and col not in excluded_cols:
                # Check if it's a reasonable variable to color by
                if not col.startswith("_") and not col.endswith("_var"):
                    color_options.append(col)

        colorby_value = st.selectbox(
            "Colour by:",
            color_options,
            index=0,
            help="Choose a variable to color the points by.",
        )

        show_partial_residuals = st.checkbox(
            "Show partial residuals",
            value=False,
            help="Display partial residuals instead of raw data points. Partial residuals show the relationship between the moderator and outcome while controlling for other variables in the model.",
        )

    with col2:
        plot_width = st.slider("Plot width", 600, 1200, 800, step=50)
        plot_height = st.slider("Plot height", 400, 800, 600, step=50)

        # # Axis limit controls
        # use_custom_limits = st.checkbox(
        #     "🎯 Custom axis limits",
        #     value=False,
        #     help="Enable manual control of plot axis limits",
        # )

        # Calculate y-limits efficiently without creating full plotter instance
        def _calculate_auto_y_limits():
            # Extract effect sizes directly from adapter
            effect_data = fitted_adapter.processed_df[
                fitted_adapter.effect_type
            ].dropna()
            min_yi, max_yi = effect_data.min(), effect_data.max()
            range_y = max_yi - min_yi
            return min_yi - range_y * 0.1, max_yi + range_y * 0.1

        auto_y_min, auto_y_max = _calculate_auto_y_limits()
        # if use_custom_limits:
        y_range = auto_y_max - auto_y_min
        y_buffer = y_range * 0.2  # 20% buffer for slider limits

        st.markdown("**Axis limits:**")

        # Y-axis limits
        col2a, col2b = st.columns(2)
        with col2a:
            y_min_slider = st.slider(
                "Y min",
                float(auto_y_min - y_buffer),
                float(auto_y_max + y_buffer),
                float(auto_y_min),
                step=float(y_range / 100),
                format="%.2f",
            )
        with col2b:
            y_max_slider = st.slider(
                "Y max",
                float(auto_y_min - y_buffer),
                float(auto_y_max + y_buffer),
                float(auto_y_max),
                step=float(y_range / 100),
                format="%.2f",
            )

        # # Reset button for axis limits
        # if st.button("🔄 Auto Limits", help="Reset to automatic axis limits"):
        #     st.rerun()

        # Validation warnings
        if y_min_slider >= y_max_slider:
            st.warning("⚠️ Y min should be less than Y max")

        # else:
        #     # Use automatic limits
        #     y_min_slider, y_max_slider = auto_y_min, auto_y_max

    if selected_moderator:
        try:
            # Create plotter
            plotter = StreamlitMetaRegressionPlotter(
                fitted_adapter,
                selected_moderator,
                colorby=colorby_value.lower(),
                # debug=debug_mode,
            )

            # Create plot
            fig = plotter.create_plotly_figure(
                title=f"Meta-regression: {fitted_adapter.effect_type} vs {selected_moderator}",
                width=plot_width,
                height=plot_height,
                show_partial_residuals=show_partial_residuals,
                custom_y_limits=(y_min_slider, y_max_slider),
            )

            # Display plot
            st.plotly_chart(fig, use_container_width=True)

            # Display summary statistics
            with st.expander("📊 Plot Statistics"):
                stats = plotter.create_summary_stats()
                for key, value in stats.items():
                    st.write(f"**{key}:** {value}")

            # Add download button for plot
            with st.expander("💾 Download Options"):
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("📊 Download Plot as HTML"):
                        html_str = fig.to_html()
                        st.download_button(
                            label="💾 Download HTML",
                            data=html_str,
                            file_name=f"metaregression_{selected_moderator}.html",
                            mime="text/html",
                        )
                with col2:
                    if st.button("📈 Download Plot Data as CSV"):
                        plot_data = pd.DataFrame(
                            {
                                selected_moderator: plotter.xi,
                                fitted_adapter.effect_type: plotter.yi,
                                f"{fitted_adapter.effect_type}_var": plotter.vi,
                                "study_weight": plotter.seinv,
                                "doi": plotter.dois,
                            }
                        )
                        csv = plot_data.to_csv(index=False)
                        st.download_button(
                            label="💾 Download CSV",
                            data=csv,
                            file_name=f"metaregression_data_{selected_moderator}.csv",
                            mime="text/csv",
                        )

        except Exception as e:
            st.error(f"Error creating plot: {e}")
            st.write(
                "Please check that the model was fitted successfully and try again."
            )


def test_streamlit_plotter():
    """Test the streamlit plotter with sample data."""
    print("🧪 TESTING STREAMLIT PLOTTER")
    print("============================")

    try:
        import pandas as pd

        from app.hybrid_metafor_adapter import StreamlitMetaforAdapter

        # Load sample data
        data = pd.read_csv("data/clean/analysis_ready_data.csv")
        sample_data = data[data["treatment"] == "temp"].head(15)

        print(f"✅ Loaded sample data: {sample_data.shape}")

        # Create and fit adapter
        adapter = StreamlitMetaforAdapter(
            df=sample_data,
            effect_type="st_relative_calcification",
            treatment="temp",
            verbose=True,
        )

        fitted_adapter = adapter.fit_model()
        print("✅ Model fitted")

        # Test plotter
        if "delta_t" in sample_data.columns:
            plotter = StreamlitMetaRegressionPlotter(fitted_adapter, "delta_t")
            print("✅ Plotter created")

            # Test figure creation
            fig = plotter.create_plotly_figure()
            print(f"✅ Figure created with {len(fig.data)} traces")

            # Test statistics
            stats = plotter.create_summary_stats()
            print(f"✅ Statistics created: {list(stats.keys())}")

            print("\n🎉 Streamlit plotter test passed!")
            return True
        else:
            print("⚠️  No 'delta_t' column found for testing")
            return False

    except Exception as e:
        print(f"❌ Streamlit plotter test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_streamlit_plotter()
