#!/usr/bin/env python3
"""
Streamlit-compatible MetaforModel adapter.

This adapter inherits from the original MetaforModel and overrides only the methods
that need special rpy2 context handling for Streamlit compatibility.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.metafor import MetaforModel  # noqa
from app.infrastructure import RContextManager  # noqa


class StreamlitMetaforAdapter(MetaforModel):
    """
    Streamlit-compatible adapter for metafor models.

    Inherits from MetaforModel and overrides only the methods that need
    special rpy2 context handling for Streamlit compatibility.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        effect_type: str = "st_relative_calcification",
        effect_type_var: str = None,
        treatment: str = None,
        formula: str = None,
        random: str = "~ 1 | original_doi/ID",
        required_columns: list = None,
        save_summary: bool = False,
        dvar_threshold: float = 100,
        process_data: bool = True,
        verbose: bool = True,
        **kwargs,
    ):
        """Initialize the adapter, safely calling parent constructor."""
        # Temporarily disable R operations in parent init to avoid problems with rpy2/streamlit contexts
        self._initializing = True

        # basic attributes first
        self.df = df.copy()
        self.original_df = df.copy()  # Keep original for plotting moderator selection
        self.effect_type = effect_type
        self.effect_type_var = effect_type_var or f"{effect_type}_var"
        self.treatment = treatment
        self.random = random
        self.verbose = verbose
        self.save_summary = save_summary
        self.dvar_threshold = dvar_threshold
        self.fitted = False

        # Streamlit-specific attributes
        self.model_dict = {}  # Python-friendly model data
        self.r_model = None  # Original R model object (OrdDict)

        # Handle formula using parent's logic but with safe context
        try:
            if formula is None:
                self.formula = self._get_model_formula()
            else:
                self.formula = formula

            # Get formula components using parent's method
            self.formula_components = self._get_formula_components()

            # Get required columns using parent's method
            from calcification.analysis import analysis_utils

            self.required_columns = analysis_utils._get_required_columns(
                self.effect_type,
                self.formula_components,
                required_columns,
            )

            # Prepare data using parent's method
            if process_data:
                self.processed_df = self._prepare_data()
            else:  # assume already processed
                self.processed_df = self.df

            # Set up R DataFrame using safe context
            self._safe_setup_r_df()

        except Exception as e:
            print(f"⚠️ Error during initialization: {e}")
            raise

        self._initializing = False

        if verbose:
            print("StreamlitMetaforAdapter initialized")
            print(f"   Formula: {self.formula}")
            print(f"   Treatment: {self.treatment}")
            print(f"   Data shape: {self.processed_df.shape}")

    def _safe_setup_r_df(self):
        """Set up the R DataFrame using safe context."""
        with RContextManager() as r_ctx:
            pandas2ri = r_ctx["pandas2ri"]

            # Use parent's logic for subsetting
            df_subset = self.processed_df[self.required_columns]

            # if factor moderators, check for n_levels > 2
            for mod in self.formula_components["factor_terms"]:
                if self.processed_df[mod].nunique() < 2:
                    raise ValueError(
                        f"⚠️ Factor moderator {mod} has {self.processed_df[mod].nunique()} levels, which is less than 2. This may cause problems with the model."
                    )
            self.df_subset = df_subset
            # Convert to R using safe context
            self.df_r = pandas2ri.py2rpy(df_subset)

            if self.verbose:
                print(f"R DataFrame created with {len(df_subset)} rows")

    def fit_model(self, formula: str = None):
        """Fit the model using safe R context (override parent's method)."""
        try:
            with RContextManager() as r_ctx:
                ro = r_ctx["ro"]

                # Import metafor and base within safe context
                metafor = ro.packages.importr("metafor")

                if self.verbose:
                    print(f"Fitting model with formula: {self.formula}")

                # Fit the model (returns OrdDict object)
                self.r_model = metafor.rma_mv(
                    yi=ro.FloatVector(self.df_r.rx2(self.effect_type)),
                    V=ro.FloatVector(self.df_r.rx2(self.effect_type_var)),
                    data=self.df_r,
                    mods=ro.Formula(self.formula)
                    if formula is None
                    else ro.Formula(formula),
                    random=ro.Formula(self.random),
                )

                # Extract model components for Streamlit compatibility
                self._extract_model_components()

                self.fitted = True

                if self.verbose:
                    print("Model fitted successfully")
                    print(f"   Components extracted: {list(self.model_dict.keys())}")

                return self

        except Exception as e:
            print(f"❌ Model fitting failed: {e}")
            raise RuntimeError(f"Model fitting failed: {e}")

    def extract_model_coefficient_info(self):
        """Extract model coefficients from the OrdDict R model object for Streamlit use."""
        try:
            coefficients = np.array(list(self.r_model["beta"]))
            coeff_names = self._extract_coefficient_names_from_model()
            result_coeffs = np.full(len(coeff_names), np.nan)
            # Determine which columns in the dataframe have all zeros
            zero_cols = set(self.df_subset.columns[self.df_subset.eq(0).all()])
            # this assumes (correctly) that the order matches for non-zero columns
            coeff_idx = 0
            for i, name in enumerate(coeff_names):
                # Check if name is a substring of any of the zero_cols
                if any(zero_col in name for zero_col in zero_cols):
                    result_coeffs[i] = 0
                else:
                    if coeff_idx < len(coefficients):
                        result_coeffs[i] = coefficients[coeff_idx]
                        coeff_idx += 1

            self.coefficients = result_coeffs
            self.coefficient_names = coeff_names

        except Exception as e:
            print(f"❌ Failed to extract model coefficients: {e}")

    def _extract_model_components(self):
        """Extract components from the OrdDict R model object for Streamlit use."""
        try:
            self.model_dict = {}
            self.extract_model_coefficient_info()

            # Extract other key statistics
            for key in [
                "method",
                "k",
                "QE",
                "QEp",
                "QM",
                "QMp",
                "pval",
                "se",
                "zval",
                "ci.lb",
                "ci.ub",
                "fit.stats",
            ]:
                if key in self.r_model:
                    try:
                        value = self.r_model[key]
                        if isinstance(value, pd.DataFrame):
                            row_names = {"ll": "LogLik", "dev": "Deviance"}
                            # rename rows using key
                            value.index = [
                                row_names[idx] if idx in row_names else idx
                                for idx in value.index
                            ]
                            self.model_dict[key] = value.to_dict()
                        elif hasattr(value, "__iter__") and not isinstance(value, str):
                            self.model_dict[key] = list(value)
                        else:
                            self.model_dict[key] = value
                    except Exception as e:
                        print(f"⚠️ Error extracting model component {key}: {e}")

            if self.verbose:
                print(f"Extracted model components: {list(self.model_dict.keys())}")
                if hasattr(self, "coefficient_names"):
                    print(f"Coefficient names: {self.coefficient_names}")

        except Exception as e:
            print(f"⚠️ Failed to extract some model components: {e}")

    def _extract_coefficient_names_from_model(self) -> list[str]:
        """Extract coefficient names from the model formula and structure."""
        try:
            # Method 1: get from call attribute via context handler
            with RContextManager() as r_ctx:
                ro = r_ctx["ro"]
                ro.globalenv["cl"] = self.r_model["call"]
                labels = ro.r(
                    "local({"
                    "  t <- terms(cl$mods); "
                    '  labs <- attr(t, "term.labels"); '
                    '  if (isTRUE(attr(t, "intercept") == 1L)) c("(Intercept)", labs) else labs'
                    "})"
                )
                return list(labels)

        except Exception as e:
            if self.verbose:
                print(f"⚠️  Could not extract coefficient names: {e}")
            # Fallback to generic names
            n_coef = (
                len(self.coefficients.flatten()) if hasattr(self, "coefficients") else 1
            )
            return [f"Coef {i + 1}" for i in range(n_coef)]

    def _try_extract_from_r_context(self) -> list[str]:
        """Try to extract coefficient names directly from R context."""
        try:
            with RContextManager() as r_ctx:
                ro = r_ctx["ro"]

                # Try to create a temporary R object and extract rownames
                # We'll use the beta values to reconstruct the model object temporarily
                if "beta" in self.r_model and hasattr(self.r_model["beta"], "__iter__"):
                    # Create a simple matrix in R and see if we can get names
                    beta_values = list(self.r_model["beta"])
                    ro.r.assign("temp_beta", ro.FloatVector(beta_values))

                    # Try to get the original model's beta rownames if available
                    # This is a bit hacky but might work
                    return None  # For now, return None and rely on other methods

        except Exception as e:
            if self.verbose:
                print(f"⚠️  R context extraction failed: {e}")
            return None

    def _parse_coefficient_names_from_formula(self) -> list[str]:
        """Parse coefficient names from formula with improved factor handling."""
        coef_names = []

        # Split formula to get the right side
        if "~" not in self.formula:
            return coef_names

        formula_rhs = self.formula.split("~")[1].strip()

        # Handle intercept
        has_intercept = "- 1" not in formula_rhs and "-1" not in formula_rhs
        if has_intercept:
            coef_names.append("(Intercept)")

        # Remove intercept indicators
        formula_rhs = formula_rhs.replace("- 1", "").replace("-1", "").strip()

        if formula_rhs and formula_rhs != "1":
            # Split by + and clean up terms
            terms = [term.strip() for term in formula_rhs.split("+") if term.strip()]

            for term in terms:
                # Handle different term types
                if term.startswith("factor(") and term.endswith(")"):
                    # Factor term - extract actual factor levels from the data
                    var_name = term[7:-1]  # Remove "factor(" and ")"
                    if var_name in self.df.columns:
                        # Get unique values as they appear in the data
                        unique_vals = sorted(
                            self.processed_df[var_name].dropna().unique()
                        )

                        # When there's no intercept (- 1), R creates coefficients for ALL levels
                        # When there's an intercept, R creates n-1 dummy variables
                        if has_intercept:
                            # Skip reference level (first alphabetically)
                            for val in unique_vals[1:]:
                                coef_names.append(f"factor({var_name}){val}")
                        else:
                            # Include all levels when no intercept
                            for val in unique_vals:
                                coef_names.append(f"factor({var_name}){val}")
                elif term.startswith("I(") and term.endswith(")"):
                    # Identity function (e.g., I(x^2))
                    coef_names.append(term)
                else:
                    # Simple term
                    coef_names.append(term)

        return coef_names

    def get_coefficient_names(self) -> list[str]:
        """Get the names of the model coefficients."""
        if not self.fitted:
            raise RuntimeError(
                "Model must be fitted before extracting coefficient names."
            )
        return getattr(
            self,
            "coefficient_names",
            [f"Coef {i + 1}" for i in range(len(self.coefficients))],
        )

    def get_coefficients(self) -> np.ndarray:
        """
        Extract coefficients (override parent to work with OrdDict).
        Maintains compatibility with original MetaforModel API.
        """
        if not self.fitted:
            raise RuntimeError("Model must be fitted before extracting coefficients.")

        return self.coefficients

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
            coef_df = pd.DataFrame(
                {
                    "Estimate": coef_values,
                    "SE": self.model_dict["se"],
                    "Z-value": self.model_dict["zval"],
                    "P-value": self.model_dict["pval"],
                    "CI Lower": self.model_dict["ci.lb"],
                    "CI Upper": self.model_dict["ci.ub"],
                    "Significance": significance_stars,
                    "Significant": [
                        p < 0.05 if not np.isnan(p) else False
                        for p in self.model_dict["pval"]
                    ],
                },
                index=row_names,
            )

            return coef_df

        except Exception as e:
            print(f"⚠️ Error creating coefficients DataFrame: {e}")
            return pd.DataFrame({"Error": [str(e)]})

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
            else:
                row_names = [f"Coef {i + 1}" for i in range(n_coef)]

            # Extract all statistics
            stats = {}
            stats["Estimate"] = [f"{float(coef):.4f}" for coef in coef_values]

            for key, label in [
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
            lines.append("=" * len(self.formula))
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
            if "beta" in self.model_dict:
                lines.append("\nCoefficients:")
                coef_table = self._format_coefficients_table()
                lines.append(coef_table)

            return "\n".join(lines)

        except Exception as e:
            return f"Error generating summary: {e}"

    def get_model_metadata(self) -> dict:
        """Get model metadata (uses parent's method with additional info)."""
        base_metadata = super().get_model_metadata()

        # Add Streamlit-specific metadata
        base_metadata.update(
            {
                "adapter_type": "StreamlitMetaforAdapter",
                "model_dict_keys": list(self.model_dict.keys())
                if hasattr(self, "model_dict")
                else [],
                "fitted": self.fitted,
            }
        )

        return base_metadata
