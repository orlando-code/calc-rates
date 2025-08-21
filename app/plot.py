#!/usr/bin/env python3
"""
Unified Post-Meta-Analysis Plotting Module

Provides concise, streamlit-ready functions for creating forest plots, funnel plots,
and meta-regression plots from fitted metafor models. Combines and improves upon
functionality from streamlit_plotter.py and calcification/plotting/analysis.py.
"""

import sys
from pathlib import Path
from typing import Any, Tuple, Union

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from app import helpers, metafor
from calcification.plotting import plot_config

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    import streamlit as st

    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False


class MetaAnalysisPlotter:
    """Unified plotter for post-meta-analysis visualizations."""

    def __init__(self, model: Any, verbose: bool = False):
        """Initialize plotter with a fitted metafor model."""
        self.model = model
        self.verbose = verbose
        self.fitted = getattr(model, "fitted", False)

        if not self.fitted:
            raise ValueError("Model must be fitted before plotting")


def plot_forest(
    model: Any,
    title: str = "Forest Plot",
    effect_label: str = None,
    study_labels: str = "original_doi",
    confidence_level: float = 95,
    interactive: bool = True,
    **kwargs,
) -> Union[go.Figure, Tuple[plt.Figure, plt.Axes]]:
    """Create a forest plot showing individual study effects."""
    plotter = MetaAnalysisPlotter(model)

    if plotter.df is None:
        raise ValueError("No data available from model for forest plot")

    # Extract study-level data
    df = plotter.df.copy()
    effect_col = plotter.effect_type
    var_col = plotter.effect_var_type

    # Calculate confidence intervals
    z_score = 1.96  # 95% CI
    df = df.copy()
    df["ci_lower"] = df[effect_col] - z_score * np.sqrt(df[var_col])
    df["ci_upper"] = df[effect_col] + z_score * np.sqrt(df[var_col])
    df["study_weight"] = 1 / df[var_col]

    # Get study labels
    if study_labels in df.columns:
        df["study_label"] = df[study_labels].astype(str)
    else:
        df["study_label"] = [f"Study {i + 1}" for i in range(len(df))]

    df = df.sort_values(effect_col).reset_index(drop=True)
    effect_label = effect_label or plotter.effect_type.replace("_", " ").title()

    # if interactive:
    return _create_forest_plot_plotly(df, effect_col, title, effect_label, **kwargs)
    # else:
    #     return _create_forest_plot_matplotlib(
    #         df, effect_col, title, effect_label, **kwargs
    # )


def plot_funnel(
    model: Any,
    title: str = "Funnel Plot",
    effect_label: str = None,
    precision_metric: str = "se",
    interactive: bool = True,
    **kwargs,
) -> Union[go.Figure, Tuple[plt.Figure, plt.Axes]]:
    """Create a funnel plot for assessing publication bias."""
    plotter = MetaAnalysisPlotter(model)

    if plotter.df is None:
        raise ValueError("No data available from model for funnel plot")

    df = plotter.df.copy()
    effect_col = plotter.effect_type
    var_col = plotter.effect_var_type

    # Calculate precision metrics
    df["se"] = np.sqrt(df[var_col])
    df["vi"] = df[var_col]
    df["seinv"] = 1 / df["se"]
    df["vinv"] = 1 / df["vi"]

    effect_label = effect_label or plotter.effect_type.replace("_", " ").title()

    # if interactive:
    return _create_funnel_plot_plotly(
        df, effect_col, precision_metric, title, effect_label, **kwargs
    )
    # else:
    # return _create_funnel_plot_matplotlib(
    #     df, effect_col, precision_metric, title, effect_label, **kwargs
    # )


class MetaRegressionPlotter:
    """Unified plotter for meta-regression plots."""

    def __init__(
        self,
        model: Any,
        moderator_name: str,
        verbose: bool = False,
        colorby: str = None,
    ):
        """Initialize plotter with a fitted metafor model."""
        self.model = model
        self.moderator_name = moderator_name
        self.verbose = verbose
        self.colorby = colorby
        self.fitted = getattr(model, "fitted", False)

        # Initialize partial residuals attributes
        self.partial_residuals_x = None
        self.partial_residuals_y = None

        self.model.get_model_data_for_plotting(moderator_name)
        self.get_plotting_data()

        if not self.fitted:
            raise ValueError("Model must be fitted before plotting")

    def _get_y_limits(self, show_partial_residuals: bool = False):
        """Get the y limits of the plot."""
        min_yi, max_yi = np.min(self.model.yi), np.max(self.model.yi)

        # if partial residuals available, make sure these are within the y limits
        if (
            show_partial_residuals
            and hasattr(self, "partial_residuals_y")
            and self.partial_residuals_y is not None
        ):
            min_partial, max_partial = (
                np.min(self.partial_residuals_y),
                np.max(self.partial_residuals_y),
            )
            min_yi, max_yi = min(min_partial, min_yi), max(max_partial, max_yi)

        # TODO: frame within prediction interval (if available)

        range_y = max_yi - min_yi
        return min_yi - range_y * 0.1, max_yi + range_y * 0.1

    def _get_x_limits(self, show_partial_residuals: bool = False):
        """Get the x limits of the plot - consistent for both data and partial residuals."""
        min_xi, max_xi = np.min(self.model.xi), np.max(self.model.xi)

        # Always use the same x limits whether showing data or partial residuals
        # This ensures the x-axis doesn't change when toggling
        if (
            hasattr(self, "partial_residuals_x")
            and self.partial_residuals_x is not None
        ):
            min_partial_x, max_partial_x = (
                np.min(self.partial_residuals_x),
                np.max(self.partial_residuals_x),
            )
            min_xi, max_xi = min(min_partial_x, min_xi), max(max_partial_x, max_xi)

        range_x = max_xi - min_xi
        return min_xi - range_x * 0.05, max_xi + range_x * 0.05

    def get_plotting_data(self):
        """Extract data needed for plotting from the adapter."""

        # self.seinv = 1 / np.sqrt(
        #     self.model.vi
        # )  # Inverse standard error for point sizing

        # Get values for hover text
        dois = self.model.df_processed.get(
            "original_doi", ["Unknown"] * len(self.model.xi)
        )
        st_control_calcification = self.model.df_processed.get(
            "st_control_calcification", ["Unknown"] * len(self.model.xi)
        )
        st_treatment_calcification = self.model.df_processed.get(
            "st_treatment_calcification", ["Unknown"] * len(self.model.xi)
        )

        # Generate predictions
        self.pred, self.se, self.ci_lb, self.ci_ub, self.pred_lb, self.pred_ub = (
            self.model.predict_on_moderator(self.moderator_name)
        )

        self.dois = dois
        self.st_control_calcification = st_control_calcification
        self.st_treatment_calcification = st_treatment_calcification

        #  get prediction range
        x_min, x_max = np.min(self.model.xi), np.max(self.model.xi)
        x_range = x_max - x_min
        self.xs = np.linspace(
            x_min - 0.1 * x_range,
            x_max + 0.1 * x_range,
            100,
        )

        # get predictions
        predictions = self.model.predict_on_moderator(self.moderator_name)
        self.pred = predictions["pred"]
        self.se = predictions["se"]
        self.ci_lb = predictions["ci_lb"]
        self.ci_ub = predictions["ci_ub"]
        self.pred_lb = predictions["pred_lb"]
        self.pred_ub = predictions["pred_ub"]

    def _calculate_point_sizes(self, variances: list[float]):
        max_vi = np.max(variances) if np.max(variances) > 0 else 1
        point_sizes = (variances / max_vi) * 40 + 8
        return point_sizes

    def calculate_partial_residuals(self):
        """Calculate partial residuals leveraging existing model fitting infrastructure."""
        if self.verbose:
            print(f"Calculating partial residuals for moderator: {self.moderator_name}")

        return calculate_partial_residuals(self.model, self.moderator_name)

    def _determine_point_colours(self, fig) -> tuple[list[str], bool]:
        """Determine point colours based on the colorby variable."""
        is_numeric_color = False
        if self.colorby == "core_grouping":
            scatter_points_colours = get_core_grouping_colours(self.model.df_processed)
            self._add_discrete_color_legend(fig)
        elif self.colorby:
            # Try to find the color variable in processed df first, then original df
            color_values = None
            if self.colorby in self.model.df.columns:
                color_values = self.model.df[self.colorby]
            elif hasattr(self, "original_df") and self.colorby in self.model.df.columns:
                # Get values from original df but only for the rows that exist in processed df
                # Match by index to ensure alignment
                color_values = self.model.df.loc[
                    self.model.df_processed.index, self.colorby
                ]

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
                    self._add_discrete_color_legend(fig)
            else:
                scatter_points_colours = "white"
        else:
            scatter_points_colours = "white"

        return scatter_points_colours, is_numeric_color

    def _add_discrete_color_legend(self, fig):
        """Add discrete color legend for categorical variables."""

        values = [v for v in self.model.df_processed[self.colorby].dropna().unique()]

        if self.colorby == "core_grouping":
            color_map = plot_config.CG_COLOURS
            for value in values:
                if value in color_map:
                    color = normalize_color(color_map[value])
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
                color = normalize_color(palette[i % n])
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

    def _plot_regression_line(self, fig, show_partial_residuals: bool = False):
        """Plot the regression line."""

        if hasattr(self, "pred_lb") and hasattr(self, "pred_ub"):
            fig.add_trace(
                go.Scatter(
                    x=np.concatenate([self.xs, self.xs[::-1]]),
                    y=np.concatenate([self.pred_ub, self.pred_lb[::-1]]),
                    fill="toself",
                    fillcolor="rgba(173, 216, 230, 0.25)",  # lighter blue, lower alpha
                    line=dict(color="rgba(255,255,255,0)"),
                    name="95% Prediction Interval",
                    showlegend=True,
                    hoverinfo="skip",
                    legendgroup="intervals",
                )
            ) if not show_partial_residuals else None

        # Add confidence interval (narrower, darker)
        fig.add_trace(
            go.Scatter(
                x=np.concatenate([self.xs, self.xs[::-1]]),
                y=np.concatenate([self.ci_ub, self.ci_lb[::-1]]),
                fill="toself",
                fillcolor="rgba(30, 144, 255, 0.3)",  # darker blue, higher alpha
                line=dict(color="rgba(255,255,255,0)"),
                name="95% Confidence Interval",
                showlegend=True,
                hoverinfo="skip",
                legendgroup="intervals",
            )
        ) if not show_partial_residuals else None

        # Add regression line
        fig.add_trace(
            go.Scatter(
                x=self.xs,
                y=self.pred,
                mode="lines",
                line=dict(color="blue", width=3),
                # name=f"Meta-regression: {self.model.effect_type} ~ {self.moderator_name}",
                showlegend=False,
            )
        ) if not show_partial_residuals else None

    def _format_axis(
        self, custom_y_limits: tuple = None, show_partial_residuals: bool = False
    ):
        if custom_y_limits is not None:
            self.y_min, self.y_max = custom_y_limits
        else:
            self.y_min, self.y_max = self._get_y_limits(
                show_partial_residuals=show_partial_residuals
            )

    def _get_partial_residuals_hovertext(self):
        """Get the hover text for the partial residuals."""
        return [
            f"<b>DOI:</b> {doi}<br><b>Ordinary Residual:</b> {ordr:.3f}<br><b>Partial Residual:</b> {pr:.3f}<br><b>Original {self.model.effect_type}:</b> {y:.3f}<br><b>Core Grouping:</b> {cg}<br>"
            for doi, ordr, pr, y, cg in zip(
                self.model.df_processed["original_doi"],
                self.model.ord_residuals,
                self.partial_residuals_y,
                self.model.yi,
                self.model.df_processed["core_grouping"],
            )
        ]

    def _get_data_hovertext(self):
        """Get the hover text for the data points."""
        merged_data = self.model.df.merge(
            self.model.df_processed, how="left"
        )  # TODO: may be wrong way round

        return [
            f"<b>DOI:</b> {doi}<br><b>Effect Size:</b> {ys:.3f}<br><b>Control Calcification:</b> {st_control_calc:.3f}<br><b>Treatment Calcification:</b> {st_treatment_calc:.3f}<br><b>Core Grouping:</b> {cg}"
            for doi, ys, st_control_calc, st_treatment_calc, cg in zip(
                self.model.df_processed["original_doi"],
                self.model.yi,
                merged_data["st_control_calcification"],
                merged_data["st_treatment_calcification"],
                merged_data["core_grouping"],
            )
        ]

    def _plot_scatter_points(
        self,
        fig,
        xs,
        ys,
        hover_text,
        point_sizes,
        point_colours,
        is_numeric_color,
        name=None,
        showlegend=True,
    ):
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="markers",
                marker=dict(
                    size=point_sizes,
                    color=point_colours,
                    line=dict(color="navy", width=2),
                    opacity=0.8,
                    colorscale=get_colorscale_for_continuous_color_values(self.colorby)
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
                name=name or "Samples (size ∝ precision)",
                text=hover_text,
                hovertemplate="%{text}<extra></extra>",
                showlegend=showlegend,
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
            n_samples = len(self.model.xi)

            # Extract model statistics
            qe = (
                self.model.model_dict.get("QE", [None])[0]
                if "QE" in self.model.model_dict
                else None
            )
            qm = (
                self.model.model_dict.get("QM", [None])[0]
                if "QM" in self.model.model_dict
                else None
            )
            stats = {
                "Number of samples": n_samples,
                "Moderator": self.moderator_name,
                "Formula": self.model.formula,
                "Residual heterogeneity (QE)": f"{float(qe):.3f}"
                if qe is not None
                else "N/A",
                "Model test statistic (QM)": f"{float(qm):.3f}"
                if qm is not None
                else "N/A",
                "Effect size range": f"{np.min(self.model.yi):.3f} to {np.max(self.model.yi):.3f}",
                "Moderator range": f"{np.min(self.model.xi):.3f} to {np.max(self.model.xi):.3f}",
            }

            return stats

        except Exception as e:
            return {"Error": f"Could not generate statistics: {e}"}

    def plot_plotly_meta_regression(
        self,
        title: str = None,
        custom_y_limits: tuple = None,
        width: int = 800,
        height: int = 600,
        show_partial_residuals: bool = False,
    ) -> go.Figure:
        """Plot a meta-regression plot using Plotly."""
        fig = go.Figure()

        # get point colours
        point_colours, is_numeric_color = self._determine_point_colours(fig)

        # Calculate partial residuals if requested
        if show_partial_residuals:
            if self.verbose:
                print(f"🔮 Calculating partial residuals for {self.moderator_name}")

            # Temporarily force verbose for debugging
            original_verbose = self.model.verbose
            self.model.verbose = True

            partial_residuals = self.calculate_partial_residuals()

            # Restore original verbose setting
            self.model.verbose = original_verbose

            if partial_residuals is not None:
                self.partial_residuals_x = partial_residuals[0]
                self.partial_residuals_y = partial_residuals[1]
                if self.verbose:
                    print(
                        f"   ✅ Partial residuals calculated: {len(self.partial_residuals_y)} points"
                    )
                    print(
                        f"   📊 Partial residuals x-range: {np.min(self.partial_residuals_x):.3f} to {np.max(self.partial_residuals_x):.3f}"
                    )
                    print(
                        f"   📊 Partial residuals y-range: {np.min(self.partial_residuals_y):.3f} to {np.max(self.partial_residuals_y):.3f}"
                    )
            else:
                if self.verbose:
                    print("   ❌ Failed to calculate partial residuals")
                self.partial_residuals_x = None
                self.partial_residuals_y = None

        # format axis
        self._format_axis(custom_y_limits, show_partial_residuals)

        # plot partial residuals or data points
        if (
            show_partial_residuals
            and self.partial_residuals_x is not None
            and self.partial_residuals_y is not None
            and len(self.partial_residuals_x) > 0
            and len(self.partial_residuals_y) > 0
        ):
            self._plot_scatter_points(
                fig,
                self.partial_residuals_x,
                self.partial_residuals_y,
                self._get_partial_residuals_hovertext(),
                self._calculate_point_sizes(self.model.vi),
                point_colours,
                is_numeric_color,
                name="Partial Residuals",
                showlegend=False,  # Don't show legend for partial residuals to avoid duplication
            )
        else:
            # plot regression line
            self._plot_regression_line(
                fig, show_partial_residuals=show_partial_residuals
            )
            # plot data points
            self._plot_scatter_points(
                fig,
                self.model.xi,
                self.model.yi,
                self._get_data_hovertext(),
                self._calculate_point_sizes(self.model.vi),
                point_colours,
                is_numeric_color,
                name="Samples (size ∝ precision)",
                showlegend=True,
            )

        # add reference line at zero
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

        # update layout
        fig.update_layout(
            title=title or f"Meta-regression: {self.model.formula}",
            xaxis_title=self._format_axis_label(self.moderator_name),
            yaxis_title=self._format_axis_label(self.model.effect_type)
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
            yaxis_range=[self.y_min, self.y_max],
            xaxis_range=self._get_x_limits(show_partial_residuals),
        )

        return fig


# Helper functions for Plotly plots
def _create_forest_plot_plotly(df, effect_col, title, effect_label, **kwargs):
    """Create interactive forest plot using Plotly."""
    fig = go.Figure()

    # Add individual studies
    fig.add_trace(
        go.Scatter(
            x=df[effect_col],
            y=df["study_label"],
            mode="markers",
            marker=dict(
                size=np.sqrt(df["study_weight"]) * 5,
                color="lightblue",
                line=dict(color="navy", width=1),
            ),
            error_x=dict(
                type="data",
                symmetric=False,
                array=df["ci_upper"] - df[effect_col],
                arrayminus=df[effect_col] - df["ci_lower"],
                color="blue",
                thickness=2,
            ),
            name="Individual Studies",
            hovertemplate="<b>%{y}</b><br>"
            + f"{effect_label}: %{{x:.3f}}<br>"
            + "CI: [%{customdata[0]:.3f}, %{customdata[1]:.3f}]<extra></extra>",
            customdata=np.column_stack([df["ci_lower"], df["ci_upper"]]),
        )
    )

    # Add reference line at zero
    fig.add_vline(x=0, line_dash="dash", line_color="red", opacity=0.7)

    fig.update_layout(
        title=title,
        xaxis_title=effect_label,
        yaxis_title="Studies",
        height=max(400, len(df) * 25),
        showlegend=False,
        template="plotly_white",
    )

    return fig


def _create_funnel_plot_plotly(
    df, effect_col, precision_metric, title, effect_label, **kwargs
):
    """Create interactive funnel plot using Plotly."""
    fig = go.Figure()

    # Add data points
    fig.add_trace(
        go.Scatter(
            x=df[effect_col],
            y=df[precision_metric],
            mode="markers",
            marker=dict(size=8, color="lightblue", line=dict(color="navy", width=1)),
            name="Studies",
            hovertemplate="<b>Effect Size:</b> %{x:.3f}<br>"
            + f"<b>{precision_metric.upper()}:</b> %{{y:.3f}}<extra></extra>",
        )
    )

    # Add reference line at overall effect (if calculable)
    try:
        overall_effect = np.average(df[effect_col], weights=1 / df["se"] ** 2)
        fig.add_vline(
            x=overall_effect,
            line_dash="dash",
            line_color="red",
            annotation_text="Overall Effect",
        )
    except Exception:
        pass

    fig.update_layout(
        title=title,
        xaxis_title=effect_label,
        yaxis_title=precision_metric.upper(),
        yaxis=dict(autorange="reversed")
        if precision_metric in ["se", "vi"]
        else dict(),
        template="plotly_white",
    )

    return fig


def _create_regression_plot_plotly(
    df, moderator, effect_col, var_col, x_range, predictions, title, color_by, **kwargs
):
    """Create interactive meta-regression plot using Plotly."""
    fig = go.Figure()

    # Add confidence intervals if available
    if predictions and "ci_lower" in predictions:
        fig.add_trace(
            go.Scatter(
                x=np.concatenate([x_range, x_range[::-1]]),
                y=np.concatenate(
                    [predictions["ci_upper"], predictions["ci_lower"][::-1]]
                ),
                fill="toself",
                fillcolor="rgba(173, 216, 230, 0.5)",
                line=dict(color="rgba(255,255,255,0)"),
                name="95% Confidence Interval",
                showlegend=True,
                hoverinfo="skip",
            )
        )

    # Add regression line if available
    if predictions and "prediction" in predictions:
        fig.add_trace(
            go.Scatter(
                x=x_range,
                y=predictions["prediction"],
                mode="lines",
                line=dict(color="blue", width=3),
                name="Regression Line",
                showlegend=True,
            )
        )

    # Determine point colors and sizes
    if color_by and color_by in df.columns:
        color_vals = df[color_by]
        if pd.api.types.is_numeric_dtype(color_vals):
            colors = color_vals
            colorscale = "Viridis"
        else:
            unique_vals = color_vals.unique()
            color_map = {
                val: px.colors.qualitative.Set3[i % len(px.colors.qualitative.Set3)]
                for i, val in enumerate(unique_vals)
            }
            colors = [color_map[val] for val in color_vals]
            colorscale = None
    else:
        colors = "lightblue"
        colorscale = None

    se_vals = np.sqrt(df[var_col])
    max_se = np.max(se_vals)
    point_sizes = (1 / se_vals) / (1 / max_se) * 20 + 5

    # Add data points
    fig.add_trace(
        go.Scatter(
            x=df[moderator],
            y=df[effect_col],
            mode="markers",
            marker=dict(
                size=point_sizes,
                color=colors,
                colorscale=colorscale,
                line=dict(color="navy", width=1),
                showscale=pd.api.types.is_numeric_dtype(colors)
                if hasattr(colors, "dtype")
                else False,
            ),
            name="Studies",
            hovertemplate=f"<b>{moderator}:</b> %{{x:.3f}}<br>"
            + f"<b>{effect_col}:</b> %{{y:.3f}}<br>"
            + "<b>SE:</b> %{customdata:.3f}<extra></extra>",
            customdata=se_vals,
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title=moderator.replace("_", " ").title(),
        yaxis_title=effect_col.replace("_", " ").title(),
        template="plotly_white",
        height=600,
    )

    return fig


def plot_contour(
    model: Any,
    moderatorx: str,
    moderatory: str,
    modx_range: tuple[float, float] = (0, 10),
    mody_range: tuple[float, float] = (0, -1),
    title: str = None,
    # n_points: int = 50,
) -> Union[go.Figure, Tuple[plt.Figure, plt.Axes]]:
    """
    Create a 2D contour plot showing predicted effect sizes across two moderators.

    This function demonstrates the reusability of the MetaforPredictor class
    for different plot types beyond simple meta-regression plots.

    Args:
        model: Fitted metafor model
        moderatorx: Name of first moderator (x-axis)
        moderatory: Name of second moderator (y-axis)
        title: Plot title
        n_points: Number of grid points in each dimension
        interactive: Whether to return Plotly (True) or matplotlib (False) figure
        **kwargs: Additional plotting arguments

    Returns:
        Plotly Figure if interactive=True, else matplotlib Figure and Axes
    """
    try:
        # Create the ranges with consistent default parameters
        n_points = 50
        surface, meshgrids = model.predict_nd_surface_from_model(
            moderator_names=[moderatorx, moderatory],
            moderator_values=[
                np.linspace(*modx_range, n_points),
                np.linspace(*mody_range, n_points),
            ],
        )

        return _create_contour_plot_plotly(
            meshgrids[0],
            meshgrids[1],
            surface,
            moderatorx,
            moderatory,
            title,
        )

    except ImportError:
        raise ImportError("Contour plots require the prediction module")
    except Exception as e:
        import traceback

        st.code(traceback.format_exc(), language="python")

        raise RuntimeError(f"Could not create contour plot: {e}")


def _create_contour_plot_plotly(X1, X2, surface, modx, mody, title):
    """Create interactive contour plot using Plotly."""
    # Extract the 1D arrays from the meshgrid edges for Plotly
    # Plotly expects 1D arrays for x and y coordinates
    x_coords = X1[:, 0]  # First column (constant x, varying y)
    y_coords = X2[0, :]  # First row (constant y, varying x)

    # Prepare customdata as a (n_points, n_points, 3) array: [mod1, mod2, effect]
    # Each element is [X1, X2, surface] at that grid point
    customdata = np.stack([X1, X2, surface], axis=-1)

    # Debug info - print shapes to verify alignment
    print(
        f"Debug contour plot: x_coords shape: {x_coords.shape}, y_coords shape: {y_coords.shape}, surface shape: {surface.shape}"
    )

    fig = go.Figure(
        data=go.Contour(
            x=x_coords,
            y=y_coords,
            z=surface,
            # colorscale="reds",
            colorscale="rdylbu",
            zmin=-np.abs(surface).max(),
            zmax=np.abs(surface).max(),
            # zmid=0,  # Set 0 as the midpoint of the color scale
            contours=dict(showlabels=True, labelfont=dict(size=12, color="white")),
            customdata=customdata,
            hovertemplate=(
                f"<b>{map_name_to_display_name(modx)}:</b> %{{customdata[0]:.3f}}<br>"
                f"<b>{map_name_to_display_name(mody)}:</b> %{{customdata[1]:.3f}}<br>"
                f"<b>Effect Size:</b> %{{customdata[2]:.3f}}<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        title=title
        or f"Contour plot: {map_name_to_display_name(modx)} vs {map_name_to_display_name(mody)}",
        xaxis_title=map_name_to_display_name(modx),
        yaxis_title=map_name_to_display_name(mody),
        template="plotly_white",
    )

    return fig


def map_name_to_display_name(name: str) -> str:
    """Map a name to a display name."""
    return helpers.VAR_NAME_MAP.get(name, name)


# Streamlit convenience function
def plot_meta_analysis_suite(
    model,
    moderators: list[str] = None,
    show_forest: bool = False,
    show_funnel: bool = False,
    show_regression: bool = False,
    show_contour: bool = False,
):
    """Create a complete suite of meta-analysis plots in Streamlit."""
    if not STREAMLIT_AVAILABLE:
        raise ImportError("This function requires Streamlit")

    st.subheader("📊 Meta-Analysis Plots")
    show_partial_residuals = st.checkbox(
        "Show partial residuals",
        value=False,
        help="Display partial residuals instead of raw data points. Partial residuals show the relationship between the moderator and outcome while controlling for other variables in the model.",
    )

    if show_forest:
        with st.expander("🌲 Forest Plot", expanded=True):
            try:
                fig = plot_forest(model, interactive=True)
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Could not create forest plot: {e}")

    if show_funnel:
        with st.expander("🔍 Funnel Plot", expanded=True):
            try:
                fig = plot_funnel(model, interactive=True)
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Could not create funnel plot: {e}")

    if show_regression and moderators:
        with st.expander("📈 Meta-Regression Plots", expanded=True):
            if len(moderators) > 1:
                st.error(
                    f"Only one moderator can be plotted at a time. Defaulting to the first moderator ({moderators[0]})."
                )

            try:
                st.subheader(f"Regression: {moderators[0]}")
                fig = MetaRegressionPlotter(
                    model,
                    moderators[0],
                    verbose=True,
                ).plot_plotly_meta_regression(
                    show_partial_residuals=show_partial_residuals
                )
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Could not create regression plot for {moderators[0]}: {e}")
                import traceback

                st.code(traceback.format_exc(), language="python")

    if show_contour and moderators:
        with st.expander("📊 Contour Plot", expanded=True):
            if len(moderators) != 2:
                st.error(
                    f"Contour plots require exactly two moderators. Defaulting to the first two moderators ({moderators[0]} and {moderators[1]})."
                )
            try:
                fig = plot_contour(
                    model, moderators[0], moderators[1], interactive=True
                )
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Could not create contour plot: {e}")


__all__ = [
    "MetaAnalysisPlotter",
    "plot_forest",
    "plot_funnel",
    "MetaRegressionPlotter",
    "plot_contour",
    "plot_meta_analysis_suite",
]


# --- helpers (may be moved to plot_utils.py) ---
def get_core_grouping_colours(df: pd.DataFrame) -> list[str]:
    """Get colors for core_grouping using standard color scheme."""

    core_grouping_values = df.core_grouping
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


def coefficient_involves_moderator(coef_name: str, moderator_name: str) -> bool:
    """Check if a coefficient name involves the specified moderator."""
    # direct match
    if coef_name == moderator_name:
        return True

    # check for transformations like I(delta_t^2), I(delta_t + delta_ph), etc.
    if coef_name.startswith("I(") and moderator_name in coef_name:
        return True

    # check for interaction terms like delta_t:delta_ph
    if ":" in coef_name and moderator_name in coef_name.split(":"):
        return True

    # check for factor transformations like factor(delta_t)level
    if coef_name.startswith(f"factor({moderator_name})"):
        return True

    # check for polynomial terms
    if coef_name.startswith("poly(") and moderator_name in coef_name:
        return True

    # check for spline terms like bs(delta_t)1, ns(delta_t)2
    if any(
        coef_name.startswith(f"{spline}({moderator_name})")
        for spline in ["bs", "ns", "s", "rcs"]
    ):
        return True

    return False


def evaluate_moderator_term(
    term: str, moderator_name: str, moderator_values: np.ndarray
) -> np.ndarray:
    # TODO: check for duplicate code
    """Evaluate a formula term given moderator values."""
    # simple linear term
    if term == moderator_name:
        return moderator_values

    # identity transformations like I(delta_t^2), I(delta_t^3), etc.
    if term.startswith("I(") and term.endswith(")"):
        expression = term[2:-1]  # Remove I( and )

        # handle powers: delta_t^2, delta_t^3, etc.
        if "^" in expression:
            base, power = expression.split("^")
            if base.strip() == moderator_name:
                return moderator_values ** float(power.strip())

    # factor levels - return indicator variables (would need actual factor levels)
    if term.startswith(f"factor({moderator_name})"):
        # ignore these: can't be adjusted for
        return moderator_values

    # default fallback
    return moderator_values


def extract_variable_name_from_coef(coef_name: str) -> str:
    # CHECK FOR DUPLICATION AND MOVE TO ANALYSIS_UTILS
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


def create_reduced_formula(model, moderator_name: str) -> str:
    """Create a reduced formula by removing the moderator of interest."""
    formula_str = str(model.formula)
    if "~" not in formula_str:
        raise ValueError("Invalid formula format")

    lhs, rhs = formula_str.split("~", 1)
    terms = [t.strip() for t in rhs.split("+")]

    # remove terms that contain the plotting moderator
    reduced_terms = []
    for term in terms:
        # skip terms that are the moderator itself or transformations of it
        if moderator_name not in term and not (
            term.startswith("I(") and moderator_name in term
        ):
            reduced_terms.append(term)

    if reduced_terms:
        return f"{lhs.strip()} ~ {' + '.join(reduced_terms)}"
    else:
        return f"{lhs.strip()} ~ 1"  # intercept only


def get_colorscale_for_continuous_color_values(colorby: str) -> str:
    """Get the color values for continuous variables, dependent on the moderator of interest."""
    if colorby == "delta_t":
        return "Reds"
    elif colorby == "delta_ph":
        return "Reds_r"
    else:
        return "Viridis"


def normalize_color(c):
    # Keep valid Plotly color strings as-is; convert tuples/lists to hex
    if isinstance(c, (tuple, list)):
        arr = np.array(c, dtype=float)
        if arr.max() > 1.0:
            arr = arr / 255.0
        return mcolors.to_hex(arr[:3])
    return str(c)


# --- residuals ---


def extract_residuals_from_model(model: "metafor.MetaforModel") -> np.ndarray:
    """Extract residuals from an adapter using the existing R context handling."""
    try:
        from app.infrastructure import RContextManager

        with RContextManager() as r_ctx:
            ro = r_ctx["ro"]
            localconverter = r_ctx["localconverter"]

            # use the existing R model stored in the adapter
            ro.globalenv["cl"] = model.r_model["call"]
            ro.globalenv["d"] = model.df_r

            with localconverter(ro.default_converter):
                # reconstruct model and extract residuals
                ro.r("local({ cl$data <- d; eval(cl) })")
                ro.r("r_model <- local({ cl$data <- d; eval(cl) })")
                residuals = np.array(ro.r("residuals(r_model)"))

                return residuals

    except Exception as e:
        print(f"⚠️ Failed to extract residuals: {e}")
        return None


def calculate_partial_residuals(
    model: "metafor.MetaforModel", moderator: str
) -> pd.DataFrame:
    """Calculate partial residuals for a given moderator.

    Partial residuals show the relationship between a moderator and the outcome
    after accounting for all other variables in the model.

    Formula: partial_residuals = residuals(reduced_model) + β_moderator × moderator_values
    """

    try:
        # Use existing coefficient names if available
        coef_names = getattr(model, "coefficient_names", [])
        coefficients = model.coefficients.flatten()

        if len(coef_names) != len(coefficients):
            raise ValueError("Length of coefficient names and values don't match")

        # Find ALL coefficients related to the moderator of interest
        moderator_coefficients = []
        moderator_terms = []

        for i, name in enumerate(coef_names):
            # Check if this coefficient involves the moderator
            if coefficient_involves_moderator(name, moderator):
                moderator_coefficients.append(float(coefficients[i]))
                moderator_terms.append(name)

        if not moderator_coefficients:
            raise ValueError(
                f"Could not find any coefficients for moderator '{moderator}'"
            )
        # create a reduced formula by removing the moderator term
        reduced_formula = create_reduced_formula(model, moderator)

        # create a new adapter instance for the reduced model
        reduced_model = fit_reduced_model(model, model.df_processed, reduced_formula)

        if reduced_model is None:
            raise ValueError("Failed to fit reduced model")

        ord_residuals = extract_residuals_from_model(reduced_model)
        if ord_residuals is None:
            raise ValueError("Failed to extract residuals from reduced model")
        model.ord_residuals = ord_residuals

        # calculate partial residual contribution from relevant moderators
        moderator_contribution = calculate_moderator_contribution(
            moderator,
            moderator_terms,
            moderator_coefficients,
            model.xi,
            debug=model.verbose,  # Enable debug output
        )  # TODO: this is janky (shouldn't be dependent on df or df_subset). There should also be the same number of partial residuals as datapoints in the reduced df
        partial_residuals_y = ord_residuals + moderator_contribution
        partial_residuals_x = (
            model.xi
        )  # Use the same x-values as the model, not the raw data

        if model.verbose:
            print(f"✅ Calculated partial residuals for {moderator}")
            print(f"   Full formula: {model.formula}")
            print(f"   Reduced formula: {reduced_formula}")
            print(f"   Moderator terms: {len(moderator_terms)} terms found")
            for term, coef in zip(moderator_terms, moderator_coefficients):
                print(f"     {term}: {coef:.4f}")
            print(
                f"   Moderator contribution range: {np.min(moderator_contribution):.3f} to {np.max(moderator_contribution):.3f}"
            )
            print(
                f"   Reduced residuals range: {np.min(ord_residuals):.3f} to {np.max(ord_residuals):.3f}"
            )
            print(
                f"   Moderator values range: {np.min(model.xi):.3f} to {np.max(model.xi):.3f}"
            )
            print(
                f"   Partial residuals range: {np.min(partial_residuals_y):.3f} to {np.max(partial_residuals_y):.3f}"
            )
            print(
                f"   Original data range: {np.min(model.yi):.3f} to {np.max(model.yi):.3f}"
            )

        return partial_residuals_x, partial_residuals_y
    except Exception as e:
        print(f"⚠️ Failed to calculate partial residuals: {e}")
        return None


def calculate_moderator_contribution(
    moderator_name: str,
    moderator_terms: list[str],
    moderator_coefficients: list[float],
    moderator_values: np.ndarray,
    debug: bool = False,
) -> np.ndarray:
    """Calculate the total contribution of all moderator terms to the predictions."""
    total_contribution = np.zeros_like(moderator_values)

    for term, coef in zip(moderator_terms, moderator_coefficients):
        # Calculate the transformed values for this term
        transformed_values = evaluate_moderator_term(
            term, moderator_name, moderator_values
        )

        # Add this term's contribution
        contribution = coef * transformed_values
        total_contribution += contribution

        if debug:
            print(
                f"     Term '{term}': coef={coef:.4f}, contribution range={np.min(contribution):.3f} to {np.max(contribution):.3f}"
            )

    return total_contribution


def fit_reduced_model(
    model: "metafor.MetaforModel", df: pd.DataFrame, reduced_formula: str
):
    """Fit a reduced model using the existing adapter infrastructure."""
    try:
        # create identical model but with reduced formula
        reduced_model = metafor.MetaforModel(
            df=df,
            effect_type=model.effect_type,
            effect_type_var=model.effect_type_var,
            treatment=model.treatment,
            formula=reduced_formula,
            random=model.random,
            required_columns=model.required_columns,
            process_data=False,  # necessary since otherwise sometimes cooks removes newfound outliers. TODO: further explore via sensitivity analysis
            verbose=False,  # suppress verbose output for reduced model
        )
        reduced_model.fit_model()

        return reduced_model

    except Exception as e:
        print(f"⚠️ Failed to fit reduced model: {e}")
        return None


# def _create_regression_plot_matplotlib(
#     df, moderator, effect_col, var_col, x_range, predictions, title, **kwargs
# ):
#     """Create meta-regression plot using matplotlib."""
#     figsize = kwargs.get("figsize", (10, 6))
#     fig, ax = plt.subplots(figsize=figsize)

#     # Add confidence intervals if available
#     if predictions and "ci_lower" in predictions:
#         ax.fill_between(
#             x_range,
#             predictions["ci_lower"],
#             predictions["ci_upper"],
#             alpha=0.5,
#             color="lightblue",
#             label="95% Confidence Interval",
#         )

#     # Add regression line if available
#     if predictions and "prediction" in predictions:
#         ax.plot(
#             x_range,
#             predictions["prediction"],
#             "b-",
#             linewidth=3,
#             label="Regression Line",
#         )

#     # Calculate point sizes based on precision
#     se_vals = np.sqrt(df[var_col])
#     max_se = np.max(se_vals)
#     point_sizes = (1 / se_vals) / (1 / max_se) * 100 + 20

#     ax.scatter(
#         df[moderator],
#         df[effect_col],
#         s=point_sizes,
#         alpha=0.7,
#         color="white",
#         edgecolors="navy",
#         linewidths=2,
#     )

#     ax.set_xlabel(moderator.replace("_", " ").title())
#     ax.set_ylabel(effect_col.replace("_", " ").title())
#     ax.set_title(title)
#     ax.grid(True, alpha=0.3)
#     ax.legend()

#     plt.tight_layout()
#     return fig, ax


# def _create_funnel_plot_matplotlib(
#     df, effect_col, precision_metric, title, effect_label, **kwargs
# ):
#     """Create funnel plot using matplotlib."""
#     figsize = kwargs.get("figsize", (8, 8))
#     fig, ax = plt.subplots(figsize=figsize)

#     ax.scatter(
#         df[effect_col],
#         df[precision_metric],
#         alpha=0.7,
#         s=60,
#         color="lightblue",
#         edgecolors="navy",
#     )

#     try:
#         overall_effect = np.average(df[effect_col], weights=1 / df["se"] ** 2)
#         ax.axvline(
#             x=overall_effect, color="red", linestyle="--", label="Overall Effect"
#         )
#     except Exception:
#         pass

#     ax.set_xlabel(effect_label)
#     ax.set_ylabel(precision_metric.upper())
#     ax.set_title(title)
#     ax.grid(True, alpha=0.3)

#     if precision_metric in ["se", "vi"]:
#         ax.invert_yaxis()

#     plt.tight_layout()
#     return fig, ax


# def _create_forest_plot_matplotlib(df, effect_col, title, effect_label, **kwargs):
#     """Create forest plot using matplotlib."""
#     figsize = kwargs.get("figsize", (10, 8))
#     fig, ax = plt.subplots(figsize=figsize)

#     y_pos = np.arange(len(df))

#     ax.errorbar(
#         df[effect_col],
#         y_pos,
#         xerr=[df[effect_col] - df["ci_lower"], df["ci_upper"] - df[effect_col]],
#         fmt="o",
#         capsize=5,
#         capthick=2,
#         markersize=8,
#         color="blue",
#         ecolor="lightblue",
#     )

#     ax.axvline(x=0, color="red", linestyle="--", alpha=0.7)
#     ax.set_yticks(y_pos)
#     ax.set_yticklabels(df["study_label"])
#     ax.set_xlabel(effect_label)
#     ax.set_ylabel("Studies")
#     ax.set_title(title)
#     ax.grid(True, alpha=0.3)

#     plt.tight_layout()
#     return fig, ax


# def _create_contour_plot_matplotlib(X1, X2, predictions, mod1, mod2, title, **kwargs):
#     """Create contour plot using matplotlib."""
#     figsize = kwargs.get("figsize", (8, 6))
#     fig, ax = plt.subplots(figsize=figsize)

#     contour = ax.contour(X1, X2, predictions, levels=15, colors="black", alpha=0.6)
#     contourf = ax.contourf(X1, X2, predictions, levels=15, cmap="viridis", alpha=0.8)

#     ax.clabel(contour, inline=True, fontsize=8)
#     fig.colorbar(contourf, ax=ax, label="Predicted Effect Size")

#     ax.set_xlabel(mod1.replace("_", " ").title())
#     ax.set_ylabel(mod2.replace("_", " ").title())
#     ax.set_title(title)
#     ax.grid(True, alpha=0.3)

#     plt.tight_layout()
#     return fig, ax
