# general
from dataclasses import dataclass
from typing import List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rpy2.robjects as ro
import seaborn as sns
from matplotlib.colorbar import ColorbarBase
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from rpy2.robjects.packages import importr

# stats
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    RBF,
    ConstantKernel,
    WhiteKernel,
)

# custom
from calcification.analysis import analysis, analysis_utils, meta_regression
from calcification.plotting import plot_config, plot_utils
from calcification.processing import climatology as climatology_processing
from calcification.utils import config, utils

# R
metafor = importr("metafor")
grdevices = importr("grDevices")


@dataclass
class MetaRegressionConfig:
    color: str = "#1f77b4"
    ci_color: str = "#aec7e8"
    figsize: Tuple[int, int] = (8, 6)
    dpi: int = 150
    title: str = None
    xlabel: str = "Predictor"
    ylabel: str = "Effect size"
    legend_loc: str = "best"
    point_size: str = "seinv"
    point_border_colours: str | list[str] = "black"
    point_fill_colours: str | list[str] = "white"
    colorby: list = None
    line_color: str = "blue"
    ci_color: str = "lightblue"
    ci_line_color: str = "blue"
    pi_color: str = "lightblue"
    pi_line_color: str = "blue"
    point_alpha: float = 0.8
    refline: float = 0
    full_legend: bool = False
    legend_fontsize: int = 12
    ylimits: tuple[float, float] = None
    add_climatology_lines: bool = True


class MetaRegressionPlotter:
    def __init__(
        self,
        model: ro.vectors.ListVector,
        x_axis_moderator: str,
        fig: plt.Figure = None,
        ax: plt.Axes = None,
        confidence_level_val: float = 0.95,
        weight_pt_by: str = "seinv",
        prediction_limits: tuple[float, float] = None,
        config: Optional[MetaRegressionConfig] = None,
        future_global_anomaly_df: pd.DataFrame = None,
    ):
        self.model_results = MetaRegressionResults(
            model, x_axis_moderator, prediction_limits
        )
        self.config = config or MetaRegressionConfig()
        self.model_results.get_plotting_values()  # get the data for plotting
        self.fig = fig
        self.ax = ax
        self.future_global_anomaly_df = future_global_anomaly_df

    def plot(
        self,
    ):
        """Plot meta-regression results with confidence intervals."""

        fig, ax = self._define_axes()

        # plot scatter points
        point_border_colours, point_fill_colours = self._get_point_colours()
        point_weights = self._get_point_weights()
        sorted_indices = np.argsort(-point_weights)  # larger points plot at the back
        for i in sorted_indices:
            ax.scatter(
                self.model_results.xi[i],
                self.model_results.yi[i],
                s=point_weights[i] ** 2,
                edgecolor=point_border_colours[i],
                facecolor=point_fill_colours[i],
                zorder=3,
                alpha=self.config.point_alpha,
            )

        # plot regression line
        regression_line = self._plot_line(
            ax,
            self.model_results.xs,
            self.model_results.pred,
            color=self.config.color,
            label=f"Meta-regression: effect ~ {self.model_results.polynomial_string}",
            style="-",
            linewidth=2,
        )
        self._plot_confidence_interval(
            ax,
            self.model_results.xs,
            self.model_results.ci_lb,
            self.model_results.ci_ub,
            color=self.config.ci_color,
            ci_line_color=self.config.ci_line_color,
        )
        self._plot_prediction_interval(
            ax,
            self.model_results.xs,
            self.model_results.pred_lb,
            self.model_results.pred_ub,
            color=self.config.pi_color,
            ci_line_color=self.config.pi_line_color,
        )
        # plot dummy data for legend
        ax.scatter(
            [],
            [],
            edgecolor=self.config.point_border_colours[0],
            facecolor=self.config.point_fill_colours[0],
            alpha=self.config.point_alpha,
            label="Study points",
        )  # points
        ax.plot(
            [],
            [],
            linestyle="--",
            color=self.config.ci_line_color,
            label="95% Confidence interval",
        )
        ax.plot(
            [],
            [],
            linestyle=":",
            color=self.config.ci_line_color,
            label="95% Prediction interval",
        )
        # format legend
        self._plot_legend(ax, regression_line)
        self._format_axes(ax)
        self._format_fig(fig)
        ax.legend(loc=self.config.legend_loc)
        self._add_climatology_lines_to_plot(
            ax, self.future_global_anomaly_df
        ) if self.future_global_anomaly_df is not None else None
        return fig, ax

    def _define_axes(self):
        if self.ax is None:
            fig, ax = plt.subplots(figsize=self.config.figsize, dpi=self.config.dpi)
        else:
            fig = self.ax.figure
            ax = self.ax
        return fig, ax

    def _format_axes(self, ax):
        ax.set_xlabel(self.model_results.regression_model_str)
        ax.set_ylabel(self.config.ylabel)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.axhline(
            y=0, color="gray", linestyle="-", linewidth=1, label="Zero effect level"
        ) if self.config.refline is not None else None
        ax.set_xlim(
            self.model_results.prediction_limits
        ) if self.model_results.prediction_limits else None
        ax.set_ylim(
            self.model_results.prediction_limits
        ) if self.config.ylimits else None

    def _plot_legend(self, ax, regression_line):
        ax.legend(loc=self.config.legend_loc, fontsize=self.config.legend_fontsize)
        # TODO: clarify what this actually does
        if self.config.full_legend:
            if self.config.colorby is not None:
                # if colorby represents discrete values (not continuous)
                if len(self.unique_colors) < 5:
                    legend_elements = []
                    for color in self.unique_colors:
                        legend_elements.append(
                            Line2D(
                                [0],
                                [0],
                                marker="o",
                                color="w",
                                label=color,
                                markerfacecolor=self.color_map[color],
                                markersize=10,
                            )
                        )
                    ax.legend(legend_elements, fontsize=self.config.legend_fontsize)
        else:
            ax.legend(
                [regression_line[0]],
                [f"Meta-regression: effect ~ {self.model_results.polynomial_string}"],
            )

    def _plot_line(self, ax, x, y, color, label, style, linewidth):
        return ax.plot(
            x, y, color=color, label=label, linestyle=style, linewidth=linewidth
        )

    def _format_fig(self, fig):
        fig.suptitle(self.config.title, fontsize=14)
        plt.tight_layout(rect=[0, 0, 1, 0.95])

    def _plot_confidence_interval(
        self, ax, xs, ci_lb, ci_ub, color, ci_line_color, label=None
    ):
        ax.fill_between(xs, ci_lb, ci_ub, color=color, alpha=0.3)
        for bound in (ci_lb, ci_ub):
            self._plot_line(
                ax, xs, bound, color=ci_line_color, style="--", linewidth=1, label=label
            )

    def _plot_prediction_interval(
        self, ax, xs, pred_lb, pred_ub, color, ci_line_color, label=None
    ):
        ax.fill_between(xs, pred_lb, pred_ub, color=color, alpha=0.1)
        for bound in (pred_lb, pred_ub):
            self._plot_line(
                ax, xs, bound, color=ci_line_color, style=":", linewidth=1, label=label
            )

    def _get_point_weights(self):
        return meta_regression._compute_point_weights(
            self.model_results.vi, self.config.point_size
        )

    def _get_point_colours(self):
        # point border colours
        if isinstance(self.config.point_border_colours, str):
            point_border_colours = [self.config.point_border_colours] * len(
                self.model_results.xi
            )
        elif isinstance(self.config.point_border_colours, list):
            point_border_colours = self.config.point_border_colours
        else:
            raise ValueError("point_border_colours must be a string or list of strings")

        # point fill colours
        if isinstance(self.config.point_fill_colours, str):
            point_fill_colours = [self.config.point_fill_colours] * len(
                self.model_results.xi
            )
        elif isinstance(self.config.point_fill_colours, LinearSegmentedColormap):
            point_fill_colours = self.config.point_fill_colours
        elif isinstance(self.config.point_fill_colours, list):
            # create a color map for the points
            colorby = np.array(self.config.colorby)
            self.unique_colors = np.unique(colorby)
            self.color_map = {
                color: plt.cm.viridis(i / len(self.unique_colors))
                for i, color in enumerate(self.unique_colors)
            }
            point_fill_colours = [self.color_map[color] for color in colorby]
        else:
            raise ValueError(
                "point_fill_colours must be a string, a numpy array, or LinearSegmentedColormap object"
            )
        return point_border_colours, point_fill_colours

    def _get_anomaly_variable_from_moderator(self):
        if self.model_results.moderator == "delta_ph":
            return "ph"
        elif self.model_results.moderator == "delta_t":
            return "sst"
        else:
            raise ValueError(f"Moderator {self.model_results.moderator} not supported")

    def _add_climatology_lines_to_plot(
        self,
        ax: matplotlib.axes.Axes,
        future_global_anomaly_df: pd.DataFrame,
        # scenario_variable: str,
    ) -> matplotlib.axes.Axes:
        """
        Add climatology lines to the plot for different scenarios.

        Args:
            ax (matplotlib.axes.Axes): The axis to add the lines to.
            future_global_anomaly_df (pd.DataFrame): DataFrame containing future climate scenario data. Must have scenario, time_frame, and mean_{scenario_variable}_anomaly columns.

        Returns:
            matplotlib.axes.Axes: The axis with the added lines.
        """
        if self.future_global_anomaly_df is None:
            raise ValueError(
                "future_global_anomaly_df argument is required to add climatology lines"
            )
        # determine variable from moderator
        scenario_variable = self._get_anomaly_variable_from_moderator()

        scenarios = future_global_anomaly_df["scenario"].unique()
        scenario_colours = sns.color_palette("Reds", len(scenarios))
        scenario_colour_dict = {
            scenario: scenario_colours[i] for i, scenario in enumerate(scenarios)
        }
        original_ylim = ax.get_ylim()  # get the original y-axis limits
        scenario_lines = []
        for scenario in scenarios:
            # add climatology lines to the plot
            predicted_effect_sizes = future_global_anomaly_df[
                (
                    future_global_anomaly_df["time_frame"] == 2090
                )  # TODO: extrapolate to 2100 and make dynamic
                & (future_global_anomaly_df["scenario"] == scenario)
            ][:][f"mean_{scenario_variable}_anomaly"]

            # plot vertical lines for each predicted effect size
            for effect_size in predicted_effect_sizes:
                line = ax.vlines(
                    x=effect_size,
                    ymin=original_ylim[0],
                    ymax=original_ylim[1],
                    color=scenario_colour_dict[scenario],
                    linestyle="--",
                    label=plot_config.SCENARIO_MAP[scenario],
                    zorder=5,
                )
                scenario_lines.append(line)
        ax.set_ylim(original_ylim[0], original_ylim[1])  # crop to y lim
        return ax


def plot_multiple_metaregression_axes(
    model: ro.vectors.ListVector,
    x_axis_moderators: list[str],
    future_global_anomaly_df: pd.DataFrame = None,
    annotate_axes: bool = True,
    annotate_axes_fontsize: int = 16,
) -> tuple[plt.Figure, plt.Axes]:
    """Wrapper for MetaRegressionPlotter to plot multiple axes on the same figure."""
    fig, ax = plt.subplots(1, len(x_axis_moderators), figsize=(12, 6), dpi=150)
    for axis_index, x_axis_moderator in enumerate(x_axis_moderators):
        MetaRegressionPlotter(
            ax=ax[axis_index],
            model=model,
            x_axis_moderator=x_axis_moderator,
            future_global_anomaly_df=future_global_anomaly_df,
        ).plot()
    if annotate_axes:
        plot_utils.annotate_axes_with_letters(ax, fontsize=annotate_axes_fontsize)
    return fig, ax


class MetaRegressionResults:
    def __init__(
        self,
        model: ro.vectors.ListVector,
        moderator: str = None,
        prediction_limits: tuple[float, float] = None,
        confidence_level: float = 0.95,
    ):
        self.model = model
        self.moderator = moderator
        self.prediction_limits = prediction_limits
        self.confidence_level_val = confidence_level
        self.get_model_summary()  # moderator-agnostic methods
        self.moderator_index = (
            analysis_utils.get_moderator_index(self.model, self.moderator)
            if self.moderator is not None
            else None
        )

    # --- Moderator agnostic methods ---

    def get_model_summary(self) -> None:
        self.coeffs_vals = self._get_coeffs_vals()
        self.moderator_names = analysis_utils.get_moderator_names(self.model)
        self.moderator_stats = self._get_moderator_stats()
        self.fit_method = str(self.model.rx2("method")[0])
        self.model_metadata = self._get_model_metadata()
        self.headline_stats = self._get_headline_stats()
        self.test_stats = self._get_test_stats()

    def _get_coeffs_vals(self):
        return np.array(self.model.rx2("b"))

    def _get_moderator_stats(self) -> pd.DataFrame:
        """Get the stats for each moderator from the model."""
        # table of estimate, se, zval, pval, ci.lb, ci.ub, prediction intervals?
        stats_df = pd.DataFrame()
        for stat_name in ["beta", "se", "zval", "pval", "ci.lb", "ci.ub"]:
            stats_df[stat_name] = np.array(self.model.rx2(stat_name)).squeeze()
        stats_df["moderator"] = self.moderator_names
        stats_df = stats_df.set_index("moderator")
        return stats_df

    def _get_model_metadata(self) -> pd.DataFrame:
        # k, fit method
        metadata_df = pd.DataFrame()
        for meta_name in ["k", "method"]:
            metadata_df[meta_name] = self.model.rx2(meta_name)
        return metadata_df

    def _get_test_stats(self) -> pd.DataFrame:
        """Get the test stats from the model."""
        test_df = pd.DataFrame()
        for test_name in ["QE", "QEp", "QM", "QMp"]:
            test_df[test_name] = self.model.rx2(test_name)
        return test_df

    def _get_headline_stats(self):
        """Get the headline stats from the model."""
        headline_vars = ["logLik", "Deviance", "AIC", "BIC", "AICc"]
        headline_vars_df = pd.DataFrame(self.model.rx2("fit.stats"))
        if self.fit_method == "REML":
            headline_vars_df = headline_vars_df.iloc[1]
        elif self.fit_method == "ML":
            headline_vars_df = headline_vars_df.iloc[0]
        else:
            raise ValueError(f"Unknown fit method: {self.fit_method}")
        headline_vars_df.columns = headline_vars
        return headline_vars_df

    # --- Moderator specific methods ---
    def _get_regression_model_str(self):
        # TODO: automate this better
        return "$\\Delta$ pH" if self.moderator == "delta_ph" else "$\\Delta$ T"

    def _get_polynomial_string(self):
        return (
            plot_utils.Polynomial(
                np.concatenate(
                    [self.coeffs_vals[self.moderator_index], self.coeffs_vals[0]]
                )
            )
            .__str__()
            .replace("x", self.regression_model_str)
        )

    def _get_basic_model_data_for_moderator(self):
        self.xi, self.yi, self.vi = meta_regression._extract_model_components(
            self.model, self.moderator
        )
        return self.xi, self.yi, self.vi

    def _get_prediction_values(
        self,
        xs: np.ndarray = None,
        prediction_limits: tuple[float, float] = None,
    ):
        if xs is None:
            self.xs, self.prediction_limits = (
                meta_regression._get_xs_and_prediction_limits(
                    self.xi, prediction_limits
                )
            )
        else:
            self.xs, self.prediction_limits = xs, prediction_limits
        # get xs for plotting regression line

        self.pred, self.se, self.ci_lb, self.ci_ub, self.pred_lb, self.pred_ub = (
            meta_regression.metafor_predict_from_model(
                self.model,
                self.moderator,
                self.xs,
                self.confidence_level_val,
            )
        )

    def get_plotting_values(self):
        self.coeffs_vals = self._get_coeffs_vals()
        self.regression_model_str = self._get_regression_model_str()
        self.polynomial_string = self._get_polynomial_string()
        self._get_basic_model_data_for_moderator()
        self._get_prediction_values(self.prediction_limits)


def plot_model_surface_2d(
    model: meta_regression.MetaforModel,
    moderator_names: list[str],
    prediction_limits: dict[str, tuple[float, float]] = None,
    num_points: int = 100,
):
    """Plot a surface of the model."""
    # check that len(moderator_names) == 2
    if len(moderator_names) != 2:
        raise ValueError("plot_model_surface only supports 2 moderators")

    # generate xs, ys from prediction_limits and num_points
    xs = np.linspace(*sorted(prediction_limits[moderator_names[0]]), num_points)
    ys = np.linspace(*sorted(prediction_limits[moderator_names[1]]), num_points)
    xs, ys = np.meshgrid(xs, ys)

    # get prediction surface
    pred_surface, meshgrids = meta_regression.predict_nd_surface_from_model(
        model,
        moderator_names,
        [xs, ys],
    )

    # get absolute max of pred_surface
    abs_max = np.abs(pred_surface).max()

    # plot
    fig, ax = plt.subplots(figsize=(10, 8))
    contour_ax = ax.contourf(
        meshgrids[0],
        meshgrids[1],
        pred_surface,
        levels=50,
        cmap="coolwarm_r",
        vmin=-abs_max,
        vmax=abs_max,
    )
    # format
    ax.set_xlabel(moderator_names[0])
    ax.set_ylabel(moderator_names[1])
    cbar = plt.colorbar(contour_ax, orientation="horizontal", fraction=0.05, shrink=0.8)
    cbar.set_label(label="Relative calcification ($\\Delta$%)", size="large")
    return fig, ax


def plot_funnel_from_model(
    model,
    main: str = None,
    effect_type: str = "Effect Size",
    shade_colors: list[str] = ["white", "gray55", "gray75"],
    back_color: str = "gray90",
    level: list[float] = [0.1, 0.05, 0.01],
    legend: bool = True,
    hlines: list[float] = None,
    yaxis: str = "sei",
    digits: list[int] = [1, 3],
    las: int = 1,
    xaxs: str = "i",
    yaxs: str = "i",
    xlim: list = None,
    ylim: list = [0, 10],  # this range is ridiculous, adjust accordingly
    plot_in_python: bool = False,
    figsize: tuple[int, int] = (10, 8),
    save_path: [str] = None,
) -> None | tuple[plt.Figure, plt.Axes]:
    """
    Create a funnel plot from an existing metafor model object.

    Parameters
    ----------
    model : rpy2.robjects.vectors.ListVector
        A fitted metafor model object (from rma, rma.uni, etc.)
    main : str,
        Main title for the plot
    effect_type : str,
        Label for x-axis
    shade_colors : List[str],
        Colors for confidence contours (from innermost to outermost)
    back_color : str,
        Background color for the plot
    level : List[float],
        Confidence levels for contours
    legend : bool,
        Whether to include a legend
    hlines : List[float],
        Y-positions for horizontal reference lines
    yaxis : str,
        Y-axis scale ('se', 'vi', 'seinv', 'vinv')
    digits : List[int],
        Number of digits for axis values
    las : int,
        Orientation of axis labels (0-3)
    xaxs : str,
        X-axis style ('r' for regular, 'i' for internal)
    yaxs : str,
        Y-axis style ('r' for regular, 'i' for internal)
    plot_in_python : bool,
        Whether to capture the R plot and display it in Python
    figsize : tuple,
        Figure size for matplotlib plot (if plot_in_python=True)
    save_path : str,
        Path to save the plot (if plot_in_python=True)

    Returns
    -------
    None or Tuple[plt.Figure, plt.Axes]
        If plot_in_python=True, returns the matplotlib figure and axes

    Notes
    -----
    This function requires that you have already fitted a metafor model
    using rpy2. The model should be created with metafor's rma() or similar
    functions before passing to this function.
    """
    # Convert Python lists to R vectors where needed
    r_shade = ro.StrVector(shade_colors)
    r_level = ro.FloatVector(level)
    r_digits = ro.IntVector(digits)

    # Handle horizontal lines
    r_hlines = ro.NULL if hlines is None else ro.FloatVector(hlines)

    if plot_in_python:
        # Create a temporary file for the R plot
        import os
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
            temp_filename = temp_file.name

        # Create the plot in R and save to file with high resolution
        # Multiply dimensions by 3 and use high DPI for better quality
        grdevices.png(
            temp_filename, width=figsize[0], height=figsize[1], units="in", res=300
        )

        # Call metafor's funnel function with the provided parameters
        metafor.funnel(
            x=model,
            main=ro.NULL if main is None else main,
            shade=r_shade,
            back=back_color,
            level=r_level,
            legend=legend,
            hlines=r_hlines,
            xlab=effect_type,
            yaxis=yaxis,
            las=las,
            digits=r_digits,
            xaxs=xaxs,
            yaxs=yaxs,
            xlim=ro.NULL if xlim is None else ro.FloatVector(xlim),
            # ylim=ro.NULL if ylim is None else ro.FloatVector(ylim)
        )

        grdevices.dev_off()

        # Display the plot in Python
        fig, ax = plt.subplots(figsize=figsize, dpi=300)
        img = plt.imread(temp_filename)
        ax.imshow(
            img, interpolation="nearest"
        )  # Use 'nearest' for sharper image rendering
        ax.axis("off")
        plt.tight_layout(pad=0)  # Reduce padding

        # Save if requested
        if save_path:
            plt.savefig(save_path, bbox_inches="tight", dpi=300)

        # Clean up temporary file
        os.unlink(temp_filename)

        return fig, ax
    else:
        # Call metafor's funnel function with the provided parameters
        metafor.funnel(
            x=model,
            main=ro.NULL if main is None else main,
            shade=r_shade,
            back=back_color,
            level=r_level,
            legend=legend,
            hlines=r_hlines,
            xlab=effect_type,
            yaxis=yaxis,
            las=las,
            digits=r_digits,
            xaxs=xaxs,
            # yaxs=yaxs
        )
        # xlim=ro.NULL if xlim is None else ro.FloatVector(xlim),
        # save the plot to a file if save_path is provided
        if save_path:
            # Use higher resolution settings
            grdevices.png(
                str(save_path), width=figsize[0], height=figsize[1], units="in", res=300
            )
            metafor.funnel(
                x=model,
                main=ro.NULL if main is None else main,
                shade=r_shade,
                back=back_color,
                level=r_level,
                legend=legend,
                hlines=r_hlines,
                xlab=effect_type,
                yaxis=yaxis,
                las=las,
                digits=r_digits,
                xaxs=xaxs,
                yaxs=yaxs,
                xlim=ro.NULL if xlim is None else ro.FloatVector(xlim),
                # ylim=ro.NULL if ylim is None else ro.FloatVector(ylim)
            )
            grdevices.dev_off()
            plt.close()

        return None


def plot_contour(ax, x, y, df, title, legend_label="Calcification Rate"):
    # Create a grid of points
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    for i in range(len(X)):  # interpolate the data to create a smooth contour plot
        for j in range(len(Y)):
            # Find the nearest data point
            nearest_idx = (
                (df["temp"] - X[i, j]) ** 2 + (df["phtot"] - Y[i, j]) ** 2
            ).idxmin()
            Z[i, j] = df.loc[nearest_idx, "st_calcification"]

    # Plot the contour
    contour = ax.contourf(X, Y, Z, levels=20, cmap="viridis")
    ax.set_xlabel("Temperature ($^\\circ C$)")
    ax.set_ylabel("pH$_T$")
    ax.set_title(title)
    plt.colorbar(contour, label=legend_label)


def plot_contour_gp(
    ax: matplotlib.axes.Axes,
    x: np.ndarray,
    y: np.ndarray,
    df: pd.DataFrame,
    title: str,
    legend_label: str = "Calcification Rate",
) -> None:
    """
    Create a smooth contour plot using Gaussian Process regression

    Args:
        ax (matplotlib.axes.Axes): The axes to plot on
        x (np.ndarray): The x-coordinates for the grid
        y (np.ndarray): The y-coordinates for the grid
        df (pd.DataFrame): DataFrame containing 'temp', 'phtot', and 'st_calcification' columns
        title (str): The title for the plot
        legend_label (str, optional): The label for the colorbar
    """
    X_train = df[["temp", "phtot"]].values
    y_train = df["st_calcification"].values

    # define the kernel - variable-independent RBFs with noise
    k1 = ConstantKernel(1.0) * RBF(
        length_scale=2.5, length_scale_bounds=(0.1, 10.0)
    )  # for temperature
    k2 = ConstantKernel(1.0) * RBF(
        length_scale=0.1, length_scale_bounds=(0.01, 0.5)
    )  # for pH
    kernel = k1 + k2 + WhiteKernel(noise_level=0.1)

    # fit GP
    gp = GaussianProcessRegressor(
        kernel=kernel,
        n_restarts_optimizer=10,
        alpha=np.asarray(df["st_calcification_sd"]) ** 2,
    )
    gp.fit(X_train, y_train)

    # create prediction grid and predict
    X, Y = np.meshgrid(x, y)
    grid_points = np.column_stack([X.ravel(), Y.ravel()])
    Z = gp.predict(grid_points).reshape(X.shape)

    # plot the contour
    contour = ax.contourf(X, Y, Z, levels=20, cmap="viridis")
    ax.set_xlabel("Temperature ($^\\circ C$)")
    ax.set_ylabel("pH$_T$")
    ax.set_title(title)
    ax.scatter(
        df["temp"],
        df["phtot"],
        c="white",
        s=10,
        alpha=0.6,
        edgecolors="black",
        linewidths=0.5,
    )

    # format
    plt.colorbar(contour, ax=ax, label=legend_label)


def plot_global_timeseries_grid(
    groups: dict[str, pd.DataFrame], figsize=(16, 10), dpi=300, title_org: str = None
):
    """
    Plot global timeseries in a grid layout with SSPs vertically and group types horizontally.

    Parameters:
        groups: dict mapping group names (str) to DataFrames (pd.DataFrame)
    """
    # Extract scenarios and group names
    scenarios = next(iter(groups.values()))["scenario"].unique()
    group_names = list(groups.keys())

    # Define colors and formatting
    historic_colour, historic_alpha = "darkgrey", 0.5
    forecast_alpha, zero_effect_colour, zero_effect_alpha = 0.3, "black", 0.5
    # group_colours = sns.color_palette("tab10", len(groups))  # Different color per group
    # group_colour_dict = {group: group_colours[i] for i, group in enumerate(groups)}
    scenario_colours = sns.color_palette("Reds", len(scenarios))
    scenario_colour_dict = {
        scenario: scenario_colours[i] for i, scenario in enumerate(scenarios)
    }

    x_points = next(iter(groups.values()))["time_frame"].unique()
    time_discontinuity = 2025  # present day

    # Create grid layout: rows are SSPs, columns are group types
    fig, axes = plt.subplots(
        len(scenarios),
        len(groups),
        figsize=figsize,
        sharex=False,
        sharey=False,
        dpi=dpi,
    )

    # Global min/max for y-axis synchronization
    global_min_ylim, global_max_ylim = float("inf"), float("-inf")

    # First pass to calculate global y-limits
    for group_name, df in groups.items():
        for scenario in scenarios:
            scenario_df = df[df["scenario"] == scenario]
            means_df = scenario_df[scenario_df["percentile"] == "mean"]
            min_val = min(
                means_df["ci.lb"].min() if not means_df.empty else 0, 0
            )  # Include 0
            max_val = max(
                means_df["ci.ub"].max() if not means_df.empty else 0, 0
            )  # Include 0

            global_min_ylim = min(global_min_ylim, min_val)
            global_max_ylim = max(global_max_ylim, max_val)

    # Add some padding to y-limits
    y_padding = 0.1 * (global_max_ylim - global_min_ylim)
    global_min_ylim -= y_padding
    global_max_ylim += y_padding

    # Plot each cell in the grid
    for i, scenario in enumerate(scenarios):
        for j, group_name in enumerate(group_names):
            ax = (
                axes[i, j]
                if len(scenarios) > 1 and len(group_names) > 1
                else axes[i]
                if len(group_names) == 1
                else axes[j]
                if len(scenarios) == 1
                else axes
            )

            df = groups[group_name]
            scenario_df = df[df["scenario"] == scenario]

            if not scenario_df.empty:
                means_df = scenario_df[scenario_df["percentile"] == "mean"]
                x_fine, mean_spline = plot_utils.interpolate_spline(
                    x_points, means_df["pred"]
                )
                x_fine, low_spline = plot_utils.interpolate_spline(
                    x_points, means_df["ci.lb"]
                )
                x_fine, up_spline = plot_utils.interpolate_spline(
                    x_points, means_df["ci.ub"]
                )

                historic_mask = x_fine < time_discontinuity
                forecast_mask = x_fine >= time_discontinuity

                # Historical period
                ax.fill_between(
                    x_fine[historic_mask],
                    low_spline[historic_mask],
                    up_spline[historic_mask],
                    alpha=historic_alpha,
                    color=historic_colour,
                    linewidth=0,
                )
                ax.plot(
                    x_fine[historic_mask],
                    mean_spline[historic_mask],
                    ":",
                    color=scenario_colour_dict[scenario],
                    alpha=0.8,
                )

                # Forecast period
                ax.fill_between(
                    x_fine[forecast_mask],
                    low_spline[forecast_mask],
                    up_spline[forecast_mask],
                    alpha=forecast_alpha,
                    color=scenario_colour_dict[scenario],
                    linewidth=0,
                )
                ax.plot(
                    x_fine[forecast_mask],
                    mean_spline[forecast_mask],
                    "--",
                    linewidth=2,
                    color=scenario_colour_dict[scenario],
                    label=group_name,
                )

                first_negative_index = np.argmax(
                    up_spline < 0
                )  # Find the first index where the mean spline is negative
                if first_negative_index is not None and not np.isnan(
                    first_negative_index
                ):
                    ax.annotate(
                        "",
                        xy=(x_fine[first_negative_index], 0),
                        xytext=(
                            x_fine[first_negative_index],
                            up_spline[first_negative_index] + 2,
                        ),
                        arrowprops=dict(arrowstyle="-|>", color="black", lw=1.5),
                        fontsize=8,
                        ha="center",
                        va="center",
                    )

            # Zero effect line
            ax.axhline(
                y=0,
                color=zero_effect_colour,
                alpha=zero_effect_alpha,
                linestyle="-",
                linewidth=1,
            )

            # Set grid and limits
            ax.grid(ls="--", alpha=0.3)
            ax.set_xlim(2000, 2100)
            ax.set_ylim(-200, global_max_ylim)

            from matplotlib import ticker

            x_tickspacing, y_tickspacing = 50, 100
            ax.xaxis.set_major_locator(ticker.MultipleLocator(x_tickspacing))
            ax.yaxis.set_major_locator(ticker.MultipleLocator(y_tickspacing))

            fig.text(0.5, 0.0, "Year", ha="center", fontsize=10)
            fig.text(
                0.0,
                0.5,
                "Percentage change in calcification rate",
                va="center",
                rotation="vertical",
                fontsize=10,
            )

            # Add group name as title only on first row
            if i == 0:
                ax.set_title(group_name, fontsize=10, color="black")

            # Hide tick labels except for edge cells
            if j > 0:  # Not first column
                ax.set_yticklabels([])
            if i < len(scenarios) - 1:  # Not last row
                ax.set_xticklabels([])

            # decrease size of axis tick labels
            ax.tick_params(axis="both", labelsize=8)

    # Add a single legend at the top
    handles = (
        [
            plt.Line2D(
                [],
                [],
                linestyle="-",
                color=zero_effect_colour,
                label="Zero effect",
                alpha=zero_effect_alpha,
            )
        ]
        + [plt.Rectangle((0, 0), 1, 1, color="grey", alpha=0.3, label="95% CI")]
        + [
            plt.Line2D(
                [], [], linestyle=":", color=historic_colour, label="Historical"
            ),
            plt.Line2D([], [], linestyle="--", color=historic_colour, label="Forecast"),
        ]
        + [
            plt.Line2D(
                [],
                [],
                linestyle="--",
                color=scenario_colour_dict[scenario],
                label=plot_config.SCENARIO_MAP[scenario],
            )
            for scenario in scenarios
        ]
    )

    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.98),
        ncol=len(handles),
        fontsize=10,
        frameon=False,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to accommodate suptitle

    return fig, axes


def plot_global_timeseries_multi(
    groups: dict[str, pd.DataFrame], figsize=(10, 10), dpi=300, title_org: str = None
):
    """
    Plot global timeseries for multiple groups on the same axes.

    Parameters:
        groups: dict mapping group names (str) to DataFrames (pd.DataFrame)
    """
    scenarios = next(iter(groups.values()))["scenario"].unique()

    # Define colors and formatting
    historic_colour, historic_alpha = "darkgrey", 0.5
    forecast_alpha, zero_effect_colour, zero_effect_alpha = 0.2, "black", 0.5
    group_colours = sns.color_palette("tab10", len(groups))  # Different color per group
    group_colour_dict = {group: group_colours[i] for i, group in enumerate(groups)}

    x_points = next(iter(groups.values()))["time_frame"].unique()
    time_discontinuity = 2025  # present day

    fig, axes = plt.subplots(len(scenarios), 1, figsize=figsize, sharex=True, dpi=dpi)

    for i, scenario in enumerate(scenarios):
        ax = axes[i]

        for group_name, df in groups.items():
            scenario_df = df[df["scenario"] == scenario]

            means_df = scenario_df[scenario_df["percentile"] == "mean"]
            x_fine, mean_spline = plot_utils.interpolate_spline(
                x_points, means_df["pred"]
            )
            x_fine, up_spline = plot_utils.interpolate_spline(
                x_points, means_df["ci.lb"]
            )
            x_fine, low_spline = plot_utils.interpolate_spline(
                x_points, means_df["ci.ub"]
            )

            historic_mask = x_fine < time_discontinuity
            forecast_mask = x_fine >= time_discontinuity

            ax.plot(
                x_fine[forecast_mask],
                mean_spline[forecast_mask],
                "--",
                alpha=0.8,
                color=group_colour_dict[group_name],
                label=group_name,
            )
            ax.fill_between(
                x_fine[forecast_mask],
                low_spline[forecast_mask],
                up_spline[forecast_mask],
                alpha=forecast_alpha,
                color=group_colour_dict[group_name],
                linewidth=0,
            )
            ax.plot(
                x_fine[historic_mask],
                mean_spline[historic_mask],
                ":",
                color=group_colour_dict[group_name],
                alpha=0.8,
            )
            ax.fill_between(
                x_fine[historic_mask],
                low_spline[historic_mask],
                up_spline[historic_mask],
                alpha=historic_alpha,
                color=historic_colour,
                linewidth=0,
            )

        # Zero effect line
        ax.hlines(
            0,
            xmin=x_fine.min(),
            xmax=x_fine.max(),
            color=zero_effect_colour,
            alpha=zero_effect_alpha,
        )

        # Set labels and grid
        ax.set_ylabel("Relative calcification rate", fontsize=8)
        ax.grid(ls="--", alpha=0.5)

    # Global legend
    handles = [
        plt.Line2D([], [], linestyle=":", color=historic_colour, label="Historical"),
        plt.Line2D(
            [],
            [],
            linestyle="-",
            color=zero_effect_colour,
            label="Zero effect",
            alpha=zero_effect_alpha,
        ),
    ] + [
        plt.Line2D([], [], linestyle="--", color=group_colours[i], label=group)
        for i, group in enumerate(groups)
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.95),
        ncol=len(handles),
        fontsize=8,
        frameon=False,
    )
    # annotate each axis with the scenario name
    for i, ax in enumerate(axes):
        ax.annotate(
            plot_config.SCENARIO_MAP[scenarios[i]],
            xy=(0.95, 0.9),
            xycoords="axes fraction",
            ha="center",
            va="center",
            fontsize=10,
            color="black",
        )
        # ax.set_title(scenario, fontsize=10, fontweight='bold', color=group_colours[i])

    # Synchronize ylims and xlims
    global_min_ylim, global_max_ylim = 0, 0
    for axis in axes:
        ymin, ymax = axis.get_ylim()
        global_min_ylim = min(global_min_ylim, ymin)
        global_max_ylim = max(global_max_ylim, ymax)

    for axis in axes:
        axis.tick_params(axis="both", labelsize=8)
        axis.set_xlim(1995, 2100)
        axis.set_ylim(global_min_ylim, global_max_ylim)

    if len(axes) > 0:
        axes[-1].set_xlabel("Year", fontsize=8)

    plt.suptitle(
        f"{title_org}\nProjected relative calcification rates under different climate scenarios"
        if title_org
        else "Projected relative calcification rates under different climate scenarios",
        fontsize=12,
    )

    return fig, axes


def prepare_emissions_predictions(
    reshaped_preds_df,
    emissions_data,
    scenario="ssp585",
):
    """
    Prepare emissions predictions DataFrame for plotting.

    Args:
        reshaped_preds_df (pd.DataFrame): DataFrame of predictions.
        emissions_data (pd.DataFrame): DataFrame with emissions data.
        scenario (str): Scenario to filter for.

    Returns:
        pd.DataFrame: Merged and processed DataFrame with predictions and emissions.
    """
    plot_preds = climatology_processing.interpolate_and_extrapolate_predictions(
        reshaped_preds_df.copy()
    )
    plot_preds = plot_preds[plot_preds.scenario == scenario]
    emissions = emissions_data[["year", "SSP5-8.5"]].rename(columns={"SSP5-8.5": "co2"})
    emissions_predictions = pd.merge(
        plot_preds, emissions, left_on="time_frame", right_on="year", how="left"
    )
    emissions_predictions.drop(columns="year", inplace=True)
    emissions_predictions.sort_values(by="core_grouping", inplace=True)
    # calculate p-scores and certainty
    emissions_predictions["p_score"] = emissions_predictions.apply(
        lambda row: analysis.p_score(row["pred"], row["se"], null_value=0), axis=1
    )
    emissions_predictions["certainty"] = emissions_predictions["p_score"].apply(
        analysis.assign_certainty
    )
    return emissions_predictions


# --- Burning Embers figures ---


@dataclass
class BurningEmbersConfig:
    insufficient_data_cols: Optional[List[str]] = None
    cmap_colors: Optional[List[str]] = None
    vmin: float = -75  # Changed from 0 to -75
    vmax: float = 0  # Changed from -75 to 0
    n_levels: int = 100
    figsize: Tuple[int, int] = (14, 6)
    dpi: int = 300
    end_year: int = 2100
    title: str = f"Projected impacts of climate change on reef calcifiers in {end_year}"
    forcing_col: str = "anomaly_value_sst"
    emissions_scenario: str = "ssp585"


class BurningEmbersPlotter:
    # TODO: add config, dynamically vary the left and right axis variables
    def __init__(
        self,
        predictions_df: pd.DataFrame = None,
        config: Optional[BurningEmbersConfig] = None,
    ):
        """
        Args:
            config: Optional[BurningEmbersConfig] = None
            predictions_df: pd.DataFrame = None. Must contain scenario, time_frame, core_grouping, pred, se, p_score, and certainty columns.
        """
        if config is None:
            config = BurningEmbersConfig()
        self.config = config
        if self.config.insufficient_data_cols is None:
            self.config.insufficient_data_cols = ["Foraminifera", "Molluscs"]
        if self.config.cmap_colors is None:
            self.config.cmap_colors = ["#ffffff", "#f9cb0f", "#c72529", "#812066"][::-1]
        self.predictions_df = predictions_df
        self.predictions_df = self._prepare_forcing_data()
        self.forcing_vals = self.predictions_df[self.config.forcing_col]

    def _get_cmap(self):
        """Get the colormap in line with IPCC report colors e.g. https://www.ipcc.ch/report/ar6/wg2/figures/chapter-11/figure-11-006."""
        return LinearSegmentedColormap.from_list(
            "burning_embers", self.config.cmap_colors, N=256
        )

    def _get_cnorm(self):
        """Normalise colormap to the range of the data."""
        # TODO: fix this
        # Get the actual range of prediction values from the data
        if self.predictions_df is not None and "pred" in self.predictions_df.columns:
            pred_values = self.predictions_df["pred"].dropna()
            if len(pred_values) > 0:
                data_min = pred_values.min()
                data_max = pred_values.max()
                # Ensure vmin < vmax and handle edge cases
                if data_min == data_max:
                    # If all values are the same, create a small range around the value
                    vmin = data_min - 0.1
                    vmax = data_max + 0.1
                else:
                    vmin = data_min
                    vmax = data_max
                return Normalize(vmin=vmin, vmax=vmax)

        # Fallback to config values if data is not available
        return Normalize(vmin=self.config.vmin, vmax=self.config.vmax)

    def _get_bar_categories(self):
        """Get the categories to plot, combining categories with insufficient data cols."""
        categories = list(self.predictions_df.core_grouping.unique())
        # combine categories with insufficient data cols, removing these
        categories = [
            category
            for category in categories
            if not any(
                subcategory in category
                for subcategory in self.config.insufficient_data_cols
            )
        ]
        insufficient_data_category = "/".join(self.config.insufficient_data_cols)
        # categories = categories + [insufficient_data_category]
        return categories, insufficient_data_category

    def plot(self) -> tuple[plt.Figure, plt.Axes]:
        """Plot the forcing predictions. 'Forcing' here refers to either CO2 concentration or SST anomaly."""
        categories, insufficient_data_category = self._get_bar_categories()
        cmap = self._get_cmap()
        cnorm = self._get_cnorm()

        fig, ax = plt.subplots(figsize=self.config.figsize, dpi=self.config.dpi)
        for i, category in enumerate(categories):
            category_data = self.predictions_df[
                self.predictions_df["core_grouping"] == category
            ].sort_values("time_frame")
            if len(category_data) == 0:
                print(f"No data for category: {category}")
                self.config.insufficient_data_cols.append(category)
                continue
            interp_preds, interp_forcing = self._interpolate_preds(category_data)
            self._draw_gradient_bar(ax, i, interp_preds)
            self._draw_bar_border(ax, i, interp_forcing)
            self._draw_certainty_dots(ax, i)
        # draw single (grouped) insufficient data bar
        self._draw_insufficient_data_bar(ax, len(categories), self.forcing_vals)
        self._format_axes(ax, categories + [insufficient_data_category])
        self._draw_present_day_line(
            ax,
        )
        self._draw_colorbar(fig, cmap, cnorm)
        self._format_fig()
        self._format_axes(ax, categories + [insufficient_data_category])
        return fig, ax

    def _draw_insufficient_data_bar(
        self, ax: plt.Axes, i: int, forcing_values: pd.Series
    ):
        """Draw a bar for groups with insufficient data."""
        ax.add_patch(
            plt.Rectangle(
                (i - 0.1, forcing_values.min()),
                0.2,
                forcing_values.max() - forcing_values.min(),
                edgecolor="black",
                facecolor="whitesmoke",
                linewidth=1.5,
                alpha=1,
                zorder=10,
            )
        )
        ax.text(
            i,
            (forcing_values.min() + forcing_values.max()) / 2,
            "Insufficient data",
            ha="center",
            va="center",
            fontsize=10,
            rotation=270,
            color="black",
            zorder=20,
        )

    def _interpolate_preds(self, category_data):
        """Interpolate predictions and forcings to the number of levels specified in the config."""
        # Get the actual forcing value range for this category
        forcing_values = category_data[self.config.forcing_col].values.astype(float)
        pred_values = category_data["pred"].values

        # Create evenly spaced forcing values across the range
        forcing_range = np.linspace(
            forcing_values.min(), forcing_values.max(), self.config.n_levels
        )

        # Interpolate predictions to match the forcing value range
        interpolated_predictions = np.interp(
            forcing_range,
            forcing_values,
            pred_values,
        )

        # The interpolated forcings are just the evenly spaced range
        interpolated_forcings = forcing_range

        return interpolated_predictions, interpolated_forcings

    def _draw_gradient_bar(
        self,
        ax: plt.Axes,
        i: int,
        interp_preds: np.ndarray,
    ) -> None:
        """Draw a gradient bar with the necessary color map and normalization.

        Args:
            ax (plt.Axes): Axes object.
            i (int): Index of the bar.
            interp_preds (np.ndarray): Interpolated predictions.
        """
        ax.imshow(
            np.atleast_2d(interp_preds[::-1]).T,
            extent=(i - 0.1, i + 0.1, self.forcing_vals.min(), self.forcing_vals.max()),
            aspect="auto",
            cmap=self._get_cmap(),
            norm=self._get_cnorm(),
            alpha=1,
            zorder=10,
        )

    def _draw_bar_border(
        self,
        ax: plt.Axes,
        i: int,
        interpolated_forcings_values: np.ndarray,
    ) -> None:
        """Draw a border around the gradient bar, purely for aesthetics."""
        ax.add_patch(
            plt.Rectangle(
                (i - 0.1, interpolated_forcings_values.min()),
                0.2,
                interpolated_forcings_values.max() - interpolated_forcings_values.min(),
                edgecolor="black",
                facecolor="none",
                linewidth=1.5,
                zorder=11,
            )
        )

    def _draw_certainty_dots(
        self,
        ax: plt.Axes,
        i: int,
    ) -> None:
        """Draw dots to indicate the certainty of the prediction at each forcing level."""
        step = 0.5 if self.config.forcing_col == "anomaly_value_sst" else 200
        max_val = utils.round_down_to_nearest(self.forcing_vals.max(), step)
        min_val = utils.round_down_to_nearest(self.forcing_vals.min(), step)

        dot_levels = np.arange(min_val, max_val + step, step)
        # thin down to integer spacing if forcing_col is SST
        if self.config.forcing_col == "anomaly_value_sst":
            dot_levels = dot_levels[1::2]
        for forcing_level in dot_levels:
            closest_data = self.predictions_df.iloc[
                (self.predictions_df[self.config.forcing_col] - forcing_level)
                .abs()
                .argsort()[:1]
            ]
            if not closest_data.empty:
                certainty = closest_data["certainty"].values[0]
                if not np.isnan(certainty):
                    dot_positions = np.linspace(-0.05, 0.05, 4)
                    for certainty_level in range(certainty):
                        ax.plot(
                            i + 0.25 + dot_positions[certainty_level],
                            forcing_level,
                            "o",
                            color="black",
                            markersize=4,
                            alpha=1,
                            markeredgecolor="white",
                        )

    def _format_axes(self, ax: plt.Axes, categories: list[str]) -> None:
        """Format axes: remove spines, add grid, set and label x ticks, set x limits"""
        for spine in ["top", "right", "left", "bottom"]:
            ax.spines[spine].set_visible(False)
        ax.yaxis.grid(True, linestyle="--", alpha=0.7, zorder=-20)

        ax.set_xticklabels(
            categories,
            rotation=0,
            ha="center",
            fontsize=10,
        )
        ax.set_xlim(-0.5, len(categories) - 0.5)
        ax.set_xticks(range(len(categories)))
        ax.set_ylabel(
            "SST anomaly (°C)"
            if "sst" in self.config.forcing_col
            else "CO₂ concentration (ppm)",
            fontsize=12,
        )

    def _format_fig(self):
        """Format the figure: add title, adjust layout"""
        plt.suptitle(self.config.title, fontsize=14, y=1.05)
        plt.tight_layout(rect=[0, 0.15, 1, 0.95])

    def _draw_present_day_line(self, ax):
        """Draw a horizontal line at the present day forcing (CO2 or SST value)."""
        present_day_index = np.where(
            self.forcing_vals
            == self.predictions_df[self.predictions_df.time_frame == 2025][
                self.config.forcing_col
            ].iloc[0]
        )[0][0]
        ax.axhline(
            y=self.forcing_vals.iloc[present_day_index],
            color="black",
            linestyle="--",
            linewidth=1.5,
            label="Present day (2025)",
            zorder=20,
        )
        present_day_value_label = (
            f"{self.forcing_vals.iloc[present_day_index]:.02f}°C"
            if "sst" in self.config.forcing_col
            else f"{self.forcing_vals.iloc[present_day_index]:.0f}ppm"
        )
        ax.text(
            0.2,
            self.forcing_vals.iloc[present_day_index] * 1.3,
            f"Present day\n({present_day_value_label})",
            color="black",
            fontsize=10,
            ha="left",
            va="center",
            rotation=0,
            zorder=20,
        )

    def _draw_colorbar(self, fig, cmap, cnorm):
        """Draw a colorbar with annotation of severity."""
        cax = fig.add_axes([0.25, 0.05, 0.5, 0.03])
        reversed_cmap = cmap.reversed()
        cb = ColorbarBase(cax, cmap=reversed_cmap, norm=cnorm, orientation="horizontal")
        cb.set_ticks(np.linspace(cnorm.vmin, cnorm.vmax, 4))
        cb.set_ticklabels(
            [
                "Undetectable\n(0%)",
                "Moderate\n(25%)",
                "High\n(50%)",
                "Very high\n(>75%)",
            ]
        )
        cax.set_title(
            "Percentage decrease in calcification (increase in dissolution) vs. historical baseline",
            fontsize=10,
        )

    def _generate_new_year_range(self, predictions_df: pd.DataFrame):
        """Generate a dataframe containing the list of years to which predictions will be interpolated/extrapolated."""
        nys = np.arange(
            predictions_df["time_frame"].min(), predictions_df["time_frame"].max() + 1
        )
        return pd.DataFrame({"time_frame": nys})

    def _generate_full_predictions_grid(self, predictions_df: pd.DataFrame):
        """Generate a full grid containing core_grouping, scenario, and time_frame columns ready to merge with interpolated/extrapolated predictions."""
        unique_groups = predictions_df[["core_grouping", "scenario"]].drop_duplicates()
        full_grid = unique_groups.merge(
            self._generate_new_year_range(predictions_df), how="cross"
        )
        # merge metadata grid with mean predictions
        return pd.merge(
            full_grid,
            predictions_df[predictions_df.percentile == "mean"],
            on=["core_grouping", "scenario", "time_frame"],
            how="left",
        )

    def _merge_with_emissions_data(self, predictions_grid_df: pd.DataFrame):
        """Add emissions to the climatology range."""
        # TODO: this should probably outside of the class
        scenario_names = [
            scenario_name.replace(" ", "")
            for scenario_name in list(plot_config.SCENARIO_MAP.values())
        ]
        emissions_data = climatology_processing.get_emissions_data_from_file(
            config.climatology_data_dir
            / "SUPPLEMENT_DataTables_Meinshausen_6May2020.xlsx",
            scenario_names,
        )
        # merge with predictions
        merged = pd.merge(
            predictions_grid_df,
            emissions_data,
            left_on="time_frame",
            right_on="year",
            how="left",
        )
        return merged.drop(columns="year").sort_values(by="core_grouping")

    def _interpolate_predictions_to_climatology_range(
        self, predictions_df: pd.DataFrame
    ):
        """Interpolate predictions to the climatology range. Quadratic interpolation used due to form of climatology data."""
        # select mean prediction
        predictions_df[self.config.forcing_col] = predictions_df.groupby(
            ["core_grouping", "scenario"], observed=False
        )[self.config.forcing_col].transform(
            lambda x: x.interpolate(method="quadratic")
        )
        predictions_df["pred"] = predictions_df.groupby(
            ["core_grouping", "scenario"], observed=False
        )["pred"].transform(lambda x: x.interpolate(method="quadratic"))
        predictions_df["se"] = predictions_df.groupby(
            ["core_grouping", "scenario"], observed=False
        )["se"].transform(lambda x: x.interpolate(method="quadratic"))
        return predictions_df

    def _prepare_forcing_data(
        self,
    ):
        """Prepare forcing data for plotting.

        Args:
            predictions_df: DataFrame with scenario, time_frame, core_grouping, pred and se columns.

        Returns:
            DataFrame with scenario, time_frame, core_grouping, pred, se, p_score, and certainty columns ready for Burning Embers plot.
        """
        # extrapolate predictions to the end year
        self.predictions_df = (
            climatology_processing.interpolate_and_extrapolate_predictions(
                self.predictions_df, target_year=self.config.end_year
            )
        )
        # generate predictions over full range of years
        predictions_grid_df = self._generate_full_predictions_grid(self.predictions_df)
        # interpolate predictions to the climatology range
        predictions_grid_df = self._interpolate_predictions_to_climatology_range(
            predictions_grid_df
        )
        # sort by core_grouping and scenario to be ready for plotting
        predictions_grid_df.sort_values(by=["core_grouping", "scenario"], inplace=True)
        # calculate p-scores for each prediction
        predictions_grid_df["p_score"] = predictions_grid_df.apply(
            lambda row: analysis_utils.p_score(row["pred"], row["se"], null_value=0),
            axis=1,
        )
        # assign certainty to each prediction
        predictions_grid_df["certainty"] = predictions_grid_df["p_score"].apply(
            analysis_utils.assign_certainty
        )
        # add emissions to the predictions
        predictions_grid_df = self._merge_with_emissions_data(predictions_grid_df)
        # Keep sorted by core_grouping and scenario for consistent plotting
        return predictions_grid_df.sort_values(by=["core_grouping", "scenario"])
