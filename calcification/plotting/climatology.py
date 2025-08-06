# general
import cartopy.crs as ccrs
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import gridspec

from calcification.plotting import plot_config, plot_utils


def plot_spatial_effect_distribution(
    predictions_df: pd.DataFrame,
    var_to_plot: str = "predicted_effect_size",
    time_frame: int = 2090,
    figsize: tuple[float, float] = (10, 10),
    dpi: int = 300,
    title: str = "Spatial distribution of predicted effect sizes for SSP scenarios",
    cbar_label: str = "Predicted Effect Size (Hedges' g)",
    reverse_cmap: bool = False,
) -> tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    """
    Plots the spatial distribution of predicted effect sizes for SSP scenarios on a map.

    Args:
        predictions_df (pd.DataFrame): DataFrame containing predictions with columns ['longitude', 'latitude', 'scenario', 'time_frame', 'predicted_effect_size'].
        time_frame (int): The time frame to filter the data for (default is 2100).
        figsize (tuple): Size of the figure (default is (10, 10)).
        dpi (int): Resolution of the figure (default is 300).

    Returns:
        tuple: Figure and axes objects.

    Future work:
    - size as e.g. uncertainty, number of datapoints
    - interpolate within reef presence to show effect on whole ecosystem
    """
    ssp_scenarios = predictions_df["scenario"].unique()

    # create figure
    fig, axes = plt.subplots(
        len(ssp_scenarios),
        1,
        figsize=figsize,
        subplot_kw={"projection": ccrs.PlateCarree()},
        dpi=dpi,
    )

    mean_predictions_df = predictions_df.copy()
    # normalize color map to the range of predicted effect sizes
    min_effects, max_effects = (
        mean_predictions_df[var_to_plot].min(),
        mean_predictions_df[var_to_plot].max(),
    )

    for i, (name, scenario) in enumerate(
        zip(["SSP 1-2.6", "SSP 2-4.5", "SSP 3-7.0", "SSP 5-8.5"], ssp_scenarios)
    ):
        axes[i] = plot_utils.format_geo_axes(
            axes[i], extent=[-180, 180, -40, 40]
        )  # format map

        # filter data for the specific scenario and time frame
        data_to_plot = mean_predictions_df[
            (mean_predictions_df["scenario"] == scenario)
            & (mean_predictions_df["time_frame"] == time_frame)
        ]
        if data_to_plot.empty:
            raise ValueError(
                f"No data available for scenario {scenario} at time frame {time_frame}."
            )
        data_to_plot = data_to_plot.sort_values(by=var_to_plot, ascending=True)

        norm = plt.Normalize(vmin=min_effects, vmax=max_effects)  # normalise colour map

        sc = axes[i].scatter(
            data_to_plot["longitude"],
            data_to_plot["latitude"],
            c=data_to_plot[var_to_plot],
            cmap=plt.cm.Reds_r if reverse_cmap else plt.cm.Reds,
            norm=norm,
            s=5,
            edgecolor="k",
            linewidth=0.3,
            transform=ccrs.PlateCarree(),
            alpha=0.8,
        )

        axes[i].set_title(
            name, loc="left", fontsize=8
        )  # add title to the left of each axis

    # format colourbar
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)  # make room for colorbar at bottom
    cbar_ax = fig.add_axes(
        [0.125, 0.1, 0.75, 0.02]
    )  # [lower-left-x, lower-left-y, width, height]
    fig.colorbar(sc, cax=cbar_ax, orientation="horizontal", label=cbar_label)
    cbar_ax.tick_params(labelsize=8)
    plt.suptitle(title, fontsize=8)

    return fig, axes


def plot_climate_anomalies(
    data: pd.DataFrame,
    plot_vars: list[str] = ["sst", "ph"],
    plot_var_labels: dict = {
        "sst": "Sea surface temperature (°C) anomaly",
        "ph": "pH$_{total}$ anomaly",
    },
    time_discontinuity: int = 2025,
    figsize: tuple[float, float] = (10, 4),
    dpi: int = 300,
) -> tuple[matplotlib.figure.Figure, np.ndarray]:
    """
    Plots timeseries of global climate variable anomalies for given scenarios.
    This function extracts just the anomaly plotting portion from the original
    plot_global_timeseries_with_anomalies function.

    Args:
        data (pd.DataFrame): DataFrame containing the data to plot with columns for
            scenario, time_frame, scenario_var, anomaly_value_mean, anomaly_value_p10,
            and anomaly_value_p90.
        plot_vars (list[str]): List of climate variables to plot (e.g., ['sst', 'ph']).
        plot_var_labels (dict): Dictionary mapping variable names to their display labels.
        time_discontinuity (int): Year separating historical and forecast data.
        figsize (tuple[float, float]): Size of the figure as (width, height).
        dpi (int): Resolution of the figure in dots per inch.

    Returns:
        tuple[matplotlib.figure.Figure, np.ndarray]: Figure and array of axes objects.
    """
    # Extract unique scenarios
    scenarios = data.scenario.unique()
    n_cols = len(plot_vars)

    # Create figure and axes
    fig, axes = plt.subplots(1, n_cols, figsize=figsize, dpi=dpi)
    if n_cols == 1:
        axes = np.array([axes])  # Ensure axes is always an array

    # Define colors and formatting
    historic_colour, historic_alpha = "darkgrey", 0.5
    forecast_alpha = 0.2
    scenario_colours = sns.color_palette("Reds", len(scenarios))
    scenario_colour_dict = {
        scenario: scenario_colours[i] for i, scenario in enumerate(scenarios)
    }

    # Get unique time points
    x_points = data["time_frame"].unique()

    # Plot each climate variable
    for j, plot_var in enumerate(plot_vars):
        axis = axes[j]
        # data_subset = data[data['scenario_var'] == plot_var]

        for scenario in scenarios:
            # Get data for this scenario
            forecast_subset = data[data["scenario"] == scenario]
            mean_data = forecast_subset[forecast_subset["percentile"] == "mean"]
            p10_data = forecast_subset[forecast_subset["percentile"] == "p10"]
            p90_data = forecast_subset[forecast_subset["percentile"] == "p90"]
            # Create smooth interpolated lines
            x_fine, mean_spline = plot_utils.interpolate_spline(
                x_points, mean_data[f"anomaly_value_{plot_var}"]
            )
            x_fine, low_spline = plot_utils.interpolate_spline(
                x_points, p10_data[f"anomaly_value_{plot_var}"]
            )
            x_fine, up_spline = plot_utils.interpolate_spline(
                x_points, p90_data[f"anomaly_value_{plot_var}"]
            )

            # Create masks for historical vs forecast data
            historic_mask = x_fine < time_discontinuity
            forecast_mask = x_fine >= time_discontinuity

            # Plot forecast data
            axis.plot(
                x_fine[forecast_mask],
                mean_spline[forecast_mask],
                "--",
                alpha=0.7,
                color=scenario_colour_dict[scenario],
            )
            axis.fill_between(
                x_fine[forecast_mask],
                low_spline[forecast_mask],
                up_spline[forecast_mask],
                alpha=forecast_alpha,
                color=scenario_colour_dict[scenario],
                linewidth=0,
            )

        # Plot historical data (once, since it's the same for all scenarios)
        axis.plot(
            x_fine[historic_mask],
            mean_spline[historic_mask],
            "--",
            color=historic_colour,
        )
        axis.fill_between(
            x_fine[historic_mask],
            low_spline[historic_mask],
            up_spline[historic_mask],
            alpha=historic_alpha,
            color=historic_colour,
            linewidth=0,
        )

        # Add variable label
        annotate_var = f"{plot_var_labels[plot_var]}"
        axis.annotate(
            annotate_var,
            xy=(0.5, 0.92),
            xycoords="axes fraction",
            ha="center",
            fontsize=10,
        )

        # Format axis
        axis.tick_params(axis="both", labelsize=8)
        axis.set_xlim(1995, 2100)
        axis.set_xlabel("Year", fontsize=8)

    # Add legend
    handles = [
        plt.Line2D([], [], linestyle="--", color=historic_colour, label="Historical")
    ] + [
        plt.Line2D(
            [],
            [],
            linestyle="--",
            color=scenario_colours[i],
            label=scenario
            if "SCENARIO_MAP" not in globals()
            else plot_config.SCENARIO_MAP.get(scenario.lower(), scenario),
        )
        for i, scenario in enumerate(scenarios)
    ]

    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.98),
        ncol=len(handles),
        fontsize=8,
        frameon=False,
    )

    # Adjust layout
    plt.subplots_adjust(top=0.85)  # Make room for the title
    # fig.tight_layout()

    return fig, axes


def plot_global_timeseries_with_anomalies(
    data: pd.DataFrame,
    plot_vars: list[str, str] = ["sst", "ph"],
    plot_var_labels: list[str] = {
        "sst": "Sea surface temperature anomaly (°C)",
        "ph": "pH$_{total}$ anomaly",
    },
    time_discontinuity: int = 2025,
    figsize: tuple[float, float] = (10, 10),
    dpi: int = 300,
) -> tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    """
    Plots timeseries of global impacts for given scenarios and variables, showing both
    climate variable trajectories and resulting effect sizes in a grid layout.

    Args:
        data (pd.DataFrame): DataFrame containing the data to plot with columns for
            scenario, time_frame, scenario_var, anomaly_value_mean, anomaly_value_p10,
            anomaly_value_p90, predicted_effect_size_mean, predicted_effect_size_p10,
            and predicted_effect_size_p90.
        plot_vars (list[str]): List of climate variables to plot (e.g., ['sst', 'ph']).
        plot_var_labels (dict): Dictionary mapping variable names to their display labels.
        time_discontinuity (int, optional): Year separating historical and forecast data. Default is 2025.
        figsize (tuple[float, float], optional): Size of the figure as (width, height). Default is (10, 10).
        dpi (int, optional): Resolution of the figure in dots per inch. Default is 300.

    Returns:
        tuple[matplotlib.figure.Figure, numpy.ndarray]: Figure and array of axes objects.
    """
    scenarios = data.scenario.unique()
    # calculate subplot layout
    total_rows = 1 + len(scenarios) + 1
    n_cols = len(plot_vars)

    # create figure and GridSpec (for inserting whitespace between rows)
    fig = plt.figure(figsize=figsize, dpi=dpi)
    gs = gridspec.GridSpec(
        total_rows, n_cols, height_ratios=[1] + [0.4] + [1] * len(scenarios)
    )

    # create axes
    ax = np.empty((total_rows, n_cols), dtype=object)
    for i in range(total_rows):
        if i == 1:  # skip spacer row
            continue
        for j in range(n_cols):
            ax[i, j] = fig.add_subplot(gs[i, j])

    # add axis annotations
    fig.text(
        0.5,
        0.92,
        "Predicted global conditions for each scenario",
        ha="center",
        fontsize=12,
    )
    fig.text(0.5, 0.7, "Resulting effect sizes, by scenario", ha="center", fontsize=12)

    # define colors and formatting
    historic_colour, historic_alpha = "darkgrey", 0.5
    forecast_alpha, zero_effect_colour, zero_effect_alpha = 0.2, "black", 0.5
    scenario_colours = sns.color_palette("Reds", len(scenarios))
    scenario_colour_dict = {
        scenario: scenario_colours[i] for i, scenario in enumerate(scenarios)
    }

    x_points = data["time_frame"].unique()

    # flatten axes for easier iteration
    axes = [ax[i, j] for i in range(total_rows) for j in range(n_cols) if i != 1]
    for i, axis in enumerate(axes):
        plot_var = plot_vars[i % n_cols]
        data_subset = data[data["scenario_var"] == plot_var]
        if i < n_cols:  # forecast trajectories (first row)
            for scenario in scenarios:
                forecast_subset = data_subset[data_subset["scenario"] == scenario]
                x_fine, mean_spline = plot_utils.interpolate_spline(
                    x_points, forecast_subset["anomaly_value_mean"]
                )
                x_fine, low_spline = plot_utils.interpolate_spline(
                    x_points, forecast_subset["anomaly_value_p10"]
                )
                x_fine, up_spline = plot_utils.interpolate_spline(
                    x_points, forecast_subset["anomaly_value_p90"]
                )

                historic_mask = x_fine < time_discontinuity
                forecast_mask = x_fine >= time_discontinuity

                axis.plot(
                    x_fine[forecast_mask],
                    mean_spline[forecast_mask],
                    "--",
                    alpha=0.7,
                    color=scenario_colour_dict[scenario],
                )
                axis.fill_between(
                    x_fine[forecast_mask],
                    low_spline[forecast_mask],
                    up_spline[forecast_mask],
                    alpha=forecast_alpha,
                    color=scenario_colour_dict[scenario],
                    linewidth=0,
                )

            axis.plot(
                x_fine[historic_mask],
                mean_spline[historic_mask],
                "--",
                color=historic_colour,
            )
            axis.fill_between(
                x_fine[historic_mask],
                low_spline[historic_mask],
                up_spline[historic_mask],
                alpha=historic_alpha,
                color=historic_colour,
                linewidth=0,
            )

            annotate_var = f"{plot_var_labels[plot_var]}"
            axis.annotate(
                annotate_var,
                xy=(0.05, 1.1),
                xycoords="axes fraction",
                ha="left",
                fontsize=10,
            )
        else:  # effect sizes
            scenario = scenarios[(i - n_cols) // n_cols]
            subset = data_subset[data_subset["scenario"] == scenario]

            x_points = list(subset["time_frame"].unique())
            x_fine, mean_spline = plot_utils.interpolate_spline(
                x_points, subset["predicted_effect_size_mean"]
            )
            x_fine, up_spline = plot_utils.interpolate_spline(
                x_points, subset["predicted_effect_size_p90"]
            )
            x_fine, low_spline = plot_utils.interpolate_spline(
                x_points, subset["predicted_effect_size_p10"]
            )

            axis.plot(
                x_fine[forecast_mask],
                mean_spline[forecast_mask],
                "--",
                alpha=0.7,
                color=scenario_colour_dict[scenario],
            )
            axis.fill_between(
                x_fine[forecast_mask],
                low_spline[forecast_mask],
                up_spline[forecast_mask],
                alpha=forecast_alpha,
                color=scenario_colour_dict[scenario],
                linewidth=0,
            )
            axis.plot(
                x_fine[historic_mask],
                mean_spline[historic_mask],
                "--",
                color=historic_colour,
            )
            axis.fill_between(
                x_fine[historic_mask],
                low_spline[historic_mask],
                up_spline[historic_mask],
                alpha=historic_alpha,
                color=historic_colour,
                linewidth=0,
            )
            axis.hlines(
                0,
                xmin=x_fine.min(),
                xmax=x_fine.max(),
                color=zero_effect_colour,
                alpha=zero_effect_alpha,
            )

    # add legend
    handles = [
        plt.Line2D([], [], linestyle="--", color=historic_colour, label="Historical"),
        plt.Line2D(
            [],
            [],
            linestyle="-",
            color=zero_effect_colour,
            label="Zero effect",
            alpha=zero_effect_alpha,
        ),
    ] + [
        plt.Line2D(
            [],
            [],
            linestyle="--",
            color=scenario_colours[i],
            label=plot_config.SCENARIO_MAP[scenario.lower()],
        )
        for i, scenario in enumerate(scenarios)
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.98),
        ncol=len(handles),
        fontsize=8,
        frameon=False,
    )

    # format
    global_min_ylim, global_max_ylim = float("inf"), float("-inf")
    for i, axis in enumerate(axes):  # get global y limits
        if i >= n_cols:
            ymin, ymax = axis.get_ylim()
            global_min_ylim = min(global_min_ylim, ymin)
            global_max_ylim = max(global_max_ylim, ymax)
    for i, axis in enumerate(axes):
        axis.tick_params(axis="both", labelsize=8)
        axis.set_xlim(1995, 2100)
        if i >= n_cols:  # effect sizes
            axis.set_ylim(global_min_ylim, global_max_ylim)
            if i < len(axes) - n_cols:
                axis.set_xticks([])
            else:
                axis.set_xlabel("Year", fontsize=8)
    return fig, ax


def plot_global_timeseries(
    pred_anomaly_df: pd.DataFrame,
    figsize: tuple[float, float] = (10, 10),
    dpi: int = 300,
    title_org: str = None,
) -> None:
    """
    Plot global timeseries for the given variables.
    """
    scenarios = pred_anomaly_df["scenario"].unique()
    # Define colors and formatting
    historic_colour, historic_alpha = "darkgrey", 0.5
    forecast_alpha, zero_effect_colour, zero_effect_alpha = 0.2, "black", 0.5
    scenario_colours = sns.color_palette("Reds", len(scenarios))
    scenario_colour_dict = {
        scenario: scenario_colours[i] for i, scenario in enumerate(scenarios)
    }

    x_points = pred_anomaly_df["time_frame"].unique()
    time_discontinuity = 2025  # present day

    # Plot each scenario
    fig, axes = plt.subplots(len(scenarios), 1, figsize=(10, 10), sharex=True)
    for i, scenario in enumerate(scenarios):
        ax = axes[i]
        scenario_df = pred_anomaly_df[pred_anomaly_df["scenario"] == scenario]

        means_df = scenario_df[scenario_df["percentile"] == "mean"]
        x_fine, mean_spline = plot_utils.interpolate_spline(x_points, means_df["pred"])
        x_fine, up_spline = plot_utils.interpolate_spline(x_points, means_df["ci.lb"])
        x_fine, low_spline = plot_utils.interpolate_spline(x_points, means_df["ci.ub"])

        # Masks and formatting
        historic_mask = x_fine < time_discontinuity
        forecast_mask = x_fine >= time_discontinuity

        ax.plot(
            x_fine[forecast_mask],
            mean_spline[forecast_mask],
            "--",
            alpha=0.7,
            color=scenario_colour_dict[scenario],
        )
        ax.fill_between(
            x_fine[forecast_mask],
            low_spline[forecast_mask],
            up_spline[forecast_mask],
            alpha=forecast_alpha,
            color=scenario_colour_dict[scenario],
            linewidth=0,
        )
        ax.plot(
            x_fine[historic_mask],
            mean_spline[historic_mask],
            "--",
            color=historic_colour,
        )
        ax.fill_between(
            x_fine[historic_mask],
            low_spline[historic_mask],
            up_spline[historic_mask],
            alpha=historic_alpha,
            color=historic_colour,
            linewidth=0,
        )
        ax.hlines(
            0,
            xmin=x_fine.min(),
            xmax=x_fine.max(),
            color=zero_effect_colour,
            alpha=zero_effect_alpha,
        )

        # Add zero effect line
        ax.hlines(
            0,
            xmin=means_df["time_frame"].min(),
            xmax=means_df["time_frame"].max(),
            color=zero_effect_colour,
            alpha=zero_effect_alpha,
        )

        # Set title and labels
        ax.set_ylabel("Relative calcification rate", fontsize=8)
        ax.grid(ls="--", alpha=0.5)

    # Add global legend
    handles = [
        plt.Line2D([], [], linestyle="--", color=historic_colour, label="Historical"),
        plt.Line2D(
            [],
            [],
            linestyle="-",
            color=zero_effect_colour,
            label="Zero effect",
            alpha=zero_effect_alpha,
        ),
    ] + [
        plt.Line2D(
            [],
            [],
            linestyle="--",
            color=scenario_colours[i],
            label=plot_config.SCENARIO_MAP[scenario.lower()],
        )
        for i, scenario in enumerate(scenarios)
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.95),
        ncol=len(handles),
        fontsize=8,
        frameon=False,
    )

    global_min_ylim, global_max_ylim = 0, 0
    for i, axis in enumerate(axes):
        ymin, ymax = axis.get_ylim()
        global_min_ylim = min(global_min_ylim, ymin)
        global_max_ylim = max(global_max_ylim, ymax)

    for i, axis in enumerate(axes):
        axis.tick_params(axis="both", labelsize=8)
        axis.set_xlim(1995, 2100)
        axis.set_ylim(global_min_ylim, global_max_ylim)
        if i == len(scenarios) - 1:
            axis.set_xlabel("Year", fontsize=8)

    plt.suptitle(
        f"{title_org}\nProjected relative calcification rates under different climate scenarios"
        if title_org
        else "Projected relative calcification rates under different climate scenarios",
        fontsize=12,
    )

    return fig, axes
