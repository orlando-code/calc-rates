# general
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable

from calcification.plotting import plot_config


def plot_study_timeseries(
    df: pd.DataFrame, ax=None, colorby="core_grouping"
) -> plt.Axes:
    """
    Plot the temporal distribution of studies and observation counts.
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(9, 3), dpi=300)

    # Drop NA from year columns
    df.dropna(subset=["year"], inplace=True)

    # Count occurrences of year of each doi
    year_counts = df.groupby("year")["doi"].nunique()

    # Define smaller font size for most elements
    small_fontsize = 0.8 * plt.rcParams["font.size"]

    # Create the bar chart
    ax.bar(
        year_counts.index,
        year_counts.values,
        color="royalblue",
        width=150,
        alpha=0.5,
        edgecolor="black",
        linewidth=0.5,
    )
    ax.set_ylabel("Number of studies", fontsize=plt.rcParams["font.size"])
    ax.tick_params(axis="both", which="major", labelsize=small_fontsize)

    # Set y-ticks to appear every 5 units
    max_count = year_counts.max()
    y_ticks = np.arange(0, max_count + 5, 5)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_ticks, fontsize=small_fontsize)

    ax.grid(axis="y", linestyle="--", alpha=0.6)

    # Group df by selected group
    grouped_df = df.groupby([colorby, "year"])

    # Sum total n for each unique group and by year
    group_counts = grouped_df["n"].sum()

    # Plot number of observations each year, by selected group
    unique_group = df[colorby].unique()
    n_ax = ax.twinx()
    group_palette = sns.color_palette("colorblind", len(unique_group))
    group_color_map = dict(zip(unique_group, group_palette))

    for group in unique_group:
        group_data = group_counts[group_counts.index.get_level_values(colorby) == group]
        n_ax.scatter(
            group_data.index.get_level_values("year"),
            group_data.values,
            color=group_color_map[group],
            alpha=1,
            s=20,
            label=group,
            edgecolor="white",
            linewidth=0.5,
        )

    n_ax.set_ylabel(
        "Number of observations",
        rotation=-90,
        labelpad=12,
        fontsize=plt.rcParams["font.size"],
    )
    n_ax.tick_params(axis="y", labelsize=small_fontsize)

    # Align y-axis ticks with the bar chart
    # max_observations = group_counts.max()
    n_yticks = np.arange(0, 4001, 1000)
    n_ax.set_yticks(n_yticks)
    n_ax.set_yticklabels(n_yticks, fontsize=small_fontsize)

    # Create compact legend with smaller font
    legend = n_ax.legend(
        title=colorby.capitalize().replace("_", " "),
        loc="upper left",
        fontsize=small_fontsize,
        framealpha=0.7,
        title_fontsize=small_fontsize,
    )
    legend.get_title().set_ha("left")
    plt.xlabel("Year", fontsize=plt.rcParams["font.size"])
    plt.tight_layout()
    return ax


def plot_effect_size_distributions(
    df,
    effect_sizes=[
        "cohens_d",
        "hedges_g",
        "relative_calcification",
        "absolute_calcification",
        "st_relative_calcification",
        "st_absolute_calcification",
    ],
    outlier_limits=None,
    title="Effect Size Distributions",
):
    """
    Plots histograms and boxplots for the given effect sizes in the dataframe.

    Parameters:
    - df (pd.DataFrame): The dataframe containing the effect size data.
    - effect_sizes (list): List of effect size column names to plot.
    - outlier_limits (tuple): Optional tuple (lower_limit, upper_limit) to filter outliers.
    """
    fig, axes = plt.subplots(len(effect_sizes), 1, figsize=(5, 8), dpi=300)

    for i, effect_size in enumerate(effect_sizes):
        ax = axes[i]
        data = df[effect_size].dropna()

        # Remove outliers if limits are provided
        if outlier_limits:
            lower_limit, upper_limit = outlier_limits
            data = data[(data > lower_limit) & (data < upper_limit)]

        # Plot histogram
        ax.hist(data, bins=100, color="skyblue", edgecolor="black")
        ax.set_xlabel(effect_size, fontsize=6)
        ax.set_ylabel("Frequency", fontsize=6)
        ax.grid(ls="--", alpha=0.5)
        ax.vlines(0, *ax.get_ylim(), color="red", linestyle="--", linewidth=1)

        # Add boxplot above using divider
        divider = make_axes_locatable(ax)
        box_ax = divider.append_axes("top", size="20%", pad=0.01, sharex=ax)
        box_ax.boxplot(
            data, vert=False, patch_artist=True, boxprops=dict(facecolor="lightgray")
        )
        box_ax.axis("off")
        for outlier in box_ax.findobj(match=plt.Line2D):
            outlier.set_markersize(3)
            outlier.set_alpha(0.1)

        # Optional: log scale if necessary
        if max([p.get_height() for p in ax.patches]) > 10:
            ax.set_yscale("log")

        ax.tick_params(axis="both", which="major", labelsize=6)

    plt.suptitle(title, fontsize=8)
    plt.tight_layout()


def plot_effect_size_grid(
    results_df: pd.DataFrame,
    rate_types: list,
    x_var: str,
    y_vars: list[str],
    col_titles: list[str] = None,
    figure_title: str = None,
    figsize: tuple[float] = (10, 8),
    dpi: int = 300,
    s: float = 1,
) -> tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    """
    Create a grid of plots for effect sizes.

    Args:
        results_df (pandas.DataFrame): The dataframe containing the data to plot
        rate_types (list): List of rate types to plot rows for
        x_var (str): Column name to use as x variable for each grid column
        y_vars (list[str]): List of column names to use as y variables for each grid column
        col_titles (list[str], optional): Titles for each column of plots
        figure_title (str, optional): Title for the overall figure
        figsize (tuple[float], optional): Size of the figure (width, height)
        dpi (int, optional): Resolution of the figure
        s (float, optional): Size of scatter points

    Returns:
        tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]
    """
    fig, axes = plt.subplots(len(rate_types), len(y_vars), figsize=figsize, dpi=dpi)

    if len(y_vars) == 1:  # if only one column, convert axes to 2D array
        axes = axes.reshape(-1, 1)

    # share x axis within columns
    for col in range(len(y_vars)):
        for i in range(1, len(rate_types)):
            axes[i, col].sharex(axes[0, col])

    for i, rate_type in enumerate(rate_types):
        rate_df = results_df[results_df["st_calcification_unit"] == rate_type]

        # Add rate type label only to first column (for the whole row)
        display_name = plot_config.RATE_TYPE_MAPPING.get(rate_type, rate_type)
        axes[i][0].set_ylabel(display_name, fontsize=6)

        for j, y_var in enumerate(y_vars):
            axes[i][j].scatter(rate_df[x_var], rate_df[y_var], s=s)

    # add column titles to the first row only
    if col_titles:
        for j, col_title in enumerate(col_titles):
            axes[0][j].set_title(col_title, fontsize=6)

    # format axes
    for ax in axes.flatten():
        ax.grid(visible=True, alpha=0.3)
        # zero effect line
        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.7, linewidth=0.8)
        ax.tick_params(axis="both", which="major", labelsize=6)
        ax.ticklabel_format(
            style="scientific", axis="y", scilimits=(-2, 2), useMathText=True
        )
        ax.yaxis.get_offset_text().set_fontsize(6)

    # clear x tick labels for all but the bottom row
    for i in range(len(rate_types) - 1):
        for j in range(len(y_vars)):
            axes[i, j].tick_params(axis="x", labelbottom=False)
    for j in range(len(y_vars)):
        axes[-1, j].set_xlabel(x_var, fontsize=6)

    if figure_title:
        plt.suptitle(figure_title, fontsize=10)
    plt.tight_layout()

    return fig, axes
