# general
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# spatial
# R
import seaborn as sns
from scipy import interpolate

from calcification.plotting import config as plot_config
from calcification.utils import config, utils


def save_fig(
    fig: object,
    fig_name: str,
    run_key: str = None,
) -> None:
    print("Saving figure to", config.fig_dir)
    config.fig_dir.mkdir(parents=True, exist_ok=True)
    run_key = (
        run_key + "_" + utils.get_now_timestamp_formatted()
        if run_key
        else utils.get_now_timestamp_formatted()
    )

    fig_fp = config.fig_dir / f"{fig_name}_{run_key}.png"
    fig.savefig(fig_fp, dpi=300)

    print(f"Figure saved to {fig_fp}")


def interpolate_spline(
    xs: np.ndarray, ys: np.ndarray, npoints: int = 100, k: int = 2
) -> tuple[np.ndarray, np.ndarray]:
    """Interpolates a spline for given x and y values.

    Args:
        xs (np.ndarray): x values.
        ys (np.ndarray): y values.
        npoints (int, optional): Number of points for interpolation. Default is 100.
        k (int, optional): Degree of the spline. Default is 2.

    Returns:
        tuple: Interpolated x values and corresponding y values.
    """
    x_fine = np.linspace(min(xs), max(xs) + 10, 100)
    spline = interpolate.splrep(xs, ys, k=k)
    smooth = interpolate.splev(x_fine, spline)
    return x_fine, smooth


def set_up_regression_plot(var: str):
    if var == "phtot":
        xlab = "$\\Delta$ pH"
        xlim = (-1, 0.1)
        predlim = xlim
        scenario_var = "ph"
    elif var == "temp":
        xlab = "$\\Delta$ Temperature ($^\\circ C$)"
        xlim = (-1, 10)
        predlim = xlim
        scenario_var = "sst"
    return xlab, xlim, predlim, scenario_var


def add_climatology_lines_to_plot(
    ax: matplotlib.axes.Axes,
    future_global_anomaly_df: pd.DataFrame,
    scenario_var: str,
    xlim: tuple[float, float],
) -> matplotlib.axes.Axes:
    """
    Add climatology lines to the plot for different scenarios.

    Args:
        ax (matplotlib.axes.Axes): The axis to add the lines to.
        future_global_anomaly_df (pd.DataFrame): DataFrame containing future climate scenario data. Must have scenario, time_frame, and mean_{scenario_var}_anomaly columns.
        scenario_var (str): Variable name in future_global_anomaly_df to use for reference lines.
        xlim (tuple[float, float]): x-axis limits for the plot.

    Returns:
        matplotlib.axes.Axes: The axis with the added lines.
    """
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
            (future_global_anomaly_df["time_frame"] == 2090)
            & (future_global_anomaly_df["scenario"] == scenario)
        ][:][f"mean_{scenario_var}_anomaly"]

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


class Polynomial:
    """Fancy formatting for polynomial strings

    Args:
        coefficients (list): List of coefficients for the polynomial.
        dps (int, optional): Decimal places to round coefficients to. Default is 2.

    Returns:
        str: Formatted polynomial string.

    Methods:
        __str__(): Returns the formatted polynomial string.
        format_coeff(coeff): Formats the coefficient for display.
        format_power(power): Formats the power for display.
    """

    def __init__(self, coefficients, dps=2):
        self.coeffs = coefficients
        self.dps = dps

    def __str__(self):
        chunks = []
        for coeff, power in zip(self.coeffs, range(len(self.coeffs) - 1, -1, -1)):
            if coeff == 0:
                continue
            chunks.append(self.format_coeff(round(coeff, self.dps)))
            chunks.append(self.format_power(power))
        chunks[0] = chunks[0].lstrip("+")
        return "".join(chunks)

    @staticmethod
    def format_coeff(coeff):
        return str(coeff) if coeff < 0 else "+{0}".format(coeff)

    @staticmethod
    def format_power(power):
        return "x" if power == 1 else "x^{0}".format(power) if power != 0 else ""


def format_geo_axes(
    ax: plt.Axes, extent: tuple | list = (-180, 180, -40, 50)
) -> plt.Axes:
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, facecolor="white")
    ax.add_feature(cfeature.OCEAN, alpha=0.3)
    ax.add_feature(cfeature.COASTLINE, edgecolor="lightgray", zorder=-1)
    ax.add_feature(
        cfeature.BORDERS, linestyle=":", edgecolor="gray", alpha=0.1, zorder=-1
    )

    return ax
