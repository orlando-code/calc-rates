# general
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np

# spatial
# R
from scipy import interpolate

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


def annotate_axes_with_letters(
    axes: plt.Axes, fontsize: int = 16, xy: tuple[float, float] = (0.97, 0.05)
) -> plt.Axes:
    """Annotate successive axes with letters for publication-ready plots."""
    for i, ax in enumerate(axes.flatten()):
        # annotate with letter
        ax.annotate(
            chr(65 + i),  # from A to Z
            xy=xy,
            xycoords="axes fraction",
            fontsize=fontsize,
            ha="right",
            va="center",
        )


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


class Polynomial:
    """Infer mathematical equation form and return in standard notation with 'x' as variable

    Takes coefficient names and values, infers the mathematical structure, and returns
    a standardized equation using 'x' as the variable (e.g., 'x$^2$+x+3').

    Args:
        coefficient_names (list): List of coefficient names.
            - "intrcpt" for intercept terms (power 0)
            - "I(var^power)" for R-style power terms (e.g., "I(temp^2)" becomes x^2)
            - "var^power" for power terms (e.g., "ph^2" becomes x^2)
            - "var" for linear terms (e.g., "temp" becomes x)
        coefficient_values (list): List of numerical coefficient values.
        dps (int, optional): Decimal places to round coefficients to. Default is 2.

    Returns:
        str: Standardized mathematical equation string using 'x' as variable.

    Methods:
        __str__(): Returns the standardized equation string.
        format_coeff(coeff): Formats the coefficient for display (legacy).
        format_power(power): Formats the power for display (legacy).
        format_power_term(var_name, power): Formats a power term with variable and power (legacy).
    """

    def __init__(self, coefficient_names, coefficient_values, dps=2):
        self.coeff_names = coefficient_names
        self.coeff_vals = coefficient_values
        self.dps = dps

    def __str__(self):
        # Dictionary to store terms by power: {power: coefficient}
        term_powers = {}

        for coeff_name, coeff_val in zip(self.coeff_names, self.coeff_vals):
            if coeff_val == 0:
                continue

            # Round coefficient value
            coeff_val = round(coeff_val, self.dps)

            # Determine the power of this term
            if coeff_name.lower() == "intrcpt":
                # Intercept term - power 0
                power = 0
            elif coeff_name.startswith("I(") and coeff_name.endswith(")"):
                # R-style I(var^power) term
                inner_expr = coeff_name[2:-1]  # Remove "I(" and ")"
                if "^" in inner_expr:
                    var_name, power_str = inner_expr.split("^", 1)
                    try:
                        power = int(power_str)
                    except ValueError:
                        # If power is not an integer, treat as linear
                        power = 1
                else:
                    # I(var) without power - treat as linear
                    power = 1
            elif "^" in coeff_name:
                # Power term - extract power
                var_name, power_str = coeff_name.split("^", 1)
                try:
                    power = int(power_str)
                except ValueError:
                    # If power is not an integer, treat as linear
                    power = 1
            else:
                # Linear term
                power = 1

            # Store coefficient for this power
            if power in term_powers:
                term_powers[power] += coeff_val
            else:
                term_powers[power] = coeff_val

        if not term_powers:
            return "0"

        # Build the equation in standard form, sorted by decreasing powers
        terms = []
        for power in sorted(term_powers.keys(), reverse=True):
            coeff = term_powers[power]
            if coeff == 0:
                continue

            # Format the term based on power and coefficient
            if power == 0:
                # Constant term
                term = str(coeff)
            elif power == 1:
                # Linear term
                if coeff == 1:
                    term = "x"
                elif coeff == -1:
                    term = "-x"
                else:
                    term = f"{coeff}x"
            else:
                # Higher power term
                if coeff == 1:
                    term = f"x$^{power}$"
                elif coeff == -1:
                    term = f"-x$^{power}$"
                else:
                    term = f"{coeff}x$^{power}$"

            terms.append(term)

        # Join terms with appropriate signs
        if not terms:
            return "0"

        result = terms[0]
        for term in terms[1:]:
            if term.startswith("-"):
                result += term
            else:
                result += f"+{term}"

        return result

    @staticmethod
    def format_coeff(coeff):
        return str(coeff) if coeff < 0 else "+{0}".format(coeff)

    @staticmethod
    def format_power(power):
        return "x" if power == 1 else f"x$^{power}$" if power != 0 else ""

    @staticmethod
    def format_power_term(var_name, power):
        """Format a power term with variable name and power."""
        if power == 0:
            return ""
        elif power == 1:
            return var_name
        else:
            return f"{var_name}$^{power}$"


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


# --- Deprecated ---

# def set_up_regression_plot(var: str):
#     if var == "phtot":
#         xlab = "$\\Delta$ pH"
#         xlim = (-1, 0.1)
#         predlim = xlim
#         scenario_var = "ph"
#     elif var == "temp":
#         xlab = "$\\Delta$ Temperature ($^\\circ C$)"
#         xlim = (-1, 10)
#         predlim = xlim
#         scenario_var = "sst"
#     return xlab, xlim, predlim, scenario_var


# def add_climatology_lines_to_plot(
#     ax: matplotlib.axes.Axes,
#     future_global_anomaly_df: pd.DataFrame,
#     scenario_var: str,
#     xlim: tuple[float, float],
# ) -> matplotlib.axes.Axes:
#     """
#     Add climatology lines to the plot for different scenarios.

#     Args:
#         ax (matplotlib.axes.Axes): The axis to add the lines to.
#         future_global_anomaly_df (pd.DataFrame): DataFrame containing future climate scenario data. Must have scenario, time_frame, and mean_{scenario_var}_anomaly columns.
#         scenario_var (str): Variable name in future_global_anomaly_df to use for reference lines.
#         xlim (tuple[float, float]): x-axis limits for the plot.

#     Returns:
#         matplotlib.axes.Axes: The axis with the added lines.
#     """
#     scenarios = future_global_anomaly_df["scenario"].unique()
#     scenario_colours = sns.color_palette("Reds", len(scenarios))
#     scenario_colour_dict = {
#         scenario: scenario_colours[i] for i, scenario in enumerate(scenarios)
#     }
#     original_ylim = ax.get_ylim()  # get the original y-axis limits
#     scenario_lines = []
#     for scenario in scenarios:
#         # add climatology lines to the plot
#         predicted_effect_sizes = future_global_anomaly_df[
#             (future_global_anomaly_df["time_frame"] == 2090)
#             & (future_global_anomaly_df["scenario"] == scenario)
#         ][:][f"mean_{scenario_var}_anomaly"]

#         # plot vertical lines for each predicted effect size
#         for effect_size in predicted_effect_sizes:
#             line = ax.vlines(
#                 x=effect_size,
#                 ymin=original_ylim[0],
#                 ymax=original_ylim[1],
#                 color=scenario_colour_dict[scenario],
#                 linestyle="--",
#                 label=plot_config.SCENARIO_MAP[scenario],
#                 zorder=5,
#             )
#             scenario_lines.append(line)
#     ax.set_ylim(original_ylim[0], original_ylim[1])  # crop to y lim
#     return ax
