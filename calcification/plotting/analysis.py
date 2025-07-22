# general
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import rpy2.robjects as ro
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
from calcification.analysis import analysis
from calcification.plotting import plot_utils

# R
metafor = importr("metafor")
grdevices = importr("grDevices")


def meta_regplot(
    model: ro.vectors.ListVector,
    model_comps: tuple,
    x_mod: str,
    level: float = 95,
    ci: bool = True,
    pi: bool = True,
    shade: bool = True,
    point_size: str = "seinv",
    point_color: str = "black",
    point_fill: str = "white",
    colorby: list = None,
    line_color: str = "blue",
    ci_color: str = "lightblue",
    ci_line_color: str = "blue",
    xlab: str = None,
    ylab: str = None,
    xlim: tuple[float, float] = None,
    ylim: tuple[float, float] = None,
    predlim: tuple[float, float] = None,
    refline: float = 0,
    figsize: tuple[float, float] = (10, 7),
    title: str = None,
    ax: matplotlib.axes.Axes = None,
    future_global_anomaly_df: pd.DataFrame = None,
    scenario_var: str = None,
    all_legend: bool = True,
) -> tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    """
    Create a meta-regression plot from an rma.mv or rma object.

    This function visualizes a meta-regression model, showing the effect of a moderator variable
    on the outcome, along with confidence and prediction intervals.

    Args:
        model (rpy2.robjects.vectors.ListVector): An R rma.mv or rma model object from metafor package.
        model_comps (tuple): Model components containing predictor and response info.
        x_mod (str): Name of the moderator variable to plot on x-axis.
        level (float, optional): Confidence level for intervals in percent. Default is 95.
        ci (bool, optional): Whether to plot confidence intervals. Default is True.
        pi (bool, optional): Whether to plot prediction intervals. Default is True.
        shade (bool or str, optional): Whether to shade confidence intervals. If string, specifies the shade color. Default is True.
        point_size (str or array-like, optional): Point sizes - either "seinv" (inverse of standard error), "vinv"
            (inverse of variance), or an array of custom sizes. Default is "seinv".
        point_color (str or array-like, optional): Color for point borders. Default is 'black'.
        point_fill (str or array-like, optional): Fill color for points. Default is 'white'.
        colorby (list, optional): Values to color points by. If provided, creates a color gradient. Default is None.
        line_color (str, optional): Color for regression line. Default is 'blue'.
        ci_color (str, optional): Color for CI shading. Default is 'lightblue'.
        ci_line_color (str, optional): Color for CI lines. Default is 'blue'.
        xlab (str, optional): Label for x-axis.
        ylab (str, optional): Label for y-axis.
        xlim (tuple[float, float], optional): Limits for x-axis (min, max).
        ylim (tuple[float, float], optional): Limits for y-axis (min, max).
        predlim (tuple[float, float], optional): Limits for predicted x-axis values (min, max).
        refline (float, optional): Reference line to add at specific y-value. Default is 0.
        figsize (tuple[float, float], optional): Figure size (width, height) in inches. Default is (10, 7).
        title (str, optional): Plot title.
        ax (matplotlib.axes.Axes, optional): Existing axis to plot on. If None, a new figure and axis will be created.
        future_global_anomaly_df (pd.DataFrame, optional): DataFrame containing future climate scenario data for reference lines.
        scenario_var (str, optional): Variable name in future_global_anomaly_df to use for reference lines.

    Returns:
        tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]: The figure and axis objects containing the plot.
    """
    xi, yi, vi, norm_weights, xs, pred, ci_lb, ci_ub, pred_lb, pred_ub, mod_pos = (
        analysis.process_meta_regplot_data(
            model, model_comps, x_mod, level, point_size, predlim
        )
    )

    ### plot
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    if isinstance(point_color, str):
        point_color = [point_color] * len(xi)
    if isinstance(point_fill, str):
        point_fill = [point_fill] * len(xi)

    if colorby is not None:  # create a color map for the points
        colorby = np.array(colorby)
        unique_colors = np.unique(colorby)
        color_map = {
            color: plt.cm.viridis(i / len(unique_colors))
            for i, color in enumerate(unique_colors)
        }
        point_colors = [color_map[color] for color in colorby]
    else:
        point_colors = point_fill

    sorted_indices = np.argsort(-norm_weights)  # larger points plot at the back
    for i in sorted_indices:
        ax.scatter(
            xi[i],
            yi[i],
            s=norm_weights[i] ** 2,
            edgecolor=point_color[i],
            facecolor=point_colors[i] if colorby is not None else point_fill[i],
            zorder=3,
            alpha=0.8,
        )

    if ci and shade:  # add confidence interval
        if isinstance(shade, str):
            poly_color = shade
        else:
            poly_color = ci_color

        ax.fill_between(xs, ci_lb, ci_ub, color=poly_color, alpha=0.3)

    if ci:  # add confidence interval lines
        ax.plot(xs, ci_lb, color=ci_line_color, linestyle="--", linewidth=1)
        ax.plot(xs, ci_ub, color=ci_line_color, linestyle="--", linewidth=1)
    if pi:  # add prediction interval lines
        ax.plot(xs, pred_lb, color=ci_line_color, linestyle=":", linewidth=1)
        ax.plot(xs, pred_ub, color=ci_line_color, linestyle=":", linewidth=1)
        if shade:
            ax.fill_between(xs, pred_lb, pred_ub, color=ci_color, alpha=0.1)

    ax.scatter(
        [],
        [],
        edgecolor=point_color[0],
        facecolor=point_fill[0],
        alpha=0.8,
        label="Studies",
    )
    # Get coefficients and create a polynomial string for the legend
    coeffs = np.array(model.rx2("b"))
    # polynomial = Polynomial(coeffs[mod_pos])
    polynomial = plot_utils.Polynomial(
        np.concatenate([coeffs[mod_pos], coeffs[0]])
    ).__str__()
    regression_mod_str = "$\\Delta pH$" if x_mod == "delta_ph" else "$\\Delta T$"
    # replace x with the variable name
    polynomial = polynomial.replace("x", regression_mod_str)
    regression_line = ax.plot(
        xs, pred, color=line_color, linewidth=2, label=f"Meta-regression: {polynomial}"
    )

    ax.plot(
        [], [], linestyle="--", color=ci_line_color, label="95% Confidence interval"
    ) if ci else None
    ax.plot(
        [], [], linestyle=":", color=ci_line_color, label="95% Prediction interval"
    ) if pi else None

    ### formatting
    ax.axhline(
        y=refline, color="gray", label="Zero effect level", linestyle="-", linewidth=1
    ) if refline is not None else None  # add reference line
    ax.set_xlabel(
        xlab
        if xlab
        else model_comps["predictors"][mod_pos]
        if not model_comps["intercept"]
        else model_comps["predictors"][mod_pos - 1],
        fontsize=12,
    )
    ax.set_ylabel(
        ylab if ylab else f"Effect size ({model_comps['response']})", fontsize=12
    )
    ax.set_title(title, fontsize=14) if title else None
    ax.set_xlim(xlim) if xlim else predlim if predlim else None
    ax.set_ylim(ylim) if ylim else None
    ax.grid(True, linestyle="--", alpha=0.3)

    if all_legend:
        ax.legend(fontsize=10)
        # if colorby, make legend for values
        if colorby is not None:
            # if colorby represents discrete values (not continuous)
            if len(unique_colors) < 5:
                legend_elements = []
                for color in unique_colors:
                    legend_elements.append(
                        Line2D(
                            [0],
                            [0],
                            marker="o",
                            color="w",
                            label=color,
                            markerfacecolor=color_map[color],
                            markersize=10,
                        )
                    )
                ax.legend(
                    handles=legend_elements,
                    title="Point colours",
                    fontsize=10,
                    loc="upper left",
                )
            # if colorby represents continuous values, make a colorbar
            else:
                sm = plt.cm.ScalarMappable(
                    cmap="viridis",
                    norm=plt.Normalize(vmin=min(colorby), vmax=max(colorby)),
                )
                sm.set_array([])
                cbar = plt.colorbar(sm, ax=ax)
                cbar.set_label("Color by", fontsize=10)
    else:  # only show regression line in legend if all_legend is False
        ax.legend(
            [regression_line[0]],
            [f"Meta-regression: {polynomial}"],
            fontsize=10,
            loc="upper left",
        )

    if ax is None:  # Only apply tight_layout if we created a new figure
        fig.tight_layout()

    return fig, ax


def meta_regplot_plotly(
    model,
    model_comps: tuple,
    x_mod: str,
    level: float = 95,
    ci: bool = True,
    pi: bool = True,  # TODO: finish updating this
    point_size: str = "seinv",
    point_color: str = "black",
    point_fill: str = "white",
    doi_list: list = None,
    line_color: str = "blue",
    ci_color: str = "rgba(173, 216, 230, 0.3)",
    ci_line_color: str = "blue",
    xlab: str = None,
    ylab: str = None,
    xlim: list[float, float] = None,
    ylim: list[float, float] = None,
    predlim: list[float, float] = None,
    refline: float = 0,
    figsize: tuple[float, float] = (900, 600),
    title: str = None,
):
    """
    Create an interactive meta-regression plot from an rma.mv or rma object using Plotly.

    Args:
        model (rpy2.robjects.vectors.ListVector): An R rma.mv or rma model object.
        mod_pos (int or str, optional): Position or name of the moderator variable to plot.
        level (float, default=95): Confidence level for confidence intervals.
        ci (bool, default=True): Whether to plot confidence intervals.
        pi (bool, default=True): Whether to plot prediction intervals.
        point_size (str or array-like, default="seinv"): Point sizes - either "seinv" (inverse of standard error),
            "vinv" (inverse of variance), or an array of custom sizes.
        point_color (str or array-like, default='black'): Color for point borders.
        point_fill (str or array-like, default='white'): Fill color for points.
        doi_list (list, optional): List of DOIs or identifiers for each study point.
        line_color (str, default='blue'): Color for regression line.
        ci_color (str, default='rgba(173, 216, 230, 0.3)'): Color for CI shading.
        ci_line_color (str, default='blue'): Color for CI lines.
        xlab (str, optional): Label for x-axis.
        ylab (str, optional): Label for y-axis.
        xlim (tuple, optional): Limits for x-axis (min, max).
        ylim (tuple, optional): Limits for y-axis (min, max).
        predlim (tuple, optional): Limits for predicted x-axis values (min, max).
        refline (float, optional): Reference line to add at specific y-value.
        figsize (tuple, default=(900, 600)): Figure size (width, height) in pixels.
        title (str, optional): Plot title.

    Returns:
        plotly.graph_objects.Figure: Interactive Plotly figure
    """
    ### process data for plot
    xi, yi, vi, norm_weights, xs, pred, ci_lb, ci_ub, pred_lb, pred_ub, mod_pos = (
        analysis.process_meta_regplot_data(
            model, model_comps, x_mod, level, point_size, predlim
        )
    )

    ### plot
    fig = go.Figure()

    hover_text = []
    if doi_list is not None:
        for i in range(len(xi)):
            hover_text.append(
                f"DOI: {doi_list.iloc[i]}<br>X: {xi[i]:.4f}<br>Y: {yi[i]:.4f}<br>SE: {np.sqrt(vi[i]):.4f}"
            )
    else:
        for i in range(len(xi)):
            hover_text.append(
                f"X: {xi[i]:.4f}<br>Y: {yi[i]:.4f}<br>SE: {np.sqrt(vi[i]):.4f}"
            )

    # create colormap if DOIs are provided
    if doi_list is not None:
        unique_dois = list(set(doi_list))
        colorscale = [
            "#1f77b4",
            "#ff7f0e",
            "#2ca02c",
            "#d62728",
            "#9467bd",
            "#8c564b",
            "#e377c2",
            "#7f7f7f",
            "#bcbd22",
            "#17becf",
        ]
        if len(unique_dois) > len(colorscale):  # extend color palette if needed
            import plotly.express as px

            colorscale = px.colors.qualitative.Plotly + px.colors.qualitative.D3

        color_map = {
            doi: colorscale[i % len(colorscale)] for i, doi in enumerate(unique_dois)
        }
        marker_colors = [color_map[doi] for doi in doi_list]
    else:
        marker_colors = [point_fill] * len(xi)

    # add study points and scale marker sizes appropriately for Plotly's way of visualising
    fig.add_trace(
        go.Scatter(
            x=xi,
            y=yi,
            mode="markers",
            marker=dict(
                size=norm_weights
                * (figsize[0] / 1000),  # scale marker size based on figure width
                color=marker_colors,
                line=dict(width=2, color=point_color),
            ),
            text=hover_text,
            hoverinfo="text",
            name="Studies",
        )
    )

    if ci:  # add confidence intervals
        fig.add_trace(
            go.Scatter(
                x=list(xs) + list(xs[::-1]),
                y=list(ci_ub) + list(ci_lb[::-1]),
                fill="toself",
                fillcolor=ci_color,
                line=dict(color="rgba(255,255,255,0)"),
                hoverinfo="skip",
                showlegend=False,
                name="95% Confidence Interval",
            )
        )  # shading

        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ci_lb,
                mode="lines",
                line=dict(color=ci_line_color, width=1, dash="dash"),
                hoverinfo="skip",
                showlegend=True,
                name="95% Confidence Interval",
            )
        )  # lower lines

        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ci_ub,
                mode="lines",
                line=dict(color=ci_line_color, width=1, dash="dash"),
                hoverinfo="skip",
                showlegend=False,
            )
        )  # upper lines

    if pi:  # add prediction intervals
        fig.add_trace(
            go.Scatter(
                x=list(xs) + list(xs[::-1]),
                y=list(pred_ub) + list(pred_lb[::-1]),
                fill="toself",
                fillcolor="rgba(173, 216, 230, 0.1)",
                line=dict(color="rgba(255,255,255,0)"),
                hoverinfo="skip",
                showlegend=False,
                name="95% Prediction Interval",
            )
        )  # shading

        # Add PI lines
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=pred_lb,
                mode="lines",
                line=dict(color=ci_line_color, width=1, dash="dot"),
                hoverinfo="skip",
                showlegend=True,
                name="95% Prediction Interval",
            )
        )  # lower lines

        fig.add_trace(
            go.Scatter(
                x=xs,
                y=pred_ub,
                mode="lines",
                line=dict(color=ci_line_color, width=1, dash="dot"),
                hoverinfo="skip",
                showlegend=False,
            )
        )  # upper lines

    fig.add_trace(
        go.Scatter(
            x=xs,
            y=pred,
            mode="lines",
            line=dict(color=line_color, width=2),
            name="Regression line",
        )
    )  # regression line

    if refline is not None:  # add reference line if specified
        fig.add_trace(
            go.Scatter(
                x=[predlim[0], predlim[1]],
                y=[refline, refline],
                mode="lines",
                line=dict(color="gray", width=1),
                name="Zero effect level",
            )
        )

    # format
    fig.update_layout(
        title=title,
        xaxis=dict(title=xlab, range=xlim if xlim else predlim),
        yaxis=dict(title=ylab, range=ylim),
        width=figsize[0],
        height=figsize[1],
        hovermode="closest",
        template="plotly_white",
    )

    return fig


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


def plot_3d_surface_gp(
    df: pd.DataFrame,
    title: str = "Calcification Rate vs. Temperature and pH",
    z_label: str = "Calcification Rate",
    resolution: int = 50,
) -> go.Figure:
    """
    Args:
        df (pd.DataFrame): DataFrame containing 'temp', 'phtot', and 'st_calcification' columns
        title (str, optional): The title for the plot
        z_label (str, optional): The label for the z-axis
        resolution (int, optional): Resolution of the grid for interpolation

    Returns:
        plotly.graph_objects.Figure: The plotly figure object
    """
    X_train = df[["temp", "phtot"]].values
    y_train = df["st_calcification"].values

    # define the kernel - variable-independent RBFs with noise
    k1 = ConstantKernel(1.0) * RBF(
        length_scale=1.0, length_scale_bounds=(0.1, 10.0)
    )  # for temperature
    k2 = ConstantKernel(1.0) * RBF(
        length_scale=0.25, length_scale_bounds=(0.01, 0.5)
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
    x = np.linspace(df["temp"].min(), df["temp"].max(), resolution)
    y = np.linspace(df["phtot"].min(), df["phtot"].max(), resolution)
    X, Y = np.meshgrid(x, y)
    grid_points = np.column_stack([X.ravel(), Y.ravel()])
    Z = gp.predict(grid_points).reshape(X.shape)

    # plot
    fig = go.Figure()
    fig.add_trace(
        go.Surface(
            x=X,
            y=Y,
            z=Z,
            colorscale="viridis",
            colorbar=dict(title=z_label),
            opacity=0.8,
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=df["temp"],
            y=df["phtot"],
            z=df["st_calcification"],
            mode="markers",
            marker=dict(
                size=4, color="white", line=dict(color="black", width=0.5), opacity=0.2
            ),
            name="Data Points",
        )
    )

    # format
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="Temperature (°C)",
            yaxis_title="pH<sub>T</sub>",
            zaxis_title=z_label,
            aspectmode="cube",
            xaxis=dict(nticks=10, range=[df["temp"].min(), df["temp"].max()]),
            yaxis=dict(nticks=10, range=[df["phtot"].min(), df["phtot"].max()]),
            zaxis=dict(nticks=10, range=[-10, 10]),
        ),
        width=900,
        height=700,
        margin=dict(l=65, r=50, b=65, t=90),
    )

    return fig
