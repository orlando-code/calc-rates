import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel

from calcification.plotting import analysis


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
