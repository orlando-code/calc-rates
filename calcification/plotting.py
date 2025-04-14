# general
import matplotlib
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from matplotlib import gridspec
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns    
    
# spatial
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import geopandas as gpd
from shapely.geometry import Point

# R
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr
metafor = importr('metafor')
grdevices = importr('grDevices')
    
# stats
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import interpolate

# custom
from calcification import config, utils

SCENARIO_MAP = {
    'ssp126': 'SSP 1-2.6',
    'ssp245': 'SSP 2-4.5',
    'ssp370': 'SSP 3-7.0',
    'ssp585': 'SSP 5-8.5',
}


def plot_effect_size_grid(results_df, rate_types,
                          x_var, y_vars, col_titles=None, rate_type_mapping=None, figure_title=None,
                          figsize=(10, 8), dpi=300, s=1):
    """
    Create a grid of plots for effect sizes.
    
    Parameters:
    -----------
    results_df : pandas.DataFrame
        The dataframe containing the data to plot
    rate_types : list
        List of rate types to plot rows for
    x_var : str
        Column name to use as x variable for each grid column
    y_vars : list of str
        List of column names to use as y variables for each grid column
    col_titles : list of str
        Titles for each column of plots
    rate_type_mapping : dict, optional
        Mapping of rate type codes to display names
    figure_title : str, optional
        Title for the overall figure
    figsize : tuple, optional
        Size of the figure (width, height)
    dpi : int, optional
        Resolution of the figure
    s : float, optional
        Size of scatter points
        
    Returns:
    --------
    fig, axes : tuple
        The figure and axes objects for further customization if needed
    """
    fig, axes = plt.subplots(len(rate_types), len(y_vars), figsize=figsize, dpi=dpi)
    
    if len(y_vars) == 1:    # if only one column, convert axes to 2D array
        axes = axes.reshape(-1, 1)
    
    # share x axis within columns
    for col in range(len(y_vars)):
        for i in range(1, len(rate_types)):
            axes[i, col].sharex(axes[0, col])
    
    for i, rate_type in enumerate(rate_types):
        rate_df = results_df[results_df['st_calcification_unit'] == rate_type]
        
        # Add rate type label only to first column (for the whole row)
        display_name = rate_type_mapping.get(rate_type, rate_type) if rate_type_mapping else rate_type
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
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.7, linewidth=0.8)
        ax.tick_params(axis='both', which='major', labelsize=6)
        ax.ticklabel_format(style='scientific', axis='y', scilimits=(-2, 2), useMathText=True)
        ax.yaxis.get_offset_text().set_fontsize(6)
    
    # clear x tick labels for all but the bottom row
    for i in range(len(rate_types)-1):
        for j in range(len(y_vars)):
            axes[i, j].tick_params(axis='x', labelbottom=False)
    for j in range(len(y_vars)):
        axes[-1, j].set_xlabel(x_var, fontsize=6)
            
    if figure_title:
        plt.suptitle(figure_title, fontsize=10)
    plt.tight_layout()
    
    return fig, axes


class Polynomial:
    """Completely unnecessary class for formatting polynomial strings nicely"""
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
        return ''.join(chunks)

    @staticmethod
    def format_coeff(coeff):
        return str(coeff) if coeff < 0 else "+{0}".format(coeff)

    @staticmethod
    def format_power(power):
        return 'x' if power == 1 else 'x^{0}'.format(power) if power != 0 else ''
    

def meta_regplot(model, model_comps: tuple, x_mod:str, level=95, ci=True, pi=True, shade=True, 
                 point_size="seinv", point_color='black', point_fill='white', colorby: list=None,
                 line_color='blue', ci_color='lightblue', ci_line_color='blue',
                 xlab=None, ylab=None, xlim=None, ylim=None, predlim=None,
                 refline=0, figsize=(10, 7), title=None, ax=None, future_global_anomaly_df=None, scenario_var=None):
    """
    Create a meta-regression plot from an rma.mv or rma object.
    
    Args:
        model (rpy2.robjects.vectors.ListVector): An R rma.mv or rma model object.
        mod (int or str, optional): Position or name of the moderator variable to plot.
        level (float, default=95): Confidence level for confidence intervals.
        ci (bool, default=True): Whether to plot confidence intervals.
        shade (bool or str, default=True): Whether to shade confidence intervals (True) or just plot lines (False). 
            Can also be a color string.
        point_size (str or array-like, default="seinv"): Point sizes - either "seinv" (inverse of standard error), "vinv" 
            (inverse of variance), or an array of custom sizes.
        point_color (str or array-like, default='black'): Color for point borders.
        point_fill (str or array-like, default='white'): Fill color for points.
        line_color (str, default='blue'): Color for regression line.
        ci_color (str, default='lightblue'): Color for CI shading.
        ci_line_color (str, default='blue'): Color for CI lines.
        xlab (str, optional): Label for x-axis.
        ylab (str, optional): Label for y-axis.
        xlim (tuple, optional): Limits for x-axis (min, max).
        ylim (tuple, optional): Limits for y-axis (min, max).
        predlim (tuple, optional): Limits for predicted x-axis values (min, max).
        refline (float, optional): Reference line to add at specific y-value.
        figsize (tuple, default=(10, 7)): Figure size (width, height) in inches.
        title (str, optional): Plot title.
        ax (matplotlib.axes._axes.Axes, optional): Existing axis to plot on. If None, a new figure and axis will be created.
        
    Returns:
        fig, ax (matplotlib.figure.Figure, matplotlib.axes._axes.Axes): Matplotlib figure and axis objects.
    """    
    ### process
    pandas2ri.activate()    # enable automatic conversion between R and pandas
    # get index of x_mod in predictors
    mod_pos = model_comps['predictors'].index(x_mod) if isinstance(x_mod, str) else 0
        
    # extract model components
    yi = np.array(model.rx2('yi.f'))
    vi = np.array(model.rx2('vi.f'))
    X = np.array(model.rx2('X.f'))    
    xi = X[:, mod_pos]
    
    mask = ~np.isnan(yi) & ~np.isnan(vi) & ~np.isnan(xi).any(axis=0)    # handle missing values
    if not all(mask):
        yi = yi[mask]
        vi = vi[mask]
        xi = xi[mask]
    
    # create weight vector for point sizes
    if point_size == "seinv":
        weights = 1 / np.sqrt(vi + 0.001)   # correct for no apparent variance
    elif point_size == "vinv":
        weights = 1 / vi + 0.001    # correct for no apparent variance
    elif isinstance(point_size, (list, np.ndarray)):
        weights = np.array(point_size)
    else:
        weights = np.ones_like(yi)
    
    if len(weights) > 0:    # normalize weights for point sizes
        min_w, max_w = min(weights), max(weights)
        if max_w - min_w > np.finfo(float).eps:
            norm_weights = 30 * (weights - min_w) / (max_w - min_w) + 1
        else:
            norm_weights = np.ones_like(weights) * 20
    else:
        norm_weights = np.ones_like(yi) * 20
    
    range_xi = max(xi) - min(xi)    # create sequence of x values for the regression line
    predlim = (min(xi) - 0.1*range_xi, max(xi) + 0.1*range_xi) if predlim is None else predlim
    xs = np.linspace(predlim[0], predlim[1], 1000)
    
    r_xs = ro.FloatVector(xs)
    
    # create prediction data for the regression line
    # This requires creating a new matrix with mean values for all predictors
    # and varying only the moderator of interest    # TODO: adapt to vary multiple moderators
    predict_function = ro.r('''
    function(model, xs, mod_pos, level) {
        # Get mean values for all predictors
        X_means <- colMeans(model$X.f)
        
        # Create new data for predictions
        Xnew <- matrix(rep(X_means, each=length(xs)), nrow=length(xs))
        colnames(Xnew) <- colnames(model$X.f)
        
        # Set the moderator of interest to the sequence of values
        Xnew[,mod_pos] <- xs
        
        # Remove intercept if present in the model
        if (model$int.incl) {
            Xnew <- Xnew[,-1, drop=FALSE]
        }
        
        # Make predictions
        pred <- predict(model, newmods=Xnew, level=(level/100))
        
        # Return results
        return(pred)
    }
    ''')
    
    ### get predictions
    try:
        pred_res = predict_function(model, r_xs, mod_pos + 1, level)  # R is 1-indexed
        pred = np.array(pred_res.rx2('pred'))
        ci_lb = np.array(pred_res.rx2('ci.lb'))
        ci_ub = np.array(pred_res.rx2('ci.ub'))
        pred_lb = np.array(pred_res.rx2('pi.lb'))
        pred_ub = np.array(pred_res.rx2('pi.ub'))
    except Exception as e:
        print(f"Error in prediction: {e}")
        print("Falling back to simplified prediction")
        # Simplified fallback to at least get a regression line
        coeffs = np.array(model.rx2('b'))
        if len(coeffs) > 1:  # Multiple coefficients
            if model.rx2('int.incl')[0]:  # Model includes intercept
                pred = coeffs[0] + coeffs[mod_pos] * xs
            else:
                pred = coeffs[mod_pos-1] * xs
        else:  # Single coefficient
            pred = coeffs[0] * xs
        ci_lb = pred - 1.96 * 0.5  # Rough approximation
        ci_ub = pred + 1.96 * 0.5  # Rough approximation
    
    ### plot
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    
    if isinstance(point_color, str):
        point_color = [point_color] * len(xi)
    if isinstance(point_fill, str):
        point_fill = [point_fill] * len(xi)
        
    if colorby is not None: # create a color map for the points
        colorby = np.array(colorby)
        unique_colors = np.unique(colorby)
        color_map = {color: plt.cm.viridis(i / len(unique_colors)) for i, color in enumerate(unique_colors)}
        point_colors = [color_map[color] for color in colorby]
    else:
        point_colors = point_fill
    
    sorted_indices = np.argsort(-norm_weights)  # larger points plot at the back
    for i in sorted_indices:
        sc = ax.scatter(xi[i], yi[i], s=norm_weights[i]**2, 
                  edgecolor=point_color[i], facecolor=point_colors[i] if colorby is not None else point_fill[i],
                  zorder=3, alpha=0.8)
        
    if ci and shade:    # add confidence interval
        if isinstance(shade, str):
            poly_color = shade
        else:
            poly_color = ci_color
        
        ax.fill_between(xs, ci_lb, ci_ub, color=poly_color, alpha=0.3)
    
    if ci:  # add confidence interval lines
        ax.plot(xs, ci_lb, color=ci_line_color, linestyle='--', linewidth=1)
        ax.plot(xs, ci_ub, color=ci_line_color, linestyle='--', linewidth=1)
    if pi:  # add prediction interval lines
        ax.plot(xs, pred_lb, color=ci_line_color, linestyle=':', linewidth=1)
        ax.plot(xs, pred_ub, color=ci_line_color, linestyle=':', linewidth=1)
        if shade:
            ax.fill_between(xs, pred_lb, pred_ub, color=ci_color, alpha=0.1)
    
    ax.scatter([], [], edgecolor=point_color[0], facecolor=point_fill[0], alpha=0.8, label='Studies')
    # Get coefficients and create a polynomial string for the legend
    coeffs = np.array(model.rx2('b'))
    polynomial = Polynomial(coeffs[mod_pos])
    ax.plot(xs, pred, color=line_color, linewidth=2, label=f'Regression line: {polynomial}{x_mod}')    # plot regression line with equation

    ax.plot([],[], linestyle='--', color=ci_line_color, label='95% Confidence interval') if ci else None
    ax.plot([],[], linestyle=':', color=ci_line_color, label='95% Prediction interval') if pi else None
    
    ### formatting
    ax.axhline(y=refline, color='gray', label='Zero effect level', linestyle='-', linewidth=1) if refline is not None else None    # add reference line
    ax.set_xlabel(xlab if xlab else model_comps['predictors'][mod_pos], fontsize=12)
    ax.set_ylabel(ylab if ylab else f'Effect size ({model_comps['response']})', fontsize=12)
    ax.set_title(title, fontsize=14) if title else None
    ax.set_xlim(xlim) if xlim else predlim if predlim else None
    ax.set_ylim(ylim) if ylim else None    
    ax.grid(True, linestyle='--', alpha=0.3)

    ax.legend(fontsize=10)

    if ax is None:  # Only apply tight_layout if we created a new figure
        fig.tight_layout()
    
    return fig, ax


def add_climatology_lines_to_plot(ax, future_global_anomaly_df, scenario_var, xlim):
    scenarios = future_global_anomaly_df['scenario'].unique()
    scenario_colours = sns.color_palette("Reds", len(scenarios))
    scenario_colour_dict = {scenario: scenario_colours[i] for i, scenario in enumerate(scenarios)}
    original_ylim = ax.get_ylim()  # Get the original y-axis limits
    scenario_lines = []
    for scenario in scenarios:
        # Add climatology lines to the plot
        predicted_effect_sizes = future_global_anomaly_df[
            (future_global_anomaly_df['time_frame'] == 2090) &
            (future_global_anomaly_df['scenario'] == scenario)
        ][:][f'mean_{scenario_var}_anomaly']

        # Plot vertical lines for each predicted effect size
        for effect_size in predicted_effect_sizes:
            line = ax.vlines(
                x=effect_size,
                ymin=original_ylim[0],  # Use the fixed original y-axis minimum
                ymax=original_ylim[1],  # Use the fixed original y-axis maximum
                color=scenario_colour_dict[scenario],
                linestyle='--',  
                label=SCENARIO_MAP[scenario],  
                zorder=5, 
            )
            scenario_lines.append(line)
    # Crop to y lim
    ax.set_ylim(original_ylim[0], original_ylim[1])
    # Add second legend for the climatology lines
    # legend1 = ax.get_legend()
    # legend2 = plt.legend(
    #     handles=scenario_lines, 
    #     title='SSP scenarios', 
    #     loc='lower left', 
    # )
    # # Add the new legend to the graph along with the original
    # ax.add_artist(legend1)
    # ax.add_artist(legend2)
    return ax


def meta_regplot_plotly(model, model_comps: tuple, x_mod:str, level=95, ci=True, pi=True,   # TODO: finish updating this
                      point_size="seinv", point_color='black', point_fill='white', 
                      doi_list=None,
                      line_color='blue', ci_color='rgba(173, 216, 230, 0.3)', ci_line_color='blue',
                      xlab=None, ylab=None, xlim=None, ylim=None, predlim=None,
                      refline=0, figsize=(900, 600), title=None):
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
    # Enable automatic conversion between R and pandas
    pandas2ri.activate()
    
    # Extract model components
    yi = np.array(model.rx2('yi.f'))
    vi = np.array(model.rx2('vi.f'))
    X = np.array(model.rx2('X.f'))
    xi = X[:, mod_pos]  # moderator of interest
    
    # Handle missing values
    mask = ~np.isnan(yi) & ~np.isnan(vi) & ~np.isnan(xi)
    if not all(mask):
        yi = yi[mask]
        vi = vi[mask]
        xi = xi[mask]
        if doi_list is not None:
            doi_list = [doi_list[i] for i in range(len(mask)) if mask[i]]
    
    # Create weight vector for point sizes
    if point_size == "seinv":
        weights = 1 / np.sqrt(vi)
    elif point_size == "vinv":
        weights = 1 / vi
    elif isinstance(point_size, (list, np.ndarray)):
        weights = np.array(point_size)
    else:
        weights = np.ones_like(yi)
    
    # Normalize weights for point sizes
    if len(weights) > 0:
        min_w, max_w = min(weights), max(weights)
        if max_w - min_w > np.finfo(float).eps:
            norm_weights = 30 * (weights - min_w) / (max_w - min_w) + 1
        else:
            norm_weights = np.ones_like(weights) * 20
    else:
        norm_weights = np.ones_like(yi) * 20
    
    # Create sequence of x values for the regression line
    range_xi = max(xi) - min(xi)
    predlim = (min(xi) - 0.1*range_xi, max(xi) + 0.1*range_xi) if predlim is None else predlim
    xs = np.linspace(predlim[0], predlim[1], 1000)
    
    r_xs = ro.FloatVector(xs)
    
    # Create prediction function in R
    predict_function = ro.r('''
    function(model, xs, mod_pos, level) {
        # Get mean values for all predictors
        X_means <- colMeans(model$X.f)
        
        # Create new data for predictions
        Xnew <- matrix(rep(X_means, each=length(xs)), nrow=length(xs))
        colnames(Xnew) <- colnames(model$X.f)
        
        # Set the moderator of interest to the sequence of values
        Xnew[,mod_pos] <- xs
        
        # Remove intercept if present in the model
        if (model$int.incl) {
            Xnew <- Xnew[,-1, drop=FALSE]
        }
        
        # Make predictions
        pred <- predict(model, newmods=Xnew, level=(level/100))
        
        # Return results
        return(pred)
    }
    ''')
    
    # Get predictions
    try:
        pred_res = predict_function(model, r_xs, mod_pos + 1, level)  # R is 1-indexed
        pred = np.array(pred_res.rx2('pred'))
        ci_lb = np.array(pred_res.rx2('ci.lb'))
        ci_ub = np.array(pred_res.rx2('ci.ub'))
        pred_lb = np.array(pred_res.rx2('pi.lb'))
        pred_ub = np.array(pred_res.rx2('pi.ub'))
    except Exception as e:
        print(f"Error in prediction: {e}")
        print("Falling back to simplified prediction")
        # Simplified fallback to at least get a regression line
        coeffs = np.array(model.rx2('b'))
        if len(coeffs) > 1:  # Multiple coefficients
            if model.rx2('int.incl')[0]:  # Model includes intercept
                pred = coeffs[0] + coeffs[mod_pos] * xs
            else:
                pred = coeffs[mod_pos-1] * xs
        else:  # Single coefficient
            pred = coeffs[0] * xs
        ci_lb = pred - 1.96 * 0.5  # Rough approximation
        ci_ub = pred + 1.96 * 0.5  # Rough approximation
        pred_lb = pred - 1.96 * 1.0  # Rough approximation for prediction interval
        pred_ub = pred + 1.96 * 1.0  # Rough approximation for prediction interval
    
    # Create Plotly figure
    fig = go.Figure()
    
    
    # Prepare study point data
    hover_text = []
    if doi_list is not None:
        for i in range(len(xi)):
            hover_text.append(f"DOI: {doi_list.iloc[i]}<br>X: {xi[i]:.4f}<br>Y: {yi[i]:.4f}<br>SE: {np.sqrt(vi[i]):.4f}")
    else:
        for i in range(len(xi)):
            hover_text.append(f"X: {xi[i]:.4f}<br>Y: {yi[i]:.4f}<br>SE: {np.sqrt(vi[i]):.4f}")
    
    # Create colormap if DOIs are provided
    if doi_list is not None:
        unique_dois = list(set(doi_list))
        colorscale = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                     '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        # Extend color palette if needed
        if len(unique_dois) > len(colorscale):
            import plotly.express as px
            colorscale = px.colors.qualitative.Plotly + px.colors.qualitative.D3
        
        color_map = {doi: colorscale[i % len(colorscale)] for i, doi in enumerate(unique_dois)}
        marker_colors = [color_map[doi] for doi in doi_list]
    else:
        marker_colors = [point_fill] * len(xi)
    
    # Add study points
    # Scale marker sizes appropriately for Plotly
    fig.add_trace(go.Scatter(
        x=xi,
        y=yi,
        mode='markers',
        marker=dict(
            size=norm_weights * (figsize[0] / 1000),  # Scale marker size based on figure width
            color=marker_colors,
            line=dict(width=2, color=point_color)
        ),
        text=hover_text,
        hoverinfo='text',
        name='Studies'
    ))
    
    # Add confidence intervals
    if ci:
        # Add CI filling
        fig.add_trace(go.Scatter(
            x=list(xs) + list(xs[::-1]),
            y=list(ci_ub) + list(ci_lb[::-1]),
            fill='toself',
            fillcolor=ci_color,
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo='skip',
            showlegend=False,
            name='95% Confidence Interval'
        ))
        
        # Add CI lines
        fig.add_trace(go.Scatter(
            x=xs,
            y=ci_lb,
            mode='lines',
            line=dict(color=ci_line_color, width=1, dash='dash'),
            hoverinfo='skip',
            showlegend=True,
            name='95% Confidence Interval'
        ))
        
        fig.add_trace(go.Scatter(
            x=xs,
            y=ci_ub,
            mode='lines',
            line=dict(color=ci_line_color, width=1, dash='dash'),
            hoverinfo='skip',
            showlegend=False
        ))
    
    # Add prediction intervals
    if pi:
        # Add PI filling
        fig.add_trace(go.Scatter(
            x=list(xs) + list(xs[::-1]),
            y=list(pred_ub) + list(pred_lb[::-1]),
            fill='toself',
            fillcolor='rgba(173, 216, 230, 0.1)',
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo='skip',
            showlegend=False,
            name='95% Prediction Interval'
        ))
        
        # Add PI lines
        fig.add_trace(go.Scatter(
            x=xs,
            y=pred_lb,
            mode='lines',
            line=dict(color=ci_line_color, width=1, dash='dot'),
            hoverinfo='skip',
            showlegend=True,
            name='95% Prediction Interval'
        ))
        
        fig.add_trace(go.Scatter(
            x=xs,
            y=pred_ub,
            mode='lines',
            line=dict(color=ci_line_color, width=1, dash='dot'),
            hoverinfo='skip',
            showlegend=False
        ))
    
    # Add regression line
    fig.add_trace(go.Scatter(
        x=xs,
        y=pred,
        mode='lines',
        line=dict(color=line_color, width=2),
        name='Regression line'
    ))
    
    # Add reference line if specified
    if refline is not None:
        fig.add_trace(go.Scatter(
            x=[predlim[0], predlim[1]],
            y=[refline, refline],
            mode='lines',
            line=dict(color='gray', width=1),
            name='Zero effect level'
        ))
    
    # Update layout
    fig.update_layout(
        title=title,
        xaxis=dict(
            title=xlab,
            range=xlim if xlim else predlim
        ),
        yaxis=dict(
            title=ylab,
            range=ylim
        ),
        width=figsize[0],
        height=figsize[1],
        hovermode='closest',

        template="plotly_white"
    )
    
    return fig


def interpolate_spline(xs: np.ndarray, ys: np.ndarray, npoints=100, k=2):
    """Interpolates a spline for given x and y values."""
    x_fine = np.linspace(min(xs), max(xs) + 10, 100)
    spline = interpolate.splrep(xs, ys, k=k)
    smooth = interpolate.splev(x_fine, spline)
    return x_fine, smooth


def plot_spatial_effect_distribution(predictions_df, var_to_plot: str = 'predicted_effect_size', time_frame=2090, figsize=(10, 10), dpi=300, title:str='Spatial distribution of predicted effect sizes for SSP scenarios', cbar_label:str='Predicted Effect Size (Hedges\' g)', reverse_cmap:bool=False):
    """
    Plots the spatial distribution of predicted effect sizes for SSP scenarios on a map.

    Parameters:
        predictions_df (pd.DataFrame): DataFrame containing predictions with columns ['longitude', 'latitude', 'scenario', 'time_frame', 'predicted_effect_size'].
        time_frame (int): The time frame to filter the data for (default is 2100).
        figsize (tuple): Size of the figure (default is (10, 10)).
        dpi (int): Resolution of the figure (default is 300).
    # TODO: future thoughts
    - size as e.g. uncertainty, number of datapoints
    - interpolate within reef presence to show effect on whole ecosystem
    """
    ssp_scenarios = predictions_df['scenario'].unique()

    # Create figure
    fig, axes = plt.subplots(len(ssp_scenarios), 1, figsize=figsize, subplot_kw={'projection': ccrs.PlateCarree()}, dpi=dpi)

    # mean_predictions_df = predictions_df[predictions_df['percentile'] == "mean"]
    mean_predictions_df = predictions_df.copy()
    # Normalize color map to the range of predicted effect sizes    
    min_effects, max_effects = mean_predictions_df[var_to_plot].min(), mean_predictions_df[var_to_plot].max()

    for i, (name, scenario) in enumerate(zip(['SSP 1-2.6', 'SSP 2-4.5', 'SSP 3-7.0', 'SSP 5-8.5'], ssp_scenarios)):
        # Add map features
        axes[i] = format_geo_axes(axes[i], extent=[-180, 180, -40, 40])

        # Filter data for the specific scenario and time frame
        data_to_plot = mean_predictions_df[(mean_predictions_df['scenario'] == scenario) & (mean_predictions_df['time_frame'] == time_frame)]
        # get the mean values
        # data_to_plot = data_to_plot[data_to_plot['percentile'] == "mean"]
        
        if data_to_plot.empty:
            raise ValueError(f"No data available for scenario {scenario} at time frame {time_frame}.")
        data_to_plot = data_to_plot.sort_values(by=var_to_plot, ascending=True)

        norm = plt.Normalize(vmin=min_effects, vmax=max_effects)    # normalise colour map

        sc = axes[i].scatter(
            data_to_plot['longitude'],
            data_to_plot['latitude'],
            c=data_to_plot[var_to_plot],
            cmap=plt.cm.Reds_r if reverse_cmap else plt.cm.Reds,
            norm=norm,
            s=5,
            edgecolor='k',
            linewidth=0.3,
            transform=ccrs.PlateCarree(),
            alpha=0.8,
        )

        # Add title to the left of each axis
        axes[i].set_title(name, loc='left', fontsize=8)

    # Add overall colorbar for the collection of axes - position it inside the figure
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)  # Make room for colorbar at bottom
    cbar_ax = fig.add_axes([0.125, 0.1, 0.75, 0.02])  # [lower-left-x, lower-left-y, width, height]
    fig.colorbar(sc, cax=cbar_ax, orientation='horizontal', label=cbar_label)
    cbar_ax.tick_params(labelsize=8)  # Adjust colorbar tick size

    plt.suptitle(title, fontsize=8)
    plt.show()
    
    return fig, axes


def plot_global_timeseries(data, plot_vars=['sst', 'ph'], plot_var_labels: list[str] = {'sst': "Sea surface temperature anomaly (°C)", 'ph': "pH$_{total}$ anomaly"}, time_discontinuity=2025, figsize=(10, 10), dpi=300):
    """
    Plots timeseries of global impacts for given scenarios and variables.

    Parameters:
        data (pd.DataFrame): DataFrame containing the data to plot.
        plot_vars (list): List of variables to plot (e.g., ['sst', 'ph']).
        time_discontinuity (int): Year separating historical and forecast data.
        figsize (tuple): Size of the figure.
        dpi (int): Resolution of the figure.
    """
    scenarios = data.scenario.unique()
    # Calculate dimensions
    total_rows = 1 + len(scenarios) + 1
    n_cols = len(plot_vars)

    # Create figure and GridSpec
    fig = plt.figure(figsize=figsize, dpi=dpi)
    gs = gridspec.GridSpec(total_rows, n_cols, height_ratios=[1] + [0.4] + [1] * len(scenarios))

    # Create axes
    ax = np.empty((total_rows, n_cols), dtype=object)
    for i in range(total_rows):
        if i == 1:  # Skip spacer row
            continue
        for j in range(n_cols):
            ax[i, j] = fig.add_subplot(gs[i, j])

    # Add annotations
    fig.text(0.5, 0.92, "Predicted global conditions for each scenario", ha='center', fontsize=12)
    fig.text(0.5, 0.7, "Resulting effect sizes, by scenario", ha='center', fontsize=12)

    # Define colors and formatting
    historic_colour, historic_alpha = 'darkgrey', 0.5
    forecast_alpha, zero_effect_colour, zero_effect_alpha = 0.2, "black", 0.5
    scenario_colours = sns.color_palette("Reds", len(scenarios))
    scenario_colour_dict = {scenario: scenario_colours[i] for i, scenario in enumerate(scenarios)}

    # Flatten axes for easier iteration
    axes = [ax[i, j] for i in range(total_rows) for j in range(n_cols) if i != 1]

    x_points = data['time_frame'].unique()
    
    for i, axis in enumerate(axes):
        plot_var = plot_vars[i % n_cols]
        data_subset = data[data['scenario_var'] == plot_var]
        if i < n_cols:  # Forecast trajectories
            for scenario in scenarios:
                forecast_subset = data_subset[data_subset['scenario'] == scenario]
                x_fine, mean_spline = interpolate_spline(
                    x_points, forecast_subset['anomaly_value_mean'])
                x_fine, low_spline = interpolate_spline(
                    x_points, forecast_subset['anomaly_value_p10'])
                x_fine, up_spline = interpolate_spline(
                    x_points, forecast_subset['anomaly_value_p90'])

                # Masks and formatting
                historic_mask = x_fine < time_discontinuity
                forecast_mask = x_fine >= time_discontinuity

                axis.plot(x_fine[forecast_mask], mean_spline[forecast_mask], '--', alpha=0.7, color=scenario_colour_dict[scenario])
                axis.fill_between(x_fine[forecast_mask], low_spline[forecast_mask], up_spline[forecast_mask], alpha=forecast_alpha, color=scenario_colour_dict[scenario], linewidth=0)

            axis.plot(x_fine[historic_mask], mean_spline[historic_mask], '--', color=historic_colour)
            axis.fill_between(x_fine[historic_mask], low_spline[historic_mask], up_spline[historic_mask], alpha=historic_alpha, color=historic_colour, linewidth=0)

            annotate_var = f'{plot_var_labels[plot_var]}'
            axis.annotate(annotate_var, xy=(0.05, 1.1), xycoords='axes fraction', ha='left', fontsize=10)
        else:  # Effect sizes
            scenario = scenarios[(i - n_cols) // n_cols]
            subset = data_subset[data_subset['scenario'] == scenario]

            x_points = list(subset['time_frame'].unique())
            x_fine, mean_spline = interpolate_spline(x_points, subset['predicted_effect_size_mean'])
            x_fine, up_spline = interpolate_spline(x_points, subset['predicted_effect_size_p90'])
            x_fine, low_spline = interpolate_spline(x_points, subset['predicted_effect_size_p10'])

            axis.plot(x_fine[forecast_mask], mean_spline[forecast_mask], '--', alpha=0.7, color=scenario_colour_dict[scenario])
            axis.fill_between(x_fine[forecast_mask], low_spline[forecast_mask], up_spline[forecast_mask], alpha=forecast_alpha, color=scenario_colour_dict[scenario], linewidth=0)
            axis.plot(x_fine[historic_mask], mean_spline[historic_mask], '--', color=historic_colour)
            axis.fill_between(x_fine[historic_mask], low_spline[historic_mask], up_spline[historic_mask], alpha=historic_alpha, color=historic_colour, linewidth=0)
            axis.hlines(0, xmin=x_fine.min(), xmax=x_fine.max(), color=zero_effect_colour, alpha=zero_effect_alpha)

    # Add legend
    handles = [
        plt.Line2D([], [], linestyle='--', color=historic_colour, label='Historical'),
        plt.Line2D([], [], linestyle='-', color=zero_effect_colour, label='Zero effect', alpha=zero_effect_alpha)
    ] + [
        plt.Line2D([], [], linestyle='--', color=scenario_colours[i], label=SCENARIO_MAP[scenario.lower()])
        for i, scenario in enumerate(scenarios)
    ]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 0.98), ncol=len(handles), fontsize=8, frameon=False)

    # Set y-labels, x-labels, limits, and formatting
    global_min_ylim, global_max_ylim = float('inf'), float('-inf')

    for i, axis in enumerate(axes):
        if i >= n_cols:
            ymin, ymax = axis.get_ylim()
            global_min_ylim = min(global_min_ylim, ymin)
            global_max_ylim = max(global_max_ylim, ymax)

    for i, axis in enumerate(axes):
        axis.tick_params(axis='both', labelsize=8)
        axis.set_xlim(1995, 2100)
        if i >= n_cols: # effect sizes
            # axis.set_ylabel("Effect size", fontsize=8)
            # Set y-ticks to include 0 and be spaced every 0.5
            # ymin = np.floor(global_min_ylim * 2) / 2  # Round down to nearest 0.5
            # ymax = np.ceil(global_max_ylim * 2) / 2   # Round up to nearest 0.5
            # ticks = np.arange(ymin, ymax + 0.1, 0.5)  # Add 0.1 to include ymax
            # # Make sure 0 is included in the ticks
            # if 0 not in ticks:
            #     ticks = np.sort(np.append(ticks, 0))
            # axis.set_yticks(ticks)

            axis.set_ylim(global_min_ylim, global_max_ylim)
            if i < len(axes) - n_cols:
                axis.set_xticks([])
            else:
                axis.set_xlabel("Year", fontsize=8)

    plt.show()
    return fig, ax


def save_fig(
    fig: object,
    fig_name: str,
    run_key: str = None,
) -> None:
    print('Saving figure to', config.fig_dir)
    config.fig_dir.mkdir(parents=True, exist_ok=True)
    run_key = run_key + "_" + utils.get_formatted_timestamp() if run_key else utils.get_formatted_timestamp()
    
    fig_fp = config.fig_dir / f"{fig_name}_{run_key}.png"
    fig.savefig(fig_fp, dpi=300)
    
    print(f"Figure saved to {fig_fp}")
    
    
### single-use functions, here for tidyness
def plot_study_timeseries(df: pd.DataFrame, ax=None, colorby='core_grouping') -> plt.Axes:
    """
    Plot the temporal distribution of studies and observation counts.
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(9, 3), dpi=300)

    # Drop NA from year columns
    dates_df = df.dropna(subset=["year"])

    # Count occurrences of year of each doi
    year_counts = df.groupby('year')['doi'].nunique()

    # Define smaller font size for most elements
    small_fontsize = 0.8 * plt.rcParams['font.size']

    # Create the bar chart
    ax.bar(year_counts.index, year_counts.values, color='royalblue', width=150, alpha=0.5, edgecolor='black', linewidth=0.5)
    ax.set_ylabel("Number of studies", fontsize=plt.rcParams['font.size'])
    ax.tick_params(axis='both', which='major', labelsize=small_fontsize)

    # Set y-ticks to appear every 5 units
    max_count = year_counts.max()
    y_ticks = np.arange(0, max_count + 5, 5)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_ticks, fontsize=small_fontsize)

    ax.grid(axis='y', linestyle='--', alpha=0.6)

    # Group df by selected group
    grouped_df = df.groupby([colorby, 'year'])

    # Sum total n for each unique group and by year
    group_counts = grouped_df['n'].sum()

    # Plot number of observations each year, by selected group
    unique_group = df[colorby].unique()
    n_ax = ax.twinx()
    group_palette = sns.color_palette("colorblind", len(unique_group))
    group_color_map = dict(zip(unique_group, group_palette))

    for group in unique_group:
        group_data = group_counts[group_counts.index.get_level_values(colorby) == group]
        n_ax.scatter(group_data.index.get_level_values('year'), group_data.values,
                     color=group_color_map[group], alpha=1, s=20, label=group, edgecolor='white', linewidth=0.5)

    n_ax.set_ylabel("Number of observations", rotation=-90, labelpad=12, fontsize=plt.rcParams['font.size'])
    n_ax.tick_params(axis='y', labelsize=small_fontsize)

    # Align y-axis ticks with the bar chart
    max_observations = group_counts.max()
    n_yticks = np.arange(0, 4001, 1000)
    n_ax.set_yticks(n_yticks)
    n_ax.set_yticklabels(n_yticks, fontsize=small_fontsize)

    # Create compact legend with smaller font
    legend = n_ax.legend(title=colorby.capitalize().replace('_', ' '),
                         loc='upper left',
                         fontsize=small_fontsize,
                         framealpha=0.7,
                         title_fontsize=small_fontsize)
    legend.get_title().set_ha('left')

    plt.xlabel('Year', fontsize=plt.rcParams['font.size'])
    plt.title("Temporal distribution of studies", fontsize=plt.rcParams['font.size'])
    plt.tight_layout()
    return ax


def create_faceted_dotplot_with_percentages(df: pd.DataFrame, top_n: int = 10, groupby: str = 'taxa', omission_threshold: int = 10) -> plt.Figure:
    """
    Create a faceted dotplot with percentages for the top N species in each taxonomic group.
    """
    count_df = df.groupby(['species', 'genus', groupby])['n'].sum().reset_index()
    
    # Get taxa that meet the omission threshold
    group_counts = count_df.groupby(groupby)['n'].sum()
    unique_in_group = sorted([group for group in group_counts.index 
                          if group_counts[group] >= omission_threshold])
    n_in_group = len(unique_in_group)
    
    fig, axes = plt.subplots(1, n_in_group, figsize=(5 * n_in_group, 10), sharey=False)
    
    if n_in_group == 1:
        axes = [axes]
    
    for i, group in enumerate(unique_in_group):
        ax = axes[i]
        
        group_data = count_df[count_df[groupby] == group].copy()
        total_count = group_data['n'].sum()
        
        group_data.n = group_data.n.astype(int)
        topn = group_data.nlargest(top_n, 'n')
        other_species = group_data[~group_data['species'].isin(topn['species'])]
        
        if len(other_species) > 0:
            other_count = other_species['n'].sum()
            other_percentage = (other_count / total_count) * 100
            
            other_sum = pd.DataFrame({
                'species': [f'Other ({len(other_species)} species)'],
                'genus': ['Various'],
                groupby: [group],
                'n': [other_count]
            })
            
            plot_data = pd.concat([topn, other_sum], ignore_index=True)
        else:
            plot_data = topn
        
        plot_data['species_label'] = plot_data.apply(
            lambda row: f"{row['species']} ({(row['n']/total_count*100):.1f}%)", axis=1
        )
        
        plot_data = plot_data.sort_values('n', ascending=True)
        
        unique_genera = sorted(plot_data['genus'].unique())
        genus_palette = dict(zip(unique_genera, sns.color_palette("Spectral", len(unique_genera))))
        # specify 'various' genus as black
        genus_palette['Various'] = 'black'
        
        colors = [genus_palette[genus] for genus in plot_data['genus']]
        y_positions = range(len(plot_data))
        ax.scatter(plot_data['n'], y_positions, c=colors, s=100)
        
        # add count labels
        for j, (_, row) in enumerate(plot_data.iterrows()):
            ax.text(row['n'] + 0.02 * ax.get_xlim()[1] + len(str(row['n']))*0.02*ax.get_xlim()[1], j, f"{row['n']}", va='center')
        for j, (_, row) in enumerate(plot_data.iterrows()):
            ax.plot([0, row['n']], [j, j], 'gray', alpha=0.3)
        
        # formatting
        ax.set_yticks(y_positions)
        ax.set_yticklabels(plot_data['species_label'])
        max_count = plot_data['n'].max()
        ax.set_xlim(0, max_count * 1.3)  # extra space for the count labels
        ax.grid(axis='x', linestyle='--', alpha=0.7)
        ax.set_xlabel('Count', fontsize=12)
        if i == 0:
            ax.set_ylabel('Species', fontsize=12)
        # legend
        legend_handles = [Line2D([0], [0], marker='o', color='w',
                              markerfacecolor=genus_palette[genus], markersize=10)
                         for genus in unique_genera]
        
        total_species = len(group_data)
        ax.set_title(f'{group.capitalize()}\n(Total: {int(total_count)} samples, {total_species} species)', fontsize=14)
        ax.legend(legend_handles, unique_genera, 
                 title='Genus', loc='lower right')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.suptitle(f'Top {int(top_n)} Species Counts by Taxonomic Group', fontsize=16)
    
    return fig


def plot_filtered_meow_regions(filter_dict: dict={'Lat_Zone': 'Tropical'}, 
                               extent=(-180, 180, -40, 40),
                               figsize=(15, 10),
                               dpi=300,
                               cmap='viridis'):
    """
    Plots filtered marine ecoregions of the world (MEOW) on a map.

    Parameters:
    - filter_dict: Dictionary specifying column_name: value pairs to filter the MEOW shapefile.
    - extent: Tuple specifying the map extent in the format (min_lon, max_lon, min_lat, max_lat).
    - figsize: Figure size as (width, height) in inches.
    - dpi: Resolution in dots per inch.
    - cmap: Colormap to use for the regions.

    Returns:
    - fig, ax: Matplotlib figure and axis objects for further customization.
    """
    # Read in shapefiles from directory
    gdf = gpd.read_file(config.climatology_data_dir / "MEOW" / "meow_ecos.shp")
    
    # Filter areas_df based on filter_dict
    if filter_dict:
        for col, value in filter_dict.items():
            gdf = gdf[gdf[col] == value]

    # Create figure and axis
    fig, ax = plt.subplots(1, 1, figsize=figsize, subplot_kw={'projection': ccrs.PlateCarree()}, dpi=dpi)

    ax = format_geo_axes(ax, extent=extent)

    # Plot filtered areas on world map
    gdf.plot(ax=ax, column='REALM', legend=True, 
             cmap=cmap, alpha=0.5, 
             legend_kwds={'bbox_to_anchor': (0.5, -0.3),
                          'ncol': gdf.REALM.nunique(),
                          'loc': 'lower center',
                          'fontsize': 8})

    # Add gridlines
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
    gl.xlabel_style = {'size': 8}
    gl.ylabel_style = {'size': 8}
    gl.top_labels = False
    gl.right_labels = False

    return fig, ax


def plot_areas_with_study_locations(df, filter_dict: dict={'Lat_Zone': 'Tropical'}, extent=(-180, 180, -40, 40)):
    """
    Plots areas and study locations on a world map.

    Parameters:
    - df: pandas.DataFrame containing study locations with latitude and longitude.
    - filter_dict: Dictionary specifying column_name: value pairs to filter areas_df.
    - extent: Tuple specifying the map extent in the format (min_lon, max_lon, min_lat, max_lat).
    """
    # Create base map with filtered regions
    fig, ax = plot_filtered_meow_regions(filter_dict=filter_dict, extent=extent)

    # Plot study locations
    locs_df = df.drop_duplicates('doi', keep='first')
    for i, (doi, data) in enumerate(locs_df.iterrows()):
        # Convert coordinates to float if needed
        lat = float(data["latitude"]) if isinstance(data["latitude"], str) else data["latitude"]
        lon = float(data["longitude"]) if isinstance(data["longitude"], str) else data["longitude"]

        # Only plot if coordinates are valid
        if pd.notna(lat) and pd.notna(lon):
            ax.plot(lon, lat, 'o', markeredgecolor='white', markersize=5, 
                    color='red', transform=ccrs.PlateCarree())

    # Add title
    ax.set_title('Spatial distribution of studies', fontsize=10)
    
    plt.show()
    
    return fig, ax
    
    
def format_geo_axes(ax: plt.Axes, extent: tuple | list = (-180, 180, -40, 50)) -> plt.Axes:
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, facecolor='white')
    ax.add_feature(cfeature.OCEAN, alpha=0.3)
    ax.add_feature(cfeature.COASTLINE, edgecolor='lightgray', zorder=-1)
    ax.add_feature(cfeature.BORDERS, linestyle=':', edgecolor='gray', alpha=0.1, zorder=-1)

    return ax


def plot_effect_size_distributions(df, effect_sizes=['cohens_d', 'hedges_g', 'relative_calcification', 'absolute_calcification', 'st_relative_calcification', 'st_absolute_calcification'], outlier_limits=None, title='Effect Size Distributions'):
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
        ax.hist(data, bins=100, color='skyblue', edgecolor='black')
        ax.set_xlabel(effect_size, fontsize=6)
        ax.set_ylabel('Frequency', fontsize=6)
        ax.grid(ls='--', alpha=0.5)
        ax.vlines(0, *ax.get_ylim(), color='red', linestyle='--', linewidth=1)

        # Add boxplot above using divider
        divider = make_axes_locatable(ax)
        box_ax = divider.append_axes("top", size="20%", pad=0.01, sharex=ax)
        box_ax.boxplot(data, vert=False, patch_artist=True,
                       boxprops=dict(facecolor='lightgray'))
        box_ax.axis('off')
        for outlier in box_ax.findobj(match=plt.Line2D):
            outlier.set_markersize(3)
            outlier.set_alpha(0.1)

        # Optional: log scale if necessary
        if max([p.get_height() for p in ax.patches]) > 10:
            ax.set_yscale('log')

        ax.tick_params(axis='both', which='major', labelsize=6)

    plt.suptitle(title, fontsize=8)
    plt.tight_layout()


def set_up_regression_plot(var:str):
    if var == 'phtot':
        xlab = '$\\Delta$ pH'
        xlim = (-1, 0)
        predlim = (-1, 0)
        scenario_var = 'ph'
    elif var == 'temp':
        xlab = '$\\Delta$ Temperature ($^\\circ C$)'
        xlim = (0,11)
        predlim = None
        scenario_var = 'sst'
    return xlab, xlim, predlim, scenario_var


def plot_contour(ax, x, y, df, title, legend_label='Calcification Rate'):
    # Create a grid of points
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    for i in range(len(X)): # interpolate the data to create a smooth contour plot
        for j in range(len(Y)):
            # Find the nearest data point
            nearest_idx = ((df['temp'] - X[i, j])**2 + (df['phtot'] - Y[i, j])**2).idxmin()
            Z[i, j] = df.loc[nearest_idx, 'st_calcification']

    # Plot the contour
    contour = ax.contourf(X, Y, Z, levels=20, cmap='viridis')
    ax.set_xlabel('Temperature ($^\\circ C$)')
    ax.set_ylabel('pH$_T$')
    ax.set_title(title)
    plt.colorbar(contour, label=legend_label)


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
    xlim: list=None,
    ylim: list=[0,10],  # this range is ridiculous, adjust accordingly
    plot_in_python: bool = False,
    figsize: tuple[int, int] = (10, 8),
    save_path: [str] = None
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
        import tempfile
        import os
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
            temp_filename = temp_file.name
            
        # Create the plot in R and save to file with high resolution
        # Multiply dimensions by 3 and use high DPI for better quality
        grdevices.png(temp_filename, 
                      width=figsize[0], height=figsize[1], 
                      units='in', res=300)
        
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
        ax.imshow(img, interpolation='nearest')  # Use 'nearest' for sharper image rendering
        ax.axis('off')
        plt.tight_layout(pad=0)  # Reduce padding
        
        # Save if requested
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
        
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
            grdevices.png(str(save_path), width=figsize[0], height=figsize[1], units='in', res=300)
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




### DEPRECATED
# def generic_plot_stacked_hist(df: pd.DataFrame, ax: matplotlib.axes.Axes, filter_col_name: str, filter_col_val: str | float, response_col_name: str, group_col_name: str, group_values_list: list, color_map: dict, title: str="no title") -> None:
#     """
#     Plot a stacked histogram of calcification values for a given unit condition.
    
#     Args:
#         ax (matplotlib.axes.Axes): Matplotlib axes object
#         unit (str): value to plot
#         title (str): Plot title
#         color_map (dict): Dictionary mapping DOI to color
    
#     Returns:
#         None    
#     """
#     filtered_df = df[df[filter_col_name] == filter_col_val]
    
#     # Group data by DOI
#     grouped_data = [filtered_df[filtered_df[group_col_name] == group][response_col_name] for group in group_values_list]
    
#     # Plot histogram
#     ax.hist(grouped_data, bins=50, stacked=True, label=group_values_list, color=[color_map[group] for group in group_values_list], alpha=0.8)
#     ax.set_title(title)
#     ax.set_ylabel("Frequency")
    
    
# def plot_stacked_hist_interactive(df: pd.DataFrame, unit: str, title: str, doi_list: list, color_map: dict) -> None:
#     """
#     Plot an interactive stacked histogram of calcification values for a given unit condition.
    
#     Args:
#         unit (str): value to plot
#         title (str): Plot title
#         color_map (dict): Dictionary mapping DOI to color
    
#     Returns:
#         None    
#     """
#     filtered_df = df[df["new_unit"] == unit]
    
#     # Group data by DOI
#     fig = go.Figure()
#     for doi in doi_list:
#         doi_data = filtered_df[filtered_df["doi"] == doi]["corr_calcification"]
#         fig.add_trace(go.Histogram(
#             x=doi_data,
#             name=doi,
#             marker_color=color_map[doi],
#             opacity=0.75,
#             hoverinfo='x+y+name'
#         ))
    
#     fig.update_layout(
#         barmode='stack',
#         title=title,
#         xaxis_title='Calcification',
#         yaxis_title='Frequency',
#         hovermode='x'
#     )
    
#     fig.show()
    
    
# def plot_stacked_hist(df: pd.DataFrame, ax: matplotlib.axes.Axes, unit: str, title: str, doi_list: list, color_map: dict) -> None:
#     """
#     Plot a stacked histogram of calcification values for a given unit condition.
    
#     Args:
#         ax (matplotlib.axes.Axes): Matplotlib axes object
#         unit (str): value to plot
#         title (str): Plot title
#         color_map (dict): Dictionary mapping DOI to color
    
#     Returns:
#         None    
#     """
#     filtered_df = df[df["new_unit"] == unit]
    
#     # Group data by DOI
#     doi_data = [filtered_df[filtered_df["doi"] == doi]["corr_calcification"] for doi in doi_list]
    
#     # Plot histogram
#     ax.hist(doi_data, bins=50, stacked=True, label=doi_list, color=[color_map[doi] for doi in doi_list], alpha=0.8)
#     ax.set_title(title)
#     ax.set_ylabel("Frequency")
    
    
# def simple_regplot(
#     x: np.ndarray | pd.Series, y: np.ndarray | pd.Series, n_pts: int=100, ax=None, line_kws=None, ci_kws=None
# ):
#     """ Draw a regression line with 95% confidence interval. """
#     ax = plt.gca() if ax is None else ax

#     # calculate best-fit line and interval
#     x_fit = sm.add_constant(x)
#     fit_results = sm.OLS(y, x_fit).fit()

#     eval_x = sm.add_constant(np.linspace(np.min(x), np.max(x), n_pts))
#     pred = fit_results.get_prediction(eval_x)
#     ci = pred.conf_int(alpha=0.05)  # 95% confidence interval

#     # draw the fit line and confidence interval
#     ci_kws = {} if ci_kws is None else ci_kws
#     ax.fill_between(
#         eval_x[:, 1],
#         ci[:, 0],
#         ci[:, 1],
#         alpha=0.3,
#         **ci_kws,
#     )
#     line_kws = {} if line_kws is None else line_kws
#     h = ax.plot(eval_x[:, 1], pred.predicted_mean, **line_kws)

#     return fit_results