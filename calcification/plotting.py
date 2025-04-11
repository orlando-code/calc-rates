# general
import matplotlib
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from matplotlib import gridspec
import seaborn as sns
# spatial
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# R
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri

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
    

def meta_regplot(model, mod_pos=None, level=95, ci=True, pi=True, shade=True, 
                 point_size="seinv", point_color='black', point_fill='white', colorby: list=None,
                 line_color='blue', ci_color='lightblue', ci_line_color='blue',
                 xlab=None, ylab=None, xlim=None, ylim=None, predlim=None,
                 refline=0, figsize=(10, 7), title=None):
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
        
    Returns:
        fig, ax (matplotlib.figure.Figure, matplotlib.axes._axes.Axes): Matplotlib figure and axis objects.
    """    
    ### process
    pandas2ri.activate()    # enable automatic conversion between R and pandas
    
    # extract model components
    yi = np.array(model.rx2('yi.f'))
    vi = np.array(model.rx2('vi.f'))
    X = np.array(model.rx2('X.f'))    
    xi = X[:, mod_pos]  # TODO: determine this better dynamically
    
    mask = ~np.isnan(yi) & ~np.isnan(vi) & ~np.isnan(xi).any(axis=0)    # handle missing values
    if not all(mask):
        yi = yi[mask]
        vi = vi[mask]
        xi = xi[mask]
    
    # create weight vector for point sizes
    if point_size == "seinv":
        weights = 1 / np.sqrt(vi)
    elif point_size == "vinv":
        weights = 1 / vi
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
    print(predlim)
    
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
    fig, ax = plt.subplots(figsize=figsize)
    
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
    
    if ci:  # add confidence interval lines
        ax.plot(xs, ci_lb, color=ci_line_color, linestyle='--', linewidth=1)
        ax.plot(xs, ci_ub, color=ci_line_color, linestyle='--', linewidth=1)
    if pi:  # add prediction interval lines
        ax.plot(xs, pred_lb, color=ci_line_color, linestyle=':', linewidth=1)
        ax.plot(xs, pred_ub, color=ci_line_color, linestyle=':', linewidth=1)
        if shade:
            ax.fill_between(xs, pred_lb, pred_ub, color=ci_color, alpha=0.1)
    
    
    plt.scatter([], [], edgecolor=point_color[0], facecolor=point_fill[0], alpha=0.8, label='Studies')
    ax.plot(xs, pred, color=line_color, linewidth=2, label='Regression line')    # plot regression line
    plt.plot([],[], linestyle='--', color=ci_line_color, label='95% Confidence interval') if ci else None
    plt.plot([],[], linestyle=':', color=ci_line_color, label='95% Prediction interval') if pi else None
    
    ### formatting
    ax.axhline(y=refline, color='gray', label='Zero effect level', linestyle='-', linewidth=1) if refline is not None else None    # add reference line
    ax.set_xlabel(xlab, fontsize=12) if xlab else None
    ax.set_ylabel(ylab, fontsize=12) if ylab else None
    ax.set_title(title, fontsize=14) if title else None
    ax.set_xlim(xlim) if xlim else predlim if predlim else None
    ax.set_ylim(ylim) if ylim else None    
    ax.grid(True, linestyle='--', alpha=0.3)

    ax.legend(fontsize=10)
    fig.tight_layout()
    
    return fig, ax


def meta_regplot_plotly(model, mod_pos=None, level=95, ci=True, pi=True, 
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
    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import pandas as pd
    from rpy2.robjects import pandas2ri
    import rpy2.robjects as ro
    
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


def interpolate_spline(xs, ys, npoints=100, k=2):
    
    x_fine = np.linspace(min(xs), max(xs) + 10, 100)
    spline = interpolate.splrep(xs, ys, k=k)
    smooth = interpolate.splev(x_fine, spline)
    return x_fine, smooth


def plot_spatial_effect_distribution(predictions_df, time_frame=2900, figsize=(10, 10), dpi=300):
    """
    Plots the spatial distribution of predicted effect sizes for SSP scenarios on a map.

    Parameters:
        predictions_df (pd.DataFrame): DataFrame containing predictions with columns ['longitude', 'latitude', 'scenario', 'time_frame', 'predicted_effect_size'].
        time_frame (int): The time frame to filter the data for plotting (default is 2100).
        figsize (tuple): Size of the figure (default is (10, 10)).
        dpi (int): Resolution of the figure (default is 300).
    # TODO: future thoughts
    - size as e.g. uncertainty, number of datapoints
    - interpolate within reef presence to show effect on whole ecosystem
    """
    ssp_scenarios = predictions_df['scenario'].unique()

    # Create figure
    fig, axes = plt.subplots(len(ssp_scenarios), 1, figsize=figsize, subplot_kw={'projection': ccrs.PlateCarree()}, dpi=dpi)

    # Normalize color map to the range of predicted effect sizes
    min_effects = predictions_df['predicted_effect_size'].min()
    max_effects = predictions_df['predicted_effect_size'].max()

    for i, (name, scenario) in enumerate(zip(['SSP 1-2.6', 'SSP 2-4.5', 'SSP 3-7.0', 'SSP 5-8.5'], ssp_scenarios)):
        # Add map features
        axes[i].set_extent([-180, 180, -40, 40], crs=ccrs.PlateCarree())
        axes[i].add_feature(cfeature.LAND)
        axes[i].add_feature(cfeature.OCEAN, alpha=0.3)
        axes[i].add_feature(cfeature.COASTLINE, edgecolor='lightgray', zorder=-1)
        axes[i].add_feature(cfeature.BORDERS, linestyle=':', edgecolor='gray', alpha=0.2)

        # Filter data for the specific scenario and time frame
        data_to_plot = predictions_df[(predictions_df['scenario'] == scenario) & (predictions_df['time_frame'] == time_frame)]
        if data_to_plot.empty:
            raise ValueError(f"No data available for scenario {scenario} at time frame {time_frame}.")
            continue
        data_to_plot = data_to_plot.sort_values(by='predicted_effect_size', ascending=True)

        # Normalize color map
        norm = plt.Normalize(vmin=min_effects, vmax=max_effects)

        # Plot scatter points
        sc = axes[i].scatter(
            data_to_plot['longitude'],
            data_to_plot['latitude'],
            c=data_to_plot['predicted_effect_size'],
            cmap=plt.cm.Reds_r,
            norm=norm,
            s=10,
            edgecolor='k',
            linewidth=0.3,
            transform=ccrs.PlateCarree(),
        )

        # Add title to the left of each axis
        axes[i].set_title(name, loc='left', fontsize=8)

    # Add overall colorbar for the collection of axes - position it inside the figure
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)  # Make room for colorbar at bottom
    cbar_ax = fig.add_axes([0.125, 0.1, 0.75, 0.02])  # [lower-left-x, lower-left-y, width, height]
    fig.colorbar(sc, cax=cbar_ax, orientation='horizontal', label='Predicted Effect Size (Hedges\' g)')
    cbar_ax.tick_params(labelsize=8)  # Adjust colorbar tick size

    plt.suptitle('Spatial distribution of predicted effect sizes for SSP scenarios', fontsize=8)
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
            ymin = np.floor(global_min_ylim * 2) / 2  # Round down to nearest 0.5
            ymax = np.ceil(global_max_ylim * 2) / 2   # Round up to nearest 0.5
            ticks = np.arange(ymin, ymax + 0.1, 0.5)  # Add 0.1 to include ymax
            # Make sure 0 is included in the ticks
            if 0 not in ticks:
                ticks = np.sort(np.append(ticks, 0))
            axis.set_yticks(ticks)

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