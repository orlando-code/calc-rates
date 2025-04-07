import matplotlib.pyplot as plt
import matplotlib
import plotly.graph_objects as go

# R
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri

# stats
import numpy as np
import pandas as pd
import statsmodels.api as sm


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
                 point_size="seinv", point_color='black', point_fill='white', 
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
    
    sorted_indices = np.argsort(-norm_weights)  # larger points plot at the back
    for i in sorted_indices:
        sc = ax.scatter(xi[i], yi[i], s=norm_weights[i]**2, 
                  edgecolor=point_color[i], facecolor=point_fill[i], 
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
    plt.plot([],[], linestyle='--', color=ci_line_color, label='95% Confidence interval')
    plt.plot([],[], linestyle=':', color=ci_line_color, label='95% Prediction interval')    
    
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