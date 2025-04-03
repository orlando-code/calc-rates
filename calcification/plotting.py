import matplotlib.pyplot as plt
import matplotlib
import plotly.graph_objects as go

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