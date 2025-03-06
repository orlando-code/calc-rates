import matplotlib.pyplot as plt
import matplotlib
import plotly.graph_objects as go

# stats
import numpy as np
import pandas as pd
import statsmodels.api as sm


def generic_plot_stacked_hist(df: pd.DataFrame, ax: matplotlib.axes.Axes, filter_col_name: str, filter_col_val: str | float, response_col_name: str, group_col_name: str, group_values_list: list, color_map: dict, title: str="no title") -> None:
    """
    Plot a stacked histogram of calcification values for a given unit condition.
    
    Args:
        ax (matplotlib.axes.Axes): Matplotlib axes object
        unit (str): value to plot
        title (str): Plot title
        color_map (dict): Dictionary mapping DOI to color
    
    Returns:
        None    
    """
    filtered_df = df[df[filter_col_name] == filter_col_val]
    
    # Group data by DOI
    grouped_data = [filtered_df[filtered_df[group_col_name] == group][response_col_name] for group in group_values_list]
    
    # Plot histogram
    ax.hist(grouped_data, bins=50, stacked=True, label=group_values_list, color=[color_map[group] for group in group_values_list], alpha=0.8)
    ax.set_title(title)
    ax.set_ylabel("Frequency")
    
    
def plot_stacked_hist_interactive(df: pd.DataFrame, unit: str, title: str, doi_list: list, color_map: dict) -> None:
    """
    Plot an interactive stacked histogram of calcification values for a given unit condition.
    
    Args:
        unit (str): value to plot
        title (str): Plot title
        color_map (dict): Dictionary mapping DOI to color
    
    Returns:
        None    
    """
    filtered_df = df[df["new_unit"] == unit]
    
    # Group data by DOI
    fig = go.Figure()
    for doi in doi_list:
        doi_data = filtered_df[filtered_df["doi"] == doi]["corr_calcification"]
        fig.add_trace(go.Histogram(
            x=doi_data,
            name=doi,
            marker_color=color_map[doi],
            opacity=0.75,
            hoverinfo='x+y+name'
        ))
    
    fig.update_layout(
        barmode='stack',
        title=title,
        xaxis_title='Calcification',
        yaxis_title='Frequency',
        hovermode='x'
    )
    
    fig.show()
    
    
def plot_stacked_hist(df: pd.DataFrame, ax: matplotlib.axes.Axes, unit: str, title: str, doi_list: list, color_map: dict) -> None:
    """
    Plot a stacked histogram of calcification values for a given unit condition.
    
    Args:
        ax (matplotlib.axes.Axes): Matplotlib axes object
        unit (str): value to plot
        title (str): Plot title
        color_map (dict): Dictionary mapping DOI to color
    
    Returns:
        None    
    """
    filtered_df = df[df["new_unit"] == unit]
    
    # Group data by DOI
    doi_data = [filtered_df[filtered_df["doi"] == doi]["corr_calcification"] for doi in doi_list]
    
    # Plot histogram
    ax.hist(doi_data, bins=50, stacked=True, label=doi_list, color=[color_map[doi] for doi in doi_list], alpha=0.8)
    ax.set_title(title)
    ax.set_ylabel("Frequency")
    
    
def simple_regplot(
    x: np.ndarray | pd.Series, y: np.ndarray | pd.Series, n_pts: int=100, ax=None, line_kws=None, ci_kws=None
):
    """ Draw a regression line with 95% confidence interval. """
    ax = plt.gca() if ax is None else ax

    # calculate best-fit line and interval
    x_fit = sm.add_constant(x)
    fit_results = sm.OLS(y, x_fit).fit()

    eval_x = sm.add_constant(np.linspace(np.min(x), np.max(x), n_pts))
    pred = fit_results.get_prediction(eval_x)
    ci = pred.conf_int(alpha=0.05)  # 95% confidence interval

    # draw the fit line and confidence interval
    ci_kws = {} if ci_kws is None else ci_kws
    ax.fill_between(
        eval_x[:, 1],
        ci[:, 0],
        ci[:, 1],
        alpha=0.3,
        **ci_kws,
    )
    line_kws = {} if line_kws is None else line_kws
    h = ax.plot(eval_x[:, 1], pred.predicted_mean, **line_kws)

    return fit_results