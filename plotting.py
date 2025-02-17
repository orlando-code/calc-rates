import matplotlib.pyplot as plt
import matplotlib
import pandas as pd


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
    doi_data = [filtered_df[filtered_df["doi"] == doi]["calcification"] for doi in doi_list]
    
    # Plot histogram
    ax.hist(doi_data, bins=50, stacked=True, label=doi_list, color=[color_map[doi] for doi in doi_list], alpha=0.8)
    ax.set_title(title)
    ax.set_ylabel("Frequency")