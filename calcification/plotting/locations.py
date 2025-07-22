# general
import cartopy.crs as ccrs
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd

from calcification.plotting import plot_utils
from calcification.utils import config


def plot_areas_with_study_locations(
    df, filter_dict: dict = {"Lat_Zone": "Tropical"}, extent=(-180, 180, -40, 40)
):
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
    locs_df = df.drop_duplicates("doi", keep="first")
    for i, (doi, data) in enumerate(locs_df.iterrows()):
        # Convert coordinates to float if needed
        lat = (
            float(data["latitude"])
            if isinstance(data["latitude"], str)
            else data["latitude"]
        )
        lon = (
            float(data["longitude"])
            if isinstance(data["longitude"], str)
            else data["longitude"]
        )

        # Only plot if coordinates are valid
        if pd.notna(lat) and pd.notna(lon):
            ax.plot(
                lon,
                lat,
                "o",
                markeredgecolor="white",
                markersize=5,
                color="red",
                transform=ccrs.PlateCarree(),
            )

    # Add title
    ax.set_title("Spatial distribution of studies", fontsize=10)

    plt.show()

    return fig, ax


def plot_filtered_meow_regions(
    filter_dict: dict = {"Lat_Zone": "Tropical"},
    extent=(-180, 180, -40, 40),
    figsize=(15, 10),
    dpi=300,
    cmap="viridis",
):
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
    fig, ax = plt.subplots(
        1, 1, figsize=figsize, subplot_kw={"projection": ccrs.PlateCarree()}, dpi=dpi
    )

    ax = plot_utils.format_geo_axes(ax, extent=extent)

    # Plot filtered areas on world map
    gdf.plot(
        ax=ax,
        column="REALM",
        legend=True,
        cmap=cmap,
        alpha=0.5,
        legend_kwds={
            "bbox_to_anchor": (0.5, -0.3),
            "ncol": gdf.REALM.nunique(),
            "loc": "lower center",
            "fontsize": 8,
        },
    )

    # Add gridlines
    gl = ax.gridlines(
        draw_labels=True, linewidth=0.5, color="gray", alpha=0.5, linestyle="--"
    )
    gl.xlabel_style = {"size": 8}
    gl.ylabel_style = {"size": 8}
    gl.top_labels = False
    gl.right_labels = False

    return fig, ax
