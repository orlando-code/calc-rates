# general
import cartopy.crs as ccrs
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd

from calcification.plotting import plot_utils
from calcification.utils import config


def plot_areas_with_study_locations(
    df: pd.DataFrame,
    filter_dict: dict | None = None,
    extent: tuple[float, float, float, float] = (-180, 180, -40, 40),
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot areas and study locations on a world map.

    Args:
        df: DataFrame with 'latitude', 'longitude', and 'doi' columns.
        filter_dict: Optional dict to filter regions.
        extent: (min_lon, max_lon, min_lat, max_lat).

    Returns:
        (fig, ax): Matplotlib Figure and Axes.
    """
    if filter_dict is None:
        filter_dict = {"Lat_Zone": "Tropical"}

    # create base map with filtered regions
    fig, ax = plot_filtered_meow_regions(filter_dict=filter_dict, extent=extent)

    locs_df = df.drop_duplicates("doi", keep="first")
    # plot study locations and plotting
    lat = pd.to_numeric(locs_df["latitude"], errors="coerce")
    lon = pd.to_numeric(locs_df["longitude"], errors="coerce")
    valid = lat.notna() & lon.notna()
    ax.plot(
        lon[valid],
        lat[valid],
        "o",
        markeredgecolor="white",
        markersize=5,
        color="red",
        transform=ccrs.PlateCarree(),
    )

    ax.set_title("Spatial distribution of studies", fontsize=10)
    plt.show()
    return fig, ax


def plot_filtered_meow_regions(
    filter_dict: dict | None = None,
    extent: tuple[float, float, float, float] = (-180, 180, -40, 40),
    figsize: tuple[float, float] = (15, 10),
    dpi: int = 300,
    cmap: str = "viridis",
) -> tuple[plt.Figure, plt.Axes]:
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
    if filter_dict is None:
        filter_dict = {"Lat_Zone": "Tropical"}

    # read in shapefiles from directory
    gdf = gpd.read_file(config.climatology_data_dir / "MEOW" / "meow_ecos.shp")
    for col, val in filter_dict.items():
        gdf = gdf[gdf[col] == val]

    fig, ax = plt.subplots(
        1, 1, figsize=figsize, subplot_kw={"projection": ccrs.PlateCarree()}, dpi=dpi
    )

    ax = plot_utils.format_geo_axes(ax, extent=extent)

    # plot filtered areas on world map
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
    gl = ax.gridlines(
        draw_labels=True, linewidth=0.5, color="gray", alpha=0.5, linestyle="--"
    )
    gl.xlabel_style = {"size": 8}
    gl.ylabel_style = {"size": 8}
    gl.top_labels = False
    gl.right_labels = False

    return fig, ax
