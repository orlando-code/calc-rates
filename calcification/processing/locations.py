import logging
import re

import geopandas as gpd
import googlemaps
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from calcification.utils import config, file_ops, utils

logger = logging.getLogger(__name__)


# --- Google Maps and Coordinate Utilities ---


def get_google_maps_coordinates(locations: list) -> dict:
    """Get coordinates for a list of locations using Google Maps API, with local YAML caching."""
    yaml_path = config.resources_dir / "gmaps_locations.yaml"
    gmaps_coords = file_ops.read_yaml(yaml_path) if yaml_path.exists() else {}
    missing = set(locations) - set(gmaps_coords.keys())
    if not missing:
        logger.info(f"Using cached locations in {yaml_path}")
        return gmaps_coords
    logger.info(f"Fetching {len(missing)} new locations from Google Maps API.")
    api_key = file_ops.read_yaml(config.resources_dir / "api_keys.yaml")[
        "google_maps_api"
    ]
    gmaps_client = googlemaps.Client(key=api_key)
    for loc in tqdm(missing, desc="Querying Google Maps for coordinates"):
        gmaps_coords[loc] = tuple(
            _get_coord_pair_from_google_maps(loc, gmaps_client).values
        )
    file_ops.append_to_yaml(
        {k: v for k, v in gmaps_coords.items() if k in missing}, yaml_path
    )
    logger.info(f"Saved {len(missing)} new locations to {yaml_path}")
    return gmaps_coords


def _get_coord_pair_from_google_maps(location_name: str, gmaps_client) -> pd.Series:
    """Get the latitude and longitude of a location using the Google Maps API."""
    try:
        result = gmaps_client.geocode(location_name)
        if result:
            lat = result[0]["geometry"]["location"]["lat"]
            lon = result[0]["geometry"]["location"]["lng"]
            return pd.Series([lat, lon])
        else:
            logger.warning(f"No result for {location_name}")
            return pd.Series([None, None])
    except Exception as e:
        logger.error(f"Error for {location_name}: {e}")
        return pd.Series([None, None])


# --- Main Location Processing Functions ---


def assign_coordinates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assign coordinates to locations in the DataFrame. This function will:
    1. Check if coordinates are already present in the 'coords' column.
    2. If not, it will check the 'cleaned_coords' column.
    3. If 'cleaned_coords' is NaN, it will use the Google Maps API to get coordinates.
    4. If 'cleaned_coords' is not NaN, it will convert them to decimal degrees.
    """
    ### get locations for which there are no coordinates (from location, where cleaned_coords is NaN)
    locs = df.loc[df["cleaned_coords"].isna(), "location"].unique()
    # temp_df = df.copy()

    gmaps_coords = get_google_maps_coordinates(locs)
    df["loc"] = df.apply(lambda row: _resolve_coordinates(row, gmaps_coords), axis=1)
    df["latitude"] = df["loc"].apply(lambda x: x[0] if isinstance(x, tuple) else None)
    df["longitude"] = df["loc"].apply(lambda x: x[1] if isinstance(x, tuple) else None)
    return df


def _resolve_coordinates(row, gmaps_coords):
    """Resolve coordinates from cleaned_coords, coords, or Google Maps."""
    if pd.notna(row.get("cleaned_coords")):
        return standardize_coordinates(row["cleaned_coords"])
    elif pd.notna(row.get("coords")):
        return standardize_coordinates(row["coords"])
    elif row.get("location") in gmaps_coords:
        return gmaps_coords[row["location"]]
    return None


def assign_ecoregions(df: pd.DataFrame) -> pd.DataFrame:
    """Assign ecoregions to locations based intersections of points with ecoregion polygons."""
    # copy df
    df = df.copy()
    ecoregions_gdf = gpd.read_file(
        config.climatology_data_dir / "MEOW/meow_ecos.shp"
    ).to_crs(epsg=4326)
    df["geometry"] = gpd.points_from_xy(df["longitude"], df["latitude"])
    df = gpd.GeoDataFrame(df, geometry="geometry", crs=4326)
    df = gpd.sjoin(
        df,
        ecoregions_gdf[["geometry", "ECOREGION", "REALM", "Lat_Zone"]],
        how="left",
        predicate="intersects",
    )
    df = df.drop(columns=["geometry"])
    df.columns = df.columns.str.lower()
    return df


def save_locations_information(df):
    """Save locations information to YAML and CSV files if not already present."""
    yaml_path = config.resources_dir / "locations.yaml"
    csv_path = config.resources_dir / "locations.csv"
    if not (yaml_path.exists() and csv_path.exists()):
        locs_df = df.drop_duplicates(["doi"], keep="first")[
            ["doi", "location", "latitude", "longitude"]
        ].set_index("doi")
        file_ops.write_yaml(locs_df.to_dict(orient="index"), yaml_path)
        logger.info(f"Saved locations to {yaml_path}")
        locs_df.to_csv(csv_path, index=True, index_label="doi")
        logger.info(f"Saved locations to {csv_path}")


# --- Coordinate Parsing Utilities ---


def dms_to_decimal(
    degrees: float,
    minutes: float | int = 0,
    seconds: float | int = 0,
    direction: str = "",
) -> float:
    """Convert degrees, minutes, and seconds to decimal degrees."""
    try:
        degrees, minutes, seconds = float(degrees), float(minutes), float(seconds)
        if not (0 <= minutes < 60) or not (0 <= seconds < 60):
            raise ValueError("Invalid minutes or seconds range.")
        decimal = degrees + minutes / 60 + seconds / 3600
        if direction in ["S", "W"]:
            decimal = abs(decimal) * -1
        return decimal
    except Exception as e:
        logger.error(
            f"Error converting DMS: {degrees}° {minutes}′ {seconds}″ {direction} → {e}"
        )
        return None


def standardize_coordinates(coord_string: str) -> tuple[float, float] | None:
    """Convert various coordinate formats into decimal degrees."""
    if not isinstance(coord_string, str):
        return coord_string
    if coord_string.strip() == "":
        return None
    if "°" not in coord_string and not any(
        d in coord_string for d in ["N", "S", "E", "W"]
    ):
        try:
            return tuple(
                float(coord.replace("−", "-")) for coord in coord_string.split(",")
            )
        except Exception as e:
            logger.error(f"Failed to parse decimal degrees: {coord_string} → {e}")
            return None
    coord_string = _normalize_coord_string(coord_string)
    parts = re.split(r"\s*/\s*", coord_string)
    decimal_coords = []
    lat_lng_pattern = re.compile(
        r"""([ NSEW])?\s*(\d+(?:\.\d+)?)\s*[° ]?\s*(?:(\d+(?:\.\d+)?)\s*[′\']\s*)?(?:(\d+(?:\.\d+)?)\s*[\"]\s*)?([ NSEW])?""",
        re.VERBOSE,
    )
    for part in parts:
        lat_lng = lat_lng_pattern.findall(part)
        if len(lat_lng) < 2:
            logger.warning(f"Invalid coordinate pair: {part}")
            continue
        lat, lng = None, None
        for coord in lat_lng:
            dir1, deg, mins, secs, dir2 = coord
            direction = (dir1 + dir2).strip()
            decimal = dms_to_decimal(deg, mins or 0, secs or 0, direction)
            if "N" in coord or "S" in coord:
                lat = decimal
            elif "E" in coord or "W" in coord:
                lng = decimal
        if lat is None or lng is None:
            logger.warning(f"Failed to parse: {part}")
            continue
        decimal_coords.append((lat, lng))
    if not decimal_coords:
        logger.error(f"Failed to parse: {coord_string}")
        return None
    if len(decimal_coords) == 1:
        return decimal_coords[0]
    avg_lat = np.mean([c[0] for c in decimal_coords if c[0] is not None])
    avg_lng = np.mean([c[1] for c in decimal_coords if c[1] is not None])
    return (avg_lat, avg_lng)


def _normalize_coord_string(coord_string):
    """Standardize quotes and symbols in coordinate strings."""
    coord_string = (
        coord_string.replace("`", "'")
        .replace("′", "'")
        .replace("’", "'")
        .replace("ʹ", "'")
        .replace("˙", "'")
        .replace("″", '"')
        .replace("''", '"')
        .replace("”", '"')
    )
    return coord_string


def uniquify_multilocation_study_dois(df: pd.DataFrame) -> pd.DataFrame:
    """Uniquify DOIs for studies with multiple locations. N.B. requires 'location', 'coords', and 'cleaned_coords' columns."""
    temp_df = df.copy()
    temp_df["original_doi"] = temp_df["doi"]
    temp_df["location_lower"] = temp_df["location"].str.lower()
    locs_df = temp_df.drop_duplicates(
        ["doi", "location_lower", "coords", "cleaned_coords"]
    )
    locs_df.loc[:, "doi"] = utils.uniquify_repeated_values(locs_df.doi)
    temp_df = temp_df.merge(
        locs_df["doi"],
        how="left",
        left_index=True,
        right_index=True,
        suffixes=("_old", ""),
    )
    temp_df.drop(columns=["doi_old"], inplace=True)
    temp_df["doi"] = temp_df.groupby("original_doi")["doi"].ffill()
    return temp_df
