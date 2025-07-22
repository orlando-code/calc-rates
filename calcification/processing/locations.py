# general
import re

import geopandas as gpd

# spatial
import googlemaps
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from calcification.utils import config, file_ops, utils


### spatial
def get_google_maps_coordinates(locations: list[str]) -> dict:
    """
    Get coordinates for a list of locations using Google Maps API. Checks if coordinates are already present in a yaml file.
    If not, it will append a new yaml file with the coordinates.
    """
    if (config.resources_dir / "gmaps_locations.yaml").exists():
        # read in coordinates from current yaml (values in dictionary of yaml)
        gmaps_coords = file_ops.read_yaml(config.resources_dir / "gmaps_locations.yaml")
        coord_keys = gmaps_coords.keys()
        # check if all locations are in the yaml file
        # Use set difference to find locations not in yaml coords
        extra_locations = set(locations) - set(coord_keys)
        if extra_locations:
            print(
                f"Found extra locations (not already in {config.resources_dir / 'gmaps_locations.yaml'}"
            )
        else:
            print(f"Using locations in {config.resources_dir / 'gmaps_locations.yaml'}")
            return gmaps_coords

    # if necessary, generate or append to a locations yaml file
    if not (config.resources_dir / "gmaps_locations.yaml").exists():
        print(f"Creating gmaps_locations.yaml file in {config.resources_dir}")
        # create new yaml file
        file_ops.write_yaml({}, config.resources_dir / "gmaps_locations.yaml")

    # get coordinates for these locations using Google Maps API
    gmaps_coords = {}
    GMAPS_API_KEY = file_ops.read_yaml(config.resources_dir / "api_keys.yaml")[
        "google_maps_api"
    ]
    gmaps_client = googlemaps.Client(key=GMAPS_API_KEY)

    for loc in tqdm(
        locations, desc="Querying Google Maps to retrieve coordinates of locations"
    ):
        gmaps_coords[loc] = tuple(
            get_coord_pair_from_google_maps(loc, gmaps_client).values
        )  # slightly hacky formatting since originally written for processing dataframe column
    # save locations to yaml file
    file_ops.append_to_yaml(gmaps_coords, config.resources_dir / "gmaps_locations.yaml")
    return gmaps_coords


def get_coord_pair_from_google_maps(location_name: str, gmaps_client) -> pd.Series:
    """
    Get the latitude and longitude of a location using the Google Maps API. Used row-wise on a DataFrame.

    Args:
        location_name (str): Location name
        gmaps_client (googlemaps.client.Client): Google Maps API client. Requires an active, authorised Google Maps client.
    """
    try:
        result = gmaps_client.geocode(location_name)
        if result:
            lat = result[0]["geometry"]["location"]["lat"]
            lon = result[0]["geometry"]["location"]["lng"]
            return pd.Series([lat, lon])
        else:
            return pd.Series([None, None])
    except Exception as e:
        print(f"Error for {location_name}: {e}")
        return pd.Series([None, None])


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
    temp_df = df.copy()

    gmaps_coords = get_google_maps_coordinates(locs)
    # first convert cleaned_coords to decimal degrees
    temp_df["loc"] = temp_df["cleaned_coords"].apply(
        lambda x: standardize_coordinates(x) if pd.notna(x) else None
    )
    # where no cleaned_coords value but coords value exists (is clean already), fill with decimal degrees
    mask = temp_df["loc"].isna() & temp_df["coords"].notna()
    temp_df.loc[mask, "loc"] = temp_df.loc[mask, "coords"].apply(
        standardize_coordinates
    )
    # for locations where coordinates from cleaned_coords are available, fill them in
    # for remaining locations, use the coordinates from Google Maps API if available
    temp_df["loc"] = temp_df["loc"].fillna(
        temp_df.apply(
            lambda row: tuple(gmaps_coords.get(row["location"]))
            if row["location"] in gmaps_coords
            else None,
            axis=1,
        )
    )

    # extract latitude and longitude from the coordinate tuples, with proper type checking
    temp_df["latitude"] = temp_df["loc"].apply(
        lambda x: x[0] if isinstance(x, tuple) else None
    )
    temp_df["longitude"] = temp_df["loc"].apply(
        lambda x: x[1] if isinstance(x, tuple) else None
    )

    return temp_df


def assign_ecoregions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assign ecoregions to locations based intersections of points with ecoregion polygons.
    """
    # copy df
    df = df.copy()
    # load ecoregion data
    ecoregions_gdf = gpd.read_file(config.climatology_data_dir / "MEOW/meow_ecos.shp")
    ecoregions_gdf = ecoregions_gdf.to_crs(epsg=4326)
    # generate geometry columnn and convert to geodataframe for matching
    df.loc[:, "geometry"] = gpd.points_from_xy(df["longitude"], df["latitude"])
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


def save_locations_information(df: pd.DataFrame) -> None:
    """
    Save locations information to YAML and CSV files.
    """
    if not (
        (config.resources_dir / "locations.yaml").exists()
        and (config.resources_dir / "locations.csv").exists()
    ):
        # send to dictionary
        locs_df = df.drop_duplicates(["doi"], keep="first")[
            ["doi", "location", "latitude", "longitude"]
        ].set_index("doi")
        # save dictionary to yaml
        file_ops.write_yaml(
            locs_df.to_dict(orient="index"), config.resources_dir / "locations.yaml"
        )
        print(f"Saved locations to {config.resources_dir / 'locations.yaml'}")
        locs_df.to_csv(
            config.resources_dir / "locations.csv", index=True, index_label="doi"
        )
        print(f"Saved locations to {config.resources_dir / 'locations.csv'}")


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
            decimal = abs(decimal)  # for the rare cases that stated as e.g. -3ºW
            decimal *= -1
        return decimal
    except Exception as e:
        print(
            f"Error converting DMS: {degrees}° {minutes}′ {seconds}″ {direction} → {e}"
        )
        return None


def standardize_coordinates(coord_string: str) -> tuple[float, float] | None:
    """Convert various coordinate formats into decimal degrees."""
    # check if not string (already decimal degrees)
    if not isinstance(coord_string, str):
        return coord_string
    if coord_string == " ":
        return None
    # already decimal degrees
    if "°" not in coord_string and not any(
        direction in coord_string for direction in ["N", "S", "E", "W"]
    ):
        return tuple(
            [float(coord.replace("−", "-")) for coord in coord_string.split(",")]
        )

    # Standardize quotes for minutes and seconds
    coord_string = (
        coord_string.replace("`", "'")
        .replace("′", "'")
        .replace("’", "'")
        .replace("ʹ", "'")
    )
    coord_string = coord_string.replace("″", '"').replace("''", '"').replace("”", '"')
    parts = re.split(r"\s*/\s*", coord_string)  # Split at '/' if present
    decimal_coords = []

    lat_lng_pattern = re.compile(
        r"""
        ([ NSEW])?\s*  # optional leading direction
        (\d+(?:\.\d+)?)\s*[° ]?\s*  # degrees (mandatory)
        (?:(\d+(?:\.\d+)?)\s*[′'’]\s*)?  # optional Minutes
        (?:(\d+(?:\.\d+)?)\s*[″"]\s*)?  # optional Seconds
        ([ NSEW])?  # optional trailing direction
    """,
        re.VERBOSE,
    )

    for part in parts:
        lat_lng = lat_lng_pattern.findall(part)
        # drop any empty strings from the list

        if len(lat_lng) < 2:
            print(f"Invalid coordinate pair: {part}")
            continue

        lat, lng = None, None  # Initialize variables
        for coord in lat_lng:
            dir1, deg, mins, secs, dir2 = coord
            if dir1 is None and dir2 is None:
                print(f"Could not determine direction in: {part}")
                continue
            decimal = dms_to_decimal(
                deg, mins or 0, secs or 0, (dir1 + dir2).strip(" ")
            )  # TODO: slightly hacky
            if "N" in coord or "S" in coord:
                lat = decimal
            elif "E" in coord or "W" in coord:
                lng = decimal
        if not lat and not lng:
            print("Failed to parse:", part)
            return None
        decimal_coords.append((lat, lng))

    if len(decimal_coords) == 0:
        print(f"Failed to parse: {coord_string}")
        return None
    if len(decimal_coords) == 1:
        return decimal_coords[0]
    else:
        avg_lat = np.mean([c[0] for c in decimal_coords if c[0] is not None])
        avg_lng = np.mean([c[1] for c in decimal_coords if c[1] is not None])
        return (avg_lat, avg_lng)


def uniquify_multilocation_study_dois(df: pd.DataFrame) -> pd.DataFrame:
    """Uniquify DOIs for studies with multiple locations. N.B. requires 'location', 'coords', and 'cleaned_coords' columns."""
    temp_df = df.copy()
    temp_df["original_doi"] = temp_df["doi"]
    temp_df["location_lower"] = temp_df[
        "location"
    ].str.lower()  # lower to make not case sensitive
    locs_df = temp_df.drop_duplicates(
        ["doi", "location_lower", "coords", "cleaned_coords"]
    )  # drop duplicates to get truly unique

    locs_df.loc[:, "doi"] = utils.uniquify_repeated_values(locs_df.doi)

    temp_df = temp_df.merge(
        locs_df["doi"],
        how="left",
        left_index=True,
        right_index=True,
        suffixes=("_old", ""),
    )
    # drop original doi column (now redundant)
    temp_df.drop(columns=["doi_old"], inplace=True)
    # group by original doi to fill down the new, uniquified doi
    temp_df["doi"] = temp_df.groupby("original_doi")["doi"].ffill()
    return temp_df
