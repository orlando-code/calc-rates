import pandas as pd
import numpy as np
import re

### spatial
def get_coordinates_from_gmaps(location_name: str, gmaps_client) -> pd.Series:
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


def dms_to_decimal(degrees, minutes=0, seconds=0, direction=''):
    """Convert degrees, minutes, and seconds to decimal degrees."""
    try:
        degrees, minutes, seconds = float(degrees), float(minutes), float(seconds)
        if not (0 <= minutes < 60) or not (0 <= seconds < 60):
            raise ValueError("Invalid minutes or seconds range.")
        decimal = degrees + minutes / 60 + seconds / 3600
        if direction in ['S', 'W']:
            decimal = abs(decimal)  # for the rare cases that stated as e.g. -3ºW
            decimal *= -1
        return decimal
    except Exception as e:
        print(f"Error converting DMS: {degrees}° {minutes}′ {seconds}″ {direction} → {e}")
        return None
        

def standardize_coordinates(coord_string):
    """Convert various coordinate formats into decimal degrees."""
    # check if not string (already decimal degrees)
    if not isinstance(coord_string, str):
        return coord_string
    if coord_string == ' ':
        return None
    # already decimal degrees
    if '°' not in coord_string and not any(direction in coord_string for direction in ['N', 'S', 'E', 'W']):
        return tuple([float(coord) for coord in coord_string.split(',')])

    
    coord_string = coord_string.replace("′′", '″')
    coord_string = coord_string.replace("’’", '"')
    parts = re.split(r'\s*/\s*', coord_string)  # Split at '/' if present
    decimal_coords = []

    lat_lng_pattern = re.compile(r'''
        ([ NSEW])?\s*  # optional leading direction
        (\d+(?:\.\d+)?)\s*[° ]\s*  # degrees (mandatory)
        (?:(\d+(?:\.\d+)?)\s*[′'’]\s*)?  # optional Minutes
        (?:(\d+(?:\.\d+)?)\s*[″"]\s*)?  # optional Seconds
        ([ NSEW])?  # optional trailing direction
    ''', re.VERBOSE)
    for part in parts:
        lat_lng = lat_lng_pattern.findall(part)
        # drop any empty strings from the list
        
        if len(lat_lng) != 2:
            print(f"Invalid coordinate pair: {part}")
            continue
        
        lat, lng = None, None  # Initialize variables
        for coord in lat_lng:
            dir1, deg, mins, secs, dir2 = coord
            if dir1 is None and dir2 is None:
                print(f"Could not determine direction in: {part}")
                continue
            decimal = dms_to_decimal(deg, mins or 0, secs or 0, (dir1 + dir2).strip(' '))   # TODO: slightly hacky
            if "N" in coord or "S" in coord:
                lat = decimal
            elif "E" in coord or "W" in coord:
                lng = decimal
        if not lat and not lng:
            print('Failed to parse:', part)
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