import yaml
import pandas as pd


# function to read in yamls
def read_yaml(yaml_fp) -> dict:
    with open(yaml_fp, "r") as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            return None


def write_yaml(data: dict, fp="unnamed.yaml") -> None:
    with open(fp, "w") as file:
        yaml.dump(data, file)


def append_to_yaml(data: dict, fp="unnamed.yaml") -> None:
    with open(fp, "a") as file:
        yaml.dump(data, file)


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
