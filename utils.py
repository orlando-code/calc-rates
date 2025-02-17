import yaml
import pandas as pd

# universal constants
MOLAR_MASS_CACO3 = 100.0869    # g/mol
PREFIXES = {'m': 1e-3, 'μ': 1e-6, 'n': 1e-9}
DURATIONS = {'hr': 24, 'd': 1, 'wk': 1/7}


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


def process_df(df: pd.DataFrame, **selection_kws: dict) -> pd.DataFrame:
    # Default selection values
    default_selection = {'Extractor': 'Orlando', 'Include': 'yes'}
    
    # Merge defaults with user-provided values (user values take priority)
    selection_kws = {**default_selection, **selection_kws}
    
    # Apply selection
    for key, value in selection_kws.items():
        df = df[df[key] == value]
    
    
    # general processing
    df.columns = df.columns.str.lower() # columns lower case headers
    df.columns = df.columns.str.replace(' ', '_')   # process columns to replace whitespace with underscore
    df.columns = df.columns.str.replace('[()]', '', regex=True) # remove '(' and ')' from column names
    df['year'] = pd.to_datetime(df['year'], format='%Y')    # datetime format for later plotting
    df[['doi', 'year', 'authors']] = df[['doi', 'year', 'authors']].ffill()    # where I haven't these values for every row
    
    # missing values
    df = df[~df['n'].str.contains('~', na=False)]   # remove any rows in which 'n' has '~' in the string
    df = df[df.n != 'M']    # remove any rows in which 'n' is 'M'
    df = df.dropna(subset=['n', 'calcification', 'calcification_units'])    # keep only rows with all the necessary data

    return df


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


def rate_conversion(rate_val: float, rate_unit: str) -> float:
    """Conversion into gCaCO3 ... day-1 for absolute rates"""
    
    # convert moles to mass
    if 'mol' in rate_unit:
        rate_val *= MOLAR_MASS_CACO3
        mol_prefix = rate_unit.split('mol')[0]
        if mol_prefix in PREFIXES:
            rate_val *= PREFIXES[mol_prefix]

    # convert time unit
    for time_unit, factor in DURATIONS.items():
        if time_unit in rate_unit:
            rate_val *= factor
            break
    
    # area conversion
    if 'cm-2' in rate_unit.split(' ')[1]:
        rate_val *= 1e4   # cm2 to m2
    # mass conversion
    mass_prefix = rate_unit.split(' ')[1][0]
    if mass_prefix in PREFIXES:
        rate_val *= PREFIXES[mass_prefix]   # convert to g
    
    # TODO: add non-CaCO3 conversions

    
    return rate_val


def unit_name_conversion(rate_unit: str) -> str:
    if "cm-2" in rate_unit:
        return "gCaCO3 m-2d-1"  # specific area unit
    elif "g-1" in rate_unit or "mg-1" in rate_unit:
        return "gCaCO3 g-1d-1"  # specific mass unit
    else:
        return "Unknown"
    
    
### sensitivity analysis
def select_by_stat(ds, variables_stats: dict):
    """
    Selects values from an xarray dataset based on the specified statistics for given variables.
    
    Parameters:
        ds (xr.Dataset): The input dataset with a 'param_combination' dimension and coordinates for the variables.
        variables_stats (dict): A dictionary where keys are variable names and values are the statistics to use for selection ('min', 'max', 'mean').
    
    Returns:
        xr.Dataset: Dataset subset at the selected coordinate values.
    """
    selected_coords = {}
    
    for var, stat in variables_stats.items():
        if stat == "min":
            selected_coords[var] = ds[var].min().item()
        elif stat == "max":
            selected_coords[var] = ds[var].max().item()
        elif stat == "mean":
            selected_coords[var] = ds[var].mean().item()
        else:
            raise ValueError(f"stat for {var} must be 'min', 'max', or 'mean'")
    
    # Select the closest matching values
    ds_selected = ds.sel(selected_coords, method="nearest")
    
    return ds_selected
