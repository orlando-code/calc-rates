# files
import yaml
from openpyxl import load_workbook

# general
import numpy as np
import pandas as pd
import unicodedata
import re
import itertools
import string

# custom
from calcification import utils, config
import cbsyst as cb
from tqdm import tqdm
import cbsyst.helpers as cbh

### global constants
MOLAR_MASS_CACO3 = 100.0869    # g/mol
PREFIXES = {'m': 1e-3, 'μ': 1e-6, 'n': 1e-9}
DURATIONS = {'hr': 24, 'd': 1, 'wk': 1/7, 's': 86400}


### file handling
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

### processing files
def process_df(df: pd.DataFrame, require_results: bool=False, selection_dict: dict={'include': 'yes'}) -> pd.DataFrame:
    df.columns = df.columns.str.normalize("NFKC").str.replace("μ", "u") # replace any unicode versions of 'μ' with 'u'

    # general processing
    df.rename(columns=read_yaml(config.resources_dir / "mapping.yaml")['sheet_column_map'], inplace=True)    # rename columns to agree with cbsyst output    
    df.columns = df.columns.str.lower() # columns lower case headers for less confusing access later on
    df.columns = df.columns.str.replace(' ', '_')   # process columns to replace whitespace with underscore
    df.columns = df.columns.str.replace('[()]', '', regex=True) # remove '(' and ')' from column names
    df['year'] = pd.to_datetime(df['year'], format='%Y')    # datetime format for later plotting
    # fill down necessary repeated metadata values
    df[['doi', 'year', 'authors', 'location', 'species_types']] = df[['doi', 'year', 'authors', 'location', 'species_types']].infer_objects(copy=False).ffill()
    
    df['genus'] = df.species_types.apply(lambda x: binomial_to_genus_species(x)[0])
    df['species'] = df.species_types.apply(lambda x: binomial_to_genus_species(x)[1])    # apply selection
    if selection_dict:
        for key, value in selection_dict.items():
            df = df[df[key] == value]
        
    # missing sample size values
    df = df[~df['n'].str.contains('~', na=False)]   # remove any rows in which 'n' has '~' in the string
    df = df[df.n != 'M']    # remove any rows in which 'n' is 'M'
    
    df.loc[:, df.columns != 'year'] = df.loc[:, df.columns != 'year'].apply(safe_to_numeric)
    
    problem_cols = ['irr', 'ipar']  # some columns have rogue strings when they should all contain numbers: in this case, convert unconvertable values to NaN
    for col in problem_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df['irr'] = df.apply(lambda row: irradiance_conversion(row['ipar'], 'PAR') if pd.notna(row['ipar']) else row['irr'], axis=1)  # convert irradiance to consistent unit
    
    if require_results:
        df = df.dropna(subset=['n', 'calcification'])    # keep only rows with all the necessary data
    
    # calculate calcification standard deviation only when 'calcification_se' and 'n' are not NaN
    df['calcification_sd'] = df.apply(lambda row: calc_sd_from_se(row['calcification_se'], row['n']) if pd.notna(row['calcification_se']) and pd.notna(row['n']) else row['calcification_sd'], axis=1)
    
    return df


def aggregate_df(df, method: str='mean') -> pd.DataFrame:
    # Define aggregation functions
    aggregation_funcs = {col: method if pd.api.types.is_numeric_dtype(df[col]) else lambda x: x.iloc[0] for col in df.columns}

    # Aggregate DataFrame
    return df.agg(aggregation_funcs)


def calc_sd_from_se(se: float, n: int) -> float:
    """Calculate standard deviation from standard error and sample size
    
    Args:
        se (float): standard error
        n (int): number of samples
        
    Returns:
        float: standard deviation
    """
    return se * np.sqrt(n)


def safe_to_numeric(col):
    """Convert column to numeric if possible, otherwise return as is."""
    try:
        return pd.to_numeric(col)
    except (ValueError, TypeError):
        return col  # Return original column if conversion fails
    
    
def extract_year_from_str(s) -> pd.Timestamp:
    """Extract year from a string. Useful for cases where 'authors' has been provided with attached year.
    
    Args:
        s (str): String containing year.
        
    Returns:
        pd.Timestamp: Year extracted from string.
    """
    try:
        match = re.search(r'\d{4}', str(s))
        if match:
            return pd.to_datetime(match.group(), format='%Y')
        return pd.NaT
    except ValueError:
        return None


def irradiance_conversion(irr_val: float, irr_unit: str="PAR") -> float:
    # convert from mol quanta m-2 day-1 to μmol quanta m-2 s-1
    s_in_day = DURATIONS['s']
    return irr_val / (s_in_day * PREFIXES['μ']) if irr_unit == "PAR" else irr_val
    

def binomial_to_genus_species(binomial):
    """
    Convert a binomial name to genus and species.
    
    Args:
        binomial (str): Binomial name.
        
    Returns:
        tuple: Genus and species names.
    """
    # strip periods, 'cf' (used to compare with known species)
    binomial = binomial.replace('.', '')
    binomial = binomial.replace('cf', '')
    split = binomial.split(' ')
    # remove any empty strings (indicative of leading/trailing whitespace)
    split = [s for s in split if s]
    
    if 'spp' in binomial or 'sp' in split:
        genus = split[0]
        species = 'spp'
    else:
        genus = split[0]
        species = split[-1] if len(split) > 1 else 'spp'
    return genus, species


def get_highlighted_mask(fp: str, sheet_name: str, rgb_color: str = "FFFFC000") -> pd.DataFrame:
    """
    Creates a boolean mask DataFrame where True indicates a highlighted cell.

    Args:
        fp (str): Path to the Excel file.
        sheet_name (str): Name of the worksheet.
        rgb_color (str): RGB color to filter (e.g., 'FFFFC000' for calculated values).

    Returns:
        pd.DataFrame: Boolean mask DataFrame with True for highlighted cells.
    """
    wb = load_workbook(fp, data_only=True, read_only=True)
    ws = wb[sheet_name]

    df = pd.DataFrame([[cell.value for cell in row] for row in ws.iter_rows()])
    df = df.dropna(axis=1, how='all')   # remove any empty columns
    df.columns = df.iloc[0]  # set the first row as the column headers
    # df = df.drop(0)  # remove the header row
    mask = pd.DataFrame(False, index=df.index, columns=df.columns)  # mask with False by default, same shape as df
    # mask.drop(0, inplace=True)  # drop the first row (header)

    # set highlighted cells to True, all else False
    for row_idx, row in enumerate(ws.iter_rows()):
        for col_idx, cell in enumerate(row):
            if cell.fill and cell.fill.fgColor and cell.fill.fgColor.rgb == rgb_color:
                mask.iat[row_idx, col_idx] = True
                
    # Always mark 'include' and 'n' columns as True for future processing
    for col in ['Include', 'n', 'Species types']:
        if col in mask.columns:
            mask[col] = True

    return mask


def get_highlighted(fp: str, sheet_name: str, rgb_color: str = "FFFFC000") -> pd.DataFrame:
    """
    Loads an Excel sheet into a DataFrame and masks non-highlighted values with NaN,
    except for a specified subset of columns.

    Args:
        fp (str): Path to the Excel file.
        sheet_name (str): Name of the worksheet.
        rgb_color (str): RGB color to filter (e.g., 'FFFFC000' for calculated values).
        keep_cols (list): List of column names to keep unchanged.

    Returns:
        pd.DataFrame: DataFrame with non-highlighted values masked as NaN.
    """
    df = pd.read_excel(fp, sheet_name=sheet_name, engine="openpyxl", dtype=str)
    mask = get_highlighted_mask(fp, sheet_name, rgb_color)  # generate mask of highlighted cells
    return df.where(mask.drop(0).reset_index(drop=True), np.nan)     # remove first row of mask and reset index


def uniquify_repeated_values(vals: list) -> list:
    """
    Append a unique suffix to repeated values in a list.
    
    Parameters:
        vals (list): List of values.
    
    Returns:
        list: List of values with unique suffixes.
    """
    def zip_letters(l):
        """Zip a list of strings with uppercase letters."""
        al = string.ascii_uppercase
        return ['-LOC-'.join(i) for i in zip(l, al)] if len(l) > 1 else l
    return [j for _, i in itertools.groupby(vals) for j in zip_letters(list(i))]


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
            decimal *= -1
        return decimal
    except Exception as e:
        print(f"Error converting DMS: {degrees}° {minutes}′ {seconds}″ {direction} → {e}")
        return None
        

def standardize_coordinates(coord_string):
    """Convert various coordinate formats into decimal degrees."""
    coord_string = coord_string.replace("′′", '″')
    parts = re.split(r'\s*/\s*', coord_string)  # Split at '/' if present
    decimal_coords = []

    lat_lng_pattern = re.compile(r'''
        ([NSEW])?\s*  # optional leading direction
        (\d+(?:\.\d+)?)\s*[° ]\s*  # degrees (mandatory)
        (?:(\d+(?:\.\d+)?)\s*[′'’]\s*)?  # optional Minutes
        (?:(\d+(?:\.\d+)?)\s*[″"]\s*)?  # optional Seconds
        ([NSEW])?  # optional trailing direction
    ''', re.VERBOSE)
    for part in parts:
        lat_lng = lat_lng_pattern.findall(part)
        
        if len(lat_lng) != 2:
            print(f"Invalid coordinate pair: {part}")
            continue
        
        for coord in lat_lng:
            dir1, deg, mins, secs, dir2 = coord
            if dir1 is None and dir2 is None:
                print(f"Could not determine direction in: {part}")
                continue
            decimal = dms_to_decimal(deg, mins or 0, secs or 0, dir1 or dir2)
            if "N" in coord or "S" in coord:
                lat = decimal
            elif "E" in coord or "W" in coord:
                lng = decimal
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


### carbonate chemistry
def populate_carbonate_chemistry(fp: str, sheet_name: str="all_data") -> pd.DataFrame:
    # df = pd.read_excel(fp, sheet_name=sheet_name)
    df = process_df(pd.read_excel(fp, sheet_name=sheet_name), require_results=False, selection_dict={'include': 'yes'})
    # return df
    ### load measured values
    print("Loading measured values...")
    measured_df = get_highlighted(fp, sheet_name=sheet_name)    # keeping all cols
    # return measured_df
    measured_df = process_df(measured_df, require_results=False, selection_dict={'include': 'yes'})
    
    ### convert nbs values to total scale using cbsyst     # TODO: implement uncertainty propagation
    print("Converting pH values to total scale...")
    measured_df.loc[:, 'phtot'] = measured_df.apply(
        lambda row: cbh.pH_scale_converter(
            pH=row['phnbs'], scale='NBS', Temp=row['t_in'], Sal=row['s_in'] if pd.notna(row['s_in']) else 35
        ).get('pHtot', None) if pd.notna(row['phnbs']) and pd.notna(row['t_in'])
        else row['phtot'],
        axis=1
    )
    
    ### calculate carbonate chemistry
    print("Calculating carbonate chemistry parameters...")
    carb_metadata = read_yaml(config.resources_dir / "mapping.yaml")
    carb_chem_cols = carb_metadata['carbonate_chemistry_cols']
    out_values = carb_metadata['carbonate_chemistry_params']
    carb_df = measured_df[carb_chem_cols].copy()

    # apply function row-wise
    tqdm.pandas(desc="Calculating carbonate chemistry")
    carb_df.loc[:, out_values] = carb_df.progress_apply(lambda row: pd.Series(calculate_carb_chem(row, out_values)), axis=1)
    return df.combine_first(carb_df)
    # return measured_df


def calculate_carb_chem(row, out_values: list) -> dict:
    """Calculate carbonate chemistry parameters and return a dictionary."""
    try:
        out_dict = cb.Csys(
            pHtot=row['phtot'],
            TA=row['ta'],
            T_in=row['t_in'],
            S_in=row['s_in'],
        )
        out_dict = {key.lower(): value for key, value in out_dict.items()}  # lower the keys of the dictionary to ensure case-insensitivity

        return {
            key: (out_dict.get(key.lower(), None)[0] if isinstance(out_dict.get(key.lower()), (list, np.ndarray)) else out_dict.get(key.lower(), None))
            for key in out_values
            }    
    except Exception as e:
        print(f"Error: {e}")


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
    # TODO: expand for all types of rate
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


# Function to determine optimal number of clusters using silhouette score
def optimal_kmeans(data, max_clusters=8):
    best_k = 2  # Minimum sensible number of clusters
    best_score = -1
    scores = []

    for k in range(2, min(len(data), max_clusters + 1)):  # Avoid excessive clustering
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=max_clusters)
        labels = kmeans.fit_predict(data)
        score = silhouette_score(data, labels)
        scores.append((k, score))

        if score > best_score:
            best_score = score
            best_k = k

    return best_k, scores