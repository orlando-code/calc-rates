# files
import yaml
from openpyxl import load_workbook
import requests

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
from tqdm.auto import tqdm
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
    df = df.map(lambda x: unicodedata.normalize("NFKD", str(x)).replace('\xa0', ' ') if isinstance(x, str) else x)  # clean non-breaking spaces from string cells
    # general processing
    df.rename(columns=read_yaml(config.resources_dir / "mapping.yaml")['sheet_column_map'], inplace=True)    # rename columns to agree with cbsyst output    
    df.columns = df.columns.str.lower() # columns lower case headers for less confusing access later on
    df.columns = df.columns.str.replace(' ', '_')   # process columns to replace whitespace with underscore
    df.columns = df.columns.str.replace('[()]', '', regex=True) # remove '(' and ')' from column names
    df['year'] = pd.to_datetime(df['year'], format='%Y')    # datetime format for later plotting
    
    if selection_dict:  # filter for selected values
        for key, value in selection_dict.items():
            df = df[df[key] == value]
    # TODO: perhaps this would be a good step to make all same case
    # make all string values in the dataframe (excluding 'species_types') lowercase
    # for col in df.select_dtypes(include=['object']).columns:
    #     if col not in ['species_types', 'doi', 'coords', 'cleaned_coords', 'location']:
    #         df[col] = df[col].str.lower()    
    
            
    df = df.assign(
        genus=df.species_types.apply(lambda x: binomial_to_genus_species(x)[0]),
        species=df.species_types.apply(lambda x: binomial_to_genus_species(x)[1])
    )   # separate binomials into genus and species columns
    # append family column: if no species_mapping file, generate
    if not (config.resources_dir / 'species_mapping.yaml').exists():
        create_species_mapping_yaml(df.species_types.unique())
    else:
        print(f'Using species mapping in {config.resources_dir / 'species_mapping.yaml'}.')
    species_mapping = read_yaml(config.resources_dir / 'species_mapping.yaml')
    # Extract nested dictionary values for each species
    df['family'] = df.species_types.apply(lambda x: species_mapping.get(x, {}).get('family', 'Unknown'))
    df['functional_group'] = df.species_types.apply(lambda x: species_mapping.get(x, {}).get('functional_group', 'Unknown'))
    
    # flag up duplicate dois (only if they have also have 'include' as 'yes')
    inclusion_df = df[df['include'] == 'yes']
    duplicate_dois = inclusion_df[inclusion_df.duplicated(subset='doi', keep=False)]
    if not duplicate_dois.empty and not all(pd.isna(duplicate_dois['doi'])):
        print("\nDuplicate DOIs found, treat with caution:")
        print([doi for doi in duplicate_dois.doi.unique() if doi is not np.nan])        
        
    # fill down necessary repeated metadata values
    df[['doi', 'year', 'authors', 'location', 'species_types', 'taxa']] = df[['doi', 'year', 'authors', 'location', 'species_types', 'taxa']].infer_objects(copy=False).ffill()
    df[['coords', 'cleaned_coords']] = df.groupby('doi')[['coords', 'cleaned_coords']].ffill()  # fill only as far as the next DOI

    # missing sample size values
    if df['n'].dtype == 'object':  # Only perform string operations if column contains strings
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


def get_species_info_from_worms(species_binomial: str) -> dict:
    """Query WoRMS API to get the family name of a coral species.
    
    Args:
        species_binomial (str): Scientific name in 'Genus species' format
    
    Returns:
        dict: Dictionary with family name and additional taxonomic information
    """
    
    # clean species name
    if any(x in species_binomial.replace('.', '').split() for x in ['sp', 'spp', 'cf']): # remove general indication from species, leaving just genus
        species_binomial = species_binomial.split()[0]  # take the first word as the genus
    if species_binomial in ['Massive porites', 'Porites lutea/lobata']:
        species_binomial = 'Porites'
    if 'CCA' in species_binomial:
        return {'species': species_binomial, 'family': 'Corallinaceae-Sporolithaceae', 'status': 'Best guess', 'functional_group': 'Crustose coralline algae'}
        
    # URL encode the species name to handle spaces properly
    encoded_species = requests.utils.quote(species_binomial)
    base_url = f"https://www.marinespecies.org/rest/AphiaRecordsByName/{encoded_species}?like=false&marine_only=true"
    
    try:
        response = requests.get(base_url)
        response.raise_for_status()  # raise exception for 4XX/5XX status codes
        data = response.json()
        
        if data and isinstance(data, list) and len(data) > 0:   # assign response to dictionary if data found
            if len(data) > 1:   # if more than one record, take the most recent one which is 'accepted' in the 'status' field
                data = [record for record in data if record.get('status', '') == 'accepted'][0]
                if not data:    # if technically none accepted, take the first record (haven't seen this happen yet)
                    data = data[0]
            else:
                data = data[0]  # list containing single record dictionary: take the dictionary

            result = {
                'species': species_binomial,
                'family': data.get('family', 'Not Found'),
                'status': 'Found',
                'rank': data.get('rank', 'Not Found'),
                'aphia_id': data.get('AphiaID', 'Not Found'),
                'accepted_name': data.get('valid_name', species_binomial),
                'kingdom': data.get('kingdom', 'Not Found'),
                'phylum': data.get('phylum', 'Not Found'),
                'class': data.get('class', 'Not Found'),
                'order': data.get('order', 'Not Found')
            }
            result['functional_group'] = assign_functional_group(result)

            return result
        else:
            return {'species': species_binomial, 'family': 'Not Found', 'status': 'No Data', 'functional_group': 'Unknown'}
    
    except requests.exceptions.RequestException as e:
        print(f"API Request Error for {species_binomial}: {e}")
        return {'species': species_binomial, 'family': 'Error', 'status': str(e)}


def create_species_mapping_yaml(species_list) -> None:
    """Create a YAML file with species-family mapping for a list of species.
    
    Args:
        species_list (list): List of coral species names in 'Genus species' format
    """
    import yaml
    species_mapping = {}
    for species in tqdm(species_list, desc='Querying WoRMS API to retrieve organism taxonomic data'):
        species_info = get_species_info_from_worms(species)
        species_mapping[species] = {
            'family': species_info['family'],
            'functional_group': species_info['functional_group']}
    # save family: genus, species mapping to YAML file
    with open(config.resources_dir / 'species_mapping.yaml', 'w') as file:
        yaml.dump(species_mapping, file)
    print(f'Species mapping saved to {config.resources_dir / "species_mapping.yaml"}')
    
    
def assign_functional_group(taxon_info):
    """
    Assign a functional group based on taxonomic information.
    N.B. this mapping is not exhaustive: only checked with ~130 species.
    There is also likely subjectivity in assignment.

    Args:
        taxon_info (dict): Dictionary containing taxonomic information
    
    Returns:
        str: Functional group name
    """
    family = taxon_info.get('family', '').lower()
    order = taxon_info.get('order', '').lower()
    class_name = taxon_info.get('class', '').lower()
    phylum = taxon_info.get('phylum', '').lower()
    genus = taxon_info.get('species', '').split()[0].lower()  # Extract genus from species name
    binomial = taxon_info.get('species', '').lower()

    if genus in ['jania', 'amphiroa']:
        return 'Articulate coralline algae'
    
    # Crustose coralline algae
    if family in ['corallinaceae', 'sporolithaceae', 'hapalidiaceae', 'hydrolithaceae', 'lithophyllaceae', 'mesophyllumaceae', 'spongitidaceae', 'porolithaceae']:
        return 'Crustose coralline algae'
    
    # Fleshy algae
    if (phylum in ['chlorophyta', 'ochrophyta', 'rhodophyta'] or 
        order in ['dictyotales', 'ectocarpales', 'fucales', 'gigartinales'] or
        class_name in ['phaeophyceae', 'ulvophyceae', 'florideophyceae']):
        # Special check for calcareous algae that aren't CCA
        if genus in ['galaxaura', 'padina']:
            return 'Calcareous algae'
        if genus in ['peyssonnelia']:
            return 'Calcareous red algae'
        if family in ['halimedaceae']:
            return 'Calcareous green algae'
        return 'Fleshy algae'
    
    # Hard corals (scleractinian)
    if order == 'scleractinia' or family in ['pocilloporidae', 'acroporidae', 'poritidae', 'faviidae', 'fungiidae', 'agariciidae']:
        return 'Hard coral'
    
    # Soft corals
    if order in ['alcyonacea', 'gorgonacea'] or 'alcyoniidae' in family:
        return 'Soft coral'
    
    # Sponges
    if phylum == 'porifera':
        return 'Sponge'
    
    # Foraminifera
    if phylum in ['foraminifera', 'retaria'] or class_name == 'foraminifera':
        return 'Foraminifera'
    
    # Turf algae - often identified by growth form rather than taxonomy
    if 'turf' in taxon_info.get('species', ''):
        return 'Turf algae'
    
    # Bryozoans
    if phylum == 'bryozoa':
        return 'Bryozoan'
    
    # catch-all case
    return 'Other benthic organism'



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


### carbonate chemistry
def populate_carbonate_chemistry(fp: str, sheet_name: str="all_data", selection_dict: dict={'include': 'yes'}) -> pd.DataFrame:
    # df = pd.read_excel(fp, sheet_name=sheet_name)
    df = process_df(pd.read_excel(fp, sheet_name=sheet_name), require_results=False, selection_dict=selection_dict)
    # return df
    ### load measured values
    print("Loading measured values...")
    measured_df = get_highlighted(fp, sheet_name=sheet_name)    # keeping all cols
    # return measured_df
    measured_df = process_df(measured_df, require_results=False, selection_dict=selection_dict)
    
    ### convert nbs values to total scale using cbsyst     # TODO: implement uncertainty propagation
    print("Calculating total pH values...")
    measured_df.loc[:, 'phtot'] = measured_df.apply(
        lambda row: cbh.pH_scale_converter(
            pH=row['phnbs'], scale='NBS', Temp=row['temp'], Sal=row['sal'] if pd.notna(row['sal']) else 35
        ).get('pHtot', None) if pd.notna(row['phnbs']) and pd.notna(row['temp'])
        else row['phtot'],
        axis=1
    )
    # if phtot and phnbs are both nan, calculate phtot from temp, salinity, dic, and ta
    measured_df.loc[:, 'phtot'] = measured_df.apply(
        lambda row: cb.cbsyst.Csys(
            TA=row['ta'], DIC=row['dic'], T_in=row['temp'], S_in=row['sal'] if pd.notna(row['sal']) else 35,
        ).get('pHtot', None) if pd.isna(row['phnbs']) and pd.isna(row['phtot']) and pd.notna(row['temp']) and pd.notna(row['dic']) and pd.notna(row['ta'])
        else np.nan,
        axis=1
    )
    
    ### calculate carbonate chemistry
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
            T_in=row['temp'],
            S_in=row['sal'],
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