# general
import numpy as np
from tqdm.auto import tqdm

# spatial
import xarray as xa

# custom
import cbsyst.helpers as cbh


### cbsyst sensitivity analysis
def create_st_ft_sensitivity_array(param_combinations: list, pertubation_percentage: float, resolution: int=20) -> xa.DataArray:
    
    # check if more than three parameters are passed
    if len(param_combinations[0]) != 3:
        raise ValueError("param_combinations should be a list of tuples containing salinity, temperature, and pH_NBS values")
    
    for Sal, Temp, pH_NBS in tqdm(param_combinations):
        ST_base = cbh.calc_ST(Sal)
        FT_base = cbh.calc_FT(Sal)
        
        # define perturbation ranges for ST and FT
        ST_values = np.linspace(ST_base-pertubation_percentage/100 * ST_base,  ST_base+pertubation_percentage/100 * ST_base, 20)
        FT_values = np.linspace(FT_base-pertubation_percentage/100 * FT_base, FT_base+pertubation_percentage/100 * FT_base, 20)

        # initialize empty array for pH_total values
        pH_grid = np.zeros((20, 20))

        for i, ST in enumerate(ST_values):
            for j, FT in enumerate(FT_values):
                pH_Total = cbh.pH_scale_converter(
                    pH=pH_NBS, scale='NBS', ST=ST, FT=FT, Temp=Temp, Sal=Sal
                ).get('pHtot', None)
                pH_grid[i, j] = pH_Total
        results_dict[(Sal, Temp, pH_NBS)] = pH_grid.copy()

    # extract unique values for each dimension
    Sal_values = sorted(set(k[0] for k in results_dict.keys()))
    Temp_values = sorted(set(k[1] for k in results_dict.keys()))
    pH_NBS_values = sorted(set(k[2] for k in results_dict.keys()))
    # store arrays with dimensions metadata
    data_array = np.empty((len(Sal_values), len(Temp_values), len(pH_NBS_values), len(ST_values), len(FT_values)))
    for (v1, v2, v3), arr in results_dict.items():
        i = Sal_values.index(v1)
        j = Temp_values.index(v2)
        k = pH_NBS_values.index(v3)
        data_array[i, j, k, :, :] = arr

    return xa.DataArray(
        data_array,
        dims=["salinity", "temperature", "ph_nbs", "ST", "FT"],
        coords={
            "salinity": Sal_values,
            "temperature": Temp_values,
            "ph_nbs": pH_NBS_values,
            "ST": ST_values,
            "FT": FT_values,
        },
        name="pH_Total"
    )
    

def select_by_stat(ds: xa.Dataset, variables_stats: dict):
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


### DEPRECATED

# Function to determine optimal number of clusters using silhouette score
# def optimal_kmeans(data, max_clusters=8):
#     best_k = 2  # Minimum sensible number of clusters
#     best_score = -1
#     scores = []

#     for k in range(2, min(len(data), max_clusters + 1)):  # Avoid excessive clustering
#         kmeans = KMeans(n_clusters=k, random_state=42, n_init=max_clusters)
#         labels = kmeans.fit_predict(data)
#         score = silhouette_score(data, labels)
#         scores.append((k, score))

#         if score > best_score:
#             best_score = score
#             best_k = k

#     return best_k, scores


# def rate_conversion(rate_val: float, rate_unit: str) -> tuple[float, str]:
#     """Conversion into gCaCO3 ... day-1 for absolute rates"""
#     rate_components = rate_unit.split(' ')
#     # print(rate_components)
#     num = rate_components[0]
#     denom = rate_components[1]
    
#     # if rate_unit == 'um d-1':
#     #     print('problem')
#     # process numerator
#     if 'delta' in num:  # no conversion needed
#         # rate_val = rate_val
#         new_rate_unit_num = num.replace('delta%', 'delta%')
#     else:
#         # Initialize num_prefix as an empty string
#         num_prefix = ""
        
#         new_rate_unit_num = num
#         if 'mol' in num:    # moles
#             rate_val *= MOLAR_MASS_CACO3
#             num_prefix = num.split('mol')[0]
#             num = num.replace('mol', 'g')
#             new_rate_unit_num = num.replace(num_prefix, '')
    
#         elif 'g' in num:    # mass
#             num_prefix = num.split('g')[0]
#             # new_rate_unit_num = num.replace(num_prefix, 'g')
#             new_rate_unit_num = num.replace(num_prefix, '')
#         elif 'm2' in num:    # area (need to square)
#             num_prefix = num.split('m2')[0]
#             # if more than one 'num_prefix' occurence, do this
#             if num.count(num_prefix) > 1 and num_prefix != '':
#                 new_rate_unit_num = num[1:]
#         elif 'm' in num:    # extension
#             num_prefix = num[0] # TODO: always valid?
#             if num.count(num_prefix) > 1:   # mm
#                 new_rate_unit_num = num[1:]
#             else:
#                 new_rate_unit_num = num.replace(num_prefix, '')
                    
#         # check if prefix exists and is valid
#         if num_prefix in PREFIXES:
#             if 'm2' in num:  # area
#                 rate_val *= PREFIXES[num_prefix]**2
#             else:
#                 rate_val *= PREFIXES[num_prefix]
        
#     denom_prefix = ""
#     new_rate_unit_denom = denom
#     # process denominator
#     if 'm-2' in denom:  # specific area
#         denom_prefix = denom.split('m-2')[0]
#         if denom.count(denom_prefix) > 1 and denom_prefix != '':    # mm
#             new_rate_unit_denom = 'c'+denom[1:]
#             denom_prefix = 'm'
#         else:
#             new_rate_unit_denom = denom.replace(denom_prefix, 'c')
                
#         if denom_prefix in PREFIXES:
#             rate_val /= (PREFIXES[denom_prefix]/PREFIXES['c'])**2
#     elif 'g' in denom:  # specific mass
#         denom_prefix = denom.split('g')[0]
#         if denom_prefix in PREFIXES:
#             rate_val /= PREFIXES[denom_prefix]

#         new_rate_unit_denom = denom.replace(denom_prefix, '')
#     # time conversion
#     if any(duration in denom for duration in DURATIONS.keys()):
#         for duration, factor in DURATIONS.items():
#             if duration in denom:
#                 rate_val *= factor
#                 new_rate_unit_denom = new_rate_unit_denom.replace(duration, 'd')
#                 break
            
#     new_rate_unit = f"{new_rate_unit_num} {new_rate_unit_denom}"

#     return rate_val, new_rate_unit   

    
    
    
    # def uniquify_dois(df):
#     """
#     Uniquify dois to reflect locations (for studies with multiple locations)
    
#     Args:
#         df (pd.DataFrame): dataframe containing doi and location columns
    
#     Returns:
#         pd.DataFrame: dataframe with uniquified dois, and copies of original
#     """
#     df['original_doi'] = df['doi']
#     temp_df = df.copy()
#     temp_df['location_lower'] = temp_df['location'].str.lower()
    
    
    
#     unique_locs = temp_df.drop_duplicates(['location_lower', 'coords', 'cleaned_coords'])[['doi', 'location']]
#     # unique_locs.dropna(subset=['latitude', 'longitude'], inplace=True) # remove empty rows
#     dois = unique_locs.doi
#     temp_df.index = uniquify_repeated_values(dois)
#     doi_location_map = dict(zip(zip(temp_df.drop_duplicates(subset=['doi', 'location_lower', 'coords', 'cleaned_coords'])['doi'], 
#                                 temp_df.drop_duplicates(subset=['doi', 'location_lower', 'coords', 'cleaned_coords'])['location']), 
#                             dois))
#     temp_df['doi'] = [doi_location_map.get((doi, loc), doi) for doi, loc in zip(temp_df['doi'], temp_df['location'])]
#     # temp_df['doi'] = temp_df.index
#     return temp_df

    
    
    # unique_locs.doi = utils.uniquify_repeated_values(df.drop_duplicates(subset=['doi', 'location']).doi)

    # # create a dictionary mapping from original (doi, location) pairs to uniquified dois
    # doi_location_map = dict(zip(zip(df.drop_duplicates(subset=['doi', 'location'])['doi'], 
    #                             df.drop_duplicates(subset=['doi', 'location'])['location']), 
    #                         unique_locs['doi']))
    # df['doi'] = [doi_location_map.get((doi, loc), doi) for doi, loc in zip(df['doi'], df['location'])]
    
    # return df