import numpy as np
import pandas as pd
import xarray as xa
from itertools import combinations
from tqdm.auto import tqdm

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# custom
import utils


# custom
import cbsyst.helpers as cbh

def calc_cohens_d(mu1: float, mu2: float, sd_pooled: float) -> float:
    """Calculate Hedges G metric: https://www.itl.nist.gov/div898/software/dataplot/refman2/auxillar/hedgeg.htm
    
    Args:
        mu1 (float): mean of group 1
        mu2 (float): mean of group 2
    2 (float): mean of group 2
        sd_pooled (float): pooled standard deviation of both groups
        
    Returns:
        float: Hedges G metric
    """
    return (mu1 - mu2) / sd_pooled


def calc_sd_pooled(n1: int, n2: int, sd1: float, sd2: float) -> float:
    """Calculate pooled standard deviation: https://www.itl.nist.gov/div898/software/dataplot/refman2/auxillar/hedgeg.htm
    
    Args:
        n1 (int): number of samples in group 1
        n2 (int): number of samples in group 2
        sd1 (float): standard deviation of group 1
        sd2 (float): standard deviation of group 2
        
    Returns:
        float: pooled standard deviation
    """
    return np.sqrt(((n1 - 1) * sd1 ** 2 + (n2 - 1) * sd2 ** 2) / (n1 + n2 - 2))


def calc_cohens_d_var(n1: int, n2: int, d: float) -> float:
    """Calculate variance of Cohen's d metric: https://www.campbellcollaboration.org/calculator/equations
    
    Args:
        n1 (int): number of samples in group 1
        n2 (int): number of samples in group 2
        d (float): Hedges G metric
        
    Returns:
        float: variance of Hedges G metric
    """
    return (n1 + n2) / (n1 * n2) + d ** 2 / (2 * (n1 + n2))


def calc_bias_correction(n1: int, n2: int) -> float:
    """Calculate bias correction for Cohen's d metric: https://www.campbellcollaboration.org/calculator/equations
    
    Args:
        n1 (int): number of samples in group 1
        n2 (int): number of samples in group 2
        
    Returns:
        float: bias correction factor
    """
    return 1 - 3 / (4 * (n1 + n2 - 2) - 1)


def calc_hedges_g(mu1: float, mu2: float, sd1: float, sd2: float, n1: int, n2: int) -> float:
    """Calculate Hedges G metric: https://www.campbellcollaboration.org/calculator/equations
    
    Args:
        mu1 (float): mean of group 1
        mu2 (float): mean of group 2
        sd1 (float): standard deviation of group 1
        sd2 (float): standard deviation of group 2
        n1 (int): number of samples in group 1
        n2 (int): number of samples in group 2
        
    Returns:
        float: Hedges G metric
    """
    sd_pooled = calc_sd_pooled(n1, n2, sd1, sd2)
    d = calc_cohens_d(mu1, mu2, sd_pooled)
    var = calc_cohens_d_var(n1, n2, d)
    bias_correction = calc_bias_correction(n1, n2)
    
    hedges_g = d * bias_correction
    # calculate 95% confidence intervals
    se_g = np.sqrt(var*bias_correction**2)  # standard error
    hedges_g_lower = hedges_g - 1.959964 * se_g
    hedges_g_upper = hedges_g + 1.959964 * se_g
    
    return hedges_g, (hedges_g_lower, hedges_g_upper)


def create_st_ft_sensitivity_array(param_combinations: list, pertubation_percentage: float, resolution: int=20) -> xa.DataArray:
    # TODO: make this generic for any sensitivity     results_dict = {}
    
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


def cluster_treatments(df, vars_to_cluster):
    """Cluster treatments based on independent variables and species types."""
    
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning, module='sklearn.cluster._kmeans')  # Suppress convergence warnings

    df = df.copy()  # Avoid modifying the original dataframe
    # remove nans which would cause issues for clustering
    # df = df.drop(columns=[col for col in independent_vars if df[col].isna().all()])    
    # df = df.dropna(subset=[col for col in independent_vars if col in df.columns], how='any', axis=0)

    # Ensure 'treatment_group' column exists
    df['treatment_group'] = np.nan

    # Cluster separately for each species type
    for species, group_df in df.groupby('species_types'):
        treatment_data = group_df[vars_to_cluster].dropna(axis=1)  # Only keep non-missing variables

        if treatment_data.shape[0] < 2:  # Skip clustering if there's only one sample
            print(f"Skipping {species}: Not enough samples for clustering: {len(group_df)}")
            continue

        try:
            optimal_k, _ = optimal_kmeans(treatment_data)  # Determine optimal clusters
            kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
            df.loc[group_df.index, 'treatment_group'] = kmeans.fit_predict(treatment_data)
        except ValueError:
            print(f"Error: Could not cluster {species}, df of length {len(group_df)}")
            df.loc[group_df.index, 'treatment_group'] = np.nan

    return df


def determine_control_conditions(df):
    """Identify the rows corresponding to min temperature and/or max pH."""
    grouped = df.groupby('treatment_group')

    control_treatments = {}

    for group, sub_df in grouped:
        group = int(group)  # convert group to integer for semantics
        min_temp = sub_df.loc[sub_df['t_in'].idxmin()]['t_in'] if not any(sub_df['phtot'].isna()) else None # Row with minimum temperature
        max_pH = sub_df.loc[sub_df['phtot'].idxmax()]['phtot'] if not any(sub_df['phtot'].isna()) else None # Row with maximum pH

        control_treatments[group] = {
            'control_t_in': min_temp,
            'control_phtot': max_pH,
        }

    return control_treatments


def assign_treatment_groups_multilevel(df: pd.DataFrame, t_atol: float=0.5, pH_atol: float=0.1, irr_atol: float=30) -> pd.DataFrame:
    """
    Assign treatment groups to each row based on temperature and pH values,
    recognizing multiple levels of treatments.
    
    Args:
        df (pd.DataFrame): Input dataframe with columns 'doi', 't_in', 'phtot', etc.
        t_atol (float): Absolute tolerance for temperature comparison.
        pH_atol (float): Absolute tolerance for pH comparison.
        irr_rtol (float): Relative tolerance for irradiance grouping.
        
    Returns:
        pd.DataFrame: Original dataframe with added 'treatment_group' and 'treatment_level' columns.
    """
    # Create a copy of the original dataframe
    result_df = df.copy()
    
    # Initialize treatment columns as object dtype
    result_df['treatment_group'] = pd.Series(dtype='object')
    result_df['treatment_level_t'] = pd.Series(dtype='object')
    result_df['treatment_level_ph'] = pd.Series(dtype='object')
    
    # Process each study separately
    for study_doi, study_df in df.groupby('doi'):
        # Group by irradiance within this study
        study_with_irr_groups = group_irradiance(study_df, atol=irr_atol)
        
        # Process each (irradiance group, species) combination separately
        for (irr_group, species), group_df in study_with_irr_groups.groupby(['irr_group', 'species_types']):
            # Skip if too few samples
            if len(group_df) <= 1:
                continue
                
            # Find control values (min T, max pH)
            control_T = group_df['t_in'].min() if not group_df['t_in'].isna().all() else None
            control_pH = group_df['phtot'].max() if not group_df['phtot'].isna().all() else None
            
            # Cluster temperature values
            t_values = group_df['t_in'].dropna().unique()
            t_clusters = cluster_values(t_values, t_atol)
            
            # Cluster pH values
            ph_values = group_df['phtot'].dropna().unique()
            ph_clusters = cluster_values(ph_values, pH_atol)
            
            # Map each value to its cluster index
            t_mapping = {val: cluster_idx for cluster_idx, cluster in enumerate(t_clusters) for val in cluster}
            ph_mapping = {val: cluster_idx for cluster_idx, cluster in enumerate(ph_clusters) for val in cluster}
            
            # Apply classification to each row in this group
            for idx in group_df.index:
                row = group_df.loc[idx]
                
                # Get temperature cluster level (0 is control)
                t_level = None
                if not np.isnan(row['t_in']) and control_T is not None:
                    t_cluster_idx = t_mapping.get(row['t_in'])
                    control_cluster_idx = t_mapping.get(control_T)
                    if t_cluster_idx is not None and control_cluster_idx is not None:
                        t_level = t_cluster_idx - control_cluster_idx
                
                # Get pH cluster level (0 is control)
                ph_level = None
                if not np.isnan(row['phtot']) and control_pH is not None:
                    ph_cluster_idx = ph_mapping.get(row['phtot'])
                    control_cluster_idx = ph_mapping.get(control_pH)
                    if ph_cluster_idx is not None and control_cluster_idx is not None:
                        ph_level = control_cluster_idx - ph_cluster_idx  # Reverse order since higher pH is control
                
                # Determine if values are in control clusters
                is_control_T = t_level == 0 if t_level is not None else False
                is_control_pH = ph_level == 0 if ph_level is not None else False
                
                # Classify the treatment
                if is_control_T and is_control_pH:
                    treatment = 'cTcP'
                elif is_control_T:
                    treatment = 'cTtP'
                elif is_control_pH:
                    treatment = 'tTcP'
                elif not (is_control_T or is_control_pH):
                    treatment = 'tTtP'
                else:
                    treatment = 'uncertain'
                
                # Update the treatment info in the result dataframe
                result_df.loc[idx, 'treatment_group'] = treatment
                result_df.loc[idx, 'treatment_level_t'] = t_level if t_level is not None else np.nan
                result_df.loc[idx, 'treatment_level_ph'] = ph_level if ph_level is not None else np.nan
                result_df.loc[idx, 'irr_group'] = irr_group
    
    result_df['treatment'] = result_df['treatment_group'].apply(
        lambda x: 't_in-phtot' if isinstance(x, str) and 'tT' in x and 'tP' in x else 
                 't_in' if isinstance(x, str) and 'tT' in x else 
                 'phtot' if isinstance(x, str) and 'tP' in x else 
                 'control' if isinstance(x, str) and x == 'cTcP' else np.nan
    )
    return result_df


def group_irradiance(df, irr_col='irr', atol=30):
    """
    Assigns an 'irr_group' to values that are within a relative tolerance.
    
    Args:
        df (pd.DataFrame): Input dataframe with an 'irr' column.
        irr_col (str): Column name for irradiance values.
        rtol (float): Relative tolerance (e.g., 0.10 for 10%).
        
    Returns:
        pd.DataFrame: Dataframe with new 'irr_group' column.
    """
    df = df.copy().sort_values(by=irr_col)  # copy to prevent modification of original, sort for efficiency
    irr_groups = np.zeros(len(df))  # initialize array for group assignments
    group_id = 0
    prev_irr = None

    for i, irr in enumerate(df[irr_col]):
        if np.isnan(irr):   # handling grouping where irr is NaN
            irr_groups[i] = -1  # Use a distinct value for NaN irradiance
        elif prev_irr is None:
            # First non-NaN value starts group 0
            group_id = 0
            irr_groups[i] = group_id
        elif np.abs(irr - prev_irr) > atol:
            # Outside tolerance, create new group
            group_id += 1
            irr_groups[i] = group_id
        else:
            # Within tolerance, use current group
            irr_groups[i] = group_id
            
        # Update prev_irr only for non-NaN values
        if not np.isnan(irr):
            prev_irr = irr

    # Add irr_group column to dataframe
    df['irr_group'] = irr_groups
    return df


def cluster_values(values, tolerance):
    """
    Cluster values based on their proximity.
    
    Args:
        values (array-like): Values to cluster.
        tolerance (float): Tolerance for clustering.
        
    Returns:
        list: List of clusters, where each cluster is a list of values.
    """
    if len(values) == 0:
        return []
        
    # Sort values
    sorted_values = np.sort(values)
    
    # Initialize clusters
    clusters = [[sorted_values[0]]]
    
    # Cluster remaining values
    for val in sorted_values[1:]:
        # Check if value is close to the last value in the current cluster
        if np.abs(val - clusters[-1][-1]) <= tolerance:
            # Add to current cluster
            clusters[-1].append(val)
        else:
            # Start new cluster
            clusters.append([val])
    
    return clusters


def hedges_g_for_row(treatment_row, control_data):
    """
    Calculate Hedges' g and append additional columns for a treatment row.
    
    Args:
        treatment_row: Row containing treatment data
        control_data: Dictionary containing control group data
    
    Returns:
        pandas.Series: Modified row with Hedges' g calculations
    """
    mu_t, sd_t, n_t = treatment_row['calcification'], treatment_row['calcification_sd'], treatment_row['n']
    mu_c, sd_c, n_c = control_data['mu'], control_data['sd'], control_data['n']
    t_in_c, ph_c = control_data['t_in'], control_data['ph']
    
    if np.isnan(mu_t) or np.isnan(mu_c) or np.isnan(sd_t) or np.isnan(sd_c):
        print(f"Missing data for Hedges' g calculation. mu_t: {mu_t:.3f}, mu_c: {mu_c:.3f}, sd_t: {sd_t:.3f}, sd_c: {sd_c:.3f}, n_t: {n_t:.3f}, n_c: {n_c:.3f}")
    
    # Calculate Hedges' g
    h_g, (h_g_l, h_g_u) = calc_hedges_g(mu_t, mu_c, sd_t, sd_c, n_t, n_c)
    
    row_copy = treatment_row.copy() # create a copy to avoid SettingWithCopyWarning
    
    row_copy['delta_t'] = row_copy['t_in'] - t_in_c
    row_copy['delta_pH'] = row_copy['phtot'] - ph_c
    row_copy['treatment_val'] = row_copy['t_in'] if row_copy['treatment'] == 't_in' else row_copy['phtot']
    row_copy['hedges_g'] = h_g
    row_copy['hedges_g_l'] = h_g_l
    row_copy['hedges_g_u'] = h_g_u
    row_copy['control_calcification'] = mu_c
    row_copy['treatment_calcification'] = mu_t
    
    return row_copy


def aggregate_by_treatment_group(df: pd.DataFrame) -> pd.Series:
    
    aggregation = df.agg({
        'calcification': ['mean', 'std'],
        'n': 'count'
    })
    control_row = df.iloc[0]
    control_row = control_row.copy()
    control_row['calcification'] = aggregation['calcification']['mean']
    control_row['calcification_sd'] = aggregation['calcification']['std']
    control_row['n'] = aggregation['n']['count']
    return control_row
    
    

def process_group(species_df):
    """
    Process a group of species data to calculate Hedges' g.
    
    Args:
        species_df: DataFrame containing data for a specific species
        
    Returns:
        pandas.DataFrame: DataFrame with Hedges' g calculations
    """
    loi = ['phtot', 't_in', 'calcification', 'calcification_sd', 'n']
    # Extract control data
    control_df = species_df[species_df['treatment_group'] == 'cTcP']
    if control_df.empty:
        return species_df
    if len(control_df) > 1:
        # aggregate control data by taking the mean of the rows. Insert the standard deviation of the 'calcification' column in the 'calcification_sd' column
        control_row = aggregate_by_treatment_group(control_df)
    else:
        control_row = control_df.iloc[0]
        
    control_data = {
        'mu': control_row['calcification'],
        'sd': control_row['calcification_sd'],
        'n': control_row['n'],
        't_in': control_row['t_in'],
        'ph': control_row['phtot']
    }
    
    
    # process each treatment group
    result_dfs = []
    for treatment_group, treatment_df in species_df.groupby('treatment_group'):
        # in the case that all n == 1 (individual datapoints), aggregate by treatment_group
        if np.all(treatment_df.n == 1):
            treatment_row = aggregate_by_treatment_group(treatment_df) # hedges_g_for_row requires df input
            result_dfs.append(pd.DataFrame(hedges_g_for_row(treatment_row, control_data)).T)
            continue
        if treatment_group in ['tTtP', 'uncertain', 'cTcP']:
            # result_dfs.append(treatment_df)
            continue                
                
        # apply hedges_g calculation to each row
        processed_df = treatment_df.apply(
            lambda row: hedges_g_for_row(row, control_data),
            axis=1
        )
        result_dfs.append(processed_df) if not processed_df.empty else None
    
    # return pd.concat(result_dfs, axis=0) if result_dfs else species_df
    return pd.concat(result_dfs, axis=0)


def hedges_g_for_df(df) -> pd.DataFrame:
    """
    Calculate Hedges' g for a DataFrame of experimental results.
    
    Args:
        df: DataFrame containing experimental data
        
    Returns:
        pandas.DataFrame: DataFrame with added Hedges' g calculations
    """
    # copy to avoid modifying original
    result_df = df.copy()
    
    hedges_cols = ['delta_t', 'delta_pH', 'hedges_g', 'hedges_g_l', 'hedges_g_u']
    for col in hedges_cols:
        result_df[col] = np.nan
    
    # group by relevant factors and apply processing
    grouped_data = []
    for doi, study_df in tqdm(result_df.groupby('doi')):
        print(doi)
        for irr_group, irr_df in study_df.groupby('irr_group'):
            for species, species_df in irr_df.groupby('species_types'):
                grouped_data.append(process_group(species_df))
    
    # combine processed data
    if grouped_data:
        result_df = pd.concat(grouped_data)
    
    return result_df



# def compute_heds
    # def calculate_effect_size(df1_sample, df2_sample, var, group1, group2):
    #     other_var = 't_in' if var == 'phtot' else 'phtot'
    #     if np.isclose(df1_sample[other_var], df2_sample[other_var], atol=0.1):
    #         if (var == 'phtot' and df1_sample[var] < df2_sample[var]) or (var == 't_in' and df1_sample[var] > df2_sample[var]):
    #             group1, group2 = group2, group1
    #             df1_sample, df2_sample = df2_sample, df1_sample
    #         delta_var = abs(df2_sample[var] - df1_sample[var])
    #         mu1, std1, n1 = df1_sample['calcification'], df1_sample['calcification_sd'], df1_sample['n']
    #         mu2, std2, n2 = df2_sample['calcification'], df2_sample['calcification_sd'], df2_sample['n']
    #         g = calc_hedges_g(mu2, mu1, std2, std1, n2, n1)
    #         return {
    #             'doi': df.doi.iloc[0],
    #             'species_types': species,
    #             'group1': group1,
    #             'group2': group2,
    #             'variable': var,
    #             'delta_var': delta_var,
    #             'hedges_g': g[0],
    #             'hg_ci_l': g[1][0],
    #             'hg_ci_u': g[1][1],
    #             'control_val': df1_sample[var],
    #             'treatment_val': df2_sample[var],
    #         }
    #     return None

    # results = []
    # for species, group_df in df.groupby('species_types'):
    #     treatment_groups = group_df['treatment_group'].unique()
    #     for group1, group2 in combinations(treatment_groups, 2):
    #         df1 = group_df[group_df['treatment_group'] == group1]
    #         df2 = group_df[group_df['treatment_group'] == group2]
    #         if df1.n.all() == 1 and df2.n.all() == 1:
    #             df1['calcification_sd'] = np.std(df1['calcification'])
    #             df2['calcification_sd'] = np.std(df2['calcification'])
    #             n1, n2 = len(df1), len(df2)
    #             df1 = utils.aggregate_df(df1)
    #             df2 = utils.aggregate_df(df2)
    #             df1['n'] = n1
    #             df2['n'] = n2
    #         if len(df1) != len(df2):
    #             print(f"Skipping comparison between {group1} and {group2} treatments: Different sample sizes")
    #             continue
    #         for sample in range(len(df1)):
    #             df1_sample = df1.iloc[sample] if isinstance(df1, pd.DataFrame) else df1
    #             df2_sample = df2.iloc[sample] if isinstance(df2, pd.DataFrame) else df2
    #             for var in vars_to_compare:
    #                 result = calculate_effect_size(df1_sample, df2_sample, var, group1, group2)
    #                 if result:
    #                     results.append(result)
    # return results


# def compute_hedges_g(df, vars_to_compare=['t_in', 'phtot', 'irr']):
#     """
#     Compute Hedges' g effect size for each treatment compared to the control within each species.

#     Args:
#         df (pd.DataFrame): DataFrame containing treatment groups.
#         vars_to_compare (list): List of numeric variables to compute Hedges' g for.

#     Returns:
#         pd.DataFrame: DataFrame containing effect sizes for each treatment group.
#     """
#     cols = vars_to_compare + ['species_types', 'treatment_group', 'calcification', 'n']
#     results = []
#     # remove any vars to compare that are not in the df (accounts for singular response studies)
#     df['multi_var'] = 1
#     # remove any of vars_to_compare columns which are all nan
#     specific_vars_to_compare = [col for col in vars_to_compare if df[col].notna().all()]
     
#     # if vars_to_compare updated, print message
#     if len(specific_vars_to_compare) != len(vars_to_compare):
#         print(f"Vars to compare updated to {specific_vars_to_compare} for {df.doi.iloc[0]}")
#         # update value of 'multi-var' column to 1
#         df['multi_var'] = 0

#     for species, group_df in df.groupby('species_types'):
#         treatment_groups = group_df['treatment_group'].unique()
#         for tg1, tg2 in combinations(treatment_groups, 2):
#             df1 = group_df[group_df['treatment_group'] == tg1]
#             df2 = group_df[group_df['treatment_group'] == tg2]

#             if len(df1) != len(df2):
#                 print(f"Skipping comparison between {tg1} and {tg2} treatments: Different sample sizes")
#                 continue
            
#             if (df1['n'] == 1).all() and(df2['n'] == 1).all():
#                 df1['calcification_sd'] = np.std(df1['calcification'])
#                 df2['calcification_sd'] = np.std(df2['calcification'])
#                 n1, n2 = len(df1), len(df2)
#                 df1 = utils.aggregate_df(df1)
#                 df2 = utils.aggregate_df(df2)
#                 df1['n'] = n1
#                 df2['n'] = n2
                
#             for sample in range(len(df1)):
#                 df1_sample = df1.iloc[sample] if isinstance(df1, pd.DataFrame) else df1
#                 df2_sample = df2.iloc[sample] if isinstance(df2, pd.DataFrame) else df2
#                 for var in specific_vars_to_compare:
#                     other_var = 't_in' if var == 'phtot' else 'phtot'
#                     if np.isclose(df1_sample[other_var], df2_sample[other_var], atol=0.1):  # if agree within 0.1, likely control variable
#                         # ensure that df1_sample is control group (smaller value for t_in, larger value for phtot). # TODO: allow switching for deltavar relative to climatology
#                         if (var == 'phtot' and df1_sample[var] < df2_sample[var]) or (var == 't_in' and df1_sample[var] > df2_sample[var]):
#                             tg1, tg2 = tg2, tg1
#                             df1_sample, df2_sample = df2_sample, df1_sample
#                         delta_var = abs(df2_sample[var] - df1_sample[var])
#                         print('df1_sample', df1_sample[cols])
#                         print('df2_sample', df2_sample[cols])
#                         mu1, std1, n1 = df1_sample['calcification'], df1_sample['calcification_sd'], df1_sample['n']    # control
#                         mu2, std2, n2 = df2_sample['calcification'], df2_sample['calcification_sd'], df2_sample['n']    # treatment
#                         g = calc_hedges_g(mu2, mu1, std2, std1, n2, n1)
#                         results.append({
#                             'doi': df.doi.iloc[0],
#                             'location': df.location.iloc[0],
#                             'species_types': species,
#                             'group1': tg1,
#                             'group2': tg2,
#                             'variable': var,
#                             'delta_var': delta_var,
#                             'hedges_g': g[0],
#                             'hg_ci_l': g[1][0],
#                             'hg_ci_u': g[1][1],
#                             'control_val': df1_sample[var],
#                             'treatment_val': df2_sample[var],
#                             'multi_var': df['multi_var'].iloc[0]
#                         })
#     return results
