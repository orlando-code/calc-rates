import numpy as np
import pandas as pd
import xarray as xa
from itertools import combinations
from tqdm.auto import tqdm

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


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
    # TODO: make this generic for any sensitivity analysis
    results_dict = {}
    
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

    # Ensure 'treatment_group' column exists
    df['treatment_group'] = np.nan

    # Cluster separately for each species type
    for species, group_df in df.groupby('species_types'):
        treatment_data = group_df[vars_to_cluster].dropna(axis=1)  # Only keep non-missing variables

        if treatment_data.shape[0] < 2:  # Skip clustering if there's only one sample
            print(f"Skipping {species}: Not enough samples for clustering")
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
        group = int(group)  # convert group to integer for consistency
        min_temp = sub_df.loc[sub_df['t_in'].idxmin()]['t_in'] if not any(sub_df['phtot'].isna()) else None # Row with minimum temperature
        max_pH = sub_df.loc[sub_df['phtot'].idxmax()]['phtot'] if not any(sub_df['phtot'].isna()) else None # Row with maximum pH

        control_treatments[group] = {
            'control_t_in': min_temp,
            'control_phtot': max_pH,
        }

    return control_treatments


def compute_hedges_g(df, vars_to_compare=['t_in', 'phtot']):
    """
    Compute Hedges' g effect size for each treatment compared to the control within each species.

    Args:
        df (pd.DataFrame): DataFrame containing treatment groups.
        vars_to_compare (list): List of numeric variables to compute Hedges' g for.

    Returns:
        pd.DataFrame: DataFrame containing effect sizes for each treatment group.
    """
    def calculate_effect_size(df1_sample, df2_sample, var, group1, group2):
        other_var = 't_in' if var == 'phtot' else 'phtot'
        if np.isclose(df1_sample[other_var], df2_sample[other_var], atol=0.05):
            if (var == 'phtot' and df1_sample[var] < df2_sample[var]) or (var == 't_in' and df1_sample[var] > df2_sample[var]):
                group1, group2 = group2, group1
                df1_sample, df2_sample = df2_sample, df1_sample
            delta_var = abs(df2_sample[var] - df1_sample[var])
            mu1, std1, n1 = df1_sample['calcification'], df1_sample['calcification_sd'], df1_sample['n']
            mu2, std2, n2 = df2_sample['calcification'], df2_sample['calcification_sd'], df2_sample['n']
            g = calc_hedges_g(mu2, mu1, std2, std1, n2, n1)
            return {
                'doi': df.doi.iloc[0],
                'species_types': species,
                'group1': group1,
                'group2': group2,
                'variable': var,
                'delta_var': delta_var,
                'hedges_g': g[0],
                'hg_ci_l': g[1][0],
                'hg_ci_u': g[1][1],
                'control_val': df1_sample[var],
                'treatment_val': df2_sample[var],
            }
        return None

    results = []
    for species, group_df in df.groupby('species_types'):
        treatment_groups = group_df['treatment_group'].unique()
        for group1, group2 in combinations(treatment_groups, 2):
            df1 = group_df[group_df['treatment_group'] == group1]
            df2 = group_df[group_df['treatment_group'] == group2]
            if df1.n.all() == 1 and df2.n.all() == 1:
                df1['calcification_sd'] = np.std(df1['calcification'])
                df2['calcification_sd'] = np.std(df2['calcification'])
                n1, n2 = len(df1), len(df2)
                df1 = utils.aggregate_df(df1)
                df2 = utils.aggregate_df(df2)
                df1['n'] = n1
                df2['n'] = n2
            if len(df1) != len(df2):
                print(f"Skipping comparison between {group1} and {group2} treatments: Different sample sizes")
                continue
            for sample in range(len(df1)):
                df1_sample = df1.iloc[sample] if isinstance(df1, pd.DataFrame) else df1
                df2_sample = df2.iloc[sample] if isinstance(df2, pd.DataFrame) else df2
                for var in vars_to_compare:
                    result = calculate_effect_size(df1_sample, df2_sample, var, group1, group2)
                    if result:
                        results.append(result)
    return results


### deprecated
def compute_hedges_g_dep(df, vars_to_compare=['t_in', 'phtot']):
    """
    Compute Hedges' g effect size for each treatment compared to the control within each species.

    Args:
        df (pd.DataFrame): DataFrame containing treatment groups.
        vars_to_compare (list): List of numeric variables to compute Hedges' g for.

    Returns:
        pd.DataFrame: DataFrame containing effect sizes for each treatment group.
    """
    results = []

    for species, group_df in study_df.groupby('species_types'):
        treatment_groups = group_df['treatment_group'].unique()
        
        # Compare each pair of treatment groups
        for group1, group2 in combinations(treatment_groups, 2):
            df1 = group_df[group_df['treatment_group'] == group1]
            df2 = group_df[group_df['treatment_group'] == group2]
            
            if df1.n.all() == 1 and df2.n.all() == 1:
                # aggregate data (take mean calcification and sd)
                df1['calcification_sd'] = np.std(df1['calcification'])
                df2['calcification_sd'] = np.std(df2['calcification'])
                n1, n2 = len(df1), len(df2)
                
                df1 = utils.aggregate_df(df1)
                df2 = utils.aggregate_df(df2)
                df1['n'] = n1
                df2['n'] = n2

            if len(df1) != len(df2):
                print(f"Skipping comparison between {group1} and {group2} treatments: Different sample sizes")
                continue
            
            if isinstance(df1, pd.DataFrame):   # if there are multiple samples
                for sample in range(len(df1)):
                    df1_sample = df1.iloc[sample]
                    df2_sample = df2.iloc[sample]
                    
                    for var in vars_to_compare:

                        other_var = 't_in' if var == 'phtot' else 'phtot'  # The variable that must remain constant
                        
                        # Check if the other variable is approximately the same in both groups
                        if np.isclose(df1_sample[other_var], df2_sample[other_var], atol=0.05):  # Adjust tolerance as needed
                            if var == 'phtot':
                                # mu1 is greater than mu2, switch groups such that control = mu1
                                if df1_sample[var] < df2_sample[var]:
                                    group1, group2 = group2, group1
                                    df1_sample, df2_sample = df2_sample, df1_sample
                                delta_var = df2_sample[var] - df1_sample[var]
                            elif var == 't_in':
                                # mu1 is less than mu2, switch groups such that control = mu1
                                if df1_sample[var] > df2_sample[var]:
                                    group1, group2 = group2, group1
                                    df1_sample, df2_sample = df2_sample, df1_sample
                                delta_var = df1_sample[var] - df2_sample[var]
                                    
                            mu1, std1, n1 = df1_sample['calcification'], df1_sample['calcification_sd'], df1_sample['n']
                            mu2, std2, n2 = df2_sample['calcification'], df2_sample['calcification_sd'], df2_sample['n']

                            g = analysis.calc_hedges_g(mu2, mu1, std2, std1, n2, n1)    # TODO: check signage
                            results.append({
                                'doi': study_df.doi.iloc[0],
                                'species_types': species,
                                'group1': group1,
                                'group2': group2,
                                'variable': var,
                                'delta_var': delta_var,
                                'hedges_g': g[0],
                                'hg_ci_l': g[1][0],
                                'hg_ci_u': g[1][1],
                                })
            else:
                df1_sample = df1
                df2_sample = df2
                for var in vars_to_compare:
                    
                    other_var = 't_in' if var == 'phtot' else 'phtot'  # The variable that must remain constant
                    
                    # Check if the other variable is approximately the same in both groups
                    if np.isclose(df1_sample[other_var], df2_sample[other_var], atol=0.05):  # Adjust tolerance as needed
                        if var == 'phtot':
                            # mu1 is greater than mu2, switch groups such that control = mu1
                            if df1_sample[var] < df2_sample[var]:
                                group1, group2 = group2, group1
                                df1_sample, df2_sample = df2_sample, df1_sample
                            delta_var = df2_sample[var] - df1_sample[var]
                        elif var == 't_in':
                            # mu1 is less than mu2, switch groups such that control = mu1
                            if df1_sample[var] > df2_sample[var]:
                                group1, group2 = group2, group1
                                df1_sample, df2_sample = df2_sample, df1_sample
                            delta_var = df2_sample[var] - df1_sample[var]
                                
                        mu1, std1, n1 = df1_sample['calcification'], df1_sample['calcification_sd'], df1_sample['n']
                        mu2, std2, n2 = df2_sample['calcification'], df2_sample['calcification_sd'], df2_sample['n']

                        g = analysis.calc_hedges_g(mu2, mu1, std2, std1, n2, n1)    # TODO: check signage
                        results.append({
                            'doi': study_df.doi.iloc[0],
                            'location': study_df.location.iloc[0],
                            'species_types': species,
                            'group1': group1,
                            'group2': group2,
                            'variable': var,
                            'delta_var': delta_var,
                            'hedges_g': g[0],
                            'hg_ci_l': g[1][0],
                            'hg_ci_u': g[1][1],
                            'control_val': df1_sample[var],
                            'treatment_val': df2_sample[var],
                        })
    return results

