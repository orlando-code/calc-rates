# general
import os
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import statsmodels.api as sm

# R
import rpy2.robjects as ro
import rpy2.robjects.packages as rpackages
# from rpy2.robjects import pandas2ri
metafor = rpackages.importr("metafor")
base = rpackages.importr("base")

# custom
from calcification import config, processing, file_ops


### core analysis calculations
def calc_relative_rate(mu1, mu2, sd1=None, sd2=None, n1=None, n2=None, epsilon=1e-6):
    """
    Calculate percent change between two means with error propagation.
    
    For different scenarios:
    - 10 to 20: +100%
    - 10 to 0: -100%
    - 10 to -10: -200%
    
    Parameters:
    -----------
    mu1, mu2 : float
        Mean values to compare (mu1=reference/baseline, mu2=new value)
    se1, se2 : float, optional
        Standard errors of mu1 and mu2
    epsilon : float, optional
        Small value to stabilize calculations when means are close to zero
        
    Returns:
    --------
    pc : float
        Percent change
    se_pc : float or None
        Standard error of the percent change
    """
    se1, se2 = sd1/np.sqrt(n1), sd2/np.sqrt(n2)
    # Special case: both means are exactly zero
    if mu1 == 0 and mu2 == 0:
        pc = 0  # No change between the means
        
        # If SEs are provided, calculate the uncertainty
        if se1 is not None and se2 is not None:
            # When both means are zero, consider the ratio of SEs to estimate uncertainty
            # This represents how much percentage change we would expect if the values 
            # fluctuated by ±1 SE from zero
            if se1 > 0:
                # Scale based on the potential percentage fluctuations around zero
                se_pc = 100 * se2 / se1
            else:
                # If se1 is zero but se2 is not, technically infinite uncertainty
                se_pc = float('inf') if se2 > 0 else 0
            return pc, se_pc
        return pc
    
    # Special case: baseline is zero
    if mu1 == 0:
        # Handle the zero baseline case
        # Return signed 100% change
        pc = np.sign(mu2) * 100
        
        # If SE is provided, calculate the uncertainty
        if se1 is not None and se2 is not None:
            # When baseline is zero, use the ratio of SEs to treatment
            if abs(mu2) > epsilon:
                # Error in baseline causes very large fluctuations in percent change
                se_pc = 100 * se1 / abs(mu2)
                
                # Also consider uncertainty in the treatment
                se_pc = np.sqrt(se_pc**2 + (100 * se2 / abs(mu2))**2)
            else:
                # If both are essentially zero, high uncertainty
                se_pc = float('inf') if (se1 > 0 or se2 > 0) else 0
            return pc, se_pc
        return pc
    
    # Standard percent change calculation
    pc = ((mu2 - mu1) / abs(mu1)) * 100
    
    # Return only PC if no standard errors provided
    if se1 is None or se2 is None:
        return pc
    
    # Error propagation - calculate partial derivatives
    dpc_dmu1 = (-mu2 / (mu1**2)) * 100
    dpc_dmu2 = (1 / abs(mu1)) * 100
    
    # Calculate standard error using error propagation
    var_pc = (dpc_dmu1**2 * se1**2) + (dpc_dmu2**2 * se2**2)
    
    return pc, var_pc


def calc_absolute_rate(mu1, mu2, sd1=None, sd2=None, n1=None, n2=None):
    """Calculate the simple difference between two means with error propagation.
    
    Args:
        mu1 (float): mean of group 1 (control)
        mu2 (float): mean of group 2 (treatment)
        sd1 (float): standard deviation of group 1
        sd2 (float): standard deviation of group 2
        n1 (int): number of samples in group 1
        n2 (int): number of samples in group 2
    
    Returns:
        tuple: absolute difference between means, standard error of the difference
    """
    abs_diff = mu2 - mu1
    
    # if standard deviations are provided, calculate the uncertainty
    if sd1 is not None and sd2 is not None and n1 is not None and n2 is not None:
        se1 = sd1 / np.sqrt(n1)
        se2 = sd2 / np.sqrt(n2)
        
        # Error propagation - calculate partial derivatives
        d_abs_diff_dmu1 = -1
        d_abs_diff_dmu2 = 1
        
        # Calculate standard error using error propagation
        var_abs_diff = (d_abs_diff_dmu1**2 * se1**2) + (d_abs_diff_dmu2**2 * se2**2)
        
        return abs_diff, var_abs_diff
    
    return abs_diff, None
    

def calc_bias_correction(n1: int, n2: int) -> float:
    """Calculate bias correction for Cohen's d metric: https://www.campbellcollaboration.org/calculator/equations
    
    Args:
        n1 (int): number of samples in group 1 (control)
        n2 (int): number of samples in group 2 (treatment)
        
    Returns:
        float: bias correction factor
    """
    return 1 - 3 / (4 * (n1 + n2 - 2) - 1)


def calc_cohens_d(mu1: float, mu2: float, sd1: float, sd2: float, n1: int, n2: int) -> tuple[float, float]:
    """Calculate Cohen's d metric: https://www.itl.nist.gov/div898/software/dataplot/refman2/auxillar/hedgeg.htm
    
    Args:
        mu1 (float): mean of group 1 (control)
        mu2 (float): mean of group 2 (treatment)
        sd1 (float): standard deviation of group 1
        sd2 (float): standard deviation of group 2
        n1 (int): number of samples in group 1
        n2 (int): number of samples in group 2
                
    Returns:
        tuple[float, float]: Cohen's d and its variance
    """
    sd_pooled = calc_pooled_sd(n1, n2, sd1, sd2) 
    d = (mu2 - mu1) / sd_pooled
    d_var = (n1 + n2) / (n1 * n2) + d ** 2 / (2 * (n1 + n2))
    return d, d_var


def calc_pooled_sd(n1: int, n2: int, sd1: float, sd2: float) -> float:
    """Calculate pooled standard deviation for two groups.
    N.B. BH (2021) uses simple average
    Args:
        n1 (int): number of samples in group 1 (control)
        n2 (int): number of samples in group 2 (treatment)
        sd1 (float): standard deviation of group 1
        sd2 (float): standard deviation of group 2
        
    Returns:
        float: pooled standard deviation
    """
    return np.sqrt(((n1 - 1) * sd1 ** 2 + (n2 - 1) * sd2 ** 2) / (n1 + n2 - 2))


def calc_hedges_g(mu1: float, mu2: float, sd1: float, sd2: float, n1: int, n2: int) -> float:
    """Calculate Hedges G metric: https://www.campbellcollaboration.org/calculator/equations
    
    Args:
        mu1 (float): mean of group 1 (control)
        mu2 (float): mean of group 2 (treatment)
        sd1 (float): standard deviation of group 1
        sd2 (float): standard deviation of group 2
        n1 (int): number of samples in group 1
        n2 (int): number of samples in group 2
        
    Returns:
        float: Hedges G metric
    """
    d, d_var = calc_cohens_d(mu1, mu2, sd1, sd2, n1, n2)
    bias_correction = calc_bias_correction(n1, n2)
    
    hg = d * bias_correction
    hg_var = d_var * bias_correction ** 2
    # # calculate 95% confidence intervals
    # se_g = np.sqrt(var*bias_correction**2)  # standard error
    # hg_lower = hg - 1.959964 * se_g
    # hg_upper = hg + 1.959964 * se_g
    return hg, hg_var


### meta-analysis functions
def calc_cooks_distance(data: pd.Series) -> pd.Series:
    """
    Calculate Cook's distance for a given data series.
    """
    # if data is not numeric
    if not pd.api.types.is_numeric_dtype(data):
        # convert data to numeric
        data = pd.to_numeric(data, errors='coerce')
    
    # fit OLS model
    X = sm.add_constant(data)
    try:
        model = sm.OLS(data, X).fit()
    except ValueError:
        # convert data to numeric if it is not already
        data = pd.to_numeric(data, errors='coerce')
        model = sm.OLS(data, X).fit()
    # calculate Cook's distance
    influence = model.get_influence()
    cooks_d = influence.cooks_distance[0]
    
    return cooks_d


def calc_cooks_threshold(data: pd.Series, nparams: int) -> float:
    """
    Calculate the Cook's distance threshold for outlier detection via 2√((k+1)/(n - k - 1)): a reproducible numerical replacement of eyeballing for outliers in the distance-study graph.
    """
    n = len(data)
    threshold = 2 * np.sqrt((nparams + 1) / (n - nparams - 1))
    return threshold


def remove_cooks_outliers(df: pd.DataFrame, effect_type: str = 'hedges_g', nparams: int=3) -> pd.DataFrame:
    data = df.copy()
    # calculate cooks distance
    cooks_threshold = calc_cooks_threshold(data[effect_type], nparams=nparams)
    # calculate cooks distance
    data["cooks_d"] = calc_cooks_distance(
        data[effect_type]
    )

    # remove outliers
    data_no_outliers = data[data["cooks_d"] < cooks_threshold]
    outliers = data[data["cooks_d"] >= cooks_threshold]
    print(f"\nRemoved {len(outliers)} outlier(s) (from {len(data)} samples) based on Cook's distance threshold of {cooks_threshold:.2f}")
    return data_no_outliers, outliers


### calculating treatment effects
def calculate_effect_for_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Hedges' g for a DataFrame of experimental results.
    
    Args:
        df: DataFrame containing experimental data
        
    Returns:
        pandas.DataFrame: DataFrame with calculated effect sizes
    """
    # copy to avoid modifying original
    result_df = df.copy()
    
    effect_cols = ['delta_t', 'delta_ph', 
                   'cohens_d', 'cohens_d_var', 'hedges_g', 'hedges_g_var',
                   'relative_calcification', 'relative_calcification_var',
                   'absolute_calcification', 'absolute_calcification_var',
                   'st_relative_calcification', 'st_relative_calcification_var',
                   'st_absolute_calcification', 'st_absolute_calcification_var',]
    for col in effect_cols:
        result_df[col] = np.nan
    
    # group by relevant factors and apply processing
    grouped_data = []
    doi_bar = tqdm(result_df.doi.unique())
    for doi in doi_bar:
        doi_bar.set_description(f"Processing {doi}")
        study_df = result_df[result_df['doi'] == doi]
        for irr_group, irr_df in study_df.groupby('irr_group'):
            for species, species_df in irr_df.groupby('species_types'):
                df = process_group_multivar(species_df)
                if isinstance(df, pd.Series):
                    # Convert Series to DataFrame   (necessary when not using apply)
                    df = pd.DataFrame([df].T)
                if df is not None:
                    grouped_data.extend(df) # was previously extend
    
    if isinstance(grouped_data, list):
        # Fix for performance warnings
        valid_dfs = [df for df in grouped_data if df is not None and not df.empty and not df.isna().all().all()]
        if valid_dfs:
            df = pd.concat(valid_dfs)
            # Sort index to avoid lexsort depth warning and create a copy to avoid fragmentation
            df = df.sort_index().copy()
            df["ID"] = df.index
        else:
            # Return empty DataFrame with same columns and dtypes as expected output
            df = pd.DataFrame(columns=df.columns if len(grouped_data) > 0 and grouped_data[0] is not None else None)
            df["ID"] = df.index
        return df
    elif isinstance(grouped_data, pd.DataFrame):
        # Sort index and create a copy to avoid performance warnings
        df = grouped_data.sort_index().copy()
        df["ID"] = df.index
        return df
    else:
        raise ValueError("Invalid data type for grouped_data. Expected list or DataFrame.")


def process_group_multivar(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process a group of species data to calculate effect size.
    
    Args:
        df: DataFrame containing data for a specific species
    
    Returns:
        pandas.DataFrame: DataFrame with effect size calculations
    """
    def process_group(group, control_level_col):
        control_level = min(group[control_level_col])
        control_df = group[group[control_level_col] == control_level]
        treatment_df = group[group[control_level_col] > control_level]
        
        if treatment_df.empty:  # skip if there's no treatment data
            return
            
        # convert control to series if necessary    # TODO: mean?
        # Convert control_df to Series: take mean of numeric columns and first value of others
        if isinstance(control_df, pd.DataFrame):
            if len(control_df) > 1:
                # Get numeric columns
                numeric_cols = control_df.select_dtypes(include='number').columns
                # Create a Series with first values
                control_series = control_df.iloc[0].copy()
                # Replace numeric columns with their means
                for col in numeric_cols:
                    control_series[col] = control_df[col].mean()
            else:
                control_series = control_df.iloc[0]
        else:
            control_series = control_df
        
        # Calculate effect size for each row in treatment_df and create a list of results
        effect_rows = []
        for _, row in treatment_df.iterrows():
            effect_row = calc_treatment_effect_for_row(row, control_series)
            effect_rows.append(effect_row)
        
        # Concatenate all rows to create the effect_size DataFrame
        if effect_rows:
            effect_size = pd.concat(effect_rows, axis=1).T.copy()
            
            # update treatment label
            if control_level_col == "treatment_level_t":
                effect_size['treatment_level_ph'] = group.name
                if group.name >= 1:
                    effect_size['treatment'] = 'temp_mv'
            elif control_level_col == "treatment_level_ph":
                effect_size['treatment_level_t'] = group.name
                if group.name >= 1:
                    effect_size['treatment'] = 'phtot_mv'
            
            return effect_size
        return None

    # process each group and append results
    results_ph = grouped_by_ph = df.groupby('treatment_level_ph').apply(
        process_group, control_level_col='treatment_level_t'
    )
    results_t = grouped_by_t = df.groupby('treatment_level_t').apply(
        process_group, control_level_col='treatment_level_ph'
    )
    # TODO: this doesn't add effects for where BOTH treatment levels change at once i.e. multivariate, the part which isn't in-level comparison
    
    
    
    
    # Filter out None values before returning
    results = []
    if not results_ph.empty:
        results.append(results_ph)
    if not results_t.empty:
        results.append(results_t)
    
    return results
    
    
    
    # results_df = []
    # control_df = df[df['treatment'] == "control"]
    # if control_df.empty:
    #     print(f"No control data found for this group (index {df.index} DOI {df['doi'].iloc[0]} species {df['species_types'].iloc[0]})")
    #     return None
    # # aggregate if necessary
    # control_row = aggregate_by_treatment_group(control_df) if len(control_df) > 1 else control_df.iloc[0]
    
    # for i, row in df.iterrows():
    #     if row['treatment'] == "control":
    #         continue
    #     # calculate effect size for each row in treatment_df
    #     effect_size = calc_treatment_effect_for_row(row, control_row)
        
    #     results_df.append(pd.DataFrame(effect_size).T)
    
    # return results_df


#     results_df = []
#     # for each treatment value in pH, calculate effect size varying temperature
#     grouped_by_ph = df.groupby('treatment_level_ph')
#     grouped_by_t = df.groupby('treatment_level_t')
    
#     def process_group(group, control_level_col):
#         control_level = min(group[control_level_col])
#         control_df = group[group[control_level_col] == control_level]
#         treatment_df = group[group[control_level_col] > control_level]
        
#         if treatment_df.empty:  # skip if there's no treatment data
#             return

#         # covered (better) before being passed
#         # # aggregate control data if n > 1 (more precise than just taking first row)
#         # control_data = aggregate_by_treatment_group(control_df) if len(control_df) > 1 else control_df.iloc[0]
#         # # Aggregate treatment data if all n=1
#         # if np.all(treatment_df.n == 1):
#         #     treatment_df = pd.DataFrame(aggregate_by_treatment_group(treatment_df)).T
            
            
#         # update treatment label
#         if control_level_col == "treatment_level_t":
#             treatment_df.loc[:, 'treatment'] = 'temp'
#         elif control_level_col == "treatment_level_ph":
#             treatment_df.loc[:, 'treatment'] = 'phtot'
            
#         # calculate effect size varying treatment condition
#         # First make sure we convert control_data DataFrame to Series if needed
#         control_series = control_df.iloc[0] if isinstance(control_df, pd.DataFrame) else control_df
#         # Calculate effect size for each row in treatment_df
#         effect_size = treatment_df.apply(lambda row: calc_treatment_effect_for_row(row, control_series), axis=1)
#         return effect_size
    
#     results_df.append(grouped_by_ph.apply(process_group, control_level_col='treatment_level_t', include_groups=False))
#     results_df.append(grouped_by_t.apply(process_group, control_level_col='treatment_level_ph', include_groups=False))
    
#     return pd.concat(results_df).reset_index(drop=True)


def aggregate_by_treatment_group(df: pd.DataFrame) -> pd.Series:
    """
    Aggregate a DataFrame by treatment group. Useful for when samples are individual datapoints, or multiple slightly-different controls are present.
    
    Args:
        df: DataFrame containing data for a specific treatment group
        
    Returns:
        pandas.Series: Series containing aggregated data
    """
    aggregation = df.agg({
        'calcification': ['mean', 'std'],
        'n': 'count'
    })
    control_row = df.iloc[0].copy()
    control_row['calcification'] = aggregation['calcification']['mean']
    control_row['calcification_sd'] = aggregation['calcification']['std']
    control_row['n'] = aggregation['n']['count']
    return control_row


def calc_treatment_effect_for_row(treatment_row: pd.Series, control_data: pd.Series) -> pd.Series:
    """
    Calculate the effect size (Hedges' g or relative calcification) and append additional columns for a treatment row.
    
    Args:
        treatment_row: Row containing treatment data
        control_data: Dictionary containing control group data
    
    Returns:
        pandas.Series: Row with calculated effect sizes and additional metadata
    """
    mu_t, sd_t, n_t = treatment_row['calcification'], treatment_row['calcification_sd'], treatment_row['n']
    mu_c, sd_c, n_c = control_data['calcification'], control_data['calcification_sd'], control_data['n']
    # standardised values
    s_mu_t, s_sd_t, _ = treatment_row['st_calcification'], treatment_row['st_calcification_sd'], treatment_row['n']
    s_mu_c, s_sd_c, _ = control_data['st_calcification'], control_data['st_calcification_sd'], control_data['n']
    t_in_c, ph_c = control_data['temp'], control_data['phtot']
    
    if np.isnan(mu_t) or np.isnan(mu_c) or np.isnan(sd_t) or np.isnan(sd_c):
        print(f"Missing data for effect size calculation. mu_t: {mu_t:.3f}, mu_c: {mu_c:.3f}, sd_t: {sd_t:.3f}, sd_c: {sd_c:.3f}, n_t: {n_t:.3f}, n_c: {n_c:.3f} at \n[index {treatment_row.name} DOI {treatment_row['doi']}]")
        print(treatment_row.doi)

    row_copy = treatment_row.copy() # create a copy to avoid SettingWithCopyWarning
    
    d_effect, d_var = calc_cohens_d(mu_c, mu_t, sd_c, sd_t, n_c, n_t)   # Cohen's d
    hg_effect, hg_var = calc_hedges_g(mu_c, mu_t, sd_c, sd_t, n_c, n_t) # Hedges' g
    
    # handle relative calcification (use raw value if already stated relative to baseline)
    rc_effect, rc_var = (mu_t, sd_t) if isinstance(treatment_row['calcification_unit'], str) and 'delta' in treatment_row['calcification_unit'] else calc_relative_rate(mu_c, mu_t, sd_c, sd_t, n_c, n_t)
    
    abs_effect, abs_var = calc_absolute_rate(mu_c, mu_t, sd_c, sd_t, n_c, n_t)  # absolute differences
    
    st_d_effect, st_d_var = calc_cohens_d(s_mu_c, s_mu_t, s_sd_c, s_sd_t, n_c, n_t) # standardised cohen's d
    st_hg_effect, st_hg_var = calc_hedges_g(s_mu_c, s_mu_t, s_sd_c, s_sd_t, n_c, n_t)   # standardised hedges' g
    
    # absolute differences between standardised calcification
    st_abs_effect, st_abs_var = calc_absolute_rate(s_mu_c, s_mu_t, s_sd_c, s_sd_t, n_c, n_t)
    # relative differences between standardised calcification
    st_rc_effect, st_rc_var = (s_mu_t, s_sd_t) if isinstance(treatment_row['st_calcification_unit'], str) and 'delta' in treatment_row['st_calcification_unit'] else calc_relative_rate(s_mu_c, s_mu_t, s_sd_c, s_sd_t, n_c, n_t)
    
    # assign effect sizes
    row_copy.update({
        'cohens_d': d_effect, 'cohens_d_var': d_var,
        'hedges_g': hg_effect, 'hedges_g_var': hg_var,
        'relative_calcification': rc_effect, 'relative_calcification_var': rc_var,
        'absolute_calcification': abs_effect, 'absolute_calcification_var': abs_var,
        'st_relative_calcification': st_rc_effect, 'st_relative_calcification_var': st_rc_var,
        'st_absolute_calcification': st_abs_effect, 'st_absolute_calcification_var': st_abs_var,
    })
    
    # calculate metadata
    row_copy['control_temp'] = control_data['temp']
    row_copy['treatment_temp'] = treatment_row['temp']
    row_copy['delta_t'] = row_copy['temp'] - t_in_c
    row_copy['control_phtot'] = control_data['phtot']
    row_copy['treatment_phtot'] = treatment_row['phtot']
    row_copy['delta_ph'] = row_copy['phtot'] - ph_c
    row_copy['treatment_val'] = row_copy['temp'] if row_copy['treatment'] == 'temp' else row_copy['phtot']
    row_copy['control_calcification'] = mu_c
    row_copy['control_calcification_sd'] = sd_c
    row_copy['treatment_calcification'] = mu_t
    row_copy['treatment_calcification_sd'] = sd_t
    row_copy['st_control_calcification'] = s_mu_c
    row_copy['st_control_calcification_sd'] = s_sd_c
    row_copy['st_treatment_calcification'] = s_mu_t
    row_copy['st_treatment_calcification_sd'] = s_sd_t
    row_copy['treatment_n'] = n_t
    row_copy['control_n'] = n_c
    
    return row_copy


def calculate_effect_sizes_end_to_end(raw_data_fp, data_sheet_name: str, climatology_data_fp: str=None, selection_dict: dict={'include': 'yes'}):
    """
    # TODO: replace with new processing functions?
    Calculate effect sizes from raw data and align with climatology data.
    
    Args:
        raw_data_fp (str or Path): Path to raw data file
        data_sheet_name (str): Name of the sheet containing data
        climatology_data_fp (str or Path): Path to climatology data file
        selection_dict (dict): Dictionary of selection criteria
        
    Returns:
        pd.DataFrame: DataFrame with calculated effect sizes
    """
    # load and process carbonate chemistry data
    carbonate_df = processing.populate_carbonate_chemistry(raw_data_fp, data_sheet_name, selection_dict=selection_dict)
    
    # prepare for alignment with climatology by uniquifying DOIs
    # carbonate_df = utils.uniquify_multilocation_study_dois(carbonate_df)
    print(f"\nShape of dataframe with all rows marked for inclusion: {carbonate_df.shape}")
    
    # save selected columns of carbonate dataframe to file for reference
    carbonate_save_fields = file_ops.read_yaml(config.resources_dir / 'mapping.yaml')["carbonate_save_columns"]
    carbonate_df[carbonate_save_fields].to_csv(config.tmp_data_dir / 'carbonate_chemistry.csv', index=False)

    # assign treatment groups
    carbonate_df_tgs = processing.assign_treatment_groups_multilevel(carbonate_df)

    carbonate_df_tgs_no_ones = processing.aggregate_treatments_with_individual_samples(carbonate_df_tgs)
    # calculate effect size
    print(f"\nCalculating effect sizes...")
    effects_df = calculate_effect_for_df(carbonate_df_tgs_no_ones).reset_index(drop=True)
    
    # save results
    save_cols = file_ops.read_yaml(config.resources_dir / "mapping.yaml")["save_cols"]
    effects_df['year'] = pd.to_datetime(effects_df['year']).dt.strftime('%Y')  # cast year from pd.timestamp to integer
    # Check for missing columns in save_cols
    missing_columns = [col for col in save_cols if col not in effects_df.columns]
    if missing_columns:
        print(f"\nWARNING: The following columns in save_cols are not in effects_df: {missing_columns}")
        # Filter save_cols to only include columns that exist in effects_df
        available_save_cols = [col for col in save_cols if col in effects_df.columns]
        effects_df[available_save_cols].to_csv(config.tmp_data_dir / f"effect_sizes.csv", index=False)
    else:
        effects_df[save_cols].to_csv(config.tmp_data_dir / f"effect_sizes.csv", index=False)

    print(f"\nShape of dataframe with effect sizes: {effects_df.shape}")
    
    return effects_df


### curve fitting
def fit_curve(df: pd.DataFrame, variable: str, effect_type: str, order: int) -> sm.regression.linear_model.RegressionResultsWrapper:
    """
    Fit a polynomial curve to the data.

    Parameters:
    - df (pd.DataFrame): The dataframe containing the data.
    - variable (str): The independent variable.
    - effect_type (str): The dependent variable.
    - order (int): The order of the polynomial to fit.

    Returns:
    - model: The fitted regression model.
    """
    # Remove NaNs
    df = df[df[variable].notna() & df[effect_type].notna()]

    # Create polynomial features
    X = np.vander(df[variable], N=order + 1, increasing=True)
    
    # Fit the model
    model = sm.OLS(df[effect_type], X).fit()
    return model


def predict_curve(model, x, alpha=0.05):
    """
    Predict values using the fitted model with confidence intervals.

    Parameters:
    - model: The fitted regression model.
    - x (np.ndarray): The independent variable values.
    - alpha (float): Significance level for confidence intervals (default: 0.05 for 95% CI)

    Returns:
    - tuple: (predicted values, lower confidence bound, upper confidence bound)
    """
    X = np.vander(x, N=model.params.shape[0], increasing=True)
    prediction = model.get_prediction(X)
    
    # Get the predicted values
    predicted = prediction.predicted_mean
    
    # Get the confidence intervals
    conf_int = prediction.conf_int(alpha=alpha)
    lower = conf_int[:, 0]
    upper = conf_int[:, 1]
    
    return predicted, lower, upper


### Meta-analysis
def generate_formula(effect_type: str, treatment: str=None, variables: list[str] = None, include_intercept: bool = False) -> str:
    if variables is None:
        variables = []
        
    # Add treatment-specific variable if needed
    if treatment:
        if treatment == "phtot":
            variables.append("delta_ph")
        elif treatment == "temp":
            variables.append("delta_t")
        else:
            raise ValueError(f"Unknown treatment: {treatment}")
    
    # Remove duplicates and process variables
    variables = list(set(variables))
    variable_mapping = file_ops.read_yaml(config.resources_dir / 'mapping.yaml')["meta_model_factor_variables"]
    
    # Split into factor and regular variables
    factor_vars = [f"factor({v})" for v in variables if v in variable_mapping]
    other_vars = [v for v in variables if v not in variable_mapping]
    
    # Combine all parts into formula string
    parts = other_vars + factor_vars
    formula = f"{effect_type} ~ {' + '.join(parts)}"
    
    # Remove intercept if specified
    return formula + " - 1" if not include_intercept else formula


def get_formula_components(formula):
    """
    Extracts the response variable, predictors, and intercept flag from a formula string.
    """
    formula_comps = {}
    response_part, predictor_part = formula.split('~')
    formula_comps['response'] = response_part.strip()

    # Clean and normalize predictor part
    predictor_part = predictor_part.replace(' ', '').replace('-1', '+intercept_off')  # temporarily mark intercept removal
    predictors_raw = predictor_part.split('+')

    # Determine intercept
    intercept = 'intercept_off' not in predictors_raw
    predictors_raw = [p for p in predictors_raw if p != 'intercept_off']
    formula_comps['predictors'] = [p for p in predictors_raw if p]  # remove empty strings
    formula_comps['intercept'] = intercept

    return formula_comps


def preprocess_df_for_meta_model(df, effect_type: str = 'hedges_g', effect_type_var=None, treatment=None, necessary_vars: list[str] = None, formula: str=None) -> pd.DataFrame:
    # TODO: get necessary variables more dynamically (probably via a mapping including factor)
    data = df.copy()
    
    effect_type_var = effect_type_var or f"{effect_type}_var"
    
    ### specify model
    if not formula:
        formula = generate_formula(effect_type, treatment, variables=necessary_vars)

    # select only rows relevant to treatment
    if treatment:
        data = data[data["treatment"].astype(str).str.contains(treatment, na=False)]
        # data = data[data['treatment'] == treatment]
        
    n_investigation = len(data)
    # remove nans for subset effect_type
    required_columns = [effect_type, effect_type_var, 'original_doi', 'ID'] + (necessary_vars or [])
    data = data.dropna(subset=required_columns)
    
    # Ensure all numeric columns are explicitly converted to correct type
    for col in data.columns:
        data[col] = processing.safe_to_numeric(data[col])
    # convert any object columns to strings
    for col in data.select_dtypes(include=['object']).columns:
        data[col] = data[col].astype(str)
        
        
    n_nans = n_investigation - len(data)
    
    ### summarise processing
    print("\n----- PROCESSING SUMMARY -----")
    print('Treatment: ', treatment)
    print('Total samples in input data: ', len(df))
    print('Total samples of relevant investigation: ', n_investigation)
    print('Dropped due to NaN values in required columns:', n_nans)
    print(f'Final sample count: {len(data)} ({n_nans+(len(df)-n_investigation)} rows dropped)')
    
    # remove outliers
    nparams = len(formula.split('+'))
    data, outliers = remove_cooks_outliers(data, effect_type=effect_type, nparams=nparams)
    
    return formula, data


def run_metafor_mv(
    df: pd.DataFrame,
    effect_type: str = 'hedges_g',
    effect_type_var: str = None,
    treatment: str = None,
    necessary_vars: list[str] = None,
    formula: str = None,
) -> tuple[ro.vectors.DataFrame, ro.vectors.DataFrame, pd.DataFrame]:
    """
    Run the metafor model on the given dataframe.
    
    Args:
        df (pd.DataFrame): The dataframe to run the model on.
        effect_type (str): The type of effect to use.
        treatment (str): The treatment to use.
        necessary_vars (list[str]): The necessary variables to use.
        formula (str): The formula to use.
    
    Returns:
        ro.vectors.DataFrame: The results of the metafor model.
    """
    effect_type_var = effect_type_var or f"{effect_type}_var"
    # preprocess the dataframe
    formula, df = preprocess_df_for_meta_model(df, effect_type, effect_type_var, treatment, necessary_vars, formula)
    print(f'Using formula {formula}')    
    # activate R conversion
    ro.pandas2ri.activate()
    
    all_necessary_vars = ['original_doi', 'ID'] + (necessary_vars or []) + [effect_type, effect_type_var] + ['delta_ph' if treatment == 'phtot' else 'delta_t']
    # ensure original_doi is string type to avoid conversion issues
    df = df.copy()
    df['original_doi'] = df['original_doi'].astype(str)
    
    df_subset = df[all_necessary_vars]
    df_r = ro.pandas2ri.py2rpy(df_subset)
    
    # run the metafor model
    print('\nRunning metafor model...')
    model = metafor.rma_mv(
        yi=ro.FloatVector(df_r.rx2(effect_type)),
        V=ro.FloatVector(df_r.rx2(effect_type_var)),
        data=df_r,
        mods=ro.Formula(formula),
        random=ro.Formula("~ 1 | original_doi/ID")
    )
    print('Model fitting complete.')
    return model, base.summary(model), formula, df


def run_parallel_dredge(df, global_formula=None, effect_type='hedges_g', x_var='temp', n_cores=16):
    """TODO: get actually working in parallel
    Runs a parallel dredge analysis using MuMIn in R.

    Parameters:
        df (rpy2.robjects.vectors.DataFrame): The dataframe in R format.
        effect_type (str): The effect type (e.g., 'hedges_g').
        x_var (str): The independent variable (e.g., 'delta_t').
        n_cores (int): Number of cores for parallel processing.

    Returns:
        pandas.DataFrame: The dredge result converted to a pandas DataFrame.
    """
    os.environ['LC_ALL'] = 'en_US.UTF-8'  # Set locale to UTF-8

    # Assign variables to R environment
    ro.r.assign("df_r", ro.pandas2ri.py2rpy(df))    # convert to R dataframe
    df_r = ro.pandas2ri.py2rpy(df)
    ro.r.assign("effect_col", effect_type)
    ro.r.assign("var_col", f"{effect_type}_var")
    ro.r.assign("original_doi", df_r.rx2("original_doi"))
    ro.r.assign("ID", df_r.rx2("ID"))
    
    # set up formula
    global_formula = f"{effect_type} ~ {x_var} - 1" if global_formula is None else global_formula
    print(global_formula)
    ro.r.assign("global_formula", ro.Formula(global_formula))

    # Run the R code for parallel dredge
    ro.r(f"""
    # Set up for MuMIn
    eval(metafor:::.MuMIn)

    global_model <- rma.mv(
        yi = df_r[[effect_col]],
        V = df_r[[var_col]], 
        mods = global_formula,
        random = ~ 1 | original_doi/ID,
        data=df_r,
    )
    # Create cluster
    clu <- parallel::makeCluster({n_cores})
    # Load packages on each worker
    parallel::clusterEvalQ(clu, {{
      library(metafor)
      library(MuMIn)
    }})

    # Export required variables to each worker
    parallel::clusterExport(clu, varlist = c("df_r", "effect_col", "var_col", "global_formula", "original_doi", "ID"))

    dredge_result <- MuMIn::dredge(global_model)

    # Stop the cluster
    parallel::stopCluster(clu)
    """)

    # Retrieve the dredge result and convert to pandas DataFrame
    dredge_result = ro.r("dredge_result")
    # Convert to pandas DataFrame
    ro.pandas2ri.activate()
    df = ro.pandas2ri.rpy2py(dredge_result)
    # assign any values of '-2147483648' to NaN (R's placeholder for NA in string columns)
    return df.replace(-2147483648, np.nan)


def generate_location_specific_predictions(model, df: pd.DataFrame, scenario_var: str = "sst"):
    # Get constant terms from the model matrix (excluding intercept/first column)
    model_matrix = ro.r('model.matrix')(model)
    const_terms = np.array(model_matrix)[:, 1:].mean(axis=0)
    const_terms_list = const_terms.tolist()

    df = df.sort_index()     # Sort the index to avoid PerformanceWarning about lexsort depth
    locations = df.index.unique()
    prediction_rows = []  # to hold newmods inputs
    metadata_rows = []    # to track what each row corresponds to

    for location in tqdm(locations, desc=f"Generating batched predictions for {scenario_var}"):
        location_df = df.loc[location]
        scenarios = location_df['scenario'].unique()

        for scenario in scenarios:
            scenario_df = location_df[location_df['scenario'] == scenario]
            time_frames = [1995] + list(scenario_df.time_frame.unique())

            for time_frame in time_frames:
                if time_frame == 1995:
                    base = scenario_df[f'mean_historical_{scenario_var}_30y_ensemble'].mean()
                    mean_scenario = base - base # think this causes issues?
                    p10_scenario = scenario_df[f'percentile_10_historical_{scenario_var}_30y_ensemble'].mean() - base
                    p90_scenario = scenario_df[f'percentile_90_historical_{scenario_var}_30y_ensemble'].mean() - base
                else:
                    time_scenario_df = scenario_df[scenario_df['time_frame'] == time_frame]
                    mean_scenario = time_scenario_df[f'mean_{scenario_var}_20y_anomaly_ensemble'].mean()
                    p10_scenario = time_scenario_df[f'{scenario_var}_percentile_10_anomaly_ensemble'].mean()
                    p90_scenario = time_scenario_df[f'{scenario_var}_percentile_90_anomaly_ensemble'].mean()
                    # Generate predictions for mean, p10, and p90 scenarios
                for percentile, anomaly in [('mean', mean_scenario), ('p10', p10_scenario), ('p90', p90_scenario)]:
                    prediction_rows.append([anomaly] + const_terms_list)
                    metadata_rows.append({
                        'doi': location[0],
                        'location': location[1],
                        'longitude': location[2],
                        'latitude': location[3],
                        'scenario_var': scenario_var,
                        'scenario': scenario,
                        'time_frame': time_frame,
                        'anomaly_value': anomaly,
                        'percentile': percentile
                    })

    # Convert to R matrix (newmods)
    newmods_np = np.array(prediction_rows, dtype=float)
    newmods_r = ro.r.matrix(ro.FloatVector(newmods_np.flatten()), nrow=newmods_np.shape[0], byrow=True)

    # Predict all at once in R
    predictions_r = ro.r('predict')(model, newmods=newmods_r, digits=2)
    predicted_vals = list(predictions_r)

    # Combine metadata and predictions
    for i, val in enumerate(predicted_vals[0]): # for now, just taking mean predictions (ignoring ci, pi)
        metadata_rows[i]['predicted_effect_size'] = val

    return metadata_rows


### DEPRECATED
# def compute_heds
    # def calculate_effect_size(df1_sample, df2_sample, var, group1, group2):
    #     other_var = 'temp' if var == 'phtot' else 'phtot'
    #     if np.isclose(df1_sample[other_var], df2_sample[other_var], atol=0.1):
    #         if (var == 'phtot' and df1_sample[var] < df2_sample[var]) or (var == 'temp' and df1_sample[var] > df2_sample[var]):
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


# def compute_hedges_g(df, vars_to_compare=['temp', 'phtot', 'irr']):
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
#                     other_var = 'temp' if var == 'phtot' else 'phtot'
#                     if np.isclose(df1_sample[other_var], df2_sample[other_var], atol=0.1):  # if agree within 0.1, likely control variable
#                         # ensure that df1_sample is control group (smaller value for t_in, larger value for phtot). # TODO: allow switching for deltavar relative to climatology
#                         if (var == 'phtot' and df1_sample[var] < df2_sample[var]) or (var == 'temp' and df1_sample[var] > df2_sample[var]):
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


# def cluster_treatments(df, vars_to_cluster):
#     """Cluster treatments based on independent variables and species types."""
    
#     import warnings
#     warnings.filterwarnings("ignore", category=UserWarning, module='sklearn.cluster._kmeans')  # Suppress convergence warnings

#     df = df.copy()  # Avoid modifying the original dataframe
#     # Ensure 'treatment_group' column exists
#     df['treatment_group'] = np.nan

#     # Cluster separately for each species type
#     for species, group_df in df.groupby('species_types'):
#         treatment_data = group_df[vars_to_cluster].dropna(axis=1)  # Only keep non-missing variables

#         if treatment_data.shape[0] < 2:  # Skip clustering if there's only one sample
#             print(f"Skipping {species}: Not enough samples for clustering: {len(group_df)}")
#             continue

#         try:
#             optimal_k, _ = optimal_kmeans(treatment_data)  # Determine optimal clusters
#             kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
#             df.loc[group_df.index, 'treatment_group'] = kmeans.fit_predict(treatment_data)
#         except ValueError:
#             print(f"Error: Could not cluster {species}, df of length {len(group_df)}")
#             df.loc[group_df.index, 'treatment_group'] = np.nan

#     return df
    
    
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



# def group_irradiance(df, irr_col='irr', atol=30):
#     """
#     Assigns an 'irr_group' to values that are within a relative tolerance.
    
#     Args:
#         df (pd.DataFrame): Input dataframe with an 'irr' column.
#         irr_col (str): Column name for irradiance values.
#         rtol (float): Relative tolerance (e.g., 0.10 for 10%).
        
#     Returns:
#         pd.DataFrame: Dataframe with new 'irr_group' column.
#     """
#     df = df.copy().sort_values(by=irr_col)  # copy to prevent modification of original, sort for efficiency
#     irr_groups = np.zeros(len(df))  # initialize array for group assignments
#     group_id = 0
#     prev_irr = None

#     for i, irr in enumerate(df[irr_col]):
#         if np.isnan(irr):   # handling grouping where irr is NaN
#             irr_groups[i] = -1  # Use a distinct value for NaN irradiance
#         elif prev_irr is None:
#             # First non-NaN value starts group 0
#             group_id = 0
#             irr_groups[i] = group_id
#         elif np.abs(irr - prev_irr) > atol:
#             # Outside tolerance, create new group
#             group_id += 1
#             irr_groups[i] = group_id
#         else:
#             # Within tolerance, use current group
#             irr_groups[i] = group_id
            
#         # Update prev_irr only for non-NaN values
#         if not np.isnan(irr):
#             prev_irr = irr

#     # Add irr_group column to dataframe
#     df['irr_group'] = irr_groups
#     return df



### Previously part of process_group
    # loi = ['phtot', 'temp', 'calcification', 'calcification_sd', 'n']
    # control_df = species_df[species_df['treatment_group'] == 'cTcP']    # extract control data
    # if control_df.empty:
    #     print(f"Control group not found for {species_df['doi'].iloc[0]}, for \n{species_df[['species_types']+loi].iloc[0]}")
    #     return None
    # if len(control_df) > 1: # if multiple possible controls, aggregate
    #     control_row = aggregate_by_treatment_group(control_df)
    # else:
    #     control_row = control_df.iloc[0]
    
    # # append control_df to result_dfs
    # if not pd.isna(control_row).all():
    #     control_row_df = pd.DataFrame(control_row).T

    #     # Explicitly check for all-NA columns and remove them
    #     if not control_row_df.dropna(how="all", axis=1).empty:
    #         result_dfs.append(control_row_df)
    # # process each treatment group for univariate treatments
    # for treatment_group, treatment_df in species_df.groupby('treatment_group'):
    #     if treatment_group in ['uncertain', 'cTcP']:
    #         # doesn't make sense to process
    #         continue
        
    #     # in the case that all n == 1 (individual datapoints), aggregate by treatment_group
    #     if np.all(treatment_df.n == 1):
    #         treatment_df = pd.DataFrame(aggregate_by_treatment_group(treatment_df)).T # hedges_g_for_row requires df input

            
    #     if treatment_group == 'tTtP':
    #         ### option 1: calculate each relative to single control (simple) – the max pH and min T for species
    #         # for each row in treatment_Group, calculate hedges_g_for_row and append to result_dfs
    #         if np.any(treatment_df.n == 1):
    #             print('will need to cluster by treatment level')
    #         out = pd.DataFrame(treatment_df.apply(
    #             lambda row: hedges_g_for_row(row, control_row),
    #             axis=1
    #         ))
    #         result_dfs.append(out) if not out.empty else None
    #         ### option 2: for each treatment, calculate each relative to all other treatments (complex)
            
            
    #         # e.g. for pHs at higher T, determine highest pH and calculate effects wrt that
    #         # and for Ts, determine pH closest at a different T and calculate effect wrt that
    #         continue
    
    #     # apply hedges_g calculation to each row
    #     processed_df = treatment_df.apply(
    #         lambda row: hedges_g_for_row(row, control_row),
    #         axis=1
    #     )
    #     result_dfs.append(processed_df) if not processed_df.empty else None
    
    # if result_dfs:
    #     # Drop all-NA columns from each DataFrame before concatenation
    #     filtered_dfs = [df.dropna(how="all", axis=1) for df in result_dfs]

    #     # Ensure at least one DataFrame is non-empty before concatenation
    #     filtered_dfs = [df for df in filtered_dfs if not df.empty]

    #     return pd.concat(filtered_dfs, axis=0) if filtered_dfs else None
    # else:
    #     return None
    
    
    # def process_group(species_df: pd.DataFrame, effect_type: str = 'hedges_g') -> pd.DataFrame:
#     """
#     Process a group of species data to calculate Hedges' g.
    
#     Args:
#         species_df: DataFrame containing data for a specific species
        
#     Returns:
#         pandas.DataFrame: DataFrame with Hedges' g calculations
#     """
#     # TODO: implement simple method (single control): see deprecated code
#     result_dfs = []
#     result_dfs.extend(process_group_multivar(species_df, effect_type=effect_type))
        
#     # suppress futurewarning
#     import warnings
#     warnings.simplefilter(action='ignore', category=FutureWarning)
    
#     if result_dfs:  # TODO: fix future warning without dropping all-nan columns
#         # Identify all column names across all DataFrames
#         all_columns = set().union(*(df.columns for df in result_dfs))
        
#         # Ensure all DataFrames contain the same columns with consistent dtypes
#         for i, df in enumerate(result_dfs):
#             missing_cols = all_columns - set(df.columns)
#             for col in missing_cols:
#                 df[col] = pd.NA  # Retain columns but explicitly mark as NA
#             result_dfs[i] = df.astype({col: "object" for col in missing_cols})  # Set dtype to avoid dtype inference issues
    
#     # Concatenate DataFrames without dropping NaN-only columns
#         result_df = pd.concat(result_dfs, axis=0, ignore_index=True)
#         # # drop all-NA columns from each DataFrame before concatenation (avoids future warning)
#         # filtered_dfs = [df.dropna(how="all", axis=1) for df in result_dfs]
#         # # ensure at least one DataFrame is non-empty before concatenation
#         # filtered_dfs = [df for df in filtered_dfs if not df.empty]
#         return result_df
#         # return pd.concat(filtered_dfs, axis=0) if filtered_dfs else None
#     else:
#         return None


# def calc_cohens_d(mu1: float, mu2: float, sd_pooled: float) -> float:
#     """Calculate Hedges G metric: https://www.itl.nist.gov/div898/software/dataplot/refman2/auxillar/hedgeg.htm
    
#     Args:
#         mu1 (float): mean of group 1
#         mu2 (float): mean of group 2
#     2 (float): mean of group 2
#         sd_pooled (float): pooled standard deviation of both groups
        
#     Returns:
#         float: Hedges G metric
#     """
#     return (mu1 - mu2) / sd_pooled


# def calc_sd_pooled(n1: int, n2: int, sd1: float, sd2: float) -> float:
#     """Calculate pooled standard deviation: https://www.itl.nist.gov/div898/software/dataplot/refman2/auxillar/hedgeg.htm
    
#     Args:
#         n1 (int): number of samples in group 1
#         n2 (int): number of samples in group 2
#         sd1 (float): standard deviation of group 1
#         sd2 (float): standard deviation of group 2
        
#     Returns:
#         float: pooled standard deviation
#     """
#     return np.sqrt(((n1 - 1) * sd1 ** 2 + (n2 - 1) * sd2 ** 2) / (n1 + n2 - 2))


# def calc_sd_pooled(n1: int, n2: int, sd1: float, sd2: float) -> float:
#     """Calculate pooled standard deviation: https://www.itl.nist.gov/div898/software/dataplot/refman2/auxillar/hedgeg.htm
#     NB the simpler version (as used by Ben). Doesn't weight by sample size (although this slightly captured by sd) 
#     Args:
#         n1 (int): number of samples in group 1
#         n2 (int): number of samples in group 2
#         sd1 (float): standard deviation of group 1
#         sd2 (float): standard deviation of group 2
        
#     Returns:
#         float: pooled standard deviation
#     """
#     return np.sqrt((sd1 ** 2 + sd2 ** 2) / 2)


# def process_group_multivar(df: pd.DataFrame) -> pd.DataFrame:
#     """
#     Process a group of species data to calculate effect size.
    
#     Args:
#         df: DataFrame containing data for a specific species
    
#     Returns:
#         pandas.DataFrame: DataFrame with effect size calculations
#     """
    
#     results_df = []
#     # for each treatment value in pH, calculate effect size varying temperature
#     grouped_by_ph = df.groupby('treatment_level_ph')
#     grouped_by_t = df.groupby('treatment_level_t')
    
#     def process_group(group, control_level_col):
#         control_level = min(group[control_level_col])
#         control_df = group[group[control_level_col] == control_level]
#         treatment_df = group[group[control_level_col] > control_level]
        
#         if treatment_df.empty:  # skip if there's no treatment data
#             return

#         # covered (better) before being passed
#         # # aggregate control data if n > 1 (more precise than just taking first row)
#         # control_data = aggregate_by_treatment_group(control_df) if len(control_df) > 1 else control_df.iloc[0]
#         # # Aggregate treatment data if all n=1
#         # if np.all(treatment_df.n == 1):
#         #     treatment_df = pd.DataFrame(aggregate_by_treatment_group(treatment_df)).T
            
            
#         # update treatment label
#         if control_level_col == "treatment_level_t":
#             treatment_df.loc[:, 'treatment'] = 'temp'
#         elif control_level_col == "treatment_level_ph":
#             treatment_df.loc[:, 'treatment'] = 'phtot'
            
#         # calculate effect size varying treatment condition
#         # First make sure we convert control_data DataFrame to Series if needed
#         control_series = control_df.iloc[0] if isinstance(control_df, pd.DataFrame) else control_df
#         # Calculate effect size for each row in treatment_df
#         effect_size = treatment_df.apply(lambda row: calc_treatment_effect_for_row(row, control_series), axis=1)
#         return effect_size
    
#     results_df.append(grouped_by_ph.apply(process_group, control_level_col='treatment_level_t', include_groups=False))
#     results_df.append(grouped_by_t.apply(process_group, control_level_col='treatment_level_ph', include_groups=False))
    
#     return pd.concat(results_df).reset_index(drop=True)


# # Import required R packages
# from rpy2.robjects.packages import importr
# from rpy2.robjects import pandas2ri
# import rpy2.robjects as ro

# # Activate pandas2ri for DataFrame conversion
# pandas2ri.activate()
# r = ro.r

# # Import metafor package
# metafor = importr('metafor')

# # temp_df['esid'] = temp_df.groupby('doi').cumcount() + 1

# # Convert your DataFrame to an R DataFrame
# r_temp_df = pandas2ri.py2rpy(temp_df)
# ro.r.assign('dat', r_temp_df)
# ro.r.assign('res', basic_temp_model)


# # Now you can use the updated 'dat' and 'V' in your forest plot or further analysis
# ro.r('''
# create_forest_plot <- function(res, dat) {
#     par(tck=-.01, mgp=c(1,0.01,0), mar=c(2,4,2,2))
    
#     # Ensure 'original_doi' is numeric to avoid non-numeric argument errors
#     dat$original_doi <- as.numeric(as.factor(dat$original_doi))
#     # sort by original_doi
#     dat <- dat[order(dat$original_doi), ]
    
#     dd <- c(0, diff(dat$original_doi))
#     print(dd)
#     rows <- (1:res$k)*5 + cumsum(dd)  # Multiply by 2 to increase vertical spacing
#     forest(res, 
#     # rows=rows, 
#     # ylim=c(-500, max(rows)+500), xlim=c(-5,7),
#                  cex=0.4, efac=c(0,1), mlab="Pooled Estimate")
    
#     # abline(h = rows[c(1,diff(rows)) == 2] - 1, lty="dotted")
# }
# ''')
# from rpy2.robjects.packages import importr

# grdevices = importr('grDevices')

# # To execute the forest plot function
# ro.r('create_forest_plot(res, dat)')
# # save the plot

# # If you want to save the plot
# r('pdf("forest_plot.pdf", width=12, height=10)')
# r('create_forest_plot(res, dat)')
# r('dev.off()')

# # grdevices.png(str(config.fig_dir / 'forest_plot.png'), width=800, height=600)
# # grdevices.dev_off()
# # # view plot in python
# # from matplotlib import pyplot as plt
# # import matplotlib.image as mpimg
# # img = mpimg.imread('forest_plot.png')
# # plt.imshow(img)