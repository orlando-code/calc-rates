import xarray as xa
import numpy as np
from tqdm.auto import tqdm

# custom
import cbsyst.helpers as cbh

def calc_hedges_g(mu1: float, mu2: float, sd_pooled: float) -> float:
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