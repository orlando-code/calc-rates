import itertools
import string
from pathlib import Path

import cbsyst.helpers as cbh
import numpy as np
import pandas as pd
import xarray as xa
from PIL import Image
from tqdm.auto import tqdm


### cbsyst sensitivity analysis
def create_st_ft_sensitivity_array(
    param_combinations: list, pertubation_percentage: float, resolution: int = 20
) -> xa.DataArray:
    # check if more than three parameters are passed
    if len(param_combinations[0]) != 3:
        raise ValueError(
            "param_combinations should be a list of tuples containing salinity, temperature, and pH_NBS values"
        )

    results_dict = {}
    for Sal, Temp, pH_NBS in tqdm(param_combinations):
        ST_base = cbh.calc_ST(Sal)
        FT_base = cbh.calc_FT(Sal)

        # define perturbation ranges for ST and FT
        ST_values = np.linspace(
            ST_base - pertubation_percentage / 100 * ST_base,
            ST_base + pertubation_percentage / 100 * ST_base,
            20,
        )
        FT_values = np.linspace(
            FT_base - pertubation_percentage / 100 * FT_base,
            FT_base + pertubation_percentage / 100 * FT_base,
            20,
        )

        # initialize empty array for pH_total values
        pH_grid = np.zeros((20, 20))

        for i, ST in enumerate(ST_values):
            for j, FT in enumerate(FT_values):
                pH_Total = cbh.pH_scale_converter(
                    pH=pH_NBS, scale="NBS", ST=ST, FT=FT, Temp=Temp, Sal=Sal
                ).get("pHtot", None)
                pH_grid[i, j] = pH_Total
        results_dict[(Sal, Temp, pH_NBS)] = pH_grid.copy()

    # extract unique values for each dimension
    Sal_values = sorted(set(k[0] for k in results_dict.keys()))
    Temp_values = sorted(set(k[1] for k in results_dict.keys()))
    pH_NBS_values = sorted(set(k[2] for k in results_dict.keys()))
    # store arrays with dimensions metadata
    data_array = np.empty(
        (
            len(Sal_values),
            len(Temp_values),
            len(pH_NBS_values),
            len(ST_values),
            len(FT_values),
        )
    )
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
        name="pH_Total",
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


def convert_png_to_jpg(directory: str) -> None:
    """
    Convert all PNG files in a directory to JPG format.

    Args:
        directory (str): The path to the directory containing the PNG files.
    """
    # fetch all current png files
    png_files = Path(directory).glob("*.png")

    for png_file in png_files:
        with Image.open(png_file) as img:  # open file
            # convert to RGB (PNG can have transparency)
            rgb_img = img.convert("RGB")

            # create a new filename by replacing .png with .jpg
            jpg_file = png_file.with_suffix(".jpg")

            # save image as JPG
            rgb_img.save(jpg_file, "JPEG")
            print(f"Converted {png_file} to {jpg_file}")


def uniquify_repeated_values(vals: list, uniquify_str: str = "LOC") -> list:
    """
    Append a unique suffix to repeated values in a list.

    Parameters:
        vals (list): List of values.

    Returns:
        list: List of values with unique suffixes.
    """

    def zip_letters(letters: list[str]) -> list[str]:
        """Zip a list of strings with uppercase letters."""
        al = string.ascii_uppercase
        return (
            [f"-{uniquify_str}-".join(i) for i in zip(letters, al)]
            if len(letters) > 1
            else letters
        )

    return [j for _, i in itertools.groupby(vals) for j in zip_letters(list(i))]


def safe_to_numeric(col):
    """Convert column to numeric if possible, otherwise return as is."""
    try:
        return pd.to_numeric(col)
    except (ValueError, TypeError):
        return col  # return original column if conversion fails
