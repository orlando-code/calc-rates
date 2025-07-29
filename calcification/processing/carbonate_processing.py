### calculate cargonate chemistry from experimental data
import logging

import cbsyst as cb
import cbsyst.helpers as cbh
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from calcification.processing import cleaning
from calcification.utils import config, file_ops

logger = logging.getLogger(__name__)


### carbonate chemistry
def populate_carbonate_chemistry(
    fp: str, sheet_name: str = "all_data", selection_dict: dict = None
) -> pd.DataFrame:
    """Populate carbonate chemistry parameters for a dataset.
    Args:
        fp (str): Path to Excel file containing extracted experimental data.
        sheet_name (str): Sheet name in Excel file.
        selection_dict (dict): Optional dict for row selection.

    Returns:
        pd.DataFrame: DataFrame with carbonate chemistry parameters.
    """
    if selection_dict is None:
        selection_dict = {"include": "yes"}
    df = cleaning.process_raw_data(
        pd.read_excel(fp, sheet_name=sheet_name),
        require_results=False,
        selection_dict=selection_dict,
    )
    logger.info("Loading measured values...")
    measured_df = file_ops.get_highlighted(fp, sheet_name=sheet_name)
    measured_df = cleaning.preprocess_df(measured_df, selection_dict=selection_dict)

    measured_df = _convert_ph_scales(measured_df)
    measured_df = _calculate_missing_phtot(measured_df)

    carb_metadata = file_ops.read_yaml(config.resources_dir / "mapping.yaml")
    carb_chem_cols = carb_metadata["carbonate_chemistry_cols"]
    out_values = carb_metadata["carbonate_chemistry_params"]
    carb_df = measured_df[carb_chem_cols].copy()

    tqdm.pandas(desc="Calculating carbonate chemistry")
    carb_df.loc[:, out_values] = carb_df.progress_apply(
        lambda row: pd.Series(calculate_carb_chem(row, out_values)), axis=1
    )
    return df.combine_first(carb_df)


def _convert_ph_scales(df: pd.DataFrame) -> pd.DataFrame:
    """Convert pHnbs to pHtot where needed."""
    mask = df["phtot"].isna() & df["phnbs"].notna() & df["temp"].notna()
    df.loc[mask, "phtot"] = df.loc[mask].apply(
        lambda row: cbh.pH_scale_converter(
            pH=row["phnbs"],
            scale="NBS",
            Temp=row["temp"],
            Sal=row["sal"] if pd.notna(row["sal"]) else 35,
        ).get("pHtot", None),
        axis=1,
    )
    return df


def _calculate_missing_phtot(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate pHtot from DIC/TA or pCO2/TA if missing."""
    mask_dic = (
        df["phtot"].isna() & df["dic"].notna() & df["ta"].notna() & df["temp"].notna()
    )
    if mask_dic.any():
        df.loc[mask_dic, "phtot"] = df.loc[mask_dic].apply(
            lambda row: cb.Csys(
                TA=row["ta"],
                DIC=row["dic"],
                T_in=row["temp"],
                S_in=row["sal"] if pd.notna(row["sal"]) else 35,
            ).pHtot[0],
            axis=1,
        )
    mask_pco2 = (
        df["phtot"].isna() & df["pco2"].notna() & df["ta"].notna() & df["temp"].notna()
    )
    if mask_pco2.any():
        df.loc[mask_pco2, "phtot"] = df.loc[mask_pco2].apply(
            lambda row: cb.Csys(
                TA=row["ta"],
                pCO2=row["pco2"],
                T_in=row["temp"],
                S_in=row["sal"] if pd.notna(row["sal"]) else 35,
            ).pHtot[0],
            axis=1,
        )
    return df


def calculate_carb_chem(row: pd.Series, out_values: list) -> dict:
    """(Re)calculate carbonate chemistry parameters from the dataframe row and return a dictionary."""
    try:
        out_dict = cb.Csys(
            pHtot=row["phtot"],
            TA=row["ta"],
            T_in=row["temp"],
            S_in=35 if pd.isna(row["sal"]) else row["sal"],
        )
        out_dict = {key.lower(): value for key, value in out_dict.items()}
        return {
            key: (
                out_dict.get(key.lower(), None)[0]
                if isinstance(out_dict.get(key.lower()), (list, np.ndarray))
                else out_dict.get(key.lower(), None)
            )
            for key in out_values
        }
    except Exception as e:
        logger.error(f"Error in calculate_carb_chem: {e}")
        return {key: None for key in out_values}
