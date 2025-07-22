# general

import cbsyst as cb
import cbsyst.helpers as cbh
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from calcification.processing import cleaning
from calcification.utils import config, file_ops


### carbonate chemistry
def populate_carbonate_chemistry(
    fp: str, sheet_name: str = "all_data", selection_dict: dict = {"include": "yes"}
) -> pd.DataFrame:
    df = cleaning.process_raw_data(
        pd.read_excel(fp, sheet_name=sheet_name),
        require_results=False,
        selection_dict=selection_dict,
    )
    ### load measured values
    print("Loading measured values...")
    measured_df = file_ops.get_highlighted(
        fp, sheet_name=sheet_name
    )  # keeping all cols
    # measured_df = process_raw_data(measured_df, require_results=False, selection_dict=selection_dict)
    measured_df = cleaning.preprocess_df(measured_df, selection_dict=selection_dict)

    ### convert nbs values to total scale using cbsyst     # TODO: implement uncertainty propagation
    # if one of ph is provided, ensure total ph is calculated
    # Only convert pHnbs to pHtot if pHtot is NaN and pHnbs is available
    mask_missing_phtot_with_nbs = (
        measured_df["phtot"].isna()
        & measured_df["phnbs"].notna()
        & measured_df["temp"].notna()
    )
    measured_df.loc[mask_missing_phtot_with_nbs, "phtot"] = measured_df[
        mask_missing_phtot_with_nbs
    ].apply(
        lambda row: cbh.pH_scale_converter(
            pH=row["phnbs"],
            scale="NBS",
            Temp=row["temp"],
            Sal=row["sal"] if pd.notna(row["sal"]) else 35,
        ).get("pHtot", None),
        axis=1,
    )

    # Only calculate pHtot from DIC and TA if pHtot is still NaN and required parameters are available
    mask_missing_phtot_with_carb = (
        measured_df["phtot"].isna()
        & measured_df["dic"].notna()
        & measured_df["ta"].notna()
        & measured_df["temp"].notna()
    )
    if mask_missing_phtot_with_carb.any():
        measured_df.loc[mask_missing_phtot_with_carb, "phtot"] = measured_df[
            mask_missing_phtot_with_carb
        ].apply(
            lambda row: cb.Csys(
                TA=row["ta"],
                DIC=row["dic"],
                T_in=row["temp"],
                S_in=row["sal"] if pd.notna(row["sal"]) else 35,
            ).pHtot[0],
            axis=1,
        )
    # if phtot is still NaN, calculate from other parameters.
    mask_missing_phtot_with_alt_carb = (
        measured_df["phtot"].isna()
        & measured_df["pco2"].notna()
        & measured_df["ta"].notna()
        & measured_df["temp"].notna()
    )
    if mask_missing_phtot_with_alt_carb.any():
        measured_df.loc[mask_missing_phtot_with_alt_carb, "phtot"] = measured_df[
            mask_missing_phtot_with_alt_carb
        ].apply(
            lambda row: cb.Csys(
                TA=row["ta"],
                pCO2=row["pco2"],
                T_in=row["temp"],
                S_in=row["sal"] if pd.notna(row["sal"]) else 35,
            ).pHtot[0],
            axis=1,
        )

    ### calculate carbonate chemistry
    carb_metadata = file_ops.read_yaml(config.resources_dir / "mapping.yaml")
    carb_chem_cols = carb_metadata["carbonate_chemistry_cols"]
    out_values = carb_metadata["carbonate_chemistry_params"]
    carb_df = measured_df[carb_chem_cols].copy()

    # apply carbonate chemistry calculation row-wise
    tqdm.pandas(desc="Calculating carbonate chemistry")
    carb_df.loc[:, out_values] = carb_df.progress_apply(
        lambda row: pd.Series(calculate_carb_chem(row, out_values)), axis=1
    )
    return df.combine_first(carb_df)


def calculate_carb_chem(row: pd.Series, out_values: list) -> dict:
    """(Re)calculate carbonate chemistry parameters from the dataframe row and return a dictionary."""
    try:
        out_dict = cb.Csys(
            pHtot=row["phtot"],
            TA=row["ta"],
            T_in=row["temp"],
            S_in=35 if pd.isna(row["sal"]) else row["sal"],
        )
        out_dict = {
            key.lower(): value for key, value in out_dict.items()
        }  # lower the keys of the dictionary to ensure case-insensitivity
        return {
            key: (
                out_dict.get(key.lower(), None)[0]
                if isinstance(out_dict.get(key.lower()), (list, np.ndarray))
                else out_dict.get(key.lower(), None)
            )
            for key in out_values
        }
    except Exception as e:
        print(f"Error: {e}")
