#!/usr/bin/env python3
"""
Script to prepare calcification data for R meta-analysis
"""

import argparse

from calcification.processing import process
from calcification.utils import config

parser = argparse.ArgumentParser()
parser.add_argument("--output_filename", type=str, default="analysis_ready_data.csv")
parser.add_argument(
    "--effect_types",
    nargs="+",
    default=["hedges_g", "st_relative_calcification"],
    help="Effect types to include in the output file (can specify multiple)",
)
parser.add_argument(
    "--additional_columns",
    nargs="+",
    default=[],
    help="Additional columns to include in the output file (can specify multiple, e.g. 'delta_t delta_ph')",
)
parser.add_argument(
    "--overwrite",
    action="store_true",
    help="Force overwrite of existing output file",
)
args = parser.parse_args()


def main():
    """Load and prepare data for R meta-analysis."""

    # --- set up output file ---
    output_file = config.clean_data_dir / args.output_filename
    # create directory if it doesn't exist
    output_file.parent.mkdir(parents=True, exist_ok=True)
    if output_file.exists() and not args.overwrite:
        print(
            f"Output file {output_file} already exists. Use --overwrite to overwrite."
        )
        return None

    print("Loading calcification data...")

    # --- process data from extracted file ---
    effects_df, _ = process.process_extracted_calcification_data(
        fp=config.data_dir / "Orlando_data.xlsx",
    )

    print(f"Original dataset dimensions: {effects_df.shape}")
    print(f"Columns: {list(effects_df.columns)}")

    # Check key columns for meta-analysis
    key_columns = [
        "doi",
        "original_doi",
        "latitude",
        "longitude",
        "irr_group",
        "temp",
        "phtot",
        "delta_t",
        "delta_ph",
        "core_grouping",
        "st_calcification_unit",
        "st_relative_calcification",
        "st_relative_calcification_var",
        "st_control_calcification",
        "st_treatment_calcification",
        "calcification",
        "calcification_unit",
        "treatment",
    ]
    for effect_type in args.effect_types:
        key_columns.append(effect_type)
        key_columns.append(f"{effect_type}_var")
    # Add additional columns
    additional_cols = args.additional_columns
    if additional_cols:
        print(f"Adding additional columns to the key columns: {additional_cols}")
        key_columns.extend(additional_cols)

    # --- check for missing columns ---
    print("\nChecking key columns:")
    for col in key_columns:
        if col in effects_df.columns:
            missing = effects_df[col].isna().sum()
            total = len(effects_df)
            print(f"  {col}: {missing}/{total} missing ({missing / total * 100:.1f}%)")
        else:
            print(f"  {col}: NOT FOUND")

    # --- summarise dataset statistics ---
    print("\nSummary statistics:")
    print(
        f"Number of unique study samples (species and light treatment, per study): {effects_df['doi'].nunique()}"
    )
    print(f"Number of effect sizes: {len(effects_df)}")

    if "core_grouping" in effects_df.columns:
        # print("\nCore groupings:")
        print(effects_df["core_grouping"].value_counts())

    print("\nEffect size summaries:")
    for effect_type in args.effect_types:
        print(f"{effect_type}:\n{effects_df[effect_type].describe()}")
        print(f"{effect_type}_var:\n{effects_df[f'{effect_type}_var'].describe()}")

    if "delta_t" in effects_df.columns:
        print("\nTemperature change summary (delta_t):")
        print(effects_df["delta_t"].describe())

    if "delta_ph" in effects_df.columns:
        print("\npH change summary (delta_ph):")
        print(effects_df["delta_ph"].describe())

    # --- filter data to only include columns that exist ---
    available_columns = [col for col in key_columns if col in effects_df.columns]
    data_for_r = effects_df[available_columns].copy()
    # add ID column
    data_for_r["ID"] = range(len(data_for_r))

    print(f"Saving to {output_file}")

    data_for_r.to_csv(output_file, index=False)

    # --- summarise results ---
    print(f"\nData saved to {output_file}")
    print(f"Final dataset: {data_for_r.shape}")
    print(f"Columns saved: {list(data_for_r.columns)}")

    print("\nFirst few rows of the dataset:")
    print(data_for_r.head())

    return data_for_r


if __name__ == "__main__":
    data = main()
