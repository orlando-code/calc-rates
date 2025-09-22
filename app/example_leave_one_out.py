"""
Example usage of leave-one-out cross-validation for meta-analysis.

This script demonstrates how to use the leave_one_out function to perform
cross-validation analysis on meta-analysis models.
"""

import numpy as np
import pandas as pd

from app.infrastructure import DataLoader
from app.leave_one_out import (
    identify_influential_studies,
    leave_one_out,
    summarize_leave_one_out_results,
)


def example_leave_one_out_analysis():
    """
    Example of how to run leave-one-out cross-validation analysis.
    """
    print("🔬 Example: Leave-One-Out Cross-Validation for Meta-Analysis")
    print("=" * 60)

    # Load data
    print("📊 Loading data...")
    data_loader = DataLoader(data_path="data/clean/analysis_ready_data.csv")
    df = data_loader.load_data()

    if df is None:
        print("❌ No data found!")
        return

    # Example parameters (adjust as needed)
    effect_type = "st_relative_calcification"
    treatment = "OA"  # or ["OA", "OW"] for multiple treatments
    formula = "~ delta_t + I(delta_t^2)"  # Example quadratic formula
    random_structure = "~ 1 | original_doi"

    print(f"📈 Effect type: {effect_type}")
    print(f"🧪 Treatment: {treatment}")
    print(f"📐 Formula: {formula}")
    print(f"🎲 Random structure: {random_structure}")
    print(f"📊 Total observations: {len(df)}")
    print(f"📚 Total studies: {len(df['original_doi'].unique())}")

    # Filter data if needed
    if isinstance(treatment, str):
        filtered_df = df[df["treatment"] == treatment].copy()
    else:
        filtered_df = df[df["treatment"].isin(treatment)].copy()

    print(f"📊 Filtered observations: {len(filtered_df)}")
    print(f"📚 Filtered studies: {len(filtered_df['original_doi'].unique())}")

    # Run leave-one-out analysis
    print("\n🔄 Running leave-one-out cross-validation...")
    results_df = leave_one_out(
        df=filtered_df,
        effect_type=effect_type,
        treatment=treatment,
        formula=formula,
        random_structure=random_structure,
        verbose=True,
    )

    # Summarize results
    print("\n📋 Summarizing results...")
    summary = summarize_leave_one_out_results(results_df, verbose=True)

    # Identify influential studies
    print("\n🔍 Identifying influential studies...")
    try:
        influential_studies = identify_influential_studies(
            results_df, metric="AICc", threshold_percentile=90
        )
        print("\nTop 5 most influential studies (by AICc):")
        print(influential_studies.head().to_string(index=False))
    except Exception as e:
        print(f"⚠️ Could not identify influential studies: {e}")

    # Save results
    output_file = "results/leave_one_out_results.csv"
    print(f"\n💾 Saving results to {output_file}...")
    try:
        results_df.to_csv(output_file, index=False)
        print("✅ Results saved successfully!")
    except Exception as e:
        print(f"⚠️ Could not save results: {e}")

    return results_df, summary


def example_coefficient_stability_analysis(results_df: pd.DataFrame):
    """
    Analyze coefficient stability across leave-one-out models.

    Args:
        results_df: Results from leave_one_out function
    """
    print("\n🎯 Coefficient Stability Analysis")
    print("=" * 40)

    # Get successful models
    successful_models = results_df[results_df["model_fitted_successfully"] == True]

    if len(successful_models) == 0:
        print("❌ No successful models to analyze")
        return

    # Find coefficient columns
    coef_cols = [col for col in successful_models.columns if col.startswith("coef_")]

    if len(coef_cols) == 0:
        print("❌ No coefficient columns found")
        return

    print(
        f"📊 Analyzing stability of {len(coef_cols)} coefficients across {len(successful_models)} models"
    )

    stability_stats = []

    for coef_col in coef_cols:
        coef_name = coef_col.replace("coef_", "")
        coef_data = successful_models[coef_col].dropna()

        if len(coef_data) > 1:
            stats = {
                "coefficient": coef_name,
                "n_models": len(coef_data),
                "mean": coef_data.mean(),
                "std": coef_data.std(),
                "min": coef_data.min(),
                "max": coef_data.max(),
                "cv": coef_data.std() / abs(coef_data.mean())
                if coef_data.mean() != 0
                else np.inf,
                "range": coef_data.max() - coef_data.min(),
            }
            stability_stats.append(stats)

    stability_df = pd.DataFrame(stability_stats)

    if len(stability_df) > 0:
        # Sort by coefficient of variation (lower = more stable)
        stability_df = stability_df.sort_values("cv")

        print("\nCoefficient Stability Rankings (lower CV = more stable):")
        print(
            stability_df[["coefficient", "mean", "std", "cv", "range"]].to_string(
                index=False
            )
        )

        # Identify most and least stable coefficients
        most_stable = stability_df.iloc[0]["coefficient"]
        least_stable = stability_df.iloc[-1]["coefficient"]

        print(
            f"\n✅ Most stable coefficient: {most_stable} (CV = {stability_df.iloc[0]['cv']:.3f})"
        )
        print(
            f"⚠️ Least stable coefficient: {least_stable} (CV = {stability_df.iloc[-1]['cv']:.3f})"
        )

        return stability_df

    return None


def example_model_fit_comparison(results_df: pd.DataFrame):
    """
    Compare model fit statistics across leave-one-out models.

    Args:
        results_df: Results from leave_one_out function
    """
    print("\n📈 Model Fit Comparison")
    print("=" * 30)

    successful_models = results_df[results_df["model_fitted_successfully"] == True]

    if len(successful_models) == 0:
        print("❌ No successful models to analyze")
        return

    # Model fit statistics to analyze
    fit_stats = [
        "LogLik",
        "AIC",
        "AICc",
        "BIC",
        "residual_heterogeneity_QE",
        "model_test_QM",
    ]

    print(f"📊 Comparing model fit across {len(successful_models)} models")

    for stat in fit_stats:
        if stat in successful_models.columns:
            data = successful_models[stat].dropna()
            if len(data) > 0:
                print(f"\n{stat}:")
                print(f"  Mean: {data.mean():.3f}")
                print(f"  Std:  {data.std():.3f}")
                print(f"  Min:  {data.min():.3f}")
                print(f"  Max:  {data.max():.3f}")
                print(f"  Range: {data.max() - data.min():.3f}")


if __name__ == "__main__":
    # Run example analysis
    try:
        results_df, summary = example_leave_one_out_analysis()

        if results_df is not None and len(results_df) > 0:
            # Additional analyses
            example_coefficient_stability_analysis(results_df)
            example_model_fit_comparison(results_df)

    except Exception as e:
        print(f"❌ Example failed: {e}")
        import traceback

        traceback.print_exc()
